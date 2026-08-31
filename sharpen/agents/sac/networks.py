"""
SAC Networks — Multi-Scale CNN Encoder + Gaussian Actor + Twin Critics

Architecture:
  N × DilatedCNNEncoder (one per timescale, configurable)
    → concat + private state → fusion MLP → 256-dim representation
  Actor: fusion → (mu, log_sigma) → tanh-squashed Gaussian
  Critic: fusion + action → scalar Q-value (twin pair)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpen.agents.common.batch_renorm import BatchRenorm1d
from sharpen.agents.common.network_blocks import _CausalConv1dBlock, _tc_align


class DilatedCNNEncoder(nn.Module):
    """Per-timescale encoder using causal dilated convolutions.

    Input: (B, T, F) where T=window_size, F=features_per_scale
    Output: (B, output_dim)

    4 causal conv blocks with dilations [1,2,4,8], kernel_size=3
    → receptive field = 1 + 2*(1+2+4+8) = 31 >= 30 bars
    → take last timestep → Linear → LayerNorm → (B, output_dim)
    """

    def __init__(
        self,
        input_size: int = 7,
        channels: tuple[int, ...] = (32, 64, 64, 64),
        kernel_size: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(_CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)

        self.proj = nn.Sequential(
            nn.Linear(in_ch, output_dim),
            nn.LayerNorm(output_dim),
        )

        rf = 1 + sum((kernel_size - 1) * (2 ** i) for i in range(len(channels)))
        self._receptive_field = rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → (B, output_dim)"""
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]  # last timestep
        return self.proj(x)


class MultiScaleEncoder(nn.Module):
    """Fusion of N timescale encoders + optional LOB encoder + private state.

    Forward:
        encode each scale → [optional: encode LOB] → concat + private
        → TC-align → fusion MLP → LayerNorm → ReLU → (B, fusion_dim)
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
        lob_encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        output_dim = scale_encoder_config.get("output_dim", 64)
        self.n_scales = n_scales

        self.encoders = nn.ModuleList([
            DilatedCNNEncoder(**scale_encoder_config) for _ in range(n_scales)
        ])

        # Optional LOB encoder — separate DilatedCNN for microstructure features
        self._lob_encoder: Optional[DilatedCNNEncoder] = None
        self._lob_output_dim = 0
        if lob_encoder_config is not None:
            self._lob_encoder = DilatedCNNEncoder(**lob_encoder_config)
            self._lob_output_dim = lob_encoder_config.get("output_dim", 64)
        lob_output_dim = self._lob_output_dim

        concat_dim = output_dim * n_scales + lob_output_dim + private_dim
        tc_dim = _tc_align(concat_dim)
        self._fusion_pad = tc_dim - concat_dim

        self.fusion = nn.Sequential(
            nn.Linear(tc_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.output_dim = fusion_dim

        self._init_weights()

    def _init_weights(self):
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: torch.Tensor,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            scale_stack: (B, N, W, F) stacked scale tensors
            private: (B, private_dim)
            lob: (B, W, n_lob_features) optional LOB microstructure tensor
        Returns:
            (B, fusion_dim)
        """
        encoded = [enc(scale_stack[:, i]) for i, enc in enumerate(self.encoders)]
        if self._lob_encoder is not None:
            if lob is not None:
                encoded.append(self._lob_encoder(lob))
            else:
                # Zero-fill: fusion Linear expects fixed input dim even without LOB data
                encoded.append(torch.zeros(
                    scale_stack.shape[0], self._lob_output_dim,
                    device=scale_stack.device, dtype=scale_stack.dtype,
                ))
        combined = torch.cat(encoded + [private], dim=1)
        if self._fusion_pad > 0:
            combined = F.pad(combined, (0, self._fusion_pad))
        return self.fusion(combined)


class SummaryStatsEncoder(nn.Module):
    """v6: Flat MLP encoder for summary-stats observation mode.

    Replaces DilatedCNNEncoder + MultiScaleEncoder when obs_mode="summary_stats".
    Input: (B, summary_dim + private_dim) flat vector.
    Output: (B, fusion_dim).

    ~40K params (vs ~500K for CNN encoder). AlphaSeek-informed: simpler arch wins.
    """

    def __init__(self, input_dim: int = 50, fusion_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        tc_input = _tc_align(input_dim)
        self._pad = tc_input - input_dim
        tc_hidden = _tc_align(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(tc_input, tc_hidden),
            nn.LayerNorm(tc_hidden),
            nn.ReLU(),
            nn.Linear(tc_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.output_dim = fusion_dim

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """flat_obs: (B, input_dim) — concatenated summary stats + private."""
        if self._pad > 0:
            flat_obs = F.pad(flat_obs, (0, self._pad))
        return self.net(flat_obs)


LOG_SIGMA_MIN = -20.0
LOG_SIGMA_MAX = 2.0


class SACActorNetwork(nn.Module):
    """Gaussian policy with tanh squashing for SAC.

    Output: tanh-squashed action in [-1, 1] with log_prob correction.
    Supports multi-dimensional actions via action_dim parameter.
    v6: obs_mode="summary_stats" uses SummaryStatsEncoder (flat MLP, ~40K params).
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
        action_dim: int = 1,
        obs_mode: str = "window",
        lob_encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self._obs_mode = obs_mode

        if obs_mode == "summary_stats":
            summary_dim = scale_encoder_config.get("summary_input_dim", 50)
            self.encoder = SummaryStatsEncoder(summary_dim, fusion_dim)
        else:
            self.encoder = MultiScaleEncoder(
                scale_encoder_config, private_dim, fusion_dim, n_scales,
                lob_encoder_config=lob_encoder_config,
            )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(fusion_dim, action_dim)
        self.log_sigma_layer = nn.Linear(fusion_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.log_sigma_layer.weight)
        nn.init.zeros_(self.log_sigma_layer.bias)

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) for the Gaussian policy.

        In summary_stats mode, scale_stack is actually a flat (B, input_dim) tensor
        and private is None (already concatenated).
        """
        if self._obs_mode == "summary_stats":
            features = self.encoder(scale_stack)
        else:
            features = self.encoder(scale_stack, private, lob=lob)
        h = self.head(features)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, log_sigma

    def sample(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        lob: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with log_prob (tanh correction applied).

        In summary_stats mode, scale_stack is flat (B, input_dim), private is None.

        Returns:
            action: (B, action_dim) in [-1, 1]
            log_prob: (B, 1) — summed over action dims for multi-dim actions
        """
        mu, log_sigma = self.forward(scale_stack, private, lob=lob)
        sigma = log_sigma.exp()

        if deterministic:
            action = torch.tanh(mu)
            return action, torch.zeros_like(action)

        # Reparameterization trick
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z)

        # Log prob with tanh correction: log_prob - log(1 - tanh(z)^2)
        # FIX R4-AUD-12: Use clamp instead of addition for BF16 numerical safety.
        log_prob = dist.log_prob(z) - torch.log(torch.clamp(1 - action.pow(2), min=1e-6))

        # Sum over action dims for multi-dimensional actions → (B, 1) scalar per sample
        if self.action_dim > 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class SACCriticNetwork(nn.Module):
    """Q-value network for SAC (one of twin pair).

    Input: multi-scale obs + action → scalar Q-value
    v6: obs_mode="summary_stats" uses SummaryStatsEncoder (flat MLP).

    CrossQ (`crossq=True`): the Q head becomes a BatchRenorm MLP
    (BN → [Linear → ReLU → BN] x depth → Linear), which is what makes the
    target network removable. BN goes in the head only, never the encoder: the
    encoder normalizes with LayerNorm, which is per-sample and therefore cannot
    be a source of train/bootstrap distribution mismatch in the first place.
    Geometry defaults to the baseline critic's (width=fusion_dim, depth=1) so
    that flipping the flag is a pure mechanism swap; the paper's 2048x2 critic
    is reachable via crossq_width / crossq_depth.
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        action_dim: int = 1,
        n_scales: int = 3,
        obs_mode: str = "window",
        lob_encoder_config: Optional[dict] = None,
        crossq: bool = False,
        crossq_width: Optional[int] = None,
        crossq_depth: int = 1,
        bn_momentum: float = 0.01,
        bn_warmup_steps: int = 100_000,
    ):
        super().__init__()
        self._obs_mode = obs_mode
        self.is_crossq = crossq

        if obs_mode == "summary_stats":
            summary_dim = scale_encoder_config.get("summary_input_dim", 50)
            self.encoder = SummaryStatsEncoder(summary_dim, fusion_dim)
        else:
            self.encoder = MultiScaleEncoder(
                scale_encoder_config, private_dim, fusion_dim, n_scales,
                lob_encoder_config=lob_encoder_config,
            )

        q_input_dim = fusion_dim + action_dim
        tc_q_dim = _tc_align(q_input_dim)
        self._q_pad = tc_q_dim - q_input_dim

        if crossq:
            if crossq_depth < 1:
                raise ValueError(f"crossq_depth must be >= 1, got {crossq_depth}")
            width = _tc_align(crossq_width if crossq_width else fusion_dim)
            layers: list[nn.Module] = [
                BatchRenorm1d(tc_q_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps),
            ]
            in_dim = tc_q_dim
            for _ in range(crossq_depth):
                layers += [
                    nn.Linear(in_dim, width),
                    nn.ReLU(),
                    BatchRenorm1d(width, momentum=bn_momentum, warmup_steps=bn_warmup_steps),
                ]
                in_dim = width
            layers.append(nn.Linear(in_dim, 1))
            self.q_head = nn.Sequential(*layers)
        else:
            self.q_head = nn.Sequential(
                nn.Linear(tc_q_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for layer in self.q_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multi-scale obs to fusion features: (B, fusion_dim).

        Use with q_head_forward() to avoid redundant encoder passes
        when the same (scale_stack, private) is needed for both critic
        and actor updates (O-A encoder caching).

        In summary_stats mode, scale_stack is flat (B, input_dim), private is None.
        """
        if self._obs_mode == "summary_stats":
            return self.encoder(scale_stack)
        return self.encoder(scale_stack, private, lob=lob)

    def q_head_forward(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Q-value from pre-computed encoder features: (B, 1).

        Args:
            features: (B, fusion_dim) from encode()
            action: (B, action_dim)
        """
        combined = torch.cat([features, action], dim=1)
        if self._q_pad > 0:
            combined = F.pad(combined, (0, self._q_pad))
        return self.q_head(combined)

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns scalar Q-value: (B, 1)

        In summary_stats mode, scale_stack is flat (B, input_dim), private is None.
        """
        features = self.encode(scale_stack, private, lob=lob)
        return self.q_head_forward(features, action)
