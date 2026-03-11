"""
SAC Networks — Multi-Scale CNN Encoder + Gaussian Actor + Twin Critics

Architecture:
  N × DilatedCNNEncoder (one per timescale, configurable)
    → concat + private state → fusion MLP → 256-dim representation
  Actor: fusion → (mu, log_sigma) → tanh-squashed Gaussian
  Critic: fusion + action → scalar Q-value (twin pair)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from finrl_pro_ds.agents.deepscalper.networks import _CausalConv1dBlock, _tc_align


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
        channels: Tuple[int, ...] = (32, 64, 64, 64),
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
    """Fusion of N timescale encoders + private state.

    Forward:
        encode each scale → concat [h_0, h_1, ..., h_N-1, private]
        → TC-align → fusion MLP → LayerNorm → ReLU → (B, fusion_dim)
    """

    def __init__(
        self,
        scale_encoder_config: Dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
    ):
        super().__init__()
        output_dim = scale_encoder_config.get("output_dim", 64)
        self.n_scales = n_scales

        self.encoders = nn.ModuleList([
            DilatedCNNEncoder(**scale_encoder_config) for _ in range(n_scales)
        ])

        concat_dim = output_dim * n_scales + private_dim
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
    ) -> torch.Tensor:
        """
        Args:
            scale_stack: (B, N, W, F) stacked scale tensors
            private: (B, private_dim)
        Returns:
            (B, fusion_dim)
        """
        encoded = [enc(scale_stack[:, i]) for i, enc in enumerate(self.encoders)]
        combined = torch.cat(encoded + [private], dim=1)
        if self._fusion_pad > 0:
            combined = F.pad(combined, (0, self._fusion_pad))
        return self.fusion(combined)


LOG_SIGMA_MIN = -20.0
LOG_SIGMA_MAX = 2.0


class SACActorNetwork(nn.Module):
    """Gaussian policy with tanh squashing for SAC.

    Output: tanh-squashed action in [-1, 1] with log_prob correction.
    """

    def __init__(
        self,
        scale_encoder_config: Dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
    ):
        super().__init__()
        self.encoder = MultiScaleEncoder(scale_encoder_config, private_dim, fusion_dim, n_scales)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(fusion_dim, 1)
        self.log_sigma_layer = nn.Linear(fusion_dim, 1)

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
        private: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) for the Gaussian policy."""
        features = self.encoder(scale_stack, private)
        h = self.head(features)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, log_sigma

    def sample(
        self,
        scale_stack: torch.Tensor,
        private: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with log_prob (tanh correction applied).

        Returns:
            action: (B, 1) in [-1, 1]
            log_prob: (B, 1)
        """
        mu, log_sigma = self.forward(scale_stack, private)
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

        return action, log_prob


class SACCriticNetwork(nn.Module):
    """Q-value network for SAC (one of twin pair).

    Input: multi-scale obs + action → scalar Q-value
    """

    def __init__(
        self,
        scale_encoder_config: Dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        action_dim: int = 1,
        n_scales: int = 3,
    ):
        super().__init__()
        self.encoder = MultiScaleEncoder(scale_encoder_config, private_dim, fusion_dim, n_scales)

        q_input_dim = fusion_dim + action_dim
        tc_q_dim = _tc_align(q_input_dim)
        self._q_pad = tc_q_dim - q_input_dim

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

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Returns scalar Q-value: (B, 1)"""
        features = self.encoder(scale_stack, private)
        combined = torch.cat([features, action], dim=1)
        if self._q_pad > 0:
            combined = F.pad(combined, (0, self._q_pad))
        return self.q_head(combined)
