"""
SAC Networks — Multi-Scale CNN Encoder + Gaussian Actor + Twin Critics

Architecture:
  3 × DilatedCNNEncoder (one per timescale: 3m, 15m, 1h)
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
    """Fusion of 3 timescale encoders + private state.

    Forward:
        encode each scale → concat [h_3m, h_15m, h_1h, private]
        → TC-align → fusion MLP → LayerNorm → ReLU → (B, fusion_dim)
    """

    def __init__(
        self,
        scale_encoder_config: Dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
    ):
        super().__init__()
        output_dim = scale_encoder_config.get("output_dim", 64)

        self.encoder_3m = DilatedCNNEncoder(**scale_encoder_config)
        self.encoder_15m = DilatedCNNEncoder(**scale_encoder_config)
        self.encoder_1h = DilatedCNNEncoder(**scale_encoder_config)

        concat_dim = output_dim * 3 + private_dim  # 64*3 + 5 = 197
        tc_dim = _tc_align(concat_dim)  # 200
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
        scale_3m: torch.Tensor,
        scale_15m: torch.Tensor,
        scale_1h: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scale_3m: (B, 30, 7)
            scale_15m: (B, 30, 7)
            scale_1h: (B, 30, 7)
            private: (B, 5)
        Returns:
            (B, fusion_dim)
        """
        h_3m = self.encoder_3m(scale_3m)
        h_15m = self.encoder_15m(scale_15m)
        h_1h = self.encoder_1h(scale_1h)
        combined = torch.cat([h_3m, h_15m, h_1h, private], dim=1)
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
    ):
        super().__init__()
        self.encoder = MultiScaleEncoder(scale_encoder_config, private_dim, fusion_dim)

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
        scale_3m: torch.Tensor,
        scale_15m: torch.Tensor,
        scale_1h: torch.Tensor,
        private: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) for the Gaussian policy."""
        features = self.encoder(scale_3m, scale_15m, scale_1h, private)
        h = self.head(features)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, log_sigma

    def sample(
        self,
        scale_3m: torch.Tensor,
        scale_15m: torch.Tensor,
        scale_1h: torch.Tensor,
        private: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with log_prob (tanh correction applied).

        Returns:
            action: (B, 1) in [-1, 1]
            log_prob: (B, 1)
        """
        mu, log_sigma = self.forward(scale_3m, scale_15m, scale_1h, private)
        sigma = log_sigma.exp()

        if deterministic:
            action = torch.tanh(mu)
            # log_prob not needed for deterministic, return zeros
            return action, torch.zeros_like(action)

        # Reparameterization trick
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        action = torch.tanh(z)

        # Log prob with tanh correction: log_prob - log(1 - tanh(z)^2)
        # FIX R4-AUD-12: Use clamp instead of addition for BF16 numerical safety.
        # Near tanh saturation (action≈±1), 1-action^2 can underflow in BF16;
        # clamp guarantees the log argument stays positive.
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
    ):
        super().__init__()
        self.encoder = MultiScaleEncoder(scale_encoder_config, private_dim, fusion_dim)

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
        scale_3m: torch.Tensor,
        scale_15m: torch.Tensor,
        scale_1h: torch.Tensor,
        private: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Returns scalar Q-value: (B, 1)"""
        features = self.encoder(scale_3m, scale_15m, scale_1h, private)
        combined = torch.cat([features, action], dim=1)
        if self._q_pad > 0:
            combined = F.pad(combined, (0, self._q_pad))
        return self.q_head(combined)
