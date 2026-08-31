"""Distributional SAC Critic — Quantile-Conditional Q-Network for QR-SAC + CVaR.

Replaces the scalar Q-head of SACCriticNetwork with a quantile-conditional head
that outputs N quantile values per (state, action) pair. The encoder
(MultiScaleEncoder / SummaryStatsEncoder) is reused unchanged.

Architecture:
  state -> MultiScaleEncoder.encode() -> features (B, fusion_dim)
  tau ~ Uniform(0,1) -> QuantileEmbedding -> (B, N, fusion_dim)
  features * tau_embed -> cat(action) -> MLP -> (B, N) quantile values
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpen.agents.common.network_blocks import QuantileEmbedding, _tc_align
from sharpen.agents.sac.networks import MultiScaleEncoder, SummaryStatsEncoder

logger = logging.getLogger(__name__)


class DistributionalSACCriticNetwork(nn.Module):
    """Quantile-conditional Q-network for distributional SAC.

    Outputs (B, N) quantile values instead of scalar Q(s,a).
    Reuses MultiScaleEncoder for state encoding and QuantileEmbedding from IQN.
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        action_dim: int = 1,
        n_scales: int = 3,
        n_quantiles: int = 32,
        quantile_embed_dim: int = 64,
        obs_mode: str = "window",
        lob_encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        self._obs_mode = obs_mode
        self._n_quantiles = n_quantiles
        self._fusion_dim = fusion_dim

        # State encoder — identical to SACCriticNetwork
        if obs_mode == "summary_stats":
            summary_dim = scale_encoder_config.get("summary_input_dim", 50)
            self.encoder = SummaryStatsEncoder(summary_dim, fusion_dim)
        else:
            self.encoder = MultiScaleEncoder(
                scale_encoder_config, private_dim, fusion_dim, n_scales,
                lob_encoder_config=lob_encoder_config,
            )

        # Quantile embedding: tau (B, N) -> (B, N, fusion_dim)
        self.tau_embed = QuantileEmbedding(quantile_embed_dim, fusion_dim)

        # Q-head: operates on (fusion_dim + action_dim) per quantile
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

    def encode(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multi-scale obs to fusion features: (B, fusion_dim).

        Identical to SACCriticNetwork.encode() — enables O-A caching.
        """
        if self._obs_mode == "summary_stats":
            return self.encoder(scale_stack)
        return self.encoder(scale_stack, private, lob=lob)

    def q_head_forward(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
        n_quantiles: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantile Q-values from pre-computed encoder features.

        Args:
            features: (B, fusion_dim) from encode()
            action: (B, action_dim)
            tau: (B, N) quantile fractions in [0,1]. If None, sampled uniformly.
            n_quantiles: Number of quantiles when tau is None. Defaults to self._n_quantiles.

        Returns:
            quantile_values: (B, N) quantile Q-values
            tau: (B, N) the quantile fractions used
        """
        B = features.shape[0]
        N = n_quantiles if n_quantiles is not None else self._n_quantiles

        if tau is None:
            tau = torch.rand(B, N, device=features.device)

        N = tau.shape[1]

        # Quantile embedding: (B, N) -> (B, N, fusion_dim)
        tau_embedding = self.tau_embed(tau)

        # Hadamard product: features (B, 1, fusion_dim) * tau_embed (B, N, fusion_dim)
        hadamard = features.unsqueeze(1) * tau_embedding  # (B, N, fusion_dim)

        # Expand action: (B, action_dim) -> (B, N, action_dim)
        action_expanded = action.unsqueeze(1).expand(-1, N, -1)

        # Concat and reshape for batched MLP
        combined = torch.cat([hadamard, action_expanded], dim=-1)  # (B, N, fusion_dim + action_dim)
        if self._q_pad > 0:
            combined = F.pad(combined, (0, self._q_pad))

        # Batched forward: (B*N, dim) -> (B*N, 1) -> (B, N)
        BN = B * N
        q_values = self.q_head(combined.reshape(BN, -1)).reshape(B, N)

        return q_values, tau

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        n_quantiles: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: encode + quantile Q-head.

        Returns:
            quantile_values: (B, N) quantile Q-values
            tau: (B, N) the quantile fractions used
        """
        features = self.encode(scale_stack, private, lob=lob)
        return self.q_head_forward(features, action, tau=tau, n_quantiles=n_quantiles)
