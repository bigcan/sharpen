"""
IQN (Implicit Quantile Network) with quantile cosine embedding and dueling architecture.

Reuses MicroEncoder and MacroEncoder from networks.py. Replaces standard
Q-value heads with quantile-conditional heads using NoisyLinear layers.

Reference: Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from finrl_pro_ds.agents.deepscalper.networks import (
    MicroEncoder,
    MicroEncoderMLP,
    MicroEncoderFlat,
    MicroEncoderTCN,
    MacroEncoder,
)
from finrl_pro_ds.agents.deepscalper.noisy_linear import NoisyLinear


class QuantileEmbedding(nn.Module):
    """Cosine basis embedding for quantile fractions tau in [0, 1].

    Maps tau -> cos(pi * i * tau) for i in [1..embedding_dim], then
    projects through a linear layer to match the fusion dimension.

    Input:  tau (B, N) where N = num_quantiles
    Output: (B, N, output_dim)
    """

    def __init__(self, embedding_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        # i-values: [1, 2, ..., embedding_dim]
        self.register_buffer(
            "i_pi",
            torch.arange(1, embedding_dim + 1, dtype=torch.float32) * math.pi,
        )
        self.proj = nn.Linear(embedding_dim, output_dim)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (B, N) quantile fractions in [0, 1]
        Returns:
            (B, N, output_dim) quantile embeddings
        """
        # (B, N, 1) * (embedding_dim,) -> (B, N, embedding_dim)
        cos_features = torch.cos(tau.unsqueeze(-1) * self.i_pi)
        # (B, N, embedding_dim) -> (B, N, output_dim)
        return self.activation(self.proj(cos_features))


class IQNNetwork(nn.Module):
    """Implicit Quantile Network with dueling architecture and NoisyNet heads.

    Architecture:
        1. MicroEncoder(micro) -> (B, micro_hidden)
        2. MacroEncoder(macro) -> (B, macro_hidden)
        3. Fusion([micro_enc, macro_enc, private]) -> (B, fusion_dim)
        4. QuantileEmbedding(tau) -> (B, N, fusion_dim)
        5. Element-wise: fused * tau_embed -> (B, N, fusion_dim)
        6. Dueling per quantile:
           V = NoisyLinear -> (B, N, 1)
           A = NoisyLinear -> (B, N, n_actions)
           Q_tau = V + A - mean(A) -> (B, N, n_actions)

    Args:
        micro_config: Config dict for MicroEncoder.
        macro_config: Config dict for MacroEncoder.
        fusion_dim: Dimension of fusion layer output.
        n_actions: Number of discrete actions.
        embedding_dim: Cosine basis dimension for quantile embedding.
        noisy_sigma0: Initial noise scale for NoisyLinear.
    """

    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        n_actions: int = 3,
        embedding_dim: int = 64,
        noisy_sigma0: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.n_actions = n_actions

        # Encoders (reuse from networks.py)
        encoder_type = micro_config.get("encoder_type", "rnn").lower()
        if encoder_type == "mlp":
            self.micro_encoder = MicroEncoderMLP(**micro_config)
        elif encoder_type == "tcn":
            self.micro_encoder = MicroEncoderTCN(**micro_config)
        elif encoder_type == "flat":
            self.micro_encoder = MicroEncoderFlat(**micro_config)
        else:
            self.micro_encoder = MicroEncoder(**micro_config)

        self.macro_encoder = MacroEncoder(**macro_config)

        # Fusion dimensions
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        private_size = micro_config.get("private_input_size", 5)
        self.private_size = private_size
        fusion_in_dim = micro_out_dim + macro_out_dim + private_size

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        # Quantile embedding
        self.quantile_embedding = QuantileEmbedding(
            embedding_dim=embedding_dim, output_dim=fusion_dim
        )

        # Dueling heads with NoisyLinear
        head_hidden = max(fusion_dim // 2, 64)

        # Value stream: (B, N, fusion_dim) -> (B, N, 1)
        self.value_fc1 = NoisyLinear(fusion_dim, head_hidden, sigma0=noisy_sigma0)
        self.value_fc2 = NoisyLinear(head_hidden, 1, sigma0=noisy_sigma0)

        # Advantage stream: (B, N, fusion_dim) -> (B, N, n_actions)
        self.adv_fc1 = NoisyLinear(fusion_dim, head_hidden, sigma0=noisy_sigma0)
        self.adv_fc2 = NoisyLinear(head_hidden, n_actions, sigma0=noisy_sigma0)

        # Auxiliary volatility prediction head (from fusion, not quantile-conditional)
        self.vol_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Initialize fusion and vol_head
        self._init_heads()

    def _init_heads(self):
        for module in [self.fusion, self.vol_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        micro_in: torch.Tensor,
        private_in: torch.Tensor,
        macro_in: torch.Tensor,
        tau: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            micro_in: (B, W, micro_features)
            private_in: (B, W, private_features)
            macro_in: (B, macro_features)
            tau: (B, N) quantile fractions in [0, 1]
            hidden: Optional LSTM hidden state

        Returns:
            Q_tau: (B, N, n_actions) quantile Q-values
            pred_vol: (B, 1) volatility prediction
            new_hidden: Updated LSTM hidden state (or None)
        """
        # Encode
        h_micro, new_hidden = self.micro_encoder(micro_in, hidden)  # (B, micro_hidden)
        h_macro = self.macro_encoder(macro_in)  # (B, macro_hidden)

        # DIV-1: Private state injected at fusion layer
        private_last = private_in[:, -1, :]  # (B, private_size)

        # Fusion
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        fused = self.fusion(combined)  # (B, fusion_dim)

        # Quantile embedding
        tau_embed = self.quantile_embedding(tau)  # (B, N, fusion_dim)

        # Element-wise product: (B, 1, fusion_dim) * (B, N, fusion_dim)
        x = fused.unsqueeze(1) * tau_embed  # (B, N, fusion_dim)

        # Dueling heads (applied per quantile)
        v = torch.relu(self.value_fc1(x))
        v = self.value_fc2(v)  # (B, N, 1)

        a = torch.relu(self.adv_fc1(x))
        a = self.adv_fc2(a)  # (B, N, n_actions)

        # Q = V + A - mean(A)
        q_tau = v + a - a.mean(dim=-1, keepdim=True)  # (B, N, n_actions)

        # Auxiliary volatility prediction (from fusion, not quantile-conditional)
        pred_vol = self.vol_head(fused)  # (B, 1)

        return q_tau, pred_vol, new_hidden

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        self.value_fc1.reset_noise()
        self.value_fc2.reset_noise()
        self.adv_fc1.reset_noise()
        self.adv_fc2.reset_noise()
