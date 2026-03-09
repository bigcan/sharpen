"""
IQN (Implicit Quantile Network) with quantile cosine embedding and dueling architecture.

Reuses MicroEncoder and MacroEncoder from networks.py. Replaces standard
Q-value heads with quantile-conditional heads using NoisyLinear layers.

Supports optional multi-horizon mode: two sets of dueling heads (short + long
discount horizons) sharing the same encoder backbone. Action selection blends
both horizons. This lets the agent balance immediate momentum (short-term gain)
against strategic positioning (long-term goal).

Reference:
  Dabney et al. (2018) "Implicit Quantile Networks for Distributional RL"
  Fedus et al. (2019) "Hyperbolic Discounting and Learning over Multiple Horizons"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from finrl_pro_ds.agents.deepscalper.networks import (
    MicroEncoder,
    MicroEncoderMLP,
    MicroEncoderFlat,
    MicroEncoderTCN,
    MacroEncoder,
    _tc_align,
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


def _make_dueling_heads(fusion_dim: int, n_actions: int, noisy_sigma0: float):
    """Create a pair of NoisyLinear dueling head layers (V + A streams)."""
    head_hidden = max(fusion_dim // 2, 64)
    return nn.ModuleDict({
        "value_fc1": NoisyLinear(fusion_dim, head_hidden, sigma0=noisy_sigma0),
        "value_fc2": NoisyLinear(head_hidden, 1, sigma0=noisy_sigma0),
        "adv_fc1": NoisyLinear(fusion_dim, head_hidden, sigma0=noisy_sigma0),
        "adv_fc2": NoisyLinear(head_hidden, n_actions, sigma0=noisy_sigma0),
    })


def _dueling_forward(heads: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
    """Apply dueling V + A heads to quantile-embedded input.

    Args:
        heads: ModuleDict with value_fc1/2, adv_fc1/2.
        x: (B, N, fusion_dim) quantile-embedded features.
    Returns:
        Q_tau: (B, N, n_actions)
    """
    v = torch.relu(heads["value_fc1"](x))
    v = heads["value_fc2"](v)  # (B, N, 1)
    a = torch.relu(heads["adv_fc1"](x))
    a = heads["adv_fc2"](a)  # (B, N, n_actions)
    return v + a - a.mean(dim=-1, keepdim=True)


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

    Multi-Horizon Mode (multi_horizon=True):
        Two sets of dueling heads sharing the same encoder + fusion + quantile
        embedding. Each horizon has independent V/A streams trained with
        different discount factors (gamma_short, gamma_long). Action selection
        blends: Q = alpha * Q_short + (1-alpha) * Q_long.

        This teaches the agent to balance short-term momentum ("am I making
        money now?") with long-term strategic value ("is this direction sound
        over the next several swings?").

    Args:
        micro_config: Config dict for MicroEncoder.
        macro_config: Config dict for MacroEncoder.
        fusion_dim: Dimension of fusion layer output.
        n_actions: Number of discrete actions.
        embedding_dim: Cosine basis dimension for quantile embedding.
        noisy_sigma0: Initial noise scale for NoisyLinear.
        multi_horizon: Enable dual short/long horizon heads.
    """

    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        n_actions: int = 3,
        embedding_dim: int = 64,
        noisy_sigma0: float = 0.5,
        multi_horizon: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.multi_horizon = multi_horizon

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

        # Fusion dimensions (TC-aligned for Tensor Core utilization)
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        private_size = micro_config.get("private_input_size", 5)
        self.private_size = private_size
        fusion_in_raw = micro_out_dim + macro_out_dim + private_size
        fusion_in_dim = _tc_align(fusion_in_raw)
        self._fusion_pad = fusion_in_dim - fusion_in_raw

        # Fusion layer (shared across horizons)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        # Quantile embedding (shared across horizons)
        self.quantile_embedding = QuantileEmbedding(
            embedding_dim=embedding_dim, output_dim=fusion_dim
        )

        # Dueling heads
        head_hidden = max(fusion_dim // 2, 64)

        if multi_horizon:
            # Short-horizon heads: captures immediate momentum
            self.heads_short = _make_dueling_heads(fusion_dim, n_actions, noisy_sigma0)
            # Long-horizon heads: captures strategic value
            self.heads_long = _make_dueling_heads(fusion_dim, n_actions, noisy_sigma0)
            # Backward compat aliases (point to short horizon for single-head code paths)
            self.value_fc1 = self.heads_short["value_fc1"]
            self.value_fc2 = self.heads_short["value_fc2"]
            self.adv_fc1 = self.heads_short["adv_fc1"]
            self.adv_fc2 = self.heads_short["adv_fc2"]
        else:
            # Single-horizon (original architecture)
            self.value_fc1 = NoisyLinear(fusion_dim, head_hidden, sigma0=noisy_sigma0)
            self.value_fc2 = NoisyLinear(head_hidden, 1, sigma0=noisy_sigma0)
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

    def _encode_and_fuse(
        self,
        micro_in: torch.Tensor,
        private_in: torch.Tensor,
        macro_in: torch.Tensor,
        tau: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Shared encoder + fusion + quantile embedding.

        Returns:
            x: (B, N, fusion_dim) — quantile-embedded fusion output
            fused: (B, fusion_dim) — raw fusion (for vol head)
            pred_vol: (B, 1) — auxiliary volatility prediction
            new_hidden: LSTM state
        """
        h_micro, new_hidden = self.micro_encoder(micro_in, hidden)
        h_macro = self.macro_encoder(macro_in)
        private_last = private_in[:, -1, :]
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        if self._fusion_pad > 0:
            combined = F.pad(combined, (0, self._fusion_pad))
        fused = self.fusion(combined)
        tau_embed = self.quantile_embedding(tau)
        x = fused.unsqueeze(1) * tau_embed
        pred_vol = self.vol_head(fused)
        return x, fused, pred_vol, new_hidden

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
                   In multi_horizon mode, returns SHORT horizon Q-values.
                   Use forward_dual() for both horizons.
            pred_vol: (B, 1) volatility prediction
            new_hidden: Updated LSTM hidden state (or None)
        """
        x, fused, pred_vol, new_hidden = self._encode_and_fuse(
            micro_in, private_in, macro_in, tau, hidden
        )

        if self.multi_horizon:
            q_tau = _dueling_forward(self.heads_short, x)
        else:
            v = torch.relu(self.value_fc1(x))
            v = self.value_fc2(v)
            a = torch.relu(self.adv_fc1(x))
            a = self.adv_fc2(a)
            q_tau = v + a - a.mean(dim=-1, keepdim=True)

        return q_tau, pred_vol, new_hidden

    def forward_dual(
        self,
        micro_in: torch.Tensor,
        private_in: torch.Tensor,
        macro_in: torch.Tensor,
        tau: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Multi-horizon forward pass — returns both short and long Q-values.

        Only valid when multi_horizon=True.

        Returns:
            q_short: (B, N, n_actions) short-horizon quantile Q-values
            q_long:  (B, N, n_actions) long-horizon quantile Q-values
            pred_vol: (B, 1) volatility prediction
            new_hidden: Updated LSTM hidden state
        """
        x, fused, pred_vol, new_hidden = self._encode_and_fuse(
            micro_in, private_in, macro_in, tau, hidden
        )
        q_short = _dueling_forward(self.heads_short, x)
        q_long = _dueling_forward(self.heads_long, x)
        return q_short, q_long, pred_vol, new_hidden

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        if self.multi_horizon:
            for heads in [self.heads_short, self.heads_long]:
                for layer in heads.values():
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()
        else:
            self.value_fc1.reset_noise()
            self.value_fc2.reset_noise()
            self.adv_fc1.reset_noise()
            self.adv_fc2.reset_noise()
