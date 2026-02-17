"""
PPO Actor-Critic Network for DeepScalper.

Reuses MicroEncoder and MacroEncoder from the BDQ network.
Adds Actor (two Categorical heads) and Critic (V(s)) on top of shared fusion.

V5: Supports configurable encoder_type (mlp/lstm) via micro_config.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional

from finrl_pro_ds.agents.deepscalper.networks import (
    MicroEncoder, MicroEncoderMLP, MacroEncoder
)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with Multi-Discrete action space.

    Architecture:
        MicroEncoder (MLP/LSTM) ──┐
                                  ├── Fusion (Linear → LayerNorm → LeakyReLU)
        MacroEncoder (MLP)      ──┤
                                  │
        Private State (3)       ──┘
                                  │
                             fusion (256-dim)
                        ┌─────────┼─────────┐
                   Actor Price  Actor Qty   Critic
                   softmax(5)  softmax(9)    V(s)

    V5: encoder_type in micro_config selects stateless MLP vs LSTM.
    """

    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        action_space_dims: Tuple[int, int] = (5, 9),
        **kwargs
    ):
        super().__init__()

        # Select encoder based on config (V5: defaults to LSTM for backward compat)
        micro_config = dict(micro_config)  # Don't mutate caller's dict
        encoder_type = micro_config.pop("encoder_type", "lstm")
        # AUDIT FIX A2: Extract private_input_size before passing to encoder
        # (was silently absorbed by **kwargs — fragile if kwargs removed)
        private_size = micro_config.pop("private_input_size", 3)
        if encoder_type == "mlp":
            self.micro_encoder = MicroEncoderMLP(**micro_config)
        else:
            self.micro_encoder = MicroEncoder(**micro_config)
        self.macro_encoder = MacroEncoder(**macro_config)

        # Fusion dimensions
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        self.private_size = private_size

        fusion_in_dim = micro_out_dim + macro_out_dim + private_size
        head_hidden = max(fusion_dim // 2, 64)

        # Shared fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU()
        )

        # Action dimensions
        self.price_dims, self.qty_dims = action_space_dims

        # Actor heads — output logits for Categorical distributions
        self.actor_price = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, self.price_dims)
        )

        self.actor_qty = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, self.qty_dims)
        )

        # Critic head — scalar V(s)
        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1)
        )

        # Orthogonal init (PPO best practice)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for all Linear layers. PPO standard practice."""
        for module in [self.fusion, self.actor_price, self.actor_qty, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        # Policy output layers: small gain for initial near-uniform policy
        nn.init.orthogonal_(self.actor_price[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.actor_qty[-1].weight, gain=0.01)

        # Value output layer: gain=1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(
        self,
        micro_in: torch.Tensor,
        private_in: torch.Tensor,
        macro_in: torch.Tensor,
        qty_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning actions, log-probs, values, entropy.

        Args:
            micro_in: (B, W, 30) micro-structure features
            private_in: (B, W, 3) private state
            macro_in: (B, M) macro features
            qty_mask: Optional (B, n_qty) mask, 1=valid, 0=invalid
            deterministic: If True, take argmax actions

        Returns:
            actions: (B, 2) int64 — [price_idx, qty_idx]
            log_probs: (B,) float — sum of per-branch log-probs
            values: (B,) float — V(s) estimate
            entropy: (B,) float — sum of per-branch entropies
        """
        # Encode (hidden ignored — stateless for MLP, fresh-init for LSTM)
        h_micro, _ = self.micro_encoder(micro_in, None)
        h_macro = self.macro_encoder(macro_in)

        # Private state: take last timestep (B, W, 3) -> (B, 3)
        private_last = private_in[:, -1, :]

        # Fusion
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        features = self.fusion(combined)

        # Actor logits
        price_logits = self.actor_price(features)  # (B, price_dims)
        qty_logits = self.actor_qty(features)       # (B, qty_dims)

        # Action masking on qty branch
        if qty_mask is not None:
            qty_logits = qty_logits.masked_fill(qty_mask == 0, float('-inf'))

        # Distributions
        price_dist = Categorical(logits=price_logits)
        qty_dist = Categorical(logits=qty_logits)

        # Sample or argmax
        if deterministic:
            price_action = price_logits.argmax(dim=1)
            qty_action = qty_logits.argmax(dim=1)
        else:
            price_action = price_dist.sample()
            qty_action = qty_dist.sample()

        # Log-probs (sum of independent branches)
        log_probs = price_dist.log_prob(price_action) + qty_dist.log_prob(qty_action)

        # Entropy (sum of independent branches)
        entropy = price_dist.entropy() + qty_dist.entropy()

        # Critic
        value = self.critic(features).squeeze(-1)  # (B,)

        # Actions
        actions = torch.stack([price_action, qty_action], dim=1)  # (B, 2)

        return actions, log_probs, value, entropy

    def evaluate_actions(
        self,
        micro_in: torch.Tensor,
        private_in: torch.Tensor,
        macro_in: torch.Tensor,
        actions: torch.Tensor,
        qty_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions — used during PPO training to recompute log-probs.

        Args:
            micro_in, private_in, macro_in: observation tensors
            actions: (B, 2) int64 — [price_idx, qty_idx]
            qty_mask: Optional (B, n_qty) mask

        Returns:
            log_probs: (B,) float
            values: (B,) float
            entropy: (B,) float

        V5: Encoder is always stateless (MLP) or fresh-init (LSTM fallback).
        No hidden state carried across minibatches — the 15-step window
        provides the full temporal context.
        """
        # Encode (stateless)
        h_micro, _ = self.micro_encoder(micro_in, None)
        h_macro = self.macro_encoder(macro_in)
        private_last = private_in[:, -1, :]

        # Fusion
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        features = self.fusion(combined)

        # Actor logits
        price_logits = self.actor_price(features)
        qty_logits = self.actor_qty(features)

        # Action masking
        if qty_mask is not None:
            qty_logits = qty_logits.masked_fill(qty_mask == 0, float('-inf'))

        # Distributions
        price_dist = Categorical(logits=price_logits)
        qty_dist = Categorical(logits=qty_logits)

        # Log-probs for the given actions
        log_probs = (
            price_dist.log_prob(actions[:, 0]) +
            qty_dist.log_prob(actions[:, 1])
        )

        entropy = price_dist.entropy() + qty_dist.entropy()
        value = self.critic(features).squeeze(-1)

        return log_probs, value, entropy
