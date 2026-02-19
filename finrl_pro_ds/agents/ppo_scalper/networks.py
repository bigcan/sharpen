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
    MicroEncoder, MicroEncoderMLP, MicroEncoderTCN, MacroEncoder
)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with Discrete(6) action space (Tier 2).

    Architecture:
        MicroEncoder (MLP/LSTM) ──┐
                                  ├── Fusion (Linear → LayerNorm → LeakyReLU)
        MacroEncoder (MLP)      ──┤
        Private State (5)       ──┘
                                  │
                             fusion (256-dim)
                        ┌─────────┴─────────┐
                      Actor               Critic
                   softmax(6)              V(s)

    V5: encoder_type in micro_config selects stateless MLP vs LSTM.
    """

    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        action_space_dims: int = 6, # Tier 2: Discrete(6)
        **kwargs
    ):
        super().__init__()

        # Select encoder based on config (V5: defaults to LSTM for backward compat)
        micro_config = dict(micro_config)  # Don't mutate caller's dict
        encoder_type = micro_config.pop("encoder_type", "lstm")
        # AUDIT FIX A2: Extract private_input_size before passing to encoder
        # T2.2: Private size expanded to 5
        private_size = micro_config.pop("private_input_size", 5)
        if encoder_type == "mlp":
            self.micro_encoder = MicroEncoderMLP(**micro_config)
        elif encoder_type == "tcn":
            self.micro_encoder = MicroEncoderTCN(**micro_config)
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
        self.action_dim = action_space_dims

        # Actor head — single head for Discrete(6)
        self.actor = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, self.action_dim)
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
        for module in [self.fusion, self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        # Policy output layers: small gain for initial near-uniform policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

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
            private_in: (B, W, 5) private state
            macro_in: (B, M) macro features
            qty_mask: Optional (B, 6) mask, 1=valid, 0=invalid
            deterministic: If True, take argmax actions

        Returns:
            actions: (B,) int64 — discrete action index
            log_probs: (B,) float
            values: (B,) float — V(s) estimate
            entropy: (B,) float
        """
        # Encode
        h_micro, _ = self.micro_encoder(micro_in, None)
        h_macro = self.macro_encoder(macro_in)

        # Private state: take last timestep (B, W, 5) -> (B, 5)
        private_last = private_in[:, -1, :]

        # Fusion
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        features = self.fusion(combined)

        # Actor logits
        logits = self.actor(features)  # (B, action_dim)

        # Action masking
        # FIX FIND-PPO-01: Guard against all actions being masked (degenerate state)
        if qty_mask is not None:
            valid_count = qty_mask.sum(dim=1)
            if (valid_count == 0).any():
                # Unmask Hold (action 2) as safe fallback for fully-masked samples
                qty_mask = qty_mask.clone()
                qty_mask[valid_count == 0, 2] = 1.0
            logits = logits.masked_fill(qty_mask == 0, float('-inf'))

        # Distribution
        dist = Categorical(logits=logits)

        # Sample or argmax
        if deterministic:
            action = logits.argmax(dim=1)
        else:
            action = dist.sample()

        # Log-probs and entropy
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        # Critic
        value = self.critic(features).squeeze(-1)  # (B,)

        return action, log_probs, value, entropy

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
            actions: (B,) int64
            qty_mask: Optional (B, 6) mask

        Returns:
            log_probs: (B,) float
            values: (B,) float
            entropy: (B,) float
        """
        # Encode (stateless)
        h_micro, _ = self.micro_encoder(micro_in, None)
        h_macro = self.macro_encoder(macro_in)
        private_last = private_in[:, -1, :]

        # Fusion
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        features = self.fusion(combined)

        # Actor logits
        logits = self.actor(features)

        # Action masking
        # FIX FIND-PPO-01/03: Same guard in evaluate_actions to ensure consistency
        if qty_mask is not None:
            valid_count = qty_mask.sum(dim=1)
            if (valid_count == 0).any():
                qty_mask = qty_mask.clone()
                qty_mask[valid_count == 0, 2] = 1.0
            logits = logits.masked_fill(qty_mask == 0, float('-inf'))

        # Distribution
        dist = Categorical(logits=logits)

        # Log-probs for the given actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return log_probs, value, entropy
