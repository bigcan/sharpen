"""
Low-Level Agent for EarnHFT — Single-branch Discrete PPO.

Adapted from ppo_scalper/ppo_agent.py:
  - Single actor head (1 Categorical) instead of 2
  - No quantity mask
  - Actions are (B,) int64 instead of (B, 2)

Reuses MicroEncoder and MacroEncoder from deepscalper/networks.py.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_TORCH = False


# ============================================================================
# ROLLOUT BUFFER (numpy-only, no torch dependency)
# ============================================================================
class DiscreteRolloutBuffer:
    """Rollout buffer for single-branch discrete PPO.

    Stores transitions and computes GAE.

    Args:
        rollout_steps: Steps per rollout.
        num_envs: Number of parallel environments.
        micro_shape: (W, micro_dim)
        private_shape: (W, private_dim)
        macro_shape: (macro_dim,) or None
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        micro_shape: Tuple[int, ...],
        private_shape: Tuple[int, ...],
        macro_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.pos = 0

        self.micro = np.zeros((rollout_steps, num_envs, *micro_shape), dtype=np.float32)
        self.private = np.zeros((rollout_steps, num_envs, *private_shape), dtype=np.float32)
        if macro_shape is not None:
            self.macro = np.zeros((rollout_steps, num_envs, *macro_shape), dtype=np.float32)
        else:
            self.macro = None
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    @property
    def full(self) -> bool:
        return self.pos >= self.rollout_steps

    def store(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ):
        """Store one transition."""
        t = self.pos
        self.micro[t] = obs["micro"]
        self.private[t] = obs["private"]
        if self.macro is not None and "macro" in obs:
            self.macro[t] = obs["macro"]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self.pos += 1

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ):
        """Compute GAE advantages and returns."""
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = last_values
                next_dones = last_dones
            else:
                next_values = self.values[t + 1]
                next_dones = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_values * (1.0 - next_dones) - self.values[t]
            gae = delta + gamma * gae_lambda * (1.0 - next_dones) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def iterate_minibatches(self, batch_size: int):
        """Yield shuffled minibatches."""
        total = self.rollout_steps * self.num_envs
        indices = np.random.permutation(total)

        # Flatten
        flat_micro = self.micro.reshape(total, *self.micro.shape[2:])
        flat_private = self.private.reshape(total, *self.private.shape[2:])
        flat_macro = self.macro.reshape(total, *self.macro.shape[2:]) if self.macro is not None else None
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            batch = {
                "micro": flat_micro[idx],
                "private": flat_private[idx],
                "actions": flat_actions[idx],
                "log_probs": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
            }
            if flat_macro is not None:
                batch["macro"] = flat_macro[idx]
            yield batch

    def reset(self):
        """Reset buffer position."""
        self.pos = 0


# ============================================================================
# TORCH-DEPENDENT CLASSES (network + agent)
# ============================================================================
if HAS_TORCH:

    class DiscretePPOActorCritic(nn.Module):
        """Actor-Critic network with single discrete action head.

        Reuses MicroEncoder (LSTM) and MacroEncoder (MLP) from DeepScalper,
        but outputs a single Categorical distribution instead of 2 branches.

        Args:
            micro_config: Config dict for MicroEncoder.
            macro_config: Config dict for MacroEncoder.
            fusion_dim: Dimension of fused feature vector.
            num_actions: Number of discrete actions.
            head_hidden: Hidden size for actor/critic heads.
        """

        def __init__(
            self,
            micro_config: Dict,
            macro_config: Dict,
            fusion_dim: int = 128,
            num_actions: int = 5,
            head_hidden: int = 64,
        ):
            super().__init__()
            self.num_actions = num_actions

            # Reuse encoders from DeepScalper
            from finrl_pro_ds.agents.deepscalper.networks import MicroEncoder, MacroEncoder
            self.micro_encoder = MicroEncoder(**micro_config)
            self.macro_encoder = MacroEncoder(**macro_config)

            micro_out = micro_config["hidden_size"]
            macro_out = macro_config["hidden_sizes"][-1]
            private_dim = micro_config.get("private_input_size", 3)

            # Fusion: micro output + macro output + private state
            total_in = micro_out + macro_out + private_dim
            self.fusion = nn.Sequential(
                nn.Linear(total_in, fusion_dim),
                nn.ReLU(),
            )

            # Single actor head
            self.actor = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.Tanh(),
                nn.Linear(head_hidden, num_actions),
            )

            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.Tanh(),
                nn.Linear(head_hidden, 1),
            )

        def forward(
            self,
            micro: torch.Tensor,
            private: torch.Tensor,
            macro: Optional[torch.Tensor] = None,
            hidden: Optional[Tuple] = None,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
            """Forward pass.

            Args:
                micro: (B, W, micro_dim) LOB features.
                private: (B, W, private_dim) private state.
                macro: (B, macro_dim) macro features, or None.
                hidden: LSTM hidden state tuple.
                deterministic: If True, use argmax instead of sampling.

            Returns:
                actions: (B,) int64
                log_probs: (B,) float
                values: (B,) float
                entropy: (B,) float
                new_hidden: Updated LSTM hidden state
            """
            B = micro.shape[0]

            # Encode
            micro_out, new_hidden = self.micro_encoder(micro, private, hidden)

            # Private: take last timestep
            private_last = private[:, -1, :]

            # Macro: encode if available, else zeros
            if macro is not None and hasattr(self, 'macro_encoder'):
                macro_out = self.macro_encoder(macro)
            else:
                macro_out_dim = self.macro_encoder.layers[-1].out_features
                macro_out = torch.zeros(B, macro_out_dim, device=micro.device)

            # Fuse
            features = self.fusion(torch.cat([micro_out, macro_out, private_last], dim=1))

            # Actor
            logits = self.actor(features)
            dist = Categorical(logits=logits)

            if deterministic:
                actions = logits.argmax(dim=1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Critic
            values = self.critic(features).squeeze(-1)

            return actions, log_probs, values, entropy, new_hidden

        def evaluate_actions(
            self,
            micro: torch.Tensor,
            private: torch.Tensor,
            macro: Optional[torch.Tensor],
            actions: torch.Tensor,
            hidden: Optional[Tuple] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Re-evaluate log-probs and values for stored actions.

            Args:
                micro: (B, W, micro_dim)
                private: (B, W, private_dim)
                macro: (B, macro_dim) or None
                actions: (B,) int64 — stored actions to evaluate
                hidden: Optional LSTM hidden state

            Returns:
                log_probs: (B,)
                values: (B,)
                entropy: (B,)
            """
            B = micro.shape[0]

            micro_out, _ = self.micro_encoder(micro, private, hidden)
            private_last = private[:, -1, :]

            if macro is not None:
                macro_out = self.macro_encoder(macro)
            else:
                macro_out_dim = self.macro_encoder.layers[-1].out_features
                macro_out = torch.zeros(B, macro_out_dim, device=micro.device)

            features = self.fusion(torch.cat([micro_out, macro_out, private_last], dim=1))

            logits = self.actor(features)
            dist = Categorical(logits=logits)

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic(features).squeeze(-1)

            return log_probs, values, entropy


    class DiscretePPOAgent:
        """Single-branch discrete PPO agent for EarnHFT low-level pool.

        Args:
            network_config: Config for DiscretePPOActorCritic.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_eps: PPO clip epsilon.
            vf_coef: Value function loss coefficient.
            ent_coef: Entropy bonus coefficient.
            max_grad_norm: Gradient clipping norm.
            device: Torch device.
        """

        def __init__(
            self,
            network_config: Dict[str, Any],
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            device: str = "cpu",
        ):
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_eps = clip_eps
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.max_grad_norm = max_grad_norm
            self.device = torch.device(device)
            self.num_actions = network_config.get("num_actions", 5)

            self.network = DiscretePPOActorCritic(**network_config).to(self.device)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

            self._hidden_state = None

        def predict(
            self,
            micro: torch.Tensor,
            private: torch.Tensor,
            macro: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Predict actions.

            Args:
                micro: (B, W, micro_dim)
                private: (B, W, private_dim)
                macro: (B, macro_dim) or None
                deterministic: Use argmax.

            Returns:
                actions: (B,) int64
                log_probs: (B,) float32
                values: (B,) float32
            """
            self.network.eval()
            with torch.no_grad():
                micro = micro.to(self.device)
                private = private.to(self.device)
                if macro is not None:
                    macro = macro.to(self.device)

                actions, log_probs, values, _, new_hidden = self.network(
                    micro, private, macro,
                    hidden=self._hidden_state,
                    deterministic=deterministic,
                )
                self._hidden_state = new_hidden

            return (
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy(),
            )

        def train_step(
            self,
            buffer: DiscreteRolloutBuffer,
            n_epochs: int = 4,
            batch_size: int = 64,
        ) -> Dict[str, float]:
            """PPO training step on rollout buffer.

            Returns:
                Dict with policy_loss, value_loss, entropy, total_loss.
            """
            self.network.train()

            # Normalize advantages
            adv_mean = buffer.advantages.mean()
            adv_std = buffer.advantages.std() + 1e-8
            buffer.advantages = (buffer.advantages - adv_mean) / adv_std

            total_metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}
            n_updates = 0

            for epoch in range(n_epochs):
                for batch in buffer.iterate_minibatches(batch_size):
                    micro = torch.tensor(batch["micro"], device=self.device)
                    private = torch.tensor(batch["private"], device=self.device)
                    macro = torch.tensor(batch["macro"], device=self.device) if "macro" in batch else None
                    old_actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
                    old_log_probs = torch.tensor(batch["log_probs"], device=self.device)
                    advantages = torch.tensor(batch["advantages"], device=self.device)
                    returns = torch.tensor(batch["returns"], device=self.device)

                    new_log_probs, new_values, entropy = self.network.evaluate_actions(
                        micro, private, macro, old_actions
                    )

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(new_values, returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_metrics["policy_loss"] += policy_loss.item()
                    total_metrics["value_loss"] += value_loss.item()
                    total_metrics["entropy"] += entropy.mean().item()
                    total_metrics["total_loss"] += loss.item()
                    n_updates += 1

            # Average
            for k in total_metrics:
                total_metrics[k] /= max(n_updates, 1)

            return total_metrics

        def reset_hidden_state(self):
            """Reset LSTM hidden state."""
            self._hidden_state = None

        def save(self, path: str):
            """Save checkpoint."""
            torch.save({
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, path)
            logger.info(f"Saved DiscretePPO checkpoint to {path}")

        def load(self, path: str):
            """Load checkpoint."""
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.network.load_state_dict(ckpt["network"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info(f"Loaded DiscretePPO checkpoint from {path}")
