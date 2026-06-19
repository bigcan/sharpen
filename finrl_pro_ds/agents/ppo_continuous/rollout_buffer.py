"""On-policy rollout buffer for continuous PPO on V7 observations.

Stores T steps x B envs of (scale_stack, private, action, log_prob, value,
reward, done), computes GAE, and yields shuffled minibatches.

Differs from ``ppo_scalper.rollout_buffer.RolloutBuffer``:
  * Observation is the V7 multi-scale format: scale_stack (N, W, F) + private
    (private_dim,), NOT the legacy micro/macro/private DeepScalper format.
  * Actions are float32 (continuous Box(-1, 1)), not int64 (Discrete).
  * No qty action mask (continuous action space has no per-action masking).
"""
from typing import Generator

import numpy as np


class ContinuousRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        n_scales: int,
        window_size: int,
        features_per_scale: int,
        private_dim: int,
        action_dim: int = 1,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        T, B = rollout_steps, num_envs
        self.scales = np.zeros((T, B, n_scales, window_size, features_per_scale), dtype=np.float32)
        self.private = np.zeros((T, B, private_dim), dtype=np.float32)
        self.actions = np.zeros((T, B, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, B), dtype=np.float32)
        self.values = np.zeros((T, B), dtype=np.float32)
        self.rewards = np.zeros((T, B), dtype=np.float32)
        self.dones = np.zeros((T, B), dtype=np.float32)

        self.advantages = np.zeros((T, B), dtype=np.float32)
        self.returns = np.zeros((T, B), dtype=np.float32)

    def store(
        self,
        scale_stack: np.ndarray,   # (B, N, W, F)
        private: np.ndarray,       # (B, private_dim)
        actions: np.ndarray,       # (B, action_dim)
        log_probs: np.ndarray,     # (B,)
        rewards: np.ndarray,       # (B,)
        values: np.ndarray,        # (B,)
        dones: np.ndarray,         # (B,)  terminal flags only (Bellman bootstrap)
    ):
        assert self.pos < self.rollout_steps, (
            f"Buffer full at pos={self.pos}, rollout_steps={self.rollout_steps}. "
            f"Call reset() before storing more data."
        )
        t = self.pos
        self.scales[t] = scale_stack
        self.private[t] = private
        act = np.asarray(actions, dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 1)
        self.actions[t] = act
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self.pos += 1
        if self.pos == self.rollout_steps:
            self.full = True

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ):
        """Standard GAE(gamma, lambda). Identical recursion to the discrete buffer."""
        assert self.full, "Buffer must be full before computing GAE"
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def iterate_minibatches(
        self, batch_size: int, shuffle: bool = True,
    ) -> Generator[dict[str, np.ndarray], None, None]:
        assert self.full, "Buffer must be full before iterating"
        total = self.rollout_steps * self.num_envs

        flat_scales = self.scales.reshape(total, *self.scales.shape[2:])
        flat_private = self.private.reshape(total, *self.private.shape[2:])
        flat_actions = self.actions.reshape(total, self.action_dim)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_values = self.values.reshape(total)

        # Batch-level advantage normalization (crucial for PPO stability).
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "scales": flat_scales[idx],
                "private": flat_private[idx],
                "actions": flat_actions[idx],
                "old_log_probs": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
                "old_values": flat_values[idx],
            }

    def reset(self):
        self.pos = 0
        self.full = False
