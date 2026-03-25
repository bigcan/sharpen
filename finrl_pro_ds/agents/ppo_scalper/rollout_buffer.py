"""
On-policy Rollout Buffer for PPO.

Pre-allocated numpy arrays for fixed-length trajectory storage.
Supports GAE computation and minibatch iteration.
"""
import numpy as np
from typing import Dict, Generator, Optional, Tuple


class RolloutBuffer:
    """
    Fixed-length rollout buffer for on-policy PPO training.

    Stores T steps × B environments of experience, computes GAE advantages,
    and yields shuffled minibatches for K-epoch PPO updates.

    Args:
        rollout_steps: Number of steps per rollout (T)
        num_envs: Number of parallel environments (B)
        micro_shape: Shape of micro observation per env, e.g. (15, 30)
        private_shape: Shape of private state per env, e.g. (15, 3)
        macro_shape: Shape of macro observation per env, e.g. (15,)
        n_action_branches: Number of action branches (2 for price+qty)
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        micro_shape: Tuple[int, ...] = (15, 30),
        private_shape: Tuple[int, ...] = (15, 5),
        macro_shape: Tuple[int, ...] = (15,),
        n_action_branches: int = 1,
        n_qty_actions: int = 6,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.pos = 0
        self.full = False

        T, B = rollout_steps, num_envs

        # Observations
        self.micro = np.zeros((T, B, *micro_shape), dtype=np.float32)
        self.private = np.zeros((T, B, *private_shape), dtype=np.float32)
        self.macro = np.zeros((T, B, *macro_shape), dtype=np.float32)

        # Actions & policy outputs
        self.actions = np.zeros((T, B, n_action_branches), dtype=np.int64)
        self.log_probs = np.zeros((T, B), dtype=np.float32)
        self.values = np.zeros((T, B), dtype=np.float32)

        # Action masks (B4 fix: store masks for evaluate_actions during training)
        self.qty_masks = np.ones((T, B, n_qty_actions), dtype=np.float32)

        # Env outputs
        self.rewards = np.zeros((T, B), dtype=np.float32)
        self.dones = np.zeros((T, B), dtype=np.float32)

        # Computed during GAE
        self.advantages = np.zeros((T, B), dtype=np.float32)
        self.returns = np.zeros((T, B), dtype=np.float32)

    def store(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        qty_masks: Optional[np.ndarray] = None,
    ):
        """
        Store one timestep of data from all environments.

        Args:
            obs: Dict with 'micro' (B, W, 27), 'private' (B, W, 3), 'macro' (B, M)
            actions: (B, 2) int actions
            log_probs: (B,) float log-probabilities
            rewards: (B,) float rewards
            values: (B,) float V(s) estimates
            dones: (B,) float done flags (1.0 = done)
        """
        assert self.pos < self.rollout_steps, (
            f"Buffer full at pos={self.pos}, rollout_steps={self.rollout_steps}. "
            f"Call reset() before storing more data."
        )

        t = self.pos
        self.micro[t] = obs["micro"]
        self.private[t] = obs["private"]
        self.macro[t] = obs["macro"]
        # FIX T2-SHAPE: PPO Discrete(6) returns actions as (B,) flat array,
        # but buffer shape is (T, B, n_action_branches=1). Reshape to match.
        act = np.asarray(actions)
        if act.ndim == 1 and self.actions.ndim == 3:
            act = act.reshape(-1, 1)
        self.actions[t] = act
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        if qty_masks is not None:
            self.qty_masks[t] = qty_masks
        else:
            self.qty_masks[t] = 1.0  # All actions valid

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
        """
        Compute Generalized Advantage Estimation (GAE) in-place.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff
            last_values: (B,) V(s_{T}) for bootstrapping
            last_dones: (B,) done flags for the last step
        """
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
        self, batch_size: int, shuffle: bool = True
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Yield minibatches of flattened rollout data.

        Flattens (T, B) → (T*B,) and shuffles. Each minibatch is a dict
        of numpy arrays ready for torch conversion.

        Args:
            batch_size: Size of each minibatch
            shuffle: Whether to shuffle indices

        Yields:
            Dict with keys: micro, private, macro, actions, old_log_probs,
                            advantages, returns, old_values
        """
        assert self.full, "Buffer must be full before iterating"

        total = self.rollout_steps * self.num_envs

        # Flatten (T, B, ...) -> (T*B, ...)
        flat_micro = self.micro.reshape(total, *self.micro.shape[2:])
        flat_private = self.private.reshape(total, *self.private.shape[2:])
        flat_macro = self.macro.reshape(total, *self.macro.shape[2:])
        flat_actions = self.actions.reshape(total, -1)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_values = self.values.reshape(total)
        flat_qty_masks = self.qty_masks.reshape(total, -1)

        # Normalize advantages (crucial for PPO stability)
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        # Generate indices
        indices = np.arange(total)
        if shuffle:
            np.random.shuffle(indices)

        # Yield minibatches
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]

            yield {
                "micro": flat_micro[idx],
                "private": flat_private[idx],
                "macro": flat_macro[idx],
                "actions": flat_actions[idx],
                "old_log_probs": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
                "old_values": flat_values[idx],
                "qty_masks": flat_qty_masks[idx],
            }

    def reset(self):
        """Reset buffer position for next rollout. Does NOT zero arrays."""
        self.pos = 0
        self.full = False
