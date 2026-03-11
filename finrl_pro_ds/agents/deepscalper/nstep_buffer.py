"""
N-Step Return Buffer — Accumulates multi-step returns before pushing to replay.

Sits between the trainer and the replay buffer. For each environment, maintains
a deque of recent transitions. When n transitions accumulate (or an episode ends),
computes the n-step discounted return and pushes a single (s_0, a_0, R_n, s_n, done)
transition to the underlying replay buffer.

Usage:
    nstep = NStepBuffer(n=3, gamma=0.99, num_envs=24)
    # In training loop:
    nstep.add(obs, actions, rewards, next_obs, dones, aux_targets, replay_buffer)

When n=1, degrades to direct push (no overhead).

Reference: Mnih et al. (2016) "Asynchronous Methods for Deep RL" — n-step returns.
"""
import numpy as np
from collections import deque


class NStepBuffer:
    """Per-environment n-step return accumulator.

    Maintains a deque of length n for each environment. When full (or on done),
    computes R_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1} and pushes
    (s_0, a_0, R_n, s_n, done_any) to the replay buffer.

    Args:
        n: Number of steps for multi-step returns (1 = standard 1-step).
        gamma: Discount factor.
        num_envs: Number of parallel environments.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99, num_envs: int = 1):
        self.n = n
        self.gamma = gamma
        self.num_envs = num_envs
        # Per-env transition deques
        self._buffers = [deque(maxlen=n) for _ in range(num_envs)]
        # Precompute discount powers: [1, gamma, gamma^2, ..., gamma^(n-1)]
        self._discounts = np.array([gamma ** i for i in range(n)], dtype=np.float32)

    def add(
        self,
        obs: dict,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: dict,
        dones: np.ndarray,
        aux_targets: np.ndarray,
        replay_buffer,
        resets: np.ndarray = None,
    ):
        """Add a batch of transitions from vectorized env.

        For each environment, accumulates into the n-step deque.
        When the deque is full OR when done/reset, flushes the accumulated
        n-step transition to the replay buffer.

        Args:
            obs: Dict with 'micro', 'macro', 'private' — each (num_envs, ...)
            actions: (num_envs,) or (num_envs, 1) action indices
            rewards: (num_envs,) float rewards
            next_obs: Dict with same structure as obs
            dones: (num_envs,) float dones (1.0 = terminal, stored in replay)
            aux_targets: (num_envs,) float auxiliary targets
            replay_buffer: The underlying replay buffer (FlatReplayBuffer or PER)
            resets: (num_envs,) float — 1.0 if env was reset (term OR trunc).
                    FIX GMO1-04: Triggers n-step flush on truncation to prevent
                    cross-episode reward mixing. If None, falls back to dones.
        """
        from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer

        num_envs = self.num_envs
        is_flat = isinstance(replay_buffer, FlatReplayBuffer)
        # FIX GMO1-04: Use resets (term|trunc) for flush, dones (term only) for stored flag
        flush_signals = resets if resets is not None else dones

        for i in range(num_envs):
            # Extract per-env transition
            transition = {
                "obs": {k: v[i] for k, v in obs.items()},
                "action": actions[i],
                "reward": float(rewards[i]),
                "next_obs": {k: v[i] for k, v in next_obs.items()},
                "done": float(dones[i]),  # term|trunc (R8-AUD-01: both zero Q-bootstrap)
                "aux_target": float(aux_targets[i]),
            }
            self._buffers[i].append(transition)

            buf = self._buffers[i]
            should_flush = float(flush_signals[i]) > 0.5

            # Flush conditions: buffer full OR episode ended (term or trunc)
            if len(buf) == self.n or should_flush:
                self._flush_env(i, replay_buffer, is_flat)

            # On episode end, flush any remaining partial n-step transitions
            if should_flush:
                while len(buf) > 0:
                    self._flush_env(i, replay_buffer, is_flat)

    def _flush_env(self, env_idx: int, replay_buffer, is_flat: bool):
        """Flush one n-step transition for a single environment."""
        buf = self._buffers[env_idx]
        if len(buf) == 0:
            return

        k = len(buf)  # Actual steps (may be < n at episode boundary)

        # Compute n-step return: R = sum_{i=0}^{k-1} gamma^i * r_i
        rewards = np.array([buf[j]["reward"] for j in range(k)], dtype=np.float32)
        n_step_return = float(np.dot(self._discounts[:k], rewards))

        # done_any: if any step in the window was terminal
        done_any = max(buf[j]["done"] for j in range(k))

        # s_0 and s_n (or s_k if truncated)
        s0 = buf[0]["obs"]
        a0 = buf[0]["action"]
        s_n = buf[-1]["next_obs"]
        aux = buf[0]["aux_target"]

        # Push to replay buffer
        if is_flat:
            # Wrap as batch of 1
            states = {k: v[np.newaxis] for k, v in s0.items()}
            next_states = {k: v[np.newaxis] for k, v in s_n.items()}
            _action = np.array([a0]) if np.isscalar(a0) else a0[np.newaxis]
            if _action.ndim == 1:
                _action = _action.reshape(1, -1)
            replay_buffer.push_batch(
                states,
                _action,
                np.array([n_step_return], dtype=np.float32),
                next_states,
                np.array([done_any], dtype=np.float32),
                np.array([aux], dtype=np.float32),
            )
        else:
            # PER: per-item push
            replay_buffer.push(
                s0, a0, n_step_return, s_n, done_any > 0.5, aux
            )

        # Remove the oldest transition (shift the window)
        buf.popleft()

    def flush_all(self, replay_buffer):
        """Flush all remaining partial n-step transitions (call at episode end)."""
        from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer
        is_flat = isinstance(replay_buffer, FlatReplayBuffer)
        for i in range(self.num_envs):
            while len(self._buffers[i]) > 0:
                self._flush_env(i, replay_buffer, is_flat)

    def reset_env(self, env_idx: int):
        """Clear the buffer for a single environment (on episode reset)."""
        self._buffers[env_idx].clear()
