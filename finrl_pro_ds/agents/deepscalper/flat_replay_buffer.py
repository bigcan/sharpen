"""
Flat Numpy Replay Buffer — Pre-allocated, zero-copy uniform sampling.

Replaces the Python-list ReplayBuffer with contiguous numpy arrays.
Eliminates per-object overhead: 2M entries uses ~22 GB instead of ~60-100 GB.

Drop-in replacement for ReplayBuffer (same push/sample/__len__ interface).
"""
import numpy as np
from typing import Dict, Tuple


class FlatReplayBuffer:
    """Replay buffer backed by pre-allocated numpy arrays.

    Memory layout: one contiguous array per field, indexed by circular pointer.
    No Python objects stored — everything is raw numpy with zero GC pressure.

    Args:
        capacity: Maximum number of transitions
        micro_shape: Shape of micro observation, e.g. (15, 30)
        macro_shape: Shape of macro observation, e.g. (15,)
        private_shape: Shape of private observation, e.g. (15, 2)
        action_shape: Shape of action vector, e.g. (3,)
    """

    __slots__ = (
        'capacity', '_ptr', '_size',
        '_micro', '_macro', '_private',
        '_next_micro', '_next_macro', '_next_private',
        '_actions', '_rewards', '_dones', '_aux_targets',
    )

    def __init__(
        self,
        capacity: int,
        micro_shape: Tuple[int, ...] = (15, 30),
        macro_shape: Tuple[int, ...] = (15,),
        private_shape: Tuple[int, ...] = (15, 3),
        action_shape: Tuple[int, ...] = (3,),
        action_dtype=np.int64,
    ):
        self.capacity = capacity
        self._ptr = 0
        self._size = 0

        # State arrays (float32)
        self._micro = np.zeros((capacity, *micro_shape), dtype=np.float32)
        self._macro = np.zeros((capacity, *macro_shape), dtype=np.float32)
        self._private = np.zeros((capacity, *private_shape), dtype=np.float32)

        # Next-state arrays (float32)
        self._next_micro = np.zeros((capacity, *micro_shape), dtype=np.float32)
        self._next_macro = np.zeros((capacity, *macro_shape), dtype=np.float32)
        self._next_private = np.zeros((capacity, *private_shape), dtype=np.float32)

        # Scalar arrays
        self._actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._aux_targets = np.zeros(capacity, dtype=np.float32)

    def push(self, state: Dict, action, reward: float,
             next_state: Dict, done: bool, aux_target: float = 0.0):
        """Store a transition by writing directly to pre-allocated arrays.

        Interface is identical to the old ReplayBuffer.push() — accepts
        dict observations with 'micro', 'macro', 'private' keys.
        """
        i = self._ptr

        # State
        self._micro[i] = state["micro"]
        self._macro[i] = state["macro"]
        self._private[i] = state["private"]

        # Next state
        self._next_micro[i] = next_state["micro"]
        self._next_macro[i] = next_state["macro"]
        self._next_private[i] = next_state["private"]

        # Scalars
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._aux_targets[i] = aux_target

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        # FIX BUG-09: Log when buffer first wraps (oldest experiences being overwritten)
        if self._ptr == 0 and self._size == self.capacity:
            import logging
            logging.getLogger(__name__).info(
                f"Replay buffer full ({self.capacity}). Oldest transitions now being overwritten."
            )

    def push_batch(
        self,
        states: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: Dict[str, np.ndarray],
        dones: np.ndarray,
        aux_targets: np.ndarray,
    ):
        """Store N transitions at once using numpy slice assignment.

        All inputs have leading batch dimension N, e.g. states["micro"] is (N, W, 30).
        Handles circular wraparound when ptr + N > capacity.
        """
        n = len(rewards)
        if n == 0:
            return

        ptr = self._ptr
        cap = self.capacity

        if ptr + n <= cap:
            # Simple case: fits without wrapping
            s = slice(ptr, ptr + n)
            self._micro[s] = states["micro"]
            self._macro[s] = states["macro"]
            self._private[s] = states["private"]
            self._next_micro[s] = next_states["micro"]
            self._next_macro[s] = next_states["macro"]
            self._next_private[s] = next_states["private"]
            self._actions[s] = actions
            self._rewards[s] = rewards
            self._dones[s] = dones
            self._aux_targets[s] = aux_targets
        else:
            # Wraparound: split into two writes
            first = cap - ptr
            # First chunk: ptr → end
            self._micro[ptr:cap] = states["micro"][:first]
            self._macro[ptr:cap] = states["macro"][:first]
            self._private[ptr:cap] = states["private"][:first]
            self._next_micro[ptr:cap] = next_states["micro"][:first]
            self._next_macro[ptr:cap] = next_states["macro"][:first]
            self._next_private[ptr:cap] = next_states["private"][:first]
            self._actions[ptr:cap] = actions[:first]
            self._rewards[ptr:cap] = rewards[:first]
            self._dones[ptr:cap] = dones[:first]
            self._aux_targets[ptr:cap] = aux_targets[:first]
            # Second chunk: 0 → remainder
            second = n - first
            self._micro[:second] = states["micro"][first:]
            self._macro[:second] = states["macro"][first:]
            self._private[:second] = states["private"][first:]
            self._next_micro[:second] = next_states["micro"][first:]
            self._next_macro[:second] = next_states["macro"][first:]
            self._next_private[:second] = next_states["private"][first:]
            self._actions[:second] = actions[first:]
            self._rewards[:second] = rewards[first:]
            self._dones[:second] = dones[first:]
            self._aux_targets[:second] = aux_targets[first:]

        self._ptr = (ptr + n) % cap
        self._size = min(self._size + n, cap)

    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray,
                                                Dict, np.ndarray, np.ndarray]:
        """Sample a random batch. Returns pre-stacked numpy arrays.

        Returns:
            (state_dict, actions, rewards, next_state_dict, dones, aux_targets)

            state_dict / next_state_dict have keys: 'micro', 'macro', 'private'
            All values are numpy arrays with batch dimension first.
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        states = {
            "micro": self._micro[indices],       # (B, 50, 27)
            "macro": self._macro[indices],        # (B, 11)
            "private": self._private[indices],    # (B, W, 2)
        }
        next_states = {
            "micro": self._next_micro[indices],
            "macro": self._next_macro[indices],
            "private": self._next_private[indices],
        }

        return (
            states,
            self._actions[indices],        # (B, 3)
            self._rewards[indices],        # (B,)
            next_states,
            self._dones[indices],          # (B,)
            self._aux_targets[indices],    # (B,)
        )

    def sample_stratified(self, batch_size: int, hold_action: int = 1,
                          hold_ratio: float = 0.5) -> Tuple[Dict, np.ndarray,
                                                             np.ndarray, Dict,
                                                             np.ndarray, np.ndarray]:
        """Sample with stratified hold/non-hold ratio.

        Ensures non-hold transitions are over-represented in each batch,
        combating the ~95% Hold domination typical in scalping environments.

        Args:
            batch_size: Total samples to return.
            hold_action: Action index for Hold (default 1 for Discrete(3)).
            hold_ratio: Fraction of batch that should be Hold transitions.
                        0.5 means 50% Hold, 50% non-Hold.

        Returns:
            Same format as sample().
        """
        size = self._size
        # Build masks over the filled portion of the buffer
        actions_flat = self._actions[:size, 0] if self._actions.ndim > 1 else self._actions[:size]
        hold_mask = (actions_flat == hold_action)
        hold_indices = np.where(hold_mask)[0]
        non_hold_indices = np.where(~hold_mask)[0]

        # Fallback to uniform if not enough non-hold transitions
        if len(non_hold_indices) < 2 or len(hold_indices) < 2:
            return self.sample(batch_size)

        n_hold = int(batch_size * hold_ratio)
        n_non_hold = batch_size - n_hold

        # Sample with replacement if pool is smaller than requested
        h_idx = hold_indices[np.random.randint(0, len(hold_indices), size=n_hold)]
        nh_idx = non_hold_indices[np.random.randint(0, len(non_hold_indices), size=n_non_hold)]

        indices = np.concatenate([h_idx, nh_idx])
        np.random.shuffle(indices)

        states = {
            "micro": self._micro[indices],
            "macro": self._macro[indices],
            "private": self._private[indices],
        }
        next_states = {
            "micro": self._next_micro[indices],
            "macro": self._next_macro[indices],
            "private": self._next_private[indices],
        }

        return (
            states,
            self._actions[indices],
            self._rewards[indices],
            next_states,
            self._dones[indices],
            self._aux_targets[indices],
        )

    def nbytes(self) -> int:
        """Total pre-allocated memory in bytes."""
        return (
            self._micro.nbytes + self._macro.nbytes + self._private.nbytes +
            self._next_micro.nbytes + self._next_macro.nbytes + self._next_private.nbytes +
            self._actions.nbytes + self._rewards.nbytes +
            self._dones.nbytes + self._aux_targets.nbytes
        )

    def __len__(self) -> int:
        return self._size
