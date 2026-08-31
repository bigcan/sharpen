"""
Flat Numpy Replay Buffer — Pre-allocated, zero-copy uniform sampling.

Replaces the Python-list ReplayBuffer with contiguous numpy arrays.
Eliminates per-object overhead: 2M entries uses ~22 GB instead of ~60-100 GB.

Drop-in replacement for ReplayBuffer (same push/sample/__len__ interface).
"""

import numpy as np


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
        '_regime_codes',
    )

    def __init__(
        self,
        capacity: int,
        micro_shape: tuple[int, ...] = (15, 30),
        macro_shape: tuple[int, ...] = (15,),
        private_shape: tuple[int, ...] = (15, 3),
        action_shape: tuple[int, ...] = (3,),
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

        # RCRP: PRISM regime codes for regime-balanced replay sampling.
        # -1 = unknown (no PRISM data). 0-8 = GAHMM composite code.
        self._regime_codes = np.full(capacity, -1, dtype=np.int8)

    def push(self, state: dict, action, reward: float,
             next_state: dict, done: bool, aux_target: float = 0.0,
             regime_code: int = -1):
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
        self._regime_codes[i] = regime_code

        # Advance circular pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        # FIX BUG-09: Log when buffer first wraps (oldest experiences being overwritten)
        if self._ptr == 0 and self._size == self.capacity:
            import logging
            logging.getLogger(__name__).info(
                f"Replay buffer full ({self.capacity}). Oldest transitions now being overwritten.",
            )

    def push_batch(
        self,
        states: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: dict[str, np.ndarray],
        dones: np.ndarray,
        aux_targets: np.ndarray,
        regime_codes: np.ndarray | None = None,
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

        # Default regime codes to -1 (unknown) if not provided
        if regime_codes is None:
            rc = np.full(n, -1, dtype=np.int8)
        else:
            rc = regime_codes

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
            self._regime_codes[s] = rc
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
            self._regime_codes[ptr:cap] = rc[:first]
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
            self._regime_codes[:second] = rc[first:]

        self._ptr = (ptr + n) % cap
        self._size = min(self._size + n, cap)

    def sample(self, batch_size: int) -> tuple[dict, np.ndarray, np.ndarray,
                                                dict, np.ndarray, np.ndarray]:
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
                          hold_ratio: float = 0.5) -> tuple[dict, np.ndarray,
                                                             np.ndarray, dict,
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

    def sample_regime_balanced(
        self,
        batch_size: int,
        mode: str = "balanced",
    ) -> tuple[dict, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        """Sample with regime-balanced distribution using PRISM composite codes.

        Rebalances experience replay so underrepresented vol regimes (HIGH_VOL)
        get equal training exposure. Falls back to uniform if regime codes are
        unavailable (all -1).

        Args:
            batch_size: Total samples to return.
            mode: Sampling strategy:
                - "balanced": Equal draws from each vol-regime bucket (LOW/NORMAL/HIGH).
                - "inverse_freq": Over-sample rare regimes proportional to 1/frequency.
                - "transition_boosted": 2x weight for regime-transition bars.

        Returns:
            Same format as sample().
        """
        size = self._size
        codes = self._regime_codes[:size]

        # Vol regime from composite code: LOW=0,3,6  NORMAL=1,4,7  HIGH=2,5,8
        # -1 = unknown
        vol_regime = np.where(codes >= 0, codes % 3, -1)

        low_idx = np.where(vol_regime == 0)[0]
        normal_idx = np.where(vol_regime == 1)[0]
        high_idx = np.where(vol_regime == 2)[0]

        known_buckets = [b for b in [low_idx, normal_idx, high_idx] if len(b) >= 2]

        # Fallback: not enough regime-labeled data
        if len(known_buckets) < 2:
            return self.sample(batch_size)

        if mode == "balanced":
            # Equal draws per vol-regime bucket
            per_bucket = batch_size // len(known_buckets)
            remainder = batch_size - per_bucket * len(known_buckets)
            parts = []
            for i, bucket in enumerate(known_buckets):
                n = per_bucket + (1 if i < remainder else 0)
                parts.append(bucket[np.random.randint(0, len(bucket), size=n)])
            indices = np.concatenate(parts)

        elif mode == "inverse_freq":
            # Weight inversely proportional to frequency
            total_known = sum(len(b) for b in known_buckets)
            weights = [total_known / max(len(b), 1) for b in known_buckets]
            w_sum = sum(weights)
            counts = [max(1, int(batch_size * w / w_sum)) for w in weights]
            # Fix rounding
            counts[-1] = batch_size - sum(counts[:-1])
            parts = []
            for bucket, n in zip(known_buckets, counts):
                parts.append(bucket[np.random.randint(0, len(bucket), size=n)])
            indices = np.concatenate(parts)

        elif mode == "transition_boosted":
            # 2x weight for regime-change bars
            transitions = np.zeros(size, dtype=bool)
            transitions[1:] = codes[1:] != codes[:-1]
            transitions[0] = False
            # Only count real transitions (not -1 → -1)
            transitions &= (codes >= 0)
            trans_idx = np.where(transitions)[0]
            non_trans_idx = np.where(~transitions & (codes >= 0))[0]

            if len(trans_idx) < 2 or len(non_trans_idx) < 2:
                return self.sample(batch_size)

            # 2x weight for transitions → ~33% transitions if 15% of data
            n_trans = min(batch_size // 3, len(trans_idx))
            n_non_trans = batch_size - n_trans
            t_idx = trans_idx[np.random.randint(0, len(trans_idx), size=n_trans)]
            nt_idx = non_trans_idx[np.random.randint(0, len(non_trans_idx), size=n_non_trans)]
            indices = np.concatenate([t_idx, nt_idx])

        else:
            return self.sample(batch_size)

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
            self._dones.nbytes + self._aux_targets.nbytes +
            self._regime_codes.nbytes
        )

    def __len__(self) -> int:
        return self._size
