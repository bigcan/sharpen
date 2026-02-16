"""
Flat Numpy Replay Buffer — Pre-allocated, zero-copy uniform sampling.

Replaces the Python-list ReplayBuffer with contiguous numpy arrays.
Eliminates per-object overhead: 2M entries uses ~22 GB instead of ~60-100 GB.

Drop-in replacement for ReplayBuffer (same push/sample/__len__ interface).
"""
import numpy as np
from typing import Dict, Tuple, Optional


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
        self._actions = np.zeros((capacity, *action_shape), dtype=np.int64)
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
