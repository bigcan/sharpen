"""Replay Buffer for off-policy agents."""

import numpy as np
from typing import Dict


class ReplayBuffer:
    """Simple Replay Buffer for off-policy agents."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            "state": self.state[ind],
            "action": self.action[ind],
            "next_state": self.next_state[ind],
            "reward": self.reward[ind],
            "not_done": self.not_done[ind],
        }
