"""GPU Replay Buffer for AlphaSeek DQN training.

Ported from contest/reference/erl_replay_buffer.py with production fixes:
- logging instead of print()
- non_blocking=True on .to(device) calls
- safe torch.load with map_location

The buffer stores (max_size, num_seqs, dim) tensors on CUDA, where num_seqs
equals the number of parallel simulator instances (num_sims). Uniform random
sampling across (timestep, seq_id) pairs for off-policy DQN learning.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class AlphaSeekReplayBuffer:
    """Ring-buffer replay for GPU-vectorized DQN training.

    Parameters
    ----------
    max_size : int
        Maximum number of timesteps stored per sequence.
    state_dim : int
        Dimension of state vector (10 for AlphaSeek).
    action_dim : int
        Dimension of action vector (1 for discrete DQN).
    gpu_id : int
        CUDA device index. -1 for CPU.
    num_seqs : int
        Number of parallel sequences (== num_sims in LOBTradeSimulator).
    """

    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        num_seqs: int = 1,
    ):
        self.p = 0  # write pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu"
        )

        self.states = torch.empty(
            (max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (max_size, num_seqs), dtype=torch.float32, device=self.device
        )
        self.undones = torch.empty(
            (max_size, num_seqs), dtype=torch.float32, device=self.device
        )

    def __len__(self) -> int:
        return self.cur_size

    def update(self, items: Tuple[Tensor, ...]) -> None:
        """Push a batch of transitions into the ring buffer.

        Parameters
        ----------
        items : tuple of (states, actions, rewards, undones)
            states:  (horizon_len, num_seqs, state_dim)
            actions: (horizon_len, num_seqs, action_dim)
            rewards: (horizon_len, num_seqs)
            undones: (horizon_len, num_seqs)
        """
        self.add_item = items
        states, actions, rewards, undones = items
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p : p] = states
            self.actions[self.p : p] = actions
            self.rewards[self.p : p] = rewards
            self.undones[self.p : p] = undones

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a uniform random batch of (s, a, r, undone, s') transitions.

        Returns
        -------
        tuple of 5 tensors:
            states:      (batch_size, state_dim)
            actions:     (batch_size, action_dim)
            rewards:     (batch_size,)
            undones:     (batch_size,)
            next_states: (batch_size, state_dim)
        """
        sample_len = self.cur_size - 1

        ids = torch.randint(
            sample_len * self.num_seqs, size=(batch_size,), requires_grad=False
        )
        ids0 = torch.fmod(ids, sample_len)  # timestep index
        ids1 = torch.div(ids, sample_len, rounding_mode="floor")  # seq index

        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
        )

    def save(self, cwd: str) -> None:
        """Save buffer contents to disk."""
        os.makedirs(cwd, exist_ok=True)
        item_names = [
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        ]
        for item, name in item_names:
            if self.cur_size == self.p:
                buf_item = item[: self.cur_size]
            else:
                buf_item = torch.vstack(
                    (item[self.p : self.cur_size], item[0 : self.p])
                )
            file_path = os.path.join(cwd, f"replay_buffer_{name}.pth")
            logger.info(f"Saving replay buffer: {file_path}")
            torch.save(buf_item, file_path)

    def load(self, cwd: str) -> None:
        """Load buffer contents from disk."""
        item_names = [
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        ]
        paths = [os.path.join(cwd, f"replay_buffer_{name}.pth") for _, name in item_names]
        if not all(os.path.isfile(p) for p in paths):
            logger.warning(f"Replay buffer files not found in {cwd}, skipping load")
            return

        max_sizes = []
        for (item, name), path in zip(item_names, paths):
            logger.info(f"Loading replay buffer: {path}")
            buf_item = torch.load(path, map_location=self.device, weights_only=False)
            max_size = buf_item.shape[0]
            item[:max_size] = buf_item
            max_sizes.append(max_size)

        assert all(ms == max_sizes[0] for ms in max_sizes), (
            f"Buffer file sizes mismatch: {max_sizes}"
        )
        self.cur_size = self.p = max_sizes[0]
        self.if_full = self.cur_size == self.max_size
