"""
Router Agent for EarnHFT — Simple DQN with experience replay.

Selects which low-level agent from the pool should execute at each
minute-level decision point. Uses epsilon-greedy exploration and
a target network for stability.
"""
import numpy as np
import random
import logging
from collections import deque
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_TORCH = False


class ReplayBuffer:
    """Simple experience replay buffer.

    Stores (state, action, reward, next_state, done) transitions.

    Args:
        capacity: Maximum buffer size.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _make_router_network(obs_dim: int, pool_size: int):
    """Create RouterDQN network. Only call when torch is available."""
    class RouterDQNNetwork(nn.Module):
        """obs_dim → 256 → ReLU → 128 → ReLU → pool_size"""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, pool_size),
            )

        def forward(self, x):
            return self.net(x)

    return RouterDQNNetwork()


class RouterDQN:
    """DQN agent for routing to low-level agents.

    Uses epsilon-greedy exploration, experience replay, and a target network.

    Args:
        obs_dim: Observation dimension.
        pool_size: Number of agents in the pool.
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Decay steps for epsilon.
        buffer_capacity: Replay buffer size.
        target_update_freq: Steps between target network updates.
        batch_size: Training batch size.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        pool_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for RouterDQN")

        self.obs_dim = obs_dim
        self.pool_size = pool_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.policy_net = _make_router_network(obs_dim, pool_size).to(self.device)
        self.target_net = _make_router_network(obs_dim, pool_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self._step_count = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for exploration."""
        decay_progress = min(1.0, self._step_count / max(self.epsilon_decay, 1))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy.

        Args:
            state: Observation vector.
            eval_mode: If True, always use greedy (no exploration).

        Returns:
            Agent index from pool.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.pool_size - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Perform one DQN training step.

        Returns:
            Loss value, or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Current Q-values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._step_count += 1

        # Update target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }, path)
        logger.info(f"Saved RouterDQN checkpoint to {path}")

    def load(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)
        logger.info(f"Loaded RouterDQN checkpoint from {path}")
