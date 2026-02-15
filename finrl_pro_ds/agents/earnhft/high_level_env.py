"""
High-Level Router Environment for EarnHFT.

Operates at minute-level granularity. The router selects which low-level
agent from the pool should execute for the next minute.

Observation: minute-level aggregated features.
Action: select agent index from the pool.
Reward: cumulative PnL achieved by the selected agent over the next minute.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)


class EarnHFTHighLevelEnv(gym.Env):
    """High-level router environment for agent selection.

    The router observes minute-level features and selects which low-level
    agent should trade for the next minute. The reward is the cumulative
    PnL achieved by that agent.

    Args:
        minute_features: Minute-level feature matrix, shape (N_minutes, feature_dim).
        minute_pnls: PnL per minute per agent, shape (N_minutes, pool_size).
            Pre-computed by running each pool agent on the data.
        pool_size: Number of agents in the pool.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        minute_features: np.ndarray,
        minute_pnls: np.ndarray,
        pool_size: int = 3,
    ):
        super().__init__()

        self.minute_features = minute_features.astype(np.float32)
        self.minute_pnls = minute_pnls.astype(np.float32)
        self.pool_size = pool_size
        self.n_minutes = len(minute_features)
        self.feature_dim = minute_features.shape[1]

        assert minute_pnls.shape == (self.n_minutes, pool_size), (
            f"minute_pnls shape {minute_pnls.shape} != ({self.n_minutes}, {pool_size})"
        )

        self.action_space = spaces.Discrete(pool_size)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.feature_dim,), dtype=np.float32
        )

        self._ptr = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._ptr = 0
        obs = self.minute_features[0]
        return obs, {"minute": 0}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Select agent and get reward.

        Args:
            action: Agent index from pool.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        agent_idx = int(action)
        reward = float(self.minute_pnls[self._ptr, agent_idx])

        self._ptr += 1
        truncated = self._ptr >= self.n_minutes
        terminated = False

        if not truncated:
            obs = self.minute_features[self._ptr]
        else:
            obs = self.minute_features[-1]  # Terminal obs

        info = {
            "minute": self._ptr,
            "selected_agent": agent_idx,
            "all_pnls": self.minute_pnls[self._ptr - 1].tolist(),
        }

        return obs, reward, terminated, truncated, info
