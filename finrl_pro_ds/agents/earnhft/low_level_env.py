"""
Low-Level Environment for EarnHFT.

A Gymnasium-compatible environment that wraps LOB data with:
  - Discrete action space (target position levels)
  - Q-Teacher reward shaping: reward = pnl_delta + beta * q_advantage
  - Observation: micro (LOB features) + private state + macro features

This env operates at the tick level. Each step advances one tick.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class EarnHFTLowLevelEnv(gym.Env):
    """Low-level trading environment for EarnHFT agent pool.

    Each agent in the pool trains on this env with a different beta (reward
    shaping coefficient). Beta=0 → pure PnL, larger beta → follow Q-teacher.

    Args:
        df: LOB DataFrame (one episode/chunk).
        q_table: Pre-computed Q-table from QTeacher, shape (T, num_actions, num_actions).
        num_actions: Number of discrete position levels.
        max_holding: Maximum position size.
        beta: Q-teacher reward shaping coefficient.
        commission_fee: Trading commission rate.
        window_size: Observation window for micro features.
        micro_feature_cols: LOB feature column names for micro encoder input.
        macro_features: Optional pre-computed macro features, shape (T, macro_dim).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        q_table: np.ndarray,
        num_actions: int = 5,
        max_holding: float = 1.0,
        beta: float = 1.0,
        commission_fee: float = 0.000175,
        window_size: int = 50,
        micro_feature_cols: Optional[list] = None,
        macro_features: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.df = df
        self.q_table = q_table
        self.num_actions = num_actions
        self.max_holding = max_holding
        self.beta = beta
        self.commission_fee = commission_fee
        self.window_size = window_size
        self.scale_factor = num_actions - 1

        # Default micro features: 5-level bid/ask price+vol = 20 cols + mid, spread, imbalance = ~27
        if micro_feature_cols is None:
            self.micro_feature_cols = self._default_micro_cols()
        else:
            self.micro_feature_cols = micro_feature_cols

        self.micro_dim = len(self.micro_feature_cols)
        self.private_dim = 3  # position, balance_pnl, remaining_fraction

        # Macro features
        if macro_features is not None:
            self.macro_features = macro_features
            self.macro_dim = macro_features.shape[1]
        else:
            self.macro_features = None
            self.macro_dim = 0

        # Pre-extract micro feature matrix for performance
        self._micro_data = self._extract_micro_features()
        self.T = len(df)

        # Spaces
        self.action_space = spaces.Discrete(num_actions)
        # Observation is a dict to match existing encoder interface
        obs_spaces = {
            "micro": spaces.Box(-np.inf, np.inf, shape=(window_size, self.micro_dim), dtype=np.float32),
            "private": spaces.Box(-np.inf, np.inf, shape=(window_size, self.private_dim), dtype=np.float32),
        }
        if self.macro_dim > 0:
            obs_spaces["macro"] = spaces.Box(-np.inf, np.inf, shape=(self.macro_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        # State variables
        self._ptr = 0
        self._position_action = 0  # Previous action index (starts at 0 = no position)
        self._position = 0.0
        self._cash = 0.0  # Tracks realized PnL
        self._last_portfolio_value = 0.0

    def _default_micro_cols(self) -> list:
        """Generate default LOB column list."""
        cols = []
        for i in range(1, 6):
            cols.extend([f"bid_price_{i}", f"bid_vol_{i}", f"ask_price_{i}", f"ask_vol_{i}"])
        return cols

    def _extract_micro_features(self) -> np.ndarray:
        """Extract micro features from DataFrame into numpy array."""
        available = [c for c in self.micro_feature_cols if c in self.df.columns]
        if len(available) < len(self.micro_feature_cols):
            missing = set(self.micro_feature_cols) - set(available)
            logger.warning(f"Missing micro columns, filling zeros: {missing}")
            data = np.zeros((len(self.df), len(self.micro_feature_cols)), dtype=np.float32)
            for i, col in enumerate(self.micro_feature_cols):
                if col in self.df.columns:
                    data[:, i] = self.df[col].values
            return data
        return self.df[self.micro_feature_cols].values.astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dict at current pointer."""
        t = self._ptr

        # Micro: window of LOB features
        start = max(0, t - self.window_size + 1)
        micro_window = self._micro_data[start:t + 1]
        # Pad if at beginning
        if micro_window.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - micro_window.shape[0], self.micro_dim), dtype=np.float32)
            micro_window = np.concatenate([pad, micro_window], axis=0)

        # Private state: replicated across window
        remaining_frac = 1.0 - (t / max(self.T - 1, 1))
        private_vec = np.array([
            self._position / max(self.max_holding, 1e-6),  # Normalized position
            self._cash / 1000.0,  # Normalized PnL
            remaining_frac,
        ], dtype=np.float32)
        private_window = np.tile(private_vec, (self.window_size, 1))

        obs = {
            "micro": micro_window,
            "private": private_window,
        }

        if self.macro_features is not None:
            obs["macro"] = self.macro_features[t].astype(np.float32)

        return obs

    def _portfolio_value(self) -> float:
        """Current portfolio value = cash + position * bid_price_1."""
        if self._ptr < self.T:
            bid1 = self.df.iloc[self._ptr]["bid_price_1"]
        else:
            bid1 = self.df.iloc[-1]["bid_price_1"]
        return self._cash + self._position * bid1

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to the beginning of the episode."""
        super().reset(seed=seed)

        self._ptr = 0
        self._position_action = 0
        self._position = 0.0
        self._cash = 0.0
        self._last_portfolio_value = 0.0

        obs = self._get_obs()
        info = {"step": 0, "position": 0.0}
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Args:
            action: Target position level index.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        prev_action = self._position_action
        curr_action = int(action)

        prev_pos = prev_action / self.scale_factor * self.max_holding
        curr_pos = curr_action / self.scale_factor * self.max_holding

        row = self.df.iloc[self._ptr]

        # Execute trade
        if curr_pos > prev_pos:
            # Buying
            qty = curr_pos - prev_pos
            cost = row["ask_price_1"] * qty * (1.0 + self.commission_fee)
            self._cash -= cost
        elif curr_pos < prev_pos:
            # Selling
            qty = prev_pos - curr_pos
            proceeds = row["bid_price_1"] * qty * (1.0 - self.commission_fee)
            self._cash += proceeds

        self._position = curr_pos
        self._position_action = curr_action

        # Advance pointer
        self._ptr += 1

        # Compute PnL delta
        new_portfolio_value = self._portfolio_value()
        pnl_delta = new_portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = new_portfolio_value

        # Q-teacher advantage for reward shaping
        q_adv = 0.0
        if self.beta > 0.0 and self._ptr - 1 < self.q_table.shape[0]:
            adv_vec = self.q_table[self._ptr - 1, prev_action, :] - np.max(
                self.q_table[self._ptr - 1, prev_action, :]
            )
            q_adv = adv_vec[curr_action]

        reward = pnl_delta + self.beta * q_adv

        # Terminal conditions
        terminated = False
        truncated = self._ptr >= self.T - 1

        obs = self._get_obs()
        info = {
            "step": self._ptr,
            "position": self._position,
            "pnl_delta": pnl_delta,
            "q_advantage": q_adv,
            "portfolio_value": new_portfolio_value,
        }

        return obs, float(reward), terminated, truncated, info
