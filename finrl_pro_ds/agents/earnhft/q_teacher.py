"""
Q-Teacher: Offline backward dynamic programming on LOB data.

Computes optimal action-value table via hindsight, used for reward shaping
of low-level agents during training (EarnHFT Stage 1).

Reference: EarnHFT (AAAI 2024), Section 3.1 — Demonstration Generation.
"""
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def sell_value(
    price_info: pd.Series,
    position: float,
    commission_fee: float = 0.000175,
    max_punish: float = 1e12,
) -> float:
    """Walk bid levels to compute sell proceeds (net of commission).

    Args:
        price_info: Row from LOB DataFrame with bid_price_1..5, bid_vol_1..5.
        position: Amount to sell (positive).
        commission_fee: Trading commission rate.
        max_punish: Penalty for exceeding book depth.

    Returns:
        Net sell proceeds after commission.
    """
    value = 0.0
    remaining = position
    last_level = 1

    for i in range(1, 6):
        last_level = i
        bid_size = price_info[f"bid_vol_{i}"]
        bid_price = price_info[f"bid_price_{i}"]

        if remaining <= bid_size or i == 5:
            break
        else:
            remaining -= bid_size
            value += bid_price * bid_size

    if last_level == 5 and remaining > 1e-12 and remaining > price_info["bid_vol_5"]:
        value -= max_punish  # Penalty for exceeding book depth
    else:
        value += price_info[f"bid_price_{last_level}"] * remaining

    return value * (1.0 - commission_fee)


def buy_value(
    price_info: pd.Series,
    position: float,
    commission_fee: float = 0.000175,
    max_punish: float = 1e12,
) -> float:
    """Walk ask levels to compute buy cost (including commission).

    Args:
        price_info: Row from LOB DataFrame with ask_price_1..5, ask_vol_1..5.
        position: Amount to buy (positive).
        commission_fee: Trading commission rate.
        max_punish: Penalty for exceeding book depth.

    Returns:
        Total buy cost including commission.
    """
    value = 0.0
    remaining = position
    last_level = 1

    for i in range(1, 6):
        last_level = i
        ask_size = price_info[f"ask_vol_{i}"]
        ask_price = price_info[f"ask_price_{i}"]

        if remaining <= ask_size or i == 5:
            break
        else:
            remaining -= ask_size
            value += ask_price * ask_size

    if last_level == 5 and remaining > 1e-12 and remaining > price_info["ask_vol_5"]:
        value += max_punish  # Penalty for exceeding book depth
    else:
        value += price_info[f"ask_price_{last_level}"] * remaining

    return value * (1.0 + commission_fee)


class QTeacher:
    """Compute optimal Q-table via backward DP on LOB data.

    The Q-table has shape (T, num_actions, num_actions) where:
        - T = number of timesteps
        - axis 1 = previous action (position level at t-1)
        - axis 2 = current action (position level at t)

    Each action index maps to a position level:
        position = action_index / (num_actions - 1) * max_holding

    Args:
        num_actions: Number of discrete position levels (e.g., 5 → {0, 0.25, 0.5, 0.75, 1.0}).
        max_holding: Maximum position size.
        commission_fee: Trading commission rate.
        reward_scale: Multiply raw rewards for numerical stability.
        gamma: Discount factor for future Q-values.
        max_punish: Penalty for exceeding LOB depth.
    """

    def __init__(
        self,
        num_actions: int = 5,
        max_holding: float = 1.0,
        commission_fee: float = 0.000175,
        reward_scale: float = 1000.0,
        gamma: float = 0.999,
        max_punish: float = 1e12,
    ):
        self.num_actions = num_actions
        self.max_holding = max_holding
        self.commission_fee = commission_fee
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.max_punish = max_punish
        self.scale_factor = num_actions - 1

    def compute_q_table(self, df: pd.DataFrame) -> np.ndarray:
        """Compute Q-table via backward DP.

        Runs from T-1 backward to 0. At each step, computes the reward for
        transitioning from prev_action to curr_action using LOB execution costs,
        then adds discounted future value.

        Args:
            df: LOB DataFrame with columns bid_price_1..5, bid_vol_1..5,
                ask_price_1..5, ask_vol_1..5.

        Returns:
            Q-table of shape (T, num_actions, num_actions).
        """
        T = len(df)
        n = self.num_actions
        q_table = np.zeros((T, n, n), dtype=np.float64)

        logger.info(f"Computing Q-table: T={T}, num_actions={n}, gamma={self.gamma}")

        # Backward pass: t iterates from 2 to T (accessing df.iloc[-t] and df.iloc[-t+1])
        for t in range(2, T + 1):
            idx = T - t  # Current index in q_table (goes from T-2 down to 0)
            current_row = df.iloc[idx]
            future_row = df.iloc[idx + 1]

            bid1_price = current_row["bid_price_1"]
            bid1_price_next = future_row["bid_price_1"]

            for prev_action in range(n):
                for curr_action in range(n):
                    prev_pos = prev_action / self.scale_factor * self.max_holding
                    curr_pos = curr_action / self.scale_factor * self.max_holding

                    if curr_action > prev_action:
                        # Buying: increase position
                        pos_change = (curr_action - prev_action) / self.scale_factor * self.max_holding
                        cost = buy_value(
                            current_row, pos_change,
                            self.commission_fee, self.max_punish,
                        )
                        curr_val = bid1_price * prev_pos
                        future_val = bid1_price_next * curr_pos
                        reward = future_val - (curr_val + cost)
                    elif curr_action < prev_action:
                        # Selling: decrease position
                        pos_change = (prev_action - curr_action) / self.scale_factor * self.max_holding
                        proceeds = sell_value(
                            current_row, pos_change,
                            self.commission_fee, self.max_punish,
                        )
                        curr_val = bid1_price * prev_pos
                        future_val = bid1_price_next * curr_pos
                        reward = future_val + proceeds - curr_val
                    else:
                        # Hold: no trade
                        curr_val = bid1_price * prev_pos
                        future_val = bid1_price_next * curr_pos
                        reward = future_val - curr_val

                    reward *= self.reward_scale

                    # Bellman: Q(t, prev, curr) = reward + gamma * max_a' Q(t+1, curr, a')
                    q_table[idx][prev_action][curr_action] = (
                        reward + self.gamma * np.max(q_table[idx + 1][curr_action][:])
                    )

            if t % 10000 == 0:
                logger.info(f"  Backward DP progress: {t}/{T} steps")

        logger.info("Q-table computation complete.")
        return q_table

    def get_optimal_actions(self, q_table: np.ndarray) -> np.ndarray:
        """Extract optimal action sequence from Q-table via forward greedy.

        Starting from action=0 (zero position), greedily selects the best
        action at each timestep.

        Args:
            q_table: Shape (T, num_actions, num_actions).

        Returns:
            Optimal actions of shape (T,), dtype int.
        """
        T = q_table.shape[0]
        actions = np.zeros(T, dtype=np.int64)

        prev_action = 0  # Start with zero position
        for t in range(T):
            best_action = np.argmax(q_table[t, prev_action, :])
            actions[t] = best_action
            prev_action = best_action

        return actions

    def get_q_advantage(
        self, q_table: np.ndarray, t: int, prev_action: int
    ) -> np.ndarray:
        """Return Q-advantage for reward shaping.

        Advantage = Q(t, s, a) - V(t, s) where V = max_a Q.
        This is always <= 0, with the best action having advantage 0.

        Args:
            q_table: Shape (T, num_actions, num_actions).
            t: Current timestep.
            prev_action: Previous action index.

        Returns:
            Advantage vector of shape (num_actions,).
        """
        q_values = q_table[t, prev_action, :]
        return q_values - q_values.max()
