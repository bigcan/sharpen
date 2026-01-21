"""Dynamic Stock Trading environment with arbitrary tech_dim (Pro-only).

This environment mirrors the behavior of the upstream StockTrading env but
accepts arrays directly and does not assume a fixed number (7) of technical
features per ticker. It keeps the same observation packing style and action
space semantics.
"""

from __future__ import annotations

from abc import ABC
from typing import Optional

import gymnasium as gym
import numpy as np


def _sigmoid_sign(x: np.ndarray, thresh: float) -> np.ndarray:
    s = 1 / (1 + np.exp(-(x - thresh)))
    return (s - 0.5) * 2  # range (-1, 1)


class ProStockEnv(gym.Env, ABC):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        price_ary: np.ndarray,  # (T, stock_dim)
        tech_ary: np.ndarray,   # (T, stock_dim * tech_dim)
        turbulence_ary: Optional[np.ndarray] = None,  # (T,)
        gamma: float = 0.999,
        turbulence_thresh: float = 30.0,
        min_stock_rate: float = 0.1,
        max_stock: float = 1e2,
        initial_capital: float = 1e6,
        buy_cost_pct: float = 1e-3,
        sell_cost_pct: float = 1e-3,
        reward_scaling: float = 2 ** -13,
        turnover_penalty: float = 0.0,
    ) -> None:
        assert price_ary.ndim == 2, "price_ary must be (T, stock_dim)"
        assert tech_ary.ndim == 2, "tech_ary must be (T, stock_dim * tech_dim)"
        assert price_ary.shape[0] == tech_ary.shape[0], "price/tech time lengths must match"

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_raw = (turbulence_ary.astype(np.float32) if turbulence_ary is not None
                               else np.zeros((price_ary.shape[0],), dtype=np.float32))

        self.gamma = gamma
        self.turbulence_thresh = float(turbulence_thresh)
        self.min_stock_rate = float(min_stock_rate)
        self.max_stock = float(max_stock)
        self.initial_capital = float(initial_capital)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.reward_scaling = float(reward_scaling)
        self.turnover_penalty = float(turnover_penalty)

        # Derived shapes
        self.stock_dim = int(self.price_ary.shape[1])
        assert self.tech_ary.shape[1] % self.stock_dim == 0, "tech_ary width must be multiple of stock_dim"
        self.tech_dim = int(self.tech_ary.shape[1] // self.stock_dim)
        self.max_step = int(self.price_ary.shape[0] - 1)

        # Turbulence transforms
        self.turbulence_bool = (self.turbulence_raw > self.turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (_sigmoid_sign(self.turbulence_raw, self.turbulence_thresh) * 2 ** -5).astype(np.float32)

        # State composition
        # amount + (turbulence, turbulence_bool) + (price, stock, stock_cd)*stock_dim + tech_flat
        self.state_dim = 1 + 2 + 3 * self.stock_dim + self.tech_ary.shape[1]
        self.action_dim = self.stock_dim
        self.if_discrete = False

        # Runtime state
        self.day = 0
        self.amount = 0.0
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.stocks_cd = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = 0.0
        self.initial_total_asset = 0.0
        self.episode_return = 0.0
        self.gamma_reward = 0.0

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * self.state_dim, dtype=np.float32), # Use -np.inf for more generic bounds
            high=np.array([np.inf] * self.state_dim, dtype=np.float32), # Use np.inf for more generic bounds
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1] * self.action_dim, dtype=np.float32),
            high=np.array([1] * self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed) # Important for Gymnasium API

        self.day = 0
        price = self.price_ary[self.day]
        self.stocks = np.zeros_like(self.stocks)
        self.stocks_cd = np.zeros_like(self.stocks_cd)
        self.amount = self.initial_capital - float((self.stocks * price).sum())
        self.total_asset = self.amount + float((self.stocks * price).sum())
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.episode_return = 0.0
        return self._get_state(price), {}

    def step(self, actions):
        actions = (np.asarray(actions, dtype=np.float32) * self.max_stock).astype(np.int32)
        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cd += 1

        total_trade_value = 0.0
        for i in range(self.action_dim):
            action = actions[i]
            if action > 0:  # buy
                available_amount = self.amount // (price[i] * (1 + self.buy_cost_pct))
                delta = min(available_amount, action)
                if delta > 0:
                    trade_val = float(price[i] * delta)
                    self.amount -= trade_val * (1 + self.buy_cost_pct)
                    self.stocks[i] += delta
                    self.stocks_cd[i] = 0
                    total_trade_value += trade_val
            elif action < 0:  # sell
                delta = min(-action, self.stocks[i])
                if delta > 0:
                    trade_val = float(price[i] * delta)
                    self.amount += trade_val * (1 - self.sell_cost_pct)
                    self.stocks[i] -= delta
                    self.stocks_cd[i] = 0
                    total_trade_value += trade_val

        next_total_asset = self.amount + float((self.stocks * price).sum())
        # Reward = asset_change - turnover_penalty
        # We scale the penalty by reward_scaling so it's comparable to the asset change
        reward = (next_total_asset - self.total_asset) * self.reward_scaling
        reward -= (total_trade_value * self.turnover_penalty * self.reward_scaling)
        
        self.total_asset = next_total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        self.episode_return = self.total_asset / self.initial_total_asset

        done = self.day == self.max_step
        state = self._get_state(price)
        return state, reward, done, False, {"total_assets": self.total_asset}

    def _get_state(self, price_row: np.ndarray) -> np.ndarray:
        amount_scaled = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        tech_flat = self.tech_ary[self.day]
        return np.hstack(
            (
                amount_scaled,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price_row * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech_flat,
            )
        ).astype(np.float32)

