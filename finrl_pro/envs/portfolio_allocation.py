"""Portfolio Allocation Environment (Phase 5).

This environment trains an agent to output portfolio weights for a set of assets.
It supports:
- Vector Action Space (weights)
- Transaction Costs
- Macro-Economic Features (VIX, Yields)
- Reward Engineering (Log Return, risk penalties)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Optional

class PortfolioAllocationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        price_ary: np.ndarray,       # (T, N) Close prices
        tech_ary: np.ndarray,        # (T, N*F) Technical features
        macro_ary: Optional[np.ndarray] = None, # (T, M) Macro features
        initial_capital: float = 1e6,
        transaction_cost_pct: float = 0.001, # 10 bps
        reward_scaling: float = 1.0,
        lookback: int = 1, # Number of past steps to stack? (Simpler: 1 step state)
        turbulence_ary: Optional[np.ndarray] = None,
        turbulence_thresh: float = 30.0,
    ) -> None:
        super().__init__()
        
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.macro_ary = macro_ary.astype(np.float32) if macro_ary is not None else np.zeros((len(price_ary), 0), dtype=np.float32)
        
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        
        # Dimensions
        self.n_assets = self.price_ary.shape[1]
        self.n_tech = self.tech_ary.shape[1]
        self.n_macro = self.macro_ary.shape[1]
        self.max_step = self.price_ary.shape[0] - 1
        
        # Precompute Returns for efficiency
        # returns[t] is return from t to t+1
        # price[t+1] / price[t] - 1
        self.returns_ary = np.zeros_like(self.price_ary)
        self.returns_ary[:-1] = (self.price_ary[1:] / self.price_ary[:-1]) - 1
        self.returns_ary[-1] = 0.0 # Last step has no future return
        
        # Action Space: Weights for N assets
        # We expect the agent to output raw logits or pre-normalized weights.
        # Typically standard RL (PPO) outputs squashed tanh (-1, 1) or unbounded.
        # We will apply Softmax in the step function or wrapper.
        # Here we define Box(0, 1) as the valid range for *input* to step.
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation Space
        # [Weights(N), Tech(N*F), Macro(M), Last_Return(N)]
        self.state_dim = self.n_assets + self.n_tech + self.n_macro + self.n_assets
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # State variables
        self.day = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.weights[0] = 1.0 # Default: Invest in first asset? Or equal weight? 
        # Let's default to Equal Weight
        self.weights[:] = 1.0 / self.n_assets
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        
        return self._get_state(), {}
        
    def step(self, action: np.ndarray):
        # 1. Normalize Action to Weights (Softmax or L1)
        # If action comes from PPO (tanh), it might be negative.
        # If action comes from geometric, it might be positive.
        # We assume action is "desired weights" but needs normalization.
        # Robust normalization:
        exp_action = np.exp(action)
        target_weights = exp_action / np.sum(exp_action)
        
        # 2. Calculate Transaction Cost
        # Cost is paid on the *change* in allocation * portfolio_value
        # But wait, we have *drifted* weights from the previous day's price move.
        # W_drift = W_prev * (1 + r) / (1 + R_port)
        # For simplicity in this version:
        # Assume we rebalance from `self.weights` to `target_weights`.
        # Turnover = sum(abs(target_weights - self.weights))
        # Cost = Turnover * cost_pct
        
        turnover = np.sum(np.abs(target_weights - self.weights))
        cost = turnover * self.transaction_cost_pct
        
        # 3. Calculate Portfolio Return
        # We hold `target_weights` for the period `day` to `day+1`
        asset_returns = self.returns_ary[self.day]
        gross_return = np.sum(target_weights * asset_returns)
        net_return = gross_return - cost
        
        # 4. Update State
        self.portfolio_value *= (1 + net_return)
        
        # Update weights for next step (Drift)
        # New weight_i = w_i * (1+r_i) / (1+R)
        # This is the *starting* weight for the next rebalance.
        if 1 + gross_return > 0:
            self.weights = (target_weights * (1 + asset_returns)) / (1 + gross_return)
        else:
            # If portfolio goes to 0 or negative, reset weights (bankruptcy)
            self.weights = target_weights 
            
        self.day += 1
        done = self.day >= self.max_step
        
        # 5. Reward
        # Log Return is standard for growth maximization
        reward = np.log(1 + net_return) * self.reward_scaling
        
        state = self._get_state()
        
        return state, reward, done, False, {
            "portfolio_value": self.portfolio_value,
            "return": net_return,
            "turnover": turnover,
            "cost": cost
        }
        
    def _get_state(self):
        # Combine components
        # 1. Current Weights (N)
        # 2. Technical Features (N*F)
        # 3. Macro Features (M)
        # 4. Last Returns (N) - useful for momentum
        
        # Safety check for end of buffer
        d = min(self.day, self.max_step)
        
        obs = np.concatenate([
            self.weights,
            self.tech_ary[d],
            self.macro_ary[d],
            self.returns_ary[d-1] if d > 0 else np.zeros(self.n_assets)
        ]).astype(np.float32)
        
        return obs
