"""Trade execution logic for FinRL Pro."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from finrl_pro.execution.broker import BrokerClient

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """Orchestrates data fetching, inference, and order execution."""

    def __init__(
        self,
        broker: BrokerClient,
        agent: Any,
        symbols: List[str],
    ) -> None:
        self.broker = broker
        self.agent = agent
        self.symbols = symbols

    def run_cycle(self) -> None:
        """Execute one trading cycle."""
        logger.info("Starting execution cycle...")

        # 1. Sync Account
        try:
            account = self.broker.get_account()
            equity = float(account.get("equity", 0.0))
            buying_power = float(account.get("buying_power", 0.0))
            logger.info(f"Account Equity: {equity:.2f}, BP: {buying_power:.2f}")
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return

        # 2. Get Current Positions
        current_positions = {}
        try:
            positions = self.broker.get_positions()
            for p in positions:
                current_positions[p["symbol"]] = float(p.get("qty", 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return

        # 3. Data Fetch & Feature Engineering (TODO)
        # For now, we use random state as placeholder since Live Data feed is not connected
        # In production, this would call a DataProvider and FeatureAssembler
        logger.warning("Using MOCK state for inference (Live Data not implemented)")
        state_dim = 0
        if hasattr(self.agent, "state_dim"):
            state_dim = self.agent.state_dim
        elif hasattr(self.agent, "observation_space"):
             state_dim = self.agent.observation_space.shape[0]
        else:
            state_dim = len(self.symbols) * 5 # Fallback guess
            
        state = np.random.random((1, state_dim))

        # 4. Inference
        try:
            action = self.agent.select_action(state)
            if isinstance(action, tuple):
                action = action[0] # Unwrap (action, log_prob)
            logger.info(f"Agent raw action: {action}")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return

        # 5. Allocation Logic (Assume Softmax/Weights for Phase 5 Multi-Asset)
        # If action is allocation weights, sum should be 1.
        # If action is raw logits, apply softmax.
        # Here we assume the agent output is ready-to-use (or we apply simple normalization)
        
        # Handle batch dim if present
        if len(action.shape) > 1:
            action = action[0]

        # Heuristic: if action looks like logits (any negative), apply softmax
        if np.any(action < 0):
            weights = self._softmax(action)
        else:
            weights = action / (np.sum(action) + 1e-9)

        logger.info(f"Target Weights: {weights}")

        # 6. Rebalancing (Simple Long-Only)
        if len(weights) != len(self.symbols):
            logger.error(f"Dimension mismatch: {len(weights)} weights vs {len(self.symbols)} symbols")
            return

        for i, symbol in enumerate(self.symbols):
            target_weight = weights[i]
            target_value = equity * target_weight
            
            # Get current price (TODO: use live quote)
            # Mock price: 100.0
            current_price = 100.0 
            
            target_qty = int(target_value / current_price)
            current_qty = current_positions.get(symbol, 0)
            
            diff_qty = target_qty - current_qty
            
            if diff_qty == 0:
                continue
                
            side = "buy" if diff_qty > 0 else "sell"
            qty = abs(diff_qty)
            
            logger.info(f"{symbol}: Curr {current_qty} -> Tgt {target_qty} (Diff {diff_qty})")
            
            # Execute
            # try:
            #     self.broker.submit_order(symbol, qty, side)
            # except Exception as e:
            #     logger.error(f"Order failed for {symbol}: {e}")

        logger.info("Cycle complete.")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
