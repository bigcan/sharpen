"""Discrete position manager for AlphaSeek live trading.

Faithfully replicates the position management logic from the contest
TradeSimulator._step() (trade_simulator.py lines 102-200), including:
- Position clipping to [-max_position, +max_position]
- No flip-through: long→short must go through flat first
- Max holding force-close
- Stop-loss with best-price tracking
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionAction:
    """Result of applying an action to the current position."""
    new_position: int       # resulting position: {-1, 0, +1}
    trade_delta: int        # actual trade to execute: {-1, 0, +1}
    reason: str             # why this action was chosen
    stop_loss_triggered: bool = False
    max_holding_triggered: bool = False


class DiscretePositionManager:
    """Manages discrete position state {-1, 0, +1} with contest-matching rules.

    Rules (from TradeSimulator._step):
        1. Position clipped to [-max_position, +max_position]
        2. No flip-through: if action would flip sign, close instead
        3. Holding counter increments while in position, resets when flat
        4. Max holding: force-close after max_holding ticks
        5. Stop-loss: track best_price, force-close on drawdown > threshold
        6. On entry (flat→position): set best_price to current mid_price

    Usage:
        pm = DiscretePositionManager()
        result = pm.apply_action(action_int=1, mid_price=100000.0)
        # result.trade_delta is what to send to the broker
    """

    def __init__(
        self,
        max_position: int = 1,
        max_holding: int = 1800,  # 1 hour at 2-second ticks
        stop_loss_thresh: float = 1e-4,  # EvalTradeSimulator default
    ):
        self.max_position = max_position
        self.max_holding = max_holding
        self.stop_loss_thresh = stop_loss_thresh

        # State
        self._position: int = 0
        self._holding: int = 0
        self._best_price: float = 0.0
        self._entry_price: float = 0.0
        self._total_trades: int = 0

    @property
    def position(self) -> int:
        return self._position

    @property
    def holding(self) -> int:
        return self._holding

    @property
    def best_price(self) -> float:
        return self._best_price

    @property
    def total_trades(self) -> int:
        return self._total_trades

    def apply_action(self, action_int: int, mid_price: float) -> PositionAction:
        """Apply a discrete action and return the position change.

        Args:
            action_int: desired action delta {-1, 0, +1}
            mid_price: current midpoint price (for stop-loss tracking)

        Returns:
            PositionAction with the actual trade to execute.
        """
        if action_int not in (-1, 0, 1):
            raise ValueError(f"action_int must be in {{-1, 0, +1}}, got {action_int}")
        if mid_price <= 0:
            raise ValueError(f"mid_price must be positive, got {mid_price}")

        old_position = self._position
        reason = "agent_action"
        stop_loss_triggered = False
        max_holding_triggered = False

        # --- Step 1: Compute tentative new position ---
        tentative_position = old_position + action_int
        tentative_position = max(-self.max_position, min(self.max_position, tentative_position))
        actual_delta = tentative_position - old_position

        # --- Step 2: No flip-through (TradeSimulator lines 131-133) ---
        # If the new position would flip sign (e.g., +1 → -1), close instead
        if tentative_position * old_position < 0 and old_position != 0:
            actual_delta = -old_position  # close the position
            reason = "no_flip_through"

        # --- Step 3: Holding counter (lines 136-141) ---
        self._holding += 1
        if old_position == 0:
            self._holding = 0

        # --- Step 4: Max holding force-close (lines 137-140) ---
        if self._holding > self.max_holding and old_position != 0:
            actual_delta = -old_position
            reason = "max_holding"
            max_holding_triggered = True
            logger.info(
                "Max holding triggered: position=%d, holding=%d ticks",
                old_position, self._holding,
            )

        # --- Step 5: Stop-loss (lines 147-169) ---
        if old_position > 0:
            # Long: track highest price
            self._best_price = max(self._best_price, mid_price)
            if (self._best_price - mid_price) > self.stop_loss_thresh:
                actual_delta = -old_position
                reason = "stop_loss"
                stop_loss_triggered = True
                logger.info(
                    "Stop-loss triggered (long): best=%.6f, mid=%.6f, thresh=%.6f",
                    self._best_price, mid_price, self.stop_loss_thresh,
                )
        elif old_position < 0:
            # Short: track lowest price
            self._best_price = min(self._best_price, mid_price)
            if (mid_price - self._best_price) > self.stop_loss_thresh:
                actual_delta = -old_position
                reason = "stop_loss"
                stop_loss_triggered = True
                logger.info(
                    "Stop-loss triggered (short): best=%.6f, mid=%.6f, thresh=%.6f",
                    self._best_price, mid_price, self.stop_loss_thresh,
                )

        # --- Step 6: Apply position change ---
        new_position = old_position + actual_delta

        # --- Step 7: Entry tracking (lines 174-176) ---
        if old_position == 0 and new_position != 0:
            self._best_price = mid_price
            self._entry_price = mid_price

        # Reset holding on close
        if new_position == 0:
            self._holding = 0

        # Track trades
        if actual_delta != 0:
            self._total_trades += 1

        self._position = new_position

        return PositionAction(
            new_position=new_position,
            trade_delta=actual_delta,
            reason=reason,
            stop_loss_triggered=stop_loss_triggered,
            max_holding_triggered=max_holding_triggered,
        )

    def force_flatten(self, mid_price: float) -> PositionAction:
        """Force-close the current position (for emergency/shutdown)."""
        if self._position == 0:
            return PositionAction(
                new_position=0, trade_delta=0, reason="already_flat"
            )
        delta = -self._position
        self._position = 0
        self._holding = 0
        self._total_trades += 1
        return PositionAction(
            new_position=0, trade_delta=delta, reason="force_flatten"
        )

    def sync_position(self, exchange_position: int):
        """Sync internal state with exchange position (for reconciliation)."""
        if exchange_position != self._position:
            logger.warning(
                "Position sync: internal=%d, exchange=%d. Updating to exchange.",
                self._position, exchange_position,
            )
            self._position = exchange_position

    def get_state(self) -> dict:
        """Return state for monitoring."""
        return {
            "position": self._position,
            "holding": self._holding,
            "best_price": self._best_price,
            "entry_price": self._entry_price,
            "total_trades": self._total_trades,
        }

    def position_norm(self) -> float:
        """Normalized position for state vector (matches TradeSimulator)."""
        return float(self._position) / self.max_position

    def holding_norm(self) -> float:
        """Normalized holding for state vector (matches TradeSimulator)."""
        return float(self._holding) / self.max_holding

    def reset(self):
        """Reset to initial state."""
        self._position = 0
        self._holding = 0
        self._best_price = 0.0
        self._entry_price = 0.0

    def __repr__(self) -> str:
        return (
            f"DiscretePositionManager(pos={self._position}, "
            f"holding={self._holding}, trades={self._total_trades})"
        )
