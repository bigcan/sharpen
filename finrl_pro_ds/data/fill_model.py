"""
Fill Simulation Models for Market Making Environment

Progressive complexity:
  Level 1 — Deterministic Price-Cross: fills if bar H/L touches order price
  Level 2 — Probabilistic Volume-Based: fill probability ∝ volume through price level

All levels support adverse selection slippage on fills during fast price moves.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FillResult:
    """Result of fill check for one bar."""
    bid_filled: bool = False
    ask_filled: bool = False
    bid_fill_price: float = 0.0
    ask_fill_price: float = 0.0
    bid_fill_qty: float = 0.0
    ask_fill_qty: float = 0.0
    adverse_selection_cost: float = 0.0  # bps


class FillModel:
    """Base fill model interface."""

    def __init__(
        self,
        adverse_slippage_bps: float = 0.5,
        adverse_velocity_threshold: float = 0.001,
    ):
        self.adverse_slippage_bps = adverse_slippage_bps
        self.adverse_velocity_threshold = adverse_velocity_threshold

    def check_fills(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: float,
        ask_qty: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        prev_close: float,
    ) -> FillResult:
        raise NotImplementedError

    def _apply_adverse_selection(
        self,
        result: FillResult,
        bar_open: float,
        bar_close: float,
        prev_close: float,
    ) -> FillResult:
        """Apply adverse selection slippage on fills during fast price moves."""
        if prev_close <= 0:
            return result

        price_velocity = (bar_close - prev_close) / prev_close

        if result.bid_filled and price_velocity < -self.adverse_velocity_threshold:
            # Bid filled during downtick — adverse selection (bought into a drop)
            slippage = self.adverse_slippage_bps / 10000.0
            result.bid_fill_price *= (1.0 + slippage)  # worse fill (higher price)
            result.adverse_selection_cost += abs(slippage * result.bid_fill_price * result.bid_fill_qty)

        if result.ask_filled and price_velocity > self.adverse_velocity_threshold:
            # Ask filled during uptick — adverse selection (sold into a rally)
            slippage = self.adverse_slippage_bps / 10000.0
            result.ask_fill_price *= (1.0 - slippage)  # worse fill (lower price)
            result.adverse_selection_cost += abs(slippage * result.ask_fill_price * result.ask_fill_qty)

        return result


class PriceCrossFillModel(FillModel):
    """Level 1 — Deterministic Price-Cross fills.

    bid_filled = (bar_low <= bid_price)
    ask_filled = (bar_high >= ask_price)

    Optimistic — overestimates fills. Good for validating basic agent behavior.
    """

    def check_fills(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: float,
        ask_qty: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        prev_close: float,
    ) -> FillResult:
        result = FillResult()

        if bid_qty > 0 and bid_price > 0 and bar_low <= bid_price:
            result.bid_filled = True
            result.bid_fill_price = bid_price
            result.bid_fill_qty = bid_qty

        if ask_qty > 0 and ask_price > 0 and bar_high >= ask_price:
            result.ask_filled = True
            result.ask_fill_price = ask_price
            result.ask_fill_qty = ask_qty

        return self._apply_adverse_selection(result, bar_open, bar_close, prev_close)


class VolumeBasedFillModel(FillModel):
    """Level 2 — Probabilistic Volume-Based fills.

    Fill probability proportional to volume traded through the price level,
    relative to estimated queue depth.
    """

    def __init__(
        self,
        adverse_slippage_bps: float = 0.5,
        adverse_velocity_threshold: float = 0.001,
        queue_depth_multiplier: float = 5.0,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(adverse_slippage_bps, adverse_velocity_threshold)
        self.queue_depth_multiplier = queue_depth_multiplier
        self.rng = rng or np.random.default_rng()

    def check_fills(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: float,
        ask_qty: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        prev_close: float,
    ) -> FillResult:
        result = FillResult()
        price_range = bar_high - bar_low

        if price_range <= 0 or bar_volume <= 0:
            return result

        # Bid fill probability
        if bid_qty > 0 and bid_price > 0 and bar_low <= bid_price:
            fraction_through = max(0.0, bid_price - bar_low) / price_range
            volume_at_level = bar_volume * fraction_through
            queue_depth = bid_qty * self.queue_depth_multiplier
            fill_prob = min(1.0, volume_at_level / (queue_depth + bid_qty))

            if self.rng.random() < fill_prob:
                result.bid_filled = True
                result.bid_fill_price = bid_price
                result.bid_fill_qty = bid_qty

        # Ask fill probability
        if ask_qty > 0 and ask_price > 0 and bar_high >= ask_price:
            fraction_through = max(0.0, bar_high - ask_price) / price_range
            volume_at_level = bar_volume * fraction_through
            queue_depth = ask_qty * self.queue_depth_multiplier
            fill_prob = min(1.0, volume_at_level / (queue_depth + ask_qty))

            if self.rng.random() < fill_prob:
                result.ask_filled = True
                result.ask_fill_price = ask_price
                result.ask_fill_qty = ask_qty

        return self._apply_adverse_selection(result, bar_open, bar_close, prev_close)


def create_fill_model(model_type: str = "price_cross", **kwargs) -> FillModel:
    """Factory for fill models."""
    if model_type == "price_cross":
        return PriceCrossFillModel(
            adverse_slippage_bps=kwargs.get("adverse_slippage_bps", 0.5),
            adverse_velocity_threshold=kwargs.get("adverse_velocity_threshold", 0.001),
        )
    elif model_type == "volume_based":
        return VolumeBasedFillModel(
            adverse_slippage_bps=kwargs.get("adverse_slippage_bps", 0.5),
            adverse_velocity_threshold=kwargs.get("adverse_velocity_threshold", 0.001),
            queue_depth_multiplier=kwargs.get("queue_depth_multiplier", 5.0),
            rng=kwargs.get("rng"),
        )
    else:
        raise ValueError(f"Unknown fill model type: {model_type}")
