"""Protocol v2 §4.5 trigger #6 — live cost-drift tracker.

Realized execution cost (taker/maker fee + slippage) measured against the
cost assumption the policy was *trained / configured* under. When realized
cost moves materially above the configured assumption, the HPs and the
diversity / bootstrap analysis behind the deployed bundle were done under
the wrong friction, so a Stage 2.5-R retrain is warranted (§4.5 trigger #6).

This is a **retrain trigger, NOT a §8.3 safe-mode halt.** A ``FIRED`` status
is informational: it is logged, surfaced to WandB as ``drift/cost_ratio``,
and routed to the monthly retrain-trigger watchdog. It never flattens
positions and never writes the drift kill_file. Execution being more
expensive than assumed is a model-staleness signal, not an unsafe-to-trade
signal — the parallel here is ``AgreementDecayTracker`` (silent death), not
``ActionDriftTracker`` CRIT (distribution break).

Cost model (all fractions are **one-way**, per fill, matching the env's
per-side ``taker_fee`` + ``slippage_base_bps`` cost application):

    notional       = |filled_quantity| * avg_fill_price
    fee_frac       = |fee| / notional
    slippage_frac  = |avg_fill_price - decision_price| / decision_price
    realized_frac  = fee_frac + slippage_frac

    config_frac    = taker_fee + slippage_base_bps / 1e4      (one-way assumption)

    cost_ratio     = mean(realized_frac over last `window_trades`) / config_frac

``FIRED`` when ``n_trades >= window_trades`` and ``cost_ratio >
cost_drift_ratio`` (default 1.20). The full-window requirement matches the
protocol wording ("over rolling ``cost_drift_window_trades`` trades"): the
ratio is still computed and reported during WARMUP for telemetry, but does
not fire until a complete window has accumulated.

``decision_price`` is the bar close the policy acted on (the price it
"saw"), so ``slippage_frac`` captures the execution gap between intent and
fill. Slippage is the dominant driver of the live-vs-sim cost gap on
marketable-limit crossing (sg1-btc S552 RCA: ~5 bps cross vs an unset=0
sim assumption), so it is tracked alongside the fee rather than fee alone.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class CostDriftStatus:
    """String constants — JSON-friendly, parallel to ``DriftStatus``."""

    OK = "OK"
    WARMUP = "WARMUP"
    LOG_ONLY = "LOG_ONLY"
    FIRED = "FIRED"


@dataclass
class CostDriftReport:
    status: str
    reason: str
    n_trades: int
    window_trades: int
    config_cost_frac: Optional[float]
    cost_drift_ratio: float
    realized_cost_frac_mean: Optional[float]
    cost_ratio: Optional[float]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "n_trades": self.n_trades,
            "window_trades": self.window_trades,
            "config_cost_frac": self.config_cost_frac,
            "cost_drift_ratio": self.cost_drift_ratio,
            "realized_cost_frac_mean": self.realized_cost_frac_mean,
            "cost_ratio": self.cost_ratio,
        }


class CostDriftTracker:
    """Per-strategy rolling realized-cost-vs-config tracker (§4.5 #6).

    Args:
        config_cost_frac: the one-way per-trade cost fraction the policy was
            trained / configured under — ``taker_fee + slippage_base_bps/1e4``.
            ``None`` or ``<= 0`` disables gating (status ``LOG_ONLY``): the
            realized cost is still recorded, but no ratio (hence no fire) can
            be computed without a denominator.
        cost_drift_ratio: realized/config ratio above which the trigger fires
            (``gates.retrain.cost_drift_ratio``, default **1.20**).
        window_trades: rolling window length in *filled trades*
            (``gates.retrain.cost_drift_window_trades``, default **100**).
            The trigger requires a full window before it can fire.
    """

    def __init__(
        self,
        config_cost_frac: Optional[float],
        *,
        cost_drift_ratio: float = 1.20,
        window_trades: int = 100,
    ):
        if window_trades < 1:
            raise ValueError("window_trades must be >= 1")
        if cost_drift_ratio <= 1.0:
            raise ValueError("cost_drift_ratio must be > 1.0")
        if config_cost_frac is not None and float(config_cost_frac) < 0.0:
            raise ValueError("config_cost_frac must be >= 0 (or None to disable)")

        # A non-positive configured cost has no usable denominator → treat as
        # "no baseline" (LOG_ONLY), the same permissive contract the action /
        # agreement trackers use for a missing baseline.
        self.config_cost_frac: Optional[float] = (
            float(config_cost_frac)
            if config_cost_frac is not None and float(config_cost_frac) > 0.0
            else None
        )
        self.cost_drift_ratio = float(cost_drift_ratio)
        self.window_trades = int(window_trades)
        self._realized_fracs: collections.deque[float] = collections.deque(
            maxlen=self.window_trades,
        )

        if self.config_cost_frac is None:
            logger.warning(
                "CostDriftTracker has no positive config_cost_frac — running "
                "in LOG_ONLY (realized cost recorded, no cost_ratio / FIRED)",
            )

    # --- public API -----------------------------------------------------

    def observe(
        self,
        *,
        fee: Optional[float],
        avg_fill_price: Optional[float],
        decision_price: Optional[float],
        filled_quantity: Optional[float],
    ) -> CostDriftReport:
        """Record one filled trade's realized one-way cost. Returns status.

        Invalid / missing inputs (Bybit demo can return ``None`` for fee /
        fill / qty via ccxt — see live_engine S527-cont) are *skipped*: the
        trade is not appended and the current window evaluation is returned
        unchanged. A skip never crashes the fill path and never fabricates a
        zero-cost sample that would dilute the rolling mean downward.
        """
        realized = self._realized_cost_frac(
            fee=fee,
            avg_fill_price=avg_fill_price,
            decision_price=decision_price,
            filled_quantity=filled_quantity,
        )
        if realized is not None:
            self._realized_fracs.append(realized)
        return self._evaluate()

    def snapshot(self) -> dict:
        """Current window stats without mutating state. For WandB logging."""
        return self._evaluate().to_dict()

    # --- internals ------------------------------------------------------

    @staticmethod
    def _realized_cost_frac(
        *,
        fee: Optional[float],
        avg_fill_price: Optional[float],
        decision_price: Optional[float],
        filled_quantity: Optional[float],
    ) -> Optional[float]:
        """One-way realized cost fraction for a single fill, or None to skip.

        ``fee_frac = |fee| / (|filled_quantity| * avg_fill_price)`` plus
        ``slippage_frac = |avg_fill_price - decision_price| / decision_price``.
        Slippage is dropped (0.0) — not the whole sample — when no usable
        decision price is available, so a missing reference degrades to a
        fee-only measurement rather than discarding the fill.
        """
        if fee is None or avg_fill_price is None or filled_quantity is None:
            return None
        try:
            fee_f = abs(float(fee))
            price_f = float(avg_fill_price)
            qty_f = abs(float(filled_quantity))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(fee_f) and math.isfinite(price_f) and math.isfinite(qty_f)):
            return None
        notional = qty_f * price_f
        if notional <= 0.0:
            return None
        fee_frac = fee_f / notional

        slippage_frac = 0.0
        if decision_price is not None:
            try:
                ref = float(decision_price)
            except (TypeError, ValueError):
                ref = 0.0
            if math.isfinite(ref) and ref > 0.0:
                slippage_frac = abs(price_f - ref) / ref

        return fee_frac + slippage_frac

    def _evaluate(self) -> CostDriftReport:
        n = len(self._realized_fracs)
        mean_realized = (sum(self._realized_fracs) / n) if n else None

        if self.config_cost_frac is None:
            return CostDriftReport(
                status=CostDriftStatus.LOG_ONLY,
                reason="no positive config_cost_frac — cannot compute cost_ratio",
                n_trades=n,
                window_trades=self.window_trades,
                config_cost_frac=None,
                cost_drift_ratio=self.cost_drift_ratio,
                realized_cost_frac_mean=mean_realized,
                cost_ratio=None,
            )

        cost_ratio = (
            mean_realized / self.config_cost_frac if mean_realized is not None else None
        )

        if n < self.window_trades:
            return CostDriftReport(
                status=CostDriftStatus.WARMUP,
                reason=(
                    f"warmup: {n}/{self.window_trades} trades before first check"
                ),
                n_trades=n,
                window_trades=self.window_trades,
                config_cost_frac=self.config_cost_frac,
                cost_drift_ratio=self.cost_drift_ratio,
                realized_cost_frac_mean=mean_realized,
                cost_ratio=cost_ratio,
            )

        if cost_ratio is not None and cost_ratio > self.cost_drift_ratio:
            return CostDriftReport(
                status=CostDriftStatus.FIRED,
                reason=(
                    f"FIRED: cost_ratio={cost_ratio:.3f} "
                    f"(> {self.cost_drift_ratio}) — realized one-way cost "
                    f"{mean_realized:.6f} vs config {self.config_cost_frac:.6f} "
                    f"over {n} trades; Protocol v2 §4.5 Stage 2.5-R retrain "
                    f"trigger #6 (cost_drift). Informational — not a halt."
                ),
                n_trades=n,
                window_trades=self.window_trades,
                config_cost_frac=self.config_cost_frac,
                cost_drift_ratio=self.cost_drift_ratio,
                realized_cost_frac_mean=mean_realized,
                cost_ratio=cost_ratio,
            )

        return CostDriftReport(
            status=CostDriftStatus.OK,
            reason=f"within threshold (cost_ratio={cost_ratio:.3f})",
            n_trades=n,
            window_trades=self.window_trades,
            config_cost_frac=self.config_cost_frac,
            cost_drift_ratio=self.cost_drift_ratio,
            realized_cost_frac_mean=mean_realized,
            cost_ratio=cost_ratio,
        )
