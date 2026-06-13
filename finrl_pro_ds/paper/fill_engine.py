"""Fill engines for the cross-asset TSMOM paper executor (rung 1 + rung 2).

Step 3 of the paper-executor build (S553-cont-47; spec
``.agent/artifacts/paper_executor_spec.md``). The executor shadows the FROZEN
linear core (``evaluate_linear_core``) into paper; a ``FillEngine`` turns a
rebalance's signed weight deltas into realized fills + costs.

  - **SimFillEngine (rung 1):** no broker. Fills at the per-bar reference price and
    charges the env's EXACT cost model — ``fee = Σ|order_notional|·taker_fee_pct`` and
    ``slippage = Σ|order_notional|·(base_bps + impact_bps·participation)·1e-4`` with
    **F1-correct** ``participation = |order_notional| / dollar_volume`` (ADR-6; the
    ``volume_ary`` carried by ``build_allocator_arrays`` is shares×price). This mirrors
    ``MultiAssetAllocatorEnv._calc_transaction_costs_fast`` line-for-line so the
    rung-1 paper-sim parity vs ``evaluate_linear_core`` is ≈0 by construction.
  - **IBFillEngine (rung 2):** real Interactive Brokers paper fills/commissions —
    operator-gated (ADR-5); not built in this step (the ABC marks the seam).

Slippage is modeled as a **cost debit**, NOT a worse fill price (the env's
convention): positions enter at the reference price and the (fee + slippage) is
subtracted from the book's margin. Rung-2 ``cost_drift_ratio`` then measures the
real fill-price deviation against this modeled cost.

Invariants: no raw ``print`` (logging only); SHORT-ACCT safe (works on signed
weight deltas; the book owns the no-``notional_debt`` accounting).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Match the env's active-trade and dollar-volume guards exactly (the parity
# tripwire is only sharp if these epsilons are identical).
_TRADE_EPS = 1e-8     # |delta_weight| below this is not a trade (env: abs_delta >= 1e-8)
_VOL_EPS = 1e-6       # dollar volume below this ⇒ assume MAX impact (env: bar_vols > 1e-6)


@dataclass(frozen=True)
class FillResult:
    """Outcome of filling one rebalance's orders.

    Attributes:
        fill_prices: (N,) price each asset filled at. Rung-1 = the reference price
            (slippage is a cost debit, not a price move — matches the env).
        fees: total taker fee charged (USD), ``Σ|order_notional|·taker_fee_pct``.
        slippage: total modeled slippage cost (USD), participation-based (F1).
        traded_notional: ``Σ|order_notional|`` (USD) — the one-way turnover notional.
    """

    fill_prices: np.ndarray
    fees: float
    slippage: float
    traded_notional: float

    @property
    def realized_cost(self) -> float:
        """Total cost debited from the book this rebalance (fee + slippage)."""
        return self.fees + self.slippage


class FillEngine(ABC):
    """Abstract fill engine: signed weight deltas → realized fills + costs."""

    @abstractmethod
    def fill(
        self,
        *,
        delta_weights: np.ndarray,
        ref_prices: np.ndarray,
        pv_before: float,
        dollar_volume: np.ndarray,
    ) -> FillResult:
        """Execute ``delta_weights`` (signed target-minus-held weights) at this bar.

        Args:
            delta_weights: (N,) signed weight change to execute (target − held).
            ref_prices: (N,) execution reference price per asset (the bar the fill
                books against — next close for parity, next open for ``fill_reference``).
            pv_before: portfolio value the weight notionals are measured against
                (the book's mark-to-market BEFORE this rebalance — the env's
                ``portfolio_value_before``).
            dollar_volume: (N,) per-asset dollar volume (shares×price) used as the
                slippage-participation denominator (F1).
        """
        raise NotImplementedError


class SimFillEngine(FillEngine):
    """Rung-1 simulated fills with the env's exact participation-based cost model.

    No broker, no partial fills, no rejects: every order fills in full at the
    reference price. The whole value of rung 1 is isolating forward-pipeline bugs
    from execution slippage (ADR-3), so the cost model must equal the env's.
    """

    def __init__(
        self,
        *,
        taker_fee_pct: float,
        slippage_base_bps: float,
        slippage_impact_bps: float,
    ) -> None:
        self.taker_fee_pct = float(taker_fee_pct)
        self.slippage_base_bps = float(slippage_base_bps)
        self.slippage_impact_bps = float(slippage_impact_bps)

    def fill(
        self,
        *,
        delta_weights: np.ndarray,
        ref_prices: np.ndarray,
        pv_before: float,
        dollar_volume: np.ndarray,
    ) -> FillResult:
        delta_weights = np.asarray(delta_weights, dtype=np.float64).ravel()
        ref_prices = np.asarray(ref_prices, dtype=np.float64).ravel()
        dollar_volume = np.asarray(dollar_volume, dtype=np.float64).ravel()
        abs_delta = np.abs(delta_weights)

        # Mirror MultiAssetAllocatorEnv._calc_transaction_costs_fast EXACTLY.
        active = abs_delta >= _TRADE_EPS
        if not active.any():
            return FillResult(fill_prices=ref_prices.copy(), fees=0.0,
                              slippage=0.0, traded_notional=0.0)

        notionals = abs_delta[active] * pv_before
        total_fees = float(np.sum(notionals) * self.taker_fee_pct)

        # F1: participation = order_notional / DOLLAR volume (dimensionless). Missing
        # / zero volume ⇒ assume MAX impact (ratio 1.0), the conservative env default.
        # Numerically identical to the env's np.where(vol>eps, n/vol, 1.0) but the
        # `where=` divide skips the div-by-zero entirely (no spurious RuntimeWarning).
        bar_vols = dollar_volume[active]
        volume_ratios = np.ones_like(notionals)
        np.divide(notionals, bar_vols, out=volume_ratios, where=bar_vols > _VOL_EPS)
        slippage_bps = self.slippage_base_bps + self.slippage_impact_bps * volume_ratios
        total_slippage = float(np.sum(notionals * slippage_bps * 1e-4))

        return FillResult(
            fill_prices=ref_prices.copy(),
            fees=total_fees,
            slippage=total_slippage,
            traded_notional=float(np.sum(notionals)),
        )
