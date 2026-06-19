"""Paper-executor book + forward accounting (the sim-rung state).

Step 3 of the paper-executor build (S553-cont-47; spec
``.agent/artifacts/paper_executor_spec.md``). ``PaperState`` is the live book the
executor advances one daily bar at a time. Its per-bar accounting **mirrors
``MultiAssetAllocatorEnv.step`` line-for-line** (same fixed-entry-notional PnL,
SHORT-ACCT buyback, carry, cost debit, liquidation guard, and — critically — the
same ordering: ``pv_before`` is marked BEFORE carry, costs are charged on
``pv_before``, entries update AFTER realize). That fidelity is what makes the
rung-1 paper-sim parity vs ``evaluate_linear_core`` ≈0 by construction; the
``tests/paper`` keystone + ``test_paper_state`` pin tests fail loudly if this book
ever drifts from the env.

Why a separate forward book (not just driving the env): the env is a backtest
construct that consumes the whole price array up front. The live executor sees one
new daily bar at a time and must persist/resume its state across scheduled runs —
so it needs an independent forward accounting. Keeping it byte-faithful to the env
(and proving it with the pin test) preserves "any non-zero parity drift is a real
forward-path bug" (ADR-7), without touching the validated env (additive only).

Invariants: SHORT-ACCT (shorts carry no ``notional_debt``; buyback via signed
entry-notional PnL); LEAK-2 (the executor only ever books data ``<= as_of`` — the
book itself is causal); no raw ``print`` (logging only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.paper.fill_engine import FillResult

logger = logging.getLogger(__name__)

# Epsilons transcribed verbatim from MultiAssetAllocatorEnv — the parity tripwire
# is only sharp if every threshold matches the env exactly.
_POS_EPS = 1e-8
_NOTIONAL_EPS = 1e-8
_PRICE_EPS = 1e-10
_PV_FLOOR_FRAC = 0.001    # env: portfolio_value_before floored at initial_capital*0.001

_STATE_PARQUET = "paper_state.parquet"


def generate_orders(
    target_w: np.ndarray, held_w: np.ndarray, *, min_trade_pct: float = 0.0,
) -> np.ndarray:
    """Order weight-deltas to move ``held_w`` → ``target_w`` with a min-trade deadband.

    ``|Δw| < min_trade_pct`` is zeroed — identical to the env's dust filter (a WEIGHT
    threshold; ``order_notional = Δw·equity`` so ``|order|<min_trade_pct·equity`` ⇔
    ``|Δw|<min_trade_pct``). This is the executor's order-gen primitive (step 4); the
    rung-1 replay feeds the env-processed weight trajectory directly and does NOT
    re-deadband (those weights are already the env's post-dust realized positions).
    """
    delta = np.asarray(target_w, dtype=np.float64).ravel() - np.asarray(held_w, dtype=np.float64).ravel()
    if min_trade_pct > 0.0:
        delta = np.where(np.abs(delta) < float(min_trade_pct), 0.0, delta)
    return delta


@dataclass
class PaperState:
    """The live paper book: signed weights + fixed-notional entry accounting + margin.

    All accounting methods are verbatim transcriptions of the matching
    ``MultiAssetAllocatorEnv`` methods (referenced by name in each docstring).
    """

    n_assets: int
    initial_capital: float
    assets: list[str] = field(default_factory=list)
    # NaN sentinel (not 0.0) distinguishes "unset → default to initial_capital" from a
    # genuinely-persisted 0.0 — a liquidated book floors margin to EXACTLY 0.0, and the old
    # `== 0.0` heuristic resurrected it to full capital on reload (P10-02).
    margin_balance: float = float("nan")
    # Default to empty arrays (sized to n_assets in __post_init__); keeps the dataclass
    # field types clean (ndarray, not Optional) while still allowing a bare
    # PaperState(n_assets=N, initial_capital=C) construction.
    positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))       # (N,) signed weights
    entry_prices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))    # (N,)
    entry_notionals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # (N,) fixed notional
    realized_pnl: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_carry: float = 0.0
    peak_equity: float = float("nan")    # NaN sentinel (see margin_balance) — preserve a persisted peak
    as_of_ts: int = 0

    def __post_init__(self) -> None:
        if self.positions.size == 0:
            self.positions = np.zeros(self.n_assets, dtype=np.float64)
        if self.entry_prices.size == 0:
            self.entry_prices = np.zeros(self.n_assets, dtype=np.float64)
        if self.entry_notionals.size == 0:
            self.entry_notionals = np.zeros(self.n_assets, dtype=np.float64)
        if not self.assets:
            self.assets = [f"asset_{i}" for i in range(self.n_assets)]
        # NaN sentinel ⇒ unset ⇒ default to initial_capital; a persisted 0.0 (liquidated
        # book) is preserved (P10-02). load() passes the real persisted values.
        if np.isnan(self.margin_balance):
            self.margin_balance = float(self.initial_capital)
        if np.isnan(self.peak_equity):
            self.peak_equity = float(self.initial_capital)

    # ------------------------------------------------------------------ #
    # Mark-to-market (env: _calc_unrealized_pnl / portfolio_value_before)
    # ------------------------------------------------------------------ #
    def unrealized_pnl(self, price: np.ndarray) -> np.ndarray:
        """Per-asset unrealized PnL on fixed entry notionals (env._calc_unrealized_pnl):
        ``sign(pos)·entry_notional·(price/entry_price − 1)``."""
        pnl = np.zeros(self.n_assets, dtype=np.float64)
        active = (np.abs(self.positions) > _POS_EPS) & (self.entry_notionals > _NOTIONAL_EPS)
        if active.any():
            ratio = price[active] / (self.entry_prices[active] + _PRICE_EPS) - 1.0
            pnl[active] = np.sign(self.positions[active]) * self.entry_notionals[active] * ratio
        return pnl

    def portfolio_value(self, price: np.ndarray) -> float:
        """Equity = margin + Σ unrealized at ``price``."""
        return self.margin_balance + float(self.unrealized_pnl(price).sum())

    def pv_before(self, prev_price: np.ndarray) -> float:
        """``portfolio_value_before`` (env.step): equity at the PRIOR bar's price,
        floored at ``initial_capital·0.001`` — the notional base for this bar's costs."""
        unreal = self.unrealized_pnl(prev_price)
        return max(self.margin_balance + float(unreal.sum()), self.initial_capital * _PV_FLOOR_FRAC)

    # ------------------------------------------------------------------ #
    # Accounting primitives (verbatim from MultiAssetAllocatorEnv)
    # ------------------------------------------------------------------ #
    def _realize_pnl(self, old_positions: np.ndarray, delta_weights: np.ndarray,
                     price: np.ndarray) -> float:
        """env._realize_pnl: realize PnL on the reduced/closed/flipped portion."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)
        active = (abs_old >= _POS_EPS) & (np.abs(self.entry_prices) >= _PRICE_EPS) \
            & (self.entry_notionals >= _NOTIONAL_EPS)
        if not active.any():
            return 0.0
        closed_fraction = np.zeros(self.n_assets, dtype=np.float64)
        flipped = active & (np.sign(old_positions) != np.sign(new_positions)) & (abs_new > _POS_EPS)
        closed_fraction[flipped] = 1.0
        reduced = active & ~flipped & (abs_new < abs_old)
        if reduced.any():
            closed_fraction[reduced] = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]
        price_change = price / (self.entry_prices + _PRICE_EPS) - 1.0
        pnl = np.sign(old_positions) * closed_fraction * self.entry_notionals * price_change
        return float(pnl.sum())

    def _update_entry_prices(self, old_positions: np.ndarray, delta_weights: np.ndarray,
                             price: np.ndarray, pv_before: float) -> None:
        """env._update_entry_prices: fixed-notional bookkeeping on position changes."""
        new_positions = old_positions + delta_weights
        abs_old = np.abs(old_positions)
        abs_new = np.abs(new_positions)

        closed = abs_new < _POS_EPS
        self.entry_prices[closed] = 0.0
        self.entry_notionals[closed] = 0.0

        from_flat = ~closed & (abs_old < _POS_EPS)
        self.entry_prices[from_flat] = price[from_flat]
        self.entry_notionals[from_flat] = abs_new[from_flat] * pv_before

        flipped = ~closed & ~from_flat & (np.sign(old_positions) != np.sign(new_positions))
        self.entry_prices[flipped] = price[flipped]
        self.entry_notionals[flipped] = abs_new[flipped] * pv_before

        increased = ~closed & ~from_flat & ~flipped & (abs_new > abs_old)
        if increased.any():
            added_notional = np.abs(delta_weights[increased]) * pv_before
            old_notional = self.entry_notionals[increased]
            new_notional = old_notional + added_notional
            self.entry_prices[increased] = (
                self.entry_prices[increased] * old_notional
                + price[increased] * added_notional
            ) / (new_notional + _PRICE_EPS)
            self.entry_notionals[increased] = new_notional

        reduced = ~closed & ~from_flat & ~flipped & ~increased & (abs_new < abs_old)
        if reduced.any():
            closed_frac = (abs_old[reduced] - abs_new[reduced]) / abs_old[reduced]
            self.entry_notionals[reduced] *= (1.0 - closed_frac)

    def _apply_carry(self, price: np.ndarray, carry_rates: np.ndarray) -> float:
        """env._apply_carry: per-bar carry on OPEN positions (v1 carry == 0)."""
        active = (np.abs(self.positions) >= _POS_EPS) & (np.abs(self.entry_prices) >= _PRICE_EPS)
        if not active.any():
            return 0.0
        current_notional = self.entry_notionals[active] * (price[active] / self.entry_prices[active])
        carry_pnl = np.sign(self.positions[active]) * current_notional * carry_rates[active]
        return float(carry_pnl.sum())

    # ------------------------------------------------------------------ #
    # The per-bar forward step (env.step accounting, ordering preserved)
    # ------------------------------------------------------------------ #
    def step_bar(
        self,
        *,
        delta_weights: np.ndarray,
        fill: FillResult,
        prev_price: np.ndarray,
        price_now: np.ndarray,
        carry_rates: np.ndarray,
        pv_before: float,
        as_of_ts: int | None = None,
    ) -> dict:
        """Advance the book one bar, mirroring ``MultiAssetAllocatorEnv.step`` exactly.

        Caller supplies ``pv_before`` (== ``self.pv_before(prev_price)``, marked BEFORE
        carry) and the ``fill`` whose cost was computed on that same ``pv_before`` — so
        the cost notional base matches the env. Returns the per-bar info the trajectory
        accumulates. ``prev_price`` is accepted for signature symmetry / future
        validation; ``pv_before`` already encodes it.
        """
        del prev_price  # encoded in pv_before (kept in the signature for call-site clarity)
        delta_weights = np.asarray(delta_weights, dtype=np.float64).ravel()
        old_positions = self.positions.copy()

        # 1. Carry on OLD positions at the new price (env applies before rebalance).
        carry_pnl = self._apply_carry(price_now, np.asarray(carry_rates, dtype=np.float64).ravel())
        self.cumulative_carry += carry_pnl
        self.margin_balance += carry_pnl

        # 2. Realize on the closed/reduced portion, then move positions + entries.
        realized = self._realize_pnl(old_positions, delta_weights, price_now)
        self.positions = old_positions + delta_weights
        self._update_entry_prices(old_positions, delta_weights, price_now, pv_before)

        # 3. Cost debit (fee + slippage; slippage is a cost, not a price move).
        cost = fill.realized_cost
        self.cumulative_fees += cost
        self.realized_pnl += realized
        self.margin_balance -= cost
        self.margin_balance += realized

        # 4. Liquidation guard (env): margin < 0 ⇒ force-close + realize remainder.
        if self.margin_balance < 0:
            forced = float(self.unrealized_pnl(price_now).sum())
            self.realized_pnl += forced
            self.margin_balance += forced
            self.margin_balance = max(self.margin_balance, 0.0)
            self.positions[:] = 0.0
            self.entry_prices[:] = 0.0
            self.entry_notionals[:] = 0.0

        # 5. New equity + step return (env: relative to pv_before).
        pv = self.margin_balance + float(self.unrealized_pnl(price_now).sum())
        step_return = (pv - pv_before) / pv_before if pv_before > 1e-6 else 0.0
        self.peak_equity = max(self.peak_equity, pv)
        if as_of_ts is not None:
            self.as_of_ts = int(as_of_ts)

        return {
            "portfolio_value": pv,
            "step_return": step_return,
            "carry": carry_pnl,
            "realized": realized,
            "cost": cost,
            "turnover": float(np.abs(delta_weights).sum()),
            "gross_exposure": float(np.abs(self.positions).sum()),
            "net_exposure": float(self.positions.sum()),
        }

    # ------------------------------------------------------------------ #
    # Persistence (paper_state.parquet — resume across scheduled runs)
    # ------------------------------------------------------------------ #
    def to_frame(self) -> pd.DataFrame:
        """Per-asset book as a DataFrame; scalars broadcast as constant columns so the
        whole book round-trips through a single ``paper_state.parquet`` (ADR: persist
        positions, entry_notionals, equity-state)."""
        return pd.DataFrame({
            "asset": list(self.assets),
            "position": self.positions,
            "entry_price": self.entry_prices,
            "entry_notional": self.entry_notionals,
            "margin_balance": self.margin_balance,
            "realized_pnl": self.realized_pnl,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_carry": self.cumulative_carry,
            "peak_equity": self.peak_equity,
            "initial_capital": self.initial_capital,
            "as_of_ts": self.as_of_ts,
        })

    def save(self, state_dir: str | Path) -> Path:
        """Persist the book to ``<state_dir>/paper_state.parquet`` (atomic via tmp)."""
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / _STATE_PARQUET
        tmp = path.with_suffix(".parquet.tmp")
        self.to_frame().to_parquet(tmp, index=False)
        tmp.replace(path)
        logger.info("paper_state: saved book (%d assets, equity≈%.2f) → %s",
                    self.n_assets, self.margin_balance, path)
        return path

    @classmethod
    def load(cls, state_dir: str | Path) -> "PaperState":
        """Reconstruct a book previously written by :meth:`save`."""
        path = Path(state_dir) / _STATE_PARQUET
        df = pd.read_parquet(path)
        s0 = df.iloc[0]
        st = cls(
            n_assets=len(df),
            initial_capital=float(s0["initial_capital"]),
            assets=df["asset"].astype(str).tolist(),
            margin_balance=float(s0["margin_balance"]),
            positions=df["position"].to_numpy(np.float64),
            entry_prices=df["entry_price"].to_numpy(np.float64),
            entry_notionals=df["entry_notional"].to_numpy(np.float64),
            realized_pnl=float(s0["realized_pnl"]),
            cumulative_fees=float(s0["cumulative_fees"]),
            cumulative_carry=float(s0["cumulative_carry"]),
            peak_equity=float(s0["peak_equity"]),
            as_of_ts=int(s0["as_of_ts"]),
        )
        return st


@dataclass
class LiveTrajectory:
    """Accumulated forward path of a replay/soak — the executor's live side that the
    :class:`~finrl_pro_ds.paper.parity_harness.ParityHarness` compares to the sim oracle
    and the soak gates score.

    ``equity_curve`` has ``n_steps + 1`` points (``[0] == initial_capital``); every
    other series has ``n_steps`` points. ``timestamps[k]`` is the epoch-second stamp of
    the bar at which weights ``weights[k]`` are held/valued.
    """

    weights: np.ndarray                 # (n_steps, N)
    equity_curve: np.ndarray            # (n_steps + 1,)
    step_returns: np.ndarray            # (n_steps,)
    turnovers: np.ndarray               # (n_steps,)
    cumulative_fees: np.ndarray         # (n_steps,)
    gross_exposure: np.ndarray          # (n_steps,)
    net_exposure: np.ndarray            # (n_steps,)
    timestamps: np.ndarray              # (n_steps,)
    class_pnl: dict[str, float]         # cumulative gross-return attribution per class
    assets: list[str]
    asset_class: dict[str, str]
    initial_capital: float
    spy_returns: np.ndarray | None = None   # (n_steps,) SPY daily returns (drift gate), or None
    # Per-SLEEVE cumulative gross-return attribution (two-sleeve book only; None for the
    # single-sleeve path). Distinct from class_pnl (per asset class): a sleeve may span
    # several classes and two sleeves may overlap on a bond ETF, so this is attributed by
    # sleeve via the risk-parity-weighted sleeve weights (TwoSleeveExecutor).
    sleeve_pnl: dict[str, float] | None = None

    @property
    def n_steps(self) -> int:
        return int(len(self.step_returns))

    def max_drawdown_pct(self) -> float:
        """Worst peak-to-trough drawdown of the equity curve, in percent (>= 0)."""
        eq = self.equity_curve
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd = 1.0 - eq / np.where(peak <= 0, 1.0, peak)
        return float(dd.max() * 100.0)

    def min_daily_return_pct(self) -> float:
        """Worst single-day return, in percent (<= 0 for a real loss)."""
        return float(self.step_returns.min() * 100.0) if self.n_steps else 0.0

    def max_gross_exposure(self) -> float:
        return float(self.gross_exposure.max()) if self.n_steps else 0.0

    def corr_to_spy(self, window: int | None = None) -> float | None:
        """Correlation of portfolio daily returns to SPY daily returns over the trailing
        ``window`` days (full sample if ``window`` is None). None if SPY is absent or the
        sample is too short / degenerate."""
        if self.spy_returns is None or self.n_steps < 3:
            return None
        a = self.step_returns
        b = self.spy_returns
        if window is not None and window < len(a):
            a, b = a[-window:], b[-window:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def max_single_class_pnl_share(self) -> float:
        """Largest single asset-class share of the cumulative gross-return attribution
        MAGNITUDE: ``max_c |pnl_c| / Σ_c |pnl_c|`` (bounded [1/n_classes, 1]). The drift
        gate trips if one class dominates the book's P&L."""
        mags = {c: abs(v) for c, v in self.class_pnl.items()}
        total = sum(mags.values())
        if total <= 1e-12:
            return 0.0
        return float(max(mags.values()) / total)
