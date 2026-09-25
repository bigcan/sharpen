"""Daily NAV accounting for a fully funded long/short ETF book in a cash-funded Reg-T account.

What reaches the P&L (every risk and cost layer, by construction):
  * Execution one bar after the decision: a target decided at the close of day d trades at the close of d+1,
    so the return from d to d+1 is earned by the OLD holdings.
  * Holdings drift with total returns between trades (no free daily rebalancing).
  * Trading cost: |traded notional| x per-asset one-way cost, paid from own cash at each trade.
  * Own cash (NAV minus longs) earns the T-bill rate minus a spread. It may not go negative: no margin loan.
  * Short sale proceeds are held as collateral and earn ``short_rebate`` x T-bill (default 0: a retail broker
    pays nothing on them). Shorts pay a per-asset annual borrow fee, daily.
  * Proxy assets (pre-ETF history) can be charged the ETF's expense ratio via ``expense_bps_yr``; real ETF
    prices are already net of fees, so pass zeros for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 252


class FundingViolation(RuntimeError):
    """A target needs a margin loan (longs above NAV) — the book must never ask for one."""


@dataclass(frozen=True)
class CostModel:
    cost_bps: dict[str, float] = field(default_factory=dict)       # one-way, per asset; default below
    default_cost_bps: float = 2.0
    borrow_bps_yr: dict[str, float] = field(default_factory=dict)  # annual fee on short notional
    default_borrow_bps_yr: float = 50.0
    expense_bps_yr: dict[str, float] = field(default_factory=dict)  # proxies only
    cash_spread_bps_yr: float = 15.0                               # own cash earns rf minus this
    short_rebate: float = 0.0                                      # fraction of rf paid on short proceeds

    def vec(self, tickers: list[str], which: str) -> np.ndarray:
        table, default = {
            "cost": (self.cost_bps, self.default_cost_bps),
            "borrow": (self.borrow_bps_yr, self.default_borrow_bps_yr),
            "expense": (self.expense_bps_yr, 0.0),
        }[which]
        return np.array([float(table.get(t, default)) for t in tickers]) / 1e4

    def scaled(self, mult: float) -> CostModel:
        """Same model with every trading cost multiplied by ``mult`` (break-even search)."""
        return CostModel(cost_bps={k: v * mult for k, v in self.cost_bps.items()},
                         default_cost_bps=self.default_cost_bps * mult,
                         borrow_bps_yr=self.borrow_bps_yr, default_borrow_bps_yr=self.default_borrow_bps_yr,
                         expense_bps_yr=self.expense_bps_yr, cash_spread_bps_yr=self.cash_spread_bps_yr,
                         short_rebate=self.short_rebate)


def run(targets: pd.DataFrame, asset_rets: pd.DataFrame, rf: pd.Series, costs: CostModel, *,
        lag: int = 1, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
        tol: float = 1e-9) -> pd.DataFrame:
    """Simulate the book. ``targets``: decision-date rows of weights (fractions of NAV).

    ``asset_rets``: daily total returns (T, N) on the trading calendar; NaN = not tradeable that day (treated
    as 0 return; a target on a NaN-return asset on its execution day is refused).
    ``rf``: daily risk-free return aligned to the calendar. ``lag``: bars between decision and execution (>=1).
    Returns a daily frame: ret, rf, excess, nav, longs, shorts, gross, net, traded, cost, borrow, own_cash.
    """
    if lag < 1:
        raise ValueError("execution must be at least one bar after the decision (lag >= 1)")
    tickers = list(asset_rets.columns)
    targets = targets.reindex(columns=tickers).fillna(0.0)
    cal = asset_rets.index
    if start is not None:
        cal = cal[cal >= start]
    if end is not None:
        cal = cal[cal <= end]
    R = asset_rets.reindex(cal)
    avail = R.notna().to_numpy()
    R = R.fillna(0.0).to_numpy()
    RF = rf.reindex(cal).fillna(0.0).to_numpy()
    c_trade, c_borrow, c_exp = (costs.vec(tickers, w) for w in ("cost", "borrow", "expense"))
    spread = costs.cash_spread_bps_yr / 1e4 / ANN

    # schedule: execution row -> target vector
    pos_of = {d: i for i, d in enumerate(cal)}
    full_pos = {d: i for i, d in enumerate(asset_rets.index)}
    sched: dict[int, np.ndarray] = {}
    for d, row in targets.iterrows():
        if d not in full_pos:
            raise KeyError(f"decision date {d} not on the calendar")
        ex_full = full_pos[d] + lag
        if ex_full >= len(asset_rets.index):
            continue
        ex_date = asset_rets.index[ex_full]
        if ex_date in pos_of:
            sched[pos_of[ex_date]] = row.to_numpy(dtype=float)

    n = len(cal)
    p = np.zeros(len(tickers))          # position values (signed dollars)
    own, proceeds = 1.0, 0.0
    nav_prev = 1.0
    out = np.zeros((n, 11))
    for i in range(n):
        borrow = 0.0
        if i > 0:
            p = p * (1.0 + R[i] - c_exp / ANN)
            own = own * (1.0 + RF[i] - spread) if own > 0 else own
            proceeds = proceeds * (1.0 + costs.short_rebate * RF[i])
            borrow = float(np.sum(-np.minimum(p, 0.0) * c_borrow) / ANN)
            own -= borrow
        nav = own + proceeds + p.sum()
        traded = cost = 0.0
        if i in sched:
            w = sched[i]
            bad = (w != 0) & ~avail[i]
            if bad.any():
                raise ValueError(f"{cal[i].date()}: target on untradeable {np.array(tickers)[bad].tolist()}")
            if w.clip(min=0).sum() > 1.0 + tol:
                raise FundingViolation(f"{cal[i].date()}: longs {w.clip(min=0).sum():.4f} > NAV")
            new_p = w * nav
            traded = float(np.abs(new_p - p).sum())
            cost = float(np.sum(np.abs(new_p - p) * c_trade))
            nav_after = nav - cost
            p = w * nav_after
            proceeds = float(-np.minimum(p, 0.0).sum())
            own = nav_after - float(p.sum()) - proceeds
            if own < -tol * max(1.0, nav_after):
                raise FundingViolation(f"{cal[i].date()}: own cash {own:.6f} < 0")
            nav = nav_after
        ret = nav / nav_prev - 1.0 if i > 0 else 0.0
        longs = float(np.maximum(p, 0).sum() / nav) if nav > 0 else np.nan
        shorts = float(-np.minimum(p, 0).sum() / nav) if nav > 0 else np.nan
        out[i] = [ret, RF[i], ret - RF[i], nav, longs, shorts, longs + shorts, longs - shorts,
                  traded / nav_prev, cost / nav_prev, borrow / nav_prev]
        nav_prev = nav
    df = pd.DataFrame(out, index=cal, columns=["ret", "rf", "excess", "nav", "longs", "shorts", "gross", "net",
                                               "traded", "cost", "borrow"])
    df.iloc[0, df.columns.get_loc("rf")] = 0.0
    df.iloc[0, df.columns.get_loc("excess")] = 0.0
    return df


def buy_and_hold(asset_rets: pd.DataFrame, rf: pd.Series, ticker: str, costs: CostModel, *,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """The benchmark: 100% ``ticker`` bought at the first close and held (dividends reinvested via total return)."""
    cal = asset_rets.index
    if start is not None:
        cal = cal[cal >= start]
    first_decision = asset_rets.index[max(asset_rets.index.searchsorted(cal[0]) - 1, 0)]
    tgt = pd.DataFrame({ticker: [1.0]}, index=[first_decision]).reindex(columns=asset_rets.columns).fillna(0.0)
    start_at = first_decision if start is not None else None
    return run(tgt, asset_rets, rf, costs, start=start_at, end=end)
