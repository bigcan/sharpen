"""Fable clean-room backtest oracle.

Written from scratch, on purpose sharing NO code with the project pipeline, so it
can serve as an independent referee for legacy claims (both "edges" and "kills").

Two engines:
  1. backtest_weights  — weight-based daily/periodic portfolio backtest
                         (next-bar execution, turnover costs, borrow drag).
  2. reprice_positions — unit-position equity reconstruction for re-pricing
                         recorded trajectories (e.g. RL eval logs) against raw
                         prices with an explicit fee+slippage model.

Metrics are computed in one place (`metrics_from_returns`) and unit-tested with
hand-computed cases in test_fable_oracle.py, including a look-ahead tripwire.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = 252


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def sharpe_ann(r: pd.Series, periods: int = ANN) -> float:
    r = r.dropna()
    s = float(r.std(ddof=1))
    return float(r.mean() / s * np.sqrt(periods)) if s > 0 else 0.0


def sortino_ann(r: pd.Series, periods: int = ANN) -> float:
    r = r.dropna()
    dn = r[r < 0]
    dd = float(np.sqrt((dn**2).sum() / max(len(r), 1)))
    return float(r.mean() / dd * np.sqrt(periods)) if dd > 0 else 0.0


def profit_factor(r: pd.Series) -> float:
    x = r.dropna().to_numpy()
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def cagr(r: pd.Series, periods: int = ANN) -> float:
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    total = float((1.0 + r).prod())
    yrs = len(r) / periods
    return float(total ** (1.0 / yrs) - 1.0) if yrs > 0 and total > 0 else -1.0


def metrics_from_returns(r: pd.Series, periods: int = ANN) -> dict:
    r = r.dropna()
    return {
        "n": int(len(r)),
        "sharpe": round(sharpe_ann(r, periods), 3),
        "sortino": round(sortino_ann(r, periods), 3),
        "pf": round(profit_factor(r), 4),
        "max_dd": round(max_drawdown(r), 4),
        "cagr": round(cagr(r, periods), 4),
        "ann_vol": round(float(r.std(ddof=1) * np.sqrt(periods)), 4),
        "t_stat": round(sharpe_ann(r, periods) * np.sqrt(len(r) / periods), 2),
        "worst_day": round(float(r.min()), 4) if len(r) else 0.0,
    }


def block_bootstrap_sharpe_ci(
    r: pd.Series, n_boot: int = 2000, block: int = 21,
    seed: int = 7, periods: int = ANN,
) -> dict:
    """Circular block bootstrap CI for the annualized Sharpe."""
    x = r.dropna().to_numpy()
    n = len(x)
    if n < block * 3:
        return {"lo95": float("nan"), "hi95": float("nan"), "p_sharpe_le_0": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        smp = x[idx[:n]]
        sd = smp.std(ddof=1)
        out[i] = smp.mean() / sd * np.sqrt(periods) if sd > 0 else 0.0
    return {
        "lo95": round(float(np.quantile(out, 0.025)), 3),
        "hi95": round(float(np.quantile(out, 0.975)), 3),
        "p_sharpe_le_0": round(float((out <= 0).mean()), 4),
    }


# --------------------------------------------------------------------------- #
# Engine 1: weight-based backtest
# --------------------------------------------------------------------------- #
@dataclass
class WeightBacktestResult:
    net: pd.Series
    gross: pd.Series
    cost: pd.Series
    held: pd.DataFrame
    turnover_ann: float
    avg_gross_lev: float
    metrics: dict = field(default_factory=dict)


def backtest_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 0.0,
    exec_lag: int = 1,
    borrow_bps_yr: float = 0.0,
    financing_bps_yr: float = 0.0,
) -> WeightBacktestResult:
    """Weight-based portfolio backtest with explicit timing semantics.

    prices  : wide close prices (DatetimeIndex x asset).
    weights : target weights as fraction of NAV, indexed by DECISION time
              (any subset of prices.index). Forward-filled between decisions.
    exec_lag: bars between decision and the first return earned. exec_lag=1
              means a weight decided at bar t earns return t->t+1 (i.e. you
              trade at bar t's close). exec_lag=0 is deliberately allowed so
              the look-ahead tripwire can PROVE the engine's timing is honest.
    cost_bps: one-way proportional cost on traded notional |dw| (per side).
    borrow_bps_yr: annual borrow fee charged on short gross exposure.
    financing_bps_yr: annual financing on gross leverage above 1.
    """
    prices = prices.sort_index()
    rets = prices.pct_change()
    w = weights.reindex(prices.index).ffill().fillna(0.0)
    w = w.reindex(columns=prices.columns).fillna(0.0)
    held = w.shift(exec_lag).fillna(0.0)

    # If an asset has no price (NaN return) its held weight earns nothing.
    gross = (held * rets).sum(axis=1, min_count=1).fillna(0.0)

    trade = held.diff().abs().sum(axis=1)
    if len(held):
        trade.iloc[0] = held.iloc[0].abs().sum()
    cost = trade * (cost_bps / 1e4)

    short_gross = held.clip(upper=0.0).abs().sum(axis=1)
    cost = cost + short_gross * (borrow_bps_yr / 1e4) / ANN
    gross_lev = held.abs().sum(axis=1)
    cost = cost + (gross_lev - 1.0).clip(lower=0.0) * (financing_bps_yr / 1e4) / ANN

    net = gross - cost
    yrs = max(len(prices) / ANN, 1e-9)
    res = WeightBacktestResult(
        net=net, gross=gross, cost=cost, held=held,
        turnover_ann=float(trade.sum() / yrs),
        avg_gross_lev=float(gross_lev.mean()),
    )
    res.metrics = metrics_from_returns(net)
    return res


# --------------------------------------------------------------------------- #
# Engine 2: unit-position equity reconstruction (for recorded trajectories)
# --------------------------------------------------------------------------- #
def reprice_positions(
    price: pd.Series,
    pos_units: pd.Series,
    fee_rate: float = 0.0,
    slip_rate: float = 0.0,
    initial_equity: float = 100_000.0,
    pos_timing: str = "held_during_bar",
) -> pd.DataFrame:
    """Reconstruct equity from a recorded unit-position series and raw prices.

    pos_timing:
      "held_during_bar": pos[t] is the position held during bar t's move
                         (earns price[t] - price[t-1]); trades execute at
                         price[t-1] (the bar's open ~ prior close).
      "set_at_bar_close": pos[t] decided at bar t's close, earns
                          price[t+1] - price[t]; trades execute at price[t].
    Fees+slippage are charged on traded notional |dpos| * exec_price.
    """
    idx = price.index.intersection(pos_units.index)
    p = price.loc[idx].astype(float)
    q = pos_units.loc[idx].astype(float)

    if pos_timing == "set_at_bar_close":
        held = q.shift(1).fillna(0.0)
        exec_px = p.shift(1)
    elif pos_timing == "held_during_bar":
        held = q
        exec_px = p.shift(1)
    else:
        raise ValueError(pos_timing)

    dpos = held.diff()
    dpos.iloc[0] = held.iloc[0]
    exec_px = exec_px.fillna(p)
    dprice = p.diff().fillna(0.0)

    pnl = held * dprice
    fees = dpos.abs() * exec_px * (fee_rate + slip_rate)
    equity = initial_equity + (pnl - fees).cumsum()
    return pd.DataFrame({
        "price": p, "held": held, "pnl": pnl, "fees": fees, "equity": equity,
    })


def equity_metrics(equity: pd.Series, bars_per_day: int = 96) -> dict:
    eq = equity.dropna()
    d = eq.diff().dropna()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    run_max = eq.cummax()
    return {
        "n_bars": int(len(eq)),
        "pf_bar": round(float(pos / neg), 4) if neg > 0 else float("inf"),
        "total_return_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100, 2),
        "trailing_max_dd_pct": round(float((eq / run_max - 1.0).min()) * 100, 2),
    }
