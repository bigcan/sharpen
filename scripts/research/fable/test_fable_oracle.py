"""Hand-computed unit tests for the Fable clean-room oracle.

Run: python -m pytest scripts/research/fable/test_fable_oracle.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fable_oracle import (
    backtest_weights,
    block_bootstrap_sharpe_ci,
    equity_metrics,
    max_drawdown,
    metrics_from_returns,
    profit_factor,
    reprice_positions,
    sharpe_ann,
)


def _idx(n, freq="B", start="2020-01-01"):
    return pd.date_range(start, periods=n, freq=freq)


# ----------------------------------------------------------------- metrics
def test_profit_factor_hand():
    r = pd.Series([0.01, 0.01, -0.01])
    assert profit_factor(r) == pytest.approx(2.0)


def test_max_drawdown_hand():
    r = pd.Series([0.10, -0.20, 0.05])
    # equity: 1.10, 0.88, 0.924 ; peak 1.10 -> trough 0.88 = -20%
    assert max_drawdown(r) == pytest.approx(-0.20, abs=1e-12)


def test_sharpe_zero_mean():
    r = pd.Series([0.01, -0.01] * 100)
    assert abs(sharpe_ann(r)) < 1e-9


def test_metrics_n_and_tstat():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 504))
    m = metrics_from_returns(r)
    assert m["n"] == 504
    # t_stat must equal sharpe * sqrt(n / 252) by definition
    assert m["t_stat"] == pytest.approx(m["sharpe"] * np.sqrt(504 / 252), abs=0.01)


# ----------------------------------------------------- engine 1: weights
def test_single_asset_buy_and_hold_exact():
    px = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=_idx(3))
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=px.index)
    res = backtest_weights(px, w, cost_bps=0.0, exec_lag=1)
    # exec_lag=1: day1 return earned with weight decided day0 -> +10%, day2 -10%
    assert res.net.iloc[1] == pytest.approx(0.10)
    assert res.net.iloc[2] == pytest.approx(-0.10)
    assert res.net.iloc[0] == pytest.approx(0.0)


def test_cost_charged_on_turnover():
    px = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=_idx(4))
    w = pd.DataFrame({"A": [1.0, 1.0, -1.0, -1.0]}, index=px.index)
    res = backtest_weights(px, w, cost_bps=10.0, exec_lag=1)
    # held: [0, 1, 1, -1]; trades: |1| at d1... d0 held=0 (initial), d1 enter 1
    # cost d1 = 1 * 10bp = 0.001 ; d3 flip |dw|=2 -> 0.002
    assert res.cost.iloc[1] == pytest.approx(0.001)
    assert res.cost.iloc[3] == pytest.approx(0.002)
    assert res.cost.iloc[2] == pytest.approx(0.0)


def test_two_asset_mixed_book_hand_computed():
    px = pd.DataFrame({"A": [100, 102, 101], "B": [50, 49, 50]},
                      index=_idx(3), dtype=float)
    w = pd.DataFrame({"A": [0.5] * 3, "B": [-0.5] * 3}, index=px.index)
    res = backtest_weights(px, w, exec_lag=1)
    # day1: 0.5*(+2%) + (-0.5)*(-2%) = 1% + 1% = +2%
    assert res.net.iloc[1] == pytest.approx(0.5 * 0.02 + 0.5 * 0.02)


def test_borrow_charged_on_shorts_only():
    px = pd.DataFrame({"A": [100.0] * 253}, index=_idx(253))
    w_short = pd.DataFrame({"A": [-1.0] * 253}, index=px.index)
    w_long = pd.DataFrame({"A": [1.0] * 253}, index=px.index)
    rs = backtest_weights(px, w_short, borrow_bps_yr=100.0)
    rl = backtest_weights(px, w_long, borrow_bps_yr=100.0)
    # 100bp/yr borrow on flat prices -> ~ -1% over 252 trading days, longs free
    assert rs.net.sum() == pytest.approx(-0.01, rel=1e-6)
    assert rl.net.sum() == pytest.approx(0.0)


def test_turnover_annualization_monthly_flip():
    idx = pd.date_range("2020-01-01", periods=504, freq="B")
    px = pd.DataFrame({"A": np.full(len(idx), 100.0)}, index=idx)
    month_end = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    w = pd.DataFrame(index=idx, columns=["A"], dtype=float)
    flip = 1.0
    for d in month_end:
        w.loc[d, "A"] = flip
        flip = -flip
    res = backtest_weights(px, w, exec_lag=1)
    # full flip |dw|=2 each month -> ~24/yr one-way turnover (first entry =1)
    assert 20 < res.turnover_ann < 26


# -------------------------------------------------- look-ahead tripwire
def test_lookahead_tripwire():
    """THE core honesty test, both directions of the timing contract.

    (a) A signal using ONLY same-bar info, w[t] = sign(ret[t]):
        - exec_lag=0 grants illegitimate same-bar execution -> money printer.
        - exec_lag=1 (honest next-bar execution) -> edge vanishes on a random
          walk. If it does NOT vanish, the engine leaks.
    (b) A future-peeking signal w[t] = sign(ret[t+1]) WITH exec_lag=1 must
        print money: the engine applies w[t-1] to ret[t], proving the lag is
        exactly one bar (not accidentally zero or two)."""
    rng = np.random.default_rng(42)
    n = 4000
    r = rng.normal(0, 0.01, n)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + r)}, index=_idx(n))
    rets = px["A"].pct_change()

    same_bar = pd.DataFrame({"A": np.sign(rets).fillna(0.0)}, index=px.index)
    leak0 = backtest_weights(px, same_bar, exec_lag=0)
    honest = backtest_weights(px, same_bar, exec_lag=1)
    assert leak0.metrics["sharpe"] > 5.0, "same-bar signal + lag0 must print money"
    assert abs(honest.metrics["sharpe"]) < 1.0, (
        f"LEAK: same-bar signal with exec_lag=1 shows sharpe="
        f"{honest.metrics['sharpe']} - engine timing is broken")

    peek = pd.DataFrame({"A": np.sign(rets.shift(-1)).fillna(0.0)}, index=px.index)
    peeking = backtest_weights(px, peek, exec_lag=1)
    assert peeking.metrics["sharpe"] > 5.0, (
        "future-peek signal + lag1 must print money (lag must be exactly 1 bar)")


def test_shift_invariance_of_costs_and_pnl_alignment():
    """Decision at t, exec_lag=1: the FIRST day that can differ between two
    signals decided at t is t+1. Guards subtle off-by-one drift."""
    px = pd.DataFrame({"A": [100, 101, 102, 103, 104.0]}, index=_idx(5))
    w0 = pd.DataFrame({"A": [0, 0, 0, 0, 0.0]}, index=px.index)
    w1 = pd.DataFrame({"A": [0, 0, 1.0, 1.0, 1.0]}, index=px.index)  # decided d2
    r0 = backtest_weights(px, w0)
    r1 = backtest_weights(px, w1)
    assert (r1.net.iloc[:3] == r0.net.iloc[:3]).all()  # d0..d2 identical
    assert r1.net.iloc[3] != r0.net.iloc[3]            # first effect at d3


# ------------------------------------------- engine 2: position repricer
def test_reprice_positions_hand_computed():
    idx = _idx(4, freq="15min")
    price = pd.Series([100.0, 110.0, 105.0, 115.0], index=idx)
    pos = pd.Series([1.0, 1.0, 2.0, 2.0], index=idx)  # held during bar
    out = reprice_positions(price, pos, fee_rate=0.001, slip_rate=0.0,
                            initial_equity=1000.0, pos_timing="held_during_bar")
    # bar0: held 1 entered at price 100 (no prior px -> exec at p0), dprice 0
    #   fees = 1*100*0.001 = 0.1 ; equity = 999.9
    # bar1: held 1, dprice +10 -> +10 ; no trade ; equity 1009.9
    # bar2: held 2 (bought 1 @ exec px = prev close 110): fee 0.11
    #   pnl = 2 * (105-110) = -10 ; equity = 1009.9 - 10 - 0.11 = 999.79
    # bar3: held 2, dprice +10 -> +20 ; equity 1019.79
    assert out["equity"].iloc[0] == pytest.approx(999.9)
    assert out["equity"].iloc[1] == pytest.approx(1009.9)
    assert out["equity"].iloc[2] == pytest.approx(999.79)
    assert out["equity"].iloc[3] == pytest.approx(1019.79)


def test_equity_metrics_hand():
    eq = pd.Series([100.0, 110.0, 99.0, 108.9])
    m = equity_metrics(eq)
    # diffs: +10, -11, +9.9 -> pf = 19.9/11
    assert m["pf_bar"] == pytest.approx(19.9 / 11.0, abs=1e-4)
    assert m["trailing_max_dd_pct"] == pytest.approx(-10.0, abs=1e-6)
    assert m["total_return_pct"] == pytest.approx(8.9, abs=1e-6)


def test_bootstrap_ci_sane():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    ci = block_bootstrap_sharpe_ci(r, n_boot=500)
    assert ci["lo95"] < ci["hi95"]
    assert 0.0 < ci["p_sharpe_le_0"] < 1.0
