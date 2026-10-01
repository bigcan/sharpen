"""Tripwires for scripts/research/finrl_x_rotation_repro.py (FinRL-X reproduction re-derivation).

The cost re-pricing carries the reproduction's headline claim (10 bps/side turns 4.91x into 3.02x, below QQQ), so each
property is pinned to a hand-computed value that a plausible wrong implementation misses:
  - turnover is charged against DRIFTED weights, not target-to-target;
  - drift is normalised by NAV including cash;
  - daily marks see an intra-week drawdown that weekly marks hide;
  - the parity check passes on an exact FinRL-X table and fails when one printed value moves by one unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrl_x_rotation_repro import (  # noqa: E402
    _equity,
    _periods,
    backtest,
    benchmark,
    breakeven_bps,
    check_parity,
    metrics,
    parse_their_table,
)

DATES = ["2024-01-05", "2024-01-12", "2024-01-19"]


def _closes(prices: dict[str, list[float]], dates: list[str]) -> dict[str, pd.Series]:
    idx = pd.DatetimeIndex(dates)
    return {s: pd.Series(v, index=idx, dtype=float) for s, v in prices.items()}


def _weights(rows: list[dict], dates: list[str]) -> pd.DataFrame:
    """FinRL-X's export layout: date, <symbols>..., cash, regime."""
    df = pd.DataFrame(rows).fillna(0.0)
    df.insert(0, "date", pd.to_datetime(dates))
    df["cash"] = 1 - df.drop(columns="date").sum(axis=1)
    df["regime"] = "risk_on"
    return df


def test_zero_cost_compounds_target_weights_and_skips_the_last_row():
    closes = _closes({"A": [100, 110, 121], "B": [100, 90, 99]}, DATES)
    w = _weights([{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, {"A": 1.0}], DATES)
    m = backtest(w, closes, cost_bps=0.0)
    # period 1: 0.5*(+10%) + 0.5*(-10%) = 0; period 2: +10%; the final row is an endpoint, never applied
    assert m["cumulative_x"] == pytest.approx(1.10, abs=1e-12)


def test_cost_is_charged_on_drifted_not_target_turnover():
    closes = _closes({"A": [100, 110, 121], "B": [100, 90, 99]}, DATES)
    w = _weights([{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, {"A": 1.0}], DATES)
    m = backtest(w, closes, cost_bps=10.0)
    # rebalance 1 buys from cash (turnover 1.0 -> NAV 0.999); the book then drifts to A 0.55 / B 0.45,
    # so rebalance 2 trades 0.10 back to 0.5 / 0.5. Target-to-target would see zero and end at 0.999 * 1.10.
    assert m["cumulative_x"] == pytest.approx(0.999 * (1 - 0.10 * 1e-3) * 1.10, abs=1e-12)
    assert m["avg_turnover_per_rebalance"] == pytest.approx((1.0 + 0.10) / 2, abs=1e-12)


def test_drift_is_normalised_by_nav_including_cash():
    closes = _closes({"A": [100, 110, 110]}, DATES)
    w = _weights([{"A": 0.5}, {"A": 0.5}, {"A": 0.5}], DATES)
    m = backtest(w, closes, cost_bps=0.0)
    # A +10% with half the book in cash: A drifts to 0.55 / 1.05, so the trim back to 0.5 is 0.025 / 1.05.
    # Normalising by the invested sleeve alone would call A the whole book and charge 0.5.
    assert m["avg_turnover_per_rebalance"] == pytest.approx((0.5 + 0.025 / 1.05) / 2, abs=1e-12)


def test_an_all_cash_period_is_flat():
    closes = _closes({"A": [100, 50, 60]}, DATES)
    w = _weights([{"A": 0.0}, {"A": 1.0}, {"A": 1.0}], DATES)
    m = backtest(w, closes, cost_bps=10.0)
    # week 1 in cash, so A's -50% is not held; week 2 buys A (turnover 1.0) and makes +20%.
    # Marks are pre-trade closes, so the buy-in cost lands in the next mark rather than as its own dip.
    assert m["cumulative_x"] == pytest.approx((1 - 1e-3) * 1.20, abs=1e-12)
    assert m["max_dd_daily"] == pytest.approx(0.0, abs=1e-12)


def test_daily_marks_see_the_intra_week_drawdown_weekly_marks_hide():
    days = ["2024-01-05", "2024-01-08", "2024-01-10", "2024-01-12", "2024-01-19"]
    a = pd.Series([100, 80, 95, 100, 105], index=pd.DatetimeIndex(days), dtype=float)
    w = _weights([{"A": 1.0}, {"A": 1.0}, {"A": 1.0}], DATES)
    m = backtest(w, {"A": a}, cost_bps=0.0)
    assert m["max_dd_weekly"] == pytest.approx(0.0, abs=1e-12)
    assert m["max_dd_daily"] == pytest.approx(-0.20, abs=1e-12)


def _finrlx_table(strategy: dict, spy: dict, qqq: dict) -> str:
    """Render a table the way FinRL-X's _generate_performance_report prints it."""

    def row(label: str, fmt, key: str, with_bench: bool = True) -> str:
        line = f"{label:<25s} {fmt(strategy[key]):>12s}"
        if with_bench:
            line += f" {fmt(spy[key]):>10s} {fmt(qqq[key]):>10s}"
        return line

    def mult(v: float) -> str:
        return f"{v:.2f}x"

    def pct(v: float) -> str:
        return f"{v:.2%}"

    def num(v: float) -> str:
        return f"{v:.2f}"

    return "\n".join([
        "Performance Analysis", "=" * 60, "",
        f"{'Metric':<25s} {'Strategy':>12s} {'SPY':>10s} {'QQQ':>10s}", "-" * 62,
        row("Cumulative Return", mult, "cumulative_x"),
        row("Annualized Return", pct, "ann_return"),
        row("Annualized Volatility", pct, "ann_vol"),
        row("Sharpe Ratio", num, "sharpe_finrlx"),
        row("Max Drawdown", pct, "max_dd_weekly"),
        row("Calmar Ratio", num, "calmar", with_bench=False),
        row("Win Rate", pct, "win_rate_weekly", with_bench=False),
    ])


def _example() -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2023-01-06", periods=260)
    closes = {
        s: pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, len(days)))), index=days)
        for s in ("A", "B", "SPY", "QQQ")
    }
    reb = days[::5]
    rows = [{"A": float(rng.uniform(0.2, 0.6)), "B": float(rng.uniform(0.1, 0.4))} for _ in reb]
    w = _weights(rows, [str(d.date()) for d in reb])
    ours = backtest(w, closes, cost_bps=0.0)
    bench = {b: benchmark(closes[b], pd.DatetimeIndex(w["date"])) for b in ("SPY", "QQQ")}
    return ours, bench


def test_parity_check_passes_on_an_exact_table():
    ours, bench = _example()
    their = parse_their_table(_finrlx_table(ours, bench["SPY"], bench["QQQ"]))
    assert len(their) == 7
    assert check_parity(ours, bench, their) == []


def test_parity_check_fails_closed_on_a_missing_or_truncated_table():
    ours, bench = _example()
    assert len(check_parity(ours, bench, parse_their_table("no table in this log"))) == 7
    table = _finrlx_table(ours, bench["SPY"], bench["QQQ"])
    truncated = "\n".join(line for line in table.splitlines() if not line.startswith("Win Rate"))
    problems = check_parity(ours, bench, parse_their_table(truncated))
    assert problems == ["Win Rate: FinRL-X table has 0 of 1 values"]


def test_breakeven_cost_meets_the_target_and_reports_both_fail_directions():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-06", periods=260)
    closes = {s: pd.Series(100 * np.exp(np.cumsum(rng.normal(8e-4, 0.01, len(days)))), index=days) for s in "AB"}
    reb = days[::5]
    rows = [{"A": float(x), "B": float(1 - x)} for x in rng.uniform(0.1, 0.9, len(reb))]
    dates, turnover, ret, _ = _periods(_weights(rows, [str(d.date()) for d in reb]), closes)
    gross = metrics(_equity(dates, turnover, ret, 0.0))["ann_return"]
    be = breakeven_bps(dates, turnover, ret, gross - 0.02, "ann_return")
    assert 0 < be < 100
    assert metrics(_equity(dates, turnover, ret, be))["ann_return"] == pytest.approx(gross - 0.02, abs=1e-9)
    assert breakeven_bps(dates, turnover, ret, gross + 0.01, "ann_return") == 0.0  # never beats the target
    assert breakeven_bps(dates, turnover, ret, -0.99, "ann_return") == float("inf")  # beats it past 100 bps


def test_parity_check_fails_when_one_printed_value_moves_one_unit():
    ours, bench = _example()
    shifted = dict(ours, sharpe_finrlx=ours["sharpe_finrlx"] + 0.01)
    their = parse_their_table(_finrlx_table(shifted, bench["SPY"], bench["QQQ"]))
    problems = check_parity(ours, bench, their)
    assert len(problems) == 1 and problems[0].startswith("Sharpe Ratio [Strategy]")
