"""Engine tests for the FinRL-X strategy mission: execution lag, funding, short accounting, costs, causality,
and the seal. Each load-bearing rule has a planted-bug twin that must be caught.

Run: python -m pytest -q tests/research/test_finrlx_strategy_engine.py -p no:cacheprovider
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "scripts"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.finrlx_strategy import book, pnl, seal  # noqa: E402
from research.finrlx_strategy.metrics import breakeven_multiplier, nw_ols  # noqa: E402

CAL = pd.bdate_range("2010-01-04", periods=12)


def _rets(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=CAL[: len(next(iter(cols.values())))])


def _zero_rf(idx) -> pd.Series:
    return pd.Series(0.0, index=idx)


FREE = pnl.CostModel(default_cost_bps=0.0, default_borrow_bps_yr=0.0, cash_spread_bps_yr=0.0)


# ---------------------------------------------------------------- execution lag
def _jump_case():
    r = _rets(A=[0, 0, 0, 0.10, 0, 0], B=[0.0] * 6)
    tgt = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[CAL[2]])   # decided at close of day 2
    return r, tgt


def test_target_does_not_earn_the_bar_after_its_decision():
    r, tgt = _jump_case()
    res = pnl.run(tgt, r, _zero_rf(r.index), FREE)
    # decided day 2, executed at close of day 3: the +10% of day 3 belongs to the OLD (empty) book
    assert res["ret"].iloc[3] == pytest.approx(0.0)
    assert res["longs"].iloc[3] == pytest.approx(1.0)
    assert res["nav"].iloc[-1] == pytest.approx(1.0)


def test_planted_same_bar_execution_is_caught():
    """Planted bug: executing at the decision close (the FinRL-X convention). The lag test must notice."""
    r, tgt = _jump_case()
    planted = tgt.copy()
    planted.index = [CAL[1]]          # equivalent to lag 0 from day 2
    res = pnl.run(planted, r, _zero_rf(r.index), FREE)
    assert res["nav"].iloc[-1] == pytest.approx(1.10)   # the bug captures the jump...
    assert res["ret"].iloc[3] != pytest.approx(0.0)      # ...which the lag assertion above would reject


def test_lag_zero_is_refused():
    r, tgt = _jump_case()
    with pytest.raises(ValueError):
        pnl.run(tgt, r, _zero_rf(r.index), FREE, lag=0)


# ---------------------------------------------------------------- funding
def test_asset_without_price_keeps_its_holding():
    """A market closed on the execution day cannot be traded: the old holding stays, the rest rebalances."""
    r = _rets(A=[0.0, 0.0, 0.05, np.nan, 0.0, 0.0], B=[0.0] * 6)
    tgt = pd.DataFrame({"A": [0.5, 0.0], "B": [0.0, 0.3]}, index=[CAL[0], CAL[2]])   # 2nd executes day 3 (A has no price)
    res = pnl.run(tgt, r, _zero_rf(r.index), FREE)
    nav3 = res["nav"].iloc[3]
    assert res["longs"].iloc[3] == pytest.approx((0.5 * 1.05 + 0.3 * nav3) / nav3, rel=1e-9)   # A kept, B bought
    assert res.attrs["frozen_events"] == 1


def test_planted_trading_a_closed_market_is_caught():
    """Planted bug: trading A at a stale price on a day it has no price. The frozen-holding test must notice."""
    r = _rets(A=[0.0, 0.0, 0.05, np.nan, 0.0, 0.0], B=[0.0] * 6)
    tgt = pd.DataFrame({"A": [0.5, 0.0], "B": [0.0, 0.3]}, index=[CAL[0], CAL[2]])
    res = pnl.run(tgt, r.fillna(0.0), _zero_rf(r.index), FREE)      # the bug: NaN treated as a tradeable 0% day
    assert res.attrs["frozen_events"] == 0 and res["longs"].iloc[3] == pytest.approx(0.3)


def test_margin_loan_is_refused():
    r = _rets(A=[0.0] * 5, B=[0.0] * 5)
    tgt = pd.DataFrame({"A": [0.7], "B": [0.4]}, index=[CAL[0]])
    with pytest.raises(pnl.FundingViolation):
        pnl.run(tgt, r, _zero_rf(r.index), FREE)


def test_fund_projection_largest_feasible_scale():
    core = pd.Series({"SPY": 0.5, "TLT": 0.0, "GLD": 0.0})
    sleeve = pd.Series({"SPY": 0.2, "TLT": 0.6, "GLD": -0.8})
    w, k = book.fund(core, sleeve, long_cap=1.0, short_cap=0.5)
    assert w.clip(lower=0).sum() <= 1.0 + 1e-9 and (-w.clip(upper=0)).sum() <= 0.5 + 1e-9
    # k must be the largest feasible on the grid: 0.5+0.8k <= 1 -> 0.625, 0.8k <= 0.5 -> 0.625
    assert k == pytest.approx(0.625, abs=1 / 400)


def test_fund_nonmonotone_short_spy_offsets_core():
    core = pd.Series({"SPY": 1.0, "TLT": 0.0})
    sleeve = pd.Series({"SPY": -0.4, "TLT": 0.4})     # net longs stay 1.0 for every k
    _, k = book.fund(core, sleeve, long_cap=1.0, short_cap=0.5)
    assert k == pytest.approx(1.0)


# ---------------------------------------------------------------- short accounting
def test_short_gains_when_price_falls_and_proceeds_earn_no_rebate():
    r = _rets(A=[0, 0, -0.10, 0], B=[0.0] * 4)
    rf = pd.Series(0.001, index=r.index)
    tgt = pd.DataFrame({"A": [-0.5], "B": [0.0]}, index=[CAL[0]])     # executes at close of day 1
    cm = pnl.CostModel(default_cost_bps=0.0, default_borrow_bps_yr=0.0, cash_spread_bps_yr=0.0, short_rebate=0.0)
    res = pnl.run(tgt, r, rf, cm)
    nav1 = res["nav"].iloc[1]
    # day 2: short 0.5 of NAV gains 5% of NAV; own cash (1.0 of NAV) earns rf; proceeds earn nothing
    assert res["ret"].iloc[2] == pytest.approx((0.5 * 0.10 * nav1 + nav1 * 0.001) / nav1)


def test_borrow_fee_charged_on_short_notional():
    r = _rets(A=[0.0] * 4, B=[0.0] * 4)
    tgt = pd.DataFrame({"A": [-0.5], "B": [0.0]}, index=[CAL[0]])
    cm = pnl.CostModel(default_cost_bps=0.0, default_borrow_bps_yr=252.0, cash_spread_bps_yr=0.0)
    res = pnl.run(tgt, r, _zero_rf(r.index), cm)
    assert res["borrow"].iloc[2] == pytest.approx(0.5 * 0.0252 / 252, rel=1e-6)


# ---------------------------------------------------------------- costs
def test_trading_cost_is_notional_times_bps():
    r = _rets(A=[0.0] * 4, B=[0.0] * 4)
    tgt = pd.DataFrame({"A": [0.6], "B": [-0.2]}, index=[CAL[0]])
    cm = pnl.CostModel(default_cost_bps=10.0, default_borrow_bps_yr=0.0, cash_spread_bps_yr=0.0)
    res = pnl.run(tgt, r, _zero_rf(r.index), cm)
    assert res["cost"].iloc[1] == pytest.approx(0.8 * 0.0010)


def test_buy_and_hold_matches_total_return_index():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"SPY": rng.normal(0, 0.01, 12), "X": 0.0}, index=CAL)
    res = pnl.buy_and_hold(r, _zero_rf(r.index), "SPY", FREE, start=CAL[1])
    growth = float(np.prod(1 + r["SPY"].iloc[2:]))
    assert res["nav"].iloc[-1] / res["nav"].iloc[0] == pytest.approx(growth)


# ---------------------------------------------------------------- causality (truncation equivalence)
def _synthetic_close(n: int = 700, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    cols = ["SPY", "QQQ", "TLT", "GLD", "FXE", "DBC"]
    lv = np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(n, len(cols))), axis=0))
    return pd.DataFrame(lv, index=idx, columns=cols)


@pytest.mark.parametrize("allocator", ["linear", "cov", "class_rb"])
def test_targets_are_truncation_equivalent(allocator):
    """Tier-0-style leak test: targets decided on or before t must not change when data after t changes."""
    close = _synthetic_close()
    spec = book.BookSpec(allocator=allocator, n_tranches=2, tranche_spacing=5)
    full, _ = book.build_targets(close, spec)
    t = full.index[len(full) // 2]
    bumped = close.copy()
    bumped.loc[bumped.index > t] *= 1.7
    trunc, _ = book.build_targets(bumped, spec)
    a, b = full.loc[:t], trunc.loc[:t]
    pd.testing.assert_frame_equal(a, b)


def test_planted_lookahead_is_caught_by_truncation_test():
    """Planted bug: vol measured including the current bar's future. The truncation test must fail."""
    close = _synthetic_close()
    spec = book.BookSpec(n_tranches=1)
    t = None

    def leaky_raw(c):
        w, vol = orig(c)
        return w, c.pct_change().rolling(63).std().shift(-5) * np.sqrt(252)   # reads 5 bars ahead

    orig = book.raw_trend_weights
    book.raw_trend_weights = leaky_raw
    try:
        full, _ = book.build_targets(close, spec)
        t = full.index[len(full) // 2]
        bumped = close.copy()
        bumped.loc[bumped.index > t] *= np.exp(np.random.default_rng(1).normal(0, 0.05, (int((bumped.index > t).sum()), 6)))
        trunc, _ = book.build_targets(bumped, spec)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(full.loc[:t], trunc.loc[:t])
    finally:
        book.raw_trend_weights = orig


# ---------------------------------------------------------------- stats helpers
def test_nw_ols_recovers_alpha():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.04, 600)
    y = 0.002 + 0.5 * x + rng.normal(0, 0.01, 600)
    r = nw_ols(y, x)
    # SE(alpha) = 0.01/sqrt(600) ~ 0.0004; allow 3 SE
    assert r["alpha"] == pytest.approx(0.002, abs=0.0013) and r["beta"] == pytest.approx(0.5, abs=0.05)
    assert r["t_alpha"] > 3


def test_breakeven_bisection():
    assert breakeven_multiplier(lambda m: 3.0 - m) == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------- seal
def test_backward_view_sealed_without_prereg(tmp_path):
    s = pd.Series(1.0, index=pd.bdate_range("2005-12-20", periods=10))
    with pytest.raises(seal.SealedError):
        seal.backward_view(s, gates_path=tmp_path / "missing.yaml")
    assert (seal.dev_view(s).index > seal.SEAL_END).all()


def test_guard_blocks_sealed_rows(tmp_path):
    s = pd.Series(1.0, index=pd.bdate_range("2005-12-20", periods=10))
    with pytest.raises(seal.SealedError):
        seal.guard(s, purpose="test", gates_path=tmp_path / "missing.yaml")
    seal.guard(seal.dev_view(s), purpose="test", gates_path=tmp_path / "missing.yaml")


def test_vectorised_fund_equals_reference_loop():
    rng = np.random.default_rng(9)
    names = ["SPY", "QQQ", "TLT", "GLD", "FXE", "DBC", "UUP"]
    for _ in range(300):
        core = pd.Series(0.0, index=names)
        core["SPY"] = rng.uniform(0, 1)
        sleeve = pd.Series(rng.normal(0, 0.6, len(names)), index=names)
        lc, sc = 1.0, float(rng.choice([0.0, 0.5, 1.0]))
        a, ka = book.fund(core, sleeve, lc, sc)
        b, kb = book._fund_reference(core, sleeve, lc, sc)
        assert ka == kb
        pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-15, rtol=0)


def test_prereg_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"rule: 1\nwindow: x\n")
    b.write_bytes(b"rule: 1\r\nwindow: x\r\n")
    assert seal._sha256(a) == seal._sha256(b)
    b.write_bytes(b"rule: 2\r\nwindow: x\r\n")
    assert seal._sha256(a) != seal._sha256(b)


def test_lockbox_scores_only_days_after_registration():
    from research.finrlx_strategy import lockbox
    df = pd.DataFrame({"x": 1.0}, index=pd.bdate_range("2026-09-21", periods=10))
    fwd = lockbox.forward_only(df)
    assert fwd.index.min() > lockbox.REGISTERED and len(fwd) == 5          # 2026-09-28 .. 10-02
