"""Tests for Protocol v2 §4.5 trigger #6 CostDriftTracker."""

from __future__ import annotations

import math

import pytest

from sharpen.monitoring.cost_drift import (
    CostDriftStatus,
    CostDriftTracker,
)


# A fill whose realized one-way cost fraction is exactly `cost_frac`, decomposed
# as `slip_frac` slippage + the remainder as fee. decision_price fixed at 100.
def _fill_kwargs(cost_frac: float, *, slip_frac: float = 0.0, qty: float = 1.0):
    decision_price = 100.0
    fill_price = decision_price * (1.0 + slip_frac)        # slippage_frac = slip_frac
    notional = abs(qty) * fill_price
    fee_frac = cost_frac - slip_frac
    fee = fee_frac * notional                              # fee_frac = fee / notional
    return {
        "fee": fee,
        "avg_fill_price": fill_price,
        "decision_price": decision_price,
        "filled_quantity": qty,
    }


def _drive(t: CostDriftTracker, n: int, cost_frac: float, *, slip_frac: float = 0.0):
    report = None
    for _ in range(n):
        report = t.observe(**_fill_kwargs(cost_frac, slip_frac=slip_frac))
    return report


# ---------- Constructor & validation ----------------------------------------


def test_ctor_rejects_tiny_window():
    with pytest.raises(ValueError, match="window_trades"):
        CostDriftTracker(0.001, window_trades=0)


def test_ctor_rejects_ratio_le_one():
    with pytest.raises(ValueError, match="cost_drift_ratio"):
        CostDriftTracker(0.001, cost_drift_ratio=1.0)
    with pytest.raises(ValueError, match="cost_drift_ratio"):
        CostDriftTracker(0.001, cost_drift_ratio=0.9)


def test_ctor_rejects_negative_config_cost():
    with pytest.raises(ValueError, match="config_cost_frac"):
        CostDriftTracker(-0.001)


def test_ctor_accepts_defaults():
    t = CostDriftTracker(0.00105)
    assert t.window_trades == 100
    assert t.cost_drift_ratio == pytest.approx(1.20)
    assert t.config_cost_frac == pytest.approx(0.00105)


# ---------- LOG_ONLY (no usable denominator) --------------------------------


def test_none_config_cost_is_log_only():
    t = CostDriftTracker(None, window_trades=10)
    report = _drive(t, 20, cost_frac=0.01)  # huge realized cost
    assert report.status == CostDriftStatus.LOG_ONLY
    assert report.cost_ratio is None
    # realized mean still recorded for telemetry
    assert report.realized_cost_frac_mean == pytest.approx(0.01)


def test_zero_config_cost_is_log_only():
    # A non-positive configured cost has no usable denominator → LOG_ONLY,
    # never a divide-by-zero or a spurious FIRED.
    t = CostDriftTracker(0.0, window_trades=10)
    report = _drive(t, 20, cost_frac=0.01)
    assert report.status == CostDriftStatus.LOG_ONLY
    assert t.config_cost_frac is None


# ---------- Warmup -----------------------------------------------------------


def test_warmup_until_full_window():
    t = CostDriftTracker(0.0005, cost_drift_ratio=1.20, window_trades=10)
    report = _drive(t, 5, cost_frac=0.01)  # would fire if window were full
    assert report.status == CostDriftStatus.WARMUP
    assert "5/10" in report.reason
    # cost_ratio is still computed during warmup for WandB telemetry
    assert report.cost_ratio == pytest.approx(0.01 / 0.0005)


# ---------- OK / FIRED thresholds -------------------------------------------


def test_ok_when_realized_matches_config():
    t = CostDriftTracker(0.00105, cost_drift_ratio=1.20, window_trades=10)
    report = _drive(t, 10, cost_frac=0.00105)  # ratio = 1.0
    assert report.status == CostDriftStatus.OK
    assert report.cost_ratio == pytest.approx(1.0)


def test_fee_bump_alone_does_not_fire():
    # Audit point: the 5.0 -> 5.5 bps taker bump alone is only 1.1x and must
    # NOT fire the 1.20x trigger. config = 5.0 bps fee (no slippage),
    # realized = 5.5 bps fee-only.
    t = CostDriftTracker(0.0005, cost_drift_ratio=1.20, window_trades=10)
    report = _drive(t, 10, cost_frac=0.00055, slip_frac=0.0)  # ratio = 1.1
    assert report.status == CostDriftStatus.OK
    assert report.cost_ratio == pytest.approx(1.1)


def test_fee_plus_slippage_fires_cost_drift():
    # Done-when: injected fills that include the ~5 bps live slippage on top of
    # the 5.5 bps fee fire COST_DRIFT against the stale 5.0 bps (fee-only)
    # config assumption — exactly the ~2.1x sim->live gap the audit (P10-04)
    # says the 1.20x trigger would have caught.
    t = CostDriftTracker(0.0005, cost_drift_ratio=1.20, window_trades=10)
    report = _drive(t, 10, cost_frac=0.00105, slip_frac=0.0005)  # 5.5bps fee + 5bps slip
    assert report.status == CostDriftStatus.FIRED
    assert report.cost_ratio == pytest.approx(2.1)
    assert "cost_drift" in report.reason
    assert "not a halt" in report.reason.lower()


def test_just_above_threshold_fires():
    t = CostDriftTracker(0.001, cost_drift_ratio=1.20, window_trades=10)
    _drive(t, 10, cost_frac=0.001 * 1.21)
    assert t.snapshot()["status"] == CostDriftStatus.FIRED


def test_just_below_threshold_ok():
    t = CostDriftTracker(0.001, cost_drift_ratio=1.20, window_trades=10)
    _drive(t, 10, cost_frac=0.001 * 1.19)
    assert t.snapshot()["status"] == CostDriftStatus.OK


def test_at_threshold_is_ok_strict_gt():
    # Strict > : ratio exactly at the threshold is OK, not FIRED.
    t = CostDriftTracker(0.001, cost_drift_ratio=1.20, window_trades=10)
    _drive(t, 10, cost_frac=0.0012)  # ratio = 1.20 exactly
    assert t.snapshot()["status"] == CostDriftStatus.OK


# ---------- Window eviction --------------------------------------------------


def test_window_eviction_clears_fire():
    t = CostDriftTracker(0.0005, cost_drift_ratio=1.20, window_trades=10)
    _drive(t, 10, cost_frac=0.00105, slip_frac=0.0005)  # FIRED
    assert t.snapshot()["status"] == CostDriftStatus.FIRED
    _drive(t, 10, cost_frac=0.0005)  # evict the expensive fills; ratio back to 1.0
    report = t.snapshot()
    assert report["status"] == CostDriftStatus.OK
    assert report["cost_ratio"] == pytest.approx(1.0)


# ---------- Invalid-input guards (Bybit demo None, etc.) --------------------


def test_none_fee_is_skipped_not_zero():
    # Bybit demo can return None fee/fill/qty via ccxt (live_engine S527-cont).
    # A skip must NOT append a zero-cost sample that would drag the mean down.
    t = CostDriftTracker(0.0005, cost_drift_ratio=1.20, window_trades=4)
    _drive(t, 4, cost_frac=0.00105, slip_frac=0.0005)  # FIRED on 4 real fills
    assert t.snapshot()["status"] == CostDriftStatus.FIRED
    # Four None-fee fills would evict the window to all-zero IF appended;
    # instead they are skipped and the window is unchanged.
    for _ in range(4):
        t.observe(fee=None, avg_fill_price=100.0, decision_price=100.0,
                  filled_quantity=1.0)
    assert t.snapshot()["status"] == CostDriftStatus.FIRED
    assert t.snapshot()["n_trades"] == 4


def test_none_qty_and_none_price_skipped():
    t = CostDriftTracker(0.0005, window_trades=10)
    r1 = t.observe(fee=0.05, avg_fill_price=None, decision_price=100.0,
                   filled_quantity=1.0)
    r2 = t.observe(fee=0.05, avg_fill_price=100.0, decision_price=100.0,
                   filled_quantity=None)
    assert r1.n_trades == 0
    assert r2.n_trades == 0


def test_nonpositive_notional_skipped():
    t = CostDriftTracker(0.0005, window_trades=10)
    r = t.observe(fee=0.05, avg_fill_price=0.0, decision_price=100.0,
                  filled_quantity=1.0)
    assert r.n_trades == 0


def test_non_finite_inputs_skipped():
    t = CostDriftTracker(0.0005, window_trades=10)
    r = t.observe(fee=float("nan"), avg_fill_price=100.0, decision_price=100.0,
                  filled_quantity=1.0)
    assert r.n_trades == 0
    r = t.observe(fee=0.05, avg_fill_price=float("inf"), decision_price=100.0,
                  filled_quantity=1.0)
    assert r.n_trades == 0


def test_missing_decision_price_degrades_to_fee_only():
    # No usable decision price → slippage dropped to 0 (fee-only), NOT the whole
    # sample discarded. config = fee, realized = fee → ratio 1.0 (OK).
    t = CostDriftTracker(0.00055, cost_drift_ratio=1.20, window_trades=5)
    for _ in range(5):
        # fee = 5.5 bps of notional, no decision price
        t.observe(fee=0.00055 * 100.0, avg_fill_price=100.0,
                  decision_price=None, filled_quantity=1.0)
    report = t.snapshot()
    assert report["n_trades"] == 5
    assert report["cost_ratio"] == pytest.approx(1.0)
    assert report["status"] == CostDriftStatus.OK


def test_negative_fee_and_qty_use_abs():
    # Sign conventions vary (sells = negative qty; rebate brokers = negative
    # fee). The realized fraction uses magnitudes so a short fill is measured
    # identically to a long fill of the same size.
    t = CostDriftTracker(0.00105, cost_drift_ratio=1.20, window_trades=5)
    for _ in range(5):
        kw = _fill_kwargs(0.00105, slip_frac=0.0005, qty=-1.0)  # short
        t.observe(**kw)
    report = t.snapshot()
    assert report["cost_ratio"] == pytest.approx(1.0)


# ---------- Serialization ----------------------------------------------------


def test_to_dict_round_trip():
    t = CostDriftTracker(0.001, cost_drift_ratio=1.30, window_trades=50)
    _drive(t, 10, cost_frac=0.001)
    d = t.snapshot()
    expected_keys = {
        "status", "reason", "n_trades", "window_trades",
        "config_cost_frac", "cost_drift_ratio",
        "realized_cost_frac_mean", "cost_ratio",
    }
    assert set(d.keys()) == expected_keys
    assert d["window_trades"] == 50
    assert d["cost_drift_ratio"] == pytest.approx(1.30)
    # All values JSON-friendly (no NaN/inf, no custom objects).
    for v in d.values():
        assert v is None or isinstance(v, (str, int, float))
        if isinstance(v, float):
            assert math.isfinite(v)
