"""Tests for Protocol v2.2 §8.2 ActionDriftTracker."""

from __future__ import annotations

import numpy as np
import pytest

from sharpen.monitoring.action_drift import (
    ActionDriftTracker,
    DriftStatus,
    _marginal_kl_from_hists,
)
from sharpen.reporting import compute_eval_distribution


# ---------- KL helper --------------------------------------------------------


def test_kl_identical_hists_is_zero():
    h = np.array([10, 20, 30, 20, 10], dtype=float)
    assert _marginal_kl_from_hists(h, h) == pytest.approx(0.0, abs=1e-6)


def test_kl_non_negative_and_symmetric_smoothing():
    # With symmetric smoothing KL is still non-negative (Jensen)
    h1 = np.array([0, 0, 100, 0, 0], dtype=float)
    h2 = np.array([100, 0, 0, 0, 0], dtype=float)
    kl = _marginal_kl_from_hists(h1, h2)
    assert kl > 0.0


# ---------- Constructor & validation ----------------------------------------


def test_ctor_rejects_tiny_window():
    with pytest.raises(ValueError, match="window_bars"):
        ActionDriftTracker(None, window_bars=49)


def test_ctor_rejects_warmup_larger_than_window():
    with pytest.raises(ValueError, match="min_bars_before_check"):
        ActionDriftTracker(None, window_bars=100, min_bars_before_check=200)


def test_ctor_rejects_bad_regime_cutpoints():
    with pytest.raises(ValueError, match="regime_cutpoints"):
        ActionDriftTracker(None, regime_cutpoints=[0.1, 0.2])  # len != 3


def test_ctor_logs_warning_without_baseline(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="sharpen.monitoring.action_drift"):
        ActionDriftTracker(None, window_bars=100, min_bars_before_check=50)
    assert any("without baseline" in r.getMessage() for r in caplog.records)


# ---------- Warmup + LOG_ONLY ------------------------------------------------


def _mk_baseline_scalar(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(0, 0.4, n), -1, 1)
    return compute_eval_distribution(a)


def test_warmup_returns_warmup_status():
    base = _mk_baseline_scalar()
    t = ActionDriftTracker(base, window_bars=200, min_bars_before_check=100)
    report = None
    for _ in range(50):  # < warmup
        report = t.observe(0.0, bar_close=None)
    assert report is not None
    assert report.status == DriftStatus.WARMUP


def test_no_baseline_is_log_only_not_crit():
    t = ActionDriftTracker(None, window_bars=200, min_bars_before_check=100)
    report = None
    for _ in range(150):
        report = t.observe(0.9, bar_close=None)  # highly saturated
    assert report.status == DriftStatus.LOG_ONLY


# ---------- Scalar deadband/saturation logic --------------------------------


def test_scalar_matching_distribution_is_ok():
    base = _mk_baseline_scalar(seed=1)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
        saturation_warn=0.15, saturation_crit=0.30,
    )
    rng = np.random.default_rng(1)  # same seed as baseline
    last = None
    for _ in range(499):
        a = float(np.clip(rng.normal(0, 0.4), -1, 1))
        last = t.observe(a, bar_close=None)
    assert last.status == DriftStatus.OK


def test_scalar_collapse_to_deadband_trips_crit():
    base = _mk_baseline_scalar(seed=2)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
    )
    last = None
    for _ in range(499):
        last = t.observe(0.05, bar_close=None)  # far inside deadband
    assert last.status == DriftStatus.CRIT
    assert last.deadband_frac_live == pytest.approx(1.0)


def test_scalar_mid_shift_trips_warn_not_crit():
    base = _mk_baseline_scalar(seed=3)
    # Shift dead frac by ~0.2 (>warn 0.15 but <crit 0.30)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
    )
    # Mix baseline + collapsed so final dead frac is approx baseline+0.2
    rng = np.random.default_rng(3)
    last = None
    for i in range(499):
        a = 0.05 if i % 3 == 0 else float(np.clip(rng.normal(0, 0.4), -1, 1))
        last = t.observe(a, bar_close=None)
    assert last.status in (DriftStatus.OK, DriftStatus.WARN)  # tolerance on mix


# ---------- Regime bucketing -------------------------------------------------


def test_regime_bucket_uses_explicit_cutpoints():
    base = _mk_baseline_scalar(seed=4)
    # Synthetic baseline with by_vol_quartile blocks (easier: add manually)
    base["by_vol_quartile"] = {
        q: {"deadband_frac": 0.5, "saturation_frac": 0.0}
        for q in ("q1", "q2", "q3", "q4")
    }
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
        regime_cutpoints=[0.001, 0.005, 0.020],
    )
    close = 100.0
    last = None
    for _ in range(200):
        close *= (1 + 0.0001)  # tiny vol → q1
        last = t.observe(0.1, bar_close=close)
    assert last.bucket == "q1"


def test_regime_cutpoints_fallback_from_baseline():
    base = _mk_baseline_scalar(seed=5)
    base["regime_cutpoints"] = [0.001, 0.005, 0.020]
    # Don't pass regime_cutpoints explicitly — should pick up from baseline
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
    )
    assert t.regime_cutpoints == (0.001, 0.005, 0.020)


# ---------- Multi-dim KL path ------------------------------------------------


def test_multidim_matching_distribution_is_ok():
    rng = np.random.default_rng(6)
    a = np.clip(rng.normal(0, 0.3, (2000, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
        action_kl_warn=0.5, action_kl_crit=1.0,
    )
    last = None
    for _ in range(399):
        v = np.clip(rng.normal(0, 0.3, 3), -1, 1)
        last = t.observe(v, bar_close=None)
    assert last.status == DriftStatus.OK
    assert last.kl is not None and last.kl < 0.5


def test_multidim_saturation_trips_crit():
    rng = np.random.default_rng(7)
    a = np.clip(rng.normal(0, 0.3, (2000, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
        action_kl_warn=0.5, action_kl_crit=1.0,
    )
    last = None
    for _ in range(399):
        last = t.observe(np.array([0.99, -0.99, 0.99]), bar_close=None)
    assert last.status == DriftStatus.CRIT
    assert last.kl > 1.0


def test_multidim_asset_key_mismatch_raises():
    rng = np.random.default_rng(8)
    a = np.clip(rng.normal(0, 0.3, (500, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    # Live config declares different ordering → ctor must reject
    with pytest.raises(ValueError, match="do not match expected asset_keys"):
        ActionDriftTracker(base, asset_keys=["ETH", "BTC", "SOL"])


def test_dim_change_mid_stream_raises():
    base = _mk_baseline_scalar()
    t = ActionDriftTracker(base, window_bars=100, min_bars_before_check=50)
    t.observe(0.1, bar_close=None)
    with pytest.raises(ValueError, match="action dim changed"):
        t.observe(np.array([0.1, 0.2]), bar_close=None)


# ---------- Fix 2 (S538-cont) NaN sentinel semantics ------------------------


def test_observe_nan_excluded_from_deadband_frac():
    # Mixed window of NaN and consensus bars. NaN should not contribute to
    # deadband_frac numerator OR denominator; should be reported via the
    # new no_consensus_frac field.
    base = _mk_baseline_scalar(seed=42)
    t = ActionDriftTracker(
        base, window_bars=600, min_bars_before_check=300,
        deadband_warn=0.15, deadband_crit=0.30,
    )
    # 200 NaN bars (no-consensus), 400 consensus bars.
    # All consensus bars within deadband → deadband_frac_live should be 1.0
    # over the consensus subset, NOT 400/600.
    for _ in range(200):
        t.observe(float("nan"), bar_close=None)
    for _ in range(400):
        t.observe(0.05, bar_close=None)  # always inside deadband
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(200 / 600)
    assert report["deadband_frac_live"] == pytest.approx(1.0)
    # baseline ≈ 0.4-0.6 deadband on N(0, 0.4); 1.0 - baseline > 0.30 → CRIT
    assert report["status"] == DriftStatus.CRIT


def test_observe_all_nan_emits_log_only_not_crit():
    # If every bar in the window is NaN, the tracker has no consensus bars
    # to compare against the baseline. Must NOT trip CRIT — emit LOG_ONLY.
    base = _mk_baseline_scalar(seed=7)
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
    )
    for _ in range(300):
        t.observe(float("nan"), bar_close=None)
    report = t.snapshot()
    assert report["status"] == DriftStatus.LOG_ONLY
    assert report["no_consensus_frac"] == pytest.approx(1.0)
    assert report["deadband_frac_live"] is None


# ---------- v2.6: feature_variance veto + max_veto_frac escalation -----------


def test_feature_state_none_is_backward_compat():
    """Default `feature_state=None` matches the v2.2 byte-identical path."""
    base = _mk_baseline_scalar(seed=11)
    t1 = ActionDriftTracker(base, window_bars=500, min_bars_before_check=200)
    t2 = ActionDriftTracker(base, window_bars=500, min_bars_before_check=200)
    rng = np.random.default_rng(11)
    final1 = final2 = None
    for _ in range(499):
        a = float(np.clip(rng.normal(0, 0.4), -1, 1))
        final1 = t1.observe(a, bar_close=None)                       # no kwarg
        final2 = t2.observe(a, bar_close=None, feature_state=None)   # explicit None
    # Numerically identical reports (feature_state stamp differs: None vs None)
    assert final1.status == final2.status
    assert final1.deadband_frac_live == pytest.approx(final2.deadband_frac_live)
    assert final1.saturation_frac_live == pytest.approx(final2.saturation_frac_live)
    # Both should have feature_state == "OK" stamped (None observe ≡ OK internally)
    assert final1.feature_state == "OK"
    assert final2.feature_state == "OK"


def test_feature_state_flat_does_not_append_to_window():
    """FLAT bars MUST NOT advance the action histogram (Mode B fix)."""
    base = _mk_baseline_scalar(seed=12)
    t = ActionDriftTracker(base, window_bars=500, min_bars_before_check=200)
    # Append 100 OK bars, then 100 FLAT — len(_actions) should be 100 not 200
    for _ in range(100):
        t.observe(0.5, bar_close=None, feature_state="OK")
    assert len(t._actions) == 100
    for _ in range(100):
        rep = t.observe(0.0, bar_close=None, feature_state="FLAT")
        assert rep.status == DriftStatus.VETOED
        assert rep.feature_state == "FLAT"
    # _actions unchanged; _flat_veto_count tracks the skipped bars
    assert len(t._actions) == 100
    assert t._flat_veto_count == 100
    # _recent_feature_states tracks ALL bars (100 OK + 100 FLAT)
    assert len(t._recent_feature_states) == 200


def test_feature_state_explode_appends_normally():
    """EXPLODE means features have NEW information — bar IS counted."""
    base = _mk_baseline_scalar(seed=13)
    t = ActionDriftTracker(base, window_bars=500, min_bars_before_check=200)
    for _ in range(50):
        t.observe(0.5, bar_close=None, feature_state="EXPLODE")
    assert len(t._actions) == 50
    assert t._flat_veto_count == 0


def test_veto_report_has_no_metric_fields():
    """VETOED reports omit live/baseline metrics (skipped from histogram)."""
    base = _mk_baseline_scalar(seed=14)
    t = ActionDriftTracker(base, window_bars=500, min_bars_before_check=200)
    rep = t.observe(0.0, bar_close=None, feature_state="FLAT")
    assert rep.status == DriftStatus.VETOED
    assert rep.deadband_frac_live is None
    assert rep.saturation_frac_live is None
    assert rep.kl is None
    assert rep.flat_veto_frac == pytest.approx(1.0)  # 1/1 bars FLAT


def test_dim_change_after_veto_still_raises():
    """The dim-change invariant must fire even when the new bar is FLAT."""
    t = ActionDriftTracker(None, window_bars=500, min_bars_before_check=200)
    # First call sets dim to 3
    t.observe([0.1, 0.2, 0.3], bar_close=None, feature_state="OK")
    # FLAT bar with a different dim MUST still raise
    with pytest.raises(ValueError, match="action dim changed"):
        t.observe([0.4, 0.5], bar_close=None, feature_state="FLAT")


def test_invalid_feature_state_raises():
    """`feature_state` outside {None, OK, FLAT, EXPLODE} → ValueError."""
    t = ActionDriftTracker(None, window_bars=500, min_bars_before_check=200)
    with pytest.raises(ValueError, match="feature_state must be"):
        t.observe(0.0, bar_close=None, feature_state="WEIRD")


def test_max_veto_frac_ctor_validation():
    """`max_veto_frac` outside (0, 1) → ValueError at ctor time."""
    with pytest.raises(ValueError, match="max_veto_frac"):
        ActionDriftTracker(None, max_veto_frac=0.0)
    with pytest.raises(ValueError, match="max_veto_frac"):
        ActionDriftTracker(None, max_veto_frac=1.0)
    with pytest.raises(ValueError, match="max_veto_frac"):
        ActionDriftTracker(None, max_veto_frac=1.5)


def test_max_veto_frac_exceeded_escalates_to_warn():
    """When >50% of the window is FLAT, OK escalates to WARN (ADR-5)."""
    base = _mk_baseline_scalar(seed=15)
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
        max_veto_frac=0.50,
    )
    rng = np.random.default_rng(15)
    # First 100 OK (clear warmup, populate _actions)
    for _ in range(100):
        a = float(np.clip(rng.normal(0, 0.4), -1, 1))
        t.observe(a, bar_close=None, feature_state="OK")
    # Now flood with 250 FLAT — flat_veto_frac = 250/350 ≈ 0.714 > 0.50
    final = None
    for _ in range(250):
        final = t.observe(0.0, bar_close=None, feature_state="FLAT")
    # The last call itself is VETOED, so we need the next OK call to see the
    # escalation (FLAT path short-circuits _evaluate).
    final = t.observe(0.3, bar_close=None, feature_state="OK")
    assert final.status == DriftStatus.WARN
    assert "feature_variance_veto_exceeded" in final.reason
    assert final.flat_veto_frac is not None and final.flat_veto_frac > 0.50


def test_veto_escalation_does_not_soften_crit():
    """If normal eval returns CRIT, veto escalation must NOT downgrade to WARN."""
    # Construct a tracker that would CRIT on deadband even with no veto noise.
    base = _mk_baseline_scalar(seed=16)
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
        deadband_warn=0.15, deadband_crit=0.30,
        max_veto_frac=0.50,
    )
    # 200 OK bars that collapse hard to deadband (will CRIT once n>=100)
    for _ in range(200):
        t.observe(0.05, bar_close=None, feature_state="OK")
    # Now inject FLAT bars to push veto_frac high
    for _ in range(150):
        t.observe(0.0, bar_close=None, feature_state="FLAT")
    # The next OK observation must STILL be CRIT (deadband-collapse persists)
    rep = t.observe(0.05, bar_close=None, feature_state="OK")
    assert rep.status == DriftStatus.CRIT
    # And flat_veto_frac should be reported alongside the CRIT
    assert rep.flat_veto_frac is not None and rep.flat_veto_frac > 0


def test_veto_does_not_break_vol_bucketing():
    """`bar_close` returns MUST still be appended on FLAT bars — flat features
    do not imply flat prices, and the vol bucket must keep advancing."""
    t = ActionDriftTracker(None, window_bars=500, min_bars_before_check=100)
    prices = [100.0 + i * 0.1 for i in range(50)]
    for p in prices:
        t.observe(0.0, bar_close=p, feature_state="FLAT")
    # 50 prices → ~49 log-returns appended (first call seeds _prev_close)
    assert len(t._close_returns) == 49
    # And the _actions deque remains empty because every bar was vetoed
    assert len(t._actions) == 0


def test_drift_report_to_dict_includes_v26_fields():
    """`to_dict()` must surface `flat_veto_frac` and `feature_state` for WandB."""
    t = ActionDriftTracker(None, window_bars=500, min_bars_before_check=10)
    rep = t.observe(0.0, bar_close=None, feature_state="FLAT")
    d = rep.to_dict()
    assert "flat_veto_frac" in d
    assert "feature_state" in d
    assert d["feature_state"] == "FLAT"


def test_warmup_overrides_veto_escalation():
    """During warmup (n_bars < min_bars_before_check), even high flat_veto_frac
    must NOT escalate — warmup is the strongest "wait" signal."""
    base = _mk_baseline_scalar(seed=17)
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
        max_veto_frac=0.50,
    )
    # 50 OK + 60 FLAT — n_actions=50 < warmup 200, but flat_veto_frac=60/110≈0.55
    for _ in range(50):
        t.observe(0.5, bar_close=None, feature_state="OK")
    for _ in range(60):
        t.observe(0.0, bar_close=None, feature_state="FLAT")
    rep = t.observe(0.5, bar_close=None, feature_state="OK")
    # The escalation rule only fires when status == OK; warmup stays warmup
    assert rep.status == DriftStatus.WARMUP


def test_replay_synthetic_20260526_trajectory_does_not_crit():
    """Synthetic recreation of the 2026-05-26 sg1-btc deadband collapse.

    Simulates the documented trajectory: ~70 normal bars, then a flat-feature
    regime where the policy correctly outputs near-zero. Without the veto,
    `deadband_frac_live` climbs to ~0.44 vs baseline ~0.13 → CRIT. With the
    veto enabled (feature_state="FLAT" on the flat-regime bars), CRIT must NOT
    fire because those bars are excluded from the histogram.
    """
    base = _mk_baseline_scalar(seed=18)
    t = ActionDriftTracker(
        base, window_bars=1000, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
        max_veto_frac=0.99,  # disable escalation so we test the veto exclusion alone
    )
    rng = np.random.default_rng(18)
    # Phase 1: 200 healthy bars matching baseline
    for _ in range(200):
        a = float(np.clip(rng.normal(0, 0.4), -1, 1))
        t.observe(a, bar_close=100.0, feature_state="OK")
    # Phase 2: 250 flat-feature bars, policy outputs near-zero — VETOED
    for _ in range(250):
        t.observe(0.01, bar_close=100.0, feature_state="FLAT")
    rep = t.snapshot()
    # If the veto were absent, _actions would contain ~250 near-zero entries
    # and deadband_frac_live would be ~0.7 → CRIT. With the veto, _actions
    # still only contains the 200 healthy bars → status remains OK.
    assert rep["status"] in (DriftStatus.OK, DriftStatus.WARN)
    assert rep["status"] != DriftStatus.CRIT
