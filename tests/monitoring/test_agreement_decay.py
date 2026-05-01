"""Tests for Protocol v2.3 §8.2-extension AgreementDecayTracker."""

from __future__ import annotations

import pytest

from finrl_pro_ds.monitoring.agreement_decay import (
    CONSENSUS_RULES,
    AgreementDecayStatus,
    AgreementDecayTracker,
)


# ---------- Constructor & validation ----------------------------------------


def test_ctor_rejects_tiny_window():
    with pytest.raises(ValueError, match="window_bars"):
        AgreementDecayTracker(0.10, rule="ens_agreement", window_bars=99)


def test_ctor_rejects_warmup_larger_than_window():
    with pytest.raises(ValueError, match="min_bars_before_check"):
        AgreementDecayTracker(
            0.10, rule="ens_agreement",
            window_bars=200, min_bars_before_check=300,
        )


def test_ctor_rejects_warn_ge_crit():
    with pytest.raises(ValueError, match="warn_delta"):
        AgreementDecayTracker(
            0.10, rule="ens_agreement",
            warn_delta=0.40, crit_delta=0.20,
        )


def test_ctor_rejects_nonpositive_deadband():
    with pytest.raises(ValueError, match="deadband"):
        AgreementDecayTracker(0.10, rule="ens_agreement", deadband=0.0)


def test_ctor_logs_for_non_consensus_rule(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="finrl_pro_ds.monitoring.agreement_decay"):
        AgreementDecayTracker(0.10, rule="ens_mean", window_bars=200)
    assert any(
        "non-consensus rule" in rec.getMessage() for rec in caplog.records
    )


def test_ctor_accepts_all_consensus_rules():
    for rule in CONSENSUS_RULES:
        AgreementDecayTracker(0.10, rule=rule, window_bars=200)


# ---------- Warmup / LOG_ONLY ------------------------------------------------


def test_warmup_returns_warmup_status():
    t = AgreementDecayTracker(
        0.10, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
    )
    report = None
    for _ in range(50):
        report = t.observe(0.0)
    assert report is not None
    assert report.status == AgreementDecayStatus.WARMUP
    assert "50/100" in report.reason


def test_no_baseline_is_log_only():
    t = AgreementDecayTracker(
        None, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
    )
    report = None
    for _ in range(150):
        report = t.observe(0.0)  # 100% flat — would normally trip CRIT
    assert report.status == AgreementDecayStatus.LOG_ONLY
    assert report.flat_bar_frac_live == pytest.approx(1.0)


def test_non_consensus_rule_is_log_only_even_with_baseline():
    t = AgreementDecayTracker(
        0.10, rule="ens_mean",
        window_bars=200, min_bars_before_check=100,
    )
    for _ in range(150):
        report = t.observe(0.0)
    assert report.status == AgreementDecayStatus.LOG_ONLY
    assert "non-consensus" in report.reason


# ---------- OK / WARN / CRIT thresholds -------------------------------------


def _fill(t: AgreementDecayTracker, n: int, action: float):
    for _ in range(n):
        t.observe(action)


def test_ok_when_live_matches_baseline():
    # baseline 30% flat, live ~30% flat → delta ~0
    t = AgreementDecayTracker(
        baseline_flat_frac=0.30, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    # 30 flat (|a|<0.25) + 70 directional
    for _ in range(60):
        t.observe(0.0)
    report = None
    for _ in range(140):
        report = t.observe(0.5)
    # frac flat in last 200 = 60/200 = 0.30 → delta = 0 → OK
    assert report.status == AgreementDecayStatus.OK
    assert report.flat_bar_frac_live == pytest.approx(0.30, abs=1e-9)


def test_warn_fires_above_warn_delta():
    # baseline 0.20, live ~0.45 → delta 0.25 > warn 0.20 but < crit 0.40
    t = AgreementDecayTracker(
        baseline_flat_frac=0.20, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    # 90 flat + 110 directional → 90/200 = 0.45 → delta 0.25
    for _ in range(90):
        t.observe(0.0)
    report = None
    for _ in range(110):
        report = t.observe(0.5)
    assert report.status == AgreementDecayStatus.WARN
    assert report.flat_bar_frac_delta == pytest.approx(0.25, abs=1e-9)


def test_crit_fires_above_crit_delta():
    # baseline 0.20, live 0.80 → delta 0.60 > crit 0.40
    t = AgreementDecayTracker(
        baseline_flat_frac=0.20, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    # 160 flat + 40 directional → 160/200 = 0.80 → delta 0.60
    for _ in range(160):
        t.observe(0.0)
    report = None
    for _ in range(40):
        report = t.observe(0.5)
    assert report.status == AgreementDecayStatus.CRIT
    assert report.flat_bar_frac_delta == pytest.approx(0.60, abs=1e-9)
    assert "CRIT" in report.reason


def test_window_eviction_drops_old_bars():
    # Fill with all-flat then push directional bars to evict the flats.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.30, rule="ens_agreement",
        window_bars=100, min_bars_before_check=50,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    _fill(t, 200, 0.0)  # window full of flats → would CRIT
    report = t.snapshot()
    assert report["status"] == AgreementDecayStatus.CRIT
    _fill(t, 200, 0.5)  # evict everything; window is now 100% directional
    report = t.snapshot()
    # |0.0 - 0.30| = 0.30 → still > warn 0.20 but < crit 0.40
    assert report["status"] == AgreementDecayStatus.WARN
    assert report["flat_bar_frac_live"] == pytest.approx(0.0)


def test_consensus_failure_action_classified_as_flat():
    # ens_agreement returns exactly 0 on consensus failure; verify that
    # any deadband > 0 catches it.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.10, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.001,
    )
    for _ in range(150):
        t.observe(0.0)  # consensus failure
    report = t.snapshot()
    assert report["flat_bar_frac_live"] == pytest.approx(1.0)


def test_deadband_boundary_excludes_equal():
    # |a| < deadband → flat. Equal-to-deadband should NOT count as flat.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.0, rule="ens_agreement",
        window_bars=100, min_bars_before_check=50,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    for _ in range(100):
        t.observe(0.25)  # boundary value
    report = t.snapshot()
    assert report["flat_bar_frac_live"] == pytest.approx(0.0)


def test_to_dict_round_trip():
    t = AgreementDecayTracker(
        0.30, rule="ens_majority",
        window_bars=200, min_bars_before_check=100,
    )
    _fill(t, 150, 0.0)
    d = t.snapshot()
    expected_keys = {
        "status", "reason", "n_bars",
        "flat_bar_frac_live", "flat_bar_frac_baseline", "flat_bar_frac_delta",
        "rule",
    }
    assert set(d.keys()) == expected_keys
    assert d["rule"] == "ens_majority"
