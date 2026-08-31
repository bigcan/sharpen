"""Tests for Protocol v2.3 §8.2-extension AgreementDecayTracker."""

from __future__ import annotations

import pytest

from sharpen.monitoring.agreement_decay import (
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
    with caplog.at_level(logging.WARNING, logger="sharpen.monitoring.agreement_decay"):
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


def test_consensus_failure_emits_nan_not_flat():
    # Fix 2 (S538-cont): the aggregator emits NaN on consensus failure
    # (was 0 pre-Fix-2). The tracker counts NaN as no-consensus, NOT as flat.
    # F2-AUD-01 closure (S548-cont): 100% no-consensus is no longer silent
    # LOG_ONLY — the no_consensus rate signal fires CRIT (default crit=0.60,
    # so 1.0 > 0.60).
    t = AgreementDecayTracker(
        baseline_flat_frac=0.10, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.001,
    )
    nan = float("nan")
    for _ in range(150):
        t.observe(nan)  # consensus failure (post-Fix-2)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(1.0)
    # All bars no-consensus → no consensus bars to compute flat-frac on
    assert report["flat_bar_frac_live"] is None
    # F2-AUD-01: was LOG_ONLY pre-closure; now CRIT via no-consensus signal.
    assert report["status"] == AgreementDecayStatus.CRIT
    assert "no_consensus_frac" in report["reason"]


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
        "rule", "no_consensus_frac",
    }
    assert set(d.keys()) == expected_keys
    assert d["rule"] == "ens_majority"


# ---------- Fix 2 (S538-cont) NaN sentinel semantics ------------------------


def test_observe_nan_recorded_as_no_consensus():
    # Mixed window: 200 bars with 20% NaN, 40% flat-with-consensus, 40%
    # directional-with-consensus. Expect:
    #   no_consensus_frac = 40/200 = 0.20 (< default warn 0.30 → no-consensus OK)
    #   flat_bar_frac_live = 80/160 = 0.50 (within consensus bars only)
    # S548-cont note: this test used to fix 50% NaN as OK status — that
    # implicitly required no_consensus_warn > 0.50. After F2-AUD-01 closure
    # the default warn is 0.30, so the test was rebuilt with 20% NaN to
    # preserve the original intent (verify flat-delta math on a mixed window)
    # while keeping the no-consensus signal OK.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.50, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(40):
        t.observe(nan)
    for _ in range(80):
        t.observe(0.0)  # flat with consensus
    for _ in range(80):
        t.observe(0.5)  # directional with consensus
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.20)
    assert report["flat_bar_frac_live"] == pytest.approx(0.50)
    # flat-delta = |0.50 - 0.50| = 0 (OK); no-consensus 0.20 < warn 0.30 (OK)
    assert report["status"] == AgreementDecayStatus.OK


def test_nan_window_does_not_inflate_flat_frac():
    # Pre-Fix-2 mental model: a 0-on-disagreement bar would be counted as
    # flat → inflates flat_bar_frac_live. Post-Fix-2: NaN bars do NOT
    # count as flat — flat_bar_frac_live is None when n_consensus == 0.
    # F2-AUD-01 closure (S548-cont): 100% NaN now fires CRIT via the
    # no-consensus rate signal, not silent LOG_ONLY.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.10, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(150):
        t.observe(nan)
    report = t.snapshot()
    # All bars NaN → no flat-frac (LOG_ONLY on signal 2) but CRIT on signal 1.
    assert report["status"] == AgreementDecayStatus.CRIT
    assert report["flat_bar_frac_live"] is None
    assert report["no_consensus_frac"] == pytest.approx(1.0)


# ---------- F2-AUD-01 closure (S548-cont): no-consensus rate signal --------


def test_ctor_rejects_no_consensus_warn_out_of_range():
    with pytest.raises(ValueError, match="no_consensus_warn"):
        AgreementDecayTracker(0.10, rule="ens_agreement", no_consensus_warn=0.0)
    with pytest.raises(ValueError, match="no_consensus_warn"):
        AgreementDecayTracker(0.10, rule="ens_agreement", no_consensus_warn=1.0)


def test_ctor_rejects_no_consensus_crit_out_of_range():
    with pytest.raises(ValueError, match="no_consensus_crit"):
        AgreementDecayTracker(0.10, rule="ens_agreement", no_consensus_crit=0.0)
    with pytest.raises(ValueError, match="no_consensus_crit"):
        AgreementDecayTracker(0.10, rule="ens_agreement", no_consensus_crit=1.5)


def test_ctor_rejects_no_consensus_warn_ge_crit():
    with pytest.raises(ValueError, match="no_consensus_warn must be < no_consensus_crit"):
        AgreementDecayTracker(
            0.10, rule="ens_agreement",
            no_consensus_warn=0.50, no_consensus_crit=0.40,
        )


def test_no_consensus_signal_ok_below_warn():
    # 25% NaN < default warn 0.30 → no-consensus signal OK.
    # Mix so the flat-delta signal is also OK (baseline matches live).
    t = AgreementDecayTracker(
        baseline_flat_frac=0.50, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(50):    # 25% NaN
        t.observe(nan)
    for _ in range(75):    # 37.5% of total, 50% of consensus → flat
        t.observe(0.0)
    for _ in range(75):    # 37.5% of total, 50% of consensus → directional
        t.observe(0.5)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.25)
    assert report["flat_bar_frac_live"] == pytest.approx(0.50)
    assert report["status"] == AgreementDecayStatus.OK


def test_no_consensus_signal_warn_in_middle_band():
    # 45% NaN ∈ (warn=0.30, crit=0.60] → WARN, even if flat-delta OK.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.50, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(90):    # 45% NaN
        t.observe(nan)
    for _ in range(55):    # 50% of consensus
        t.observe(0.0)
    for _ in range(55):    # 50% of consensus
        t.observe(0.5)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.45)
    assert report["flat_bar_frac_live"] == pytest.approx(0.50)
    # flat-delta = |0.50 - 0.50| = 0 (OK); no-consensus WARN dominates.
    assert report["status"] == AgreementDecayStatus.WARN
    assert "no_consensus_frac" in report["reason"]


def test_no_consensus_signal_crit_above_crit():
    # 80% NaN > crit=0.60 → CRIT, regardless of flat-delta status.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.50, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(160):    # 80% NaN
        t.observe(nan)
    for _ in range(20):
        t.observe(0.0)
    for _ in range(20):
        t.observe(0.5)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.80)
    assert report["status"] == AgreementDecayStatus.CRIT
    assert "no_consensus_frac" in report["reason"]


def test_max_severity_resolves_both_signals_critting():
    # Both signals CRIT simultaneously: flat-delta CRIT + no-consensus CRIT.
    # Final status CRIT; reason mentions BOTH signals (pipe-joined).
    t = AgreementDecayTracker(
        baseline_flat_frac=0.05, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(140):    # 70% NaN → no-consensus CRIT (>0.60)
        t.observe(nan)
    for _ in range(60):     # 100% of 60 consensus bars flat → flat_frac_live=1.0
        t.observe(0.0)      # |1.0 - 0.05| = 0.95 > flat crit 0.40 → flat CRIT
    report = t.snapshot()
    assert report["status"] == AgreementDecayStatus.CRIT
    # Reason should mention both signals when both are at same severity.
    assert "flat_frac_delta" in report["reason"]
    assert "no_consensus_frac" in report["reason"]


def test_no_consensus_warn_with_flat_ok_takes_warn():
    # Asymmetric: no-consensus WARN, flat-delta OK. Final WARN.
    t = AgreementDecayTracker(
        baseline_flat_frac=0.50, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    for _ in range(80):     # 40% NaN → no-consensus WARN
        t.observe(nan)
    for _ in range(60):     # flat 50% of consensus
        t.observe(0.0)
    for _ in range(60):     # directional 50%
        t.observe(0.5)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.40)
    assert report["flat_bar_frac_live"] == pytest.approx(0.50)  # = baseline → flat OK
    assert report["status"] == AgreementDecayStatus.WARN


def test_no_consensus_crit_overrides_flat_warn():
    # Flat-delta WARN + no-consensus CRIT → final CRIT (max severity).
    t = AgreementDecayTracker(
        baseline_flat_frac=0.20, rule="ens_agreement",
        window_bars=200, min_bars_before_check=100,
        warn_delta=0.20, crit_delta=0.40, deadband=0.25,
    )
    nan = float("nan")
    # 130 NaN = 65% → no-consensus CRIT
    for _ in range(130):
        t.observe(nan)
    # 70 consensus bars: 35 flat → flat_frac_live=0.50; |0.50-0.20|=0.30 → flat WARN
    for _ in range(35):
        t.observe(0.0)
    for _ in range(35):
        t.observe(0.5)
    report = t.snapshot()
    assert report["no_consensus_frac"] == pytest.approx(0.65)
    assert report["flat_bar_frac_delta"] == pytest.approx(0.30)
    # CRIT dominates WARN.
    assert report["status"] == AgreementDecayStatus.CRIT
