"""F2-AUD-02 closure (S548-cont): pre-Fix-2 baseline + consensus rule guard.

Pre-Fix-2 baselines folded consensus-failure bars (aggregator returned 0)
into `deadband_frac` — both numerator and denominator included those bars.
Post-Fix-2 the live `AgreementDecayTracker` excludes NaN bars from both.
Pairing the two would compute a biased `flat_frac_delta` (apples-to-oranges)
and produce false WARN/CRIT.

The guard at `_init_agreement_decay_tracker` detects the schema mismatch
(pre-Fix-2 baseline lacks the `no_consensus_frac` field) when paired with
a consensus rule (`ens_agreement` / `ens_majority`) and degrades the
tracker to LOG_ONLY by setting `baseline_flat_frac=None`. WandB telemetry
still flows; gating is suppressed until the operator re-bakes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sharpen.crypto.live.live_engine import _init_agreement_decay_tracker


def _write_baseline(path: Path, *, rule: str, include_fix2_marker: bool) -> None:
    eed = {
        "histogram_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "counts": [10, 20, 30, 20, 10],
        "mean": 0.0,
        "std": 0.5,
        "entropy": 1.5,
        "deadband_frac": 0.25,
        "saturation_frac": 0.01,
        "n": 90,
        "composition_rule": rule,
    }
    if include_fix2_marker:
        eed["no_consensus_frac"] = 0.10
        eed["n_total_bars"] = 100
    payload = {
        "protocol": "v2.2_stage_2_5_ensemble_report",
        "ensemble_eval_distribution": eed,
    }
    path.write_text(json.dumps(payload))


def _build_config(baseline_path: Path) -> dict:
    return {
        "drift": {
            "enabled": True,
            "baseline_path": str(baseline_path),
        },
        "gates": {"drift": {"agreement_flat_window_bars": 200}},
        "trading": {"deadband_threshold": 0.25},
    }


def _mock_agent(rule: str):
    agent = MagicMock()
    agent.aggregation_rule = rule
    agent.deadband = 0.25
    return agent


def test_pre_fix2_baseline_with_ens_agreement_degrades_to_log_only(tmp_path, caplog):
    """Pre-Fix-2 baseline (no `no_consensus_frac`) + `ens_agreement` →
    tracker constructed with baseline_flat_frac=None (LOG_ONLY)."""
    baseline = tmp_path / "ensemble_report.json"
    _write_baseline(baseline, rule="ens_agreement", include_fix2_marker=False)

    with caplog.at_level(logging.WARNING):
        tracker = _init_agreement_decay_tracker(
            _build_config(baseline), _mock_agent("ens_agreement"),
        )

    assert tracker is not None
    assert tracker.baseline_flat_frac is None, (
        "guard should suppress baseline_flat_frac so flat-delta signal "
        "stays LOG_ONLY"
    )
    assert any(
        "pre-Fix-2 schema" in rec.getMessage()
        and "F2-AUD-02" in rec.getMessage()
        for rec in caplog.records
    ), f"expected pre-Fix-2 warning; got {[r.getMessage() for r in caplog.records]!r}"


def test_pre_fix2_baseline_with_ens_majority_also_degrades(tmp_path, caplog):
    """Same guard applies to the other consensus rule."""
    baseline = tmp_path / "ensemble_report.json"
    _write_baseline(baseline, rule="ens_majority", include_fix2_marker=False)

    with caplog.at_level(logging.WARNING):
        tracker = _init_agreement_decay_tracker(
            _build_config(baseline), _mock_agent("ens_majority"),
        )

    assert tracker is not None
    assert tracker.baseline_flat_frac is None


def test_fix2_baseline_with_ens_agreement_loads_baseline(tmp_path):
    """Fix-2 baseline (has `no_consensus_frac`) + `ens_agreement` → tracker
    loads baseline_flat_frac normally; gating is armed."""
    baseline = tmp_path / "ensemble_report.json"
    _write_baseline(baseline, rule="ens_agreement", include_fix2_marker=True)

    tracker = _init_agreement_decay_tracker(
        _build_config(baseline), _mock_agent("ens_agreement"),
    )

    assert tracker is not None
    assert tracker.baseline_flat_frac == pytest.approx(0.25), (
        "Fix-2 baseline should load deadband_frac into the tracker"
    )


def test_non_consensus_rule_skips_tracker_entirely(tmp_path):
    """Non-consensus rules (`ens_mean` / `ens_median` / `ens_pf_weighted`)
    return None — guard never runs. Pre-Fix-2 baseline doesn't matter
    because those rules don't produce NaN."""
    baseline = tmp_path / "ensemble_report.json"
    _write_baseline(baseline, rule="ens_mean", include_fix2_marker=False)

    tracker = _init_agreement_decay_tracker(
        _build_config(baseline), _mock_agent("ens_mean"),
    )

    assert tracker is None, "non-consensus rule should skip tracker construction"


def test_missing_baseline_field_still_degrades(tmp_path, caplog):
    """Existing path: baseline file present but no `deadband_frac` →
    LOG_ONLY (unchanged behavior, not the F2-AUD-02 guard)."""
    baseline = tmp_path / "ensemble_report.json"
    baseline.write_text(json.dumps({
        "protocol": "v2.2_stage_2_5_ensemble_report",
        "ensemble_eval_distribution": {"n": 100},
    }))

    with caplog.at_level(logging.WARNING):
        tracker = _init_agreement_decay_tracker(
            _build_config(baseline), _mock_agent("ens_agreement"),
        )

    assert tracker is not None
    assert tracker.baseline_flat_frac is None
    # Should match the OLD warning, not the F2-AUD-02 one.
    assert any(
        "has no" in rec.getMessage() and "deadband_frac" in rec.getMessage()
        for rec in caplog.records
    )
    # F2-AUD-02 specific marker should NOT be in this branch's warning.
    assert not any(
        "F2-AUD-02" in rec.getMessage() for rec in caplog.records
    )


def test_drift_disabled_returns_none(tmp_path):
    """drift.enabled=false short-circuits regardless of baseline."""
    baseline = tmp_path / "ensemble_report.json"
    _write_baseline(baseline, rule="ens_agreement", include_fix2_marker=False)

    config = _build_config(baseline)
    config["drift"]["enabled"] = False

    tracker = _init_agreement_decay_tracker(config, _mock_agent("ens_agreement"))

    assert tracker is None
