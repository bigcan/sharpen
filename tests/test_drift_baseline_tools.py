"""Regression tests for the live-drift recal helpers (S535-R3).

Covers the pure functions of:
  - scripts/recal_drift_baseline.py — `_summarise`, `_patch_baseline`
  - scripts/drift_watch.py — `_trailing_warn_streak`, `_format_recal_command`

WandB-touching code paths are exercised manually; this file only locks in
the math/aggregation/format contracts so future edits do not silently
break the patched-JSON shape that ActionDriftTracker reads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _import(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def R():
    return _import("recal_drift_baseline")


@pytest.fixture(scope="module")
def W():
    return _import("drift_watch")


# ---- recal_drift_baseline ---------------------------------------------------

def _synth_history(n_per_bucket: int = 119, mean: float = 0.7966,
                   std: float = 0.0112, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    step = 2100
    for q in ("q1", "q2", "q3", "q4"):
        for d in rng.normal(mean, std, n_per_bucket):
            rows.append({
                "_step": step, "drift/status": "OK", "drift/bucket": q,
                "drift/n_bars": 1000,
                "drift/deadband_frac_live": float(d),
                "drift/saturation_frac_live": 0.0,
            })
            step += 5
    return pd.DataFrame(rows)


def test_summarise_global_and_buckets(R):
    hist = _synth_history()
    s = R._summarise(hist, require_status="OK")
    assert s["n_emissions"] == 4 * 119
    assert abs(s["global"]["deadband"]["mean"] - 0.7966) < 0.005
    for q in ("q1", "q2", "q3", "q4"):
        assert q in s["by_vol_quartile"]
        assert s["by_vol_quartile"][q]["deadband"]["n"] == 119


def test_summarise_status_filter_drops_warn(R):
    hist = _synth_history(n_per_bucket=10)
    hist.loc[0:5, "drift/status"] = "WARN"
    s = R._summarise(hist, require_status="OK")
    assert s["n_emissions"] == len(hist) - 6


def test_summarise_raises_on_no_rows_after_filter(R):
    hist = _synth_history(n_per_bucket=2)
    hist["drift/status"] = "WARN"
    with pytest.raises(RuntimeError, match="No rows survive status filter"):
        R._summarise(hist, require_status="OK")


def test_patch_baseline_uniform_writes_global_to_all_buckets(R):
    hist = _synth_history()
    s = R._summarise(hist, require_status="OK")
    base = {
        "protocol": "v2.2_test",
        "ensemble_eval_distribution": {
            "deadband_frac": 0.54, "saturation_frac": 0.0,
            "by_vol_quartile": {
                q: {"deadband_frac": 0.5, "saturation_frac": 0.0}
                for q in ("q1", "q2", "q3", "q4")
            },
        },
    }
    patched, wc = R._patch_baseline(base, s, bucket_strategy="uniform")
    eed = patched["ensemble_eval_distribution"]
    assert abs(eed["deadband_frac"] - 0.7966) < 0.005
    for q in ("q1", "q2", "q3", "q4"):
        assert eed["by_vol_quartile"][q]["deadband_frac"] == eed["deadband_frac"]
    # what_changed records both the global key and per-bucket keys
    assert "ensemble_eval_distribution.deadband_frac" in wc
    assert "ensemble_eval_distribution.by_vol_quartile.q1.deadband_frac" in wc
    old, new = wc["ensemble_eval_distribution.deadband_frac"]
    assert abs(old - 0.54) < 1e-6 and abs(new - 0.7966) < 0.005


def test_patch_baseline_observed_uses_per_bucket_means(R):
    hist = _synth_history()
    s = R._summarise(hist, require_status="OK")
    base = {
        "protocol": "v2.2_test",
        "ensemble_eval_distribution": {
            "deadband_frac": 0.54, "saturation_frac": 0.0,
            "by_vol_quartile": {
                q: {"deadband_frac": 0.5, "saturation_frac": 0.0}
                for q in ("q1", "q2", "q3", "q4")
            },
        },
    }
    patched, _ = R._patch_baseline(base, s, bucket_strategy="observed")
    bvq = patched["ensemble_eval_distribution"]["by_vol_quartile"]
    # All buckets near target but not necessarily equal to global
    for q in ("q1", "q2", "q3", "q4"):
        assert abs(bvq[q]["deadband_frac"] - 0.7966) < 0.005


def test_patch_baseline_rejects_legacy_seed_baseline(R):
    """Stage 2 (per-seed) baselines have no `ensemble_eval_distribution`
    block — the helper must reject them rather than silently no-op."""
    hist = _synth_history(n_per_bucket=2)
    s = R._summarise(hist, require_status="OK")
    base = {"protocol": "v2.2_stage_2_seed_report",
            "per_seed_eval_distribution": {"42": {}}}
    with pytest.raises(KeyError, match="ensemble_eval_distribution"):
        R._patch_baseline(base, s, bucket_strategy="uniform")


def test_attach_metadata_bumps_protocol_once(R):
    cfg = {"protocol": "v2.2_stage_2_5_ensemble_report",
           "ensemble_eval_distribution": {}}
    summary = {"n_emissions": 1, "step_range_observed": [0, 0],
               "global": {"deadband": {"mean": 0.5}, "saturation": {"mean": 0.0}},
               "by_vol_quartile": {}}
    R._attach_metadata(cfg, source_run="e/p/r", step_range=(0, 1),
                       summary=summary, what_changed={},
                       reason="x", require_status="OK",
                       bucket_strategy="uniform", tag="TAG")
    assert "live_recal_TAG" in cfg["protocol"]
    # Calling twice does not double-bump
    R._attach_metadata(cfg, source_run="e/p/r", step_range=(0, 1),
                       summary=summary, what_changed={},
                       reason="x", require_status="OK",
                       bucket_strategy="uniform", tag="TAG2")
    assert cfg["protocol"].count("live_recal") == 1


# ---- drift_watch ------------------------------------------------------------

def test_streak_clean_history(W):
    df = pd.DataFrame({
        "_step": list(range(10)),
        "_timestamp": [1000.0 + i * 60 for i in range(10)],
        "drift/status": ["OK"] * 10,
    })
    n, h, lo, hi = W._trailing_warn_streak(df)
    assert n == 0 and h is None


def test_streak_trailing_warn_with_hours(W):
    df = pd.DataFrame({
        "_step": list(range(10)),
        "_timestamp": [1000.0 + i * 3600 for i in range(10)],
        "drift/status": ["OK"] * 5 + ["WARN"] * 5,
    })
    n, h, lo, hi = W._trailing_warn_streak(df)
    assert n == 5
    assert h == 4.0  # 4 inter-row gaps × 1h
    assert lo == 5 and hi == 9


def test_streak_breaks_at_ok(W):
    """A WARN run interrupted by OK should only count the trailing tail."""
    df = pd.DataFrame({
        "_step": [0, 1, 2, 3],
        "_timestamp": [1000.0, 2000.0, 3000.0, 4000.0],
        "drift/status": ["WARN", "OK", "WARN", "WARN"],
    })
    n, h, lo, hi = W._trailing_warn_streak(df)
    assert n == 2
    assert lo == 2 and hi == 3


def test_streak_empty_df(W):
    n, h, lo, hi = W._trailing_warn_streak(pd.DataFrame())
    assert n == 0


def test_streak_missing_timestamp_returns_none_hours(W):
    df = pd.DataFrame({
        "_step": [0, 1, 2],
        "drift/status": ["WARN", "WARN", "WARN"],
    })
    n, h, lo, hi = W._trailing_warn_streak(df)
    assert n == 3
    assert h is None  # no _timestamp -> hours unknown


def test_format_recal_command_includes_all_args(W):
    rep = W.WatchReport(
        run_id="abc123", run_name="live-sg1-btc-test",
        run_path="ent/proj/abc123", state="running",
        n_drift_rows=500, n_warn_total=120, n_crit_total=0,
        last_status="WARN", last_step=4500, last_ts=1715000000.0,
        warn_streak_n=120, warn_streak_hours=6.0,
        warn_streak_first_step=3000, warn_streak_last_step=4500,
    )
    cmd = W._format_recal_command(
        rep, baseline_hint="baselines/sg1_btc_extended_fold_07/ensemble_report.json",
    )
    for piece in (
        "scripts/recal_drift_baseline.py",
        "--baseline baselines/sg1_btc_extended_fold_07/ensemble_report.json",
        "--source-run ent/proj/abc123",
        "--step-range 3000,4500",
        "--reason ",
        "--dry-run",
    ):
        assert piece in cmd, f"missing {piece!r} in command:\n{cmd}"


def test_format_recal_command_falls_back_when_baseline_unknown(W):
    rep = W.WatchReport(
        run_id="x", run_name="x", run_path="x", state="running",
        n_drift_rows=0, n_warn_total=0, n_crit_total=0,
        last_status=None, last_step=None, last_ts=None,
        warn_streak_n=10, warn_streak_hours=2.0,
        warn_streak_first_step=100, warn_streak_last_step=200,
    )
    cmd = W._format_recal_command(rep, baseline_hint=None)
    assert "<baselines/.../ensemble_report.json>" in cmd
