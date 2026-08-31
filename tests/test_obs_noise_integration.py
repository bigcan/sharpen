"""Integration tests for the v2.7-B B2 orchestrator ``run_obs_noise_stage``.

These exercise the full wiring — real ``apply_ohlc_noise`` -> materialized
noised parquet -> ``data.file_path`` swap -> aggregate -> gate -> report — using
an injected deterministic fake rollout (``_run_rule`` / ``_load_agents`` hooks),
so no torch / checkpoint / env is required. The fake reads the (possibly noised)
parquet a config points at and builds a portfolio_value path from its close
column, so the sigma=0 nominal must reproduce a direct un-noised rollout (ADR-3)
and sigma>0 must perturb it.

Architecture: ``.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.data.splitter import TimeRange  # noqa: E402
from sharpen.eval.obs_noise import (  # noqa: E402
    FoldNoiseResult,
    InvariantViolation,
    NoiseSpec,
    aggregate_fold,
    build_obs_noise_report,
    run_obs_noise_stage,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic 1-min(-ish) parquet + fake rollout
# ---------------------------------------------------------------------------


def _write_source(tmp_path: Path, freq: str = "1h") -> Path:
    rng = np.random.default_rng(3)
    ts = pd.date_range("2025-08-15", "2025-10-01", freq=freq, tz="UTC")
    n = len(ts)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=n))
    open_ = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.003, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.clip(np.minimum(open_, close) - span, 1e-6, None)
    df = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(1, 100, n),
    })
    src = tmp_path / "source.parquet"
    df.to_parquet(src)
    return src


def _fake_load_agents(cfg, ckpts, device):
    return {"loaded": dict(ckpts)}


def _pv_from_close(parquet_path: str, test_start: str, test_end: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ds = pd.Timestamp(test_start, tz="UTC")
    de = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
    close = df.loc[(ts >= ds) & (ts < de), "close"].to_numpy(dtype=np.float64)
    rets = np.concatenate([[0.0], np.diff(close) / close[:-1]])
    return 100000.0 * np.cumprod(1.0 + rets)


def _fake_run_rule(cfg, agents, rule_name, rule_fn, device, out_dir):
    """Deterministic rollout: PV is the close-return path of whatever parquet the
    config points at, over its test window. Sensitive to the noise (close changes
    => PV changes) and identical to the source path when sigma=0."""
    pv = _pv_from_close(
        cfg["data"]["file_path"],
        cfg["data"]["test_start_date"],
        cfg["data"]["test_end_date"],
    )
    return pd.DataFrame({"portfolio_value": pv})


def _one_fold():
    return [{
        "train": TimeRange("2025-08-15 00:00:00", "2025-09-01 00:00:00"),
        "val": TimeRange("2025-09-01 00:00:00", "2025-09-01 00:00:00"),
        "test": TimeRange("2025-09-01 00:00:00", "2025-10-01 00:00:00"),
    }]


def _base_config(src: Path) -> dict:
    return {
        "protocol_version": "2.7",
        "data": {"file_path": str(src)},
        "env": {"hindsight_weight": 0.0, "deadband_threshold": 0.25},
    }


def _gates(**ov) -> dict:
    g = {
        "obs_noise_pf_floor_10bps": 0.50,
        "obs_noise_pf_floor_50bps": 0.40,
        "obs_noise_mdd_buffer_pp_10bps": 5.0,
        "obs_noise_mdd_buffer_pp_50bps": 8.0,
        "obs_noise_required_min_folds": 1,
        "obs_noise_base_seed": 20260602,
    }
    g.update(ov)
    return g


def _run(tmp_path, specs, gates=None, keep_noised=False, fold_checkpoints=None,
         config=None, folds=None):
    src = _write_source(tmp_path)
    return run_obs_noise_stage(
        config=config or _base_config(src),
        folds=folds or _one_fold(),
        fold_checkpoints=fold_checkpoints or {0: {42: "x", 789: "y", 456: "z"}},
        rule_name="ens_pf_weighted",
        rule_fn=lambda a, db: np.array([0.0]),
        specs=specs,
        gates=gates or _gates(),
        out_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        device="cpu",
        keep_noised=keep_noised,
        workstream="testws",
        ensemble_seeds=[42, 789, 456],
        _load_agents=_fake_load_agents,
        _run_rule=_fake_run_rule,
    )


# ---------------------------------------------------------------------------
# e2e wiring + report schema
# ---------------------------------------------------------------------------


def test_obs_noise_stage_synthetic_e2e(tmp_path):
    specs = [NoiseSpec(0.005, "50bps", 2)]
    results, verdict = _run(tmp_path, specs)

    assert len(results) == 1
    assert isinstance(results[0], FoldNoiseResult)
    assert verdict.n_folds == 1 and verdict.n_folds_graded in (0, 1)
    assert verdict.decision in ("PASS", "FAIL", "UNKNOWN_INSUFFICIENT_FOLDS")

    report_path = tmp_path / "out" / "obs_noise_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["schema"] == "1.0"
    assert report["stage"] == "3.5"
    assert report["protocol_version"] == "2.7"
    assert report["workstream"] == "testws"
    assert report["ensemble_seeds"] == [42, 789, 456]
    assert report["noise_model"]["volume_noised"] is False
    assert report["noise_model"]["noise_start_rule"] == "test_start"
    assert len(report["folds"]) == 1
    ps = report["folds"][0]["per_sigma"]["50bps"]
    assert len(ps["pf_seeds"]) == 2
    assert len(ps["noised_parquet_sha"]) == 2
    assert ps["noised_parquet_sha"][0] != ps["noised_parquet_sha"][1]  # distinct seeds
    assert "pf_ratio" in ps and "mdd_degradation_pp" in ps
    assert report["edge_robustness"]["decision"] == verdict.decision


def test_obs_noise_nominal_matches_direct_unnoised(tmp_path):
    """ADR-3 / OBSNOISE-3: the sigma=0 nominal PF reproduces a direct un-noised
    rollout over the same window (numerator/denominator computed identically)."""
    src = _write_source(tmp_path)
    config = _base_config(src)
    folds = _one_fold()
    results, _ = run_obs_noise_stage(
        config=config, folds=folds,
        fold_checkpoints={0: {42: "x"}},
        rule_name="ens_mean", rule_fn=lambda a, db: np.array([0.0]),
        specs=[NoiseSpec(0.005, "50bps", 1)],
        gates=_gates(),
        out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch",
        device="cpu",
        _load_agents=_fake_load_agents, _run_rule=_fake_run_rule,
    )
    from sharpen.eval.obs_noise import pf_from_pv
    direct_pv = _pv_from_close(str(src), "2025-09-01", "2025-10-01")
    assert results[0].pf_nominal == pytest.approx(pf_from_pv(direct_pv), abs=1e-9)


def test_obs_noise_sigma_perturbs_pf(tmp_path):
    """A real perturbation: at least one noisy seed PF differs from nominal."""
    specs = [NoiseSpec(0.02, "200bps", 3)]
    gates = _gates(obs_noise_pf_floor_200bps=0.40, obs_noise_mdd_buffer_pp_200bps=50.0)
    results, _ = _run(tmp_path, specs, gates=gates)
    r = results[0]
    assert any(abs(p - r.pf_nominal) > 1e-6 for p in r.pf_seeds)


# ---------------------------------------------------------------------------
# scratch management
# ---------------------------------------------------------------------------


def test_obs_noise_deletes_scratch_by_default(tmp_path):
    _run(tmp_path, [NoiseSpec(0.005, "50bps", 2)], keep_noised=False)
    leftover = list((tmp_path / "scratch").rglob("*.parquet"))
    assert leftover == [], f"scratch parquets not cleaned: {leftover}"


def test_obs_noise_keeps_scratch_when_requested(tmp_path):
    _run(tmp_path, [NoiseSpec(0.005, "50bps", 2)], keep_noised=True)
    leftover = list((tmp_path / "scratch").rglob("*.parquet"))
    # 1 nominal + 2 seeds = 3 parquets retained.
    assert len(leftover) == 3


# ---------------------------------------------------------------------------
# preflight invariants
# ---------------------------------------------------------------------------


def test_obs_noise_bug03_hindsight_nonzero_raises(tmp_path):
    src = _write_source(tmp_path)
    cfg = _base_config(src)
    cfg["env"]["hindsight_weight"] = 0.5
    with pytest.raises(InvariantViolation, match="BUG-03"):
        _run(tmp_path, [NoiseSpec(0.005, "50bps", 1)], config=cfg)


def test_obs_noise_leak1_noise_start_before_val_end_raises(tmp_path):
    bad_fold = [{
        "train": TimeRange("2025-08-15 00:00:00", "2025-09-01 00:00:00"),
        "val": TimeRange("2025-09-01 00:00:00", "2025-09-15 00:00:00"),  # val_end > test_start
        "test": TimeRange("2025-09-01 00:00:00", "2025-10-01 00:00:00"),
    }]
    with pytest.raises(InvariantViolation, match="LEAK-1"):
        _run(tmp_path, [NoiseSpec(0.005, "50bps", 1)], folds=bad_fold)


def test_obs_noise_missing_fold_checkpoints_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        _run(tmp_path, [NoiseSpec(0.005, "50bps", 1)], fold_checkpoints={1: {42: "x"}})


def test_obs_noise_missing_file_path_raises(tmp_path):
    cfg = {"protocol_version": "2.7", "data": {}, "env": {"hindsight_weight": 0.0}}
    with pytest.raises(ValueError, match="file_path"):
        _run(tmp_path, [NoiseSpec(0.005, "50bps", 1)], config=cfg)


# ---------------------------------------------------------------------------
# build_obs_noise_report (pure)
# ---------------------------------------------------------------------------


def test_build_obs_noise_report_schema():
    specs = [NoiseSpec(0.001, "10bps", 10), NoiseSpec(0.005, "50bps", 10)]
    folds = _one_fold()
    results = []
    for label, spec in zip(("10bps", "50bps"), specs):
        results.append(aggregate_fold(
            0, spec, pf_nominal=1.6, mdd_nominal=-0.04,
            pf_seeds=[1.5] * 10, mdd_seeds=[-0.05] * 10,
        ))
    from sharpen.eval.obs_noise import resolve_obs_noise_gate
    gates = {
        "obs_noise_pf_floor_10bps": 0.85, "obs_noise_pf_floor_50bps": 0.70,
        "obs_noise_mdd_buffer_pp_10bps": 3.0, "obs_noise_mdd_buffer_pp_50bps": 5.0,
        "obs_noise_required_min_folds": 1,
    }
    verdict = resolve_obs_noise_gate(results, gates)
    report = build_obs_noise_report(
        {"protocol_version": "2.7"}, results, verdict, specs,
        fold_shas={(0, "10bps"): ["a"] * 10, (0, "50bps"): ["b"] * 10},
        folds=folds, rule_name="ens_pf_weighted", gates=gates,
        workstream="ws", ensemble_seeds=[1, 2, 3], device="cpu",
        wall_time_seconds=12.0, wandb_run_id="wid", git_sha="sha",
    )
    assert report["schema"] == "1.0"
    assert report["sigma_levels"] == [
        {"label": "10bps", "sigma": 0.001}, {"label": "50bps", "sigma": 0.005},
    ]
    assert report["thresholds_used"]["obs_noise_pf_floor_10bps"] == 0.85
    assert report["thresholds_used"]["obs_noise_n_seeds"] == 10
    assert report["wall_time_seconds"] == 12.0
    assert report["folds"][0]["test_range"] == ["2025-09-01", "2025-10-01"]
    assert report["folds"][0]["val_end"] == "2025-09-01"
