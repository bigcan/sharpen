"""Offline end-to-end smoke test for the v2.7-B B4 CLI ``stage_3_5_obs_noise``.

Drives ``main()`` in ``--dry-run`` mode over a synthetic source parquet + minimal
WF config + gates yaml. ``--dry-run`` substitutes a deterministic synthetic
rollout but STILL materializes and swaps real noised parquets, so this exercises
the full CLI wiring (splitter folds -> sigma specs -> per-fold nominal + noisy
cells -> aggregate -> gate -> obs_noise_report.json + exit code) without torch
or checkpoints.

Architecture: ``.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md`` (IC-5)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_3_5_obs_noise import (  # noqa: E402  (torch-free at import)
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_PREFLIGHT,
    EXIT_UNKNOWN,
    main,
)


def _write_source(path: Path) -> None:
    rng = np.random.default_rng(5)
    ts = pd.date_range("2025-08-01", "2025-12-01", freq="1h", tz="UTC")
    n = len(ts)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=n))
    open_ = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.003, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.clip(np.minimum(open_, close) - span, 1e-6, None)
    pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(1, 100, n),
    }).to_parquet(path)


def _write_config(path: Path, src: Path) -> None:
    cfg = {
        "protocol_version": "2.7",
        "data": {
            "file_path": str(src),
            "start_date": "2025-08-01",
            "end_date": "2025-12-01",
        },
        "splitter": {
            "train_months": 1, "val_months": 1, "test_months": 1,
            "step_months": 1, "buffer_days": 0,
        },
        "env": {"hindsight_weight": 0.0, "deadband_threshold": 0.25, "max_leverage": 1.0},
        "ensemble": {
            "seeds": [1, 2, 3], "rule": "ens_mean",
            "seed_pfs": {1: 1.5, 2: 1.4, 3: 1.3},
        },
        "wandb": {"tags": ["research"]},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_gates(path: Path) -> None:
    gates = {
        "aggregation_rule": "ens_mean",
        "gates": {
            "obs_noise_sigma_levels": {"10bps": 0.001, "50bps": 0.005},
            "obs_noise_n_seeds": 2,
            "obs_noise_pf_floor_10bps": 0.50,
            "obs_noise_pf_floor_50bps": 0.40,
            "obs_noise_mdd_buffer_pp_10bps": 5.0,
            "obs_noise_mdd_buffer_pp_50bps": 8.0,
            "obs_noise_required_min_folds": 1,
            "obs_noise_required": False,
        },
    }
    path.write_text(yaml.safe_dump(gates, sort_keys=False), encoding="utf-8")


def test_cli_dry_run_e2e(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    src = tmp_path / "source.parquet"
    cfg = tmp_path / "wf.yaml"
    gates = tmp_path / "gates.yaml"
    out_dir = tmp_path / "out"
    scratch = tmp_path / "scratch"
    _write_source(src)
    _write_config(cfg, src)
    _write_gates(gates)

    rc = main([
        "--workstream", "smoke",
        "--config", str(cfg),
        "--gates-file", str(gates),
        "--out-dir", str(out_dir),
        "--scratch-dir", str(scratch),
        "--device", "cpu",
        "--dry-run",
    ])
    assert rc in (EXIT_PASS, EXIT_FAIL, EXIT_UNKNOWN), f"unexpected exit {rc}"

    report_path = out_dir / "obs_noise_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["schema"] == "1.0"
    assert report["stage"] == "3.5"
    assert report["workstream"] == "smoke"
    assert len(report["folds"]) >= 1
    fold0 = report["folds"][0]
    assert set(fold0["per_sigma"]) == {"10bps", "50bps"}
    assert len(fold0["per_sigma"]["10bps"]["pf_seeds"]) == 2
    assert report["edge_robustness"]["decision"] in (
        "PASS", "FAIL", "UNKNOWN_INSUFFICIENT_FOLDS",
    )
    # scratch parquets cleaned (keep_noised not passed)
    assert list(scratch.rglob("*.parquet")) == []


def test_cli_missing_sigma_levels_preflight(tmp_path):
    src = tmp_path / "source.parquet"
    cfg = tmp_path / "wf.yaml"
    gates = tmp_path / "gates.yaml"
    _write_source(src)
    _write_config(cfg, src)
    # gates without obs_noise_sigma_levels
    gates.write_text(yaml.safe_dump({"aggregation_rule": "ens_mean", "gates": {}}),
                     encoding="utf-8")
    rc = main([
        "--workstream", "smoke", "--config", str(cfg), "--gates-file", str(gates),
        "--out-dir", str(tmp_path / "out"), "--scratch-dir", str(tmp_path / "s"),
        "--dry-run",
    ])
    assert rc == EXIT_PREFLIGHT


def test_cli_missing_config_preflight(tmp_path):
    gates = tmp_path / "gates.yaml"
    _write_gates(gates)
    rc = main([
        "--workstream", "smoke", "--config", str(tmp_path / "nope.yaml"),
        "--gates-file", str(gates), "--dry-run",
    ])
    assert rc == EXIT_PREFLIGHT
