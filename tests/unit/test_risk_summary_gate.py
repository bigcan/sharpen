from __future__ import annotations

import json
from pathlib import Path

import pytest

from finrl_pro.eval.risk_summary_gate import enforce_turnover_cost_limits


def _write_profiles(path: Path, *, max_turnover: float, max_costs: float) -> None:
    payload = {
        "profiles": [
            {
                "profile_id": "default",
                "name": "Default",
                "max_capital_at_risk": 0.1,
                "max_drawdown_pct": 0.2,
                "leverage_cap": 2.0,
                "sandbox_required": True,
                "approved_by": "ci",
                "effective_date": "2025-01-01",
                "fallback_agent": None,
                "max_avg_turnover": max_turnover,
                "max_transaction_costs_bps": max_costs,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config(path: Path, profile_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "experiment_id: test",
                "fingerprint_manifest: finrl_pro/configs/fingerprints.yaml",
                f"risk_profile_file: {profile_path.as_posix()}",
                "risk_profile_id: default",
                "sandbox_enabled: true",
            ]
        ),
        encoding="utf-8",
    )


def _write_risk_summary(path: Path, config: Path, *, avg_turnover: float, costs_bps: float) -> None:
    payload = {
        Path(config).stem: {
            "runs": [
                {
                    "config": config.as_posix(),
                    "fingerprint_id": "fp-test",
                    "seed": 41,
                    "avg_turnover": avg_turnover,
                    "total_turnover": avg_turnover * 100,
                    "transaction_costs_bps": costs_bps,
                }
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_enforce_limits_allows_compliant_runs(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    profiles = tmp_path / "profiles.json"
    _write_profiles(profiles, max_turnover=0.2, max_costs=200.0)
    cfg = tmp_path / "action_discrete.yaml"
    _write_config(cfg, profiles)
    _write_risk_summary(matrix / "risk_summary.json", cfg, avg_turnover=0.15, costs_bps=150.0)

    enforce_turnover_cost_limits(matrix)


def test_enforce_limits_raises_on_turnover_breach(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    profiles = tmp_path / "profiles.json"
    _write_profiles(profiles, max_turnover=0.2, max_costs=200.0)
    cfg = tmp_path / "action_continuous.yaml"
    _write_config(cfg, profiles)
    _write_risk_summary(matrix / "risk_summary.json", cfg, avg_turnover=0.25, costs_bps=150.0)

    with pytest.raises(RuntimeError) as err:
        enforce_turnover_cost_limits(matrix)

    assert "avg_turnover" in str(err.value)

