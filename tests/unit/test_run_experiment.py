"""Unit tests for simulated training telemetry."""

from __future__ import annotations

import csv
from pathlib import Path

from finrl_pro_ds.training.run_experiment import (
    _persist_artifacts,
    _simulate_training_outputs,
)


def _make_training_cfg() -> dict:
    return {
        "metrics": {
            "sharpe_ratio": 1.1,
            "max_drawdown": 0.15,
            "volatility": 0.25,
        },
        "dataset_hash": "dvc://tests/sim",
        "seed": 41,
        "steps": 256,
        "module_versions": {
            "agent": "PPO",
            "action.space": "CONTINUOUS_[-1,+1]",
        },
        "costs": {
            "per_turnover_bps": 2.0,
        },
    }


def test_simulation_emits_turnover_and_cost_metrics(tmp_path, monkeypatch):
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text("experiment: test\n", encoding="utf-8")

    sim = _simulate_training_outputs(cfg_path=cfg_path, training_cfg=_make_training_cfg())

    assert len(sim.turnover) == len(sim.returns)
    assert len(sim.transaction_costs) == len(sim.returns)
    assert sim.metrics["avg_turnover"] > 0
    assert sim.metrics["transaction_costs_bps"] > 0

    monkeypatch.chdir(tmp_path)
    artifacts = _persist_artifacts("fp-test", sim)
    assert any(p.endswith("execution.csv") for p in artifacts)

    exec_path = Path("reports") / "fp-test" / "execution.csv"
    rows = list(csv.DictReader(exec_path.open("r", encoding="utf-8")))
    assert rows, "execution.csv should contain telemetry rows"
    first = rows[0]
    assert {"position", "trade", "turnover", "transaction_cost"} <= set(first.keys())
