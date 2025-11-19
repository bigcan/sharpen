from __future__ import annotations

import json
from pathlib import Path

from finrl_pro.eval.update_phases import _collect


def test_collect_includes_risk_telemetry(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()

    cfg = tmp_path / "action_discrete.yaml"
    cfg.write_text("experiment: demo", encoding="utf-8")

    runs = [
        {
            "config": cfg.as_posix(),
            "fingerprint_id": "fp-1",
        }
    ]
    (matrix / "runs.json").write_text(json.dumps(runs), encoding="utf-8")

    evals = [
        {
            "fingerprint_id": "fp-1",
            "evaluated_metrics": {
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.1,
                "volatility": 0.2,
            },
            "duplicate_returns": False,
        }
    ]
    (matrix / "eval_report.json").write_text(json.dumps(evals), encoding="utf-8")

    risk_summary = {
        cfg.stem: {
            "runs": [
                {
                    "config": cfg.as_posix(),
                    "fingerprint_id": "fp-1",
                    "avg_turnover": 0.1234,
                    "transaction_costs_bps": 155.5,
                }
            ]
        }
    }
    (matrix / "risk_summary.json").write_text(json.dumps(risk_summary), encoding="utf-8")

    grouped = _collect(matrix)
    phase1 = grouped[1]
    assert phase1[0].avg_turnover == 0.1234
    assert phase1[0].transaction_costs_bps == 155.5

