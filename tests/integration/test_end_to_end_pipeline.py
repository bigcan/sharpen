"""Regression test covering the fingerprint to report workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.eval.base import EvaluationContext
from finrl_pro.eval.benchmark_catalog import BenchmarkCatalog
from finrl_pro.eval.report_pipeline import ReportPipeline
from finrl_pro.eval.walk_forward import WalkForwardEvaluator
from finrl_pro.training.trainer import Trainer


@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid external MLflow interactions during tests."""

    def _fake_mlflow_run(self, module_versions, metrics, artifact_uris):
        return "test-mlflow-run"

    monkeypatch.setattr(Trainer, "_log_mlflow_run", _fake_mlflow_run, raising=False)


def test_pipeline_generates_compliance_report(tmp_path: Path) -> None:
    """Training → evaluation → reporting produces persisted artifacts."""

    manifest_path = tmp_path / "fingerprints.yaml"
    store = FingerprintStore(manifest_path=manifest_path)
    trainer = Trainer(fingerprint_store=store)

    fingerprint = trainer.run(
        config_path="finrl_pro/configs/experiment_sp500.yaml",
        dataset_hash="dvc://datasets/sp500",
        seed=101,
        module_versions={"finrl_pro.training.trainer": "test"},
        metrics={
            "sharpe_ratio": 1.08,
            "max_drawdown": 0.12,
            "volatility": 0.25,
            "capital_at_risk": 0.04,
            "leverage": 1.5,
        },
        artifact_uris=["s3://finrl-pro/artifacts/checkpoint.pt"],
        baseline_reference="benchmarks:sp500_rolling_1y",
        sandbox_enabled=True,
    )

    assert manifest_path.exists()
    store.load()
    assert store.get(fingerprint.fingerprint_id) is not None

    replayed = trainer.reproduce(fingerprint.fingerprint_id)
    assert replayed.metrics_snapshot["sharpe_ratio"] == pytest.approx(1.08)

    returns_dir = Path("reports") / fingerprint.fingerprint_id
    returns_dir.mkdir(parents=True, exist_ok=True)
    returns_path = returns_dir / "returns.csv"
    with returns_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "return"])
        for idx in range(252):
            value = 0.0015 if idx % 2 else 0.0005
            writer.writerow([idx, value])

    catalog = BenchmarkCatalog(manifest_path=Path("finrl_pro/configs/benchmarks.yaml"))
    catalog.load()

    context = EvaluationContext(
        fingerprint_id=fingerprint.fingerprint_id,
        benchmark_id="sp500_rolling_1y",
        walk_forward_splits=4,
    )
    evaluator = WalkForwardEvaluator(catalog=catalog)
    evaluation = evaluator.evaluate(context)

    pipeline = ReportPipeline(reports_dir=tmp_path / "reports")
    report = pipeline.generate(
        fingerprint_id=context.fingerprint_id,
        evaluation=evaluation,
        author="compliance_bot",
    )

    summary_path = Path(report.summary_location)
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["report"]["fingerprint_id"] == fingerprint.fingerprint_id
    assert payload["benchmark_label"] == "S&P 500 Rolling 1Y"
    assert sum(payload["report"]["shap_summary"].values()) == pytest.approx(1.0)

    serialized = report.to_dict()
    assert serialized["metrics"]["sharpe_ratio"] > 1.05
    assert serialized["variance_vs_baseline"]["sharpe_ratio"] > 0
