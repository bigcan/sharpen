"""Integration tests for performance reporting pipeline."""

from __future__ import annotations

from pathlib import Path

import json

from finrl_pro.eval.base import EvaluationContext
from finrl_pro.eval.benchmark_catalog import BenchmarkCatalog
from finrl_pro.eval.report_pipeline import ReportPipeline
from finrl_pro.eval.walk_forward import WalkForwardEvaluator
from finrl_pro.explainability.shap_analysis import ShapAnalysis


def test_reporting_pipeline_generates_report(tmp_path: Path) -> None:
    """Walk-forward evaluation feeds the reporting pipeline end-to-end."""
    catalog = BenchmarkCatalog(manifest_path=Path("finrl_pro/configs/benchmarks.yaml"))
    catalog.load()

    evaluator = WalkForwardEvaluator(catalog=catalog, shap=ShapAnalysis())
    context = EvaluationContext(
        fingerprint_id="fingerprint-123",
        benchmark_id="sp500_rolling_1y",
        walk_forward_splits=3,
    )

    evaluation = evaluator.evaluate(context)
    pipeline = ReportPipeline(reports_dir=tmp_path / "reports")
    report = pipeline.generate(
        fingerprint_id=context.fingerprint_id,
        evaluation=evaluation,
        author="quant_team",
    )

    assert report.fingerprint_id == context.fingerprint_id
    assert report.benchmark_id == "sp500_rolling_1y"
    assert report.approval_status == "Draft"
    assert Path(report.summary_location).exists()

    serialized = report.to_dict()
    assert serialized["metrics"]["sharpe_ratio"] > 1.05
    assert serialized["variance_vs_baseline"]["sharpe_ratio"] > 0
    assert abs(sum(serialized["shap_summary"].values()) - 1.0) < 1e-6

    content = json.loads(Path(report.summary_location).read_text(encoding="utf-8"))
    assert content["report"]["report_id"] == report.report_id
    assert content["benchmark_label"] == "S&P 500 Rolling 1Y"
