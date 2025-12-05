"""Walk-forward evaluation routines for FinRL Pro."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Dict, List

from finrl_pro.eval.base import EvaluationContext
from finrl_pro.eval.benchmark_catalog import BenchmarkCatalog, BenchmarkCatalogEntry
from finrl_pro.explainability.shap_analysis import ShapAnalysis
from finrl_pro.eval.statistics import sharpe_ratio as _sharpe_ratio, std as _std


@dataclass(slots=True)
class EvaluationResult:
    """Represents the outcome of a walk-forward evaluation."""

    benchmark: BenchmarkCatalogEntry
    evaluated_metrics: Dict[str, float]
    variance_vs_baseline: Dict[str, float]
    shap_summary: Dict[str, float]
    walk_forward_splits: int
    returns: List[float] = field(default_factory=list)


class WalkForwardEvaluator:
    """Run walk-forward evaluations against catalog benchmarks."""

    def __init__(self, catalog: BenchmarkCatalog, shap: ShapAnalysis | None = None) -> None:
        self._catalog = catalog
        self._shap = shap or ShapAnalysis()

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Evaluate the supplied fingerprint against its benchmark."""
        if not context.benchmark_id:
            raise ValueError("benchmark_id is required for walk-forward evaluation.")

        entry = self._catalog.get(context.benchmark_id)
        baseline = entry.metrics_baseline
        # Placeholder baseline is used only for variance reference; evaluation now
        # requires artifact-first metrics sourced from returns.csv per fingerprint.
        evaluated_metrics: Dict[str, float] = {}
        variance: Dict[str, float] = {}
        shap_summary: Dict[str, float] = {}

        # Artifact-first: try to read returns.csv
        report_dir = Path("reports") / str(context.fingerprint_id)
        returns_csv = report_dir / "returns.csv"
        rets: List[float] = []
        if not returns_csv.exists():
            raise FileNotFoundError(
                f"returns.csv not found for fingerprint '{context.fingerprint_id}'. "
                "Training runs must emit artifact returns before evaluation."
            )

        with returns_csv.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            hdr = [c.strip().lower() for c in (rdr.fieldnames or [])]
            col = next((candidate for candidate in ("return", "daily_return", "ret") if candidate in hdr), None)
            if not col:
                raise ValueError(
                    f"returns.csv for fingerprint '{context.fingerprint_id}' does not include a supported column header."
                )
            for row in rdr:
                try:
                    rets.append(float(row[col]))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid return value in reports/{context.fingerprint_id}/returns.csv"
                    ) from exc

        if not rets:
            raise ValueError(
                f"No return rows parsed for fingerprint '{context.fingerprint_id}'. "
                "Ensure training emits realized returns before running walk-forward evaluation."
            )

        # Compute equity curve and drawdown from returns
        equity: List[float] = []
        dd: List[float] = []
        cum = 1.0
        peak = 1.0
        for r in rets:
            cum *= (1.0 + r)
            equity.append(cum)
            peak = max(peak, cum)
            dd.append((cum / peak) - 1.0)

        # Emit artifacts under reports/<fingerprint_id>/
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            # equity_curve.csv
            eq_csv = report_dir / "equity_curve.csv"
            eq_lines = ["t,equity"]
            for i, v in enumerate(equity):
                eq_lines.append(f"{i},{v}")
            eq_csv.write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
            # drawdown.csv
            dd_csv = report_dir / "drawdown.csv"
            dd_lines = ["t,drawdown"]
            for i, v in enumerate(dd):
                dd_lines.append(f"{i},{v}")
            dd_csv.write_text("\n".join(dd_lines) + "\n", encoding="utf-8")
        except Exception:
            # Non-fatal: proceed even if artifact write fails
            pass

        # Compute metrics from realized returns and artifacts
        try:
            sharpe = float(_sharpe_ratio(rets))
            vol_realized = float(_std(rets) * (252.0 ** 0.5))
            mdd = float(abs(min(dd)) if dd else 0.0)
        except Exception:
            sharpe = 0.0
            vol_realized = 0.0
            mdd = 0.0

        evaluated_metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "volatility": vol_realized,
        }
        variance = {
            k: evaluated_metrics.get(k, 0.0) - float(baseline.get(k, 0.0)) for k in evaluated_metrics
        }
        shap_summary = self._shap.summarize(variance)

        return EvaluationResult(
            benchmark=entry,
            evaluated_metrics=evaluated_metrics,
            variance_vs_baseline=variance,
            shap_summary=shap_summary,
            walk_forward_splits=context.walk_forward_splits,
            returns=rets,
        )
