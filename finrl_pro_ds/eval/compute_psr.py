"""CLI: compute PSR and Sharpe CI from a returns CSV and optionally log to MLflow.

CSV expectations: has a header and one of these columns: ['return', 'daily_return', 'ret'].
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
 

from finrl_pro_ds.eval.statistics import (
    SharpeCI,
    bootstrap_sharpe_ci,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def _read_returns(path: Path) -> list[float]:
    cols = {"return", "daily_return", "ret"}
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        header = {c.strip().lower() for c in rdr.fieldnames or []}
        target = next((c for c in cols if c in header), None)
        if not target:
            raise ValueError(
                f"CSV '{path}' is missing a returns column; expected one of {sorted(cols)}; got {sorted(header)}"
            )
        out: list[float] = []
        for row in rdr:
            try:
                out.append(float(row[target]))
            except (TypeError, ValueError):
                continue
    return out


def _maybe_log_mlflow(payload: dict[str, float], run_id: str | None) -> None:
    if not run_id:
        return
    try:
        import mlflow
    except Exception:
        return
    active = mlflow.active_run()
    if active and active.info.run_id != run_id:
        mlflow.end_run()
        active = None
    started = False
    if not active:
        mlflow.start_run(run_id=run_id)
        started = True
    try:
        mlflow.log_metrics(payload)
    finally:
        if started:
            mlflow.end_run()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro_ds.eval.compute_psr", description="Compute PSR and CI from returns CSV")
    ap.add_argument("--returns-csv", required=True, help="Path to CSV with a returns column.")
    ap.add_argument("--mlflow-run-id", default=None, help="Optional MLflow run id to log metrics to.")
    ap.add_argument("--out-json", default=None, help="Optional path to write JSON with PSR and CI.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level for CI (default 0.05 for 95% CI)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for CI.")
    args = ap.parse_args(argv or None)

    returns = _read_returns(Path(args.returns_csv))
    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    psr = probabilistic_sharpe_ratio(returns)
    ci: SharpeCI = bootstrap_sharpe_ci(returns, alpha=float(args.alpha), B=int(args.bootstrap))

    metrics = {
        "sharpe_ratio": float(sr),
        "sortino_ratio": float(so),
        "psr": float(psr),
        "sharpe_ci_lower": float(ci.lower),
        "sharpe_ci_upper": float(ci.upper),
    }

    # Log to MLflow if requested
    if args.mlflow_run_id:
        _maybe_log_mlflow(metrics, str(args.mlflow_run_id))

    # Persist JSON if requested; otherwise print
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
