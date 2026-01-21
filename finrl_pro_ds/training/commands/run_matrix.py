"""CLI to execute a matrix of experiment configs, expand sweeps, and generate reports.

This orchestrates multiple calls to `finrl_pro_ds.training.run_experiment` and then
reproduces + evaluates each fingerprint to produce a consolidated JSON and Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from finrl_pro_ds.configs.fingerprint_store import FingerprintStore
from finrl_pro_ds.eval.base import EvaluationContext
from finrl_pro_ds.eval.benchmark_catalog import BenchmarkCatalog
from finrl_pro_ds.eval.walk_forward import WalkForwardEvaluator
from finrl_pro_ds.eval.statistics import probabilistic_sharpe_ratio
from finrl_pro_ds.training.trainer import Trainer


@dataclass(slots=True)
class RunRecord:
    config: str
    fingerprint_id: str
    manifest: str


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _expand_sweeps(base_path: Path, cfg: dict[str, Any]) -> List[tuple[Path, dict[str, Any]]]:
    """Expand sweep metadata into distinct config variants.

    Supported keys under `sweep`:
      - risk_profile_ids: list[str]
      - agents: list[str] (written to training.module_versions.agent)
      - seeds: list[int] (written to training.seed)
    """
    variants: List[tuple[Path, dict[str, Any]]] = []
    sweep = dict(cfg.get("sweep", {}) or {})
    risk_ids: List[str] = list(sweep.get("risk_profile_ids", []) or [])
    agents: List[str] = list(sweep.get("agents", []) or [])
    seeds: List[int] = list(sweep.get("seeds", []) or [])

    # If no sweep keys, return original
    if not risk_ids and not agents and not seeds:
        return [(base_path, cfg)]

    # Build cartesian over present dimensions (only those provided)
    if not risk_ids:
        risk_ids = [cfg.get("risk_profile_id", "default")]
    if not agents:
        # fall back to current agent in module_versions or placeholder
        agent_cur = (
            cfg.get("training", {})
            .get("module_versions", {})
            .get("agent", "PPO")
        )
        agents = [str(agent_cur)]
    if not seeds:
        seeds = [int(cfg.get("training", {}).get("seed", 42))]

    for rid in risk_ids:
        for agent in agents:
            for seed in seeds:
                clone = json.loads(json.dumps(cfg))  # deep copy via JSON
                clone["risk_profile_id"] = rid
                t = clone.setdefault("training", {})
                t.setdefault("module_versions", {})["agent"] = str(agent)
                t["seed"] = int(seed)
                # synthesize filename
                name = base_path.stem + f"__risk-{rid}__agent-{agent}__seed-{seed}.yaml"
                variants.append((base_path.with_name(name), clone))
    return variants


def _run_experiment(config_path: Path) -> RunRecord:
    """Invoke the training entrypoint as a subprocess and parse its JSON output."""
    cmd = [
        "python",
        "-m",
        "finrl_pro_ds.training.run_experiment",
        "--config",
        str(config_path),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    # Parse JSON object from stdout block (handles pretty-printed multi-line JSON)
    stdout = proc.stdout
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(
            f"Failed to locate JSON in output for config {config_path}.\nSTDOUT:\n{stdout}\nSTDERR:\n{proc.stderr}"
        )
    blob = stdout[start : end + 1]
    payload: Dict[str, Any] = json.loads(blob)
    return RunRecord(
        config=str(payload["config"]),
        fingerprint_id=str(payload["fingerprint_id"]),
        manifest=str(payload["manifest"]),
    )


def _evaluate_fingerprints(
    records: List[RunRecord],
    *,
    splits: int,
    benchmarks_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    store = FingerprintStore(manifest_path=Path(records[0].manifest))
    store.load()
    trainer = Trainer(fingerprint_store=store)
    catalog = BenchmarkCatalog(benchmarks_path)
    catalog.load()
    wf = WalkForwardEvaluator(catalog)

    evaluations: list[dict[str, Any]] = []
    seen_digests: dict[str, str] = {}
    telemetry: list[dict[str, Any]] = []
    for rec in records:
        fp = trainer.reproduce(rec.fingerprint_id)
        ref = fp.baseline_reference or ""
        bench_id = ref.split(":", 1)[1] if ":" in ref else ref
        ctx = EvaluationContext(
            fingerprint_id=fp.fingerprint_id,
            benchmark_id=bench_id,
            walk_forward_splits=int(splits),
        )
        res = wf.evaluate(ctx)
        psr = probabilistic_sharpe_ratio(res.returns) if res.returns else None
        # Duplicate artifact guard: mark evaluations that reuse identical returns.csv
        dup_flag = False
        dup_of = None
        try:
            rpath = Path("reports") / fp.fingerprint_id / "returns.csv"
            if rpath.exists():
                data = rpath.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_digests:
                    dup_flag = True
                    dup_of = seen_digests[digest]
                else:
                    seen_digests[digest] = fp.fingerprint_id
        except Exception:
            pass

        risk_metrics = {
            "avg_turnover": fp.metrics_snapshot.get("avg_turnover"),
            "total_turnover": fp.metrics_snapshot.get("total_turnover"),
            "transaction_costs_bps": fp.metrics_snapshot.get("transaction_costs_bps"),
        }

        evaluations.append(
            {
                "config": rec.config,
                "fingerprint_id": fp.fingerprint_id,
                "mlflow_run_id": fp.mlflow_run_id,
                "seed": fp.seed,
                "benchmark_id": bench_id,
                "evaluated_metrics": res.evaluated_metrics,
                "variance_vs_baseline": res.variance_vs_baseline,
                "shap_summary": res.shap_summary,
                "walk_forward_splits": res.walk_forward_splits,
                "duplicate_returns": dup_flag,
                "duplicate_of": dup_of or "",
                "psr": psr,
                "risk_metrics": risk_metrics,
            }
        )
        telemetry.append(
            {
                "fingerprint_id": fp.fingerprint_id,
                "config": rec.config,
                "seed": fp.seed,
                "metrics_snapshot": dict(fp.metrics_snapshot),
            }
        )
    return evaluations, telemetry


def _write_markdown_report(path: Path, *, runs: list[RunRecord], evals: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FinRL Pro Matrix Report")
    lines.append("")
    lines.append("## Runs")
    for r in runs:
        lines.append(
            f"- config: `{r.config}` — fingerprint: `{r.fingerprint_id}` — manifest: `{r.manifest}`"
        )
    lines.append("")
    lines.append("## Evaluations (Walk-Forward)")
    for ev in evals:
        em = ev["evaluated_metrics"]
        psr = ev.get("psr")
        psr_str = f"{psr:.2f}" if isinstance(psr, (int, float)) else "n/a"
        lines.append(
            f"- fp `{ev['fingerprint_id']}` | bench `{ev['benchmark_id']}` | splits {ev['walk_forward_splits']} "
            f"| Sharpe {em.get('sharpe_ratio'):.2f} | MaxDD {em.get('max_drawdown'):.2f} "
            f"| Vol {em.get('volatility'):.2f} | PSR {psr_str}"
        )
        risk = ev.get("risk_metrics") or {}
        avg_turn = risk.get("avg_turnover")
        total_turn = risk.get("total_turnover")
        cost_bps = risk.get("transaction_costs_bps")
        if any(value is not None for value in (avg_turn, total_turn, cost_bps)):
            avg_str = f"{avg_turn:.4f}" if isinstance(avg_turn, (int, float)) else "n/a"
            total_str = f"{total_turn:.2f}" if isinstance(total_turn, (int, float)) else "n/a"
            cost_str = f"{cost_bps:.1f}" if isinstance(cost_bps, (int, float)) else "n/a"
            lines.append(
                f"  ↳ turnover(avg {avg_str}, total {total_str}) | costs {cost_str} bps"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_base(config_path: str) -> str:
    stem = Path(config_path).stem
    return stem.split("__", 1)[0]


def _mean(values: Iterable[float | None]) -> float | None:
    data = [float(v) for v in values if isinstance(v, (int, float))]
    if not data:
        return None
    return sum(data) / len(data)


def _write_summary_report(path: Path, evals: list[dict[str, Any]]) -> None:
    summary: dict[str, dict[str, Any]] = {}
    for ev in evals:
        cfg = _config_base(ev["config"])
        bucket = summary.setdefault(
            cfg,
            {
                "runs": [],
            },
        )
        bucket["runs"].append(ev)

    payload: dict[str, Any] = {}
    for cfg, rows in summary.items():
        runs = rows["runs"]
        payload[cfg] = {
            "count": len(runs),
            "mean_sharpe": _mean(r["evaluated_metrics"].get("sharpe_ratio") for r in runs),
            "mean_psr": _mean(r.get("psr") for r in runs),
            "mean_maxdd": _mean(r["evaluated_metrics"].get("max_drawdown") for r in runs),
            "mean_avg_turnover": _mean(
                (r.get("risk_metrics") or {}).get("avg_turnover") for r in runs
            ),
            "mean_total_turnover": _mean(
                (r.get("risk_metrics") or {}).get("total_turnover") for r in runs
            ),
            "mean_transaction_costs_bps": _mean(
                (r.get("risk_metrics") or {}).get("transaction_costs_bps") for r in runs
            ),
            "runs": [
                {
                    "fingerprint_id": r["fingerprint_id"],
                    "seed": r.get("seed"),
                    "config": r["config"],
                    "sharpe_ratio": r["evaluated_metrics"].get("sharpe_ratio"),
                    "max_drawdown": r["evaluated_metrics"].get("max_drawdown"),
                    "psr": r.get("psr"),
                    "avg_turnover": (r.get("risk_metrics") or {}).get("avg_turnover"),
                    "total_turnover": (r.get("risk_metrics") or {}).get("total_turnover"),
                    "transaction_costs_bps": (r.get("risk_metrics") or {}).get("transaction_costs_bps"),
                }
                for r in runs
            ],
        }

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro_ds.training.commands.run_matrix",
        description="Run a matrix of FinRL Pro experiments and produce a consolidated report.",
    )
    parser.add_argument(
        "--experiments-dir",
        default="finrl_pro_ds/configs/experiments",
        help="Directory to scan for experiment YAMLs.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/matrix",
        help="Directory to write aggregated outputs.",
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=5,
        help="Number of splits for evaluation.",
    )
    parser.add_argument(
        "--benchmarks",
        default="finrl_pro_ds/configs/benchmarks.yaml",
        help="Path to benchmark catalog.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    exp_dir = Path(args.experiments_dir)
    out_dir = Path(args.output_dir)
    tmp_dir = Path("tmp/matrix_inputs")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Collect YAMLs
    candidates = sorted([p for p in exp_dir.glob("*.yaml")])

    # Expand sweeps
    scheduled: list[Path] = []
    for cfg_path in candidates:
        cfg = _load_yaml(cfg_path)
        for variant_path, payload in _expand_sweeps(cfg_path, cfg):
            # if unchanged, keep original reference; else write to tmp
            if variant_path == cfg_path:
                scheduled.append(cfg_path)
            else:
                tmp_yaml = tmp_dir / variant_path.name
                _write_yaml(tmp_yaml, payload)
                scheduled.append(tmp_yaml)

    # Deduplicate schedule preserving order
    seen: set[str] = set()
    unique_schedule: list[Path] = []
    for p in scheduled:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_schedule.append(p)

    # Run all experiments
    run_records: list[RunRecord] = []
    for cfg in unique_schedule:
        try:
            rec = _run_experiment(cfg)
            run_records.append(rec)
        except subprocess.CalledProcessError as exc:  # risk policy or other failure
            # Log a minimal failure artifact and continue
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "failures.log").write_text(
                f"FAILED: {cfg} -> {exc}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}\n",
                encoding="utf-8",
            )

    # Persist runs
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dicts = [{"config": r.config, "fingerprint_id": r.fingerprint_id, "manifest": r.manifest} for r in run_records]
    (out_dir / "runs.json").write_text(
        json.dumps(run_dicts, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Reproduce and evaluate
    evaluations, telemetry = _evaluate_fingerprints(
        run_records,
        splits=int(args.walk_forward_splits),
        benchmarks_path=Path(args.benchmarks),
    )
    (out_dir / "fingerprints.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "eval_report.json").write_text(
        json.dumps(evaluations, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown_report(out_dir / "report.md", runs=run_records, evals=evaluations)
    _write_summary_report(out_dir / "risk_summary.json", evaluations)

    # Best-effort: update leaderboard from the newly generated matrix artifacts
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "finrl_pro_ds.eval.update_leaderboard",
                "--matrix-dir",
                str(out_dir),
                "--leaderboard",
                "docs/leaderboard.md",
            ],
            check=True,
            text=True,
        )
    except Exception:
        # Non-fatal if leaderboard update fails
        pass

    print(json.dumps({
        "runs": str((out_dir / "runs.json").as_posix()),
        "fingerprints": str((out_dir / "fingerprints.json").as_posix()),
        "evaluations": str((out_dir / "eval_report.json").as_posix()),
        "markdown": str((out_dir / "report.md").as_posix()),
        "risk_summary": str((out_dir / "risk_summary.json").as_posix()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
