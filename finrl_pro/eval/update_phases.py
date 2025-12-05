"""CLI: Generate phase run lists and summaries from matrix results.

Produces Markdown files under reports/matrix:
- phase1_runs.md, phase1_summary.md, ..., phase5_*.md
Optionally appends a concise re-run update to final_report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from finrl_pro.eval.statistics import probabilistic_sharpe_ratio
from finrl_pro.eval.risk_summary_gate import (
    enforce_turnover_cost_limits,
    load_risk_summary_runs,
    normalize_config_key,
)


@dataclass(slots=True)
class RunEval:
    fingerprint: str
    config: str
    sharpe: Optional[float]
    maxdd: Optional[float]
    vol: Optional[float]
    psr: Optional[float]
    duplicate: bool
    avg_turnover: Optional[float]
    transaction_costs_bps: Optional[float]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_returns(path: Path) -> list[float]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        out: list[float] = []
        for row in rdr:
            try:
                out.append(float(row.get("return") or row.get("daily_return") or row.get("ret") or ""))
            except Exception:
                continue
    return out


def _classify_phase(config_path: str) -> Optional[int]:
    stem = Path(config_path).stem.lower()
    if any(tag in stem for tag in ("multi", "basket", "multiasset")):
        return 5
    if any(tag in stem for tag in ("cost", "ope", "paper", "stress", "robust", "robustness", "slippage")):
        return 4
    if "agent_" in stem:
        return 3
    if "action_" in stem:
        return 1
    if any(k in stem for k in ("feature", "fracdiff", "wavelet")):
        return 2
    if "sp500_daily" in stem:
        return 0
    return None


def _collect(matrix_dir: Path) -> Dict[Optional[int], List[RunEval]]:
    runs = _load_json(matrix_dir / "runs.json")
    evals = _load_json(matrix_dir / "eval_report.json")
    eval_by_fp: Dict[str, Dict[str, Any]] = {str(e["fingerprint_id"]): e for e in evals}
    risk_runs = load_risk_summary_runs(matrix_dir / "risk_summary.json")
    grouped: Dict[Optional[int], List[RunEval]] = {}
    for r in runs:
        fp = str(r.get("fingerprint_id"))
        cfg = str(r.get("config"))
        phase = _classify_phase(cfg)
        if phase is None:
            continue
        ev = eval_by_fp.get(fp)
        if not ev:
            continue
        metrics = dict(ev.get("evaluated_metrics") or {})
        sharpe = float(metrics.get("sharpe_ratio", 0.0)) if metrics.get("sharpe_ratio") is not None else None
        maxdd = float(metrics.get("max_drawdown", 0.0)) if metrics.get("max_drawdown") is not None else None
        vol = float(metrics.get("volatility", 0.0)) if metrics.get("volatility") is not None else None
        dup = bool(ev.get("duplicate_returns", False))
        # PSR from artifacts
        rets_csv = Path("reports") / fp / "returns.csv"
        try:
            rets = _read_returns(rets_csv)
            psr = float(probabilistic_sharpe_ratio(rets)) if rets else None
        except Exception:
            psr = None
        telemetry = risk_runs.get(normalize_config_key(cfg))
        rec = RunEval(
            fingerprint=fp,
            config=cfg,
            sharpe=sharpe,
            maxdd=maxdd,
            vol=vol,
            psr=psr,
            duplicate=dup,
            avg_turnover=telemetry.avg_turnover if telemetry else None,
            transaction_costs_bps=telemetry.transaction_costs_bps if telemetry else None,
        )
        grouped.setdefault(phase, []).append(rec)
    return grouped


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "-"


def _write_phase_files(out_dir: Path, phase: int, rows: List[RunEval]) -> Tuple[Optional[RunEval], Path, Path]:
    runs_md = out_dir / f"phase{phase}_runs.md"
    summ_md = out_dir / f"phase{phase}_summary.md"
    runs_lines: List[str] = []
    runs_lines.append(f"# Phase {phase} Runs")
    runs_lines.append("")
    runs_lines.append("| Fingerprint | Config | Sharpe | PSR | MaxDD | Vol | AvgTurn | Cost (bps) | Duplicate |")
    runs_lines.append("|-------------|--------|--------|-----|-------|-----|---------|-----------|-----------|")
    best: Optional[RunEval] = None
    for r in rows:
        if best is None or (r.sharpe or float("-inf")) > (best.sharpe or float("-inf")):
            best = r
        runs_lines.append(
            f"| {r.fingerprint} | `{Path(r.config).name}` | "
            f"{(f'{r.sharpe:.2f}' if r.sharpe is not None else '-')} | "
            f"{(f'{r.psr:.2f}' if r.psr is not None else '-')} | "
            f"{_fmt_pct(r.maxdd)} | {(f'{r.vol:.2f}' if r.vol is not None else '-')} | "
            f"{(f'{r.avg_turnover:.4f}' if r.avg_turnover is not None else '-')} | "
            f"{(f'{r.transaction_costs_bps:.1f}' if r.transaction_costs_bps is not None else '-')} | {str(r.duplicate)} |"
        )
    runs_md.write_text("\n".join(runs_lines) + "\n", encoding="utf-8")

    summ_lines: List[str] = []
    summ_lines.append(f"# Phase {phase} Summary")
    summ_lines.append("")
    if best:
        summ_lines.append("Best run:")
        summ_lines.append(
            f"- Fingerprint: {best.fingerprint}"
        )
        summ_lines.append(
            f"- Config: `{Path(best.config).name}`"
        )
        summ_lines.append(
            f"- Sharpe {best.sharpe:.2f} | PSR {(best.psr if best.psr is not None else 0.0):.2f} | "
            f"MaxDD {_fmt_pct(best.maxdd)} | Vol {(best.vol if best.vol is not None else 0.0):.2f}"
        )
        if best.avg_turnover is not None or best.transaction_costs_bps is not None:
            summ_lines.append(
                f"- Avg Turnover {(best.avg_turnover if best.avg_turnover is not None else 0.0):.4f} | "
                f"Costs {(best.transaction_costs_bps if best.transaction_costs_bps is not None else 0.0):.1f} bps"
            )
    else:
        summ_lines.append("No runs classified for this phase.")
    summ_md.write_text("\n".join(summ_lines) + "\n", encoding="utf-8")
    return best, runs_md, summ_md


def _append_final_report(final_report: Path, bests: Dict[int, Optional[RunEval]]) -> None:
    today = date.today().isoformat()
    lines = [
        "",
        "---",
        f"\n## Re-run Updates ({today})\n",
    ]
    for ph in sorted(bests.keys()):
        r = bests[ph]
        if not r:
            continue
        lines.append(
            f"- Phase {ph}: `{Path(r.config).name}` -> Fingerprint {r.fingerprint}; "
            f"Sharpe {r.sharpe:.2f}, PSR {(r.psr if r.psr is not None else 0.0):.2f}, "
            f"MaxDD {_fmt_pct(r.maxdd)}, Vol {(r.vol if r.vol is not None else 0.0):.2f}"
        )
    with final_report.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.update_phases", description="Generate per-phase run summaries")
    ap.add_argument("--matrix-dir", default="reports/matrix", help="Directory with runs.json & eval_report.json")
    ap.add_argument("--out-dir", default="reports/matrix", help="Output directory for phase markdown files")
    ap.add_argument("--final-report", default="reports/matrix/final_report.md", help="Path to append re-run updates")
    args = ap.parse_args(list(argv) if argv is not None else None)

    matrix_dir = Path(args.matrix_dir)
    enforce_turnover_cost_limits(matrix_dir)
    grouped = _collect(matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bests: Dict[int, Optional[RunEval]] = {}
    for phase, rows in sorted(grouped.items()):
        if phase is None or not rows:
            continue
        best, _, _ = _write_phase_files(out_dir, phase, rows)
        bests[int(phase)] = best

    # Append concise updates to final_report
    _append_final_report(Path(args.final_report), bests)


if __name__ == "__main__":  # pragma: no cover
    main()
