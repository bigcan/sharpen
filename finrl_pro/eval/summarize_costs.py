"""Summarize cost sensitivity from matrix artifacts.

Reads runs.json and eval_report.json under reports/matrix, attempts to match
base vs stress configs by filename, and writes a simple Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_stress_config(path: str) -> bool:
    name = Path(path).name.lower()
    return "stress" in name or "cost" in name


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.summarize_costs")
    ap.add_argument("--matrix-dir", default="reports/matrix", help="Directory with runs.json & eval_report.json")
    ap.add_argument("--out", default="reports/matrix/cost_sensitivity.md", help="Output Markdown path")
    args = ap.parse_args(argv or None)

    mdir = Path(args.matrix_dir)
    runs: List[Dict[str, str]] = _load_json(mdir / "runs.json")
    evals: List[Dict[str, Any]] = _load_json(mdir / "eval_report.json")

    # Map fingerprint -> metrics
    metrics_by_fp = {e["fingerprint_id"]: e["evaluated_metrics"] for e in evals}

    rows: List[str] = []
    rows.append("# Cost Sensitivity Summary")
    rows.append("")
    rows.append("| Config | Fingerprint | Sharpe | MaxDD | Volatility | Note |")
    rows.append("|--------|-------------|--------|-------|------------|------|")

    for r in runs:
        cfg = r.get("config", "")
        fp = r.get("fingerprint_id", "")
        met = metrics_by_fp.get(fp, {})
        sharpe = met.get("sharpe_ratio")
        mdd = met.get("max_drawdown")
        vol = met.get("volatility")
        note = "stress" if _is_stress_config(cfg) else "base"
        rows.append(
            f"| `{Path(cfg).name}` | `{fp}` | {sharpe if sharpe is not None else '—'} | {mdd if mdd is not None else '—'} | {vol if vol is not None else '—'} | {note} |"
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()

