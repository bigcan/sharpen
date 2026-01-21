"""Regime analysis on returns.csv for a fingerprint.

Labels days into bull/bear/sideways using a simple rolling mean filter and
computes per-regime Sharpe and hit rate. Outputs markdown and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_returns(path: Path) -> List[float]:
    rets: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rets.append(float(row.get("return", "")))
            except Exception:
                continue
    return rets


def _rolling_mean(x: List[float], window: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(x):
        s += v
        if i >= window:
            s -= x[i - window]
        if i + 1 >= window:
            out.append(s / window)
        else:
            out.append(0.0)
    return out


def _label_regimes(r: List[float], window: int = 63, thr: float = 0.0) -> List[str]:
    m = _rolling_mean(r, window)
    labels: List[str] = []
    for v in m:
        if v > thr:
            labels.append("bull")
        elif v < -thr:
            labels.append("bear")
        else:
            labels.append("sideways")
    return labels


def _sharpe(r: List[float], periods_per_year: int = 252) -> float:
    n = len(r)
    if n == 0:
        return 0.0
    mu = sum(r) / n
    m = mu
    var = sum((v - m) ** 2 for v in r) / n if n > 0 else 0.0
    sd = var ** 0.5
    return (mu / sd) * (periods_per_year ** 0.5) if sd > 0 else 0.0


def _hit_rate(r: List[float]) -> float:
    n = len(r)
    if n == 0:
        return 0.0
    return sum(1 for v in r if v > 0) / n


def _per_regime_metrics(r: List[float], labels: List[str]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {"bull": [], "bear": [], "sideways": []}
    for v, lab in zip(r, labels):
        buckets.setdefault(lab, []).append(v)
    out: Dict[str, Dict[str, float]] = {}
    for lab, seq in buckets.items():
        out[lab] = {
            "count": float(len(seq)),
            "sharpe": float(_sharpe(seq)),
            "hit_rate": float(_hit_rate(seq)),
        }
    return out


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro_ds.eval.regime_analysis", description="Per-regime metrics from returns.csv")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id to analyze")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    ap.add_argument("--window", type=int, default=63, help="Rolling window for regime filter")
    ap.add_argument("--threshold", type=float, default=0.0, help="Mean return threshold to separate bull/bear from sideways")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    rcsv = rdir / "returns.csv"
    if not rcsv.exists():
        raise FileNotFoundError(f"returns.csv not found under {rdir}")
    rets = _read_returns(rcsv)
    labels = _label_regimes(rets, window=int(args.window), thr=float(args.threshold))
    metrics = _per_regime_metrics(rets, labels)

    # Persist JSON and Markdown
    out_json = rdir / "regime_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Regime Analysis")
    lines.append("")
    lines.append(f"Window: {int(args.window)} | Threshold: {float(args.threshold)}")
    lines.append("")
    lines.append("| Regime | Count | Sharpe | Hit Rate |")
    lines.append("|--------|-------|--------|----------|")
    for lab in ("bull", "bear", "sideways"):
        m = metrics.get(lab, {})
        lines.append(
            f"| {lab} | {int(m.get('count', 0.0))} | {m.get('sharpe', 0.0):.2f} | {m.get('hit_rate', 0.0):.2f} |"
        )
    (rdir / "regime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
