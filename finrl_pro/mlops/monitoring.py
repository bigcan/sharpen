"""Basic monitoring hooks: drift stats and alerts from returns CSV.

Computes simple population stats (mean, std), approximate PSI via binning,
and emits a Markdown/JSON summary under reports/<fingerprint_id>/monitoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def _read_returns(path: Path) -> List[float]:
    vals: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                vals.append(float(row.get("return", "")))
            except Exception:
                continue
    return vals


def _mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: List[float], ddof: int = 1) -> float:
    n = len(x)
    if n <= ddof:
        return 0.0
    m = _mean(x)
    v = sum((v - m) ** 2 for v in x) / (n - ddof)
    return v ** 0.5


def _psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    if not expected or not actual:
        return 0.0
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    step = (hi - lo) / max(1, bins)
    if step == 0.0:
        return 0.0
    # bin counts
    def hist(data: List[float]) -> List[int]:
        h = [0] * bins
        for v in data:
            idx = int((v - lo) / step)
            if idx >= bins:
                idx = bins - 1
            if idx < 0:
                idx = 0
            h[idx] += 1
        return h

    he = hist(expected)
    ha = hist(actual)
    n_e = sum(he)
    n_a = sum(ha)
    psi = 0.0
    for i in range(bins):
        pe = he[i] / n_e if n_e else 0.0
        pa = ha[i] / n_a if n_a else 0.0
        if pe > 0 and pa > 0:
            psi += (pa - pe) * (0 if pe == 0 else (pa / pe))
    return float(psi)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.mlops.monitoring", description="Compute monitoring stats for returns")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    ap.add_argument("--window", type=int, default=252, help="Window size for recent vs baseline split")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    rcsv = rdir / "returns.csv"
    if not rcsv.exists():
        raise FileNotFoundError(f"returns.csv not found under {rdir}")
    r = _read_returns(rcsv)
    if not r:
        raise RuntimeError("empty returns series")
    w = int(args.window)
    baseline = r[:-w] if len(r) > w else r
    recent = r[-w:] if len(r) > w else r
    stats = {
        "baseline_mean": _mean(baseline),
        "baseline_std": _std(baseline),
        "recent_mean": _mean(recent),
        "recent_std": _std(recent),
        "psi": _psi(baseline, recent),
        "window": w,
        "n_total": len(r),
        "n_baseline": len(baseline),
        "n_recent": len(recent),
    }
    out_dir = rdir / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Monitoring Stats",
        "",
        f"Window: {w}",
        "",
        f"- baseline_mean: {stats['baseline_mean']:.6f}",
        f"- baseline_std: {stats['baseline_std']:.6f}",
        f"- recent_mean: {stats['recent_mean']:.6f}",
        f"- recent_std: {stats['recent_std']:.6f}",
        f"- psi: {stats['psi']:.4f}",
    ]
    (out_dir / "monitor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
