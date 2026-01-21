"""Basic monitoring hooks: drift stats and alerts from returns CSV.

Computes simple population stats (mean, std), approximate PSI via binning,
and emits a Markdown/JSON summary under reports/<fingerprint_id>/monitoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List

from finrl_pro_ds.mlops.alerting import RiskAlertDispatcher


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


def _read_csv_column(path: Path, column: str) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                val = row.get(column, "")
                if val:
                    vals.append(float(val))
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
    epsilon = 1e-6
    for i in range(bins):
        pe = he[i] / n_e if n_e else 0.0
        pa = ha[i] / n_a if n_a else 0.0
        
        # Apply epsilon to avoid zero division or log(0)
        pe = max(pe, epsilon)
        pa = max(pa, epsilon)
        
        psi += (pa - pe) * math.log(pa / pe)
    return float(psi)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro_ds.mlops.monitoring", description="Compute monitoring stats for returns")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    ap.add_argument("--window", type=int, default=252, help="Window size for recent vs baseline split")
    ap.add_argument("--max-drawdown", type=float, default=0.20, help="Max drawdown threshold (positive float, e.g. 0.20 for 20%)")
    ap.add_argument("--max-turnover", type=float, default=0.50, help="Max daily turnover threshold")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    rcsv = rdir / "returns.csv"
    if not rcsv.exists():
        raise FileNotFoundError(f"returns.csv not found under {rdir}")
    
    r = _read_returns(rcsv)
    dd_vals = _read_csv_column(rdir / "drawdown.csv", "drawdown")
    to_vals = _read_csv_column(rdir / "execution.csv", "turnover")

    if not r:
        raise RuntimeError("empty returns series")
    
    w = int(args.window)
    baseline = r[:-w] if len(r) > w else r
    recent = r[-w:] if len(r) > w else r
    
    # Calculate recent drawdown/turnover stats
    recent_dd = dd_vals[-w:] if len(dd_vals) > w else dd_vals
    recent_to = to_vals[-w:] if len(to_vals) > w else to_vals
    
    min_recent_dd = min(recent_dd) if recent_dd else 0.0
    max_recent_to = max(recent_to) if recent_to else 0.0
    mean_recent_to = _mean(recent_to)

    dispatcher = RiskAlertDispatcher()
    
    # Alerts
    # Drawdown is typically negative in CSV (e.g. -0.05). Threshold is positive (0.20).
    # So check if min_recent_dd < -0.20
    if min_recent_dd < -args.max_drawdown:
        dispatcher.emit(
            level="CRITICAL", 
            message="Drawdown limit breached", 
            details={"current": f"{min_recent_dd:.4f}", "limit": f"{-args.max_drawdown}"}
        )
        
    if max_recent_to > args.max_turnover:
        dispatcher.emit(
            level="WARNING", 
            message="Turnover spike detected", 
            details={"current": f"{max_recent_to:.4f}", "limit": f"{args.max_turnover}"}
        )

    stats = {
        "baseline_mean": _mean(baseline),
        "baseline_std": _std(baseline),
        "recent_mean": _mean(recent),
        "recent_std": _std(recent),
        "recent_min_drawdown": min_recent_dd,
        "recent_max_turnover": max_recent_to,
        "recent_mean_turnover": mean_recent_to,
        "psi": _psi(baseline, recent),
        "window": w,
        "n_total": len(r),
        "n_baseline": len(baseline),
        "n_recent": len(recent),
        "alerts": list(dispatcher.history()),
    }
    
    out_dir = rdir / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    
    lines = [
        "# Monitoring Stats",
        "",
        f"Window: {w}",
        "",
        "## Drift Metrics",
        f"- baseline_mean: {stats['baseline_mean']:.6f}",
        f"- baseline_std: {stats['baseline_std']:.6f}",
        f"- recent_mean: {stats['recent_mean']:.6f}",
        f"- recent_std: {stats['recent_std']:.6f}",
        f"- psi: {stats['psi']:.4f}",
        "",
        "## Risk Metrics (Recent)",
        f"- Min Drawdown: {min_recent_dd:.4f}",
        f"- Max Turnover: {max_recent_to:.4f}",
        f"- Mean Turnover: {mean_recent_to:.4f}",
        "",
        "## Alerts",
    ]
    
    if stats["alerts"]:
        for alert in stats["alerts"]:
            lines.append(f"- **[{alert['level']}]** {alert['message']}: {alert.get('details', '')}")
    else:
        lines.append("- No alerts.")
        
    (out_dir / "monitor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
