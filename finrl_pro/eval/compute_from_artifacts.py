"""Compute evaluation metrics from emitted artifacts for a fingerprint.

Reads returns/equity/drawdown CSVs under reports/<fingerprint_id>/ and emits
JSON with Sharpe, Sortino, MaxDD, and Calmar estimates. Intended for quick
leaderboard population when evaluator outputs are placeholders.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Tuple


def _read_series(path: Path, value_col: str) -> list[float]:
    out: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                out.append(float(row[value_col]))
            except Exception:
                continue
    return out


def mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def std(x: list[float], ddof: int = 1) -> float:
    n = len(x)
    if n <= ddof:
        return 0.0
    m = mean(x)
    v = sum((v - m) ** 2 for v in x) / (n - ddof)
    return v ** 0.5


def sharpe(returns: list[float], periods_per_year: int = 252) -> float:
    s = std(returns)
    return (mean(returns) / s) * sqrt(periods_per_year) if s else 0.0


def sortino(returns: list[float], periods_per_year: int = 252) -> float:
    downside = [min(0.0, r) for r in returns if r < 0.0]
    ds = std(downside, ddof=1) if downside else 0.0
    return (mean(returns) / ds) * sqrt(periods_per_year) if ds else 0.0


def max_drawdown(dd: list[float]) -> float:
    return min(dd) if dd else 0.0


def calmar(equity: list[float], dd: list[float]) -> float:
    if not equity:
        return 0.0
    total_return = (equity[-1] / equity[0]) - 1.0 if equity[0] != 0 else 0.0
    years = max(1.0, len(equity) / 252.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(dd))
    return (cagr / mdd) if mdd > 0 else 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.compute_from_artifacts")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory")
    ap.add_argument("--out-json", default=None, help="Optional output JSON path")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    returns = _read_series(rdir / "returns.csv", "return")
    equity = _read_series(rdir / "equity_curve.csv", "equity")
    dd = _read_series(rdir / "drawdown.csv", "drawdown")

    metrics = {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(dd),
        "calmar": calmar(equity, dd),
    }

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

