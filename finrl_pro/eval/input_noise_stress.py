"""Input-noise stress: add Gaussian noise to returns and summarize impact.

Reads `reports/<fingerprint_id>/returns.csv`, generates noisy variants at
multiple noise levels (as a fraction of the empirical std of returns), writes
noisy CSVs, and produces a Markdown summary comparing metrics.
"""

from __future__ import annotations

import argparse
import csv
from math import sqrt
from pathlib import Path
from typing import List, Tuple


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


def _write_returns(path: Path, rets: List[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "return"])
        for i, r in enumerate(rets):
            wr.writerow([i, r])


def _mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: List[float], ddof: int = 1) -> float:
    n = len(x)
    if n <= ddof:
        return 0.0
    m = _mean(x)
    v = sum((v - m) ** 2 for v in x) / (n - ddof)
    return v ** 0.5


def _equity_drawdown(returns: List[float]) -> Tuple[List[float], List[float]]:
    eq: List[float] = []
    dd: List[float] = []
    cum = 1.0
    peak = 1.0
    for r in returns:
        cum *= (1.0 + r)
        eq.append(cum)
        peak = max(peak, cum)
        dd.append((cum / peak) - 1.0)
    return eq, dd


def _sharpe(returns: List[float], periods_per_year: int = 252) -> float:
    s = _std(returns)
    return (_mean(returns) / s) * (periods_per_year ** 0.5) if s else 0.0


def _sortino(returns: List[float], periods_per_year: int = 252) -> float:
    downside = [min(0.0, r) for r in returns if r < 0.0]
    ds = _std(downside) if downside else 0.0
    return (_mean(returns) / ds) * (periods_per_year ** 0.5) if ds else 0.0


def _maxdd(dd: List[float]) -> float:
    return min(dd) if dd else 0.0


def _calmar(equity: List[float], dd: List[float]) -> float:
    if not equity:
        return 0.0
    total_return = (equity[-1] / equity[0]) - 1.0 if equity[0] != 0 else 0.0
    years = max(1.0, len(equity) / 252.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    m = abs(_maxdd(dd))
    return (cagr / m) if m > 0 else 0.0


def _metrics(returns: List[float]) -> dict:
    eq, dd = _equity_drawdown(returns)
    return {
        "sharpe": _sharpe(returns),
        "sortino": _sortino(returns),
        "max_drawdown": _maxdd(dd),
        "calmar": _calmar(eq, dd),
        "n": len(returns),
    }


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.input_noise_stress", description="Gaussian input-noise stress")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id to stress")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    ap.add_argument("--levels", default="0.25,0.50", help="Noise sigma as fractions of empirical std, comma-separated")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    base = rdir / "returns.csv"
    if not base.exists():
        raise FileNotFoundError(f"returns.csv not found under {rdir}")
    r = _read_returns(base)
    import random

    sigma = _std(r, ddof=1)
    levels = [float(x) for x in str(args.levels).split(",") if x]

    lines: List[str] = []
    lines.append("# Input Noise Stress (Gaussian)")
    lines.append("")
    lines.append("| Series | N | Sharpe | Sortino | MaxDD | Calmar |")
    lines.append("|--------|---|--------|---------|-------|--------|")
    def fmt(m: dict) -> str:
        return f"{int(m['n'])} | {m['sharpe']:.2f} | {m['sortino']:.2f} | {abs(m['max_drawdown']):.1%} | {m['calmar']:.2f}"

    # Base
    m_base = _metrics(r)
    lines.append(f"| base | {fmt(m_base)} |")

    rnd = random.Random(int(args.seed))
    for lvl in levels:
        s = float(lvl) * sigma
        noise = [rnd.gauss(0.0, s) for _ in r]
        r_noisy = [a + b for a, b in zip(r, noise)]
        _write_returns(rdir / f"returns_noise_{lvl:.2f}.csv", r_noisy)
        m = _metrics(r_noisy)
        lines.append(f"| noise_sigma={lvl:.2f}×std | {fmt(m)} |")

    (rdir / "input_noise_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()

