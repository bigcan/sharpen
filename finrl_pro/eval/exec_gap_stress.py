"""Execution gap stress: evaluate impact of 1–2 tick delay on realized returns.

Reads `reports/<fingerprint_id>/returns.csv`, constructs stressed return series
with 1- and 2-step execution delays, writes stressed CSVs, and emits a Markdown
summary comparing Sharpe/Sortino/MaxDD/Calmar across original and stressed.
"""

from __future__ import annotations

import argparse
import csv
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


def _mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: List[float], ddof: int = 1) -> float:
    n = len(x)
    if n <= ddof:
        return 0.0
    m = _mean(x)
    v = sum((v - m) ** 2 for v in x) / (n - ddof)
    return v ** 0.5


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
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.exec_gap_stress", description="Execution gap stress testing")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id to stress")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    base = rdir / "returns.csv"
    if not base.exists():
        raise FileNotFoundError(f"returns.csv not found under {rdir}")
    r = _read_returns(base)
    # Apply 1- and 2-step execution gaps: realized returns are shifted later
    gap1 = r[1:] + [0.0]
    gap2 = r[2:] + [0.0, 0.0]
    _write_returns(rdir / "returns_execution_gap_1.csv", gap1)
    _write_returns(rdir / "returns_execution_gap_2.csv", gap2)

    m_base = _metrics(r)
    m_g1 = _metrics(gap1)
    m_g2 = _metrics(gap2)

    # Markdown summary
    lines: List[str] = []
    lines.append("# Execution Gap Stress")
    lines.append("")
    lines.append("| Series | N | Sharpe | Sortino | MaxDD | Calmar |")
    lines.append("|--------|---|--------|---------|-------|--------|")
    def fmt(m: dict) -> str:
        return f"{int(m['n'])} | {m['sharpe']:.2f} | {m['sortino']:.2f} | {abs(m['max_drawdown']):.1%} | {m['calmar']:.2f}"
    lines.append(f"| base | {fmt(m_base)} |")
    lines.append(f"| gap+1 | {fmt(m_g1)} |")
    lines.append(f"| gap+2 | {fmt(m_g2)} |")
    (rdir / "execution_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()

