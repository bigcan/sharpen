"""CLI: plot equity curve and drawdown from per-fingerprint CSV artifacts.

Reads `reports/<fingerprint_id>/equity_curve.csv` and `drawdown.csv`, then
produces `equity_curve.png` and `drawdown.png` in the same directory.
Silently no-ops if matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple


def _read_series(path: Path, value_col: str) -> Tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                xs.append(int(row.get("t", len(xs))))
                ys.append(float(row[value_col]))
            except (TypeError, ValueError, KeyError):
                continue
    return xs, ys


def _plot_png(x: list[int], y: list[float], out_path: Path, *, title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y, linewidth=1.2)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro.eval.plot_artifacts", description="Plot equity and drawdown images")
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id whose artifacts to plot")
    ap.add_argument("--reports-dir", default="reports", help="Reports directory (default: reports)")
    args = ap.parse_args(argv or None)

    rdir = Path(args.reports_dir) / args.fingerprint
    eq_csv = rdir / "equity_curve.csv"
    dd_csv = rdir / "drawdown.csv"
    if not eq_csv.exists() or not dd_csv.exists():
        return
    x_eq, y_eq = _read_series(eq_csv, "equity")
    x_dd, y_dd = _read_series(dd_csv, "drawdown")
    _plot_png(x_eq, y_eq, rdir / "equity_curve.png", title="Equity Curve", ylabel="Equity")
    _plot_png(x_dd, y_dd, rdir / "drawdown.png", title="Drawdown", ylabel="Drawdown")


if __name__ == "__main__":  # pragma: no cover
    main()

