"""Runs INSIDE FinRL-X's own venv (C:/FinRL/FinRL-Trading/.venv), with cwd = the clone. No sharpen imports.

Reads prices.csv and weights.csv (wide: date index x ticker columns; weights already lagged by the caller so
row t is what trades at close t), runs FinRL-X's unmodified ``BacktestEngine.run_backtest`` and writes
nav.csv. ``benchmark_tickers=[]`` keeps FinRL-X from fetching benchmarks through FMP.

usage: <clone>/.venv/Scripts/python.exe finrlx_bt_runner.py <io_dir> <transaction_cost> <name>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    io_dir, tc, name = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    clone = Path.cwd()
    sys.path.insert(0, str(clone))
    sys.path.insert(0, str(clone / "src"))
    from src.backtest.backtest_engine import BacktestConfig, BacktestEngine  # noqa: PLC0415

    prices = pd.read_csv(io_dir / "prices.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(io_dir / "weights.csv", index_col=0, parse_dates=True)
    cfg = BacktestConfig(start_date=str(prices.index.min().date()), end_date=str(prices.index.max().date()),
                         transaction_cost=tc, benchmark_tickers=[], integer_positions=False,
                         initial_capital=1_000_000.0)
    res = BacktestEngine(cfg).run_backtest(name, prices, weights)
    res.portfolio_values.rename("nav").to_csv(io_dir / "nav.csv")
    (io_dir / "finrlx_metrics.json").write_text(json.dumps({k: float(v) for k, v in res.metrics.items()
                                                            if isinstance(v, (int, float))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
