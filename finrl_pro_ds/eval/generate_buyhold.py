"""Generate Buy&Hold baseline returns and metrics for a ticker and date range.

Fetches OHLCV via YahooLoader, computes daily close-to-close returns,
emits `reports/baselines/<ticker>_<start>_<end>/returns.csv` and metrics JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import pandas as pd

from finrl_pro_ds.data.yahoo_loader import YahooLoader
from finrl_pro_ds.eval.statistics import (
    SharpeCI,
    bootstrap_sharpe_ci,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def _compute_returns(df: pd.DataFrame) -> List[float]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    rets = [float(x) for x in df["ret"].tolist() if pd.notna(x)]
    return rets


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrl_pro_ds.eval.generate_buyhold")
    ap.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    ap.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD (default: 2023-01-01)")
    ap.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD (default: 2025-12-31)")
    ap.add_argument("--out-dir", default="reports/baselines", help="Output directory base")
    args = ap.parse_args(argv or None)

    loader = YahooLoader()
    df = loader.fetch(tickers=[args.ticker], start=args.start, end=args.end, interval="1d")
    if df.empty:
        raise RuntimeError("No data returned from Yahoo for given parameters.")

    rets = _compute_returns(df)
    # Write returns CSV
    label = f"{args.ticker}_{args.start}_{args.end}"
    out_dir = Path(args.out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "returns.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "return"])
        for i, r in enumerate(rets):
            wr.writerow([i, r])

    # Metrics
    sr = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    psr = probabilistic_sharpe_ratio(rets)
    ci: SharpeCI = bootstrap_sharpe_ci(rets)
    metrics = {
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "sharpe_ratio": float(sr),
        "sortino_ratio": float(so),
        "psr": float(psr),
        "sharpe_ci_lower": float(ci.lower),
        "sharpe_ci_upper": float(ci.upper),
        "n": len(rets),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"returns_csv": str(out_dir / "returns.csv"), "metrics": str(out_dir / "metrics.json")}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

