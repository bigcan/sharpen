"""Simple buy/hold backtest to produce artifact-based returns for verification.

Writes `reports/<fingerprint_id>/returns.csv` using adjusted close prices
downloaded via yfinance. Designed for quick Gate 0.0 metric verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("yfinance is required for simple_bh_backtest") from e
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} {start}..{end}")
    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        raise RuntimeError("Expected 'Adj Close' or 'Close' in downloaded data")
    series_or_df = df[col]
    if isinstance(series_or_df, pd.DataFrame):
        s = series_or_df.iloc[:, 0].astype(float)
        s.name = "adj_close"
    else:
        s = series_or_df.astype(float)
        s.name = "adj_close"
    s.index = pd.to_datetime(s.index)
    return s


def _returns_from_prices(prices: pd.Series) -> pd.Series:
    rets = prices.pct_change().fillna(0.0).astype(float)
    rets.name = "return"
    return rets


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="finrl_pro.eval.simple_bh_backtest", description="Buy/Hold returns to artifacts"
    )
    ap.add_argument("--fingerprint", required=True, help="Fingerprint id to write under reports/<fp>")
    ap.add_argument("--ticker", default="SPY", help="Ticker (default: SPY)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = ap.parse_args(list(argv) if argv is not None else None)

    s = _fetch_prices(args.ticker, args.start, args.end)
    r = _returns_from_prices(s)

    out_dir = Path("reports") / str(args.fingerprint)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "returns.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("t,return\n")
        for i, v in enumerate(r.values.tolist()):
            f.write(f"{i},{v}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
