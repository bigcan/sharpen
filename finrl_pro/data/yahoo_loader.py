"""Yahoo Finance data loader for FinRL Pro.

Fetches OHLCV bars via yfinance and returns a standardized DataFrame
with columns: timestamp, ticker, open, high, low, close, volume.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for k in ["open", "high", "low", "close", "volume"]:
        # find matching column ignoring case
        for c in df.columns:
            if c.lower() == k:
                rename[c] = k
                break
    out = df.rename(columns=rename)
    out = out[[c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]]
    out["timestamp"] = out.index
    return out


class YahooLoader:
    """Minimal yfinance-backed loader."""

    def fetch(
        self,
        *,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        frames: list[pd.DataFrame] = []
        # If multiple tickers, columns are MultiIndex: (ticker, field)
        if isinstance(data.columns, pd.MultiIndex):
            for tic in tickers:
                if tic not in data.columns.get_level_values(0):
                    continue
                df = data[tic].copy()
                df = _standardize(df)
                df["ticker"] = tic
                frames.append(df)
        else:
            df = _standardize(data)
            df["ticker"] = tickers[0]
            frames.append(df)

        combined = pd.concat(frames, axis=0, ignore_index=False)
        combined = combined.reset_index(drop=True)
        # Ensure column order
        cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
        cols_present = [c for c in cols if c in combined.columns]
        return combined[cols_present]

