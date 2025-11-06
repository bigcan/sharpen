"""Alpaca data loader for FinRL Pro using alpaca-py.

Fetches OHLCV bars via Alpaca Market Data API and returns a standardized
DataFrame with columns: timestamp, ticker, open, high, low, close, volume.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def _tf_from_str(interval: str):
    from alpaca.data.timeframe import TimeFrame

    mapping = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame.FiveMinutes,
        "15m": TimeFrame.FifteenMinutes,
        "30m": TimeFrame.ThirtyMinutes,
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }
    return mapping.get(interval, TimeFrame.Day)


class AlpacaLoader:
    """Minimal Alpaca-backed loader (requires env credentials)."""

    def __init__(self, *, api_key: str, api_secret: str) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, api_secret)

    def fetch(
        self,
        *,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest

        timeframe = _tf_from_str(interval)
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment="raw",
        )
        bars = self._client.get_stock_bars(req)
        df = bars.df  # MultiIndex (symbol, timestamp) or single
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()  # columns: symbol, timestamp, ...
            df.rename(columns={"symbol": "ticker"}, inplace=True)
        else:
            df = df.reset_index()  # index → timestamp
            df.insert(0, "ticker", tickers[0])

        # Standardize column names
        rename = {c: c.lower() for c in df.columns}
        df.rename(columns=rename, inplace=True)
        cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
        present = [c for c in cols if c in df.columns]
        return df[present]

