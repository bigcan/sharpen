"""Generate synthetic SPY daily and multi-asset daily datasets.

Outputs:
- data/sp500_daily_2016_2025.parquet
- data/sp500_multi_2016_2025/<TICKER>.parquet (for 10 tickers)
Schema: timestamp, ticker, open, high, low, close, volume
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _date_range(start: str, end: str) -> List[datetime]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out: List[datetime] = []
    cur = s
    while cur <= e:
        # weekdays only
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _synth_series(n: int, start_price: float, drift: float, vol: float, seed: int) -> List[float]:
    import random

    rnd = random.Random(seed)
    price = start_price
    series = []
    for _ in range(n):
        ret = rnd.gauss(drift, vol)
        price *= (1.0 + ret)
        series.append(price)
    return series


def _ohlcv_from_close(close: List[float], seed: int) -> tuple[List[float], List[float], List[float], List[int]]:
    import random

    rnd = random.Random(seed)
    opens = []
    highs = []
    lows = []
    vols = []
    prev = close[0]
    for c in close:
        o = prev * (1.0 + rnd.uniform(-0.002, 0.002))
        h = max(c, o) * (1.0 + rnd.uniform(0.0, 0.005))
        l = min(c, o) * (1.0 - rnd.uniform(0.0, 0.005))
        v = int(1_000_000 * (1.0 + rnd.uniform(-0.2, 0.2)))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        vols.append(v)
        prev = c
    return opens, highs, lows, vols


def make_spy_daily(out_path: Path, start: str = "2016-01-01", end: str = "2025-12-31") -> None:
    days = _date_range(start, end)
    close = _synth_series(len(days), start_price=200.0, drift=0.0002, vol=0.01, seed=7)
    o, h, l, v = _ohlcv_from_close(close, seed=11)
    df = pd.DataFrame(
        {
            "timestamp": days,
            "ticker": ["SPY"] * len(days),
            "open": o,
            "high": h,
            "low": l,
            "close": close,
            "volume": v,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def make_multi(out_dir: Path, start: str = "2016-01-01", end: str = "2025-12-31") -> None:
    tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "JPM",
        "XOM",
        "UNH",
        "V",
    ]
    days = _date_range(start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, tic in enumerate(tickers):
        close = _synth_series(len(days), start_price=100.0 + i * 10.0, drift=0.00025, vol=0.012, seed=100 + i)
        o, h, l, v = _ohlcv_from_close(close, seed=200 + i)
        df = pd.DataFrame(
            {
                "timestamp": days,
                "ticker": [tic] * len(days),
                "open": o,
                "high": h,
                "low": l,
                "close": close,
                "volume": v,
            }
        )
        df.to_parquet(out_dir / f"{tic}.parquet", index=False)


if __name__ == "__main__":
    make_spy_daily(Path("data/sp500_daily_2016_2025.parquet"))
    make_multi(Path("data/sp500_multi_2016_2025"))

