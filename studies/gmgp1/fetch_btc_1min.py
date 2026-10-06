"""Fetch Bybit BTCUSDT (linear) 1-minute candles from Bybit's public API into
data/btc_usdt_1min_bybit.parquet, then verify them against the checksums the
study was run on.

    python studies/gmgp1/fetch_btc_1min.py --start 2024-01-01 --end 2026-05-01 --out data/btc_usdt_1min_bybit.parquet

Public endpoint, no API key. Restartable: monthly pieces are cached under --cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=1&limit=1000&start={s}&end={e}"
MIN_MS = 60_000


def get(s: int, e: int, tries: int = 6) -> list:
    for k in range(tries):
        try:
            with urllib.request.urlopen(URL.format(s=s, e=e), timeout=30) as r:
                j = json.loads(r.read())
            if j.get("retCode") == 0:
                return j["result"]["list"]
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"Bybit request failed for {s}-{e}")


def fetch_range(t0: int, t1: int) -> pd.DataFrame:
    rows, s = [], t0
    while s < t1:
        e = min(s + 999 * MIN_MS, t1 - MIN_MS)
        rows += get(s, e)
        s = e + MIN_MS
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype("float64")
    return df.drop(columns="turnover").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def checksum(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(df[["open", "high", "low", "close", "volume"]].round(6).to_numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-05-01")
    ap.add_argument("--out", type=Path, default=Path("data/btc_usdt_1min_bybit.parquet"))
    ap.add_argument("--cache", type=Path, default=Path("data/_bybit_cache"))
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(a.start, a.end, freq="MS", tz="UTC")
    parts = []
    for m0, m1 in zip(months[:-1], months[1:]):
        f = a.cache / f"{m0:%Y-%m}.parquet"
        if not f.exists():
            df = fetch_range(int(m0.timestamp() * 1000), int(m1.timestamp() * 1000))
            df.to_parquet(f)
            print("fetched", f.name, len(df), flush=True)
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print("rows", len(df), "first", df.timestamp.iloc[0], "last", df.timestamp.iloc[-1], "sha256", checksum(df))


if __name__ == "__main__":
    main()
