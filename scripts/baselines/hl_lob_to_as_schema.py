"""Convert Hyperliquid 1s LOB parts + trades parts into the schema expected by
`avellaneda_stoikov_mm.py`.

A-S baseline expects columns:
    timestamp (datetime, UTC), open, high, low, close, volume,
    bbo_bid_qty, bbo_ask_qty, spread_mean

This adapter aggregates 1s LOB snapshots into OHLCV bars on `mid_price`, sums
trade volume by bar, and takes BBO sizes at bar end.

Usage:
    python scripts/baselines/hl_lob_to_as_schema.py \\
        --lob-glob 'data/hyperliquid/btc/lob_*_part_*.parquet' \\
        --trades-glob 'data/hyperliquid/btc/trades_*_part_*.parquet' \\
        --bar-seconds 10 \\
        --out data/processed/btc_hl_lob_10s.parquet
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def load_concat(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def aggregate(
    lob: pd.DataFrame,
    trades: pd.DataFrame | None,
    bar_seconds: int,
) -> pd.DataFrame:
    lob = lob.sort_values("timestamp_ms").reset_index(drop=True)
    lob["ts"] = pd.to_datetime(lob["timestamp_ms"], unit="ms", utc=True)
    lob = lob.set_index("ts")

    rule = f"{bar_seconds}s"
    ohlc = lob["mid_price"].resample(rule).ohlc()
    bbo_bid_qty = lob["bid_size_1"].resample(rule).last()
    bbo_ask_qty = lob["ask_size_1"].resample(rule).last()
    spread_mean = lob["spread"].resample(rule).mean()

    bars = pd.concat(
        {
            "open": ohlc["open"],
            "high": ohlc["high"],
            "low": ohlc["low"],
            "close": ohlc["close"],
            "bbo_bid_qty": bbo_bid_qty,
            "bbo_ask_qty": bbo_ask_qty,
            "spread_mean": spread_mean,
        },
        axis=1,
    )

    if trades is not None and not trades.empty:
        trades = trades.copy()
        trades["ts"] = pd.to_datetime(trades["timestamp_ms"], unit="ms", utc=True)
        trades = trades.set_index("ts")
        volume = trades["sz"].resample(rule).sum()
    else:
        volume = pd.Series(0.0, index=bars.index, name="volume")
    bars["volume"] = volume.reindex(bars.index).fillna(0.0)

    # Drop bars with no LOB activity (will have NaN OHLC).
    bars = bars.dropna(subset=["open", "high", "low", "close"]).copy()
    bars.index.name = "ts"
    bars = bars.reset_index().rename(columns={"ts": "timestamp"})
    bars["timestamp"] = bars["timestamp"].astype("datetime64[ns, UTC]")
    return bars[
        ["timestamp", "open", "high", "low", "close", "volume",
         "bbo_bid_qty", "bbo_ask_qty", "spread_mean"]
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lob-glob", required=True, help="Glob for LOB parquet parts")
    p.add_argument("--trades-glob", default=None, help="Glob for trades parquet parts (optional)")
    p.add_argument("--bar-seconds", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    lob = load_concat(args.lob_glob)
    trades = load_concat(args.trades_glob) if args.trades_glob else None
    print(f"[adapt] lob rows={len(lob):,} "
          f"trades rows={len(trades) if trades is not None else 0:,}")

    bars = aggregate(lob, trades, args.bar_seconds)
    print(f"[adapt] bars={len(bars):,} "
          f"from {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.out, index=False, engine="pyarrow")
    print(f"[adapt] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
