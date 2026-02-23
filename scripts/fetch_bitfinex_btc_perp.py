"""
Fetch BTC/USDT Perpetual OHLCV-1m from Bitfinex Public API.
=============================================================

No authentication required. 10,000 candles per request.
Full year 2025 ≈ 525,600 1-min bars = ~53 requests.

Ticker: tBTCF0:USTF0  (Bitfinex BTC perpetual settled in USDt)

Fee structure (since Dec 17, 2025):
  - Maker: 0.00% (zero)
  - Taker: 0.00% (zero)
  - Only cost: funding rate (8h settlement, avoidable by flattening)
  - Effective spread: TBD (need LOB analysis)

Output: data/bitfinex/btc_usdt_perp_2025_1min.parquet

Usage:
  python scripts/fetch_bitfinex_btc_perp.py
  python scripts/fetch_bitfinex_btc_perp.py --start 2024-01-01 --end 2024-12-31
  python scripts/fetch_bitfinex_btc_perp.py --also-funding  # Include funding rate history
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api-pub.bitfinex.com/v2"
SYMBOL = "tBTCF0:USTF0"  # BTC perpetual (USDt-settled)
DATA_DIR = Path(__file__).parent.parent / "data" / "bitfinex"


def fetch_candles(symbol: str, timeframe: str, start_ms: int, limit: int = 10000) -> list:
    """Fetch OHLCV candles from Bitfinex public API (no auth needed).

    Returns list of [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME].
    Note: Bitfinex returns [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME] — not standard OHLCV order.
    """
    url = f"{BASE_URL}/candles/trade:{timeframe}:{symbol}/hist"
    params = {"limit": limit, "sort": 1, "start": start_ms}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in str(data):
        raise RuntimeError(f"Bitfinex API error: {data}")
    return data


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical derivatives status (funding rate, mark price, OI).

    Returns DataFrame with funding rate columns.
    """
    url = f"{BASE_URL}/status/deriv/{symbol}/hist"
    all_rows = []
    cursor = start_ms

    print("[FUNDING] Fetching funding rate history...")
    while cursor < end_ms:
        params = {"limit": 5000, "sort": 1, "start": cursor}
        if end_ms:
            params["end"] = end_ms
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data or len(data) == 0:
            break

        all_rows.extend(data)
        cursor = data[-1][0] + 1  # mts is first field
        print(f"  Fetched {len(all_rows)} funding records, latest: {pd.Timestamp(cursor, unit='ms')}")
        time.sleep(1.5)

    if not all_rows:
        print("[FUNDING] No funding data returned")
        return pd.DataFrame()

    # Derivatives status fields (from Bitfinex API docs):
    # [MTS, _, DERIV_PRICE, SPOT_PRICE, _, INSURANCE_FUND_BALANCE,
    #  _, NEXT_FUNDING_EVT_TIMESTAMP_MS, NEXT_FUNDING_ACCRUED,
    #  _, _, _, _, CURRENT_FUNDING, _, MARK_PRICE, _, _,
    #  OPEN_INTEREST, _, _, _, _, CLAMP_MIN, CLAMP_MAX, ...]
    cols = ['mts', '_1', 'deriv_price', 'spot_price', '_4',
            'insurance_fund', '_6', 'next_funding_ts', 'next_funding_accrued',
            '_9', '_10', '_11', '_12', 'current_funding', '_14',
            'mark_price', '_16', '_17', 'open_interest']

    # Truncate columns to match actual data width
    n_cols = min(len(cols), min(len(row) for row in all_rows))
    df = pd.DataFrame([row[:n_cols] for row in all_rows], columns=cols[:n_cols])
    df['timestamp'] = pd.to_datetime(df['mts'], unit='ms')

    # Keep useful columns
    keep = ['timestamp', 'deriv_price', 'spot_price', 'current_funding',
            'next_funding_accrued', 'mark_price', 'open_interest']
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Bitfinex BTC perp OHLCV-1m")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=SYMBOL, help="Bitfinex symbol")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe")
    parser.add_argument("--also-funding", action="store_true",
                        help="Also fetch funding rate history")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Sleep between requests (seconds)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end + " 23:59:59").timestamp() * 1000)

    print(f"[BITFINEX] Fetching {args.symbol} {args.timeframe} candles")
    print(f"[BITFINEX] Range: {args.start} → {args.end}")
    print(f"[BITFINEX] Expected: ~{(end_ms - start_ms) / 60000:,.0f} 1-min bars")
    print()

    all_candles = []
    cursor = start_ms
    request_count = 0

    while cursor < end_ms:
        try:
            batch = fetch_candles(args.symbol, args.timeframe, cursor)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  Rate limited! Waiting 60s...")
                time.sleep(60)
                continue
            raise
        except requests.exceptions.RequestException as e:
            print(f"  Connection error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        if not batch or len(batch) == 0:
            break

        all_candles.extend(batch)
        request_count += 1
        latest_ts = pd.Timestamp(batch[-1][0], unit='ms')
        cursor = batch[-1][0] + 60000  # +1 minute

        print(f"  Request {request_count}: +{len(batch)} candles, "
              f"total={len(all_candles):,}, latest={latest_ts}")

        time.sleep(args.sleep)

    if not all_candles:
        print("[BITFINEX] ERROR: No data returned!")
        sys.exit(1)

    # Build DataFrame
    # Bitfinex format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
    df = pd.DataFrame(all_candles, columns=['mts', 'open', 'close', 'high', 'low', 'volume'])
    df['timestamp'] = pd.to_datetime(df['mts'], unit='ms')

    # Reorder to standard OHLCV
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Dedup (overlapping requests may produce duplicates)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # Remove any data outside requested range
    df = df[(df['timestamp'] >= args.start) & (df['timestamp'] <= args.end + " 23:59:59")]

    # Basic validation
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0].reset_index(drop=True)

    # Compute mid_price for consistency with our pipeline
    df['mid_price'] = (df['open'] + df['close']) / 2

    print(f"\n[BITFINEX] Fetch complete:")
    print(f"  Total candles: {len(df):,}")
    print(f"  Requests made: {request_count}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Trading days: {df['timestamp'].dt.date.nunique()}")
    print(f"  Avg bars/day: {len(df) / max(df['timestamp'].dt.date.nunique(), 1):.0f}")
    print(f"  Price range: ${df['close'].min():,.2f} — ${df['close'].max():,.2f}")
    print(f"  Avg price: ${df['close'].mean():,.2f}")
    print(f"  Avg volume/bar: {df['volume'].mean():.2f} BTC")

    # Fee analysis
    avg_price = df['close'].mean()
    print(f"\n  Fee Analysis (Bitfinex zero-fee model):")
    print(f"    Maker fee: 0.00 bps")
    print(f"    Taker fee: 0.00 bps")
    print(f"    Only cost: spread + funding (8h)")
    print(f"    vs BTC Binance (5.0 bps/side): ∞x cheaper (zero fee)")
    print(f"    vs BTC Hyperliquid (2.5 bps/side): ∞x cheaper (zero fee)")
    print(f"    vs GC CME (0.35 bps/side): still cheaper (zero fee)")

    # Return analysis (for breakeven comparison)
    closes = df['close'].values
    returns_1 = np.diff(closes) / closes[:-1] * 10000  # 1-min returns in bps
    returns_5 = (closes[5:] - closes[:-5]) / closes[:-5] * 10000  # 5-min approx

    print(f"\n  Return Statistics:")
    print(f"    1-min avg |return|: {np.abs(returns_1).mean():.2f} bps")
    print(f"    5-min avg |return|: {np.abs(returns_5).mean():.2f} bps")
    print(f"    1-min std:          {returns_1.std():.2f} bps")
    print(f"    5-min std:          {returns_5.std():.2f} bps")

    # Breakeven at zero fees
    for horizon_label, rets in [("1-min", returns_1), ("5-min", returns_5)]:
        mu = np.abs(rets).mean()
        for venue, c_oneway in [("Bitfinex (0bps)", 0.0), ("Hyperliquid (2.5bps)", 2.5),
                                 ("Binance (5bps)", 5.0)]:
            c_rt = c_oneway * 2
            p_break = (c_rt / mu + 1) / 2 if mu > 0 else 1.0
            print(f"    {horizon_label} breakeven @ {venue}: {p_break*100:.1f}%")

    # Save
    out_path = DATA_DIR / f"btc_usdt_perp_2025_1min.parquet"
    df.to_parquet(out_path, index=False, engine='pyarrow')
    print(f"\n[BITFINEX] Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Funding rate history
    if args.also_funding:
        funding_df = fetch_funding_history(args.symbol, start_ms, end_ms)
        if len(funding_df) > 0:
            funding_path = DATA_DIR / "btc_usdt_perp_2025_funding.parquet"
            funding_df.to_parquet(funding_path, index=False, engine='pyarrow')
            print(f"[BITFINEX] Funding data saved: {funding_path} ({len(funding_df):,} records)")

            # Funding rate summary
            if 'current_funding' in funding_df.columns:
                fr = funding_df['current_funding'].dropna()
                print(f"\n  Funding Rate Summary:")
                print(f"    Records: {len(fr):,}")
                print(f"    Mean:    {fr.mean()*100:.4f}%")
                print(f"    Median:  {fr.median()*100:.4f}%")
                print(f"    Std:     {fr.std()*100:.4f}%")
                print(f"    Min:     {fr.min()*100:.4f}%")
                print(f"    Max:     {fr.max()*100:.4f}%")
                # Annualized cost of holding a position through all funding
                annual_cost_bps = abs(fr.mean()) * 3 * 365 * 10000  # 3 times/day
                print(f"    Annualized drag: {annual_cost_bps:.1f} bps")


if __name__ == "__main__":
    main()
