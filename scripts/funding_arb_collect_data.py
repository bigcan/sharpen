"""
funding_arb_collect_data.py — Fetch historical funding rates + OHLCV + open interest
from multiple exchanges, build the 4D market data array for training.

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted for FinRL-Pro_DS project structure.

Usage:
    python scripts/funding_arb_collect_data.py --days 90 --output data/funding_arb
    python scripts/funding_arb_collect_data.py --days 30 --symbols BTC/USDT ETH/USDT --exchanges binance bybit
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "DOGE/USDT",
    "AVAX/USDT", "LINK/USDT", "OP/USDT",
]
DEFAULT_EXCHANGES = ["binance", "bybit", "okx"]

PERP_SUFFIX = ":USDT"  # CCXT unified linear perp format
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


# ------------------------------------------------------------------ #
#  Exchange Helpers                                                    #
# ------------------------------------------------------------------ #

def get_exchange(name: str) -> ccxt.Exchange:
    """Initialize exchange with rate limiting."""
    ex_class = getattr(ccxt, name)
    ex = ex_class({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()
    return ex


def perp_symbol(symbol: str) -> str:
    """Convert BTC/USDT → BTC/USDT:USDT for linear perps."""
    return f"{symbol}{PERP_SUFFIX}"


# ------------------------------------------------------------------ #
#  Funding Rate History                                                #
# ------------------------------------------------------------------ #

def fetch_funding_rates(
    exchange: ccxt.Exchange,
    symbol: str,
    since: datetime,
    exchange_name: str,
) -> pd.DataFrame:
    """Fetch all funding rate history for a symbol since a given date."""
    psym = perp_symbol(symbol)
    since_ms = int(since.timestamp() * 1000)
    all_records = []

    logger.info(f"  Fetching FR history: {symbol} on {exchange_name}")
    try:
        while True:
            batch = exchange.fetch_funding_rate_history(
                symbol=psym, since=since_ms, limit=100,
            )
            if not batch:
                break
            for entry in batch:
                all_records.append({
                    "timestamp": pd.Timestamp(entry["timestamp"], unit="ms", tz="UTC"),
                    "symbol": symbol,
                    "exchange": exchange_name,
                    "funding_rate": entry.get("fundingRate", 0.0),
                })
            since_ms = batch[-1]["timestamp"] + 1
            time.sleep(RATE_LIMIT_SLEEP)

            if len(batch) < 100:
                break
    except Exception as e:
        logger.warning(f"  FR fetch failed for {symbol}/{exchange_name}: {e}")

    logger.info(f"  → {len(all_records)} FR records")
    return pd.DataFrame(all_records)


# ------------------------------------------------------------------ #
#  OHLCV History (for ATR, basis, volatility)                          #
# ------------------------------------------------------------------ #

def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    since: datetime,
    exchange_name: str,
    timeframe: str = "8h",
) -> pd.DataFrame:
    """Fetch OHLCV candles aligned to 8h funding intervals."""
    since_ms = int(since.timestamp() * 1000)
    all_candles = []

    logger.info(f"  Fetching OHLCV: {symbol} on {exchange_name} ({timeframe})")
    try:
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=500)
            if not batch:
                break
            all_candles.extend(batch)
            since_ms = batch[-1][0] + 1
            time.sleep(RATE_LIMIT_SLEEP)
            if len(batch) < 500:
                break
    except Exception as e:
        logger.warning(f"  OHLCV fetch failed for {symbol}/{exchange_name}: {e}")

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["exchange"] = exchange_name
    logger.info(f"  → {len(df)} candles")
    return df


# ------------------------------------------------------------------ #
#  Perp OHLCV (for basis spread calculation)                           #
# ------------------------------------------------------------------ #

def fetch_perp_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    since: datetime,
    exchange_name: str,
    timeframe: str = "8h",
) -> pd.DataFrame:
    """Fetch perpetual contract OHLCV for basis calculation."""
    psym = perp_symbol(symbol)
    since_ms = int(since.timestamp() * 1000)
    all_candles = []

    logger.info(f"  Fetching Perp OHLCV: {psym} on {exchange_name}")
    try:
        while True:
            batch = exchange.fetch_ohlcv(psym, timeframe, since=since_ms, limit=500)
            if not batch:
                break
            all_candles.extend(batch)
            since_ms = batch[-1][0] + 1
            time.sleep(RATE_LIMIT_SLEEP)
            if len(batch) < 500:
                break
    except Exception as e:
        logger.warning(f"  Perp OHLCV fetch failed for {psym}/{exchange_name}: {e}")

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["exchange"] = exchange_name
    return df


# ------------------------------------------------------------------ #
#  Feature Engineering                                                 #
# ------------------------------------------------------------------ #

def compute_atr(ohlcv_df: pd.DataFrame, period: int = 21) -> pd.Series:
    """Compute normalized Average True Range (ATR / close)."""
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]
    close = ohlcv_df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    return atr / close


def compute_basis(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> pd.Series:
    """Compute basis spread: (perp_close - spot_close) / spot_close."""
    merged = spot_df[["timestamp", "close"]].merge(
        perp_df[["timestamp", "close"]],
        on="timestamp",
        suffixes=("_spot", "_perp"),
    )
    basis = (merged["close_perp"] - merged["close_spot"]) / merged["close_spot"]
    return basis


# ------------------------------------------------------------------ #
#  Build 4D Market Data Array                                          #
# ------------------------------------------------------------------ #

def build_market_array(
    fr_df: pd.DataFrame,
    ohlcv_dict: dict,
    perp_ohlcv_dict: dict,
    symbols: list[str],
    exchanges: list[str],
) -> np.ndarray:
    """
    Build the 4D array: (n_steps, n_symbols, n_exchanges, 7)

    Features:
        0: fr_current          — current funding rate
        1: fr_predicted        — EMA-predicted next FR
        2: fr_7d_mean          — 7-day rolling mean FR
        3: fr_7d_std           — 7-day rolling std FR
        4: basis_spread        — (perp - spot) / spot
        5: atr_normalized      — ATR / close
        6: volume_change       — normalized volume change
    """
    fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"], utc=True)
    all_timestamps = sorted(fr_df["timestamp"].unique())
    n_steps = len(all_timestamps)
    n_sym = len(symbols)
    n_ex = len(exchanges)

    logger.info(f"Building array: {n_steps} steps × {n_sym} symbols × {n_ex} exchanges")
    data = np.zeros((n_steps, n_sym, n_ex, 7), dtype=np.float32)

    ts_to_idx = {ts: i for i, ts in enumerate(all_timestamps)}

    for s_i, symbol in enumerate(symbols):
        for e_i, exchange in enumerate(exchanges):
            # --- Funding rates ---
            mask = (fr_df["symbol"] == symbol) & (fr_df["exchange"] == exchange)
            sym_fr = fr_df[mask].sort_values("timestamp")

            if sym_fr.empty:
                continue

            fr_series = sym_fr.set_index("timestamp")["funding_rate"]

            for ts, fr_val in fr_series.items():
                if ts in ts_to_idx:
                    idx = ts_to_idx[ts]
                    data[idx, s_i, e_i, 0] = fr_val

            # Feature 1: EMA prediction
            fr_arr = data[:, s_i, e_i, 0]
            ema = np.zeros(n_steps)
            ema[0] = fr_arr[0]
            for t in range(1, n_steps):
                ema[t] = 0.7 * fr_arr[t] + 0.3 * ema[t - 1]
            data[:, s_i, e_i, 1] = ema

            # Features 2-3: Rolling mean/std (21 periods = 7 days)
            fr_pd = pd.Series(fr_arr)
            data[:, s_i, e_i, 2] = fr_pd.rolling(21, min_periods=1).mean().values
            data[:, s_i, e_i, 3] = fr_pd.rolling(21, min_periods=1).std().fillna(0).values

            # --- OHLCV features ---
            key = (symbol, exchange)

            # Feature 4: Basis spread
            if key in ohlcv_dict and key in perp_ohlcv_dict:
                spot_df = ohlcv_dict[key]
                perp_df = perp_ohlcv_dict[key]
                if not spot_df.empty and not perp_df.empty:
                    basis = compute_basis(spot_df, perp_df)
                    merged_ts = spot_df.merge(perp_df, on="timestamp")["timestamp"]
                    for j, ts in enumerate(merged_ts):
                        if ts in ts_to_idx and j < len(basis):
                            data[ts_to_idx[ts], s_i, e_i, 4] = basis.iloc[j]

            # Feature 5: Normalized ATR
            if key in ohlcv_dict:
                spot_df = ohlcv_dict[key]
                if not spot_df.empty:
                    atr = compute_atr(spot_df)
                    for j, (_, row) in enumerate(spot_df.iterrows()):
                        ts = row["timestamp"]
                        if ts in ts_to_idx and j < len(atr):
                            data[ts_to_idx[ts], s_i, e_i, 5] = atr.iloc[j]

            # Feature 6: Volume change
            if key in ohlcv_dict:
                spot_df = ohlcv_dict[key]
                if not spot_df.empty and len(spot_df) > 1:
                    vol_change = spot_df["volume"].pct_change().fillna(0).clip(-2, 2)
                    for j, (_, row) in enumerate(spot_df.iterrows()):
                        ts = row["timestamp"]
                        if ts in ts_to_idx and j < len(vol_change):
                            data[ts_to_idx[ts], s_i, e_i, 6] = vol_change.iloc[j]

    # Forward-fill any gaps
    for s_i in range(n_sym):
        for e_i in range(n_ex):
            for f_i in range(7):
                series = data[:, s_i, e_i, f_i]
                mask = series != 0
                if mask.any():
                    last_val = 0.0
                    for t in range(n_steps):
                        if series[t] != 0:
                            last_val = series[t]
                        else:
                            series[t] = last_val
                    data[:, s_i, e_i, f_i] = series

    return data


# ------------------------------------------------------------------ #
#  Main Collection Pipeline                                            #
# ------------------------------------------------------------------ #

def collect_all(
    symbols: list[str],
    exchanges: list[str],
    days: int,
    output_dir: str,
):
    """Full data collection pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    since = datetime.utcnow() - timedelta(days=days)
    logger.info(f"Collecting {days} days of data since {since.isoformat()}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Exchanges: {exchanges}")

    # 1. Collect funding rates
    all_fr = []
    for ex_name in exchanges:
        ex = get_exchange(ex_name)
        for symbol in symbols:
            df = fetch_funding_rates(ex, symbol, since, ex_name)
            if not df.empty:
                all_fr.append(df)

    fr_df = pd.concat(all_fr, ignore_index=True) if all_fr else pd.DataFrame()
    logger.info(f"Total FR records: {len(fr_df)}")

    # 2. Collect spot OHLCV
    ohlcv_dict = {}
    for ex_name in exchanges:
        ex = get_exchange(ex_name)
        for symbol in symbols:
            df = fetch_ohlcv(ex, symbol, since, ex_name, "8h")
            if not df.empty:
                ohlcv_dict[(symbol, ex_name)] = df

    # 3. Collect perp OHLCV (for basis)
    perp_ohlcv_dict = {}
    for ex_name in exchanges:
        ex = get_exchange(ex_name)
        for symbol in symbols:
            df = fetch_perp_ohlcv(ex, symbol, since, ex_name, "8h")
            if not df.empty:
                perp_ohlcv_dict[(symbol, ex_name)] = df

    # 4. Save raw DataFrames
    if not fr_df.empty:
        fr_df.to_parquet(output_path / "funding_rates.parquet", index=False)
        logger.info(f"Saved funding_rates.parquet ({len(fr_df)} rows)")

    all_ohlcv = (
        pd.concat(ohlcv_dict.values(), ignore_index=True) if ohlcv_dict else pd.DataFrame()
    )
    if not all_ohlcv.empty:
        all_ohlcv.to_parquet(output_path / "spot_ohlcv.parquet", index=False)
        logger.info(f"Saved spot_ohlcv.parquet ({len(all_ohlcv)} rows)")

    all_perp = (
        pd.concat(perp_ohlcv_dict.values(), ignore_index=True)
        if perp_ohlcv_dict
        else pd.DataFrame()
    )
    if not all_perp.empty:
        all_perp.to_parquet(output_path / "perp_ohlcv.parquet", index=False)

    # 5. Build 4D market data array
    if not fr_df.empty:
        market_data = build_market_array(
            fr_df, ohlcv_dict, perp_ohlcv_dict, symbols, exchanges,
        )
        np.save(output_path / "market_data.npy", market_data)
        logger.info(f"Saved market_data.npy: shape {market_data.shape}")

        # Train/test split
        split_idx = int(len(market_data) * 0.8)
        np.save(output_path / "train_data.npy", market_data[:split_idx])
        np.save(output_path / "eval_data.npy", market_data[split_idx:])
        logger.info(f"Train: {split_idx} steps, Eval: {len(market_data) - split_idx} steps")

    # Summary stats
    if not fr_df.empty:
        logger.info("\n=== Funding Rate Summary ===")
        summary = fr_df.groupby(["symbol", "exchange"])["funding_rate"].agg(
            ["mean", "std", "min", "max", "count"],
        )
        logger.info(f"\n{summary.to_string()}")

        logger.info("\n=== Estimated Annualized Yield (positive carry) ===")
        for (sym, ex), row in summary.iterrows():
            annual_yield = row["mean"] * 3 * 365 * 100
            logger.info(f"  {sym:12s} {ex:8s}: {annual_yield:+.1f}% APR")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect funding rate data for RL training",
    )
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--exchanges", nargs="+", default=DEFAULT_EXCHANGES)
    parser.add_argument("--output", type=str, default="data/funding_arb")
    args = parser.parse_args()

    collect_all(
        symbols=args.symbols,
        exchanges=args.exchanges,
        days=args.days,
        output_dir=args.output,
    )
