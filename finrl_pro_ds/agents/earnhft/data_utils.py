"""
Data utilities for EarnHFT agent.

Provides chunking, regime labeling, and minute-level aggregation
for the hierarchical training pipeline.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def chunk_data(df: pd.DataFrame, chunk_length: int = 3600) -> List[pd.DataFrame]:
    """Split a DataFrame into fixed-length episodes (chunks).

    Args:
        df: LOB DataFrame.
        chunk_length: Number of rows per chunk.

    Returns:
        List of DataFrames, each of length chunk_length. The last chunk
        is dropped if it's shorter than chunk_length.
    """
    n_chunks = len(df) // chunk_length
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_length
        end = start + chunk_length
        chunks.append(df.iloc[start:end].reset_index(drop=True))
    logger.info(f"Split {len(df)} rows into {n_chunks} chunks of {chunk_length}")
    return chunks


def label_regimes(
    df: pd.DataFrame,
    window: int = 300,
    price_col: str = "bid_price_1",
) -> np.ndarray:
    """Classify market regimes: trend direction × volatility level.

    Regimes (4 categories):
        0 = trending-up, low-vol
        1 = trending-up, high-vol
        2 = trending-down, low-vol
        3 = trending-down, high-vol

    Args:
        df: LOB DataFrame.
        window: Lookback window for trend and volatility estimation.
        price_col: Price column to use.

    Returns:
        Array of regime labels, shape (len(df),), dtype int.
    """
    prices = df[price_col].values.astype(np.float64)
    T = len(prices)
    regimes = np.zeros(T, dtype=np.int64)

    for t in range(T):
        start = max(0, t - window)
        segment = prices[start : t + 1]
        if len(segment) < 2:
            regimes[t] = 0
            continue

        # Trend: positive return over window
        trend_up = segment[-1] > segment[0]

        # Volatility: std of log returns
        log_rets = np.diff(np.log(np.maximum(segment, 1e-12)))
        vol = np.std(log_rets) if len(log_rets) > 1 else 0.0

        # Threshold: median volatility (approximated dynamically)
        high_vol = vol > 1e-5  # Simple threshold for tick-level data

        if trend_up and not high_vol:
            regimes[t] = 0
        elif trend_up and high_vol:
            regimes[t] = 1
        elif not trend_up and not high_vol:
            regimes[t] = 2
        else:
            regimes[t] = 3

    return regimes


def aggregate_to_minute(
    df: pd.DataFrame,
    ticks_per_minute: int = 60,
    price_col: str = "bid_price_1",
) -> pd.DataFrame:
    """Aggregate tick-level data to minute-level features for the router.

    Produces per-minute:
        - open, high, low, close
        - volume (sum of bid_vol_1)
        - return (close/open - 1)
        - volatility (std of tick returns)
        - spread_mean (mean of ask_price_1 - bid_price_1)

    Args:
        df: Tick-level LOB DataFrame.
        ticks_per_minute: Number of ticks per minute interval.
        price_col: Price column for OHLC.

    Returns:
        Minute-level DataFrame with aggregated features.
    """
    prices = df[price_col].values
    n_minutes = len(df) // ticks_per_minute

    records = []
    for m in range(n_minutes):
        start = m * ticks_per_minute
        end = start + ticks_per_minute
        seg = df.iloc[start:end]
        p = prices[start:end]

        ohlc = {
            "open": p[0],
            "high": p.max(),
            "low": p.min(),
            "close": p[-1],
            "volume": seg["bid_vol_1"].sum() if "bid_vol_1" in seg.columns else 0.0,
            "return": (p[-1] / max(p[0], 1e-12)) - 1.0,
        }

        log_rets = np.diff(np.log(np.maximum(p, 1e-12)))
        ohlc["volatility"] = np.std(log_rets) if len(log_rets) > 1 else 0.0

        if "ask_price_1" in seg.columns:
            spreads = seg["ask_price_1"].values - seg["bid_price_1"].values
            ohlc["spread_mean"] = np.mean(spreads)
        else:
            ohlc["spread_mean"] = 0.0

        records.append(ohlc)

    result = pd.DataFrame(records)
    logger.info(f"Aggregated {len(df)} ticks to {n_minutes} minute bars")
    return result
