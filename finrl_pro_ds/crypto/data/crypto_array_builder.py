"""Build numpy arrays for CryptoPerpEnv from DataFrames.

Factored out of scripts/crypto_backtest_runner.py so that library modules
(e.g. finrl_pro.automl.walk_forward) can import without depending on scripts/.

H3 fix: Resolves architectural inversion where finrl_pro/ imported from scripts/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 8 crypto-specific feature columns produced by crypto_features.py
CRYPTO_FEATURE_COLS = [
    "funding_rate", "oi_change_pct", "btc_correlation",
    "volume_profile_skew", "liquidation_intensity", "exchange_netflow",
    "btc_dominance_regime", "cost_to_rebalance",
]


def build_env_arrays(
    ohlcv: pd.DataFrame,
    crypto_feats: pd.DataFrame,
    funding: pd.DataFrame,
    assets: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    feature_cols: list[str] | None = None,
) -> dict:
    """Build numpy arrays for CryptoPerpEnv from a time window.

    Args:
        ohlcv: Cleaned OHLCV DataFrame with [timestamp, ticker, open, high, low, close, volume].
        crypto_feats: Crypto features DataFrame with [timestamp, ticker, ...feature_cols...].
        funding: Funding rates DataFrame with [timestamp, ticker, funding_rate].
        assets: Ordered list of base asset symbols.
        start_ts: Window start timestamp (inclusive).
        end_ts: Window end timestamp (inclusive).
        feature_cols: Feature columns to use. Defaults to CRYPTO_FEATURE_COLS.

    Returns:
        Dict with keys: price_ary, tech_ary, funding_rate_ary, volume_ary, timestamps.
    """
    if feature_cols is None:
        feature_cols = CRYPTO_FEATURE_COLS

    # Filter to time window
    mask = (ohlcv["timestamp"] >= start_ts) & (ohlcv["timestamp"] <= end_ts)
    window_ohlcv = ohlcv[mask].copy()

    # Build price array (T, n_assets)
    price_pivot = window_ohlcv.pivot(index="timestamp", columns="ticker", values="close")
    price_pivot = price_pivot.reindex(columns=assets).ffill().fillna(0.0)

    # C2: Warn about assets with no data in this window
    zero_assets = [a for a in assets if (price_pivot[a] == 0.0).all()]
    if zero_assets:
        logger.warning(
            f"Assets with no price data in window (will be masked in env): {zero_assets}"
        )

    timestamps = price_pivot.index

    price_ary = price_pivot.values

    # Build volume array (T, n_assets)
    vol_pivot = window_ohlcv.pivot(index="timestamp", columns="ticker", values="volume")
    vol_pivot = vol_pivot.reindex(columns=assets).ffill().fillna(0.0)
    volume_ary = vol_pivot.values

    # Build funding rate array (T, n_assets)
    fund_mask = (funding["timestamp"] >= start_ts) & (funding["timestamp"] <= end_ts)
    window_fund = funding[fund_mask].copy()
    if not window_fund.empty:
        fund_pivot = window_fund.pivot(index="timestamp", columns="ticker", values="funding_rate")
        fund_pivot = fund_pivot.reindex(index=timestamps, columns=assets).ffill().fillna(0.0)
        funding_ary = fund_pivot.values
    else:
        funding_ary = np.zeros_like(price_ary)

    # Build tech array (T, n_assets * tech_dim)
    feat_mask = (crypto_feats["timestamp"] >= start_ts) & (crypto_feats["timestamp"] <= end_ts)
    window_feats = crypto_feats[feat_mask].copy()

    # Validate feature columns exist in the features DataFrame
    available_cols = set(window_feats.columns)
    missing_cols = [c for c in feature_cols if c not in available_cols]
    if missing_cols:
        raise ValueError(
            f"OBS-DIM violation: feature columns missing from crypto_features: "
            f"{missing_cols}. Available: {sorted(available_cols)}"
        )

    # Build per-asset feature arrays and concatenate
    tech_frames = []
    for asset in assets:
        asset_feats = window_feats[window_feats["ticker"] == asset].set_index("timestamp")
        asset_feats = asset_feats.reindex(timestamps)[feature_cols].ffill().fillna(0.0)
        tech_frames.append(asset_feats.values)

    # tech_ary shape: (T, n_assets * n_features)
    tech_ary = np.hstack(tech_frames)

    # OBS-DIM invariant: validate tech_dim matches actual feature count
    actual_features_per_asset = len(feature_cols)
    if tech_ary.shape[1] != len(assets) * actual_features_per_asset:
        raise ValueError(
            f"OBS-DIM violation: tech_ary has {tech_ary.shape[1]} columns but expected "
            f"{len(assets)} assets × {actual_features_per_asset} features = "
            f"{len(assets) * actual_features_per_asset}"
        )

    # Convert timestamps to epoch seconds
    ts_epoch = timestamps.astype(np.int64) // 10**9

    return {
        "price_ary": price_ary.astype(np.float64),
        "tech_ary": tech_ary.astype(np.float32),
        "funding_rate_ary": funding_ary.astype(np.float64),
        "volume_ary": volume_ary.astype(np.float64),
        "timestamps": ts_epoch.values,
    }
