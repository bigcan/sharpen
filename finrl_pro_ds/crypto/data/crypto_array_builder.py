"""Build numpy arrays for CryptoPerpEnv from DataFrames.

Factored out of scripts/crypto_backtest_runner.py so that library modules
(e.g. finrl_pro.automl.walk_forward) can import without depending on scripts/.

H3 fix: Resolves architectural inversion where finrl_pro/ imported from scripts/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.features.funding_arb_features import (
    FUNDING_ARB_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

# The 6 crypto-specific feature columns produced by crypto_features.py.
# M1 fix: Removed exchange_netflow and cost_to_rebalance (always zero —
# wasted 25% of feature capacity and could confuse the agent).
CRYPTO_FEATURE_COLS = [
    "funding_rate", "oi_change_pct", "btc_correlation",
    "volume_profile_skew", "liquidation_intensity",
    "btc_dominance_regime",
]

# Note: SAFFS feature column names are imported from
# finrl_pro_ds.crypto.features.saffs_features (single source of truth).


def build_env_arrays(
    ohlcv: pd.DataFrame,
    crypto_feats: pd.DataFrame,
    funding: pd.DataFrame,
    assets: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    feature_cols: list[str] | None = None,
    passthrough_cols: list[str] | None = None,
    norm_window: int = 720,
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
        passthrough_cols: Feature columns to exempt from z-score normalization
            (e.g. GAHMM probability features that are already bounded [0,1]).
        norm_window: Rolling z-score window for per-window re-normalization (LEAK-1 fix).

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
            f"Assets with no price data in window (will be masked in env): {zero_assets}",
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
            f"{missing_cols}. Available: {sorted(available_cols)}",
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
            f"{len(assets) * actual_features_per_asset}",
        )

    # LEAK-1 fix: Re-normalize features using window-local statistics.
    # Features computed on the full dataset carry rolling z-score / rolling
    # correlation statistics from prior windows.  Re-applying rolling z-score
    # here uses only data within this window, breaking cross-window leakage.
    # Passthrough columns (e.g. GAHMM probabilities) are exempt — they're
    # already bounded and sum-normalized, z-scoring destroys their semantics.
    passthrough_indices = None
    if passthrough_cols and feature_cols:
        n_feats = len(feature_cols)
        pt_local = [feature_cols.index(c) for c in passthrough_cols if c in feature_cols]
        if pt_local:
            passthrough_indices = set()
            for asset_idx in range(len(assets)):
                for local_idx in pt_local:
                    passthrough_indices.add(asset_idx * n_feats + local_idx)
    tech_ary = _renormalize_within_window(tech_ary, norm_window, passthrough=passthrough_indices)

    # Convert timestamps to epoch seconds
    ts_epoch = timestamps.astype(np.int64) // 10**9

    return {
        "price_ary": price_ary.astype(np.float64),
        "tech_ary": tech_ary.astype(np.float32),
        "funding_rate_ary": funding_ary.astype(np.float64),
        "volume_ary": volume_ary.astype(np.float64),
        "timestamps": ts_epoch.values,
    }


def _renormalize_within_window(
    tech_ary: np.ndarray,
    norm_window: int,
    clip: float = 5.0,
    passthrough: set[int] | None = None,
) -> np.ndarray:
    """Re-normalize feature columns using window-local rolling z-score.

    Eliminates cross-window normalization leakage (LEAK-1) by replacing
    each column's values with z-scores computed solely from data within
    this walk-forward window.

    Args:
        tech_ary: Feature array of shape (T, n_cols).
        norm_window: Rolling window size for mean/std.
        clip: Symmetric clip range for z-scores.
        passthrough: Set of column indices to copy through WITHOUT z-scoring
            (e.g. probability features that are already bounded [0,1]).

    Returns:
        Normalized array (same shape), float32.
    """
    T, n_cols = tech_ary.shape
    result = np.empty_like(tech_ary, dtype=np.float64)
    min_periods = max(24, norm_window // 10)

    for col_idx in range(n_cols):
        if passthrough and col_idx in passthrough:
            result[:, col_idx] = tech_ary[:, col_idx]
            continue
        col = pd.Series(tech_ary[:, col_idx], dtype=np.float64)
        roll_mean = col.rolling(window=norm_window, min_periods=min_periods).mean()
        roll_std = col.rolling(window=norm_window, min_periods=min_periods).std()
        z = (col - roll_mean) / (roll_std + 1e-6)
        result[:, col_idx] = z.fillna(0.0).clip(-clip, clip).values

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Funding Arb Array Builder
# ---------------------------------------------------------------------------

def build_funding_arb_arrays(
    spot_ohlcv: pd.DataFrame,
    perp_ohlcv: pd.DataFrame,
    arb_features: pd.DataFrame,
    funding: pd.DataFrame,
    assets: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    feature_cols: list[str] | None = None,
    norm_window: int = 720,
) -> dict:
    """Build numpy arrays for FundingArbEnv from a time window.

    Args:
        spot_ohlcv: Spot OHLCV DataFrame [timestamp, ticker, open, high, low, close, volume].
        perp_ohlcv: Perp OHLCV DataFrame [timestamp, ticker, open, high, low, close, volume].
        arb_features: Funding arb features [timestamp, ticker, ...feature_cols...].
        funding: Funding rates [timestamp, ticker, funding_rate].
        assets: Ordered list of base asset symbols.
        start_ts: Window start timestamp (inclusive).
        end_ts: Window end timestamp (inclusive).
        feature_cols: Feature columns to use. Defaults to FUNDING_ARB_FEATURE_COLS.
        norm_window: Rolling z-score window for per-window re-normalization (LEAK-1).

    Returns:
        Dict with keys: spot_price_ary, perp_price_ary, funding_rate_ary,
                         spot_volume_ary, perp_volume_ary, tech_ary, timestamps.
    """
    if feature_cols is None:
        feature_cols = FUNDING_ARB_FEATURE_COLS

    # Filter to time window — perp OHLCV defines the canonical timestamps
    perp_mask = (perp_ohlcv["timestamp"] >= start_ts) & (perp_ohlcv["timestamp"] <= end_ts)
    window_perp = perp_ohlcv[perp_mask].copy()

    # Perp price array (T, n_assets)
    perp_pivot = window_perp.pivot(index="timestamp", columns="ticker", values="close")
    perp_pivot = perp_pivot.reindex(columns=assets).ffill().fillna(0.0)
    timestamps = perp_pivot.index

    perp_price_ary = perp_pivot.values

    # Spot price array (T, n_assets)
    spot_mask = (spot_ohlcv["timestamp"] >= start_ts) & (spot_ohlcv["timestamp"] <= end_ts)
    window_spot = spot_ohlcv[spot_mask].copy()
    spot_pivot = window_spot.pivot(index="timestamp", columns="ticker", values="close")
    spot_pivot = spot_pivot.reindex(index=timestamps, columns=assets).ffill().fillna(0.0)
    spot_price_ary = spot_pivot.values

    # Warn about zero-price assets
    zero_spot = [a for a in assets if (spot_pivot[a] == 0.0).all()]
    zero_perp = [a for a in assets if (perp_pivot[a] == 0.0).all()]
    if zero_spot:
        logger.warning(f"Assets with no spot price data in window: {zero_spot}")
    if zero_perp:
        logger.warning(f"Assets with no perp price data in window: {zero_perp}")

    # Perp volume array (T, n_assets)
    perp_vol_pivot = window_perp.pivot(index="timestamp", columns="ticker", values="volume")
    perp_vol_pivot = perp_vol_pivot.reindex(index=timestamps, columns=assets).ffill().fillna(0.0)
    perp_volume_ary = perp_vol_pivot.values

    # Spot volume array (T, n_assets)
    spot_vol_pivot = window_spot.pivot(index="timestamp", columns="ticker", values="volume")
    spot_vol_pivot = spot_vol_pivot.reindex(index=timestamps, columns=assets).ffill().fillna(0.0)
    spot_volume_ary = spot_vol_pivot.values

    # Funding rate array (T, n_assets)
    fund_mask = (funding["timestamp"] >= start_ts) & (funding["timestamp"] <= end_ts)
    window_fund = funding[fund_mask].copy()
    if not window_fund.empty:
        fund_pivot = window_fund.pivot(index="timestamp", columns="ticker", values="funding_rate")
        fund_pivot = fund_pivot.reindex(index=timestamps, columns=assets).ffill().fillna(0.0)
        funding_ary = fund_pivot.values
    else:
        funding_ary = np.zeros_like(perp_price_ary)

    # Tech array (T, n_assets * n_features)
    feat_mask = (arb_features["timestamp"] >= start_ts) & (arb_features["timestamp"] <= end_ts)
    window_feats = arb_features[feat_mask].copy()

    # Validate feature columns
    available_cols = set(window_feats.columns)
    missing_cols = [c for c in feature_cols if c not in available_cols]
    if missing_cols:
        raise ValueError(
            f"OBS-DIM violation: feature columns missing from arb_features: "
            f"{missing_cols}. Available: {sorted(available_cols)}",
        )

    tech_frames = []
    for asset in assets:
        asset_feats = window_feats[window_feats["ticker"] == asset].set_index("timestamp")
        asset_feats = asset_feats.reindex(timestamps)[feature_cols].ffill().fillna(0.0)
        tech_frames.append(asset_feats.values)

    tech_ary = np.hstack(tech_frames)

    # OBS-DIM invariant
    actual_features_per_asset = len(feature_cols)
    if tech_ary.shape[1] != len(assets) * actual_features_per_asset:
        raise ValueError(
            f"OBS-DIM violation: tech_ary has {tech_ary.shape[1]} columns but expected "
            f"{len(assets)} assets × {actual_features_per_asset} features = "
            f"{len(assets) * actual_features_per_asset}",
        )

    # LEAK-1 fix: Re-normalize within window
    tech_ary = _renormalize_within_window(tech_ary, norm_window)

    # Convert timestamps to epoch seconds
    ts_epoch = timestamps.astype(np.int64) // 10**9

    return {
        "spot_price_ary": spot_price_ary.astype(np.float64),
        "perp_price_ary": perp_price_ary.astype(np.float64),
        "funding_rate_ary": funding_ary.astype(np.float64),
        "spot_volume_ary": spot_volume_ary.astype(np.float64),
        "perp_volume_ary": perp_volume_ary.astype(np.float64),
        "tech_ary": tech_ary.astype(np.float32),
        "timestamps": ts_epoch.values,
    }
