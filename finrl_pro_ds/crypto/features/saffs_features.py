"""SAFFS feature integration for FinRL crypto strategies.

Imports SAFFSFeatureProvider from the SAFFS project via sys.path injection.
Set SAFFS_ROOT env var to point to the SAFFS project root (default: C:\\FinRL\\SAFFS).

Produces 13 features per asset per timestep:
    chronos_p10/p30/p50/p70/p90, chronos_spread,
    gahmm_price_bear/neutral/bull, gahmm_vol_low/normal/high,
    gahmm_composite_code
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Cross-project import via sys.path injection
SAFFS_ROOT = os.environ.get("SAFFS_ROOT", r"C:\FinRL\SAFFS")

_provider = None  # Singleton


def _get_provider(config: dict[str, Any] | None = None):
    """Get or create the SAFFSFeatureProvider singleton."""
    global _provider

    if _provider is not None:
        return _provider

    # Inject SAFFS into sys.path
    if SAFFS_ROOT not in sys.path:
        sys.path.insert(0, SAFFS_ROOT)
        logger.info(f"Added SAFFS_ROOT to sys.path: {SAFFS_ROOT}")

    from exports.finrl_feature_provider import SAFFSFeatureProvider

    config = config or {}
    _provider = SAFFSFeatureProvider(
        chronos_device=config.get("chronos_device", "cuda"),
        chronos_context_length=config.get("chronos_context_length", 512),
        gahmm_config=config.get("gahmm", {}),
        gahmm_refit_every=config.get("gahmm_refit_every", 720),
        gahmm_min_bars=config.get("gahmm_min_bars", 60),
    )
    _provider.load()
    return _provider


def compute_saffs_features(
    ohlcv_df: pd.DataFrame,
    assets: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute SAFFS features (Chronos-2 + GAHMM) for all assets.

    Args:
        ohlcv_df: OHLCV DataFrame with [timestamp, ticker, open, high, low, close, volume].
        assets: List of asset tickers.
        start_ts: Feature window start (inclusive).
        end_ts: Feature window end (inclusive).
        config: Optional SAFFS config dict (chronos_device, gahmm params, etc.).

    Returns:
        DataFrame with [timestamp, ticker] + 13 SAFFS feature columns.
    """
    provider = _get_provider(config)
    return provider.compute_features(ohlcv_df, assets, start_ts, end_ts)


def get_saffs_feature_cols() -> list[str]:
    """Return the list of SAFFS feature column names."""
    if SAFFS_ROOT not in sys.path:
        sys.path.insert(0, SAFFS_ROOT)
    from exports.finrl_feature_provider import SAFFS_FEATURE_COLS
    return list(SAFFS_FEATURE_COLS)


def unload_saffs() -> None:
    """Free SAFFS model resources (GPU memory etc.)."""
    global _provider
    if _provider is not None:
        _provider.unload()
        _provider = None
        logger.info("SAFFS provider unloaded")
