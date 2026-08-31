"""PRISM feature integration for FinRL crypto strategies.

Imports PRISMFeatureProvider from the PRISM project via sys.path injection.
Set PRISM_ROOT env var to point to the PRISM project root (default: C:\\FinRL\\PRISM).

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
# Platform-agnostic default: Windows → C:\FinRL\PRISM, others → ~/FinRL/PRISM
_DEFAULT_PRISM_ROOT = (
    r"C:\FinRL\PRISM" if sys.platform == "win32"
    else os.path.expanduser("~/FinRL/PRISM")
)
PRISM_ROOT = os.environ.get("PRISM_ROOT", _DEFAULT_PRISM_ROOT)

_provider = None  # Singleton


def _get_provider(config: dict[str, Any] | None = None):
    """Get or create the PRISMFeatureProvider singleton."""
    global _provider

    if _provider is not None:
        if config is not None:
            logger.warning(
                "PRISMFeatureProvider already loaded — ignoring config. "
                "Call unload_prism() first to reconfigure."
            )
        return _provider

    # Inject SAFFS into sys.path
    if PRISM_ROOT not in sys.path:
        sys.path.insert(0, PRISM_ROOT)
        logger.info(f"Added PRISM_ROOT to sys.path: {PRISM_ROOT}")

    from exports.finrl_feature_provider import PRISMFeatureProvider

    config = config or {}
    _provider = PRISMFeatureProvider(
        chronos_device=config.get("chronos_device", "cuda"),
        chronos_context_length=config.get("chronos_context_length", 512),
        gahmm_config=config.get("gahmm", {}),
        gahmm_refit_every=config.get("gahmm_refit_every", 720),
        gahmm_min_bars=config.get("gahmm_min_bars", 60),
    )
    _provider.load()
    return _provider


def compute_prism_features(
    ohlcv_df: pd.DataFrame,
    assets: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute PRISM features (Chronos-2 + GAHMM) for all assets.

    Args:
        ohlcv_df: OHLCV DataFrame with [timestamp, ticker, open, high, low, close, volume].
        assets: List of asset tickers.
        start_ts: Feature window start (inclusive).
        end_ts: Feature window end (inclusive).
        config: Optional PRISM config dict (chronos_device, gahmm params, etc.).

    Returns:
        DataFrame with [timestamp, ticker] + 13 PRISM feature columns.
    """
    provider = _get_provider(config)
    return provider.compute_features(ohlcv_df, assets, start_ts, end_ts)


def get_prism_feature_cols() -> list[str]:
    """Return the list of PRISM feature column names."""
    if PRISM_ROOT not in sys.path:
        sys.path.insert(0, PRISM_ROOT)
    from exports.finrl_feature_provider import PRISM_FEATURE_COLS
    return list(PRISM_FEATURE_COLS)


def get_prism_passthrough_cols() -> list[str]:
    """Return PRISM feature columns that should NOT be z-score normalized.

    These are the GAHMM probability features — already bounded [0,1] and
    sum-normalized. Z-scoring destroys their probability semantics.
    """
    if PRISM_ROOT not in sys.path:
        sys.path.insert(0, PRISM_ROOT)
    from exports.finrl_feature_provider import PRISM_PASSTHROUGH_COLS
    return list(PRISM_PASSTHROUGH_COLS)


def unload_prism() -> None:
    """Free PRISM model resources (GPU memory etc.)."""
    global _provider
    if _provider is not None:
        _provider.unload()
        _provider = None
        logger.info("PRISM provider unloaded")
