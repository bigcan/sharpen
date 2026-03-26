"""Funding rate arbitrage features for the delta-neutral spot-perp strategy.

Produces 12 features per asset tailored for funding arb timing, basis risk
assessment, and pair selection. All features respect point-in-time safety
(no look-ahead bias) and LEAK-1 compliance (window-local normalization
applied downstream in the array builder).

Features (per asset):
    1.  funding_rate_raw          — Clipped raw funding rate
    2.  funding_rate_annualized   — Annualized funding rate (raw × 3 × 365)
    3.  funding_ema_24h           — EMA(24) short-term funding trend
    4.  funding_ema_168h          — EMA(168) funding regime
    5.  funding_rate_std          — Rolling std of funding rate (720h) — volatility of funding
    6.  funding_cumsum_ffd        — FFD(d=0.4) on cumsum(funding_rate), normalized
    7.  basis_pct                 — (perp_close - spot_close) / spot_close
    8.  basis_ema_168h            — EMA(168) of basis — basis regime
    9.  basis_ffd                 — FFD(d=0.35) on basis_pct, normalized
    10. oi_change_pct             — Open interest % change (1h, lagged)
    11. log_oi_ffd                — FFD(d=0.4) on log(open_interest), normalized
    12. perp_volume_raw           — Raw perp volume (normalized downstream by array builder)
    13. btc_correlation           — Rolling correlation with BTC (168h)
    14. volatility_24h            — Realized hourly vol × sqrt(24) — basis risk proxy
    15. volume_profile_skew       — Buy vol / total vol ratio (24h proxy)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.features.crypto_features import _fractional_diff

logger = logging.getLogger(__name__)

# Publication lag registry (bars of 1H each)
PUBLICATION_LAGS: dict[str, int] = {
    "funding_rate": 0,
    "oi_change_pct": 1,
}

FUNDING_ARB_FEATURE_COLS = [
    "funding_rate_raw",
    "funding_rate_annualized",
    "funding_ema_24h",
    "funding_ema_168h",
    "funding_rate_std",
    "funding_cumsum_ffd",
    "basis_pct",
    "basis_ema_168h",
    "basis_ffd",
    "oi_change_pct",
    "log_oi_ffd",
    "perp_volume_raw",
    "btc_correlation",
    "volatility_24h",
    "volume_profile_skew",
]


def compute_funding_arb_features(
    perp_ohlcv: pd.DataFrame,
    spot_ohlcv: pd.DataFrame,
    funding_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    correlation_window: int = 168,
    volume_profile_window: int = 24,
    zscore_window: int = 720,
) -> pd.DataFrame:
    """Compute all funding-arb features for each asset.

    Args:
        perp_ohlcv: Perpetual OHLCV [timestamp, ticker, open, high, low, close, volume].
        spot_ohlcv: Spot OHLCV [timestamp, ticker, open, high, low, close, volume].
        funding_df: Funding rates [timestamp, ticker, funding_rate].
        oi_df: Open interest [timestamp, ticker, open_interest]. Optional.
        correlation_window: Window for BTC correlation (default 168h = 1 week).
        volume_profile_window: Window for volume profile skew (default 24h).
        zscore_window: Window for rolling z-scores (default 720h = 1 month).

    Returns:
        DataFrame with columns: [timestamp, ticker] + FUNDING_ARB_FEATURE_COLS
    """
    tickers = sorted(perp_ohlcv["ticker"].unique())
    all_frames = []

    # Pre-compute BTC returns for correlation
    btc_perp = perp_ohlcv[perp_ohlcv["ticker"] == "BTC"].set_index("timestamp").sort_index()
    btc_returns = btc_perp["close"].pct_change() if not btc_perp.empty else pd.Series(dtype=float)

    for ticker in tickers:
        # Get perp and spot data aligned by timestamp
        asset_perp = (
            perp_ohlcv[perp_ohlcv["ticker"] == ticker]
            .set_index("timestamp")
            .sort_index()
            .copy()
        )
        asset_spot = (
            spot_ohlcv[spot_ohlcv["ticker"] == ticker]
            .set_index("timestamp")
            .sort_index()
            .copy()
        )

        if asset_perp.empty:
            continue

        features = pd.DataFrame(index=asset_perp.index)
        features["ticker"] = ticker

        # Align spot to perp timestamps
        spot_close = asset_spot["close"].reindex(asset_perp.index).ffill().fillna(0.0)
        perp_close = asset_perp["close"]

        # --- 1. Funding rate raw (clipped) ---
        features["funding_rate_raw"] = _merge_funding(
            asset_perp.index, funding_df, ticker,
        )

        # --- 2. Funding rate annualized ---
        features["funding_rate_annualized"] = features["funding_rate_raw"] * 3 * 365

        # --- 3. Funding EMA 24h (short-term trend) ---
        features["funding_ema_24h"] = (
            features["funding_rate_raw"].ewm(span=24, min_periods=6).mean()
        )

        # --- 4. Funding EMA 168h (regime) ---
        features["funding_ema_168h"] = (
            features["funding_rate_raw"].ewm(span=168, min_periods=24).mean()
        )

        # --- 5. Funding rate std (rolling volatility of funding) ---
        min_per = min(max(24, zscore_window // 10), zscore_window)
        features["funding_rate_std"] = (
            features["funding_rate_raw"]
            .rolling(window=zscore_window, min_periods=min_per)
            .std()
            .fillna(0.0)
        )

        # --- 5b. Funding cumsum FFD ---
        funding_cumsum = features["funding_rate_raw"].cumsum()
        ffd_funding_raw = _fractional_diff(funding_cumsum, d=0.4, window=100)
        features["funding_cumsum_ffd"] = _rolling_zscore(ffd_funding_raw, window=zscore_window).clip(-5, 5)

        # --- 6. Basis % (perp - spot) / spot ---
        features["basis_pct"] = np.where(
            spot_close.abs() > 1e-10,
            (perp_close - spot_close) / spot_close,
            0.0,
        )

        # --- 7. Basis EMA 168h (basis regime) ---
        features["basis_ema_168h"] = (
            features["basis_pct"].ewm(span=168, min_periods=24).mean()
        )

        # --- 7b. Basis FFD ---
        ffd_basis_raw = _fractional_diff(features["basis_pct"], d=0.35, window=100)
        features["basis_ffd"] = _rolling_zscore(ffd_basis_raw, window=zscore_window).clip(-5, 5)

        # --- 8. Open interest % change ---
        features["oi_change_pct"] = _compute_oi_change(
            asset_perp.index, oi_df, ticker,
        )

        # --- 8b. Log OI FFD ---
        if oi_df is not None and not oi_df.empty:
            tic_oi = oi_df[oi_df["ticker"] == ticker]
            if not tic_oi.empty:
                tic_oi = tic_oi.set_index("timestamp")["open_interest"].sort_index()
                aligned_oi = tic_oi.reindex(asset_perp.index)
                oi_last = tic_oi.index.max()
                aligned_oi.loc[aligned_oi.index > oi_last] = np.nan
                aligned_oi = aligned_oi.ffill()

                # Apply publication lag
                aligned_oi = aligned_oi.shift(PUBLICATION_LAGS.get("oi_change_pct", 1)).ffill()

                # log of OI (clip lower bound to avoid log(0))
                # FFD logic
                log_oi = np.log(aligned_oi.clip(lower=1.0))
                ffd_oi_raw = _fractional_diff(log_oi, d=0.4, window=100)
                features["log_oi_ffd"] = _rolling_zscore(ffd_oi_raw, window=zscore_window).clip(-5, 5)
            else:
                features["log_oi_ffd"] = 0.0
        else:
            features["log_oi_ffd"] = 0.0

        # --- 9. Perp volume raw (normalized downstream by array builder) ---
        features["perp_volume_raw"] = asset_perp["volume"].fillna(0.0)

        # --- 10. BTC correlation ---
        if ticker == "BTC":
            features["btc_correlation"] = 1.0
        else:
            asset_returns = perp_close.pct_change()
            aligned_btc = btc_returns.reindex(asset_returns.index)
            if not btc_returns.empty:
                btc_last = btc_returns.index.max()
                aligned_btc.loc[aligned_btc.index > btc_last] = np.nan
            aligned_btc = aligned_btc.ffill().fillna(0.0)
            min_corr_periods = min(max(correlation_window // 2, 48), correlation_window)
            features["btc_correlation"] = (
                asset_returns.rolling(window=correlation_window, min_periods=min_corr_periods)
                .corr(aligned_btc)
                .fillna(0.0)
            )

        # --- 11. Volatility 24h (realized vol = std(hourly_returns) × sqrt(24)) ---
        hourly_returns = perp_close.pct_change()
        features["volatility_24h"] = (
            hourly_returns.rolling(window=24, min_periods=6).std() * np.sqrt(24)
        ).fillna(0.0)

        # --- 12. Volume profile skew ---
        is_buy = (asset_perp["close"] > asset_perp["open"]).astype(float)
        min_vp = min(max(12, volume_profile_window // 2), volume_profile_window)
        buy_vol = (is_buy * asset_perp["volume"]).rolling(
            window=volume_profile_window, min_periods=min_vp,
        ).sum()
        total_vol = asset_perp["volume"].rolling(
            window=volume_profile_window, min_periods=min_vp,
        ).sum()
        features["volume_profile_skew"] = (
            (buy_vol / (total_vol + 1e-10)).clip(0.0, 1.0).fillna(0.5)
        )

        features = features.reset_index(names="timestamp")
        all_frames.append(features)

    if not all_frames:
        raise ValueError("No funding arb features computed — empty OHLCV input?")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    logger.info(
        f"Funding arb features computed: {len(result)} rows, "
        f"{result['ticker'].nunique()} assets, "
        f"columns={list(result.columns)}",
    )
    return result


# ---------------------------------------------------------------------------
# Helper functions (reused patterns from crypto_features.py)
# ---------------------------------------------------------------------------

def _merge_funding(
    index: pd.DatetimeIndex,
    funding_df: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Merge funding rates onto OHLCV timestamps, forward-filled."""
    if funding_df is None or funding_df.empty:
        return pd.Series(0.0, index=index)

    tic_fund = funding_df[funding_df["ticker"] == ticker].copy()
    if tic_fund.empty:
        return pd.Series(0.0, index=index)

    tic_fund = tic_fund.set_index("timestamp")["funding_rate"].sort_index()
    aligned = tic_fund.reindex(index)
    fund_last = tic_fund.index.max()
    aligned.loc[aligned.index > fund_last] = np.nan
    aligned = aligned.ffill().fillna(0.0)
    return aligned.clip(-0.01, 0.01)


def _compute_oi_change(
    index: pd.DatetimeIndex,
    oi_df: pd.DataFrame | None,
    ticker: str,
) -> pd.Series:
    """Compute 1H percentage change in open interest."""
    if oi_df is None or oi_df.empty:
        return pd.Series(0.0, index=index)

    tic_oi = oi_df[oi_df["ticker"] == ticker].copy()
    if tic_oi.empty:
        return pd.Series(0.0, index=index)

    tic_oi = tic_oi.set_index("timestamp")["open_interest"].sort_index()
    aligned = tic_oi.reindex(index)
    oi_last = tic_oi.index.max()
    aligned.loc[aligned.index > oi_last] = np.nan
    aligned = aligned.ffill()
    pct_change = aligned.pct_change().fillna(0.0).clip(-1.0, 1.0)
    return pct_change.shift(PUBLICATION_LAGS["oi_change_pct"]).fillna(0.0)


def _rolling_zscore(series: pd.Series, window: int = 720) -> pd.Series:
    """Rolling Z-score normalization."""
    min_periods = min(max(24, window // 10), window)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return ((series - mean) / (std + 1e-6)).fillna(0.0)
