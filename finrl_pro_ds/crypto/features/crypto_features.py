"""Crypto-specific features for Synapse Crypto 1H.

Produces 8 crypto-specific features per asset on top of the standard
technical indicator set. All features respect point-in-time safety
(no look-ahead bias).

Features:
    1. funding_rate          — Perpetual swap funding rate (raw)
    2. oi_change_pct         — Open interest % change (1h)
    3. btc_correlation       — Rolling correlation with BTC (168h)
    4. volume_profile_skew   — Buy vol / total vol ratio (24h proxy)
    5. liquidation_intensity — Normalized liquidation volume (24h)
    6. exchange_netflow       — Net exchange inflow/outflow (24h, if available)
    7. btc_dominance_regime  — BTC dominance regime indicator
    8. cost_to_rebalance     — Estimated txn cost (derived per-step in env)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Publication lag registry (bars of 1H each)
# ---------------------------------------------------------------------------
PUBLICATION_LAGS: dict[str, int] = {
    "btc_dominance": 1,          # T-1h (CoinGecko API)
    "total_market_cap_ffd": 1,   # T-1h
    "funding_rate": 0,           # Real-time from exchange
    "oi_change_pct": 1,          # T-1h
    "fear_greed_index": 24,      # Published daily, use T-24h
    "dxy_ffd": 1,                # T-1h (market hours only)
}


def compute_crypto_features(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    oi_df: pd.DataFrame | None = None,
    btc_dominance_series: pd.Series | None = None,
    fear_greed_series: pd.Series | None = None,
    correlation_window: int = 168,
    volume_profile_window: int = 24,
) -> pd.DataFrame:
    """Compute all crypto-specific features for each asset.

    Args:
        ohlcv_df: Cleaned OHLCV with columns [timestamp, ticker, open, high, low, close, volume].
        funding_df: Funding rates with columns [timestamp, ticker, funding_rate].
        oi_df: Open interest with columns [timestamp, ticker, open_interest]. Optional.
        btc_dominance_series: BTC dominance time series (index=timestamp). Optional.
        fear_greed_series: Fear & Greed index (index=timestamp). Optional.
        correlation_window: Window for BTC correlation (default 168h = 1 week).
        volume_profile_window: Window for volume profile skew (default 24h).

    Returns:
        DataFrame with crypto features merged onto OHLCV timestamps.
        Columns: [timestamp, ticker, funding_rate, oi_change_pct, btc_correlation,
                  volume_profile_skew, liquidation_intensity, exchange_netflow,
                  btc_dominance_regime, cost_to_rebalance]
    """
    tickers = sorted(ohlcv_df["ticker"].unique())
    all_frames = []

    # Pre-compute BTC returns for correlation
    btc_ohlcv = ohlcv_df[ohlcv_df["ticker"] == "BTC"].set_index("timestamp").sort_index()
    btc_returns = btc_ohlcv["close"].pct_change() if not btc_ohlcv.empty else pd.Series(dtype=float)

    for ticker in tickers:
        asset_ohlcv = (
            ohlcv_df[ohlcv_df["ticker"] == ticker]
            .set_index("timestamp")
            .sort_index()
            .copy()
        )

        if asset_ohlcv.empty:
            continue

        features = pd.DataFrame(index=asset_ohlcv.index)
        features["ticker"] = ticker

        # --- 1. Funding rate (raw, clipped) ---
        features["funding_rate"] = _merge_funding(
            asset_ohlcv.index, funding_df, ticker
        )

        # --- 2. Open interest % change ---
        features["oi_change_pct"] = _compute_oi_change(
            asset_ohlcv.index, oi_df, ticker
        )

        # --- 3. BTC correlation (rolling) ---
        if ticker == "BTC":
            features["btc_correlation"] = 1.0
        else:
            asset_returns = asset_ohlcv["close"].pct_change()
            # Align with BTC returns
            aligned_btc = btc_returns.reindex(asset_returns.index)
            # LEAK-2: Truncate forward-fill to BTC data extent
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

        # --- 4. Volume profile skew (buy volume proxy) ---
        # Proxy: if close > open, classify as "buy bar"; ratio over window
        is_buy = (asset_ohlcv["close"] > asset_ohlcv["open"]).astype(float)
        buy_vol = (is_buy * asset_ohlcv["volume"]).rolling(
            window=volume_profile_window, min_periods=min(max(12, volume_profile_window // 2), volume_profile_window)
        ).sum()
        total_vol = asset_ohlcv["volume"].rolling(
            window=volume_profile_window, min_periods=min(max(12, volume_profile_window // 2), volume_profile_window)
        ).sum()
        features["volume_profile_skew"] = (buy_vol / (total_vol + 1e-10)).clip(0.0, 1.0).fillna(0.5)

        # --- 5. Liquidation intensity (placeholder — use volume spike as proxy) ---
        # Real liquidation data requires websocket or specialized API.
        # Proxy: extreme volume * extreme price move in same bar
        vol_zscore = _rolling_zscore(asset_ohlcv["volume"], window=168).clip(-5, 5)
        ret_abs = asset_ohlcv["close"].pct_change().abs()
        ret_zscore = _rolling_zscore(ret_abs, window=168).clip(-5, 5)
        features["liquidation_intensity"] = (
            (vol_zscore * ret_zscore).clip(0, 10).fillna(0.0)
        )

        # --- 6. Exchange netflow (placeholder — zero if unavailable) ---
        features["exchange_netflow"] = 0.0

        # --- 7. BTC dominance regime (apply publication lag before regime encoding) ---
        if btc_dominance_series is not None and not btc_dominance_series.empty:
            dom = btc_dominance_series.reindex(asset_ohlcv.index)
            # LEAK-2: ffill within available range, truncate beyond
            last_available = btc_dominance_series.index.max()
            dom = dom.ffill()
            dom.loc[dom.index > last_available] = np.nan
            dom = dom.shift(PUBLICATION_LAGS["btc_dominance"]).ffill().fillna(50.0)
            # Vectorized regime encoding (replaces slow .apply() loop)
            features["btc_dominance_regime"] = np.select(
                [dom > 55, dom < 40],
                [1.0, 0.0],
                default=0.5,
            )
        else:
            features["btc_dominance_regime"] = 0.5  # Neutral default

        # --- 8. Cost to rebalance (placeholder — computed in env per step) ---
        features["cost_to_rebalance"] = 0.0

        features = features.reset_index(names="timestamp")
        all_frames.append(features)

    if not all_frames:
        raise ValueError("No crypto features computed — empty OHLCV input?")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    logger.info(
        f"Crypto features computed: {len(result)} rows, "
        f"{result['ticker'].nunique()} assets, "
        f"columns={list(result.columns)}"
    )
    return result


def compute_macro_features(
    ohlcv_df: pd.DataFrame,
    btc_dominance_series: pd.Series | None = None,
    total_market_cap_series: pd.Series | None = None,
    fear_greed_series: pd.Series | None = None,
    dxy_series: pd.Series | None = None,
    ffd_d: float = 0.4,
    norm_window: int = 720,
) -> pd.DataFrame:
    """Compute macro / cross-asset features broadcast to all assets.

    Features:
        btc_dominance       — BTC market cap / total crypto cap
        total_market_cap_ffd — FFD on log(total_cap)
        fear_greed_index    — Crypto Fear & Greed (daily, ffilled to 1H)
        dxy_ffd             — FFD on log(DXY), resampled to 1H

    All features use lagged values matching their publication delay.
    """
    timestamps = ohlcv_df[["timestamp"]].drop_duplicates().sort_values("timestamp")
    ts_index = pd.DatetimeIndex(timestamps["timestamp"])

    macro = pd.DataFrame(index=ts_index)

    # BTC dominance (lag 1h)
    if btc_dominance_series is not None and not btc_dominance_series.empty:
        aligned = btc_dominance_series.reindex(ts_index)
        # LEAK-2: ffill within available range, truncate beyond
        last_available = btc_dominance_series.index.max()
        aligned = aligned.ffill()
        aligned.loc[aligned.index > last_available] = np.nan
        macro["btc_dominance"] = aligned.shift(PUBLICATION_LAGS["btc_dominance"]).ffill().fillna(0.5)
    else:
        macro["btc_dominance"] = 0.5

    # Total market cap FFD (lag 1h)
    if total_market_cap_series is not None and not total_market_cap_series.empty:
        aligned = total_market_cap_series.reindex(ts_index)
        last_available = total_market_cap_series.index.max()
        aligned = aligned.ffill()
        aligned.loc[aligned.index > last_available] = np.nan
        log_cap = np.log(aligned.clip(lower=1e6))
        macro["total_market_cap_ffd"] = _fractional_diff(
            log_cap.shift(PUBLICATION_LAGS["total_market_cap_ffd"]), d=ffd_d
        )
    else:
        macro["total_market_cap_ffd"] = 0.0

    # Fear & Greed index (lag 24h)
    if fear_greed_series is not None and not fear_greed_series.empty:
        aligned = fear_greed_series.reindex(ts_index)
        last_available = fear_greed_series.index.max()
        aligned = aligned.ffill()
        aligned.loc[aligned.index > last_available] = np.nan
        macro["fear_greed_index"] = aligned.shift(PUBLICATION_LAGS["fear_greed_index"]).ffill().fillna(50.0)
    else:
        macro["fear_greed_index"] = 50.0  # Neutral

    # DXY FFD (lag 1h, market hours only)
    if dxy_series is not None and not dxy_series.empty:
        aligned = dxy_series.reindex(ts_index)
        last_available = dxy_series.index.max()
        aligned = aligned.ffill()
        aligned.loc[aligned.index > last_available] = np.nan
        log_dxy = np.log(aligned.clip(lower=50.0))
        macro["dxy_ffd"] = _fractional_diff(
            log_dxy.shift(PUBLICATION_LAGS["dxy_ffd"]), d=ffd_d
        )
    else:
        macro["dxy_ffd"] = 0.0

    # Normalize macro features with rolling z-score
    for col in ["btc_dominance", "total_market_cap_ffd", "fear_greed_index", "dxy_ffd"]:
        if col in macro.columns:
            macro[col] = _rolling_zscore(macro[col], window=norm_window).clip(-5, 5).fillna(0.0)

    macro = macro.fillna(0.0)
    logger.info(f"Macro features computed: {len(macro)} timestamps, columns={list(macro.columns)}")
    return macro


# ---------------------------------------------------------------------------
# Helper functions
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
    # LEAK-2: Truncate forward-fill to available funding data extent
    fund_last = tic_fund.index.max()
    aligned.loc[aligned.index > fund_last] = np.nan
    aligned = aligned.ffill().fillna(0.0)
    # Clip to prevent extreme values
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
    # LEAK-2: Truncate forward-fill to available OI data extent
    oi_last = tic_oi.index.max()
    aligned.loc[aligned.index > oi_last] = np.nan
    aligned = aligned.ffill()
    pct_change = aligned.pct_change().fillna(0.0).clip(-1.0, 1.0)
    # Apply publication lag
    return pct_change.shift(PUBLICATION_LAGS["oi_change_pct"]).fillna(0.0)


def _rolling_zscore(series: pd.Series, window: int = 720) -> pd.Series:
    """Rolling Z-score normalization."""
    mean = series.rolling(window=window, min_periods=min(max(24, window // 10), window)).mean()
    std = series.rolling(window=window, min_periods=min(max(24, window // 10), window)).std()
    return ((series - mean) / (std + 1e-6)).fillna(0.0)


def _fractional_diff(series: pd.Series, d: float = 0.4, window: int = 100) -> pd.Series:
    """Fixed-window fractional differentiation (FFD).

    Simplified implementation for feature generation.
    Uses the same approach as Lopez de Prado's Advances in Financial ML.
    Vectorized via np.convolve (replaces Python loop).
    """
    weights = _get_ffd_weights(d, window)
    w = weights.flatten()

    values = series.ffill().fillna(0.0).values

    # Full convolution, then trim to 'valid' region and zero-pad the front
    conv = np.convolve(values, w, mode="full")
    # The valid output starts at index len(w)-1 and has len(values)-len(w)+1 entries
    valid_start = len(w) - 1
    valid = conv[valid_start : valid_start + len(values) - len(w) + 1]

    result = np.zeros(len(values), dtype=np.float64)
    result[len(w) - 1 :] = valid

    return pd.Series(result, index=series.index)


def _get_ffd_weights(d: float, window: int) -> np.ndarray:
    """Compute FFD weights."""
    weights = [1.0]
    for k in range(1, window):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < 1e-6:
            break
        weights.append(w)
    return np.array(weights[::-1]).reshape(-1, 1)
