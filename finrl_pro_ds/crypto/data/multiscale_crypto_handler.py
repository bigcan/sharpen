"""Multi-Scale Crypto Data Handler for CryptoPerpSwingEnv (Sync-2H).

Multi-asset adaptation of MultiScaleOHLCVHandler. Loads per-asset OHLCV data
from a merged DataFrame, resamples to multiple timescales, computes 8 features
per scale per asset, and provides a stepping interface for the environment.

Features per scale per asset (8 dims, TC-aligned):
  0. log_return = log(close_t / close_{t-1})
  1. atr_norm = ATR(14) / EMA(ATR, 50) - 1.0
  2. parkinson_vol = sqrt(log(H/L)^2 / (4*ln2))
  3-6. open_z, high_z, low_z, close_z = SymLog -> EMA-Z(120) -> tanh
  7. volume_z = SymLog -> EMA-Z(120) -> tanh

Optional SigBoost features (V1.1, gated by feature_set_version):
  Per asset, injected into env private state (5 dims each):
    0. funding_ema_24h   -- EMA(24) on funding rate
    1. funding_ema_168h  -- EMA(168) on funding rate
    2. funding_cumsum_ffd -- FFD(d=0.4) on cumsum(funding), z-scored
    3. momentum_spread_24h  -- asset return - BTC return (24h)
    4. momentum_spread_168h -- asset return - BTC return (168h)

LEAK-1 compliant: norm_cutoff_date splits normalization per asset.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.features.crypto_features import (
    _fractional_diff,
    _rolling_zscore,
)
from finrl_pro_ds.data.multiscale_handler import (
    _compute_scale_features,
    _resample_ohlcv,
)

logger = logging.getLogger(__name__)


class MultiScaleCryptoHandler:
    """Multi-asset, multi-scale OHLCV handler for CryptoPerpSwingEnv.

    Takes a merged OHLCV DataFrame (timestamp, ticker, OHLCV) as produced by
    the crypto data pipeline, splits by asset, resamples to configured scales,
    computes features, and provides a step interface aligned to the base scale.

    Interface:
        reset() -> None
        step() -> dict | None
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame],
        assets: list[str],
        feature_config: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        norm_cutoff_date: Optional[str] = None,
    ):
        """
        Args:
            ohlcv_df: Merged OHLCV with columns [timestamp, ticker, open, high, low, close, volume].
            funding_df: Merged funding rates with columns [timestamp, ticker, funding_rate].
                        None if funding not available.
            assets: Ordered list of base asset symbols (e.g. ["BTC", "ETH", ...]).
            feature_config: Dict with keys: scales, window_size, obs_mode, summary_feature_indices, norm_span.
            start_date: Start date for the env window (features computed on full history for EMA warmup).
            end_date: End date for the env window.
            norm_cutoff_date: LEAK-1 normalization boundary (typically val/test split date).
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.feature_config = feature_config
        self.scales = feature_config.get("scales", [1, 4, 24])
        self.window_size = feature_config.get("window_size", 30)
        self.norm_span = feature_config.get("norm_span", 120)

        # Observation mode
        self.obs_mode = feature_config.get("obs_mode", "summary_stats")
        self.summary_feature_indices = feature_config.get(
            "summary_feature_indices", [0, 1, 2, 6, 7],
        )

        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.norm_cutoff_date = pd.to_datetime(norm_cutoff_date) if norm_cutoff_date else None

        self._ptr = 0

        # Load and process
        self._load_data(ohlcv_df, funding_df)

    def _load_data(self, ohlcv_df: pd.DataFrame, funding_df: Optional[pd.DataFrame]):
        """Split by asset, resample to scales, compute features."""
        # Ensure timestamp is datetime, tz-naive
        ohlcv = ohlcv_df.copy()
        ts = pd.to_datetime(ohlcv["timestamp"], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        ohlcv["timestamp"] = ts

        if self.end_date is not None:
            ohlcv = ohlcv[ohlcv["timestamp"] <= self.end_date]

        # Base scale = smallest in scales list (should be 1 for 1H data)
        base_scale = min(self.scales)
        self._base_scale = base_scale

        # Build canonical timestamp index from base scale
        # Use the first asset that has data to establish the time grid
        sample_asset = None
        for asset in self.assets:
            asset_df = ohlcv[ohlcv["ticker"] == asset].sort_values("timestamp")
            if len(asset_df) > 0:
                sample_asset = asset
                break
        if sample_asset is None:
            raise ValueError("No OHLCV data found for any asset")

        # Get base-scale timestamps from first available asset
        base_ohlcv = ohlcv[ohlcv["ticker"] == sample_asset].sort_values("timestamp")
        if base_scale > 1:
            base_resampled = _resample_ohlcv(base_ohlcv, base_scale * 60)
        else:
            base_resampled = base_ohlcv.copy()
        base_timestamps = base_resampled["timestamp"].values

        # Per-asset per-scale features: _scale_features[scale] = (T_scale, n_assets, 8)
        self._scale_features: dict[int, np.ndarray] = {}
        self._scale_timestamps: dict[int, np.ndarray] = {}

        for scale in self.scales:
            scale_minutes = scale * 60  # Convert hours to minutes for resampler

            # Compute features for each asset at this scale
            per_asset_features = []
            scale_ts = None

            for asset in self.assets:
                asset_ohlcv = ohlcv[ohlcv["ticker"] == asset].sort_values("timestamp")

                if len(asset_ohlcv) < 10:
                    logger.warning(f"Asset {asset} has insufficient data ({len(asset_ohlcv)} bars)")

                # Resample to target scale
                if scale > 1:
                    resampled = _resample_ohlcv(asset_ohlcv, scale_minutes)
                else:
                    resampled = asset_ohlcv[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                    resampled = resampled.reset_index(drop=True)

                # Compute norm cutoff index for this scale
                norm_cutoff_idx = None
                if self.norm_cutoff_date is not None:
                    cutoff_mask = resampled["timestamp"] >= self.norm_cutoff_date
                    if cutoff_mask.any():
                        norm_cutoff_idx = cutoff_mask.idxmax()

                # Compute 8 features using shared function from multiscale_handler
                features = _compute_scale_features(resampled, norm_cutoff_idx, self.norm_span)

                # Align to common timestamp grid via reindex
                if scale_ts is None:
                    scale_ts = resampled["timestamp"].values

                per_asset_features.append(features)

            # Stack: (T_scale, n_assets, 8)
            # Pad shorter assets to max length
            max_len = max(f.shape[0] for f in per_asset_features)
            aligned = []
            for f in per_asset_features:
                if f.shape[0] < max_len:
                    pad = np.zeros((max_len - f.shape[0], f.shape[1]), dtype=np.float32)
                    f = np.concatenate([pad, f], axis=0)
                aligned.append(f)

            self._scale_features[scale] = np.stack(aligned, axis=1)  # (T, N, 8)
            self._scale_timestamps[scale] = scale_ts[:max_len] if scale_ts is not None else np.array([])

        # Apply start_date trimming AFTER feature computation (EMA warmup)
        base_features = self._scale_features[base_scale]
        trim_idx = 0
        if self.start_date is not None and len(self._scale_timestamps[base_scale]) > 0:
            start_mask = self._scale_timestamps[base_scale] >= np.datetime64(self.start_date)
            if start_mask.any():
                trim_idx = np.argmax(start_mask)
                trim_idx = max(0, trim_idx - self.window_size)

        # Trim all scales and rebuild index maps
        for scale in self.scales:
            if scale == base_scale:
                self._scale_features[scale] = self._scale_features[scale][trim_idx:]
                self._scale_timestamps[scale] = self._scale_timestamps[scale][trim_idx:]

        # Base scale data for stepping
        base_df_full = self._scale_features[base_scale]  # (T, N, 8)
        self._len = base_df_full.shape[0]

        # Extract close prices per asset from OHLCV (base scale)
        self._base_close = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_high = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_low = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_volume = np.zeros((self._len, self.n_assets), dtype=np.float64)

        for ai, asset in enumerate(self.assets):
            asset_ohlcv = ohlcv[ohlcv["ticker"] == asset].sort_values("timestamp")
            if base_scale > 1:
                asset_resampled = _resample_ohlcv(asset_ohlcv, base_scale * 60)
            else:
                asset_resampled = asset_ohlcv.copy().reset_index(drop=True)

            # Trim to match
            if self.start_date is not None:
                start_mask = asset_resampled["timestamp"] >= self.start_date
                if start_mask.any():
                    t_idx = max(0, start_mask.idxmax() - self.window_size)
                    asset_resampled = asset_resampled.iloc[t_idx:].reset_index(drop=True)

            n = min(len(asset_resampled), self._len)
            offset = self._len - n  # Right-align if shorter
            self._base_close[offset:, ai] = asset_resampled["close"].values[:n].astype(np.float64)
            self._base_high[offset:, ai] = asset_resampled["high"].values[:n].astype(np.float64)
            self._base_low[offset:, ai] = asset_resampled["low"].values[:n].astype(np.float64)
            self._base_volume[offset:, ai] = asset_resampled["volume"].values[:n].astype(np.float64)

        self._base_timestamps = self._scale_timestamps[base_scale]

        # Pre-compute ATR per asset on base scale for vol-regime scaling
        self._base_atr = np.zeros((self._len, self.n_assets), dtype=np.float64)
        for ai in range(self.n_assets):
            close = self._base_close[:, ai]
            high = self._base_high[:, ai]
            low = self._base_low[:, ai]
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
            )
            self._base_atr[:, ai] = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # Build scale index map (base -> coarser scale indices)
        self._scale_index_map: dict[int, np.ndarray] = {}
        for scale in self.scales:
            if scale == base_scale:
                self._scale_index_map[scale] = np.arange(self._len)
            else:
                # X2 / audit P2-01: map each base bar to the last COMPLETED coarse
                # bar (no forward look-ahead). _resample_ohlcv stamps coarse bars at
                # interval START (label='left'), so a coarse bar stamped t closes at
                # t + scale HOURS; it is causal for a base bar at time b only when
                # t + scale <= b. Searching on (base_ts - scale_ns) selects that
                # last-closed bar. Crypto scales are HOURS (not minutes like the V7
                # MultiScaleOHLCVHandler), so scale_ns = scale * 3600 * 1e9. The prior
                # code searched base_ts directly and returned the still-in-progress
                # coarse bar, leaking up to (scale - base_scale) hours of the base
                # bar's own future into obs (window mean/std AND the `last` value).
                coarse_ts = self._scale_timestamps[scale].astype("int64")
                base_ts = self._base_timestamps.astype("int64")
                scale_ns = scale * 3600 * 1_000_000_000
                indices = np.searchsorted(coarse_ts, base_ts - scale_ns, side="right") - 1
                indices = np.clip(indices, 0, len(coarse_ts) - 1)
                self._scale_index_map[scale] = indices

        # Load funding rates
        self._base_funding = np.zeros((self._len, self.n_assets), dtype=np.float64)
        if funding_df is not None and not funding_df.empty:
            self._load_funding(funding_df)

        # SigBoost V1.1: compute crypto-specific features (gated by config)
        self._compute_sigboost_features()

        self._ptr = self.window_size

        sigboost_str = f", sigboost={self._sigboost_features.shape}" if self._sigboost_features is not None else ""
        logger.info(
            f"MultiScaleCryptoHandler loaded: {self._len} base bars ({base_scale}h), "
            f"{self.n_assets} assets, scales={self.scales}, window={self.window_size}{sigboost_str}",
        )

        # Verify sufficient data
        if self._len < self.window_size + 10:
            raise ValueError(
                f"Insufficient data after filtering: {self._len} bars "
                f"(need at least {self.window_size + 10})",
            )

    def _load_funding(self, funding_df: pd.DataFrame):
        """Load funding rates aligned to base timestamps."""
        fund = funding_df.copy()
        ts = pd.to_datetime(fund["timestamp"], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        fund["timestamp"] = ts

        base_ts_index = pd.DatetimeIndex(self._base_timestamps)

        for ai, asset in enumerate(self.assets):
            asset_fund = fund[fund["ticker"] == asset].set_index("timestamp")
            if asset_fund.empty:
                continue
            aligned = asset_fund["funding_rate"].reindex(base_ts_index, method="ffill").fillna(0.0)
            n = min(len(aligned), self._len)
            self._base_funding[:n, ai] = aligned.values[:n]

        # Clip extreme funding rates
        self._base_funding = np.clip(self._base_funding, -0.5, 0.5)

    def _compute_sigboost_features(self):
        """Compute V1.1 SigBoost crypto features per asset at base scale.

        Gated by feature_config["feature_set_version"]. Produces (T, n_assets, 5):
          0: funding_ema_24h, 1: funding_ema_168h, 2: funding_cumsum_ffd,
          3: momentum_spread_24h, 4: momentum_spread_168h.

        Stores result in self._sigboost_features (None if V1).
        """
        version = self.feature_config.get("feature_set_version", "v1")
        if version != "v1.1":
            self._sigboost_features = None
            return

        T = self._len
        N = self.n_assets
        feats = np.zeros((T, N, 5), dtype=np.float32)

        # Find BTC index for momentum spread reference
        btc_idx = None
        for i, asset in enumerate(self.assets):
            if asset == "BTC":
                btc_idx = i
                break

        # Pre-compute BTC returns for momentum spread
        if btc_idx is not None:
            btc_close = pd.Series(self._base_close[:, btc_idx], dtype=np.float64)
            btc_ret_24 = btc_close.pct_change(24).fillna(0.0)
            btc_ret_168 = btc_close.pct_change(168).fillna(0.0)
        else:
            btc_ret_24 = pd.Series(0.0, index=range(T))
            btc_ret_168 = pd.Series(0.0, index=range(T))

        for ai in range(N):
            funding = pd.Series(self._base_funding[:, ai], dtype=np.float64)

            # 0: funding_ema_24h
            feats[:, ai, 0] = funding.ewm(span=24, min_periods=6).mean().fillna(0.0).values

            # 1: funding_ema_168h
            feats[:, ai, 1] = funding.ewm(span=168, min_periods=24).mean().fillna(0.0).values

            # 2: funding_cumsum_ffd (cumsum -> FFD -> rolling z-score -> clip)
            funding_cumsum = funding.cumsum()
            ffd_raw = _fractional_diff(funding_cumsum, d=0.4, window=100)
            feats[:, ai, 2] = (
                _rolling_zscore(ffd_raw, window=720).clip(-5, 5).fillna(0.0).values
            )

            # 3-4: momentum spread vs BTC
            if ai == btc_idx:
                # BTC vs itself = 0
                feats[:, ai, 3] = 0.0
                feats[:, ai, 4] = 0.0
            else:
                asset_close = pd.Series(self._base_close[:, ai], dtype=np.float64)
                asset_ret_24 = asset_close.pct_change(24).fillna(0.0)
                asset_ret_168 = asset_close.pct_change(168).fillna(0.0)
                feats[:, ai, 3] = (asset_ret_24 - btc_ret_24).clip(-1.0, 1.0).fillna(0.0).values
                feats[:, ai, 4] = (asset_ret_168 - btc_ret_168).clip(-1.0, 1.0).fillna(0.0).values

        self._sigboost_features = feats
        logger.info(
            f"SigBoost V1.1 features computed: {feats.shape} "
            f"(btc_idx={btc_idx}, 5 features per asset)",
        )

    def reset(self):
        """Reset pointer to start of data."""
        self._ptr = self.window_size

    def step(self) -> Optional[dict]:
        """Advance one base-scale bar, return multi-asset multi-scale obs.

        Returns:
            dict with keys:
                scale_0: (n_assets, window_size, 8) or (n_assets * n_summary,)
                scale_1: same shape
                scale_2: same shape
                close: (n_assets,) float64
                atr: (n_assets,) float64
                funding_rate: (n_assets,) float64
                volume: (n_assets,) float64
                timestamp: numpy datetime64
            or None if data exhausted
        """
        if self._ptr >= self._len:
            return None

        result = {}

        for i, scale in enumerate(self.scales):
            features = self._scale_features[scale]  # (T_scale, n_assets, 8)
            if scale == self._base_scale:
                idx = self._ptr
            else:
                idx = self._scale_index_map[scale][self._ptr]

            # Window: [idx - window_size + 1, idx + 1) for each asset
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = features[start:end]  # (W', n_assets, 8)

            # Pad if insufficient history
            if window.shape[0] < self.window_size:
                pad_len = self.window_size - window.shape[0]
                pad = np.tile(window[0:1], (pad_len, 1, 1))
                window = np.concatenate([pad, window], axis=0)

            # window is (W, n_assets, 8) -> transpose to (n_assets, W, 8)
            window = np.transpose(window, (1, 0, 2))

            key = f"scale_{i}"
            if self.obs_mode == "summary_stats":
                result[key] = self._compute_summary_stats_multi(window)
            else:
                result[key] = window.copy()

        result["close"] = self._base_close[self._ptr].copy()
        result["atr"] = self._base_atr[self._ptr].copy()
        result["funding_rate"] = self._base_funding[self._ptr].copy()
        result["volume"] = self._base_volume[self._ptr].copy()
        result["timestamp"] = self._base_timestamps[self._ptr]

        if self._sigboost_features is not None:
            result["sigboost_features"] = self._sigboost_features[self._ptr].copy()

        self._ptr += 1
        return result

    def _compute_summary_stats_multi(self, window: np.ndarray) -> np.ndarray:
        """Compute summary stats for all assets at one scale.

        Args:
            window: (n_assets, W, 8) feature window.

        Returns:
            flat (n_assets * n_summary,) float32 where n_summary = len(indices) * 3.
        """
        # Select features: (n_assets, W, n_selected)
        selected = window[:, :, self.summary_feature_indices]

        # Compute per-asset: mean, std, last over window dim
        means = selected.mean(axis=1)      # (n_assets, n_selected)
        stds = selected.std(axis=1)        # (n_assets, n_selected)
        stds = np.where(stds < 1e-8, 0.0, stds)
        last = selected[:, -1, :]          # (n_assets, n_selected)

        # Concat per asset: (n_assets, n_selected * 3)
        per_asset = np.concatenate([means, stds, last], axis=1)

        # Flatten: (n_assets * n_selected * 3,)
        return per_asset.reshape(-1).astype(np.float32)

    @property
    def data_length(self) -> int:
        """Total number of base-scale bars available."""
        return self._len
