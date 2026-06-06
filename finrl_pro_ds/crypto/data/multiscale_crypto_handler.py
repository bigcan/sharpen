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
        """Split by asset, resample to scales, compute features, align by timestamp.

        FIND-CMGP1-03 fix: per-asset features and OHLCV are reindexed onto a canonical
        timestamp grid (sorted union of all assets' resampled timestamps) per scale.
        Slots where an asset has no bar are zero-filled and flagged inactive in
        ``_base_active``. CryptoPerpSwingEnv's C2 mask gates positions on
        ``close > 1e-10``, so zero-fill OHLCV propagates "no data" without further
        wiring. For top-N universes with continuous bars, the union equals each
        asset's grid, so reindex is a no-op and outputs are bit-identical to the
        pre-fix path.
        """
        # Ensure timestamp is datetime, tz-naive
        ohlcv = ohlcv_df.copy()
        ts = pd.to_datetime(ohlcv["timestamp"], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        ohlcv["timestamp"] = ts

        if self.end_date is not None:
            ohlcv = ohlcv[ohlcv["timestamp"] <= self.end_date]

        base_scale = min(self.scales)
        self._base_scale = base_scale

        # Pre-resample each asset for each scale, store frames keyed by timestamp.
        # asset_resampled[scale][asset] = DataFrame indexed by timestamp.
        asset_resampled: dict[int, dict[str, pd.DataFrame]] = {s: {} for s in self.scales}
        any_data = False
        for asset in self.assets:
            asset_df = ohlcv[ohlcv["ticker"] == asset].sort_values("timestamp")
            if len(asset_df) > 0:
                any_data = True
            elif self.norm_cutoff_date is not None or self.start_date is not None:
                logger.warning(f"Asset {asset} has no OHLCV data in window")
            for scale in self.scales:
                if scale > 1:
                    rs = _resample_ohlcv(asset_df, scale * 60)
                else:
                    rs = asset_df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                    rs = rs.reset_index(drop=True)
                if len(rs) > 0 and len(rs) < 10:
                    logger.warning(
                        f"Asset {asset} has insufficient {scale}h data ({len(rs)} bars)",
                    )
                asset_resampled[scale][asset] = rs.set_index("timestamp").sort_index()
        if not any_data:
            raise ValueError("No OHLCV data found for any asset")

        # Canonical timestamp grid per scale = sorted union over assets.
        self._scale_timestamps: dict[int, np.ndarray] = {}
        for scale in self.scales:
            union = pd.DatetimeIndex([])
            for asset in self.assets:
                union = union.union(asset_resampled[scale][asset].index)
            self._scale_timestamps[scale] = union.values

        # Compute features per asset on its NATIVE grid (so log_return/EMA are
        # correct), then reindex onto the canonical grid with zero-fill for
        # missing rows.
        self._scale_features: dict[int, np.ndarray] = {}
        for scale in self.scales:
            canonical = pd.DatetimeIndex(self._scale_timestamps[scale])
            T = len(canonical)
            feats = np.zeros((T, self.n_assets, 8), dtype=np.float32)
            for ai, asset in enumerate(self.assets):
                rs_idx = asset_resampled[scale][asset]
                if len(rs_idx) == 0:
                    continue
                rs = rs_idx.reset_index()
                # Norm cutoff on the asset's native grid (LEAK-1).
                norm_cutoff_idx = None
                if self.norm_cutoff_date is not None:
                    cutoff_mask = rs["timestamp"] >= self.norm_cutoff_date
                    if cutoff_mask.any():
                        norm_cutoff_idx = int(cutoff_mask.idxmax())
                asset_features = _compute_scale_features(rs, norm_cutoff_idx, self.norm_span)
                # Map asset native timestamps to canonical-grid positions.
                asset_ts = pd.DatetimeIndex(rs["timestamp"].values)
                indexer = canonical.get_indexer(asset_ts)
                valid = indexer >= 0
                feats[indexer[valid], ai, :] = asset_features[valid]
            self._scale_features[scale] = feats

        # Apply start_date trim with window_size warmup buffer on base scale.
        trim_idx = 0
        base_canonical_full = pd.DatetimeIndex(self._scale_timestamps[base_scale])
        if self.start_date is not None and len(base_canonical_full) > 0:
            start_mask = base_canonical_full >= self.start_date
            if start_mask.any():
                first_idx = int(np.argmax(start_mask.values))
                trim_idx = max(0, first_idx - self.window_size)

        if trim_idx > 0:
            self._scale_features[base_scale] = self._scale_features[base_scale][trim_idx:]
            self._scale_timestamps[base_scale] = self._scale_timestamps[base_scale][trim_idx:]

        base_features = self._scale_features[base_scale]  # (T, N, 8)
        self._len = base_features.shape[0]
        self._base_timestamps = self._scale_timestamps[base_scale]

        # Reindex per-asset OHLCV and ATR onto base canonical grid (zero-fill).
        # Inactive slots: close=high=low=volume=0 → CryptoPerpSwingEnv C2 mask
        # (close > 1e-10) gates positions and PnL on those slots.
        base_canonical = pd.DatetimeIndex(self._base_timestamps)
        self._base_close = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_high = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_low = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_volume = np.zeros((self._len, self.n_assets), dtype=np.float64)
        self._base_active = np.zeros((self._len, self.n_assets), dtype=bool)
        self._base_atr = np.zeros((self._len, self.n_assets), dtype=np.float64)

        # Stash native base-scale frames for sigboost (returns must be computed
        # on the native grid to avoid spurious ±100% spikes at gap boundaries).
        self._asset_native_base: dict[str, pd.DataFrame] = asset_resampled[base_scale]

        for ai, asset in enumerate(self.assets):
            rs_idx = asset_resampled[base_scale][asset]
            if len(rs_idx) == 0:
                continue
            asset_ts = pd.DatetimeIndex(rs_idx.index)
            indexer = base_canonical.get_indexer(asset_ts)
            valid = indexer >= 0
            slot = indexer[valid]

            close_vals = rs_idx["close"].values
            high_vals = rs_idx["high"].values
            low_vals = rs_idx["low"].values
            volume_vals = rs_idx["volume"].values

            self._base_close[slot, ai] = close_vals[valid].astype(np.float64)
            self._base_high[slot, ai] = high_vals[valid].astype(np.float64)
            self._base_low[slot, ai] = low_vals[valid].astype(np.float64)
            self._base_volume[slot, ai] = volume_vals[valid].astype(np.float64)
            self._base_active[slot, ai] = True

            # ATR on the asset's native grid, then reindex onto canonical.
            close_native = close_vals.astype(np.float64)
            high_native = high_vals.astype(np.float64)
            low_native = low_vals.astype(np.float64)
            prev_close = np.roll(close_native, 1)
            if len(prev_close) > 0:
                prev_close[0] = close_native[0]
            tr_native = np.maximum(
                high_native - low_native,
                np.maximum(np.abs(high_native - prev_close), np.abs(low_native - prev_close)),
            )
            atr_native = pd.Series(tr_native).rolling(14, min_periods=1).mean().values
            self._base_atr[slot, ai] = atr_native[valid]

        # Build scale index map (base canonical -> coarser scale canonical).
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

        # Funding rates (already timestamp-aligned via reindex+ffill).
        self._base_funding = np.zeros((self._len, self.n_assets), dtype=np.float64)
        if funding_df is not None and not funding_df.empty:
            self._load_funding(funding_df)

        # SigBoost V1.1
        self._compute_sigboost_features()

        self._ptr = self.window_size

        sigboost_str = (
            f", sigboost={self._sigboost_features.shape}"
            if self._sigboost_features is not None else ""
        )
        active_pct = float(self._base_active.mean()) * 100.0 if self._len > 0 else 0.0
        logger.info(
            f"MultiScaleCryptoHandler loaded: {self._len} base bars ({base_scale}h), "
            f"{self.n_assets} assets, scales={self.scales}, window={self.window_size}, "
            f"active={active_pct:.1f}%{sigboost_str}",
        )

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

        # Returns computed on each asset's NATIVE grid then reindexed onto base
        # canonical (FIND-CMGP1-03): pct_change on the zero-filled canonical
        # close array would produce ±100% spikes at gap boundaries.
        base_canonical = pd.DatetimeIndex(self._base_timestamps)
        per_asset_ret_24: list[Optional[np.ndarray]] = [None] * N
        per_asset_ret_168: list[Optional[np.ndarray]] = [None] * N
        for ai, asset in enumerate(self.assets):
            rs = self._asset_native_base.get(asset)
            if rs is None or len(rs) == 0:
                continue
            close_native = rs["close"].astype(np.float64)
            r24 = close_native.pct_change(24).fillna(0.0)
            r168 = close_native.pct_change(168).fillna(0.0)
            per_asset_ret_24[ai] = r24.reindex(base_canonical, fill_value=0.0).values
            per_asset_ret_168[ai] = r168.reindex(base_canonical, fill_value=0.0).values

        if btc_idx is not None and per_asset_ret_24[btc_idx] is not None:
            btc_ret_24_arr = per_asset_ret_24[btc_idx]
            btc_ret_168_arr = per_asset_ret_168[btc_idx]
        else:
            btc_ret_24_arr = np.zeros(T, dtype=np.float64)
            btc_ret_168_arr = np.zeros(T, dtype=np.float64)

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
                feats[:, ai, 3] = 0.0
                feats[:, ai, 4] = 0.0
            elif per_asset_ret_24[ai] is not None:
                feats[:, ai, 3] = np.clip(
                    per_asset_ret_24[ai] - btc_ret_24_arr, -1.0, 1.0,
                ).astype(np.float32)
                feats[:, ai, 4] = np.clip(
                    per_asset_ret_168[ai] - btc_ret_168_arr, -1.0, 1.0,
                ).astype(np.float32)

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
                close: (n_assets,) float64 — 0.0 where asset has no bar at this slot
                atr: (n_assets,) float64
                funding_rate: (n_assets,) float64
                volume: (n_assets,) float64
                active: (n_assets,) bool — True iff asset had a real bar at this slot
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
        result["active"] = self._base_active[self._ptr].copy()
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
