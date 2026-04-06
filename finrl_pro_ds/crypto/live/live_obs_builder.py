"""Live Observation Builder — Rolling OHLCV buffer producing training-identical observations.

Reuses the SAME feature functions from multiscale_handler.py:
  _compute_scale_features(), _resample_ohlcv(), _ema_zscore_tanh(), _symlog()

This guarantees exact numerical parity between training and live observations.

Design:
    1. Bootstrap: fetch historical 1-min bars via CryptoLoader to warm EMAs
    2. Per-bar update: append new 1-min bars, recompute features when scale bars complete
    3. get_observation(): return obs dict matching ContinuousSwingEnv._get_observation()

The full recompute strategy (vs incremental EMA) eliminates numerical drift risk.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from finrl_pro_ds.data.multiscale_handler import (
    _compute_scale_features,
    _resample_ohlcv,
)

logger = logging.getLogger(__name__)

# Maximum 1-min bars to retain in the rolling buffer.
# ~30 days of 1-min data. Prevents unbounded memory growth.
_MAX_BUFFER_BARS = 50_000


class LiveObsBuilder:
    """Builds observations identical to MultiScaleOHLCVHandler for live trading.

    Maintains a rolling buffer of 1-min OHLCV bars. On each new bar:
      1. Append bar to buffer
      2. For each scale: if new scale bar completed, recompute features
      3. Return obs dict matching ContinuousSwingEnv format

    LEAK-1 note: In live mode there is no train/val/test split.
    All normalization uses the rolling buffer without cutoff. The
    EMA warmup requires bootstrap_bars of history on startup.
    """

    def __init__(
        self,
        scales: list[int],
        window_size: int = 30,
        norm_span: int = 120,
        n_features: int = 8,
        bootstrap_bars: int = 30_000,
        drift_detection: bool = True,
        drift_window: int = 100,
        obs_mode: str = "window",
        summary_feature_indices: Optional[list[int]] = None,
    ):
        """
        Args:
            scales: Timescale in minutes, e.g. [15, 60, 240].
            window_size: Observation window per scale (matches training).
            norm_span: EMA span for z-score normalization.
            n_features: Features per scale (8 = TC-aligned).
            bootstrap_bars: 1-min bars to pre-load for EMA warmup.
                            Default 30_000 (~20.8 days) ensures warmup
                            for EMA span=120 at 240-min scale.
            drift_detection: Enable EMA drift detection (variance monitoring).
            drift_window: Rolling window size for variance tracking.
            obs_mode: "window" (raw windows) or "summary_stats" (flat summary).
            summary_feature_indices: Feature column indices for summary stats
                (default [0,1,2,6,7] = log_return, atr_norm, parkinson, close_z, volume_z).
        """
        self.scales = sorted(scales)
        self.window_size = window_size
        self.norm_span = norm_span
        self.n_features = n_features
        self.bootstrap_bars = bootstrap_bars
        self.obs_mode = obs_mode
        self.summary_feature_indices = summary_feature_indices or [0, 1, 2, 6, 7]

        # Rolling 1-min OHLCV buffer (DataFrame)
        self._buffer_1min: Optional[pd.DataFrame] = None

        # Per-scale feature arrays and metadata
        self._scale_features: dict[int, np.ndarray] = {}
        self._scale_dfs: dict[int, pd.DataFrame] = {}
        self._scale_last_bar_count: dict[int, int] = {}

        # Base scale = finest (smallest)
        self._base_scale = self.scales[0]

        # ATR state (rolling 14-bar on base scale)
        self._base_atr_values: Optional[np.ndarray] = None
        self._atr_buffer: list[float] = []
        self._atr_rolling_mean: float = 0.0

        self._bootstrapped = False

        # EMA convergence: need ~3x norm_span bars per scale for 95% convergence
        self._ema_convergence_factor = 3

        # ---- Drift detection state ----
        self._drift_detection_enabled = drift_detection
        self._drift_window = drift_window
        # Per-scale deque of variance values, one entry per update() call
        self._variance_history: dict[int, deque[float]] = {}
        self._drift_detected = False
        # Thresholds: current variance vs rolling median
        self._drift_flat_threshold = 0.01   # <1% of median = features going flat
        self._drift_explode_threshold = 100.0  # >100x median = explosion

    # -------------------------------------------------------------------
    # Warmup quality
    # -------------------------------------------------------------------
    def compute_min_bootstrap_bars(self) -> int:
        """Minimum 1-min bars needed for EMA convergence on all scales.

        For EMA(span=N), ~3*N bars gives ~95% convergence.
        We need this many bars on the COARSEST scale, so multiply by max scale.
        """
        min_scale_bars = self._ema_convergence_factor * self.norm_span + self.window_size
        return int(max(self.scales) * min_scale_bars)

    def get_warmup_quality(self) -> dict[int, float]:
        """Check EMA convergence quality per scale.

        Returns:
            Dict of {scale_minutes: convergence_ratio} where 1.0 = fully converged.
            Values below 1.0 indicate the EMA z-scores are still biased toward
            initialization and not representative of the true running statistics.
        """
        convergence_threshold = self._ema_convergence_factor * self.norm_span
        result = {}
        for scale in self.scales:
            n_bars = len(self._scale_features.get(scale, []))
            result[scale] = min(n_bars / convergence_threshold, 1.0) if convergence_threshold > 0 else 1.0
        return result

    # -------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------
    async def bootstrap(
        self,
        loader,
        asset: str,
    ) -> None:
        """Fetch historical 1-min bars and initialize feature buffers.

        Args:
            loader: CryptoLoader instance (with async fetch_ohlcv).
            asset: Base asset symbol (e.g., "BTC").
        """
        from datetime import datetime, timezone

        logger.info(
            f"LiveObsBuilder bootstrapping: fetching {self.bootstrap_bars} "
            f"1-min bars for {asset}...",
        )

        # FIX AUD-C01: Use correct CryptoLoader.fetch_ohlcv() signature.
        # Compute start/end from bootstrap_bars instead of passing limit.
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.bootstrap_bars + 60)  # Small buffer

        for attempt in range(3):
            try:
                df = await loader.fetch_ohlcv(
                    assets=[asset],
                    start=start.isoformat(),
                    end=end.isoformat(),
                    timeframe="1m",
                )
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Bootstrap fetch attempt {attempt+1} failed: {e}, retrying...")
                    import asyncio
                    await asyncio.sleep(5.0 * (attempt + 1))
                else:
                    raise RuntimeError(f"Bootstrap failed after 3 attempts: {e}") from e

        if df is None or len(df) < self.window_size * max(self.scales):
            raise RuntimeError(
                f"Insufficient bootstrap data: got {len(df) if df is not None else 0} bars, "
                f"need at least {self.window_size * max(self.scales)}",
            )

        # FIX AUD-C02: Filter to single asset if multi-asset data returned
        if 'ticker' in df.columns:
            if df['ticker'].nunique() > 1:
                logger.warning(f"Multi-asset data returned, filtering to {asset}")
            df = df[df['ticker'] == asset].drop(columns=['ticker'])

        self._init_from_dataframe(df)
        logger.info(
            f"LiveObsBuilder bootstrapped: {len(self._buffer_1min)} 1-min bars, "
            f"scales={self.scales}, features ready",
        )

    def bootstrap_from_dataframe(self, df_1min: pd.DataFrame) -> None:
        """Bootstrap from an existing DataFrame (for testing / offline use).

        Args:
            df_1min: DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        self._init_from_dataframe(df_1min)

    def _init_from_dataframe(self, df: pd.DataFrame) -> None:
        """Initialize all internal state from a 1-min DataFrame."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()

        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df = df.reset_index()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df[['open', 'high', 'low', 'close', 'volume']] = (
            df[['open', 'high', 'low', 'close', 'volume']].ffill()
        )
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Store rolling buffer
        self._buffer_1min = df

        # Compute features for each scale
        self._recompute_all_scales()

        # Compute ATR on base scale
        self._recompute_base_atr()

        # Initialize ATR rolling stats
        if self._base_atr_values is not None and len(self._base_atr_values) > 0:
            recent_atr = self._base_atr_values[-200:]
            self._atr_buffer = list(recent_atr)
            self._atr_rolling_mean = float(np.mean(recent_atr))

        self._bootstrapped = True

        # Log warmup quality per scale
        warmup_quality = self.get_warmup_quality()
        min_bootstrap = self.compute_min_bootstrap_bars()
        for scale, quality in sorted(warmup_quality.items()):
            n_bars = len(self._scale_features.get(scale, []))
            needed = self._ema_convergence_factor * self.norm_span
            if quality < 1.0:
                logger.warning(
                    f"WARMUP INCOMPLETE — scale {scale}min: {quality:.0%} converged "
                    f"({n_bars}/{needed} bars). Agent features on this scale are "
                    f"unreliable. Increase bootstrap_bars to >= {min_bootstrap}.",
                )
            else:
                logger.info(
                    f"Warmup OK — scale {scale}min: {n_bars} bars "
                    f"(need {needed}, {quality:.0%} converged)",
                )

    def _recompute_all_scales(self) -> None:
        """Resample and compute features for every scale."""
        for scale in self.scales:
            resampled = _resample_ohlcv(self._buffer_1min, scale)
            # No norm_cutoff in live mode (LEAK-1 N/A — no splits in live)
            features = _compute_scale_features(
                resampled, norm_cutoff_idx=None, span=self.norm_span,
                n_features=self.n_features,
            )
            self._scale_dfs[scale] = resampled
            self._scale_features[scale] = features
            self._scale_last_bar_count[scale] = len(resampled)

    def _recompute_base_atr(self) -> None:
        """Compute ATR(14) on the base scale for position capping."""
        base_df = self._scale_dfs.get(self._base_scale)
        if base_df is None or len(base_df) < 2:
            return

        close = base_df['close'].values.astype(np.float64)
        high = base_df['high'].values.astype(np.float64)
        low = base_df['low'].values.astype(np.float64)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        self._base_atr_values = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # -------------------------------------------------------------------
    # Drift detection
    # -------------------------------------------------------------------
    @property
    def drift_detected(self) -> bool:
        """True if any scale has anomalous feature variance (flat or exploding)."""
        return self._drift_detected

    def _check_drift(self) -> bool:
        """Check per-scale feature variance against rolling median.

        Appends the current variance of the latest feature row for each scale
        to a rolling window. If the current variance drops below 1% of the
        rolling median (features going constant/stale) or exceeds 100x the
        rolling median (numerical explosion), logs a WARNING and returns True.

        Returns:
            True if drift detected on ANY scale, False otherwise.
        """
        if not self._drift_detection_enabled:
            return False

        any_drift = False

        for scale in self.scales:
            features = self._scale_features.get(scale)
            if features is None or len(features) == 0:
                continue

            # Variance of the latest feature row (across all feature columns)
            latest_row = features[-1]
            current_var = float(np.var(latest_row))

            # Initialize deque on first call for this scale
            if scale not in self._variance_history:
                self._variance_history[scale] = deque(maxlen=self._drift_window)

            history = self._variance_history[scale]
            history.append(current_var)

            # Need at least 10 samples to establish a baseline
            if len(history) < 10:
                continue

            rolling_median = float(np.median(list(history)))

            # Guard against zero median (all-constant features from the start)
            if rolling_median < 1e-15:
                # If median is ~0 and current is also ~0, no drift.
                # If median is ~0 but current is nonzero, that's a regime change, not drift.
                continue

            variance_ratio = current_var / rolling_median

            if variance_ratio < self._drift_flat_threshold:
                logger.warning(
                    "EMA drift detected — scale %dmin: feature variance FLAT "
                    "(ratio=%.4f, current=%.2e, median=%.2e). "
                    "Possible stale or corrupted data feed.",
                    scale, variance_ratio, current_var, rolling_median,
                )
                any_drift = True
            elif variance_ratio > self._drift_explode_threshold:
                logger.warning(
                    "EMA drift detected — scale %dmin: feature variance EXPLOSION "
                    "(ratio=%.1f, current=%.2e, median=%.2e). "
                    "Possible data corruption or extreme market event.",
                    scale, variance_ratio, current_var, rolling_median,
                )
                any_drift = True

        self._drift_detected = any_drift
        return any_drift

    # -------------------------------------------------------------------
    # Live update
    # -------------------------------------------------------------------
    def update(self, bars_1min: list[dict] | pd.DataFrame) -> None:
        """Append new 1-min bars and recompute features for completed scale bars.

        Args:
            bars_1min: New 1-min bars as list of dicts or DataFrame.
                       Each dict: {timestamp, open, high, low, close, volume}
        """
        if not self._bootstrapped:
            raise RuntimeError("Call bootstrap() or bootstrap_from_dataframe() first")

        if isinstance(bars_1min, pd.DataFrame):
            new_df = bars_1min.copy()
        else:
            new_df = pd.DataFrame(bars_1min)

        if len(new_df) == 0:
            return

        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

        # Append to buffer
        self._buffer_1min = pd.concat(
            [self._buffer_1min, new_df], ignore_index=True,
        )

        # Trim buffer to prevent unbounded growth
        if len(self._buffer_1min) > _MAX_BUFFER_BARS:
            trim = len(self._buffer_1min) - _MAX_BUFFER_BARS
            self._buffer_1min = self._buffer_1min.iloc[trim:].reset_index(drop=True)

        # Full recompute (guarantees parity, <10ms for rolling buffer)
        self._recompute_all_scales()
        self._recompute_base_atr()

        # Update ATR rolling stats
        if self._base_atr_values is not None and len(self._base_atr_values) > 0:
            current_atr = float(self._base_atr_values[-1])
            self._atr_buffer.append(current_atr)
            if len(self._atr_buffer) > 200:
                self._atr_buffer = self._atr_buffer[-200:]
            self._atr_rolling_mean = float(np.mean(self._atr_buffer))

        # Drift detection — lightweight variance check per scale
        self._check_drift()

    # -------------------------------------------------------------------
    # Observation construction
    # -------------------------------------------------------------------
    def get_observation(
        self,
        current_position: float,
        prev_close: float,
        current_close: float,
        timestamp: Optional[datetime] = None,
    ) -> dict[str, np.ndarray]:
        """Build observation dict matching ContinuousSwingEnv._get_observation().

        Args:
            current_position: Current position fraction [-1, 1].
            prev_close: Previous bar's close price.
            current_close: Current bar's close price.
            timestamp: Optional override. If None, uses the latest bar timestamp
                       from the buffer (recommended for parity with training).

        Returns:
            Dict with keys (window mode):
                scale_0: (window_size, n_features) float32
                scale_1: (window_size, n_features) float32
                ...
                private: (5,) float32
            Dict with keys (summary_stats mode):
                scale_0: (n_selected * 3,) float32
                scale_1: (n_selected * 3,) float32
                ...
                private: (5,) float32
        """
        if not self._bootstrapped:
            raise RuntimeError("Call bootstrap() or bootstrap_from_dataframe() first")

        obs = {}

        # Scale features — last window_size rows from each scale
        for i, scale in enumerate(self.scales):
            features = self._scale_features[scale]
            n = len(features)

            if n >= self.window_size:
                window = features[n - self.window_size: n]
            else:
                # Pad with first row
                pad_len = self.window_size - n
                pad = np.tile(features[0:1], (pad_len, 1))
                window = np.concatenate([pad, features], axis=0)

            if self.obs_mode == "summary_stats":
                obs[f"scale_{i}"] = self._compute_summary_stats(window)
            else:
                obs[f"scale_{i}"] = window.copy()

        # FIX AUD-C04: Use bar timestamp from buffer for time encoding parity.
        # The training env uses handler._base_timestamps[ptr-1], not an external clock.
        if timestamp is None:
            bar_ts = self.get_latest_timestamp()
            if bar_ts is not None:
                timestamp = bar_ts.to_pydatetime()
            else:
                timestamp = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

        # Private state: [position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]
        obs["private"] = self._build_private_state(
            current_position, prev_close, current_close, timestamp,
        )

        return obs

    def _compute_summary_stats(self, window: np.ndarray) -> np.ndarray:
        """Compute (mean, std, last) for selected feature columns over window.

        Mirrors MultiScaleOHLCVHandler._compute_summary_stats() exactly.
        Returns: flat (n_selected * 3,) float32 array.
        """
        selected = window[:, self.summary_feature_indices]  # (W, n_selected)
        means = selected.mean(axis=0)
        stds = selected.std(axis=0)
        stds = np.where(stds < 1e-8, 0.0, stds)
        last = selected[-1]
        return np.concatenate([means, stds, last]).astype(np.float32)

    def _build_private_state(
        self,
        current_position: float,
        prev_close: float,
        current_close: float,
        timestamp: datetime,
    ) -> np.ndarray:
        """Build 5-dim private state vector.

        Replicates ContinuousSwingEnv._get_private_state() exactly.
        """
        # 1. Current position [-1, 1]
        pos = float(current_position)

        # 2. Unrealized PnL proxy (recent return * position, clipped)
        pnl_proxy = 0.0
        if prev_close > 0 and current_close > 0:
            ret_bps = (current_close - prev_close) / prev_close * 10000.0
            pnl_proxy = float(np.clip(current_position * ret_bps / 100.0, -1.0, 1.0))

        # 3-4. Time encoding (minutes since midnight → sin/cos)
        if isinstance(timestamp, datetime):
            minutes = timestamp.hour * 60 + timestamp.minute
        else:
            # numpy datetime64 or pandas Timestamp
            ts = pd.Timestamp(timestamp)
            minutes = ts.hour * 60 + ts.minute

        time_sin = float(np.sin(2 * np.pi * minutes / 1440.0))
        time_cos = float(np.cos(2 * np.pi * minutes / 1440.0))

        # 5. ATR ratio (current / rolling mean, clipped and scaled)
        current_atr = self.get_current_atr()
        if self._atr_rolling_mean > 1e-12:
            atr_ratio = float(np.clip(current_atr / self._atr_rolling_mean, 0.0, 3.0)) / 3.0
        else:
            atr_ratio = 0.5

        return np.array(
            [pos, pnl_proxy, time_sin, time_cos, atr_ratio],
            dtype=np.float32,
        )

    # -------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------
    def get_current_atr(self) -> float:
        """Get the most recent ATR value on the base scale."""
        if self._base_atr_values is not None and len(self._base_atr_values) > 0:
            return float(self._base_atr_values[-1])
        return 0.0

    def get_current_close(self) -> float:
        """Get the most recent close price on the base scale."""
        base_df = self._scale_dfs.get(self._base_scale)
        if base_df is not None and len(base_df) > 0:
            return float(base_df['close'].iloc[-1])
        return 0.0

    def get_latest_timestamp(self) -> Optional[pd.Timestamp]:
        """Get timestamp of the most recent base-scale bar."""
        base_df = self._scale_dfs.get(self._base_scale)
        if base_df is not None and len(base_df) > 0:
            return pd.Timestamp(base_df['timestamp'].iloc[-1])
        return None

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def is_ready(self) -> bool:
        return self._bootstrapped and all(
            len(self._scale_features.get(s, [])) >= self.window_size
            for s in self.scales
        )
