"""
Multi-Scale OHLCV Data Handler

Reads 1-min OHLCV parquet, resamples to multiple timescales (configurable),
computes 7 features per scale, and provides a stepping interface for the env.

Features per scale (8 dims, TC-aligned):
  1. log_return = log(close_t / close_{t-1})
  2. atr_norm = ATR(14) / EMA(ATR, 50), centered around 1.0
  3. parkinson_vol = sqrt(log(H/L)^2 / (4·ln2))
  4-7. open_z, high_z, low_z, close_z = SymLog → EMA-Z(span=120) → tanh
  8. volume_z = SymLog → EMA-Z(span=120) → tanh (market participation)

LEAK-1 compliant: norm_cutoff_date splits normalization.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _symlog(x: np.ndarray) -> np.ndarray:
    """SymLog transform: sign(x) * log(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _ema_zscore_tanh(values: np.ndarray, span: int = 120) -> np.ndarray:
    """EMA Z-Score with causal shift → tanh soft-clip. Returns float32 array."""
    s = pd.Series(values.astype(np.float64))
    ema_mean = s.ewm(span=span, adjust=False).mean().shift(1)
    ema_std = s.ewm(span=span, adjust=False).std().shift(1)

    mu = ema_mean.values.copy()
    sigma = ema_std.values.copy()

    # Backward-fill pre-warmup rows
    first_valid = np.where(~np.isnan(mu))[0]
    if len(first_valid) > 0:
        fv = first_valid[0]
        mu[:fv] = mu[fv]
        sigma[:fv] = sigma[fv]
    else:
        mu[np.isnan(mu)] = 0.0

    sigma = np.where(np.isnan(sigma) | (sigma < 1e-8), 1.0, sigma)
    z = (values.astype(np.float64) - mu) / sigma
    return np.tanh(z * 0.5).astype(np.float32)


def _resample_ohlcv(df_1min: pd.DataFrame, scale_minutes: int) -> pd.DataFrame:
    """Standard OHLCV aggregation: first O, max H, min L, last C, sum V."""
    df = df_1min.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    freq = f'{scale_minutes}min'
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open', 'high', 'low', 'close'])  # FIX R4-AUD-06: Drop bars with any NaN OHLC
    resampled.index.name = 'timestamp'
    return resampled.reset_index()


def _compute_scale_features(df: pd.DataFrame, norm_cutoff_idx: Optional[int] = None, span: int = 120, n_features: int = 8) -> np.ndarray:
    """Compute features for a single timescale DataFrame.

    Features (8 dims, TC-aligned for Conv1d Tensor Core acceleration):
      0. log_return
      1. atr_norm (centered around 0)
      2. parkinson_vol
      3-6. open_z, high_z, low_z, close_z (SymLog → EMA-Z → tanh)
      7. volume_z (SymLog → EMA-Z → tanh) — market participation signal

    Returns: (N, n_features) float32 array
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(len(close))
    n = len(close)

    features = np.zeros((n, n_features), dtype=np.float32)

    # 1. log_return
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    safe_prev = np.where(prev_close > 0, prev_close, 1e-9)
    log_ret = np.log(close / safe_prev)
    log_ret[0] = 0.0
    features[:, 0] = np.clip(log_ret, -0.1, 0.1).astype(np.float32)

    # 2. atr_norm = ATR(14) / EMA(ATR, 50)
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    tr[0] = high[0] - low[0] if high[0] > low[0] else 0.0
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_ema_50 = pd.Series(atr_14).ewm(span=50, adjust=False).mean().values
    atr_ema_safe = np.where(atr_ema_50 > 1e-9, atr_ema_50, 1.0)
    atr_norm = atr_14 / atr_ema_safe
    features[:, 1] = np.clip(atr_norm - 1.0, -2.0, 2.0).astype(np.float32)  # Center around 0

    # 3. parkinson_vol
    hl_ratio = np.where(low > 0, high / low, 1.0)
    parkinson = np.sqrt(np.log(hl_ratio) ** 2 / (4.0 * np.log(2.0)))
    features[:, 2] = np.clip(parkinson, 0.0, 0.1).astype(np.float32)

    # 4-7. OHLC z-scores: SymLog → EMA-Z → tanh
    # FIX MATH-N03: Carry forward warm-up buffer across norm cutoff so EMA-Z
    # converges before genuine post-cutoff data begins (matching parquet_handler.py).
    WARMUP_BUFFER = 200  # ~1.7 half-lives for span=120 EMA

    def normalize_series(arr):
        if norm_cutoff_idx is not None and 0 < norm_cutoff_idx < len(arr):
            part1 = _ema_zscore_tanh(_symlog(arr[:norm_cutoff_idx]), span)
            # Carry forward last WARMUP_BUFFER rows from pre-cutoff as warm-up
            buffer_size = min(WARMUP_BUFFER, norm_cutoff_idx)
            arr_after_with_buffer = arr[norm_cutoff_idx - buffer_size:]
            part2_full = _ema_zscore_tanh(_symlog(arr_after_with_buffer), span)
            # Strip buffer rows — only keep genuine post-cutoff output
            part2 = part2_full[buffer_size:]
            return np.concatenate([part1, part2])
        return _ema_zscore_tanh(_symlog(arr), span)

    features[:, 3] = normalize_series(open_)
    features[:, 4] = normalize_series(high)
    features[:, 5] = normalize_series(low)
    features[:, 6] = normalize_series(close)

    # 8. volume_z — market participation / liquidity signal
    # TC-OPT: 8th feature aligns Conv1d input channels to multiple of 8
    if n_features >= 8:
        features[:, 7] = normalize_series(volume)

    return features


class MultiScaleOHLCVHandler:
    """Multi-scale OHLCV data handler for ContinuousSwingEnv.

    Reads 1-min OHLCV, resamples to multiple scales, computes features,
    and provides a step-by-step interface aligned to the base (finest) scale.

    Interface:
        reset() → resets pointer
        step() → dict with scale arrays, close price, ATR
    """

    def __init__(
        self,
        file_path: str,
        ticker: str,
        feature_config: Dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        norm_cutoff_date: Optional[str] = None,
    ):
        self.file_path = file_path
        self.ticker = ticker
        self.scales = feature_config.get("scales", [3, 15, 60])
        self.window_size = feature_config.get("window_size", 30)
        self.norm_span = feature_config.get("norm_span", 120)

        # v6: Summary-stats observation mode (725→50 dims)
        self.obs_mode = feature_config.get("obs_mode", "window")
        self.summary_feature_indices = feature_config.get(
            "summary_feature_indices", [0, 1, 2, 6, 7]
        )  # log_return, atr_norm, parkinson_vol, close_z, volume_z

        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.norm_cutoff_date = pd.to_datetime(norm_cutoff_date) if norm_cutoff_date else None

        self._ptr = 0

        # Load and process data
        self._load_data()

    def _load_data(self):
        """Load 1-min parquet, resample to each scale, compute features."""
        try:
            df = pd.read_parquet(self.file_path, engine='fastparquet')
        except Exception:
            df_raw = pd.read_parquet(self.file_path, engine='pyarrow')
            df = pd.DataFrame({col: np.array(df_raw[col].values, copy=True) for col in df_raw.columns})
            del df_raw

        df.columns = df.columns.astype(str).str.strip()

        # Ensure timestamp column
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df = df.reset_index()
        if 'timestamp' not in df.columns:
            raise RuntimeError(f"No timestamp column in {self.file_path}")

        ts = pd.to_datetime(df['timestamp'], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df['timestamp'] = ts

        # Apply end_date filter before feature computation (safe — no future leakage)
        if self.end_date is not None:
            df = df[df['timestamp'] <= self.end_date]

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure required OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Forward-fill gaps
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].ffill()

        self._df_1min = df

        # Base scale = smallest in scales list
        base_scale = min(self.scales)

        # Resample each scale and compute features on FULL history (warm EMAs)
        self._scale_dfs = {}
        self._scale_features = {}
        self._scale_timestamps = {}

        for scale in self.scales:
            resampled = _resample_ohlcv(df, scale)

            # Compute norm cutoff index for this scale
            norm_cutoff_idx = None
            if self.norm_cutoff_date is not None:
                cutoff_mask = resampled['timestamp'] >= self.norm_cutoff_date
                if cutoff_mask.any():
                    norm_cutoff_idx = cutoff_mask.idxmax()

            features = _compute_scale_features(resampled, norm_cutoff_idx, self.norm_span)

            # FIX GMGP1-F1: Apply start_date AFTER feature computation
            # so EMAs are warm when the environment window begins
            if self.start_date is not None:
                start_mask = resampled['timestamp'] >= self.start_date
                if start_mask.any():
                    trim_idx = start_mask.idxmax()
                    # Keep window_size extra bars before start_date for obs history
                    trim_idx = max(0, trim_idx - self.window_size)
                    resampled = resampled.iloc[trim_idx:].reset_index(drop=True)
                    features = features[trim_idx:]

            self._scale_dfs[scale] = resampled
            self._scale_features[scale] = features
            self._scale_timestamps[scale] = resampled['timestamp'].values

        # Verify sufficient data after filtering
        base_features = self._scale_features[base_scale]
        if len(base_features) < self.window_size + 10:
            raise ValueError(
                f"Insufficient data after date filtering: {len(base_features)} bars "
                f"(need at least {self.window_size + 10})"
            )

        # Base scale data for stepping
        self._base_scale = base_scale
        base_df = self._scale_dfs[base_scale]
        self._base_close = base_df['close'].values.astype(np.float64)
        self._base_high = base_df['high'].values.astype(np.float64)
        self._base_low = base_df['low'].values.astype(np.float64)
        self._base_volume = base_df['volume'].values.astype(np.float64)
        self._base_timestamps = base_df['timestamp'].values

        # Precompute ATR on base scale for position capping
        prev_close = np.roll(self._base_close, 1)
        prev_close[0] = self._base_close[0]
        tr = np.maximum(
            self._base_high - self._base_low,
            np.maximum(
                np.abs(self._base_high - prev_close),
                np.abs(self._base_low - prev_close)
            )
        )
        self._base_atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # Build mapping: for each base-scale bar, which index in each coarser scale?
        self._scale_index_map = {}
        for scale in self.scales:
            if scale == base_scale:
                # 1:1 mapping
                self._scale_index_map[scale] = np.arange(len(self._scale_features[scale]))
            else:
                # Map base timestamps to coarser scale indices
                coarse_ts = self._scale_timestamps[scale].astype('int64')
                base_ts = self._base_timestamps.astype('int64')
                indices = np.searchsorted(coarse_ts, base_ts, side='right') - 1
                indices = np.clip(indices, 0, len(coarse_ts) - 1)
                self._scale_index_map[scale] = indices

        self._len = len(self._base_close)
        self._ptr = self.window_size  # Start after initial window

        logger.info(
            f"MultiScaleOHLCVHandler loaded: {self._len} base bars ({base_scale}min), "
            f"scales={self.scales}, window={self.window_size}"
        )

    def reset(self):
        """Reset pointer to start."""
        self._ptr = self.window_size

    def step(self) -> Optional[Dict]:
        """Advance one base-scale bar, return multi-scale obs.

        Returns:
            dict with keys:
                scale_0: (window_size, 7) float32  (finest scale)
                scale_1: (window_size, 7) float32
                scale_2: (window_size, 7) float32  (coarsest scale)
                close: float
                atr: float
                timestamp: numpy datetime64
            or None if data exhausted
        """
        if self._ptr >= self._len:
            return None

        result = {}

        # For each scale, get the window of features
        for i, scale in enumerate(self.scales):
            features = self._scale_features[scale]
            if scale == self._base_scale:
                idx = self._ptr
            else:
                idx = self._scale_index_map[scale][self._ptr]

            # Window: [idx - window_size + 1, idx + 1)
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = features[start:end]

            # Pad if insufficient history
            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                pad = np.tile(window[0:1], (pad_len, 1))
                window = np.concatenate([pad, window], axis=0)

            key = f"scale_{i}"
            if self.obs_mode == "summary_stats":
                result[key] = self._compute_summary_stats(window)
            else:
                result[key] = window.copy()

        result["close"] = float(self._base_close[self._ptr])
        result["atr"] = float(self._base_atr[self._ptr])
        result["timestamp"] = self._base_timestamps[self._ptr]

        self._ptr += 1
        return result

    def _compute_summary_stats(self, window: np.ndarray) -> np.ndarray:
        """Compute (mean, std, last) for selected feature columns over window.

        v6 AlphaSeek-informed: reduce (W, 8) window → flat (n_selected * 3,) vector.
        Default indices [0,1,2,6,7] = log_return, atr_norm, parkinson_vol, close_z, volume_z.
        Drops redundant open_z(3), high_z(4), low_z(5).

        Returns: flat (n_features * 3,) float32 array
        """
        selected = window[:, self.summary_feature_indices]  # (W, n_selected)
        means = selected.mean(axis=0)
        stds = selected.std(axis=0)
        stds = np.where(stds < 1e-8, 0.0, stds)  # Zero out near-zero std
        last = selected[-1]
        return np.concatenate([means, stds, last]).astype(np.float32)

    def get_lookahead_volatility(self, horizon: int = 100) -> Optional[float]:
        """Lookahead volatility for auxiliary loss (same interface as ParquetDataHandler)."""
        if self._ptr + horizon >= self._len:
            return None
        future_close = self._base_close[self._ptr:self._ptr + horizon]
        returns = np.diff(np.log(np.where(future_close > 0, future_close, 1e-9)))
        if len(returns) < 2:
            return None
        return float(np.std(returns))
