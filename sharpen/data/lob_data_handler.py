"""
LOB Data Handler for Market Making Environment.

Loads aggregated LOB parquet (10s bars from lob_aggregate.py) and provides:
  - OHLCV features computed directly (no resampling — data is already at target resolution)
  - LOB microstructure features as a separate observation channel
  - Segment-aware stepping (never crosses data gaps)
  - Full bar data for fill model (open, high, low, close, volume)

Unlike the parent MultiScaleOHLCVHandler which resamples 1-min data to multiple
timescales, this handler treats the input parquet as pre-aggregated bars and computes
features directly — no _resample_ohlcv call. This avoids AUD-01 (10s→1min destruction).

LOB feature channels (8 dims, TC-aligned):
  0. bbo_imbalance           — raw [-1, 1], no normalization needed
  1. depth_imbalance_5       — raw [-1, 1]
  2. depth_ratio_5_centered  — depth_ratio_5 - 0.5, centered at 0
  3. bbo_bid_qty_z           — EMA-Z-tanh normalized
  4. bbo_ask_qty_z           — EMA-Z-tanh normalized
  5. spread_z                — EMA-Z-tanh normalized
  6. n_bbo_changes_z         — EMA-Z-tanh normalized (activity proxy)
  7. microprice_offset       — raw, already in bps scale, clipped
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from sharpen.data.multiscale_handler import (
    MultiScaleOHLCVHandler,
    _compute_scale_features,
    _ema_zscore_tanh,
    _symlog,
)

logger = logging.getLogger(__name__)

N_LOB_FEATURES = 8


class LOBDataHandler(MultiScaleOHLCVHandler):
    """LOB-aware data handler for MarketMakingEnv.

    Fully overrides _load_data() to:
      - Skip _resample_ohlcv (data is already at target resolution)
      - Compute LOB microstructure features
      - Track segment boundaries for episode management
    """

    def __init__(
        self,
        file_path: str,
        ticker: str,
        feature_config: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        norm_cutoff_date: Optional[str] = None,
    ):
        self._n_lob_features = feature_config.get("n_lob_features", N_LOB_FEATURES)
        self._lob_norm_span = feature_config.get("lob_norm_span", 120)
        self._has_lob = False

        # Parent __init__ calls _load_data() which we fully override
        super().__init__(
            file_path=file_path,
            ticker=ticker,
            feature_config=feature_config,
            start_date=start_date,
            end_date=end_date,
            norm_cutoff_date=norm_cutoff_date,
        )

    def _load_data(self):
        """Load LOB parquet directly — NO resampling.

        Overrides parent entirely to avoid _resample_ohlcv which would
        aggregate 10s bars into 1-min bars (AUD-01 fix).
        """
        # ---- Read parquet ----
        try:
            df = pd.read_parquet(self.file_path, engine="fastparquet")
        except Exception:
            df_raw = pd.read_parquet(self.file_path, engine="pyarrow")
            df = pd.DataFrame({
                col: np.array(df_raw[col].values, copy=True) for col in df_raw.columns
            })
            del df_raw

        df.columns = df.columns.astype(str).str.strip()

        # ---- Ensure timestamp column (tz-naive) ----
        if "timestamp" not in df.columns and df.index.name == "timestamp":
            df = df.reset_index()
        if "timestamp" not in df.columns:
            raise RuntimeError(f"No timestamp column in {self.file_path}")

        ts = pd.to_datetime(df["timestamp"], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df["timestamp"] = ts

        # ---- Detect LOB columns ----
        has_lob = "segment_id" in df.columns and "bbo_imbalance" in df.columns
        if has_lob:
            logger.info("LOBDataHandler: detected LOB parquet with microstructure features")
            self._has_lob = True
        else:
            logger.info("LOBDataHandler: standard OHLCV parquet (no LOB features)")
            self._has_lob = False

        # ---- Ensure OHLCV columns ----
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ---- Apply end_date filter ----
        if self.end_date is not None:
            df = df[df["timestamp"] <= self.end_date]

        df = df.sort_values("timestamp").reset_index(drop=True)
        df[["open", "high", "low", "close", "volume"]] = (
            df[["open", "high", "low", "close", "volume"]].ffill()
        )

        # ---- Compute OHLCV features on the FULL data (no resampling) ----
        # Norm cutoff index
        norm_cutoff_idx = None
        if self.norm_cutoff_date is not None:
            cutoff_mask = df["timestamp"] >= self.norm_cutoff_date
            if cutoff_mask.any():
                norm_cutoff_idx = cutoff_mask.idxmax()

        features = _compute_scale_features(df, norm_cutoff_idx, self.norm_span)

        # ---- Apply start_date trim AFTER feature computation (warm EMAs) ----
        trim_start = 0
        if self.start_date is not None:
            start_mask = df["timestamp"] >= self.start_date
            if start_mask.any():
                trim_idx = start_mask.idxmax()
                trim_start = max(0, trim_idx - self.window_size)

        if trim_start > 0:
            df = df.iloc[trim_start:].reset_index(drop=True)
            features = features[trim_start:]

        # ---- Store as single-scale (no multi-scale resampling needed) ----
        base_scale = min(self.scales)
        self._base_scale = base_scale
        self._scale_dfs = {base_scale: df}
        self._scale_features = {base_scale: features}
        self._scale_timestamps = {base_scale: df["timestamp"].values}
        self._scale_index_map = {base_scale: np.arange(len(features))}

        # Base arrays for stepping + fill model
        self._base_close = df["close"].values.astype(np.float64)
        self._base_high = df["high"].values.astype(np.float64)
        self._base_low = df["low"].values.astype(np.float64)
        self._base_open = df["open"].values.astype(np.float64)
        self._base_volume = df["volume"].values.astype(np.float64)
        self._base_timestamps = df["timestamp"].values

        # ATR
        prev_close = np.roll(self._base_close, 1)
        prev_close[0] = self._base_close[0]
        tr = np.maximum(
            self._base_high - self._base_low,
            np.maximum(
                np.abs(self._base_high - prev_close),
                np.abs(self._base_low - prev_close),
            ),
        )
        self._base_atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        self._len = len(self._base_close)
        self._ptr = self.window_size

        # ---- LOB features ----
        if self._has_lob:
            self._compute_lob_features(df, norm_cutoff_idx, trim_start)
        else:
            self._lob_features = None
            self._trimmed_segment_ids = None
            self._trimmed_boundaries = []

        logger.info(
            f"LOBDataHandler loaded: {self._len} bars, "
            f"LOB={'YES' if self._has_lob else 'NO'}, "
            f"segments={len(self._trimmed_boundaries) + 1 if self._has_lob else 0}, "
            f"window={self.window_size}",
        )

    def _compute_lob_features(
        self, df: pd.DataFrame, norm_cutoff_idx: Optional[int], trim_start: int,
    ):
        """Compute normalized LOB features aligned to the (already trimmed) base scale.

        Args:
            df: The trimmed DataFrame (same rows as base scale features).
            norm_cutoff_idx: Cutoff index in the PRE-trim DataFrame.
            trim_start: Number of rows trimmed from the start.
        """
        span = self._lob_norm_span
        n = len(df)

        # Adjust norm_cutoff_idx for trimming
        adjusted_cutoff = None
        if norm_cutoff_idx is not None:
            adjusted_cutoff = norm_cutoff_idx - trim_start
            if adjusted_cutoff < 0 or adjusted_cutoff >= n:
                adjusted_cutoff = None

        WARMUP_BUFFER = 200

        def normalize_lob(arr):
            if adjusted_cutoff is not None and 0 < adjusted_cutoff < len(arr):
                part1 = _ema_zscore_tanh(_symlog(arr[:adjusted_cutoff].astype(np.float64)), span)
                buf = min(WARMUP_BUFFER, adjusted_cutoff)
                after_buf = arr[adjusted_cutoff - buf:].astype(np.float64)
                part2_full = _ema_zscore_tanh(_symlog(after_buf), span)
                part2 = part2_full[buf:]
                return np.concatenate([part1, part2])
            return _ema_zscore_tanh(_symlog(arr.astype(np.float64)), span)

        lob_features = np.zeros((n, self._n_lob_features), dtype=np.float32)

        # Channel 0: bbo_imbalance — raw [-1, 1]
        lob_features[:, 0] = np.clip(df["bbo_imbalance"].values, -1.0, 1.0)

        # Channel 1: depth_imbalance_5 — raw [-1, 1]
        lob_features[:, 1] = np.clip(df["depth_imbalance_5"].values, -1.0, 1.0)

        # Channel 2: depth_ratio_5 centered — [0,1] → [-1, 1]
        lob_features[:, 2] = np.clip(df["depth_ratio_5"].values - 0.5, -0.5, 0.5) * 2.0

        # Channel 3: bbo_bid_qty normalized
        lob_features[:, 3] = normalize_lob(df["bbo_bid_qty"].values)

        # Channel 4: bbo_ask_qty normalized
        lob_features[:, 4] = normalize_lob(df["bbo_ask_qty"].values)

        # Channel 5: spread normalized
        lob_features[:, 5] = normalize_lob(df["spread_mean"].values)

        # Channel 6: n_bbo_changes normalized (activity)
        lob_features[:, 6] = normalize_lob(df["n_bbo_changes"].values.astype(np.float64))

        # Channel 7: microprice offset — already in bps, clip to [-5, 5] and normalize
        lob_features[:, 7] = np.clip(df["microprice_offset"].values / 5.0, -1.0, 1.0)

        self._lob_features = lob_features

        # Segment tracking (already trimmed — same rows as features)
        if "segment_id" in df.columns:
            self._trimmed_segment_ids = df["segment_id"].values.astype(np.int32)
            self._trimmed_boundaries = list(
                np.where(np.diff(self._trimmed_segment_ids) != 0)[0] + 1,
            )
        else:
            self._trimmed_segment_ids = None
            self._trimmed_boundaries = []

        logger.info(
            f"  LOB features: shape={self._lob_features.shape}, "
            f"boundaries={self._trimmed_boundaries}",
        )

    def get_segment_boundaries(self) -> list:
        """Return list of bar indices where segment changes (episode boundaries)."""
        return self._trimmed_boundaries if hasattr(self, "_trimmed_boundaries") else []

    def get_segment_id(self, idx: int) -> int:
        """Get segment ID for a given bar index."""
        if self._trimmed_segment_ids is not None and 0 <= idx < len(self._trimmed_segment_ids):
            return int(self._trimmed_segment_ids[idx])
        return 0

    def step(self) -> Optional[dict]:
        """Advance one bar. Returns multi-scale obs + OHLCV + LOB features.

        Returns None at data exhaustion or segment boundary (env should reset).
        """
        if self._ptr >= self._len:
            return None

        # AUD-10 fix: Check segment boundary and advance past it
        if self._trimmed_segment_ids is not None:
            if self._ptr > 0 and self._ptr < len(self._trimmed_segment_ids):
                curr_seg = self._trimmed_segment_ids[self._ptr]
                prev_seg = self._trimmed_segment_ids[self._ptr - 1]
                if curr_seg != prev_seg:
                    # Advance past boundary so next reset() + step() works
                    self._ptr += 1
                    return None

        result = super().step()
        if result is None:
            return None

        # _ptr was already incremented by super().step()
        idx = self._ptr - 1

        # OHLCV bar data for fill model
        result["open"] = float(self._base_open[idx])
        result["high"] = float(self._base_high[idx])
        result["low"] = float(self._base_low[idx])
        result["volume"] = float(self._base_volume[idx])

        # LOB features window
        if self._has_lob and self._lob_features is not None:
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = self._lob_features[start:end]

            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                pad = np.tile(window[0:1], (pad_len, 1))
                window = np.concatenate([pad, window], axis=0)

            result["lob_features"] = window.copy()
        else:
            result["lob_features"] = np.zeros(
                (self.window_size, self._n_lob_features), dtype=np.float32,
            )

        # Segment ID
        if self._trimmed_segment_ids is not None and idx < len(self._trimmed_segment_ids):
            result["segment_id"] = int(self._trimmed_segment_ids[idx])
        else:
            result["segment_id"] = 0

        return result

    def reset(self):
        """Reset pointer to start of first usable segment."""
        self._ptr = self.window_size

        # Skip past any segment boundary at the initial position
        if self._trimmed_segment_ids is not None and self._ptr < self._len:
            if self._ptr > 0:
                initial_seg = self._trimmed_segment_ids[self._ptr]
                # Ensure we have at least window_size bars of the SAME segment behind us
                seg_start = self._ptr
                while seg_start > 0 and self._trimmed_segment_ids[seg_start - 1] == initial_seg:
                    seg_start -= 1
                if self._ptr - seg_start < self.window_size:
                    # Not enough history in this segment — advance to next boundary + window_size
                    for boundary in sorted(self._trimmed_boundaries):
                        candidate = boundary + self.window_size
                        if candidate < self._len:
                            self._ptr = candidate
                            break
