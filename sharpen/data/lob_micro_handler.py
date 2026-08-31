"""
LOB Micro Data Handler for Market Making V9.

Loads 1s LOB snapshots, aggregates to N-second bars, and computes
40-dim FeatureFactory micro features for the observation channel.

Key differences from LOBDataHandler (V8, 10s aggregated, 8-dim):
  - Uses raw LOB columns (20-level depth) → FeatureFactory.process_micro() → 40 dims
  - Aggregates 1s snapshots to configurable bar size (default 5s)
  - OHLCV from mid_price movement across bar window (real price range for fill model)
  - Volume from BBO change count (activity proxy, matches lob_aggregate.py)

Data flow:
  1s parquet (3M rows) → aggregate to 5s (~600K rows, OHLCV + last LOB snapshot)
  → column rename (bid_p_0→bid_price_1) → FeatureFactory.process_micro() → 40 dims
  → _compute_scale_features on 5s OHLCV → 8 dims per scale
  → step() yields {scale_0:(W,8), lob:(W,40), close, atr, open, high, low, volume}
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from sharpen.data.feature_engineering import DeepScalperFeatureEngineer as FeatureFactory
from sharpen.data.multiscale_handler import (
    MultiScaleOHLCVHandler,
    _compute_scale_features,
)

logger = logging.getLogger(__name__)

# Number of LOB depth levels to use for FeatureFactory (top 5 of 20 available)
_N_LEVELS = 5
# Number of micro features produced by FeatureFactory at n_levels=5
_N_MICRO_FEATURES = 40


def _remap_lob_columns(df: pd.DataFrame, n_levels: int = _N_LEVELS) -> pd.DataFrame:
    """Rename 1s parquet LOB columns to FeatureFactory convention.

    Parquet uses 0-indexed: bid_p_0, bid_q_0, ask_p_0, ask_q_0
    FeatureFactory uses 1-indexed: bid_price_1, bid_vol_1, ask_price_1, ask_vol_1
    """
    remap = {}
    for i in range(n_levels):
        remap[f"bid_p_{i}"] = f"bid_price_{i + 1}"
        remap[f"bid_q_{i}"] = f"bid_vol_{i + 1}"
        remap[f"ask_p_{i}"] = f"ask_price_{i + 1}"
        remap[f"ask_q_{i}"] = f"ask_vol_{i + 1}"
    return df.rename(columns=remap)


def _aggregate_to_bars(
    df: pd.DataFrame,
    bar_size_s: int = 5,
    n_levels: int = _N_LEVELS,
    min_ticks_per_bar: int = 2,
) -> pd.DataFrame:
    """Aggregate 1s LOB snapshots to N-second bars.

    For each bar:
      - OHLCV from mid_price (real price range for fill model)
      - Volume = count of BBO price changes (activity proxy)
      - Raw LOB columns = LAST snapshot in bar (for FeatureFactory)
      - Segment boundaries preserved (never aggregate across gaps)

    Returns DataFrame with ~N/bar_size_s rows.
    """
    bar_size_ms = bar_size_s * 1000
    df = df.copy()

    # Ensure mid_price exists
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0

    # BBO change detection (per-segment diff to avoid cross-segment contamination)
    df["bbo_changed"] = (
        (df.groupby("segment_id")["best_bid_price"].diff().abs().fillna(0) > 1e-8)
        | (df.groupby("segment_id")["best_ask_price"].diff().abs().fillna(0) > 1e-8)
    ).astype(np.int32)

    # Bar group assignment
    df["bar_group"] = df["timestamp_ms"] // bar_size_ms

    # Build aggregation dict
    agg_dict = {
        "timestamp_ms": ("timestamp_ms", "first"),
        "open": ("mid_price", "first"),
        "high": ("mid_price", "max"),
        "low": ("mid_price", "min"),
        "close": ("mid_price", "last"),
        "n_ticks": ("mid_price", "count"),
        "n_bbo_changes": ("bbo_changed", "sum"),
        # BBO quantities (last snapshot — for volume proxy)
        "best_bid_qty": ("best_bid_qty", "last"),
        "best_ask_qty": ("best_ask_qty", "last"),
        "best_bid_price": ("best_bid_price", "last"),
        "best_ask_price": ("best_ask_price", "last"),
    }

    # Raw LOB columns: take LAST snapshot per bar (for FeatureFactory)
    for i in range(n_levels):
        agg_dict[f"bid_p_{i}"] = (f"bid_p_{i}", "last")
        agg_dict[f"bid_q_{i}"] = (f"bid_q_{i}", "last")
        agg_dict[f"ask_p_{i}"] = (f"ask_p_{i}", "last")
        agg_dict[f"ask_q_{i}"] = (f"ask_q_{i}", "last")

    agg = df.groupby(["segment_id", "bar_group"]).agg(**agg_dict).reset_index()

    # Drop thin bars (partial bars at segment boundaries)
    n_before = len(agg)
    agg = agg[agg["n_ticks"] >= min_ticks_per_bar].reset_index(drop=True)
    if n_before - len(agg) > 0:
        logger.info(f"  Dropped {n_before - len(agg)} bars with <{min_ticks_per_bar} ticks")

    # Volume = total BBO activity (matches lob_aggregate.py convention)
    agg["volume"] = (agg["best_bid_qty"] + agg["best_ask_qty"]) * agg["n_ticks"]

    # Timestamp
    agg["timestamp"] = pd.to_datetime(agg["timestamp_ms"], unit="ms", utc=True)
    agg["timestamp"] = agg["timestamp"].dt.tz_convert(None)

    agg = agg.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"  Aggregated {len(df):,} 1s snapshots → {len(agg):,} {bar_size_s}s bars "
        f"({agg['segment_id'].nunique()} segments)"
    )
    return agg


class LOBMicroDataHandler(MultiScaleOHLCVHandler):
    """1s LOB → N-second bars with 40-dim FeatureFactory micro features.

    Fully overrides _load_data() to:
      1. Load 1s LOB parquet
      2. Aggregate to N-second bars (OHLCV from mid_price, LOB from last snapshot)
      3. Remap column names for FeatureFactory
      4. Compute 40-dim micro features via FeatureFactory.process_micro()
      5. Compute 8-dim OHLCV scale features via _compute_scale_features()
      6. Track segment boundaries for episode management
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
        self._bar_size_s = int(feature_config.get("bar_size_s", 5))
        self._n_levels = int(feature_config.get("n_levels", _N_LEVELS))
        self._n_micro_features = _N_MICRO_FEATURES
        self._has_lob = False
        self._asset_class = feature_config.get("asset_class", "crypto")

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
        """Load 1s LOB parquet → aggregate → compute features."""
        # ---- 1. Read 1s parquet (only columns we need to save memory) ----
        logger.info(f"LOBMicroDataHandler: loading {self.file_path}...")
        needed_cols = (
            ["timestamp_ms", "segment_id", "mid_price",
             "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]
            + [f"bid_p_{i}" for i in range(self._n_levels)]
            + [f"bid_q_{i}" for i in range(self._n_levels)]
            + [f"ask_p_{i}" for i in range(self._n_levels)]
            + [f"ask_q_{i}" for i in range(self._n_levels)]
        )
        df_raw = pd.read_parquet(self.file_path, columns=needed_cols)
        df_raw.columns = df_raw.columns.astype(str).str.strip()
        logger.info(f"  Loaded {len(df_raw):,} rows, {df_raw['segment_id'].nunique()} segments")

        # ---- Verify required columns ----
        required_lob = [f"bid_p_{i}" for i in range(self._n_levels)]
        missing = [c for c in required_lob if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Missing LOB columns: {missing}")
        if "timestamp_ms" not in df_raw.columns:
            raise ValueError("Missing timestamp_ms column")

        # ---- Apply end_date filter on raw data (before aggregation) ----
        if self.end_date is not None:
            ts = pd.to_datetime(df_raw["timestamp_ms"], unit="ms")
            end_ts = pd.Timestamp(self.end_date)
            df_raw = df_raw[ts <= end_ts].reset_index(drop=True)

        # ---- 2. Aggregate to N-second bars ----
        df = _aggregate_to_bars(
            df_raw,
            bar_size_s=self._bar_size_s,
            n_levels=self._n_levels,
        )
        # Free raw data
        del df_raw

        # ---- 3. Remap LOB column names for FeatureFactory ----
        df = _remap_lob_columns(df, self._n_levels)

        # ---- Ensure OHLCV columns ----
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column after aggregation: {col}")
        df[["open", "high", "low", "close", "volume"]] = (
            df[["open", "high", "low", "close", "volume"]].ffill()
        )

        # ---- Norm cutoff index ----
        norm_cutoff_idx = None
        if self.norm_cutoff_date is not None:
            cutoff_mask = df["timestamp"] >= self.norm_cutoff_date
            if cutoff_mask.any():
                norm_cutoff_idx = cutoff_mask.idxmax()

        # ---- 4. Compute 40-dim micro features via FeatureFactory ----
        ff = FeatureFactory(config={
            "n_levels": self._n_levels,
            "vol_norm_window": self.norm_span,
            "asset_class": self._asset_class,
        })
        df = ff.process_micro(df)
        self._has_lob = True

        # Extract micro feature columns as numpy array
        micro_cols = ff.micro_feature_cols
        # Verify all columns exist
        available = [c for c in micro_cols if c in df.columns]
        if len(available) < len(micro_cols):
            missing_feats = set(micro_cols) - set(available)
            logger.warning(f"  Missing {len(missing_feats)} micro features (defaulting to 0): {missing_feats}")
            for c in missing_feats:
                df[c] = np.float32(0.0)
        lob_features = df[micro_cols].values.astype(np.float32)
        self._n_micro_features = len(micro_cols)

        # ---- Apply start_date trim AFTER feature computation (warm EMAs) ----
        trim_start = 0
        if self.start_date is not None:
            start_mask = df["timestamp"] >= self.start_date
            if start_mask.any():
                trim_idx = start_mask.idxmax()
                trim_start = max(0, trim_idx - self.window_size)

        # ---- 5. Compute 8-dim OHLCV scale features ----
        scale_features = _compute_scale_features(df, norm_cutoff_idx, self.norm_span)

        # ---- Trim ----
        if trim_start > 0:
            df = df.iloc[trim_start:].reset_index(drop=True)
            scale_features = scale_features[trim_start:]
            lob_features = lob_features[trim_start:]

        # ---- 6. Store as single-scale ----
        base_scale = min(self.scales)
        self._base_scale = base_scale
        self._scale_dfs = {base_scale: df}
        self._scale_features = {base_scale: scale_features}
        self._scale_timestamps = {base_scale: df["timestamp"].values}
        self._scale_index_map = {base_scale: np.arange(len(scale_features))}

        # Base arrays for stepping + fill model
        self._base_close = df["close"].values.astype(np.float64)
        self._base_high = df["high"].values.astype(np.float64)
        self._base_low = df["low"].values.astype(np.float64)
        self._base_open = df["open"].values.astype(np.float64)
        self._base_volume = df["volume"].values.astype(np.float64)
        self._base_timestamps = df["timestamp"].values

        # ATR (use longer window for sub-minute bars)
        prev_close = np.roll(self._base_close, 1)
        prev_close[0] = self._base_close[0]
        tr = np.maximum(
            self._base_high - self._base_low,
            np.maximum(
                np.abs(self._base_high - prev_close),
                np.abs(self._base_low - prev_close),
            ),
        )
        # 60-bar ATR ≈ 5 minutes at 5s bars
        atr_window = max(14, 60 // max(self._bar_size_s, 1) * 14)
        self._base_atr = pd.Series(tr).rolling(atr_window, min_periods=1).mean().values

        self._len = len(self._base_close)
        self._ptr = self.window_size

        # ---- LOB micro features ----
        self._lob_features = lob_features

        # ---- Segment boundaries ----
        if "segment_id" in df.columns:
            self._trimmed_segment_ids = df["segment_id"].values.astype(np.int32)
            self._trimmed_boundaries = list(
                np.where(np.diff(self._trimmed_segment_ids) != 0)[0] + 1,
            )
        else:
            self._trimmed_segment_ids = None
            self._trimmed_boundaries = []

        logger.info(
            f"LOBMicroDataHandler loaded: {self._len} bars ({self._bar_size_s}s), "
            f"LOB={self._n_micro_features}-dim, "
            f"segments={len(self._trimmed_boundaries) + 1}, "
            f"window={self.window_size}",
        )

    def step(self) -> Optional[dict]:
        """Advance one bar. Returns multi-scale obs + OHLCV + 40-dim LOB features.

        Returns None at data exhaustion or segment boundary (env should reset).
        """
        if self._ptr >= self._len:
            return None

        # Segment boundary check
        if self._trimmed_segment_ids is not None:
            if self._ptr > 0 and self._ptr < len(self._trimmed_segment_ids):
                curr_seg = self._trimmed_segment_ids[self._ptr]
                prev_seg = self._trimmed_segment_ids[self._ptr - 1]
                if curr_seg != prev_seg:
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

        # LOB micro features window (40-dim)
        if self._lob_features is not None:
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
                (self.window_size, self._n_micro_features), dtype=np.float32,
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
                seg_start = self._ptr
                while seg_start > 0 and self._trimmed_segment_ids[seg_start - 1] == initial_seg:
                    seg_start -= 1
                if self._ptr - seg_start < self.window_size:
                    for boundary in sorted(self._trimmed_boundaries):
                        candidate = boundary + self.window_size
                        if candidate < self._len:
                            self._ptr = candidate
                            break

    def get_segment_boundaries(self) -> list:
        """Return list of bar indices where segment changes."""
        return self._trimmed_boundaries if hasattr(self, "_trimmed_boundaries") else []
