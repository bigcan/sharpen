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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd

from finrl_pro_ds.data.multiscale_handler import (
    _compute_scale_features,
    _resample_ohlcv,
    compute_features_with_warmup,
)

logger = logging.getLogger(__name__)

# Maximum 1-min bars to retain in the rolling buffer.
# ~30 days of 1-min data. Prevents unbounded memory growth.
_MAX_BUFFER_BARS = 50_000


def resolve_norm_warmup_path(config: dict) -> Optional[str]:
    """Resolve per-fold norm warmup buffer path (S509 train-serve skew fix).

    Two strategies can share a checkpoint directory while having different
    feature scales (e.g. sg1-xauusd `[3,15,60]` vs gmgp1-xauusd `[15,60,240]`
    both pointing at `WF_seed42_fold_07_*/`), so the buffer file is named
    after the scales it was extracted for.

    Resolution order, where ``ckpt_dir`` ranges over (1) explicit, (2) ensemble
    `_resolved_paths`, (3) solo `agent.checkpoint_path`:
        a. `features.norm_warmup_path` (explicit override)
        b. ``ckpt_dir/norm_warmup_<scales-dash-joined>.pkl``
        z. None → caller's LiveObsBuilder falls back to legacy rolling EMA-Z

    Returns the resolved path string, or None (with logger.warning).
    """
    from pathlib import Path
    feat_cfg = config.get("features", {})
    explicit = feat_cfg.get("norm_warmup_path")
    if explicit:
        return explicit
    scales = feat_cfg.get("scales") or []
    suffixed_name = f"norm_warmup_{'-'.join(str(s) for s in scales)}.pkl" if scales else None
    candidates_per_dir = [n for n in [suffixed_name] if n]

    agent_cfg = config.get("agent", {}) or {}
    ensemble_cfg = agent_cfg.get("ensemble") or {}
    ckpt_dirs: list[Path] = []
    for ckpt_path in (ensemble_cfg.get("_resolved_paths") or {}).values():
        ckpt_dirs.append(Path(ckpt_path).parent)
    solo_ckpt = agent_cfg.get("checkpoint_path")
    if solo_ckpt:
        ckpt_dirs.append(Path(solo_ckpt).parent)
    for ckpt_dir in ckpt_dirs:
        for name in candidates_per_dir:
            candidate = ckpt_dir / name
            if candidate.exists():
                return str(candidate)
    logger.warning(
        f"S509: no norm warmup buffer found alongside checkpoint(s) "
        f"(searched {candidates_per_dir} in {len(ckpt_dirs)} dir(s)); "
        f"LiveObsBuilder will use legacy rolling EMA-Z (train-serve skew possible).",
    )
    return None


@dataclass(frozen=True)
class FeatureVarianceStatus:
    """Read-only snapshot returned by :meth:`LiveObsBuilder.feature_variance_status`.

    Consumed by :class:`LiveTradingEngine` to pass ``feature_state`` into
    :meth:`ActionDriftTracker.observe` (S551-cont-4 v2.6 amendment). ``OK`` is
    the conservative default: insufficient samples or disabled drift detection
    return ``state="OK"`` so the tracker never vetoes on missing data.
    """

    state: str                                       # "OK" | "FLAT" | "EXPLODE"
    base_scale: int
    base_scale_ratio: Optional[float]
    per_scale_ratio: dict[int, Optional[float]] = field(default_factory=dict)
    flat_threshold: float = 0.01
    explode_threshold: float = 100.0


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

    # S546 GAP-B fix: weekend/session-closed asset classes lose ~30% of
    # calendar minutes to halts (Gold CFDs close Fri 22:00 UTC → Sun 22:00
    # UTC).  A naïve `start = end - timedelta(minutes=bootstrap_bars)`
    # under-fetches active bars, producing a 95% warmup gate fail on the
    # coarsest scale and a 10-bar engine lockout.  Expand the calendar
    # window by the listed factor per asset class.  Tuned so that the
    # 60-min scale's 360-bar EMA-convergence target lands cleanly with a
    # small safety margin.  See
    # docs/research/sg1_xauusd_sim_to_live_gap_audit.md §3 + §4.2.
    _CALENDAR_EXPANSION_FACTOR: dict[str, float] = {
        "crypto": 1.0,
        "cfd_gold": 1.45,
        "cfd_forex": 1.45,
        "cme_futures": 1.45,
    }

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
        norm_warmup_path: Optional[str] = None,
        asset_class: str = "crypto",
        max_leverage: float = 1.0,
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
            asset_class: Drives bootstrap calendar-window expansion for assets
                with closed sessions ("cfd_gold", "cfd_forex", "cme_futures";
                see ``_CALENDAR_EXPANSION_FACTOR``).  Default "crypto" → 1.0×
                (no expansion).
            max_leverage: Training env's ``env.max_leverage``. The training
                env reports position and pnl_proxy in the private state
                DIVIDED by max_leverage (B1 fix,
                ``ContinuousSwingEnv._get_private_state``); live must mirror
                that or a leverage-trained agent sees out-of-distribution
                private dims (FE-02, 2026-07-08 FE audit). At the default 1.0
                the division is a no-op — identical to prior behavior for
                every current live config.
        """
        self.scales = sorted(scales)
        self.window_size = window_size
        self.norm_span = norm_span
        self.n_features = n_features
        self.bootstrap_bars = bootstrap_bars
        self.obs_mode = obs_mode
        self.summary_feature_indices = summary_feature_indices or [0, 1, 2, 6, 7]
        self.asset_class = asset_class
        self.max_leverage = max(float(max_leverage), 1e-9)
        if asset_class not in self._CALENDAR_EXPANSION_FACTOR:
            logger.warning(
                f"LiveObsBuilder: unknown asset_class={asset_class!r}; "
                "defaulting bootstrap calendar-expansion factor to 1.0 "
                "(may under-fetch on weekend-closed markets — declare the "
                "asset_class explicitly in features.asset_class).",
            )

        # S509 fix: per-scale pre-cutoff warmup buffer that reproduces training's
        # freeze-and-restart EMA-Z behavior. Without this, live's rolling EMA
        # tracks recent regime → train-serve skew → silent capital starvation
        # (sg1-xauusd 2026-04-30 drift_crit was the witness incident).
        self._norm_warmup: dict[int, pd.DataFrame] = {}
        if norm_warmup_path:
            self._load_norm_warmup(norm_warmup_path)

        # Rolling 1-min OHLCV buffer (DataFrame)
        self._buffer_1min: Optional[pd.DataFrame] = None

        # Per-scale feature arrays and metadata
        self._scale_features: dict[int, np.ndarray] = {}
        self._scale_dfs: dict[int, pd.DataFrame] = {}
        self._scale_last_bar_count: dict[int, int] = {}
        # X2 / audit P2-01: index of the last CAUSAL (completed) bar per scale,
        # used to terminate each observation window. Mirrors
        # MultiScaleOHLCVHandler._scale_index_map evaluated at the latest base bar.
        self._scale_end_idx: dict[int, int] = {}

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
    def calendar_expansion_factor(self) -> float:
        """Calendar-window expansion factor for the configured asset class.

        Crypto-style 24/7 assets return 1.0.  Weekend-closed assets
        (cfd_gold, cfd_forex, cme_futures) return 1.45 so that
        ``bootstrap_bars`` of *active* 1-min data lands inside the fetched
        calendar window.  Unknown asset classes fall back to 1.0 (already
        warned at construction time).
        """
        return self._CALENDAR_EXPANSION_FACTOR.get(self.asset_class, 1.0)

    def compute_bootstrap_calendar_minutes(self) -> int:
        """Calendar-minute window for the bootstrap fetch.

        ``bootstrap_bars + 60`` minutes of *active* data on a 24/7 market;
        scaled by ``calendar_expansion_factor()`` for weekend-closed asset
        classes (S546 GAP-B).
        """
        return int((self.bootstrap_bars + 60) * self.calendar_expansion_factor())

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

        calendar_minutes = self.compute_bootstrap_calendar_minutes()
        factor = self.calendar_expansion_factor()
        logger.info(
            f"LiveObsBuilder bootstrapping: requesting {calendar_minutes} "
            f"calendar minutes for {self.bootstrap_bars} active 1-min bars on "
            f"{asset} (asset_class={self.asset_class}, expansion={factor:.2f}x).",
        )

        # FIX AUD-C01: Use correct CryptoLoader.fetch_ohlcv() signature.
        # Compute start/end from bootstrap_bars instead of passing limit.
        # S546: window is calendar-minute-expanded for weekend-closed assets.
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=calendar_minutes)

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

        # S509: force tz-naive UTC to match the warmup buffer's stripping (see
        # extract_norm_warmup_buffer.py). Brokers that return ISO8601 with `Z`
        # or `+00:00` would otherwise produce a tz-aware Series; concatenating
        # that with the tz-naive warmup buffer in compute_features_with_warmup
        # raises TypeError on modern pandas.
        ts = pd.to_datetime(df['timestamp'], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        df['timestamp'] = ts
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df[['open', 'high', 'low', 'close', 'volume']] = (
            df[['open', 'high', 'low', 'close', 'volume']].ffill()
        )
        # FE-01: dedup on the bootstrap path too — loaders dedup internally,
        # but the buffer invariant (unique, sorted 1-min timestamps) is
        # enforced here so update()'s resample math can rely on it.
        df = (
            df.sort_values('timestamp', kind='stable')
            .drop_duplicates(subset='timestamp', keep='last')
            .reset_index(drop=True)
        )

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

    def _load_norm_warmup(self, path: str) -> None:
        """Load per-scale pre-cutoff warmup buffer (S509 train-serve skew fix).

        Schema (written by scripts/extract_norm_warmup_buffer.py):
            {"schema_version": "v1", "scales": [...], "n_warmup": 200,
             "norm_span": 120, "fold": int, "seed": int,
             "train_end_date": str, "buffers": {scale: pd.DataFrame}}
        """
        import pickle
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"norm warmup buffer not found: {p}")
        with open(p, "rb") as f:
            payload = pickle.load(f)
        if payload.get("schema_version") != "v1":
            raise ValueError(f"unsupported norm_warmup schema: {payload.get('schema_version')}")
        if int(payload.get("norm_span", 0)) != int(self.norm_span):
            raise ValueError(
                f"norm_span mismatch: warmup={payload.get('norm_span')} live={self.norm_span}"
            )
        for scale in self.scales:
            if scale not in payload["buffers"]:
                raise ValueError(
                    f"warmup buffer missing scale={scale}min "
                    f"(have {sorted(payload['buffers'].keys())})"
                )
            self._norm_warmup[scale] = payload["buffers"][scale].reset_index(drop=True)
        logger.info(
            f"S509: loaded norm warmup buffer fold={payload.get('fold')} "
            f"train_end={payload.get('train_end_date')} "
            f"scales={sorted(self._norm_warmup.keys())}",
        )

    def _recompute_all_scales(self) -> None:
        """Resample and compute features for every scale.

        S509: when a per-scale warmup buffer is loaded, features are computed
        via `compute_features_with_warmup` to reproduce training's freeze-and-
        restart EMA-Z. Falls back to rolling-EMA-Z (legacy/buggy) otherwise.
        """
        for scale in self.scales:
            resampled = _resample_ohlcv(self._buffer_1min, scale)
            warmup = self._norm_warmup.get(scale)
            if warmup is not None and len(warmup) > 0:
                features = compute_features_with_warmup(
                    warmup, resampled, span=self.norm_span,
                    n_features=self.n_features,
                )
            else:
                # No warmup → legacy rolling EMA-Z (LEAK-1 N/A path; left as
                # fallback for tests / strategies that have not yet shipped a
                # warmup buffer alongside their checkpoint).
                features = _compute_scale_features(
                    resampled, norm_cutoff_idx=None, span=self.norm_span,
                    n_features=self.n_features,
                )
            self._scale_dfs[scale] = resampled
            self._scale_features[scale] = features
            self._scale_last_bar_count[scale] = len(resampled)

        # X2: refresh the causal window end-index once all scale dfs exist.
        self._recompute_scale_end_indices()

    def _recompute_scale_end_indices(self) -> None:
        """Index of the last CAUSAL bar to terminate each scale's obs window.

        X2 / audit P2-01: mirrors ``MultiScaleOHLCVHandler._scale_index_map``
        evaluated at the latest base bar. The base (finest) scale terminates at
        its last bar; each coarser scale terminates at the most recent bar that
        has already CLOSED at or before the latest base bar's START timestamp —
        never the in-progress bar. A coarse bar stamped ``t`` (label='left')
        closes at ``t + scale`` minutes, so it is only causal for a base bar at
        time ``b`` when ``t + scale <= b``; searching on ``base_ts - scale_ns``
        selects that last-closed bar. Without this the window would end on the
        partial in-progress coarse bar, leaking up to ``scale - base_scale``
        minutes of the base bar's own future (train<->live divergence).
        """
        self._scale_end_idx = {}
        base_df = self._scale_dfs.get(self._base_scale)
        if base_df is None or len(base_df) == 0:
            return
        # Same int64-ns conversion path as the training handler so the
        # searchsorted is bit-identical.
        base_ts_last = base_df["timestamp"].values.astype("int64")[-1]
        for scale in self.scales:
            n = len(self._scale_features.get(scale, []))
            if n == 0:
                continue
            if scale == self._base_scale:
                # 1:1 mapping — the current base bar itself is the decision bar.
                self._scale_end_idx[scale] = n - 1
                continue
            coarse_ts = self._scale_dfs[scale]["timestamp"].values.astype("int64")
            scale_ns = scale * 60 * 1_000_000_000
            idx = int(np.searchsorted(coarse_ts, base_ts_last - scale_ns, side="right") - 1)
            self._scale_end_idx[scale] = int(np.clip(idx, 0, n - 1))

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

    def _compute_variance_ratio(self, scale: int) -> Optional[float]:
        """Read-only variance ratio for ``scale`` against its rolling median.

        Does NOT mutate ``self._variance_history`` — callers that want to advance
        the rolling buffer (e.g. :meth:`_check_drift`) must append separately.

        Returns:
            ``current_var / rolling_median``, or ``None`` when:
              - drift detection is disabled
              - the scale has no features yet
              - the rolling history is too short (<10 samples)
              - the rolling median is degenerate (<1e-15)
        """
        if not self._drift_detection_enabled:
            return None
        features = self._scale_features.get(scale)
        if features is None or len(features) == 0:
            return None
        history = self._variance_history.get(scale)
        if history is None or len(history) < 10:
            return None
        rolling_median = float(np.median(list(history)))
        if rolling_median < 1e-15:
            return None
        current_var = float(np.var(features[-1]))
        return current_var / rolling_median

    def _ratio_to_state(self, ratio: Optional[float]) -> str:
        """Classify a variance ratio. ``None`` → ``"OK"`` (conservative — don't veto on missing data)."""
        if ratio is None:
            return "OK"
        if ratio < self._drift_flat_threshold:
            return "FLAT"
        if ratio > self._drift_explode_threshold:
            return "EXPLODE"
        return "OK"

    def feature_variance_status(
        self,
        *,
        scale: Union[str, int] = "base",
    ) -> FeatureVarianceStatus:
        """Read-only snapshot of feature-variance health.

        Args:
            scale:
                ``"base"`` (default) — state of the finest scale, the only scale
                that updates every bar.
                ``"all"`` — ``"FLAT"`` if any scale is FLAT, else ``"EXPLODE"``
                if any scale exploded, else ``"OK"``.
                ``int`` — state of that specific scale (must be in ``self.scales``).

        Returns:
            :class:`FeatureVarianceStatus` with the selected state plus per-scale
            ratios for downstream telemetry. Insufficient samples are reported as
            ``"OK"`` (never veto on missing data — operator should rely on the
            existing 10-sample minimum in :meth:`_check_drift`).
        """
        per_scale: dict[int, Optional[float]] = {
            s: self._compute_variance_ratio(s) for s in self.scales
        }
        if scale == "base":
            target_state = self._ratio_to_state(per_scale.get(self._base_scale))
        elif scale == "all":
            states = [self._ratio_to_state(r) for r in per_scale.values()]
            if "FLAT" in states:
                target_state = "FLAT"
            elif "EXPLODE" in states:
                target_state = "EXPLODE"
            else:
                target_state = "OK"
        elif isinstance(scale, int):
            if scale not in self.scales:
                raise ValueError(
                    f"feature_variance_status: scale {scale} not in configured "
                    f"scales {self.scales}"
                )
            target_state = self._ratio_to_state(per_scale.get(scale))
        else:
            raise ValueError(
                f"feature_variance_status: scale must be 'base', 'all', or int "
                f"in self.scales; got {scale!r}"
            )
        return FeatureVarianceStatus(
            state=target_state,
            base_scale=self._base_scale,
            base_scale_ratio=per_scale.get(self._base_scale),
            per_scale_ratio=per_scale,
            flat_threshold=self._drift_flat_threshold,
            explode_threshold=self._drift_explode_threshold,
        )

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
            state = self._ratio_to_state(variance_ratio)

            if state == "FLAT":
                logger.warning(
                    "EMA drift detected — scale %dmin: feature variance FLAT "
                    "(ratio=%.4f, current=%.2e, median=%.2e). "
                    "Possible stale or corrupted data feed.",
                    scale, variance_ratio, current_var, rolling_median,
                )
                any_drift = True
            elif state == "EXPLODE":
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

        # S509: same tz-stripping as _init_from_dataframe — guarantees the
        # rolling buffer stays tz-naive so the warmup-buffer concat in
        # compute_features_with_warmup never hits the tz-aware/naive collision.
        ts = pd.to_datetime(new_df['timestamp'], utc=True)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        new_df['timestamp'] = ts

        # Append to buffer.
        # FE-01 (2026-07-08 FE audit): dedup by timestamp, keep='last'.
        # _fetch_new_bars re-fetches from the last BASE-bar start (+1min), so
        # every call overlaps the buffer by up to (base_scale-1) already-held
        # minutes. Without dedup those rows accumulate and _resample_ohlcv
        # SUMS their volume — every closed base bar except the newest carried
        # ~2x volume while the decision bar stayed 1x, biasing volume_z (and
        # the SG-1 signal gate) low on exactly the bar the agent acts on.
        # keep='last' also lets a re-fetched (finalized/revised) candle
        # replace its earlier partial version; the overlap now degrades into
        # harmless gap-healing. Stable sort preserves fetch order for equal
        # timestamps so 'last' == most recent fetch.
        self._buffer_1min = (
            pd.concat([self._buffer_1min, new_df], ignore_index=True)
            .sort_values('timestamp', kind='stable')
            .drop_duplicates(subset='timestamp', keep='last')
            .reset_index(drop=True)
        )

        # audit F19: drop duplicate/overlapping 1-min bars (e.g. a re-fetch where
        # the loader returns the candle containing `since`). Without this, a dup
        # bar double-counts volume (resample sums volume while OHLC uses
        # first/max/min/last), silently corrupting the volume_z feature. Keep the
        # last occurrence (freshest fetch) and re-sort.
        self._buffer_1min = (
            self._buffer_1min.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
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

        # Scale features — causal window terminating at the last COMPLETED bar.
        # X2 / audit P2-01: coarser scales must NOT include the in-progress bar
        # (whose [t, t+scale) interval has not closed at the latest base bar);
        # the base scale terminates at its last bar. Window construction is
        # identical to MultiScaleOHLCVHandler.step() (incl. front-padding with
        # the window's first row) so train<->live observations stay bit-equal.
        for i, scale in enumerate(self.scales):
            features = self._scale_features[scale]
            n = len(features)
            idx = self._scale_end_idx.get(scale, n - 1)

            # Window: [idx - window_size + 1, idx + 1)
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = features[start:end]
            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                pad = np.tile(window[0:1], (pad_len, 1))
                window = np.concatenate([pad, window], axis=0)

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

        Replicates ContinuousSwingEnv._get_private_state() exactly —
        including the B1 leverage normalization: position and pnl_proxy are
        reported as fraction-of-cap (divided by max_leverage) so the private
        dims stay in the same range the agent trained on regardless of the
        leverage knob (FE-02). No-op at max_leverage=1.0.
        """
        # 1. Current position normalized to [-1, 1] regardless of max_leverage.
        pos = float(current_position) / self.max_leverage

        # 2. Unrealized PnL proxy (recent return * position-as-fraction-of-cap, clipped)
        pnl_proxy = 0.0
        if prev_close > 0 and current_close > 0:
            ret_bps = (current_close - prev_close) / prev_close * 10000.0
            pnl_proxy = float(np.clip(pos * ret_bps / 100.0, -1.0, 1.0))

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

    def get_current_hl(self) -> tuple[float, float]:
        """Return (high, low) of the most recent base-scale bar. (0.0, 0.0) if unavailable."""
        base_df = self._scale_dfs.get(self._base_scale)
        if base_df is not None and len(base_df) > 0:
            return (
                float(base_df['high'].iloc[-1]),
                float(base_df['low'].iloc[-1]),
            )
        return (0.0, 0.0)

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
