"""
DeepScalper Feature Engineering v2 — Evidence-Ranked Feature Pipeline
=====================================================================

Micro Features (~30 dims):  LOB-derived → LSTM
Macro Features (~15 dims):  OHLCV-derived → MLP

Normalization Pipeline (all unbounded features):
  raw → SymLog → EMA-Z(span=120, shift=1) → tanh  ∈ (-1, 1)

Bounded features (OBI, RSI, %B, slope asym, sin/cos) bypass normalization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union

# ─── Column name constants (used by env, parquet_handler, tests) ──────────

# 30 micro features consumed by the env's _build_frame()
MICRO_FEATURE_COLS: List[str] = (
    ['microprice_basis']                                     # 1   dim  0
    + [f'dofi_{i}' for i in range(1, 6)]                     # 5   dims 1-5
    + [f'dofi_int_{i}' for i in range(1, 6)]                 # 5   dims 6-10
    + ['total_obi']                                          # 1   dim  11  (Fix A: replaces VOI)
    + [f'dist_bid_{i}' for i in range(1, 6)]                 # 5   dims 12-16
    + [f'dist_ask_{i}' for i in range(1, 6)]                 # 5   dims 17-21
    + [f'obi_{i}' for i in range(1, 6)]                      # 5   dims 22-26
    + ['slope_asym']                                         # 1   dim  27
    + ['spread_bps']                                         # 1   dim  28
    + ['dofi_velocity']                                      # 1   dim  29
)
NUM_MICRO_FEATURES = len(MICRO_FEATURE_COLS)  # 30

# 15 macro features consumed by the env's _update_macro_state()
MACRO_FEATURE_COLS: List[str] = [
    'logret_1', 'logret_3', 'logret_5', 'logret_15',        # 4  multi-horizon returns
    'parkinson_vol',                                         # 1  realized volatility
    'vol_regime_ratio',                                      # 1  15m/60m Parkinson ratio
    'cvd_proxy_5', 'cvd_proxy_15',                           # 2  taker aggression proxy
    'rvol',                                                  # 1  relative volume
    'rsi_14',                                                # 1  RSI
    'bbpct_20',                                              # 1  Bollinger %B
    'funding_sin', 'funding_cos',                            # 2  funding rate clock
    'session_sin', 'session_cos',                            # 2  time-of-day session
]
NUM_MACRO_FEATURES = len(MACRO_FEATURE_COLS)  # 15

# Features that are naturally bounded and bypass SymLog→EMA-Z→tanh
# All bounded features MUST output in [-1, 1] range (Fix B)
_BOUNDED_FEATURES = {
    'total_obi', 'slope_asym',          # Fix A: voi → total_obi
    *[f'obi_{i}' for i in range(1, 6)],
    'rsi_14', 'bbpct_20',               # Fix B: centered to [-1, 1]
    'funding_sin', 'funding_cos',
    'session_sin', 'session_cos',
    'vol_regime_ratio',                  # Fix B: centered to [-1, 1]
}


class DeepScalperFeatureEngineer:
    """
    Feature Engineering v2 for DeepScalper.
    Handles Micro-level (LOB) and Macro-level (OHLCV) features.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.norm_span = int(self.config.get('vol_norm_window', 120))  # EMA span

    # ──────────────────────────────────────────────────────────────────────
    # NORMALIZATION PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _symlog(x: np.ndarray) -> np.ndarray:
        """SymLog transform: sign(x) * log(1 + |x|).  Dampens power-law tails."""
        return np.sign(x) * np.log1p(np.abs(x))

    def _ema_zscore_tanh(self, series: pd.Series, span: int = None) -> np.ndarray:
        """
        EMA Z-Score with strict causal shift → tanh soft-clip.  (Fix C)

        Uses pandas .ewm().std() for numerically correct EMA standard
        deviation instead of manual variance → sqrt.

        Steps:
          1. μ = EMA(x, span).shift(1)           ← no current-bar leakage
          2. σ = EMA_std(x, span).shift(1)        ← pandas built-in
          3. z = (x - μ) / (σ + ε)
          4. return tanh(z)  ∈ (-1, 1)
        """
        if span is None:
            span = self.norm_span
        x = series.values.astype(np.float64)
        s = pd.Series(x)

        # Fix C: Use pandas .ewm().std() — correct EMA std dev, not manual variance
        ema_mean = s.ewm(span=span, adjust=False).mean().shift(1)
        ema_std = s.ewm(span=span, adjust=False).std().shift(1)

        mu = ema_mean.values
        sigma = ema_std.values

        # Fill NaN from shift(1) on first row
        mu[0] = 0.0
        sigma[0] = 1.0  # avoid div-by-zero on first row

        # Guard against zero/NaN sigma
        sigma = np.where(np.isnan(sigma) | (sigma < 1e-8), 1.0, sigma)

        z = (x - mu) / sigma
        return np.tanh(z).astype(np.float32)

    def _normalize_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Apply the full normalization pipeline to a list of columns:
          raw → SymLog → EMA-Z(span, shift=1) → tanh

        Bounded features (listed in _BOUNDED_FEATURES) are left unchanged.
        """
        for col in cols:
            if col not in df.columns:
                continue
            if col in _BOUNDED_FEATURES:
                # Already bounded — cast to float32 and skip
                df[col] = df[col].astype(np.float32)
                continue

            raw = df[col].values.astype(np.float64)
            symlog_vals = self._symlog(raw)
            df[col] = self._ema_zscore_tanh(pd.Series(symlog_vals), self.norm_span)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # MICRO FEATURES  (LOB → 30 dims)
    # ──────────────────────────────────────────────────────────────────────

    def process_micro(self, lob_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Level 2 LOB data into 30-dim micro features.

        Computes features IN-PLACE on lob_df (same contract as v1).
        Raw LOB columns (bid_price_1, ask_price_1, bid_vol_1, ask_vol_1)
        are preserved for order execution in the environment.

        Output columns (30): see MICRO_FEATURE_COLS
        """
        df = lob_df

        # ── 0. Mid Price (needed by many features, kept for order execution) ──
        bp1 = df['bid_price_1'].values.astype(np.float64)
        ap1 = df['ask_price_1'].values.astype(np.float64)
        mid = (bp1 + ap1) / 2.0
        mid_safe = np.where(mid > 0, mid, 1e-9)
        df['mid_price'] = mid

        # ── 1. Microprice Basis (1 dim) ──
        bv1 = df['bid_vol_1'].values.astype(np.float64)
        av1 = df['ask_vol_1'].values.astype(np.float64)
        vol_sum = bv1 + av1
        microprice = (bp1 * av1 + ap1 * bv1) / (vol_sum + 1e-8)  # Fix D: epsilon guard
        mp_safe = np.where(microprice > 0, microprice, mid_safe)
        df['microprice_basis'] = np.log(mp_safe / mid_safe) * 10000.0  # bps

        # ── 2. Multi-level DOFI L1-L5 (5 dims) ──
        # Cont et al. (2014): W_b - W_a per level
        for i in range(1, 6):
            bp = df[f'bid_price_{i}'].values.astype(np.float64)
            bv = df[f'bid_vol_{i}'].values.astype(np.float64)
            ap = df[f'ask_price_{i}'].values.astype(np.float64)
            av = df[f'ask_vol_{i}'].values.astype(np.float64)

            bp_prev = np.roll(bp, 1); bp_prev[0] = bp[0]
            bv_prev = np.roll(bv, 1); bv_prev[0] = bv[0]
            ap_prev = np.roll(ap, 1); ap_prev[0] = ap[0]
            av_prev = np.roll(av, 1); av_prev[0] = av[0]

            w_b = np.where(bp > bp_prev, bv,
                    np.where(bp < bp_prev, -bv_prev, bv - bv_prev))
            w_a = np.where(ap < ap_prev, av,
                    np.where(ap > ap_prev, -av_prev, av - av_prev))

            df[f'dofi_{i}'] = w_b - w_a

        # ── 3. Integrated DOFI — 5-min rolling sum (5 dims) ──
        for i in range(1, 6):
            df[f'dofi_int_{i}'] = (
                pd.Series(df[f'dofi_{i}'].values)
                .rolling(window=5, min_periods=1)
                .sum()
                .values
            )

        # ── 4. Total Book Imbalance (1 dim) ── Fix A
        # Replaces VOI which was identical to obi_1. Total OBI captures
        # full 5-level book imbalance in a single feature.
        # Bounded [-1, 1] → bypass normalization
        total_bid_vol = np.zeros(len(df), dtype=np.float64)
        total_ask_vol = np.zeros(len(df), dtype=np.float64)
        for i in range(1, 6):
            total_bid_vol += df[f'bid_vol_{i}'].values.astype(np.float64)
            total_ask_vol += df[f'ask_vol_{i}'].values.astype(np.float64)
        total_vol_sum = total_bid_vol + total_ask_vol
        df['total_obi'] = np.clip(
            (total_bid_vol - total_ask_vol) / (total_vol_sum + 1e-8),
            -1.0, 1.0
        ).astype(np.float32)

        # ── 5. Price Distances from Mid (bps) — 5 bid + 5 ask (10 dims) ──
        for i in range(1, 6):
            bp_i = df[f'bid_price_{i}'].values.astype(np.float64)
            ap_i = df[f'ask_price_{i}'].values.astype(np.float64)
            df[f'dist_bid_{i}'] = ((bp_i - mid) / mid_safe) * 10000.0
            df[f'dist_ask_{i}'] = ((ap_i - mid) / mid_safe) * 10000.0

        # ── 6. Level-wise Volume Imbalance — OBI per level (5 dims) ──
        # Bounded [-1, 1] → bypass normalization
        for i in range(1, 6):
            bv_i = df[f'bid_vol_{i}'].values.astype(np.float64)
            av_i = df[f'ask_vol_{i}'].values.astype(np.float64)
            df[f'obi_{i}'] = np.clip(
                (bv_i - av_i) / (bv_i + av_i + 1e-8), -1.0, 1.0  # Fix D: epsilon
            ).astype(np.float32)

        # ── 7. LOB Slope Asymmetry (1 dim) ──
        # slope_bid = Σ(vol_k) / (price_1 - price_K) for bid side
        # slope_ask = Σ(vol_k) / (price_K - price_1) for ask side
        # Asymmetry = (slope_ask - slope_bid) / (slope_ask + slope_bid)
        # Bounded [-1, 1]
        bid_vol_sum = np.zeros(len(df), dtype=np.float64)
        ask_vol_sum = np.zeros(len(df), dtype=np.float64)
        for i in range(1, 6):
            bid_vol_sum += df[f'bid_vol_{i}'].values.astype(np.float64)
            ask_vol_sum += df[f'ask_vol_{i}'].values.astype(np.float64)

        bp5 = df['bid_price_5'].values.astype(np.float64)
        ap5 = df['ask_price_5'].values.astype(np.float64)

        bid_depth = np.abs(bp1 - bp5)
        ask_depth = np.abs(ap5 - ap1)

        bid_depth_safe = np.where(bid_depth > 1e-9, bid_depth, 1.0)
        ask_depth_safe = np.where(ask_depth > 1e-9, ask_depth, 1.0)

        slope_bid = bid_vol_sum / bid_depth_safe
        slope_ask = ask_vol_sum / ask_depth_safe

        slope_sum = slope_bid + slope_ask
        df['slope_asym'] = np.clip(
            (slope_ask - slope_bid) / (slope_sum + 1e-8), -1.0, 1.0  # Fix D: epsilon
        ).astype(np.float32)

        # ── 8. Relative Spread (bps) (1 dim) ──
        df['spread_bps'] = ((ap1 - bp1) / mid_safe) * 10000.0

        # ── 9. DOFI Velocity — diff of total DOFI (1 dim) ──
        total_dofi = np.zeros(len(df), dtype=np.float64)
        for i in range(1, 6):
            total_dofi += df[f'dofi_{i}'].values.astype(np.float64)
        dofi_vel = np.diff(total_dofi, prepend=0.0)
        df['dofi_velocity'] = dofi_vel

        # ── 10. Legacy compatibility: keep spread_1 alias for backward compat ──
        df['spread_1'] = df['spread_bps']

        # ── NORMALIZE all micro features ──
        self._normalize_features(df, MICRO_FEATURE_COLS)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # MACRO FEATURES  (OHLCV → 15 dims)
    # ──────────────────────────────────────────────────────────────────────

    def process_macro(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data into 15-dim macro features.

        Returns a NEW DataFrame with MACRO_FEATURE_COLS columns.
        """
        df = ohlcv_df.copy()

        # Ensure standard columns
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                raise ValueError("Missing 'close' column for macro features.")

        cl = df['close'].values.astype(np.float64)
        hi = df['high'].values.astype(np.float64)
        lo = df['low'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones_like(cl)

        cl_safe = np.where(cl > 0, cl, 1e-9)

        # ── 1. Multi-Horizon Log Returns (4 dims) ──
        for horizon in [1, 3, 5, 15]:
            lr = np.zeros_like(cl)
            if len(cl) > horizon:
                prev = cl[:-horizon]
                prev_safe = np.where(prev > 0, prev, 1e-9)
                lr[horizon:] = np.log(cl[horizon:] / prev_safe)
            lr = np.clip(lr, -0.1, 0.1)  # 10% cap
            df[f'logret_{horizon}'] = lr

        # ── 2. Parkinson Realized Volatility — 15-min rolling (1 dim) ──
        # Parkinson = sqrt( (1/4ln2) * Σ[ln(H/L)]² / n )
        hl_ratio = np.where(lo > 0, hi / lo, 1.0)
        ln_hl_sq = np.log(hl_ratio) ** 2
        parkinson_raw = pd.Series(ln_hl_sq).rolling(window=15, min_periods=1).mean().values
        df['parkinson_vol'] = np.sqrt(parkinson_raw / (4 * np.log(2)))

        # ── 3. Volatility Regime Ratio — 15m/60m Parkinson (1 dim) ──
        parkinson_60 = pd.Series(ln_hl_sq).rolling(window=60, min_periods=1).mean().values
        ratio_raw = parkinson_raw / (parkinson_60 + 1e-8)  # Fix D: epsilon
        # Fix B: Center to [-1, 1] for DQN gradient flow
        # Map [0, 3] → [0, 1] → [-1, 1]
        df['vol_regime_ratio'] = (np.clip(ratio_raw / 3.0, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)

        # ── 4. Proxy CVD — OHLCV taker aggression (2 dims) ──
        # Proxy: sign = 2*(close > open) - 1,  magnitude = volume
        # CVD_proxy = cumsum(signed_vol) over rolling window
        is_bullish = (cl >= df['open'].values.astype(np.float64)).astype(np.float64)
        signed_vol = (2 * is_bullish - 1) * vol
        df['cvd_proxy_5'] = pd.Series(signed_vol).rolling(window=5, min_periods=1).sum().values
        df['cvd_proxy_15'] = pd.Series(signed_vol).rolling(window=15, min_periods=1).sum().values

        # ── 5. Relative Volume — RVOL (1 dim) ──
        vol_ma60 = pd.Series(vol).rolling(window=60, min_periods=1).mean().values
        vol_ma60_safe = np.where(vol_ma60 > 0, vol_ma60, 1.0)
        df['rvol'] = vol / vol_ma60_safe

        # ── 6. RSI 14-minute (1 dim) ──
        # Bounded [0, 1] → bypass normalization
        delta = np.diff(cl, prepend=cl[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-8)  # Fix D: epsilon
        rsi_raw = 1.0 - 1.0 / (1.0 + rs)  # [0, 1]
        # Fix B: Center to [-1, 1] for DQN gradient flow
        df['rsi_14'] = (rsi_raw * 2.0 - 1.0).astype(np.float32)  # [-1, 1]

        # ── 7. Bollinger %B — 20-minute (1 dim) ──
        # Fix B: Soft-clip with tanh instead of hard [0,1] clip
        # %B can exceed [0,1] during extreme volatility breakouts
        sma20 = pd.Series(cl).rolling(window=20, min_periods=1).mean().values
        std20 = pd.Series(cl).rolling(window=20, min_periods=1).std().values
        std20 = np.where(np.isnan(std20) | (std20 < 1e-9), 1e-9, std20)
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        band_width = upper - lower
        pct_b_raw = (cl - lower) / (band_width + 1e-8)  # Fix D: epsilon
        # Center and soft-clip: tanh((x - 0.5) * 2) maps ~[0,1] → ~[-1,1]
        df['bbpct_20'] = np.tanh((pct_b_raw - 0.5) * 2.0).astype(np.float32)

        # ── 8. Funding Rate Clock — sin/cos (2 dims) ──
        # Binance BTC perps: funding every 8h at 00:00, 08:00, 16:00 UTC
        # Encode minutes-to-next-funding as sin/cos cycle
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            minutes_in_day = ts.dt.hour * 60 + ts.dt.minute
            # Minutes into current 8h funding cycle (480 min per cycle)
            minutes_into_cycle = minutes_in_day % 480
            phase = 2.0 * np.pi * minutes_into_cycle.values / 480.0
            df['funding_sin'] = np.sin(phase).astype(np.float32)
            df['funding_cos'] = np.cos(phase).astype(np.float32)
        else:
            df['funding_sin'] = np.zeros(len(df), dtype=np.float32)
            df['funding_cos'] = np.zeros(len(df), dtype=np.float32)

        # ── 9. Time-of-Day Session Encoding — sin/cos (2 dims) ──
        # Encode minute-of-day for Asia/London/NY session detection
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            minutes_in_day = ts.dt.hour * 60 + ts.dt.minute
            phase = 2.0 * np.pi * minutes_in_day.values / 1440.0
            df['session_sin'] = np.sin(phase).astype(np.float32)
            df['session_cos'] = np.cos(phase).astype(np.float32)
        else:
            df['session_sin'] = np.zeros(len(df), dtype=np.float32)
            df['session_cos'] = np.zeros(len(df), dtype=np.float32)

        # ── NORMALIZE unbounded macro features ──
        self._normalize_features(df, MACRO_FEATURE_COLS)

        # Fix E: Let NaNs flow through — warm-up rows are sliced off
        # downstream in parquet_handler. Do NOT fillna(0) as it poisons
        # early replay buffer with synthetic flatlined data.
        # Only forward-fill for mid-sequence gaps (e.g., missing candles).
        df[MACRO_FEATURE_COLS] = df[MACRO_FEATURE_COLS].ffill()

        result = df[MACRO_FEATURE_COLS].copy()

        # Carry timestamp forward if available (needed for alignment)
        if 'timestamp' in ohlcv_df.columns:
            result['timestamp'] = ohlcv_df['timestamp'].values

        return result

    # ──────────────────────────────────────────────────────────────────────
    # MULTIMODAL ALIGNMENT  (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────────

    def align_multimodal(self, micro_df: pd.DataFrame, macro_data: Union[pd.DataFrame, dict]) -> dict:
        """
        Aligns Macro features to Micro timestamps using Pure NumPy.

        Args:
            micro_df: DataFrame with 'timestamp' column
            macro_data: Either a DataFrame or a pre-sanitized dict of numpy arrays

        Returns:
            dict of column_name -> np.float32 array, aligned to micro_df length

        Fast-path: When N==M (same row count), returns arrays as-is.
        Normal-path: Uses searchsorted for O(N log M) alignment.
        """
        N = len(micro_df)

        # Handle dict input (pre-sanitized numpy arrays from parquet_handler)
        if isinstance(macro_data, dict):
            M = 0
            for k, v in macro_data.items():
                if k != 'timestamp' and hasattr(v, '__len__'):
                    M = len(v)
                    break

            if N == M:
                aligned_data = {k: v for k, v in macro_data.items() if k != 'timestamp'}
                return aligned_data

            if 'timestamp' not in macro_data:
                raise ValueError("macro_data dict must have 'timestamp' for alignment when lengths differ")

            micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
            macro_ts = np.array(macro_data['timestamp'], dtype='datetime64[ns]').view('int64')

            idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
            idx = np.clip(idx, 0, M - 1)

            aligned_data = {}
            for k, v in macro_data.items():
                if k == 'timestamp':
                    continue
                aligned_data[k] = v[idx]

            return aligned_data

        # Legacy DataFrame path
        M = len(macro_data)

        if N == M:
            aligned_data = {}
            for col in macro_data.columns:
                if col == 'timestamp':
                    continue
                vals = np.array(macro_data[col].values, dtype=np.float32)
                aligned_data[col] = vals
            return aligned_data

        # DataFrame searchsorted path
        micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
        macro_ts = np.array(macro_data['timestamp'].values, dtype='datetime64[ns]').view('int64')

        idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
        idx = np.clip(idx, 0, M - 1)

        aligned_data = {}
        for col in macro_data.columns:
            if col == 'timestamp':
                continue
            vals = np.array(macro_data[col].values, dtype=np.float32)
            aligned_data[col] = vals[idx]

        return aligned_data
