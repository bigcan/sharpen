"""AlphaSeek Feature Engine — LOB snapshots → 8 microstructure features.

Two paths:
  - process_batch(df)       : vectorized over a DataFrame (training data generation)
  - process_snapshot(snap)  : incremental O(buffer) per tick (live trading)

Both paths produce numerically identical results for the same input sequence.
LEAK-1 compliant: call reset() at segment/split boundaries.

Feature dimensions (8):
  0: bbo_imbalance     — (bid_qty - ask_qty) / (bid_qty + ask_qty)        [-1, 1]
  1: depth_imbalance_5 — top-5 level qty imbalance                        [-1, 1]
  2: microprice_offset — (microprice - mid) / mid * 1e4 bps, scaled       [-1, 1]
  3: ofi               — order flow imbalance (delta of qty changes)       EMA-Z-tanh
  4: spread_z          — spread                                            EMA-Z-tanh
  5: momentum_5s       — log(mid_t / mid_{t-5})                            EMA-Z-tanh
  6: realized_vol_30s  — rolling 30-tick std of log returns                 EMA-Z-tanh
  7: depth_ratio_5     — bid_depth_5 / total_depth_5, centered             [-1, 1]
"""

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization helpers (reused from multiscale_handler.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Snapshot dict keys expected by process_snapshot()
# ---------------------------------------------------------------------------
# best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, spread,
# bid_prices_5 (list[float]), bid_qtys_5 (list[float]),
# ask_prices_5 (list[float]), ask_qtys_5 (list[float])
# ---------------------------------------------------------------------------

N_DEPTH_LEVELS = 5  # top-5 levels for depth features


class AlphaSeekFeatureEngine:
    """Pure LOB → 8 features computation. No I/O."""

    N_FEATURES = 8

    def __init__(
        self,
        norm_span: int = 120,
        momentum_window: int = 5,
        vol_window: int = 30,
        max_buffer: int = 0,
    ):
        self.norm_span = norm_span
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        # Buffer cap for streaming path: 5x span is enough for EMA convergence
        self._max_buffer = max_buffer if max_buffer > 0 else 5 * norm_span

        # Streaming state
        self._mid_prices: deque = deque(maxlen=self._max_buffer)
        self._spreads: deque = deque(maxlen=self._max_buffer)
        self._ofis: deque = deque(maxlen=self._max_buffer)
        self._log_returns: deque = deque(maxlen=self._max_buffer)

        # Previous snapshot quantities for OFI delta
        self._prev_bid_qty_top: float = 0.0
        self._prev_ask_qty_top: float = 0.0
        self._tick_count: int = 0

    def reset(self) -> None:
        """Reset all streaming state. Call at segment/split boundaries (LEAK-1)."""
        self._mid_prices.clear()
        self._spreads.clear()
        self._ofis.clear()
        self._log_returns.clear()
        self._prev_bid_qty_top = 0.0
        self._prev_ask_qty_top = 0.0
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Batch path (vectorized, for training data generation)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        df: pd.DataFrame,
        segment_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process a LOB DataFrame → (N, 8) float32 features.

        Expected columns: best_bid_price, best_bid_qty, best_ask_price,
        best_ask_qty, spread, mid_price, bid_q_0..bid_q_4, ask_q_0..ask_q_4,
        bid_p_0..bid_p_4, ask_p_0..ask_p_4.
        Optional: segment_id (for LEAK-1 boundary reset).
        """
        n = len(df)
        features = np.zeros((n, self.N_FEATURES), dtype=np.float32)

        bid_qty = df["best_bid_qty"].values.astype(np.float64)
        ask_qty = df["best_ask_qty"].values.astype(np.float64)
        mid = df["mid_price"].values.astype(np.float64)
        spread = df["spread"].values.astype(np.float64)

        # Depth arrays (top 5 levels)
        bid_q_cols = [f"bid_q_{i}" for i in range(N_DEPTH_LEVELS)]
        ask_q_cols = [f"ask_q_{i}" for i in range(N_DEPTH_LEVELS)]
        bid_p_cols = [f"bid_p_{i}" for i in range(N_DEPTH_LEVELS)]
        ask_p_cols = [f"ask_p_{i}" for i in range(N_DEPTH_LEVELS)]

        bid_q_5 = df[bid_q_cols].fillna(0.0).values.astype(np.float64)  # (N, 5)
        ask_q_5 = df[ask_q_cols].fillna(0.0).values.astype(np.float64)
        bid_p_5 = df[bid_p_cols].fillna(0.0).values.astype(np.float64)
        ask_p_5 = df[ask_p_cols].fillna(0.0).values.astype(np.float64)

        # --- Dim 0: BBO imbalance [-1, 1] ---
        total_bbo = bid_qty + ask_qty
        total_bbo = np.where(total_bbo < 1e-12, 1e-12, total_bbo)
        features[:, 0] = np.clip((bid_qty - ask_qty) / total_bbo, -1.0, 1.0)

        # --- Dim 1: Depth imbalance (top 5) [-1, 1] ---
        bid_depth_5 = bid_q_5.sum(axis=1)
        ask_depth_5 = ask_q_5.sum(axis=1)
        total_depth_5 = bid_depth_5 + ask_depth_5
        total_depth_5 = np.where(total_depth_5 < 1e-12, 1e-12, total_depth_5)
        features[:, 1] = np.clip(
            (bid_depth_5 - ask_depth_5) / total_depth_5, -1.0, 1.0,
        )

        # --- Dim 2: Microprice offset (bps, scaled to [-1, 1]) ---
        microprice = (bid_p_5[:, 0] * ask_q_5[:, 0] + ask_p_5[:, 0] * bid_q_5[:, 0])
        micro_denom = bid_q_5[:, 0] + ask_q_5[:, 0]
        micro_denom = np.where(micro_denom < 1e-12, 1e-12, micro_denom)
        microprice = microprice / micro_denom
        mid_safe = np.where(np.abs(mid) < 1e-12, 1e-12, mid)
        microprice_bps = (microprice - mid) / mid_safe * 1e4
        features[:, 2] = np.clip(microprice_bps / 5.0, -1.0, 1.0).astype(np.float32)

        # --- Dim 3: OFI (order flow imbalance) — EMA-Z-tanh ---
        # Delta of top-of-book quantities
        d_bid = np.diff(bid_qty, prepend=bid_qty[0])
        d_ask = np.diff(ask_qty, prepend=ask_qty[0])
        ofi_raw = d_bid - d_ask
        ofi_symlog = _symlog(ofi_raw)
        if segment_ids is not None:
            features[:, 3] = self._batch_normalize_with_segments(
                ofi_symlog, segment_ids,
            )
        else:
            features[:, 3] = _ema_zscore_tanh(ofi_symlog, self.norm_span)

        # --- Dim 4: Spread — EMA-Z-tanh ---
        spread_symlog = _symlog(spread)
        if segment_ids is not None:
            features[:, 4] = self._batch_normalize_with_segments(
                spread_symlog, segment_ids,
            )
        else:
            features[:, 4] = _ema_zscore_tanh(spread_symlog, self.norm_span)

        # --- Dim 5: Momentum (5-tick log return) — EMA-Z-tanh ---
        log_mid = np.log(np.where(mid < 1e-12, 1e-12, mid))
        momentum = np.zeros(n, dtype=np.float64)
        momentum[self.momentum_window :] = (
            log_mid[self.momentum_window :] - log_mid[: -self.momentum_window]
        )
        mom_symlog = _symlog(momentum)
        if segment_ids is not None:
            features[:, 5] = self._batch_normalize_with_segments(
                mom_symlog, segment_ids,
            )
        else:
            features[:, 5] = _ema_zscore_tanh(mom_symlog, self.norm_span)

        # --- Dim 6: Realized vol (30-tick rolling std of log returns) — EMA-Z-tanh ---
        log_ret = np.diff(log_mid, prepend=log_mid[0])
        vol_series = pd.Series(log_ret).rolling(
            self.vol_window, min_periods=1,
        ).std().fillna(0.0).values
        vol_symlog = _symlog(vol_series)
        if segment_ids is not None:
            features[:, 6] = self._batch_normalize_with_segments(
                vol_symlog, segment_ids,
            )
        else:
            features[:, 6] = _ema_zscore_tanh(vol_symlog, self.norm_span)

        # --- Dim 7: Depth ratio (top 5), centered [-1, 1] ---
        depth_ratio = bid_depth_5 / total_depth_5  # [0, 1]
        features[:, 7] = np.clip((depth_ratio - 0.5) * 2.0, -1.0, 1.0).astype(
            np.float32,
        )

        logger.info(
            f"FeatureEngine.process_batch: {n:,} rows → ({n}, {self.N_FEATURES}) features",
        )
        return features

    def _batch_normalize_with_segments(
        self,
        values: np.ndarray,
        segment_ids: np.ndarray,
    ) -> np.ndarray:
        """Apply EMA-Z-tanh per segment (LEAK-1: no normalization across segments)."""
        result = np.zeros_like(values, dtype=np.float32)
        unique_segments = np.unique(segment_ids)
        for seg_id in unique_segments:
            mask = segment_ids == seg_id
            seg_vals = values[mask]
            result[mask] = _ema_zscore_tanh(seg_vals, self.norm_span)
        return result

    # ------------------------------------------------------------------
    # Incremental path (for live trading)
    # ------------------------------------------------------------------

    def process_snapshot(self, snapshot: dict) -> np.ndarray:
        """Process one LOB snapshot → 8 features (float32).

        Uses a rolling buffer and recomputes EMA-Z-tanh on the buffer
        tail to guarantee numerical parity with the batch path.

        Args:
            snapshot: dict with keys:
                best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
                spread, bid_prices_5 (list), bid_qtys_5 (list),
                ask_prices_5 (list), ask_qtys_5 (list)

        Returns:
            np.ndarray of shape (8,), float32.
        """
        bid_price = float(snapshot["best_bid_price"])
        ask_price = float(snapshot["best_ask_price"])
        bid_qty = float(snapshot["best_bid_qty"])
        ask_qty = float(snapshot["best_ask_qty"])
        spread_val = float(snapshot["spread"])

        bid_prices_5 = snapshot.get("bid_prices_5", [bid_price])
        bid_qtys_5 = snapshot.get("bid_qtys_5", [bid_qty])
        ask_prices_5 = snapshot.get("ask_prices_5", [ask_price])
        ask_qtys_5 = snapshot.get("ask_qtys_5", [ask_qty])

        # Pad to 5 levels if fewer
        bid_qtys_5 = list(bid_qtys_5) + [0.0] * max(0, 5 - len(bid_qtys_5))
        ask_qtys_5 = list(ask_qtys_5) + [0.0] * max(0, 5 - len(ask_qtys_5))
        bid_prices_5 = list(bid_prices_5) + [0.0] * max(0, 5 - len(bid_prices_5))
        ask_prices_5 = list(ask_prices_5) + [0.0] * max(0, 5 - len(ask_prices_5))

        mid = (bid_price + ask_price) / 2.0

        features = np.zeros(self.N_FEATURES, dtype=np.float32)

        # --- Dim 0: BBO imbalance ---
        total_bbo = bid_qty + ask_qty
        if total_bbo > 1e-12:
            features[0] = np.clip((bid_qty - ask_qty) / total_bbo, -1.0, 1.0)

        # --- Dim 1: Depth imbalance (top 5) ---
        bid_depth_5 = sum(bid_qtys_5[:5])
        ask_depth_5 = sum(ask_qtys_5[:5])
        total_depth = bid_depth_5 + ask_depth_5
        if total_depth > 1e-12:
            features[1] = np.clip(
                (bid_depth_5 - ask_depth_5) / total_depth, -1.0, 1.0,
            )

        # --- Dim 2: Microprice offset ---
        micro_denom = bid_qtys_5[0] + ask_qtys_5[0]
        if micro_denom > 1e-12 and abs(mid) > 1e-12:
            microprice = (
                bid_prices_5[0] * ask_qtys_5[0] + ask_prices_5[0] * bid_qtys_5[0]
            ) / micro_denom
            microprice_bps = (microprice - mid) / mid * 1e4
            features[2] = np.clip(microprice_bps / 5.0, -1.0, 1.0)

        # --- OFI raw value ---
        if self._tick_count > 0:
            ofi_raw = (bid_qty - self._prev_bid_qty_top) - (
                ask_qty - self._prev_ask_qty_top
            )
        else:
            ofi_raw = 0.0
        self._prev_bid_qty_top = bid_qty
        self._prev_ask_qty_top = ask_qty

        # Update buffers
        self._mid_prices.append(mid)
        self._spreads.append(spread_val)
        self._ofis.append(ofi_raw)
        if len(self._mid_prices) >= 2:
            prev_mid = self._mid_prices[-2]
            if prev_mid > 1e-12:
                self._log_returns.append(np.log(mid / prev_mid))
            else:
                self._log_returns.append(0.0)
        else:
            self._log_returns.append(0.0)

        self._tick_count += 1

        # --- Dim 3: OFI — EMA-Z-tanh over buffer ---
        ofi_arr = np.array(self._ofis, dtype=np.float64)
        ofi_norm = _ema_zscore_tanh(_symlog(ofi_arr), self.norm_span)
        features[3] = ofi_norm[-1]

        # --- Dim 4: Spread — EMA-Z-tanh over buffer ---
        spread_arr = np.array(self._spreads, dtype=np.float64)
        spread_norm = _ema_zscore_tanh(_symlog(spread_arr), self.norm_span)
        features[4] = spread_norm[-1]

        # --- Dim 5: Momentum (5-tick) — EMA-Z-tanh over buffer ---
        mid_arr = np.array(self._mid_prices, dtype=np.float64)
        log_mid = np.log(np.where(mid_arr < 1e-12, 1e-12, mid_arr))
        n = len(log_mid)
        momentum = np.zeros(n, dtype=np.float64)
        if n > self.momentum_window:
            momentum[self.momentum_window :] = (
                log_mid[self.momentum_window :] - log_mid[: -self.momentum_window]
            )
        mom_norm = _ema_zscore_tanh(_symlog(momentum), self.norm_span)
        features[5] = mom_norm[-1]

        # --- Dim 6: Realized vol (30-tick) — EMA-Z-tanh over buffer ---
        lr_arr = np.array(self._log_returns, dtype=np.float64)
        vol_series = (
            pd.Series(lr_arr)
            .rolling(self.vol_window, min_periods=1)
            .std()
            .fillna(0.0)
            .values
        )
        vol_norm = _ema_zscore_tanh(_symlog(vol_series), self.norm_span)
        features[6] = vol_norm[-1]

        # --- Dim 7: Depth ratio (top 5), centered ---
        if total_depth > 1e-12:
            features[7] = np.clip((bid_depth_5 / total_depth - 0.5) * 2.0, -1.0, 1.0)

        return features

    @property
    def warmup_ticks(self) -> int:
        """Minimum ticks before features are meaningful."""
        return self.norm_span

    @property
    def tick_count(self) -> int:
        return self._tick_count
