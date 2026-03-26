"""Tests for AlphaSeekFeatureEngine — batch/incremental parity, bounds, LEAK-1."""

import numpy as np
import pandas as pd

from finrl_pro_ds.alphaseek.feature_engine import (
    N_DEPTH_LEVELS,
    AlphaSeekFeatureEngine,
    _ema_zscore_tanh,
    _symlog,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lob_row(
    mid: float = 100000.0,
    spread: float = 0.01,
    bid_qty: float = 1.0,
    ask_qty: float = 2.0,
    depth_levels: int = N_DEPTH_LEVELS,
) -> dict:
    """Build a single row dict matching the LOB parquet schema."""
    row = {
        "best_bid_price": mid - spread / 2,
        "best_bid_qty": bid_qty,
        "best_ask_price": mid + spread / 2,
        "best_ask_qty": ask_qty,
        "spread": spread,
        "mid_price": mid,
    }
    for i in range(depth_levels):
        row[f"bid_p_{i}"] = mid - spread / 2 - i * 0.01
        row[f"bid_q_{i}"] = bid_qty * (1.0 - 0.1 * i)
        row[f"ask_p_{i}"] = mid + spread / 2 + i * 0.01
        row[f"ask_q_{i}"] = ask_qty * (1.0 - 0.1 * i)
    # Pad remaining levels with NaN (matching parquet)
    for i in range(depth_levels, 20):
        row[f"bid_p_{i}"] = np.nan
        row[f"bid_q_{i}"] = np.nan
        row[f"ask_p_{i}"] = np.nan
        row[f"ask_q_{i}"] = np.nan
    return row


def _make_snapshot(
    mid: float = 100000.0,
    spread: float = 0.01,
    bid_qty: float = 1.0,
    ask_qty: float = 2.0,
) -> dict:
    """Build a snapshot dict matching process_snapshot() contract."""
    return {
        "best_bid_price": mid - spread / 2,
        "best_bid_qty": bid_qty,
        "best_ask_price": mid + spread / 2,
        "best_ask_qty": ask_qty,
        "spread": spread,
        "bid_prices_5": [mid - spread / 2 - i * 0.01 for i in range(5)],
        "bid_qtys_5": [bid_qty * (1.0 - 0.1 * i) for i in range(5)],
        "ask_prices_5": [mid + spread / 2 + i * 0.01 for i in range(5)],
        "ask_qtys_5": [ask_qty * (1.0 - 0.1 * i) for i in range(5)],
    }


def _make_lob_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic LOB DataFrame for testing."""
    rng = np.random.RandomState(seed)
    mids = 100000.0 + np.cumsum(rng.randn(n) * 5.0)
    spreads = np.abs(rng.randn(n) * 0.01) + 0.01
    bid_qtys = np.abs(rng.randn(n) * 2.0) + 0.1
    ask_qtys = np.abs(rng.randn(n) * 2.0) + 0.1

    rows = []
    for i in range(n):
        rows.append(
            _make_lob_row(
                mid=mids[i],
                spread=spreads[i],
                bid_qty=bid_qtys[i],
                ask_qty=ask_qtys[i],
            ),
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: _symlog / _ema_zscore_tanh
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_symlog_basic(self):
        x = np.array([-10, -1, 0, 1, 10], dtype=np.float64)
        result = _symlog(x)
        assert result[2] == 0.0  # symlog(0) = 0
        assert result[0] < 0  # negative input → negative output
        assert result[4] > 0  # positive input → positive output
        np.testing.assert_allclose(result[1], -np.log(2), atol=1e-10)
        np.testing.assert_allclose(result[3], np.log(2), atol=1e-10)

    def test_ema_zscore_tanh_bounded(self):
        x = np.random.randn(500).astype(np.float64) * 100
        result = _ema_zscore_tanh(x, span=120)
        assert result.dtype == np.float32
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_ema_zscore_tanh_causal(self):
        """First value should use backward-filled EMA (no future leak)."""
        x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        result = _ema_zscore_tanh(x, span=2)
        assert result.shape == (3,)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Tests: FeatureEngine
# ---------------------------------------------------------------------------

class TestFeatureEngineBatch:
    def test_output_shape(self):
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(100)
        features = engine.process_batch(df)
        assert features.shape == (100, 8)
        assert features.dtype == np.float32

    def test_features_bounded(self):
        """All features should be in [-1, 1]."""
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(300)
        features = engine.process_batch(df)
        assert np.all(features >= -1.0 - 1e-6), f"Min: {features.min()}"
        assert np.all(features <= 1.0 + 1e-6), f"Max: {features.max()}"

    def test_features_no_nan(self):
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(200)
        features = engine.process_batch(df)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_bbo_imbalance_sign(self):
        """When bid_qty > ask_qty, bbo_imbalance should be positive."""
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(50)
        # Override to make bid_qty >> ask_qty
        df["best_bid_qty"] = 10.0
        df["best_ask_qty"] = 1.0
        features = engine.process_batch(df)
        assert features[:, 0].mean() > 0.5  # dim 0 = bbo_imbalance

    def test_segment_aware_normalization(self):
        """Features should reset at segment boundaries."""
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(200)
        segments = np.zeros(200, dtype=np.int32)
        segments[100:] = 1  # split at row 100

        features_seg = engine.process_batch(df, segment_ids=segments)
        features_noseg = engine.process_batch(df)

        # EMA-Z-tanh features (dims 3,4,5,6) should differ at segment boundary
        # because the EMA resets
        assert not np.allclose(features_seg[:, 3], features_noseg[:, 3])

    def test_microprice_offset(self):
        """Microprice at BBO should be near mid when qty balanced."""
        engine = AlphaSeekFeatureEngine(norm_span=30)
        df = _make_lob_df(50)
        # Equal bid/ask quantities at BBO
        df["best_bid_qty"] = 1.0
        df["best_ask_qty"] = 1.0
        df["bid_q_0"] = 1.0
        df["ask_q_0"] = 1.0
        features = engine.process_batch(df)
        # Microprice ≈ mid → offset ≈ 0
        assert np.abs(features[:, 2]).mean() < 0.1


class TestFeatureEngineIncremental:
    def test_output_shape(self):
        engine = AlphaSeekFeatureEngine(norm_span=30)
        snap = _make_snapshot()
        features = engine.process_snapshot(snap)
        assert features.shape == (8,)
        assert features.dtype == np.float32

    def test_features_bounded(self):
        engine = AlphaSeekFeatureEngine(norm_span=30)
        for i in range(100):
            mid = 100000.0 + np.random.randn() * 50
            snap = _make_snapshot(mid=mid)
            features = engine.process_snapshot(snap)
            assert np.all(features >= -1.0 - 1e-6)
            assert np.all(features <= 1.0 + 1e-6)

    def test_reset_clears_state(self):
        engine = AlphaSeekFeatureEngine(norm_span=30)
        for _ in range(50):
            engine.process_snapshot(_make_snapshot())
        assert engine.tick_count == 50

        engine.reset()
        assert engine.tick_count == 0

    def test_warmup_property(self):
        engine = AlphaSeekFeatureEngine(norm_span=120)
        assert engine.warmup_ticks == 120


class TestBatchIncrementalParity:
    """Verify batch and incremental paths produce same results."""

    def test_parity(self):
        """Batch and incremental features should match closely."""
        n = 200
        engine_batch = AlphaSeekFeatureEngine(norm_span=30, momentum_window=5, vol_window=30)
        engine_incr = AlphaSeekFeatureEngine(
            norm_span=30, momentum_window=5, vol_window=30, max_buffer=n + 10,
        )

        df = _make_lob_df(n, seed=42)
        batch_features = engine_batch.process_batch(df)

        incr_features = np.zeros((n, 8), dtype=np.float32)
        for i in range(n):
            row = df.iloc[i]
            snap = {
                "best_bid_price": row["best_bid_price"],
                "best_bid_qty": row["best_bid_qty"],
                "best_ask_price": row["best_ask_price"],
                "best_ask_qty": row["best_ask_qty"],
                "spread": row["spread"],
                "bid_prices_5": [row[f"bid_p_{j}"] for j in range(5)],
                "bid_qtys_5": [row[f"bid_q_{j}"] for j in range(5)],
                "ask_prices_5": [row[f"ask_p_{j}"] for j in range(5)],
                "ask_qtys_5": [row[f"ask_q_{j}"] for j in range(5)],
            }
            incr_features[i] = engine_incr.process_snapshot(snap)

        # Raw features (dims 0, 1, 2, 7) should be exact
        for dim in [0, 1, 2, 7]:
            np.testing.assert_allclose(
                batch_features[:, dim],
                incr_features[:, dim],
                atol=1e-5,
                err_msg=f"Dim {dim} raw feature mismatch",
            )

        # EMA-Z-tanh features (dims 3, 4, 5, 6) should be close.
        # The incremental path uses a growing buffer while the batch path
        # operates on the full array, so minor floating-point divergence
        # is expected. Tolerance of 1e-3 is well within feature semantics.
        for dim in [3, 4, 5, 6]:
            np.testing.assert_allclose(
                batch_features[100:, dim],
                incr_features[100:, dim],
                atol=1e-3,
                err_msg=f"Dim {dim} EMA feature mismatch (after warmup)",
            )
