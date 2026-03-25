"""Test observation parity between training MultiScaleOHLCVHandler and LiveObsBuilder.

This is the CRITICAL test for live trading correctness. If observations
differ between training and live, the agent will behave unpredictably.

Tests:
    1. Bootstrap LiveObsBuilder from the same 1-min data used by training handler
    2. Step both through N bars
    3. Assert np.allclose() on all scale features and private state
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from finrl_pro_ds.data.multiscale_handler import (
    MultiScaleOHLCVHandler,
)
from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder


def _generate_synthetic_ohlcv(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV data for testing.

    Creates realistic-looking price data with trends, volatility clustering,
    and volume variation. NOT for strategy evaluation — only for feature parity.
    """
    rng = np.random.RandomState(seed)

    # Generate price series with random walk + momentum
    returns = rng.normal(0, 0.001, n_bars)  # ~0.1% per minute
    returns = np.cumsum(returns)
    close = 50000.0 * np.exp(returns)  # Start at ~50000 (BTC-like)

    # Generate OHLCV from close
    spread = close * 0.0005  # 5 bps spread
    high = close + rng.uniform(0, 1, n_bars) * spread
    low = close - rng.uniform(0, 1, n_bars) * spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.lognormal(10, 1, n_bars)  # Realistic volume distribution

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=n_bars,
        freq="1min",
        tz="UTC",
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestLiveObsParity:
    """Test that LiveObsBuilder produces identical observations to training."""

    # Use smaller scales for faster tests with less data
    SCALES = [5, 15, 60]
    WINDOW_SIZE = 30
    NORM_SPAN = 120
    N_FEATURES = 8

    def _make_training_handler(self, parquet_path: str) -> MultiScaleOHLCVHandler:
        """Create a training-mode handler from a parquet file."""
        return MultiScaleOHLCVHandler(
            file_path=parquet_path,
            ticker="SYNTH",
            feature_config={
                "scales": self.SCALES,
                "window_size": self.WINDOW_SIZE,
                "norm_span": self.NORM_SPAN,
            },
        )

    def _make_live_builder(self) -> LiveObsBuilder:
        """Create a LiveObsBuilder with matching config."""
        return LiveObsBuilder(
            scales=self.SCALES,
            window_size=self.WINDOW_SIZE,
            norm_span=self.NORM_SPAN,
            n_features=self.N_FEATURES,
        )

    def test_scale_features_parity(self, tmp_path):
        """Scale feature arrays must be numerically identical.

        Strategy: Step training handler to the END of data so its last
        observation matches the live builder's full-data observation.
        Both produce features from the same underlying functions on
        the same data — they should be numerically identical.
        """
        # Generate enough data for all scales with window_size
        df = _generate_synthetic_ohlcv(n_bars=5000)
        parquet_path = str(tmp_path / "synth_1min.parquet")
        df.to_parquet(parquet_path, index=False)

        # Training handler — step to the very end
        handler = self._make_training_handler(parquet_path)
        handler.reset()

        last_step = None
        while True:
            step_data = handler.step()
            if step_data is None:
                break
            last_step = step_data

        assert last_step is not None, "Training handler produced no observations"

        # Live builder — bootstrap from same full DataFrame
        builder = self._make_live_builder()
        builder.bootstrap_from_dataframe(df)

        # Both should produce identical feature windows for the last bar
        live_obs = builder.get_observation(
            current_position=0.5,
            prev_close=last_step.get("close", 50000.0) * 0.999,
            current_close=last_step.get("close", 50000.0),
            timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        for i in range(len(self.SCALES)):
            key = f"scale_{i}"
            training_arr = last_step[key]
            live_arr = live_obs[key]

            assert training_arr.shape == live_arr.shape, (
                f"{key} shape mismatch: training={training_arr.shape}, live={live_arr.shape}"
            )
            assert np.allclose(training_arr, live_arr, atol=1e-5), (
                f"{key} values differ! Max diff={np.max(np.abs(training_arr - live_arr)):.8f}"
            )

    def test_private_state_construction(self):
        """Private state must match V7 env logic."""
        df = _generate_synthetic_ohlcv(n_bars=2000)
        builder = self._make_live_builder()
        builder.bootstrap_from_dataframe(df)

        obs = builder.get_observation(
            current_position=0.75,
            prev_close=50000.0,
            current_close=50050.0,
            timestamp=datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc),
        )

        private = obs["private"]
        assert private.shape == (5,), f"Private state shape: {private.shape}"
        assert private.dtype == np.float32

        # Check each dimension
        # 0: position
        assert abs(private[0] - 0.75) < 1e-6

        # 1: unrealized PnL proxy
        # ret_bps = (50050 - 50000) / 50000 * 10000 = 10 bps
        # pnl_proxy = 0.75 * 10 / 100 = 0.075
        assert abs(private[1] - 0.075) < 1e-4

        # 2-3: time encoding for 14:30 UTC
        14 * 60 + 30  # = 870
        expected_sin = float(np.sin(2 * np.pi * 870 / 1440.0))
        expected_cos = float(np.cos(2 * np.pi * 870 / 1440.0))
        assert abs(private[2] - expected_sin) < 1e-5
        assert abs(private[3] - expected_cos) < 1e-5

        # 4: ATR ratio — just check it's in valid range [0, 1]
        assert 0.0 <= private[4] <= 1.0

    def test_incremental_update_consistency(self):
        """Feeding bars one-by-one should produce same result as full bootstrap."""
        df = _generate_synthetic_ohlcv(n_bars=2000)

        # Method A: Bootstrap from full DataFrame
        builder_full = self._make_live_builder()
        builder_full.bootstrap_from_dataframe(df)

        # Method B: Bootstrap from first 1500, then update with remaining 500
        builder_incremental = self._make_live_builder()
        builder_incremental.bootstrap_from_dataframe(df.iloc[:1500].copy())
        builder_incremental.update(df.iloc[1500:].copy())

        obs_full = builder_full.get_observation(
            current_position=0.0,
            prev_close=50000.0,
            current_close=50000.0,
            timestamp=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
        )
        obs_inc = builder_incremental.get_observation(
            current_position=0.0,
            prev_close=50000.0,
            current_close=50000.0,
            timestamp=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
        )

        for i in range(len(self.SCALES)):
            key = f"scale_{i}"
            assert np.allclose(obs_full[key], obs_inc[key], atol=1e-5), (
                f"{key}: full vs incremental differ! "
                f"Max diff={np.max(np.abs(obs_full[key] - obs_inc[key])):.8f}"
            )

    def test_observation_no_nan(self):
        """Observations should never contain NaN or Inf."""
        df = _generate_synthetic_ohlcv(n_bars=2000)
        builder = self._make_live_builder()
        builder.bootstrap_from_dataframe(df)

        obs = builder.get_observation(
            current_position=0.0,
            prev_close=50000.0,
            current_close=50000.0,
            timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        for key, arr in obs.items():
            assert not np.any(np.isnan(arr)), f"{key} contains NaN"
            assert not np.any(np.isinf(arr)), f"{key} contains Inf"

    def test_observation_shapes(self):
        """Observation dict must have correct shapes."""
        df = _generate_synthetic_ohlcv(n_bars=2000)
        builder = self._make_live_builder()
        builder.bootstrap_from_dataframe(df)

        obs = builder.get_observation(
            current_position=0.0,
            prev_close=50000.0,
            current_close=50000.0,
            timestamp=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        )

        for i in range(len(self.SCALES)):
            key = f"scale_{i}"
            assert key in obs, f"Missing key: {key}"
            assert obs[key].shape == (self.WINDOW_SIZE, self.N_FEATURES), (
                f"{key} shape: {obs[key].shape}"
            )
            assert obs[key].dtype == np.float32

        assert "private" in obs
        assert obs["private"].shape == (5,)
        assert obs["private"].dtype == np.float32

    def test_is_ready_flag(self):
        """is_ready should be False before bootstrap, True after."""
        builder = self._make_live_builder()
        assert not builder.is_ready

        df = _generate_synthetic_ohlcv(n_bars=2000)
        builder.bootstrap_from_dataframe(df)
        assert builder.is_ready

    def test_atr_accessors(self):
        """ATR accessors should return valid values after bootstrap."""
        df = _generate_synthetic_ohlcv(n_bars=2000)
        builder = self._make_live_builder()
        builder.bootstrap_from_dataframe(df)

        atr = builder.get_current_atr()
        assert atr > 0, f"ATR should be positive: {atr}"

        close = builder.get_current_close()
        assert close > 0, f"Close should be positive: {close}"

        ts = builder.get_latest_timestamp()
        assert ts is not None
