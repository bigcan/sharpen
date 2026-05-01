"""Tests for AlphaSeekStateBuilder — protocol compliance, warmup, NaN handling."""

import numpy as np
import torch

from finrl_pro_ds.alphaseek.feature_engine import AlphaSeekFeatureEngine
from finrl_pro_ds.alphaseek.state_builder import AlphaSeekStateBuilder


def _make_snapshot(mid: float = 100000.0, spread: float = 0.01) -> dict:
    return {
        "best_bid_price": mid - spread / 2,
        "best_bid_qty": 1.5,
        "best_ask_price": mid + spread / 2,
        "best_ask_qty": 2.0,
        "spread": spread,
        "bid_prices_5": [mid - spread / 2 - i * 0.01 for i in range(5)],
        "bid_qtys_5": [1.5 * (1.0 - 0.1 * i) for i in range(5)],
        "ask_prices_5": [mid + spread / 2 + i * 0.01 for i in range(5)],
        "ask_qtys_5": [2.0 * (1.0 - 0.1 * i) for i in range(5)],
    }


class TestStateBuilderProtocol:
    """Verify StateBuilderProtocol compliance."""

    def test_get_state_shape(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        for _ in range(15):
            builder.ingest_snapshot(_make_snapshot())
        state = builder.get_state(position=0, holding=0)
        assert state.shape == (1, 12)  # v3: +pending_active +bars_since_last_trade
        assert state.dtype == torch.float32

    def test_get_state_device_cpu(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        builder.ingest_snapshot(_make_snapshot())
        state = builder.get_state(position=0, holding=0)
        assert state.device == torch.device("cpu")

    def test_is_ready_before_warmup(self):
        engine = AlphaSeekFeatureEngine(norm_span=20)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        assert not builder.is_ready
        for _ in range(10):
            builder.ingest_snapshot(_make_snapshot())
        assert not builder.is_ready

    def test_is_ready_after_warmup(self):
        engine = AlphaSeekFeatureEngine(norm_span=20)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        for _ in range(20):
            builder.ingest_snapshot(_make_snapshot())
        assert builder.is_ready

    def test_mid_price(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        builder.ingest_snapshot(_make_snapshot(mid=50000.0))
        assert abs(builder.get_current_mid_price() - 50000.0) < 0.01

    def test_spread(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        builder.ingest_snapshot(_make_snapshot(spread=0.5))
        assert abs(builder.get_current_spread() - 0.5) < 0.001


class TestStateBuilderPositionNorm:
    """Verify position and holding normalization in state vector."""

    def test_position_zero(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu", max_position=1)
        builder.ingest_snapshot(_make_snapshot())
        state = builder.get_state(position=0, holding=0)
        assert state[0, 0].item() == 0.0  # position_norm
        assert state[0, 1].item() == 0.0  # holding_norm

    def test_position_positive(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu", max_position=1)
        builder.ingest_snapshot(_make_snapshot())
        state = builder.get_state(position=1, holding=900)
        assert state[0, 0].item() == 1.0  # position_norm = 1/1
        assert abs(state[0, 1].item() - 0.5) < 0.01  # holding_norm = 900/1800

    def test_position_negative(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu", max_position=1)
        builder.ingest_snapshot(_make_snapshot())
        state = builder.get_state(position=-1, holding=0)
        assert state[0, 0].item() == -1.0  # position_norm = -1/1


class TestStateBuilderPreWarmup:
    """State should be zeros before any snapshot ingested."""

    def test_pre_warmup_zeros(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        state = builder.get_state(position=0, holding=0)
        assert state.shape == (1, 12)  # v3
        assert torch.all(state == 0.0)


class TestStateBuilderReset:
    def test_reset_clears_state(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        for _ in range(15):
            builder.ingest_snapshot(_make_snapshot())
        assert builder.is_ready

        builder.reset()
        assert not builder.is_ready
        assert builder.get_current_mid_price() == 0.0


class TestStateBuilderNaN:
    """Features should never contain NaN or Inf."""

    def test_no_nan_in_state(self):
        engine = AlphaSeekFeatureEngine(norm_span=10)
        builder = AlphaSeekStateBuilder(engine, device="cpu")
        rng = np.random.RandomState(42)
        for i in range(50):
            mid = 100000.0 + rng.randn() * 100
            builder.ingest_snapshot(_make_snapshot(mid=mid))
            state = builder.get_state(position=rng.choice([-1, 0, 1]), holding=i)
            assert torch.isfinite(state).all(), f"NaN/Inf at tick {i}"
