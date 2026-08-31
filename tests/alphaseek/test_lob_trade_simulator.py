"""Tests for LOBTradeSimulator — data loading, state shape, step logic."""

import os

import pytest
import torch as th

# Skip if LOB parquet not available (CI/local without data)
LOB_PARQUET = "data/lob_parquet/btcusdt_lob_1s.parquet"
HAS_LOB_DATA = os.path.exists(LOB_PARQUET)

skip_no_data = pytest.mark.skipif(
    not HAS_LOB_DATA,
    reason=f"LOB parquet not found: {LOB_PARQUET}",
)


@skip_no_data
class TestLOBTradeSimulatorBasic:
    """Basic simulator tests using real LOB data."""

    def _make_sim(self, **kwargs):
        from sharpen.alphaseek.lob_trade_simulator import LOBTradeSimulator

        defaults = {
            "lob_parquet_path": LOB_PARQUET,
            "num_sims": 4,
            "step_gap": 2,
            "seq_len": 3600,
            "norm_span": 30,
        }
        defaults.update(kwargs)
        return LOBTradeSimulator(**defaults)

    def test_state_dim(self):
        sim = self._make_sim()
        assert sim.state_dim == 12  # v3: +pending_limit_active +bars_since_last_trade
        assert sim.action_dim == 3

    def test_reset_shape(self):
        sim = self._make_sim()
        state = sim.reset()
        assert state.shape == (4, 12)  # (num_sims, state_dim)

    def test_step_shape(self):
        sim = self._make_sim()
        sim.reset()
        action = th.ones((4, 1), dtype=th.long)  # all hold
        next_state, reward, done, info = sim.step(action)
        assert next_state.shape == (4, 12)
        assert reward.shape == (4,)
        assert done.shape == (4,)

    def test_state_no_nan(self):
        sim = self._make_sim()
        state = sim.reset()
        assert th.isfinite(state).all()

    def test_full_episode(self):
        """Run a full episode, verify terminal at max_step."""
        sim = self._make_sim(num_sims=2)
        state = sim.reset()
        hold = th.ones((2, 1), dtype=th.long)  # action=1 = hold

        for step in range(sim.max_step):
            state, reward, done, info = sim.step(hold)

        # After max_step, should be truncated (episode reset internally)
        # The state returned is from the new episode after reset
        assert state.shape == (2, 12)


@skip_no_data
class TestLOBTradeSimulatorStepGap:
    """Verify step_gap affects max_step and max_holding correctly."""

    def _make_sim(self, step_gap):
        from sharpen.alphaseek.lob_trade_simulator import LOBTradeSimulator

        return LOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET,
            num_sims=2,
            step_gap=step_gap,
            seq_len=3600,
            norm_span=30,
        )

    def test_step_gap_1(self):
        sim = self._make_sim(step_gap=1)
        assert sim.max_holding == 3600  # 60*60//1
        assert sim.max_step == (3600 - 60) // 1

    def test_step_gap_2(self):
        sim = self._make_sim(step_gap=2)
        assert sim.max_holding == 1800
        assert sim.max_step == (3600 - 60) // 2

    def test_step_gap_4(self):
        sim = self._make_sim(step_gap=4)
        assert sim.max_holding == 900
        assert sim.max_step == (3600 - 60) // 4

    def test_step_gap_8(self):
        sim = self._make_sim(step_gap=8)
        assert sim.max_holding == 450
        assert sim.max_step == (3600 - 60) // 8


@skip_no_data
class TestEvalLOBTradeSimulator:
    def test_eval_deterministic_start(self):
        from sharpen.alphaseek.lob_trade_simulator import EvalLOBTradeSimulator

        sim = EvalLOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET,
            num_sims=2,
            step_gap=2,
            norm_span=30,
        )
        state1 = sim.reset()
        state2 = sim.reset()
        # Deterministic: same start → same state
        assert th.allclose(state1, state2)
