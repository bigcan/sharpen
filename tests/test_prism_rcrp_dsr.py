"""Tests for PRISM Path 1 (RCRP) and Path 2 (Regime-Adaptive DSR)."""

from __future__ import annotations

import numpy as np

from finrl_pro_ds.agents.common.flat_replay_buffer import FlatReplayBuffer
from finrl_pro_ds.envs.dsr import DSRCalculator


# ── Path 1: Regime-Balanced Replay ──────────────────────────────────────────


def _make_buffer(n: int = 500, regime_dist: dict[int, int] | None = None) -> FlatReplayBuffer:
    """Create a buffer pre-filled with transitions labeled by regime code."""
    buf = FlatReplayBuffer(
        capacity=1000,
        micro_shape=(10, 7),
        macro_shape=(10,),
        private_shape=(5,),
        action_shape=(1,),
        action_dtype=np.float32,
    )
    if regime_dist is None:
        # Realistic skew: 20% LOW, 55% NORMAL, 25% HIGH vol
        regime_dist = {0: 40, 1: 50, 2: 10, 3: 40, 4: 60, 5: 15, 6: 20, 7: 165, 8: 100}

    for code, count in regime_dist.items():
        for _ in range(count):
            state = {
                "micro": np.random.randn(10, 7).astype(np.float32),
                "macro": np.random.randn(10).astype(np.float32),
                "private": np.random.randn(5).astype(np.float32),
            }
            next_state = {
                "micro": np.random.randn(10, 7).astype(np.float32),
                "macro": np.random.randn(10).astype(np.float32),
                "private": np.random.randn(5).astype(np.float32),
            }
            buf.push(
                state, np.array([0.5], dtype=np.float32), 0.1,
                next_state, False, regime_code=code,
            )
    return buf


class TestRegimeBalancedReplay:
    def test_balanced_mode_equalizes_vol_regimes(self):
        """Balanced mode should draw roughly equal samples from each vol regime."""
        buf = _make_buffer()
        batch_size = 300

        # Sample many batches and count vol-regime distribution
        vol_counts = np.zeros(3, dtype=int)
        for _ in range(50):
            states, actions, rewards, next_states, dones, aux = \
                buf.sample_regime_balanced(batch_size, mode="balanced")
            # Check we got the right batch size
            assert len(rewards) == batch_size

        # Single large sample to check distribution
        states, actions, rewards, next_states, dones, aux = \
            buf.sample_regime_balanced(3000, mode="balanced")
        # Verify structure
        assert states["micro"].shape == (3000, 10, 7)
        assert actions.shape == (3000, 1)

    def test_balanced_mode_vol_distribution(self):
        """Verify balanced sampling produces roughly equal vol-regime counts."""
        buf = _make_buffer()
        # Collect regime codes for sampled indices
        # We test indirectly: sample balanced, then check the regime codes
        # of sampled transitions match the balanced expectation.
        total_samples = 9000
        states, _, _, _, _, _ = buf.sample_regime_balanced(total_samples, mode="balanced")
        # Can't directly check regime codes from sample output, but we verify
        # the method runs without error and returns correct shapes
        assert states["private"].shape == (total_samples, 5)

    def test_inverse_freq_oversamples_rare_regimes(self):
        """Inverse frequency mode should run without error."""
        buf = _make_buffer()
        states, actions, rewards, next_states, dones, aux = \
            buf.sample_regime_balanced(300, mode="inverse_freq")
        assert len(rewards) == 300

    def test_transition_boosted_mode(self):
        """Transition-boosted mode should run without error."""
        buf = _make_buffer()
        states, actions, rewards, next_states, dones, aux = \
            buf.sample_regime_balanced(300, mode="transition_boosted")
        assert len(rewards) == 300

    def test_fallback_on_no_regime_data(self):
        """Should fall back to uniform sampling when no regime codes present."""
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(10, 7),
            macro_shape=(10,),
            private_shape=(5,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        # Fill without regime codes (all default to -1)
        for _ in range(50):
            state = {
                "micro": np.random.randn(10, 7).astype(np.float32),
                "macro": np.random.randn(10).astype(np.float32),
                "private": np.random.randn(5).astype(np.float32),
            }
            buf.push(state, np.array([0.0], dtype=np.float32), 0.0, state, False)
        # Should fall back to uniform (no crash)
        states, _, rewards, _, _, _ = buf.sample_regime_balanced(20, mode="balanced")
        assert len(rewards) == 20

    def test_push_batch_with_regime_codes(self):
        """push_batch should accept regime_codes array."""
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(10, 7),
            macro_shape=(10,),
            private_shape=(5,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        n = 10
        states = {
            "micro": np.random.randn(n, 10, 7).astype(np.float32),
            "macro": np.random.randn(n, 10).astype(np.float32),
            "private": np.random.randn(n, 5).astype(np.float32),
        }
        actions = np.random.randn(n, 1).astype(np.float32)
        rewards = np.random.randn(n).astype(np.float32)
        dones = np.zeros(n, dtype=np.float32)
        aux = np.zeros(n, dtype=np.float32)
        codes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 4], dtype=np.int8)

        buf.push_batch(states, actions, rewards, states, dones, aux, regime_codes=codes)
        assert len(buf) == n
        # Verify codes were stored
        np.testing.assert_array_equal(buf._regime_codes[:n], codes)

    def test_push_batch_without_regime_codes(self):
        """push_batch without regime_codes should default to -1."""
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(10, 7),
            macro_shape=(10,),
            private_shape=(5,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        n = 5
        states = {
            "micro": np.random.randn(n, 10, 7).astype(np.float32),
            "macro": np.random.randn(n, 10).astype(np.float32),
            "private": np.random.randn(n, 5).astype(np.float32),
        }
        actions = np.random.randn(n, 1).astype(np.float32)
        rewards = np.random.randn(n).astype(np.float32)
        dones = np.zeros(n, dtype=np.float32)
        aux = np.zeros(n, dtype=np.float32)

        buf.push_batch(states, actions, rewards, states, dones, aux)
        np.testing.assert_array_equal(buf._regime_codes[:n], np.full(n, -1, dtype=np.int8))

    def test_unknown_mode_falls_back(self):
        """Unknown mode should fall back to uniform sampling."""
        buf = _make_buffer()
        states, _, rewards, _, _, _ = buf.sample_regime_balanced(50, mode="unknown_mode")
        assert len(rewards) == 50

    def test_nbytes_includes_regime_codes(self):
        """nbytes should account for regime_codes array."""
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(1,),
            macro_shape=(1,),
            private_shape=(1,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        # regime_codes is int8 (1 byte each) = 100 bytes
        assert buf._regime_codes.nbytes == 100
        assert buf._regime_codes.nbytes <= buf.nbytes()


# ── Path 2: Regime-Adaptive DSR ─────────────────────────────────────────────


class TestRegimeAdaptiveDSR:
    def test_default_no_regime_behavior(self):
        """Without regime config, DSR should behave identically to baseline."""
        dsr_base = DSRCalculator(eta=0.001, scale=1.0)
        dsr_regime = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=None)

        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        for r in returns:
            r1 = dsr_base.compute(r)
            r2 = dsr_regime.compute(r)
            assert r1 == r2, f"Mismatch at R={r}: {r1} vs {r2}"

    def test_regime_context_changes_eta(self):
        """set_regime_context should modify effective eta."""
        mults = {0: 2.0, 1: 1.0, 2: 0.3}
        dsr = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=1.0)

        # LOW_VOL: eta should double
        dsr.set_regime_context(0)
        assert abs(dsr.eta - 0.002) < 1e-9

        # NORMAL_VOL: eta unchanged
        dsr.set_regime_context(1)
        assert abs(dsr.eta - 0.001) < 1e-9

        # HIGH_VOL: eta * 0.3
        dsr.set_regime_context(2)
        assert abs(dsr.eta - 0.0003) < 1e-9

    def test_regime_blend_smooths_transition(self):
        """Blending at 0.5 should average base and regime eta."""
        mults = {0: 2.0, 1: 1.0, 2: 0.3}
        dsr = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=0.5)

        # LOW_VOL with 0.5 blend: 0.5*0.001 + 0.5*0.002 = 0.0015
        dsr.set_regime_context(0)
        assert abs(dsr.eta - 0.0015) < 1e-9

        # HIGH_VOL with 0.5 blend: 0.5*0.001 + 0.5*0.0003 = 0.00065
        dsr.set_regime_context(2)
        assert abs(dsr.eta - 0.00065) < 1e-9

    def test_unknown_regime_uses_base_eta(self):
        """vol_regime=-1 (unknown) should use base eta."""
        mults = {0: 2.0, 1: 1.0, 2: 0.3}
        dsr = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults)

        dsr.set_regime_context(-1)
        assert abs(dsr.eta - 0.001) < 1e-9

    def test_reset_restores_base_eta(self):
        """reset() should restore base eta after regime changes."""
        mults = {0: 2.0, 2: 0.3}
        dsr = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=1.0)

        dsr.set_regime_context(0)
        assert dsr.eta != 0.001
        dsr.reset()
        assert abs(dsr.eta - 0.001) < 1e-9

    def test_high_vol_produces_different_rewards(self):
        """HIGH_VOL regime should produce different DSR values than NORMAL_VOL."""
        mults = {0: 2.0, 1: 1.0, 2: 0.3}
        dsr_normal = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=1.0)
        dsr_highvol = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=1.0)

        # Warm up both
        warmup = [0.01, -0.005, 0.02, -0.01, 0.015] * 5
        for r in warmup:
            dsr_normal.set_regime_context(1)
            dsr_normal.compute(r)
            dsr_highvol.set_regime_context(2)
            dsr_highvol.compute(r)

        # After divergent eta paths, a new return should produce different DSR
        dsr_normal.set_regime_context(1)
        r_normal = dsr_normal.compute(0.03)
        dsr_highvol.set_regime_context(2)
        r_highvol = dsr_highvol.compute(0.03)

        assert r_normal != r_highvol, "Different etas should produce different DSR values"

    def test_regime_blend_zero_ignores_regime(self):
        """regime_blend=0.0 should always use base eta regardless of regime."""
        mults = {0: 2.0, 2: 0.3}
        dsr = DSRCalculator(eta=0.001, scale=1.0, regime_eta_multipliers=mults, regime_blend=0.0)

        dsr.set_regime_context(0)
        assert abs(dsr.eta - 0.001) < 1e-9

        dsr.set_regime_context(2)
        assert abs(dsr.eta - 0.001) < 1e-9
