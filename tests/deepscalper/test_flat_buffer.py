"""
Tests for FlatReplayBuffer — pre-allocated numpy replay buffer.

Covers:
  1. Core buffer operations (push, sample, capacity, wraparound)
  2. Shape and dtype correctness
  3. Memory footprint validation
  4. Drop-in agent integration (train_step with flat buffer)
"""
import numpy as np
import pytest

from sharpen.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MICRO_SHAPE = (15, 30)
MACRO_SHAPE = (15,)
PRIVATE_SHAPE = (2,)
ACTION_SHAPE = (3,)


def _make_buffer(capacity=100):
    return FlatReplayBuffer(
        capacity=capacity,
        micro_shape=MICRO_SHAPE,
        macro_shape=MACRO_SHAPE,
        private_shape=PRIVATE_SHAPE,
        action_shape=ACTION_SHAPE,
    )


def _push_transition(buf, idx=0):
    """Push a deterministic transition keyed by idx for verification."""
    state = {
        "micro": np.full(MICRO_SHAPE, idx, dtype=np.float32),
        "private": np.full(PRIVATE_SHAPE, idx + 0.1, dtype=np.float32),
        "macro": np.full(MACRO_SHAPE, idx + 0.2, dtype=np.float32),
    }
    next_state = {
        "micro": np.full(MICRO_SHAPE, idx + 100, dtype=np.float32),
        "private": np.full(PRIVATE_SHAPE, idx + 100.1, dtype=np.float32),
        "macro": np.full(MACRO_SHAPE, idx + 100.2, dtype=np.float32),
    }
    action = [idx % 3, idx % 5, idx % 5]
    reward = float(idx) * 0.01
    done = (idx % 7 == 0)
    aux = float(idx) * 0.001
    buf.push(state, action, reward, next_state, done, aux)


# ---------------------------------------------------------------------------
# 1. Core Buffer Operations
# ---------------------------------------------------------------------------

class TestFlatBufferCore:

    def test_push_and_len(self):
        """Length should track number of pushed transitions."""
        buf = _make_buffer(100)
        assert len(buf) == 0
        for i in range(50):
            _push_transition(buf, i)
        assert len(buf) == 50

    def test_capacity_full(self):
        """Length should cap at capacity when full."""
        buf = _make_buffer(10)
        for i in range(10):
            _push_transition(buf, i)
        assert len(buf) == 10

    def test_capacity_wraparound(self):
        """Oldest data should be overwritten after capacity is exceeded."""
        buf = _make_buffer(10)
        for i in range(25):
            _push_transition(buf, i)
        assert len(buf) == 10
        # Sample all — values should come from indices 15..24
        states, _, _, _, _, _ = buf.sample(10)
        micro_vals = states["micro"][:, 0, 0]  # First element of each micro
        # All values should be >= 15 (oldest surviving transition)
        assert micro_vals.min() >= 15.0

    def test_empty_sample_raises(self):
        """Sampling from empty buffer should raise (numpy will error on size=0)."""
        buf = _make_buffer(10)
        with pytest.raises(ValueError):
            buf.sample(1)


# ---------------------------------------------------------------------------
# 2. Shape and Dtype Correctness
# ---------------------------------------------------------------------------

class TestFlatBufferShapes:

    def test_sample_returns_correct_shapes(self):
        """All returned arrays should have correct batch dimension and shapes."""
        buf = _make_buffer(100)
        for i in range(100):
            _push_transition(buf, i)

        batch_size = 16
        states, actions, rewards, next_states, dones, aux_targets = buf.sample(batch_size)

        # State dicts
        assert states["micro"].shape == (batch_size, *MICRO_SHAPE)
        assert states["macro"].shape == (batch_size, *MACRO_SHAPE)
        assert states["private"].shape == (batch_size, *PRIVATE_SHAPE)

        # Next state dicts
        assert next_states["micro"].shape == (batch_size, *MICRO_SHAPE)
        assert next_states["macro"].shape == (batch_size, *MACRO_SHAPE)
        assert next_states["private"].shape == (batch_size, *PRIVATE_SHAPE)

        # Scalar arrays
        assert actions.shape == (batch_size, *ACTION_SHAPE)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert aux_targets.shape == (batch_size,)

    def test_sample_dtypes(self):
        """Dtypes should be float32 for observations, int64 for actions."""
        buf = _make_buffer(50)
        for i in range(50):
            _push_transition(buf, i)

        states, actions, rewards, next_states, dones, aux_targets = buf.sample(8)

        assert states["micro"].dtype == np.float32
        assert states["macro"].dtype == np.float32
        assert states["private"].dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32
        assert aux_targets.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Data Integrity
# ---------------------------------------------------------------------------

class TestFlatBufferData:

    def test_push_sample_roundtrip(self):
        """Push 1 item, sample 1 item — values should match exactly."""
        buf = _make_buffer(10)
        state = {
            "micro": np.ones(MICRO_SHAPE, dtype=np.float32) * 42.0,
            "private": np.ones(PRIVATE_SHAPE, dtype=np.float32) * 7.0,
            "macro": np.ones(MACRO_SHAPE, dtype=np.float32) * 3.14,
        }
        next_state = {
            "micro": np.ones(MICRO_SHAPE, dtype=np.float32) * 43.0,
            "private": np.ones(PRIVATE_SHAPE, dtype=np.float32) * 8.0,
            "macro": np.ones(MACRO_SHAPE, dtype=np.float32) * 2.72,
        }
        buf.push(state, [1, 2, 3], 0.5, next_state, True, 0.99)

        s, a, r, ns, d, aux = buf.sample(1)
        np.testing.assert_array_equal(s["micro"][0], state["micro"])
        np.testing.assert_array_equal(s["private"][0], state["private"])
        np.testing.assert_array_almost_equal(s["macro"][0], state["macro"])
        np.testing.assert_array_equal(ns["micro"][0], next_state["micro"])
        np.testing.assert_array_equal(a[0], [1, 2, 3])
        assert r[0] == pytest.approx(0.5)
        assert d[0] == pytest.approx(1.0)   # True → 1.0
        assert aux[0] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 4. Memory Footprint
# ---------------------------------------------------------------------------

class TestFlatBufferMemory:

    def test_nbytes_matches_expectation(self):
        """Pre-allocated memory should match theoretical calculation."""
        cap = 1000
        buf = _make_buffer(cap)

        expected = (
            2 * cap * np.prod(MICRO_SHAPE) * 4 +    # micro + next_micro (float32)
            2 * cap * np.prod(MACRO_SHAPE) * 4 +     # macro + next_macro
            2 * cap * np.prod(PRIVATE_SHAPE) * 4 +   # private + next_private
            cap * np.prod(ACTION_SHAPE) * 8 +         # actions (int64)
            3 * cap * 4 +                              # rewards, dones, aux (float32)
            cap * 1                                    # RCRP regime codes (int8)
        )
        assert buf.nbytes() == expected

    def test_nbytes_large_capacity(self):
        """Sanity: 2M capacity should be in the ~7 GB range for unit test shapes."""
        cap = 2_000_000
        buf = FlatReplayBuffer(
            capacity=cap,
            micro_shape=MICRO_SHAPE,
            macro_shape=MACRO_SHAPE,
            private_shape=PRIVATE_SHAPE,
            action_shape=ACTION_SHAPE,
        )
        gb = buf.nbytes() / (1024**3)
        assert 5 < gb < 10, f"Expected ~7 GB, got {gb:.1f} GB"


# ---------------------------------------------------------------------------
# 5. push_batch Tests
# ---------------------------------------------------------------------------

class TestFlatBufferBatchPush:

    def test_push_batch_matches_sequential(self):
        """Batch push of N transitions should match N sequential pushes."""
        N = 20
        buf_seq = _make_buffer(100)
        buf_batch = _make_buffer(100)

        # Build batched arrays
        micros = np.arange(N, dtype=np.float32).reshape(N, 1, 1) * np.ones(MICRO_SHAPE, dtype=np.float32)
        macros = np.arange(N, dtype=np.float32).reshape(N, 1) * np.ones(MACRO_SHAPE, dtype=np.float32)
        privates = np.arange(N, dtype=np.float32).reshape(N, 1) * np.ones(PRIVATE_SHAPE, dtype=np.float32)
        next_micros = micros + 100
        next_macros = macros + 100
        next_privates = privates + 100
        actions_arr = np.zeros((N, *ACTION_SHAPE), dtype=np.int64)
        rewards_arr = np.arange(N, dtype=np.float32) * 0.01
        dones_arr = np.zeros(N, dtype=np.float32)
        aux_arr = np.arange(N, dtype=np.float32) * 0.001

        # Sequential
        for i in range(N):
            state = {"micro": micros[i], "macro": macros[i], "private": privates[i]}
            next_state = {"micro": next_micros[i], "macro": next_macros[i], "private": next_privates[i]}
            buf_seq.push(state, actions_arr[i], rewards_arr[i], next_state, dones_arr[i], aux_arr[i])

        # Batch
        states = {"micro": micros, "macro": macros, "private": privates}
        next_states = {"micro": next_micros, "macro": next_macros, "private": next_privates}
        buf_batch.push_batch(states, actions_arr, rewards_arr, next_states, dones_arr, aux_arr)

        assert len(buf_seq) == len(buf_batch)
        np.testing.assert_array_equal(buf_seq._micro[:N], buf_batch._micro[:N])
        np.testing.assert_array_equal(buf_seq._rewards[:N], buf_batch._rewards[:N])
        np.testing.assert_array_equal(buf_seq._next_micro[:N], buf_batch._next_micro[:N])

    def test_push_batch_wraparound(self):
        """Batch push should handle circular wraparound correctly."""
        cap = 10
        buf = _make_buffer(cap)

        # Fill to position 8
        for i in range(8):
            _push_transition(buf, i)

        # Batch push 5 items — wraps around from ptr=8, needs positions 8,9,0,1,2
        N = 5
        micros = np.ones((N, *MICRO_SHAPE), dtype=np.float32) * 99
        macros = np.ones((N, *MACRO_SHAPE), dtype=np.float32) * 99
        privates = np.ones((N, *PRIVATE_SHAPE), dtype=np.float32) * 99
        states = {"micro": micros, "macro": macros, "private": privates}
        next_states = {"micro": micros + 1, "macro": macros + 1, "private": privates + 1}
        actions_arr = np.zeros((N, *ACTION_SHAPE), dtype=np.int64)
        rewards_arr = np.ones(N, dtype=np.float32) * 99
        dones_arr = np.zeros(N, dtype=np.float32)
        aux_arr = np.zeros(N, dtype=np.float32)

        buf.push_batch(states, actions_arr, rewards_arr, next_states, dones_arr, aux_arr)

        assert len(buf) == cap
        # Positions 8, 9 should have value 99
        assert buf._micro[8, 0, 0] == pytest.approx(99.0)
        assert buf._micro[9, 0, 0] == pytest.approx(99.0)
        # Positions 0, 1, 2 should also have been overwritten
        assert buf._micro[0, 0, 0] == pytest.approx(99.0)
        assert buf._micro[1, 0, 0] == pytest.approx(99.0)
        assert buf._micro[2, 0, 0] == pytest.approx(99.0)
        # Position 3 should still be the old value (3.0)
        assert buf._micro[3, 0, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 6. Agent Integration (drop-in replacement)
# ---------------------------------------------------------------------------

def _push_integration_transition(buf, idx=0):
    """Push a transition matching integration test shapes (windowed private, 2-branch action)."""
    state = {
        "micro": np.full((15, 30), idx, dtype=np.float32),
        "private": np.full((15, 5), idx + 0.1, dtype=np.float32),
        "macro": np.full((15,), idx + 0.2, dtype=np.float32),
    }
    next_state = {
        "micro": np.full((15, 30), idx + 100, dtype=np.float32),
        "private": np.full((15, 5), idx + 100.1, dtype=np.float32),
        "macro": np.full((15,), idx + 100.2, dtype=np.float32),
    }
    action = [idx % 5, idx % 9]
    buf.push(state, action, float(idx) * 0.01, next_state, (idx % 7 == 0), float(idx) * 0.001)


class TestFlatBufferAgentIntegration:

    def test_agent_uses_flat_buffer(self):
        """Agent with use_per=False should create FlatReplayBuffer."""
        from sharpen.agents.deepscalper.bdq_agent import DeepScalperBDQ

        network_config = {
            "micro_config": {"input_size": 30, "private_input_size": 5,
                             "hidden_size": 64, "rnn_type": "LSTM"},
            "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
            "action_space_dims": (5, 9),  # Sprint 7: 2-branch
        }
        agent = DeepScalperBDQ(
            network_config=network_config,
            lr=1e-4, gamma=0.99,
            buffer_size=256, batch_size=32,
            target_update_freq=10,
            use_per=False,
            device="cpu",
        )
        assert isinstance(agent.memory, FlatReplayBuffer)

    def test_train_step_with_flat_buffer(self):
        """train_step should produce finite loss with FlatReplayBuffer."""
        from sharpen.agents.deepscalper.bdq_agent import DeepScalperBDQ

        network_config = {
            "micro_config": {"input_size": 30, "private_input_size": 5,
                             "hidden_size": 64, "rnn_type": "LSTM"},
            "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
            "action_space_dims": (5, 9),  # Sprint 7: 2-branch
        }
        agent = DeepScalperBDQ(
            network_config=network_config,
            lr=1e-4, gamma=0.99,
            buffer_size=256, batch_size=32,
            target_update_freq=10,
            use_per=False,
            device="cpu",
        )

        # Populate buffer
        for i in range(128):
            _push_integration_transition(agent.memory, i)

        # Run train step
        metrics = agent.train_step()
        assert metrics is not None
        assert np.isfinite(metrics["loss_total"]), f"Loss not finite: {metrics['loss_total']}"
        assert "per_beta" not in metrics

    def test_multi_step_training_flat(self):
        """50 train steps should all produce finite loss."""
        from sharpen.agents.deepscalper.bdq_agent import DeepScalperBDQ

        network_config = {
            "micro_config": {"input_size": 30, "private_input_size": 5,
                             "hidden_size": 64, "rnn_type": "LSTM"},
            "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
            "action_space_dims": (5, 9),  # Sprint 7: 2-branch
        }
        agent = DeepScalperBDQ(
            network_config=network_config,
            lr=1e-4, gamma=0.99,
            buffer_size=256, batch_size=32,
            target_update_freq=10,
            use_per=False,
            device="cpu",
        )

        for i in range(128):
            _push_integration_transition(agent.memory, i)

        losses = []
        for step in range(50):
            metrics = agent.train_step()
            assert metrics is not None, f"train_step returned None at step {step}"
            assert np.isfinite(metrics["loss_total"]), (
                f"Non-finite loss at step {step}: {metrics['loss_total']}"
            )
            losses.append(metrics["loss_total"])

        assert len(set(losses)) > 1, "Loss never changed across 50 steps"
