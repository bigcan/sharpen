"""
Tests for Prioritized Experience Replay (PER) implementation.

Covers:
  1. SumTree data structure operations
  2. PrioritizedReplayBuffer sampling and IS weights
  3. BDQ agent integration with PER buffer
"""
import pytest
import numpy as np
import torch

from finrl_pro_ds.agents.deepscalper.per_buffer import SumTree, PrioritizedReplayBuffer


# ---------------------------------------------------------------------------
# 1. SumTree Unit Tests
# ---------------------------------------------------------------------------

class TestSumTree:
    """Tests for the SumTree data structure."""

    def test_add_and_total(self):
        """Total should equal sum of all inserted priorities."""
        tree = SumTree(capacity=8)
        priorities = [1.0, 3.0, 5.0, 2.0]
        for i, p in enumerate(priorities):
            tree.add(p, f"data_{i}")
        assert tree.total() == pytest.approx(sum(priorities))
        assert tree.size == len(priorities)

    def test_capacity_wraparound(self):
        """When buffer is full, old entries are overwritten and total is updated."""
        tree = SumTree(capacity=4)
        # Fill completely
        for i in range(4):
            tree.add(1.0, f"data_{i}")
        assert tree.total() == pytest.approx(4.0)
        assert tree.size == 4

        # Overwrite first entry with higher priority
        tree.add(10.0, "data_new")
        assert tree.size == 4  # Still 4 (capacity)
        assert tree.total() == pytest.approx(13.0)  # 10 + 1 + 1 + 1

    def test_get_proportional_sampling(self):
        """get() should return the correct leaf for cumulative sum queries."""
        tree = SumTree(capacity=4)
        tree.add(1.0, "A")  # [0, 1)
        tree.add(3.0, "B")  # [1, 4)
        tree.add(1.0, "C")  # [4, 5)
        tree.add(5.0, "D")  # [5, 10)

        # Query middle of B's range
        _, _, data = tree.get(2.5)
        assert data == "B"

        # Query start of D's range
        _, _, data = tree.get(5.5)
        assert data == "D"

        # Query start of A's range
        _, _, data = tree.get(0.5)
        assert data == "A"

    def test_update_priority(self):
        """Updating a leaf's priority should propagate to root correctly."""
        tree = SumTree(capacity=4)
        idx = tree.add(1.0, "A")
        tree.add(1.0, "B")
        assert tree.total() == pytest.approx(2.0)

        # Double A's priority
        tree.update(idx, 5.0)
        assert tree.total() == pytest.approx(6.0)

    def test_min_priority(self):
        """min_priority should return the smallest active leaf priority."""
        tree = SumTree(capacity=8)
        tree.add(5.0, "A")
        tree.add(1.0, "B")
        tree.add(3.0, "C")
        assert tree.min_priority() == pytest.approx(1.0)

    def test_empty_tree(self):
        """Empty tree should have total=0 and min_priority=0."""
        tree = SumTree(capacity=4)
        assert tree.total() == 0.0
        assert tree.min_priority() == 0.0


# ---------------------------------------------------------------------------
# 2. PrioritizedReplayBuffer Tests
# ---------------------------------------------------------------------------

def _make_dummy_transition(index: int = 0):
    """Create a dummy transition matching BDQ agent format."""
    state = {
        "micro": np.random.randn(15, 30).astype(np.float32),
        "private": np.random.randn(15, 3).astype(np.float32),
        "macro": np.random.randn(15).astype(np.float32),
    }
    next_state = {
        "micro": np.random.randn(15, 30).astype(np.float32),
        "private": np.random.randn(15, 3).astype(np.float32),
        "macro": np.random.randn(15).astype(np.float32),
    }
    return (state, [0, 2, 1], float(np.random.randn()), next_state, False, 0.1)


class TestPrioritizedReplayBuffer:
    """Tests for the PrioritizedReplayBuffer."""

    def test_push_and_len(self):
        """Buffer length should match number of pushed transitions."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for i in range(50):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)
        assert len(buf) == 50

    def test_sample_returns_correct_shapes(self):
        """Sample should return 8-tuple with correct array shapes."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for i in range(100):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        batch_size = 16
        result = buf.sample(batch_size)
        assert len(result) == 8  # (states, actions, rewards, ..., indices, is_weights)

        states, actions, rewards, next_states, dones, aux_targets, indices, is_weights = result
        assert len(states) == batch_size
        assert len(indices) == batch_size
        assert is_weights.shape == (batch_size,)

    def test_is_weights_normalized(self):
        """Max IS weight should be 1.0 (normalized by maximum)."""
        buf = PrioritizedReplayBuffer(capacity=200, alpha=0.6, beta_start=0.4)
        for i in range(200):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        _, _, _, _, _, _, _, is_weights = buf.sample(32)
        assert is_weights.max() == pytest.approx(1.0, abs=1e-5)
        assert (is_weights > 0).all()
        assert (is_weights <= 1.0 + 1e-5).all()

    def test_high_priority_sampled_more(self):
        """Items with higher priority should be sampled more frequently."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)  # alpha=1 for full prioritization

        # Push 10 low-priority items
        for i in range(10):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        # Now update priorities: item 0 gets very high priority
        # Tree indices for first items start at capacity-1
        high_idx = buf.tree.capacity - 1  # First leaf
        buf.tree.update(high_idx, 100.0)

        # Sample many times and count how often the high-priority item appears
        sample_count = 0
        n_samples = 1000
        for _ in range(n_samples):
            _, _, _, _, _, _, indices, _ = buf.sample(1)
            if indices[0] == high_idx:
                sample_count += 1

        # With priority 100 vs 9 items at priority ~1, expect ~100/109 ≈ 92%
        assert sample_count > n_samples * 0.5, (
            f"High-priority item sampled only {sample_count}/{n_samples} times"
        )

    def test_update_priorities_shifts_distribution(self):
        """After updating priorities, sampling distribution should change."""
        buf = PrioritizedReplayBuffer(capacity=50, alpha=0.6)
        for i in range(50):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        # Record initial total
        initial_total = buf.tree.total()

        # Update all priorities to high values
        indices = np.arange(buf.tree.capacity - 1, buf.tree.capacity - 1 + 50)
        td_errors = np.ones(50) * 10.0
        buf.update_priorities(indices, td_errors)

        # Total should have changed
        new_total = buf.tree.total()
        assert new_total != pytest.approx(initial_total, abs=0.1)

    def test_beta_annealing(self):
        """Beta should anneal from beta_start toward 1.0 as frames progress."""
        buf = PrioritizedReplayBuffer(
            capacity=100, alpha=0.6, beta_start=0.4, beta_frames=1000
        )
        for i in range(100):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        beta_initial = buf.beta
        assert beta_initial == pytest.approx(0.4, abs=0.01)

        # Sample enough to anneal
        for _ in range(100):
            buf.sample(10)  # Each sample advances frame by batch_size=10

        beta_later = buf.beta
        assert beta_later > beta_initial, f"Beta didn't increase: {beta_later}"

    def test_capacity_overflow(self):
        """Buffer should handle overflow (more pushes than capacity) gracefully."""
        buf = PrioritizedReplayBuffer(capacity=10, alpha=0.6)
        for i in range(25):
            s, a, r, ns, d, aux = _make_dummy_transition(i)
            buf.push(s, a, r, ns, d, aux)

        assert len(buf) == 10  # Capped at capacity
        result = buf.sample(5)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# 3. BDQ Agent Integration with PER
# ---------------------------------------------------------------------------

def _make_per_agent():
    """Create a BDQ agent with PER enabled for testing."""
    from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ

    network_config = {
        "micro_config": {"input_size": 30, "private_input_size": 2,
                         "hidden_size": 64, "rnn_type": "LSTM"},
        "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
        "action_space_dims": [3, 5, 5],
    }
    return DeepScalperBDQ(
        network_config=network_config,
        lr=1e-4,
        gamma=0.99,
        buffer_size=256,
        batch_size=32,
        target_update_freq=10,
        use_per=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=10000,
        device="cpu",
    )


class TestBDQWithPER:
    """Integration tests: BDQ agent training with PER buffer."""

    def test_per_agent_init(self):
        """Agent with use_per=True should use PrioritizedReplayBuffer."""
        agent = _make_per_agent()
        assert isinstance(agent.memory, PrioritizedReplayBuffer)
        assert agent.use_per is True

    def test_train_step_with_per(self):
        """train_step should succeed and return PER-specific metrics."""
        agent = _make_per_agent()

        # Populate buffer
        for _ in range(128):
            s, a, r, ns, d, aux = _make_dummy_transition()
            agent.memory.push(s, a, r, ns, d, aux)

        metrics = agent.train_step()
        assert metrics is not None
        assert np.isfinite(metrics["loss_total"]), f"Loss not finite: {metrics['loss_total']}"
        assert "per_beta" in metrics, "Missing per_beta in metrics"
        assert "per_max_priority" in metrics, "Missing per_max_priority in metrics"
        assert 0.0 < metrics["per_beta"] <= 1.0

    def test_multi_step_training(self):
        """Agent should train for 100+ steps without crashing."""
        agent = _make_per_agent()

        # Populate buffer
        for _ in range(128):
            s, a, r, ns, d, aux = _make_dummy_transition()
            agent.memory.push(s, a, r, ns, d, aux)

        # Run 100 training steps
        losses = []
        for step in range(100):
            metrics = agent.train_step()
            assert metrics is not None, f"train_step returned None at step {step}"
            assert np.isfinite(metrics["loss_total"]), (
                f"Non-finite loss at step {step}: {metrics['loss_total']}"
            )
            losses.append(metrics["loss_total"])

        # Sanity: loss should have varied (not stuck at exact same value)
        assert len(set(losses)) > 1, "Loss never changed across 100 steps"

    def test_backward_compat_no_per(self):
        """Agent with use_per=False should use FlatReplayBuffer."""
        from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
        from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer

        network_config = {
            "micro_config": {"input_size": 30, "private_input_size": 2,
                             "hidden_size": 64, "rnn_type": "LSTM"},
            "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
            "action_space_dims": [3, 5, 5],
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
        assert agent.use_per is False

        # Train step should work without PER metrics
        for _ in range(64):
            s, a, r, ns, d, aux = _make_dummy_transition()
            agent.memory.push(s, a, r, ns, d, aux)

        metrics = agent.train_step()
        assert metrics is not None
        assert "per_beta" not in metrics
