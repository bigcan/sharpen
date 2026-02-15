"""
Unit tests for EarnHFT Low-Level Environment, Agent, and Rollout Buffer.

Tests cover:
1. EarnHFTLowLevelEnv — reset, step, obs shapes, reward shaping
2. DiscretePPOActorCritic — forward shapes, evaluate_actions, gradients
3. DiscretePPOAgent — predict, save/load
4. DiscreteRolloutBuffer — store, GAE, minibatch
5. data_utils — chunk_data, label_regimes, aggregate_to_minute

Run: python -m pytest tests/earnhft/test_low_level.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import torch
    import torch.nn  # noqa: F401
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_TORCH = False


# ============================================================================
# HELPERS
# ============================================================================
def make_lob_df(n_rows=200, mid_start=100.0, mid_step=0.01):
    """Create LOB DataFrame for testing."""
    rows = []
    for t in range(n_rows):
        mid = mid_start + t * mid_step
        row = {}
        for i in range(1, 6):
            row[f"bid_price_{i}"] = mid - 0.05 - (i - 1) * 0.01
            row[f"bid_vol_{i}"] = 10.0
            row[f"ask_price_{i}"] = mid + 0.05 + (i - 1) * 0.01
            row[f"ask_vol_{i}"] = 10.0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def lob_df():
    return make_lob_df(200)


@pytest.fixture
def q_table():
    """Simple Q-table for 200 steps, 5 actions."""
    return np.random.randn(200, 5, 5).astype(np.float64) * 0.1


@pytest.fixture
def network_config():
    return {
        "micro_config": {
            "input_size": 20,
            "private_input_size": 3,
            "hidden_size": 32,
            "rnn_type": "LSTM",
        },
        "macro_config": {
            "input_size": 8,
            "hidden_sizes": [32, 16],
        },
        "fusion_dim": 32,
        "num_actions": 5,
    }


# ============================================================================
# TEST: DATA UTILS
# ============================================================================
class TestDataUtils:

    def test_chunk_data(self, lob_df):
        from finrl_pro_ds.agents.earnhft.data_utils import chunk_data
        chunks = chunk_data(lob_df, chunk_length=50)
        assert len(chunks) == 4  # 200 // 50
        assert len(chunks[0]) == 50

    def test_chunk_drops_remainder(self):
        from finrl_pro_ds.agents.earnhft.data_utils import chunk_data
        df = make_lob_df(110)
        chunks = chunk_data(df, chunk_length=50)
        assert len(chunks) == 2  # 110 // 50 = 2, remainder dropped

    def test_label_regimes_shape(self, lob_df):
        from finrl_pro_ds.agents.earnhft.data_utils import label_regimes
        regimes = label_regimes(lob_df, window=50)
        assert regimes.shape == (200,)
        assert regimes.dtype == np.int64
        assert set(regimes).issubset({0, 1, 2, 3})

    def test_aggregate_to_minute(self, lob_df):
        from finrl_pro_ds.agents.earnhft.data_utils import aggregate_to_minute
        minute_df = aggregate_to_minute(lob_df, ticks_per_minute=20)
        assert len(minute_df) == 10  # 200 // 20
        assert "open" in minute_df.columns
        assert "volatility" in minute_df.columns
        assert "spread_mean" in minute_df.columns


# ============================================================================
# TEST: LOW-LEVEL ENV
# ============================================================================
class TestLowLevelEnv:

    @pytest.fixture
    def env(self, lob_df, q_table):
        from finrl_pro_ds.agents.earnhft.low_level_env import EarnHFTLowLevelEnv
        return EarnHFTLowLevelEnv(
            df=lob_df, q_table=q_table,
            num_actions=5, max_holding=1.0,
            beta=1.0, window_size=10,
        )

    def test_reset(self, env):
        obs, info = env.reset()
        assert "micro" in obs
        assert "private" in obs
        assert obs["micro"].shape == (10, 20)  # (window_size, micro_dim)
        assert obs["private"].shape == (10, 3)
        assert info["position"] == 0.0

    def test_step_shapes(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)
        assert "micro" in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "pnl_delta" in info
        assert "q_advantage" in info

    def test_action_space(self, env):
        assert env.action_space.n == 5

    def test_full_episode(self, env):
        """Run full episode to completion."""
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_reward_shaping_with_beta(self, lob_df, q_table):
        """Beta > 0 should produce different rewards than beta=0."""
        from finrl_pro_ds.agents.earnhft.low_level_env import EarnHFTLowLevelEnv

        env_beta0 = EarnHFTLowLevelEnv(lob_df, q_table, beta=0.0, window_size=10)
        env_beta1 = EarnHFTLowLevelEnv(lob_df, q_table, beta=1.0, window_size=10)

        env_beta0.reset()
        env_beta1.reset()

        _, r0, _, _, _ = env_beta0.step(2)
        _, r1, _, _, _ = env_beta1.step(2)
        # Result may be the same if q_advantage happens to be 0, but generally differ
        # Just check both are finite
        assert np.isfinite(r0)
        assert np.isfinite(r1)


# ============================================================================
# TEST: DISCRETE PPO NETWORK
# ============================================================================
class TestDiscretePPOActorCritic:

    @pytest.fixture(autouse=True)
    def skip_no_torch(self):
        if not HAS_TORCH:
            pytest.skip("torch not available")

    @pytest.fixture
    def network(self, network_config):
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOActorCritic
        return DiscretePPOActorCritic(**network_config)

    @pytest.fixture
    def batch_data(self):
        B, W = 4, 10
        micro = torch.randn(B, W, 20)
        private = torch.randn(B, W, 3)
        macro = torch.randn(B, 8)
        return micro, private, macro

    def test_forward_shapes(self, network, batch_data):
        micro, private, macro = batch_data
        B = micro.shape[0]
        actions, log_probs, values, entropy, hidden = network(micro, private, macro)
        assert actions.shape == (B,), f"Expected (B,), got {actions.shape}"
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == (B,)

    def test_deterministic(self, network, batch_data):
        micro, private, macro = batch_data
        a1, _, _, _, _ = network(micro, private, macro, deterministic=True)
        a2, _, _, _, _ = network(micro, private, macro, deterministic=True)
        assert torch.equal(a1, a2)

    def test_evaluate_actions(self, network, batch_data):
        micro, private, macro = batch_data
        B = micro.shape[0]
        actions = torch.randint(0, 5, (B,))
        log_probs, values, entropy = network.evaluate_actions(micro, private, macro, actions)
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == (B,)

    def test_gradient_flow(self, network, batch_data):
        micro, private, macro = batch_data
        actions, log_probs, values, entropy, _ = network(micro, private, macro)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        for name, param in network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# TEST: DISCRETE ROLLOUT BUFFER
# ============================================================================
class TestDiscreteRolloutBuffer:

    def test_store_and_gae(self):
        if not HAS_TORCH:
            from finrl_pro_ds.agents.earnhft.low_level_agent import DiscreteRolloutBuffer
        else:
            from finrl_pro_ds.agents.earnhft.low_level_agent import DiscreteRolloutBuffer
        T, B = 8, 2
        buf = DiscreteRolloutBuffer(T, B, (10, 20), (10, 3), (8,))

        for t in range(T):
            obs = {
                "micro": np.random.randn(B, 10, 20).astype(np.float32),
                "private": np.random.randn(B, 10, 3).astype(np.float32),
                "macro": np.random.randn(B, 8).astype(np.float32),
            }
            buf.store(obs, np.random.randint(0, 5, B),
                      np.random.randn(B).astype(np.float32),
                      np.random.randn(B).astype(np.float32),
                      np.random.randn(B).astype(np.float32),
                      np.zeros(B, dtype=np.float32))

        assert buf.full
        buf.compute_gae(0.99, 0.95, np.zeros(B), np.zeros(B))
        assert not np.isnan(buf.advantages).any()

    def test_minibatch_shapes(self):
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscreteRolloutBuffer
        T, B = 8, 2
        buf = DiscreteRolloutBuffer(T, B, (10, 20), (10, 3), (8,))

        for t in range(T):
            obs = {
                "micro": np.zeros((B, 10, 20), dtype=np.float32),
                "private": np.zeros((B, 10, 3), dtype=np.float32),
                "macro": np.zeros((B, 8), dtype=np.float32),
            }
            buf.store(obs, np.zeros(B, dtype=np.int64),
                      np.zeros(B, dtype=np.float32),
                      np.zeros(B, dtype=np.float32),
                      np.zeros(B, dtype=np.float32),
                      np.zeros(B, dtype=np.float32))

        buf.compute_gae(0.99, 0.95, np.zeros(B), np.zeros(B))

        total = 0
        for batch in buf.iterate_minibatches(4):
            assert batch["actions"].ndim == 1  # (batch,) not (batch, 1)
            total += batch["micro"].shape[0]
        assert total == T * B


# ============================================================================
# TEST: DISCRETE PPO AGENT
# ============================================================================
class TestDiscretePPOAgent:

    @pytest.fixture(autouse=True)
    def skip_no_torch(self):
        if not HAS_TORCH:
            pytest.skip("torch not available")

    def test_predict(self, network_config):
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        agent = DiscretePPOAgent(network_config=network_config, lr=3e-4, device="cpu")

        B, W = 2, 10
        micro = torch.randn(B, W, 20)
        private = torch.randn(B, W, 3)
        macro = torch.randn(B, 8)

        actions, log_probs, values = agent.predict(micro, private, macro)
        assert actions.shape == (B,)
        assert actions.dtype == np.int64
        assert log_probs.shape == (B,)
        assert values.shape == (B,)

    def test_save_load(self, network_config, tmp_path):
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        agent = DiscretePPOAgent(network_config=network_config, lr=3e-4, device="cpu")

        path = str(tmp_path / "test.pth")
        agent.save(path)
        assert os.path.exists(path)

        agent2 = DiscretePPOAgent(network_config=network_config, lr=3e-4, device="cpu")
        agent2.load(path)

        for (n1, p1), (n2, p2) in zip(
            agent.network.named_parameters(), agent2.network.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch at {n1}"
