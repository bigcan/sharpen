"""
Unit tests for EarnHFT High-Level Router.

Tests cover:
1. EarnHFTHighLevelEnv — reset, step, full episode
2. ReplayBuffer — push, sample
3. RouterDQN — select_action, train_step (when torch available)

Run: python -m pytest tests/earnhft/test_router.py -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def minute_features():
    """60 minutes of 8-dim features."""
    return np.random.randn(60, 8).astype(np.float32)


@pytest.fixture
def minute_pnls():
    """PnL per minute per agent (3 agents)."""
    return np.random.randn(60, 3).astype(np.float32) * 0.01


# ============================================================================
# TEST: HIGH-LEVEL ENV
# ============================================================================
class TestHighLevelEnv:

    def test_reset(self, minute_features, minute_pnls):
        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size=3)
        obs, info = env.reset()
        assert obs.shape == (8,)
        assert info["minute"] == 0

    def test_step(self, minute_features, minute_pnls):
        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size=3)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert info["selected_agent"] == 1
        assert len(info["all_pnls"]) == 3

    def test_full_episode(self, minute_features, minute_pnls):
        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size=3)
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 60

    def test_action_space(self, minute_features, minute_pnls):
        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size=3)
        assert env.action_space.n == 3

    def test_reward_matches_selected(self, minute_features, minute_pnls):
        """Reward should equal PnL of selected agent at that minute."""
        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size=3)
        env.reset()
        for minute_idx in range(5):
            agent_idx = minute_idx % 3
            _, reward, _, _, _ = env.step(agent_idx)
            expected = minute_pnls[minute_idx, agent_idx]
            assert abs(reward - expected) < 1e-6


# ============================================================================
# TEST: REPLAY BUFFER
# ============================================================================
class TestReplayBuffer:

    def test_push_and_len(self):
        from finrl_pro_ds.agents.earnhft.router_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(np.zeros(8), 0, 1.0, np.zeros(8), False)
        assert len(buf) == 10

    def test_sample_shapes(self):
        from finrl_pro_ds.agents.earnhft.router_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(np.random.randn(8), i % 3, 0.1, np.random.randn(8), False)
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.shape == (5, 8)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)

    def test_capacity_overflow(self):
        from finrl_pro_ds.agents.earnhft.router_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(np.zeros(8), 0, float(i), np.zeros(8), False)
        assert len(buf) == 5


# ============================================================================
# TEST: ROUTER DQN (requires torch)
# ============================================================================
class TestRouterDQN:

    @pytest.fixture(autouse=True)
    def skip_no_torch(self):
        try:
            import torch
            import torch.nn  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch not available")

    def test_select_action(self):
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN
        router = RouterDQN(obs_dim=8, pool_size=3, device="cpu")
        state = np.random.randn(8).astype(np.float32)
        action = router.select_action(state, eval_mode=True)
        assert 0 <= action < 3

    def test_epsilon_decay(self):
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN
        router = RouterDQN(obs_dim=8, pool_size=3, epsilon_start=1.0,
                           epsilon_end=0.05, epsilon_decay=100, device="cpu")
        assert router.epsilon == 1.0
        router._step_count = 50
        assert 0.05 < router.epsilon < 1.0
        router._step_count = 200
        assert router.epsilon == 0.05

    def test_train_step(self):
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN
        router = RouterDQN(obs_dim=8, pool_size=3, batch_size=4, device="cpu")

        # Fill buffer
        for _ in range(10):
            s = np.random.randn(8).astype(np.float32)
            ns = np.random.randn(8).astype(np.float32)
            router.store_transition(s, np.random.randint(3), 0.1, ns, False)

        loss = router.train_step()
        assert loss is not None
        assert loss >= 0
