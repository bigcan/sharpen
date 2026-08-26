"""
Tests for Phase J IQN implementation.

Covers:
  1. NoisyLinear — shapes, noise reset, eval mode determinism
  2. QuantileEmbedding — output shapes
  3. IQNNetwork — forward shapes, noise propagation
  4. IQNAgent — predict, train_step, save/load roundtrip
  5. Stratified sampling — hold ratio enforcement
  6. Daily episodes — truncation + random start
"""
import os
import tempfile

import numpy as np
import torch

from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer
from finrl_pro_ds.agents.deepscalper.iqn_agent import IQNAgent
from finrl_pro_ds.agents.deepscalper.iqn_network import IQNNetwork, QuantileEmbedding
from finrl_pro_ds.agents.deepscalper.noisy_linear import NoisyLinear

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MICRO_CFG = {"input_size": 18, "private_input_size": 5, "hidden_size": 128, "encoder_type": "flat"}
MACRO_CFG = {"input_size": 15, "hidden_sizes": [128, 128]}
WINDOW = 15
B = 4  # batch size
N_ACTIONS = 3
N_QUANTILES = 8


def _make_iqn_network():
    return IQNNetwork(
        micro_config=MICRO_CFG,
        macro_config=MACRO_CFG,
        fusion_dim=128,
        n_actions=N_ACTIONS,
        embedding_dim=32,
        noisy_sigma0=0.5,
    )


def _make_iqn_agent(device="cpu"):
    net_cfg = {
        "micro_config": MICRO_CFG,
        "macro_config": MACRO_CFG,
        "action_space_dims": N_ACTIONS,
    }
    return IQNAgent(
        network_config=net_cfg,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=4,
        buffer_size=100,
        num_quantiles=N_QUANTILES,
        embedding_dim=32,
        noisy_sigma0=0.5,
        action_dims=N_ACTIONS,
        device=device,
    )


def _random_obs(batch=B):
    """Generate random observations matching network input shapes."""
    micro = torch.randn(batch, WINDOW, MICRO_CFG["input_size"])
    private = torch.randn(batch, WINDOW, MICRO_CFG["private_input_size"])
    macro = torch.randn(batch, MACRO_CFG["input_size"])
    return micro, private, macro


def _fill_agent_buffer(agent, n=20):
    """Push random transitions into agent's replay buffer."""
    for _ in range(n):
        state = {
            "micro": np.random.randn(WINDOW, MICRO_CFG["input_size"]).astype(np.float32),
            "macro": np.random.randn(MACRO_CFG["input_size"]).astype(np.float32),
            "private": np.random.randn(WINDOW, MICRO_CFG["private_input_size"]).astype(np.float32),
        }
        next_state = {
            "micro": np.random.randn(WINDOW, MICRO_CFG["input_size"]).astype(np.float32),
            "macro": np.random.randn(MACRO_CFG["input_size"]).astype(np.float32),
            "private": np.random.randn(WINDOW, MICRO_CFG["private_input_size"]).astype(np.float32),
        }
        action = np.array([np.random.randint(0, N_ACTIONS)], dtype=np.int64)
        agent.memory.push(state, action, np.random.randn(), next_state, False, 0.0)


# ===========================================================================
# NoisyLinear Tests
# ===========================================================================

class TestNoisyLinear:
    def test_output_shapes(self):
        layer = NoisyLinear(64, 32, sigma0=0.5)
        x = torch.randn(B, 64)
        out = layer(x)
        assert out.shape == (B, 32)

    def test_noise_reset_changes_buffers(self):
        layer = NoisyLinear(64, 32)
        eps_in_before = layer.epsilon_input.clone()
        eps_out_before = layer.epsilon_output.clone()
        layer.reset_noise()
        # Noise should change after reset
        assert not torch.equal(eps_in_before, layer.epsilon_input)
        assert not torch.equal(eps_out_before, layer.epsilon_output)

    def test_eval_mode_no_noise(self):
        layer = NoisyLinear(64, 32)
        x = torch.randn(1, 64)

        # Eval mode should be deterministic (uses mu only)
        layer.eval()
        out1 = layer(x)
        layer.reset_noise()  # noise changes shouldn't affect eval
        out2 = layer(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_train_mode_uses_noise(self):
        layer = NoisyLinear(64, 32)
        x = torch.randn(1, 64)

        layer.train()
        out1 = layer(x).clone()
        layer.reset_noise()
        out2 = layer(x)
        # Outputs should differ after noise reset in train mode
        assert not torch.allclose(out1, out2), "Train mode should use noise"


# ===========================================================================
# QuantileEmbedding Tests
# ===========================================================================

class TestQuantileEmbedding:
    def test_output_shapes(self):
        embed = QuantileEmbedding(embedding_dim=32, output_dim=128)
        tau = torch.rand(B, N_QUANTILES)
        out = embed(tau)
        assert out.shape == (B, N_QUANTILES, 128)

    def test_output_range(self):
        """Embedding output should be non-negative (ReLU activation)."""
        embed = QuantileEmbedding(embedding_dim=32, output_dim=128)
        tau = torch.rand(B, N_QUANTILES)
        out = embed(tau)
        assert (out >= 0).all(), "ReLU output should be non-negative"


# ===========================================================================
# IQNNetwork Tests
# ===========================================================================

class TestIQNNetwork:
    def test_forward_shapes(self):
        net = _make_iqn_network()
        micro, private, macro = _random_obs()
        tau = torch.rand(B, N_QUANTILES)

        q_tau, pred_vol, new_hidden = net(micro, private, macro, tau)

        assert q_tau.shape == (B, N_QUANTILES, N_ACTIONS)
        assert pred_vol.shape == (B, 1)

    def test_reset_noise_propagates(self):
        net = _make_iqn_network()
        # Capture noise before
        v1_eps = net.value_fc1.epsilon_input.clone()
        # Reset
        net.reset_noise()
        # Noise should change
        assert not torch.equal(v1_eps, net.value_fc1.epsilon_input)

    def test_different_tau_different_output(self):
        net = _make_iqn_network()
        net.eval()
        micro, private, macro = _random_obs(1)

        tau1 = torch.tensor([[0.1, 0.5]])
        tau2 = torch.tensor([[0.3, 0.9]])

        q1, _, _ = net(micro, private, macro, tau1)
        q2, _, _ = net(micro, private, macro, tau2)

        # Different tau should produce different quantile Q-values
        assert not torch.allclose(q1, q2, atol=1e-6)


# ===========================================================================
# IQNAgent Tests
# ===========================================================================

class TestIQNAgent:
    def test_predict_returns_valid_actions(self):
        agent = _make_iqn_agent()
        micro, private, macro = _random_obs()
        actions = agent.predict(micro, private, macro, deterministic=False)

        assert actions.shape == (B,)
        assert all(0 <= a < N_ACTIONS for a in actions)

    def test_predict_deterministic(self):
        agent = _make_iqn_agent()
        micro, private, macro = _random_obs(1)

        # Deterministic should be consistent (no noise)
        agent.policy_net.eval()
        actions = []
        for _ in range(5):
            a = agent.predict(micro, private, macro, deterministic=True)
            actions.append(a[0])
        # All should be the same (eval mode = no noise)
        assert len(set(actions)) == 1, "Deterministic predict should be consistent"

    def test_deterministic_predict_ignores_global_rng_state(self):
        """The real tripwire for the tau bug.

        Before the fix, predict() drew a fresh tau ~ U(0,1) even when
        deterministic=True, so on an untrained net (per-action Q-means nearly
        tied) the argmax flipped with the global RNG state. Reseeding between
        calls makes that dependence visible instead of leaving it to luck about
        which RNG state the suite happens to be in.
        """
        agent = _make_iqn_agent()
        micro, private, macro = _random_obs(1)

        actions = []
        for seed in range(8):
            torch.manual_seed(seed * 1000 + 17)
            actions.append(agent.predict(micro, private, macro, deterministic=True)[0])

        assert len(set(actions)) == 1, (
            "deterministic predict must not depend on global RNG state, got %r" % (actions,)
        )

    def test_deterministic_tau_is_the_midpoint_grid(self):
        agent = _make_iqn_agent()
        tau = agent._deterministic_tau(3)

        assert tau.shape == (3, N_QUANTILES)
        expected = (torch.arange(N_QUANTILES, dtype=torch.float32) + 0.5) / N_QUANTILES
        torch.testing.assert_close(tau[0], expected)
        # Every row identical, strictly inside (0, 1), and symmetric about 0.5 —
        # the last is what keeps the quadrature unbiased for the mean.
        torch.testing.assert_close(tau[1], tau[0])
        assert (tau > 0).all() and (tau < 1).all()
        assert tau[0].mean().item() == 0.5

    def test_deterministic_tau_grid_is_cached_and_rebuilt_on_change(self):
        agent = _make_iqn_agent()
        agent._deterministic_tau(2)
        cached = agent._eval_tau
        agent._deterministic_tau(5)
        assert agent._eval_tau is cached, "grid must be cached, not rebuilt per call"

        agent.num_quantiles = N_QUANTILES + 4
        rebuilt = agent._deterministic_tau(2)
        assert rebuilt.shape == (2, N_QUANTILES + 4)
        assert agent._eval_tau is not cached

    def test_stochastic_predict_still_samples_tau(self, monkeypatch):
        """Exploration must not be frozen by the fix."""
        agent = _make_iqn_agent()
        micro, private, macro = _random_obs(1)

        calls = []
        real_rand = torch.rand

        def spy(*args, **kwargs):
            calls.append(args)
            return real_rand(*args, **kwargs)

        monkeypatch.setattr(torch, "rand", spy)

        agent.predict(micro, private, macro, deterministic=True)
        assert calls == [], "deterministic predict must not draw tau"

        agent.predict(micro, private, macro, deterministic=False)
        assert calls, "stochastic predict must still sample tau"

    def test_train_step_returns_loss(self):
        agent = _make_iqn_agent()
        _fill_agent_buffer(agent, n=20)

        metrics = agent.train_step()
        assert metrics is not None
        assert "loss_total" in metrics
        assert "loss_qty" in metrics
        assert np.isfinite(metrics["loss_total"])

    def test_epsilon_always_zero(self):
        agent = _make_iqn_agent()
        assert agent.epsilon == 0.0
        agent.decay_epsilon()
        assert agent.epsilon == 0.0

    def test_save_load_roundtrip(self):
        agent = _make_iqn_agent()
        _fill_agent_buffer(agent, n=20)
        agent.train_step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "iqn_ckpt.pth")
            agent.save(path)

            agent2 = _make_iqn_agent()
            agent2.load(path)

            # Verify weights match
            net1 = agent.policy_net._orig_mod if agent._torch_compiled else agent.policy_net
            net2 = agent2.policy_net._orig_mod if agent2._torch_compiled else agent2.policy_net
            for p1, p2 in zip(net1.parameters(), net2.parameters()):
                assert torch.allclose(p1, p2), "Loaded weights should match saved weights"


# ===========================================================================
# Stratified Sampling Tests
# ===========================================================================

class TestStratifiedSampling:
    def test_hold_ratio_approximate(self):
        """Stratified sampling should produce approximately the requested hold ratio."""
        buf = FlatReplayBuffer(
            capacity=1000,
            micro_shape=(WINDOW, 18),
            macro_shape=(15,),
            private_shape=(WINDOW, 5),
            action_shape=(1,),
        )

        # Push 900 Hold (action=1) and 100 non-Hold transitions
        for i in range(1000):
            state = {
                "micro": np.zeros((WINDOW, 18), dtype=np.float32),
                "macro": np.zeros(15, dtype=np.float32),
                "private": np.zeros((WINDOW, 5), dtype=np.float32),
            }
            action = np.array([1 if i < 900 else 0], dtype=np.int64)
            buf.push(state, action, 0.0, state, False, 0.0)

        # Sample with 50% hold ratio
        _, actions, _, _, _, _ = buf.sample_stratified(256, hold_action=1, hold_ratio=0.5)
        hold_count = (actions[:, 0] == 1).sum()
        non_hold_count = (actions[:, 0] != 1).sum()

        # Should be approximately 128 each (allow ±20 for randomness)
        assert 108 <= hold_count <= 148, f"Expected ~128 hold, got {hold_count}"
        assert 108 <= non_hold_count <= 148, f"Expected ~128 non-hold, got {non_hold_count}"

    def test_fallback_to_uniform(self):
        """Should fallback to uniform sampling if not enough non-hold transitions."""
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(WINDOW, 18),
            macro_shape=(15,),
            private_shape=(WINDOW, 5),
            action_shape=(1,),
        )

        # Push only Hold transitions
        for _ in range(50):
            state = {
                "micro": np.zeros((WINDOW, 18), dtype=np.float32),
                "macro": np.zeros(15, dtype=np.float32),
                "private": np.zeros((WINDOW, 5), dtype=np.float32),
            }
            buf.push(state, np.array([1], dtype=np.int64), 0.0, state, False, 0.0)

        # Should not crash — falls back to uniform
        result = buf.sample_stratified(10, hold_action=1, hold_ratio=0.5)
        assert result is not None
        assert len(result[1]) == 10


# ===========================================================================
# Daily Episodes Tests
# ===========================================================================

class TestDailyEpisodes:
    def _make_env_config(self, episode_length=0, random_start=False):
        """Minimal env config for DeepScalperEnv."""
        return {
            "symbol": "GC",
            "initial_balance": 100000.0,
            "margin_requirement": 0.05,
            "maker_fee": 0.00001,
            "taker_fee": 0.000035,
            "window_size": 5,
            "max_drawdown_pct": 0.3,
            "episode_length": episode_length,
            "random_start": random_start,
            "action": {"discrete_dims": 3, "max_position": 5.0, "fixed_trade_qty": 0.2},
            "reward": {"scaling": 1.0, "volatility_horizon": 20, "sharpe_weight": 0.0},
        }

    def test_episode_length_config_parsing(self):
        """Verify episode_length and random_start are parsed from config."""
        from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

        cfg = self._make_env_config(episode_length=288, random_start=True)
        env = DeepScalperEnv(cfg)
        assert env.episode_length == 288
        assert env.random_start is True

    def test_default_no_episode_limit(self):
        """Without episode_length, env should not truncate early."""
        from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

        cfg = self._make_env_config()
        env = DeepScalperEnv(cfg)
        assert env.episode_length == 0
        assert env._episode_end == 0

    def test_random_start_different_indices(self):
        """Random start should produce different starting positions."""
        from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

        cfg = self._make_env_config(episode_length=10, random_start=True)
        env = DeepScalperEnv(cfg)

        # Mock handler with _len and _ptr
        class MockHandler:
            def __init__(self):
                self._len = 500
                self._ptr = 0
                self._col_to_idx = {}

            def reset(self):
                pass

            def step(self):
                return None

        env.handler = MockHandler()
        starts = set()
        for _ in range(20):
            env.reset()
            starts.add(env.handler._ptr)

        # Should have multiple different start positions
        assert len(starts) > 1, f"Random start should produce varied positions, got {starts}"
