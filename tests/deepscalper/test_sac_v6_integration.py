"""Integration tests for SAC v6 summary_stats mode.

Verifies the full agent pipeline:
  - FlatReplayBuffer handles zero-dim macro/private shapes
  - SACAgent.predict works in summary_stats mode
  - SACAgent._obs_to_buffer / _obs_batch_to_buffer produce correct shapes
  - SACAgent.update runs without error (mini forward+backward pass)
"""
import numpy as np
import pytest
import torch

from finrl_pro_ds.agents.sac.sac_agent import SACAgent
from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer


class TestFlatReplayBufferZeroDim:
    """Verify buffer works when macro/private have 0 columns."""

    def test_zero_dim_construction(self):
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(50,),
            macro_shape=(0,),
            private_shape=(0,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        assert buf._micro.shape == (100, 50)
        assert buf._macro.shape == (100, 0)
        assert buf._private.shape == (100, 0)

    def test_zero_dim_push_and_sample(self):
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(50,),
            macro_shape=(0,),
            private_shape=(0,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        state = {
            "micro": np.random.randn(50).astype(np.float32),
            "macro": np.array([], dtype=np.float32),
            "private": np.array([], dtype=np.float32),
        }
        next_state = {
            "micro": np.random.randn(50).astype(np.float32),
            "macro": np.array([], dtype=np.float32),
            "private": np.array([], dtype=np.float32),
        }
        for _ in range(20):
            buf.push(state, np.array([0.5], dtype=np.float32), 0.01, next_state, False)

        batch = buf.sample(8)
        assert batch[0]["micro"].shape == (8, 50)
        assert batch[0]["macro"].shape == (8, 0)
        assert batch[0]["private"].shape == (8, 0)

    def test_zero_dim_push_batch(self):
        buf = FlatReplayBuffer(
            capacity=100,
            micro_shape=(50,),
            macro_shape=(0,),
            private_shape=(0,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        n = 10
        states = {
            "micro": np.random.randn(n, 50).astype(np.float32),
            "macro": np.zeros((n, 0), dtype=np.float32),
            "private": np.zeros((n, 0), dtype=np.float32),
        }
        next_states = {
            "micro": np.random.randn(n, 50).astype(np.float32),
            "macro": np.zeros((n, 0), dtype=np.float32),
            "private": np.zeros((n, 0), dtype=np.float32),
        }
        buf.push_batch(
            states,
            np.random.randn(n, 1).astype(np.float32),
            np.random.randn(n).astype(np.float32),
            next_states,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )
        assert buf._size == n


class TestSACAgentSummaryStats:
    """Integration test: SACAgent with obs_mode='summary_stats'."""

    @pytest.fixture
    def agent(self):
        """Create a minimal SAC agent in summary_stats mode."""
        network_config = {
            "obs_mode": "summary_stats",
            "summary_input_dim": 50,
            "scale_encoder": {
                "input_size": 8,
                "channels": [32, 64, 64, 64],
                "kernel_size": 3,
                "output_dim": 64,
            },
            "private_dim": 5,
            "fusion_dim": 128,
            "window_size": 30,
            "action_dim": 1,
            "n_scales": 3,
        }
        return SACAgent(
            network_config=network_config,
            lr_actor=3e-4,
            lr_critic=3e-4,
            lr_alpha=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=8,
            buffer_size=200,
            initial_alpha=0.2,
            learning_starts=10,
            update_interval=1,
            actor_update_freq=1,
            gradient_clip=5.0,
            checkpoint_interval=10000,
            use_amp=False,
            torch_compile=False,
            device="cpu",
        )

    def _make_obs(self, n_scales=3, n_feat=5, batch=False):
        """Create a summary_stats observation dict."""
        n_summary = n_feat * 3  # mean, std, last per feature
        if batch:
            B = 4
            obs = {f"scale_{i}": np.random.randn(B, n_summary).astype(np.float32) for i in range(n_scales)}
            obs["private"] = np.random.randn(B, 5).astype(np.float32)
        else:
            obs = {f"scale_{i}": np.random.randn(n_summary).astype(np.float32) for i in range(n_scales)}
            obs["private"] = np.random.randn(5).astype(np.float32)
        return obs

    def test_obs_to_buffer_shape(self, agent):
        """Single obs converts to correct buffer format."""
        obs = self._make_obs()
        buf_obs = agent._obs_to_buffer(obs)
        assert buf_obs["micro"].shape == (50,), f"Expected (50,), got {buf_obs['micro'].shape}"
        assert buf_obs["macro"].shape == (0,)
        assert buf_obs["private"].shape == (0,)

    def test_obs_batch_to_buffer_shape(self, agent):
        """Batch obs converts to correct buffer format."""
        obs = self._make_obs(batch=True)
        buf_obs = agent._obs_batch_to_buffer(obs)
        assert buf_obs["micro"].shape == (4, 50), f"Expected (4, 50), got {buf_obs['micro'].shape}"
        assert buf_obs["macro"].shape == (4, 0)
        assert buf_obs["private"].shape == (4, 0)

    def test_predict(self, agent):
        """Agent predict returns valid actions."""
        obs = self._make_obs()
        buf_obs = agent._obs_to_buffer(obs)
        flat = torch.tensor(buf_obs["micro"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = agent.predict(flat, private=None, deterministic=True)
        assert action.shape == (1, 1)
        assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_push_and_update(self, agent):
        """Agent can push transitions and run update without error."""
        # Fill buffer past learning_starts
        for _ in range(20):
            obs = self._make_obs()
            next_obs = self._make_obs()
            buf_s = agent._obs_to_buffer(obs)
            buf_ns = agent._obs_to_buffer(next_obs)
            agent.replay_buffer.push(
                buf_s,
                np.array([0.1], dtype=np.float32),
                0.01,
                buf_ns,
                False,
            )

        # Run train steps — metrics only returned every 50 steps (perf optimization)
        # Just verify no exceptions are raised
        for _ in range(50):
            info = agent.train_step()
        assert info is not None, "Expected metrics after 50 train steps"
        assert "critic_loss" in info

    def test_full_cycle(self, agent):
        """Full predict → buffer → update cycle without error."""
        # Simulate 60 steps to exceed learning_starts and get metrics
        for step in range(60):
            obs = self._make_obs()
            buf_obs = agent._obs_to_buffer(obs)
            flat = torch.tensor(buf_obs["micro"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action = agent.predict(flat, private=None, deterministic=False)

            next_obs = self._make_obs()
            buf_ns = agent._obs_to_buffer(next_obs)
            agent.replay_buffer.push(
                buf_obs,
                action.cpu().numpy().flatten(),
                np.random.randn() * 0.01,
                buf_ns,
                False,
            )

            if step >= agent.learning_starts:
                agent.train_step()  # Just verify no exception
