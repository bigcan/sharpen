"""Tests for GMGP1-v6 summary-stats observation mode.

Verifies that:
  - MultiScaleOHLCVHandler summary_stats mode produces correct shapes
  - SummaryStatsEncoder forward pass produces correct output shape
  - Actor network works in summary_stats mode
  - Window mode behavior is unchanged
"""
import numpy as np
import torch

from sharpen.agents.sac.networks import (
    SACActorNetwork,
    SACCriticNetwork,
    SummaryStatsEncoder,
)
from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler


class TestSummaryStatsEncoder:
    def test_forward_shape(self):
        """Encoder produces (B, fusion_dim) from flat input."""
        encoder = SummaryStatsEncoder(input_dim=50, fusion_dim=256, hidden_dim=128)
        x = torch.randn(4, 50)
        out = encoder(x)
        assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"

    def test_param_count(self):
        """~40K params — much smaller than CNN encoder."""
        encoder = SummaryStatsEncoder(input_dim=50, fusion_dim=256, hidden_dim=128)
        n_params = sum(p.numel() for p in encoder.parameters())
        assert n_params < 60_000, f"Expected <60K params, got {n_params}"

    def test_tc_alignment(self):
        """Input padding handles non-multiple-of-8 dims."""
        encoder = SummaryStatsEncoder(input_dim=50, fusion_dim=256)
        x = torch.randn(2, 50)
        out = encoder(x)  # 50 → tc_align(50) = 56 internally
        assert out.shape == (2, 256)


class TestActorSummaryMode:
    def test_actor_forward(self):
        """Actor in summary_stats mode produces valid (mu, log_sigma)."""
        scale_cfg = {"summary_input_dim": 50}
        actor = SACActorNetwork(
            scale_cfg, private_dim=5, fusion_dim=256, n_scales=3,
            action_dim=1, obs_mode="summary_stats",
        )
        flat_obs = torch.randn(4, 50)
        mu, log_sigma = actor.forward(flat_obs)
        assert mu.shape == (4, 1)
        assert log_sigma.shape == (4, 1)

    def test_actor_sample(self):
        """Actor sample produces valid actions in [-1, 1]."""
        scale_cfg = {"summary_input_dim": 50}
        actor = SACActorNetwork(
            scale_cfg, private_dim=5, fusion_dim=256, n_scales=3,
            action_dim=1, obs_mode="summary_stats",
        )
        flat_obs = torch.randn(4, 50)
        action, log_prob = actor.sample(flat_obs, deterministic=True)
        assert action.shape == (4, 1)
        assert (action >= -1.0).all() and (action <= 1.0).all()


class TestCriticSummaryMode:
    def test_critic_forward(self):
        """Critic in summary_stats mode produces scalar Q."""
        scale_cfg = {"summary_input_dim": 50}
        critic = SACCriticNetwork(
            scale_cfg, private_dim=5, fusion_dim=256, action_dim=1,
            n_scales=3, obs_mode="summary_stats",
        )
        flat_obs = torch.randn(4, 50)
        action = torch.randn(4, 1)
        q = critic.forward(flat_obs, action=action)
        assert q.shape == (4, 1)

    def test_critic_encode_and_q_head(self):
        """Critic encode + q_head_forward path works."""
        scale_cfg = {"summary_input_dim": 50}
        critic = SACCriticNetwork(
            scale_cfg, private_dim=5, fusion_dim=256, action_dim=1,
            n_scales=3, obs_mode="summary_stats",
        )
        flat_obs = torch.randn(4, 50)
        features = critic.encode(flat_obs)
        assert features.shape == (4, 256)
        q = critic.q_head_forward(features, torch.randn(4, 1))
        assert q.shape == (4, 1)


class TestHandlerSummaryStats:
    def test_summary_stats_computation(self):
        """_compute_summary_stats produces correct (n_feat * 3,) shape."""
        # Create a minimal handler-like object to test the method
        handler = MultiScaleOHLCVHandler.__new__(MultiScaleOHLCVHandler)
        handler.summary_feature_indices = [0, 1, 2, 6, 7]

        window = np.random.randn(30, 8).astype(np.float32)
        result = handler._compute_summary_stats(window)

        expected_dim = 5 * 3  # 5 features × (mean, std, last)
        assert result.shape == (expected_dim,), f"Expected ({expected_dim},), got {result.shape}"
        assert result.dtype == np.float32

    def test_summary_stats_values(self):
        """Verify mean/std/last are computed correctly."""
        handler = MultiScaleOHLCVHandler.__new__(MultiScaleOHLCVHandler)
        handler.summary_feature_indices = [0, 1]  # Just 2 features

        window = np.array([
            [1.0, 2.0, 0, 0, 0, 0, 0, 0],
            [3.0, 4.0, 0, 0, 0, 0, 0, 0],
            [5.0, 6.0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        result = handler._compute_summary_stats(window)
        # 2 features × 3 stats = 6 dims
        assert result.shape == (6,)
        # means: [3.0, 4.0]
        np.testing.assert_allclose(result[0], 3.0, atol=1e-5)
        np.testing.assert_allclose(result[1], 4.0, atol=1e-5)
        # stds: [std([1,3,5]), std([2,4,6])]
        np.testing.assert_allclose(result[2], np.std([1, 3, 5]), atol=1e-5)
        np.testing.assert_allclose(result[3], np.std([2, 4, 6]), atol=1e-5)
        # last: [5.0, 6.0]
        np.testing.assert_allclose(result[4], 5.0, atol=1e-5)
        np.testing.assert_allclose(result[5], 6.0, atol=1e-5)


class TestWindowModeUnchanged:
    def test_actor_window_mode(self):
        """Window mode actor still works identically."""
        scale_cfg = {
            "input_size": 8, "channels": (32, 64, 64, 64),
            "kernel_size": 3, "output_dim": 64,
        }
        actor = SACActorNetwork(
            scale_cfg, private_dim=5, fusion_dim=256, n_scales=3,
            action_dim=1, obs_mode="window",
        )
        scale_stack = torch.randn(2, 3, 30, 8)
        private = torch.randn(2, 5)
        mu, log_sigma = actor.forward(scale_stack, private)
        assert mu.shape == (2, 1)
        assert log_sigma.shape == (2, 1)
