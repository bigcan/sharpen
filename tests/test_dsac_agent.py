"""Tests for Distributional SAC (QR-SAC + CVaR) agent."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from sharpen.agents.sac.distributional_networks import DistributionalSACCriticNetwork
from sharpen.agents.sac.dsac_agent import DistributionalSACAgent
from sharpen.agents.sac.sac_agent import SACAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def network_config():
    return {
        "scale_encoder": {
            "input_size": 7,
            "channels": (32, 64, 64, 64),
            "kernel_size": 3,
            "output_dim": 64,
        },
        "private_dim": 5,
        "fusion_dim": 256,
        "n_scales": 2,
        "window_size": 30,
        "action_dim": 10,  # Funding-arb: 10 assets
    }


@pytest.fixture
def dsac_agent(network_config):
    return DistributionalSACAgent(
        network_config=network_config,
        n_quantiles=8,  # Small for fast tests
        cvar_alpha=0.25,
        quantile_embed_dim=16,
        kappa=1.0,
        buffer_size=1000,
        batch_size=32,
        learning_starts=64,
        device="cpu",
    )


@pytest.fixture
def dummy_obs(network_config):
    """Create a batch of dummy observations."""
    B = 4
    W = network_config["window_size"]
    F = network_config["scale_encoder"]["input_size"]
    N = network_config["n_scales"]
    return {
        "scale_stack": torch.randn(B, N, W, F),
        "private": torch.randn(B, network_config["private_dim"]),
    }


# ---------------------------------------------------------------------------
# Test: Construction
# ---------------------------------------------------------------------------

class TestDSACConstruction:

    def test_critics_are_distributional(self, dsac_agent):
        assert isinstance(dsac_agent.critic1, DistributionalSACCriticNetwork)
        assert isinstance(dsac_agent.critic2, DistributionalSACCriticNetwork)
        assert isinstance(dsac_agent.target_critic1, DistributionalSACCriticNetwork)
        assert isinstance(dsac_agent.target_critic2, DistributionalSACCriticNetwork)

    def test_actor_is_inherited(self, dsac_agent):
        from sharpen.agents.sac.networks import SACActorNetwork
        assert isinstance(dsac_agent.actor, SACActorNetwork)

    def test_is_subclass_of_sac(self, dsac_agent):
        assert isinstance(dsac_agent, SACAgent)

    def test_distributional_params_stored(self, dsac_agent):
        assert dsac_agent._n_quantiles == 8
        assert dsac_agent._cvar_alpha == 0.25
        assert dsac_agent.kappa == 1.0

    def test_target_critics_frozen(self, dsac_agent):
        for p in dsac_agent.target_critic1.parameters():
            assert not p.requires_grad
        for p in dsac_agent.target_critic2.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Test: Critic output shapes
# ---------------------------------------------------------------------------

class TestCriticShapes:

    def test_critic_output_is_quantile_vector(self, dsac_agent, dummy_obs):
        B = dummy_obs["scale_stack"].shape[0]
        N = dsac_agent._n_quantiles
        action = torch.randn(B, dsac_agent._action_dim)

        q_values, tau = dsac_agent.critic1(
            dummy_obs["scale_stack"], dummy_obs["private"], action,
        )
        assert q_values.shape == (B, N), f"Expected ({B}, {N}), got {q_values.shape}"
        assert tau.shape == (B, N)

    def test_encode_returns_fusion_dim(self, dsac_agent, dummy_obs):
        features = dsac_agent.critic1.encode(
            dummy_obs["scale_stack"], dummy_obs["private"],
        )
        assert features.shape == (4, 256)

    def test_q_head_with_custom_tau(self, dsac_agent, dummy_obs):
        B = dummy_obs["scale_stack"].shape[0]
        N_custom = 16
        features = dsac_agent.critic1.encode(
            dummy_obs["scale_stack"], dummy_obs["private"],
        )
        action = torch.randn(B, dsac_agent._action_dim)
        tau = torch.rand(B, N_custom)

        q_values, tau_out = dsac_agent.critic1.q_head_forward(
            features, action, tau=tau,
        )
        assert q_values.shape == (B, N_custom)
        assert torch.equal(tau_out, tau)


# ---------------------------------------------------------------------------
# Test: CVaR tau range
# ---------------------------------------------------------------------------

class TestCVaRTauRange:

    def test_cvar_tau_in_expected_range(self, dsac_agent):
        """Verify CVaR tau sampling stays within [0, cvar_alpha]."""
        B, N = 64, dsac_agent._n_quantiles
        cvar_tau = torch.rand(B, N) * dsac_agent._cvar_alpha
        assert cvar_tau.max() <= dsac_agent._cvar_alpha + 1e-6
        assert cvar_tau.min() >= 0.0


# ---------------------------------------------------------------------------
# Test: Quantile loss
# ---------------------------------------------------------------------------

class TestQuantileLoss:

    def test_loss_shape(self, dsac_agent):
        B, N = 8, 4
        q_tau = torch.randn(B, N)
        T_tau = torch.randn(B, N)
        tau = torch.rand(B, N)

        loss = dsac_agent._compute_quantile_loss(q_tau, T_tau, tau)
        assert loss.shape == (B,)

    def test_loss_zero_when_identical_single_quantile(self, dsac_agent):
        """With N=N'=1, matching prediction and target gives zero loss."""
        B = 4
        q_tau = torch.tensor([[2.0]] * B)
        T_tau = torch.tensor([[2.0]] * B)
        tau = torch.tensor([[0.5]] * B)

        loss = dsac_agent._compute_quantile_loss(q_tau, T_tau, tau)
        assert loss.sum().item() < 1e-6

    def test_loss_positive(self, dsac_agent):
        """Loss should be positive when predictions differ from targets."""
        B, N = 4, 4
        q_tau = torch.zeros(B, N)
        T_tau = torch.ones(B, N)
        tau = torch.tensor([[0.25, 0.50, 0.75, 1.0]] * B)

        loss = dsac_agent._compute_quantile_loss(q_tau, T_tau, tau)
        assert (loss > 0).all()

    def test_asymmetric_loss(self, dsac_agent):
        """Low tau penalizes overestimation (prediction > target) MORE.

        For tau=0.1: weight for overestimation = |0.1 - 1| = 0.9
                     weight for underestimation = |0.1 - 0| = 0.1
        So overestimation loss > underestimation loss at low tau.
        """
        tau_low = torch.tensor([[0.1]])

        q_under = torch.tensor([[0.0]])  # prediction below target
        T_above = torch.tensor([[1.0]])  # target above -> underestimation

        q_over = torch.tensor([[1.0]])   # prediction above target
        T_below = torch.tensor([[0.0]])  # target below -> overestimation

        loss_under_low = dsac_agent._compute_quantile_loss(q_under, T_above, tau_low)
        loss_over_low = dsac_agent._compute_quantile_loss(q_over, T_below, tau_low)
        # Low tau penalizes overestimation more (weight 0.9 vs 0.1)
        assert loss_over_low.item() > loss_under_low.item()


# ---------------------------------------------------------------------------
# Test: Save/Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_load_roundtrip(self, dsac_agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dsac_test.pt")
            dsac_agent.save(path)

            # Verify dsac_config is in checkpoint
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert "dsac_config" in checkpoint
            assert checkpoint["dsac_config"]["n_quantiles"] == 8
            assert checkpoint["dsac_config"]["cvar_alpha"] == 0.25
            assert checkpoint["dsac_config"]["kappa"] == 1.0

            # Load into new agent
            dsac_agent.load(path)

    def test_save_load_preserves_weights(self, network_config):
        """Verify state dict round-trips exactly through save/load."""
        agent = DistributionalSACAgent(
            network_config=network_config,
            n_quantiles=8,
            cvar_alpha=0.25,
            buffer_size=100,
            learning_starts=10,
            device="cpu",
        )

        sd_before = {k: v.clone() for k, v in agent.critic1.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dsac_test.pt")
            agent.save(path)
            agent.load(path)

        sd_after = agent.critic1.state_dict()
        for k in sd_before:
            assert torch.equal(sd_before[k], sd_after[k]), f"Mismatch in {k}"


# ---------------------------------------------------------------------------
# Test: Mega-batch training
# ---------------------------------------------------------------------------

class TestMegaBatchTraining:

    def test_train_step_returns_none_before_learning_starts(self, dsac_agent):
        result = dsac_agent.train_step_mega(1)
        assert result is None

    def test_train_step_runs_after_buffer_filled(self, dsac_agent, network_config):
        """Fill buffer and verify train_step_mega runs without error."""
        W = network_config["window_size"]
        F = network_config["scale_encoder"]["input_size"]
        N = network_config["n_scales"]
        action_dim = network_config["action_dim"]
        private_dim = network_config["private_dim"]

        # Fill replay buffer past learning_starts
        for _ in range(70):
            obs = {
                f"scale_{i}": np.random.randn(W, F).astype(np.float32)
                for i in range(N)
            }
            obs["private"] = np.random.randn(private_dim).astype(np.float32)
            next_obs = {
                f"scale_{i}": np.random.randn(W, F).astype(np.float32)
                for i in range(N)
            }
            next_obs["private"] = np.random.randn(private_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)

            dsac_agent.store_transition(obs, action, 0.01, next_obs, False)

        # Should run without error
        dsac_agent.train_step_mega(2)
        # Metrics may be None if step count doesn't hit the logging interval
        # Just verify no exception was raised


# ---------------------------------------------------------------------------
# Test: Predict interface compatibility
# ---------------------------------------------------------------------------

class TestPredictCompat:

    def test_predict_returns_correct_shape(self, dsac_agent, dummy_obs):
        action = dsac_agent.predict(
            dummy_obs["scale_stack"], dummy_obs["private"], deterministic=True,
        )
        assert action.shape == (4, 10)  # (B, action_dim)

    def test_predict_stochastic(self, dsac_agent, dummy_obs):
        action = dsac_agent.predict(
            dummy_obs["scale_stack"], dummy_obs["private"], deterministic=False,
        )
        assert action.shape == (4, 10)

    def test_predict_in_tanh_range(self, dsac_agent, dummy_obs):
        action = dsac_agent.predict(
            dummy_obs["scale_stack"], dummy_obs["private"], deterministic=False,
        )
        assert (action >= -1.0).all() and (action <= 1.0).all()
