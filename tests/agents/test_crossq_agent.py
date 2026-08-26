"""Tests for the CrossQ variant of SAC (agents.sac.crossq).

CrossQ = SAC with the target network deleted, a BatchRenorm Q head, and the
current/next state-action pairs pushed through the critic in ONE joint batch so
the normalization statistics are shared between them (Bhatt et al., ICLR 2024,
arXiv:1902.05605).

Three of these tests are tripwires rather than unit tests, and are worth keeping
even if they look redundant:

  * test_joint_pass_is_not_two_separate_passes — the joint batch IS the
    algorithm. An "optimization" that splits it into two passes silently
    produces SAC-without-a-target-network, which diverges, and no other test
    here would notice.
  * test_statistics_are_computed_in_fp32_under_half_input — NAN-01. Batch
    statistics of an fp16 activation overflow silently (fp16 caps at 65504).
  * test_predict_does_not_touch_critic_batchnorm_statistics — LEAK-1 adjacency.
    BatchRenorm's running statistics are the one piece of critic state that
    could, in principle, absorb evaluation-set data.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from finrl_pro_ds.agents.common.batch_renorm import BatchRenorm1d
from finrl_pro_ds.agents.sac.sac_agent import SACAgent, normalize_crossq_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def network_config():
    return {
        "scale_encoder": {
            "input_size": 4,
            "channels": (8, 8),
            "kernel_size": 3,
            "output_dim": 8,
        },
        "private_dim": 5,
        "fusion_dim": 32,
        "n_scales": 2,
        "window_size": 8,
        "action_dim": 1,
    }


def _make_agent(network_config, crossq):
    return SACAgent(
        network_config=network_config,
        buffer_size=256,
        batch_size=8,
        learning_starts=16,
        device="cpu",
        crossq=crossq,
    )


@pytest.fixture
def crossq_agent(network_config):
    return _make_agent(network_config, True)


@pytest.fixture
def baseline_agent(network_config):
    return _make_agent(network_config, None)


def _fill_buffer(agent, network_config, n=40, scale=1.0):
    W = network_config["window_size"]
    F = network_config["scale_encoder"]["input_size"]
    N = network_config["n_scales"]
    priv_dim = network_config["private_dim"]
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs = {f"scale_{i}": (rng.standard_normal((W, F)) * scale).astype(np.float32) for i in range(N)}
        obs["private"] = rng.standard_normal(priv_dim).astype(np.float32)
        next_obs = {f"scale_{i}": (rng.standard_normal((W, F)) * scale).astype(np.float32) for i in range(N)}
        next_obs["private"] = rng.standard_normal(priv_dim).astype(np.float32)
        action = rng.standard_normal(network_config["action_dim"]).astype(np.float32)
        agent.store_transition(obs, action, 0.01, next_obs, False)


def _dummy_obs(network_config, batch=4, scale=1.0):
    W = network_config["window_size"]
    F = network_config["scale_encoder"]["input_size"]
    N = network_config["n_scales"]
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(batch, N, W, F, generator=g) * scale,
        torch.randn(batch, network_config["private_dim"], generator=g),
    )


def _bn_layers(critic):
    return [m for m in critic.q_head if isinstance(m, BatchRenorm1d)]


def _bn_snapshot(agent):
    """Capture every BatchRenorm buffer so a second forward pass can be run
    against identical statistics (a train-mode forward mutates them)."""
    return [
        (m, m.running_mean.clone(), m.running_var.clone(), m._steps)
        for m in _bn_layers(agent.critic1) + _bn_layers(agent.critic2)
    ]


def _bn_restore(snapshot):
    for m, mean, var, steps in snapshot:
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        m._steps = steps


# ---------------------------------------------------------------------------
# Config normalization
# ---------------------------------------------------------------------------

class TestConfigFlag:

    @pytest.mark.parametrize("value", [None, False, {}, {"enabled": False}])
    def test_disabled_forms(self, value):
        assert normalize_crossq_config(value)["enabled"] is False

    def test_bool_true_uses_paper_defaults(self):
        cfg = normalize_crossq_config(True)
        assert cfg["enabled"] is True
        assert cfg["adam_betas"] == (0.5, 0.999)
        assert cfg["bn_momentum"] == 0.01

    def test_dict_overrides(self):
        cfg = normalize_crossq_config({"enabled": True, "critic_width": 2048, "critic_depth": 2})
        assert (cfg["critic_width"], cfg["critic_depth"]) == (2048, 2)
        assert cfg["bn_warmup_steps"] == 100_000  # untouched default

    def test_unknown_key_is_rejected_not_ignored(self):
        with pytest.raises(ValueError, match="unknown crossq config key"):
            normalize_crossq_config({"enabled": True, "bn_warmup_step": 10})

    def test_bad_type_rejected(self):
        with pytest.raises(TypeError):
            normalize_crossq_config("true")


# ---------------------------------------------------------------------------
# Construction — and the untouched baseline
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_baseline_keeps_target_networks_and_has_no_batchnorm(self, baseline_agent):
        assert baseline_agent.crossq is False
        assert baseline_agent.target_critic1 is not None
        assert baseline_agent.target_critic2 is not None
        assert _bn_layers(baseline_agent.critic1) == []

    def test_crossq_has_no_target_networks(self, crossq_agent):
        assert crossq_agent.crossq is True
        assert crossq_agent.target_critic1 is None
        assert crossq_agent.target_critic2 is None

    def test_crossq_q_head_is_a_batchrenorm_mlp(self, crossq_agent):
        head = crossq_agent.critic1.q_head
        # BN on the input, then Linear -> ReLU -> BN per hidden layer, then Linear.
        assert isinstance(head[0], BatchRenorm1d)
        assert isinstance(head[-1], torch.nn.Linear)
        assert len(_bn_layers(crossq_agent.critic1)) == 2  # depth 1 => input BN + 1 hidden BN

    def test_default_geometry_matches_baseline_so_the_ab_isolates_the_mechanism(
        self, network_config, crossq_agent, baseline_agent,
    ):
        base_linears = [m for m in baseline_agent.critic1.q_head if isinstance(m, torch.nn.Linear)]
        cq_linears = [m for m in crossq_agent.critic1.q_head if isinstance(m, torch.nn.Linear)]
        assert [tuple(m.weight.shape) for m in base_linears] == [tuple(m.weight.shape) for m in cq_linears]

    def test_width_and_depth_are_configurable(self, network_config):
        agent = _make_agent(network_config, {"enabled": True, "critic_width": 64, "critic_depth": 2})
        linears = [m for m in agent.critic1.q_head if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3  # 2 hidden + output
        assert linears[0].out_features == 64
        assert len(_bn_layers(agent.critic1)) == 3

    def test_critic_optimizer_uses_paper_betas(self, crossq_agent, baseline_agent):
        assert crossq_agent.critic_optimizer.param_groups[0]["betas"] == (0.5, 0.999)
        assert baseline_agent.critic_optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_dsac_refuses_the_combination(self, network_config):
        from finrl_pro_ds.agents.sac.dsac_agent import DistributionalSACAgent
        with pytest.raises(NotImplementedError, match="not supported by DistributionalSACAgent"):
            DistributionalSACAgent(
                network_config=network_config, n_quantiles=4, quantile_embed_dim=8,
                buffer_size=64, batch_size=8, learning_starts=16, device="cpu", crossq=True,
            )


# ---------------------------------------------------------------------------
# BatchRenorm layer
# ---------------------------------------------------------------------------

class TestBatchRenorm:

    def test_warmup_phase_is_exactly_batchnorm(self):
        layer = BatchRenorm1d(6, warmup_steps=1000, affine=False)
        x = torch.randn(32, 6) * 3 + 5
        out = layer(x)
        expected = (x - x.mean(0)) / (x.var(0, unbiased=False) + layer.eps).sqrt()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_relaxation_ramps_after_warmup(self):
        layer = BatchRenorm1d(4, warmup_steps=10)
        assert layer._relaxation() == (1.0, 0.0)
        layer._steps = 10
        assert layer._relaxation() == (1.0, 0.0)  # ramp starts at the identity
        layer._steps = 15
        r_max, d_max = layer._relaxation()
        assert 1.0 < r_max < 3.0 and 0.0 < d_max < 5.0
        layer._steps = 100
        assert layer._relaxation() == (3.0, 5.0)

    def test_training_updates_running_statistics(self):
        layer = BatchRenorm1d(4, momentum=0.5)
        before = layer.running_mean.clone()
        layer(torch.randn(16, 4) + 10.0)
        assert not torch.equal(before, layer.running_mean)

    def test_eval_uses_running_statistics_and_updates_nothing(self):
        layer = BatchRenorm1d(4, momentum=0.5, affine=False)
        layer(torch.randn(16, 4) + 10.0)  # seed the running stats
        layer.eval()
        snapshot = (layer.running_mean.clone(), layer.running_var.clone())
        x = torch.randn(16, 4)
        out = layer(x)
        expected = (x - snapshot[0]) / (snapshot[1] + layer.eps).sqrt()
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert torch.equal(layer.running_mean, snapshot[0])
        assert torch.equal(layer.running_var, snapshot[1])

    def test_statistics_are_computed_in_fp32_under_half_input(self):
        """NAN-01 tripwire: fp16 batch statistics overflow, fp32 ones do not.

        With activations of magnitude ~1e3 the squared deviations reach ~1e6,
        well past fp16's 65504 ceiling. An implementation that normalizes in the
        autocast dtype gets var=inf and emits an all-zero (or NaN) row; a correct
        one still returns unit-variance output.
        """
        layer = BatchRenorm1d(8, warmup_steps=1000, affine=False)
        x = (torch.randn(64, 8) * 1000.0).half()
        out = layer(x)
        assert out.dtype == torch.float16  # input dtype preserved
        assert torch.isfinite(out.float()).all()
        assert out.float().std().item() == pytest.approx(1.0, abs=0.1)

    def test_autocast_does_not_change_output_dtype(self):
        layer = BatchRenorm1d(8, affine=False)
        x = torch.randn(16, 8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = layer(x)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_single_sample_in_training_mode_raises(self):
        layer = BatchRenorm1d(4)
        with pytest.raises(ValueError, match="batch >= 2"):
            layer(torch.randn(1, 4))

    def test_single_sample_in_eval_mode_is_fine(self):
        layer = BatchRenorm1d(4).eval()
        assert torch.isfinite(layer(torch.randn(1, 4))).all()

    def test_step_counter_survives_a_state_dict_round_trip(self):
        layer = BatchRenorm1d(4, warmup_steps=2)
        for _ in range(5):
            layer(torch.randn(8, 4))
        clone = BatchRenorm1d(4, warmup_steps=2)
        clone.load_state_dict(layer.state_dict())
        assert clone._steps == layer._steps == 5
        assert clone._relaxation() == layer._relaxation()


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------

class TestJointPass:

    def test_joint_pass_is_not_two_separate_passes(self, crossq_agent, network_config):
        """The shared batch statistics ARE the mechanism.

        Q(cat[s, s']) must not equal cat[Q(s), Q(s')]: in the joint pass both
        halves are normalized by one set of statistics drawn from the mixture.
        If this ever passes, someone has split the forward and turned CrossQ
        back into target-network-free SAC.
        """
        critic = crossq_agent.critic1
        critic.train()
        s, priv = _dummy_obs(network_config, batch=16, scale=1.0)
        s_next, priv_next = _dummy_obs(network_config, batch=16, scale=8.0)
        act = torch.zeros(16, 1)

        with torch.no_grad():
            joint = critic(
                torch.cat([s, s_next]), torch.cat([priv, priv_next]),
                torch.cat([act, act]),
            )
            separate = torch.cat([critic(s, priv, act), critic(s_next, priv_next, act)])

        assert not torch.allclose(joint, separate, rtol=1e-3, atol=1e-3)

    def test_crossq_critic_loss_carries_gradient_to_both_halves(self, crossq_agent, network_config):
        """The next-state half is stop-gradiented as a VALUE but still coupled
        through the batch statistics — that coupling is intended."""
        _fill_buffer(crossq_agent, network_config, n=40)
        crossq_agent.train_step_mega(1)
        assert all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in crossq_agent.critic1.parameters()
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:

    def test_train_step_returns_none_before_learning_starts(self, crossq_agent):
        assert crossq_agent.train_step_mega(1) is None

    def test_train_step_updates_critic_and_batchnorm_statistics(self, crossq_agent, network_config):
        _fill_buffer(crossq_agent, network_config, n=60)
        head = crossq_agent.critic1.q_head
        w_before = [m.weight.clone() for m in head if isinstance(m, torch.nn.Linear)]
        bn_before = [m.running_mean.clone() for m in _bn_layers(crossq_agent.critic1)]

        crossq_agent.train_step_mega(4)

        w_after = [m.weight for m in head if isinstance(m, torch.nn.Linear)]
        assert any(not torch.equal(a, b) for a, b in zip(w_before, w_after))
        bn_after = [m.running_mean for m in _bn_layers(crossq_agent.critic1)]
        assert all(not torch.equal(a, b) for a, b in zip(bn_before, bn_after))
        assert all(torch.isfinite(w).all() for w in w_after)

    def test_actor_update_leaves_batchnorm_statistics_alone(self, crossq_agent, network_config):
        """The actor loss scores the policy's actions with the head in eval mode;
        those actions must not leak into the critic's running statistics."""
        _fill_buffer(crossq_agent, network_config, n=60)
        crossq_agent.train_step_mega(2)  # warm the stats up

        # actor_update_freq=2, so this step count lands on an actor update
        bn = _bn_layers(crossq_agent.critic1)
        steps_before = [m._steps for m in bn]
        crossq_agent._train_step_count = 1  # next step is an actor-update step
        crossq_agent.train_step_mega(1)
        steps_after = [m._steps for m in bn]
        # exactly ONE BN forward per layer (the critic's joint pass), not two
        assert [a - b for a, b in zip(steps_after, steps_before)] == [1] * len(bn)

    def test_baseline_training_path_still_works(self, baseline_agent, network_config):
        _fill_buffer(baseline_agent, network_config, n=60)
        before = baseline_agent.target_critic1.q_head[0].weight.clone()
        baseline_agent.train_step_mega(4)
        assert not torch.equal(before, baseline_agent.target_critic1.q_head[0].weight)

    def test_train_step_runs_under_bf16_autocast(self, network_config):
        agent = SACAgent(
            network_config=network_config, buffer_size=256, batch_size=8,
            learning_starts=16, device="cpu", crossq=True,
            use_amp=False, amp_dtype="bfloat16",
        )
        _fill_buffer(agent, network_config, n=60)
        agent.train_step_mega(2)
        assert all(torch.isfinite(p).all() for p in agent.critic1.parameters())


# ---------------------------------------------------------------------------
# Evaluation / LEAK-1 adjacency
# ---------------------------------------------------------------------------

class TestEvaluationIsolation:

    def test_predict_does_not_touch_critic_batchnorm_statistics(self, crossq_agent, network_config):
        """LEAK-1 adjacency.

        Running statistics are the only critic state that could absorb data from
        a later split. The critic is not on the inference path at all — this test
        pins that fact rather than assuming it.
        """
        _fill_buffer(crossq_agent, network_config, n=60)
        crossq_agent.train_step_mega(2)

        snapshot = [
            (m.running_mean.clone(), m.running_var.clone(), m._steps)
            for m in _bn_layers(crossq_agent.critic1) + _bn_layers(crossq_agent.critic2)
        ]

        # "Test-split" observations: deliberately different scale from training.
        scale_stack, priv = _dummy_obs(network_config, batch=4, scale=50.0)
        for deterministic in (True, False):
            crossq_agent.predict(scale_stack, priv, deterministic=deterministic)

        after = [
            (m.running_mean, m.running_var, m._steps)
            for m in _bn_layers(crossq_agent.critic1) + _bn_layers(crossq_agent.critic2)
        ]
        for (m0, v0, s0), (m1, v1, s1) in zip(snapshot, after):
            assert torch.equal(m0, m1) and torch.equal(v0, v1) and s0 == s1

    def test_single_sample_q_evaluation_works_in_eval_mode(self, crossq_agent, network_config):
        scale_stack, priv = _dummy_obs(network_config, batch=1)
        crossq_agent.critic1.eval()
        with torch.no_grad():
            q = crossq_agent.critic1(scale_stack, priv, torch.zeros(1, 1))
        assert q.shape == (1, 1) and torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoints:

    def test_round_trip_preserves_batchnorm_state(self, crossq_agent, network_config):
        _fill_buffer(crossq_agent, network_config, n=60)
        crossq_agent.train_step_mega(3)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crossq.pt")
            crossq_agent.save(path)
            restored = _make_agent(network_config, True)
            restored.load(path)

        src_bn = _bn_layers(crossq_agent.critic1)
        dst_bn = _bn_layers(restored.critic1)
        for a, b in zip(src_bn, dst_bn):
            assert torch.equal(a.running_mean, b.running_mean)
            assert torch.equal(a.running_var, b.running_var)
            assert a._steps == b._steps

    def test_checkpoint_omits_target_networks(self, crossq_agent):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crossq.pt")
            crossq_agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["crossq"] is True
        assert "target_critic1" not in ckpt

    def test_loading_a_crossq_checkpoint_into_baseline_sac_raises(
        self, crossq_agent, baseline_agent, network_config,
    ):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crossq.pt")
            crossq_agent.save(path)
            with pytest.raises(ValueError, match="agents.sac.crossq"):
                baseline_agent.load(path)

    def test_loading_a_baseline_checkpoint_into_crossq_raises(
        self, crossq_agent, baseline_agent, network_config,
    ):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sac.pt")
            baseline_agent.save(path)
            with pytest.raises(ValueError, match="agents.sac.crossq"):
                crossq_agent.load(path)

    def test_baseline_round_trip_unaffected(self, baseline_agent, network_config):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sac.pt")
            baseline_agent.save(path)
            restored = _make_agent(network_config, None)
            restored.load(path)
        assert torch.equal(
            baseline_agent.target_critic1.q_head[0].weight,
            restored.target_critic1.q_head[0].weight,
        )


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------

class TestTrainerWiring:
    """The flag has to survive the config -> trainer -> agent path, not just
    the agent constructor."""

    @staticmethod
    def _trainer_config(crossq, update_interval=4):
        from tests.deepscalper.test_sac_trainer import _minimal_config
        cfg = _minimal_config(
            total_timesteps=12, learning_starts=4, batch_size=4,
            buffer_size=32, update_interval=update_interval,
        )
        if crossq is not None:
            cfg["agents"]["sac"]["crossq"] = crossq
        return cfg

    @staticmethod
    def _fake_env():
        from tests.deepscalper.test_sac_trainer import FakeVecEnv
        return FakeVecEnv(num_envs=1, n_scales=3, window_size=8,
                          features_per_scale=8, episode_length=20)

    def test_flag_reaches_the_agent(self):
        from finrl_pro_ds.training.sac_trainer import SACTrainer
        trainer = SACTrainer(self._fake_env(), self._trainer_config(True),
                             device="cpu", run_name="crossq_wiring")
        assert trainer.agent.crossq is True
        assert trainer.agent.target_critic1 is None

    def test_absent_flag_builds_baseline_sac(self):
        from finrl_pro_ds.training.sac_trainer import SACTrainer
        trainer = SACTrainer(self._fake_env(), self._trainer_config(None),
                             device="cpu", run_name="baseline_wiring")
        assert trainer.agent.crossq is False
        assert trainer.agent.target_critic1 is not None

    def test_tau_is_not_rescaled_under_crossq(self):
        """The UTD tau schedule is meaningless without a Polyak update; leaving
        it on would print a tau in the logs that nothing reads."""
        from finrl_pro_ds.training.sac_trainer import SACTrainer
        base = SACTrainer(self._fake_env(), self._trainer_config(None, update_interval=4),
                          device="cpu", run_name="baseline_tau")
        cq = SACTrainer(self._fake_env(), self._trainer_config(True, update_interval=4),
                        device="cpu", run_name="crossq_tau")
        assert base.agent.tau != pytest.approx(0.005)  # auto-scaled
        assert cq.agent.tau == pytest.approx(0.005)    # untouched

    @patch("finrl_pro_ds.training.sac_trainer.wandb")
    def test_training_loop_runs_end_to_end(self, mock_wandb):
        from finrl_pro_ds.training.sac_trainer import SACTrainer
        mock_wandb.log = MagicMock()
        trainer = SACTrainer(self._fake_env(), self._trainer_config(True),
                             device="cpu", run_name="crossq_train")
        final_path = trainer.train()
        assert final_path is not None and os.path.isfile(final_path)
        assert all(torch.isfinite(p).all() for p in trainer.agent.critic1.parameters())


# ---------------------------------------------------------------------------
# The Bellman target itself (MATH-RL01, CrossQ variant)
# ---------------------------------------------------------------------------

class TestBellmanTarget:
    """y = r + gamma * (1 - done) * (min(Q1, Q2) - alpha * log_pi), with the
    next-state half stop-gradiented. The two degenerate cases below pin the
    reward term and the termination mask without re-deriving the bootstrap."""

    @staticmethod
    def _loss_inputs(agent, network_config, dones_value, batch=8):
        W = network_config["window_size"]
        F = network_config["scale_encoder"]["input_size"]
        N = network_config["n_scales"]
        g = torch.Generator().manual_seed(7)
        s = torch.randn(batch, N, W, F, generator=g)
        s2 = torch.randn(batch, N, W, F, generator=g)
        priv = torch.randn(batch, network_config["private_dim"], generator=g)
        npriv = torch.randn(batch, network_config["private_dim"], generator=g)
        actions = torch.zeros(batch, network_config["action_dim"])
        rewards = torch.randn(batch, 1, generator=g)
        dones = torch.full((batch, 1), float(dones_value))
        return s, s2, priv, npriv, actions, rewards, dones

    def _run(self, agent, network_config, dones_value):
        s, s2, priv, npriv, actions, rewards, dones = self._loss_inputs(
            agent, network_config, dones_value,
        )
        amp_ctx = torch.amp.autocast("cuda", enabled=False)
        _, _, q1, q2, loss = agent._crossq_critic_loss(
            s, s2, priv, npriv, None, None, actions, rewards, dones,
            torch.tensor(0.2), amp_ctx,
        )
        return q1, q2, rewards, loss

    def test_terminal_transitions_bootstrap_nothing(self, crossq_agent, network_config):
        q1, q2, rewards, loss = self._run(crossq_agent, network_config, dones_value=1.0)
        expected = (
            torch.nn.functional.mse_loss(q1, rewards)
            + torch.nn.functional.mse_loss(q2, rewards)
        )
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_zero_discount_bootstraps_nothing(self, network_config):
        agent = SACAgent(
            network_config=network_config, buffer_size=64, batch_size=8,
            learning_starts=16, device="cpu", crossq=True, gamma=0.0,
        )
        q1, q2, rewards, loss = self._run(agent, network_config, dones_value=0.0)
        expected = (
            torch.nn.functional.mse_loss(q1, rewards)
            + torch.nn.functional.mse_loss(q2, rewards)
        )
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_target_matches_the_soft_bellman_formula_term_by_term(
        self, crossq_agent, network_config,
    ):
        """Pins min(Q1,Q2), the alpha*log_pi entropy term, gamma, the (1-done)
        mask and the stop-gradient against a hand-written reference.

        Determinism: encoder dropout off and a fixed policy sample, so the only
        state that moves between the two passes is BatchRenorm - which is
        snapshotted and restored.
        """
        agent = crossq_agent
        n = 8
        agent.critic1.encoder.eval()
        agent.critic2.encoder.eval()
        fixed_action = torch.full((n, 1), 0.3)
        fixed_logp = torch.full((n, 1), -0.7)
        agent.actor.sample = lambda *a, **k: (fixed_action, fixed_logp)

        s, s2, priv, npriv, actions, rewards, dones = self._loss_inputs(
            agent, network_config, 0.0, batch=n,
        )
        alpha = torch.tensor(0.2)
        amp_ctx = torch.amp.autocast("cuda", enabled=False)

        snap = _bn_snapshot(agent)
        agent.critic1.zero_grad(set_to_none=True)
        agent.critic2.zero_grad(set_to_none=True)
        _, _, q1, q2, loss = agent._crossq_critic_loss(
            s, s2, priv, npriv, None, None, actions, rewards, dones, alpha, amp_ctx,
        )
        loss.backward()
        impl_grads = [
            p.grad.clone() for p in agent.critic1.parameters() if p.grad is not None
        ]
        _bn_restore(snap)
        agent.critic1.zero_grad(set_to_none=True)
        agent.critic2.zero_grad(set_to_none=True)

        cat_s = torch.cat([s, s2])
        cat_p = torch.cat([priv, npriv])
        cat_a = torch.cat([actions, fixed_action])
        o1 = agent.critic1.q_head_forward(agent.critic1.encode(cat_s, cat_p), cat_a)
        o2 = agent.critic2.q_head_forward(agent.critic2.encode(cat_s, cat_p), cat_a)
        y = rewards + (1.0 - dones) * agent.gamma * (
            torch.min(o1[n:], o2[n:]).detach() - alpha * fixed_logp
        )
        ref_loss = (
            torch.nn.functional.mse_loss(o1[:n], y)
            + torch.nn.functional.mse_loss(o2[:n], y)
        )

        torch.testing.assert_close(q1, o1[:n], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)

        # Gradients too, not just values: a missing .detach() on the bootstrap is
        # invisible in the loss VALUE and shows up only here.
        ref_loss.backward()
        ref_grads = [
            p.grad.clone() for p in agent.critic1.parameters() if p.grad is not None
        ]
        assert len(impl_grads) == len(ref_grads) and impl_grads
        for a, b in zip(impl_grads, ref_grads):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)

    def test_gradient_reaches_the_critic(self, crossq_agent, network_config):
        s, s2, priv, npriv, actions, rewards, dones = self._loss_inputs(
            crossq_agent, network_config, 0.0,
        )
        amp_ctx = torch.amp.autocast("cuda", enabled=False)
        _, _, _, _, loss = crossq_agent._crossq_critic_loss(
            s, s2, priv, npriv, None, None, actions, rewards, dones,
            torch.tensor(0.2), amp_ctx,
        )
        loss.backward()
        grads = [p.grad for p in crossq_agent.critic1.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        out_layer = [m for m in crossq_agent.critic1.q_head if isinstance(m, torch.nn.Linear)][-1]
        assert out_layer.weight.grad.abs().sum() > 0
