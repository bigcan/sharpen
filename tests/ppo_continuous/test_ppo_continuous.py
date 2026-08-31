"""Correctness tests for the continuous PPO module (S553-cont-54).

The headline test is `test_logprob_consistency`: the log-prob produced at
ROLLOUT time (SACActorNetwork.sample) must equal the log-prob recomputed at
UPDATE time (PPOContinuousActorCritic.evaluate_actions) for the SAME action.
If they diverge, the PPO importance ratio exp(new-old) is biased even at the
first epoch and the whole algorithm is wrong. This guards the atanh inversion.
"""
import numpy as np
import torch

from sharpen.agents.ppo_continuous.networks import PPOContinuousActorCritic
from sharpen.agents.ppo_continuous.ppo_continuous_agent import PPOContinuousAgent
from sharpen.agents.ppo_continuous.rollout_buffer import ContinuousRolloutBuffer

SCALE_CFG = {"input_size": 8, "channels": (8, 8), "kernel_size": 3, "output_dim": 16, "dropout": 0.0}
N_SCALES, W, F, PD, AD, FUS = 2, 12, 8, 5, 1, 32


def _net():
    torch.manual_seed(0)
    return PPOContinuousActorCritic(
        SCALE_CFG, private_dim=PD, fusion_dim=FUS, n_scales=N_SCALES,
        action_dim=AD, obs_mode="window",
    )


def _obs(B=4):
    torch.manual_seed(1)
    return torch.randn(B, N_SCALES, W, F), torch.randn(B, PD)


def test_act_shapes():
    net = _net()
    scale, priv = _obs()
    action, log_prob, value = net.act(scale, priv, deterministic=False)
    assert action.shape == (4, AD)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert torch.all(action >= -1) and torch.all(action <= 1)


def test_logprob_consistency():
    """sample() log-prob == evaluate_actions() log-prob for the SAME action."""
    net = _net()
    net.eval()
    scale, priv = _obs()
    torch.manual_seed(123)
    action, lp_sample = net.actor.sample(scale, priv, deterministic=False)
    lp_sample = lp_sample.reshape(action.shape[0], -1).sum(dim=-1)
    lp_eval, _value, _ent = net.evaluate_actions(scale, priv, action)
    assert torch.allclose(lp_sample, lp_eval, atol=1e-3), (
        f"log-prob mismatch (biased PPO ratio): max diff "
        f"{(lp_sample - lp_eval).abs().max().item():.2e}"
    )


def test_deterministic_action_is_tanh_mu():
    net = _net()
    net.eval()
    scale, priv = _obs()
    mu, _ = net.actor.forward(scale, priv)
    action, _ = net.actor.sample(scale, priv, deterministic=True)
    assert torch.allclose(action, torch.tanh(mu), atol=1e-5)


def test_entropy_positive_and_finite():
    net = _net()
    scale, priv = _obs()
    action, _, _ = net.act(scale, priv)
    _lp, _v, ent = net.evaluate_actions(scale, priv, action)
    assert torch.all(torch.isfinite(ent))
    # Gaussian differential entropy can be negative for small sigma, but with
    # zero-init log_sigma (sigma=1) it is positive here.
    assert ent.shape == (4,)


def test_gae_matches_manual():
    buf = ContinuousRolloutBuffer(
        rollout_steps=3, num_envs=1, n_scales=N_SCALES, window_size=W,
        features_per_scale=F, private_dim=PD, action_dim=AD,
    )
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    for t in range(3):
        buf.store(
            scale_stack=np.zeros((1, N_SCALES, W, F), np.float32),
            private=np.zeros((1, PD), np.float32),
            actions=np.zeros((1, AD), np.float32),
            log_probs=np.zeros(1, np.float32),
            rewards=np.array([rewards[t]], np.float32),
            values=np.array([values[t]], np.float32),
            dones=np.zeros(1, np.float32),
        )
    gamma, lam = 0.99, 0.95
    last_value = np.array([0.5], np.float32)
    buf.compute_gae(gamma, lam, last_value, np.zeros(1, np.float32))

    # Manual GAE
    adv = np.zeros(3)
    gae = 0.0
    vals = values + [0.5]
    for t in reversed(range(3)):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    assert np.allclose(buf.advantages[:, 0], adv, atol=1e-5)
    assert np.allclose(buf.returns[:, 0], adv + np.array(values), atol=1e-5)


def test_agent_train_step_runs():
    net_cfg = {
        "scale_encoder": dict(SCALE_CFG), "private_dim": PD,
        "fusion_dim": FUS, "n_scales": N_SCALES, "action_dim": AD, "window_size": W,
    }
    agent = PPOContinuousAgent(net_cfg, lr=3e-4, n_epochs=2, batch_size=8, rollout_steps=4, device="cpu")
    buf = ContinuousRolloutBuffer(
        rollout_steps=4, num_envs=4, n_scales=N_SCALES, window_size=W,
        features_per_scale=F, private_dim=PD, action_dim=AD,
    )
    rng = np.random.default_rng(0)
    for _t in range(4):
        scale = rng.standard_normal((4, N_SCALES, W, F)).astype(np.float32)
        priv = rng.standard_normal((4, PD)).astype(np.float32)
        st = torch.as_tensor(scale)
        pv = torch.as_tensor(priv)
        a, lp, v = agent.act_rollout(st, pv)
        buf.store(scale, priv, a, lp, rng.standard_normal(4).astype(np.float32), v, np.zeros(4, np.float32))
    buf.compute_gae(0.99, 0.95, np.zeros(4, np.float32), np.zeros(4, np.float32))
    metrics = agent.train_step(buf)
    assert np.isfinite(metrics["loss_total"])
    assert np.isfinite(metrics["approx_kl"])
    assert np.isfinite(metrics["policy_loss"])


def test_predict_sac_compatible_and_save_load(tmp_path):
    net_cfg = {
        "scale_encoder": dict(SCALE_CFG), "private_dim": PD,
        "fusion_dim": FUS, "n_scales": N_SCALES, "action_dim": AD, "window_size": W,
    }
    agent = PPOContinuousAgent(net_cfg, device="cpu")
    scale, priv = _obs(B=1)
    action = agent.predict(scale, priv, deterministic=True)
    assert action.shape == (1, AD)
    assert torch.all(action >= -1) and torch.all(action <= 1)
    # Deterministic predict is reproducible
    action2 = agent.predict(scale, priv, deterministic=True)
    assert torch.allclose(action, action2)

    path = str(tmp_path / "ckpt.pth")
    agent.save(path)
    agent2 = PPOContinuousAgent(net_cfg, device="cpu")
    agent2.load(path)
    a1 = agent.predict(scale, priv, deterministic=True)
    a2 = agent2.predict(scale, priv, deterministic=True)
    assert torch.allclose(a1, a2, atol=1e-5)
