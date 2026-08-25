"""NAN-01 — detection (SACTrainer) and containment (SACAgent._guard_alpha_grad).

Origin: randd_log S553-cont-164/165. One poisoned observation row made the actor's forward
pass emit NaN. The twin critics and the actor were protected by ``GradScaler`` (which skips a
step when ``unscale_`` finds inf/NaN), but the alpha update runs deliberately OUTSIDE AMP and
took the NaN gradient — latching ``log_alpha`` to NaN permanently. Because ``alpha`` multiplies
the entropy term of the target for the WHOLE batch, one bad row then poisoned every subsequent
target for every row. Meanwhile nothing inspected the metrics, so run ``pqwttrqd`` reached
75,000/75,000, exited ``state=finished``, and wrote a full gate verdict JSON built on it.

Two separable mechanisms, tested separately because they must stay separable — the guard must
CONTAIN without suppressing the signal the assertion DETECTS:
  * ``SACTrainer._assert_finite_metrics`` — halts the run (detection).
  * ``SACAgent._guard_alpha_grad``        — zeroes a non-finite log_alpha grad (containment).

All negative tests: each fails if its half is reverted.
"""
from __future__ import annotations

import math

import pytest
import torch

from finrl_pro_ds.agents.sac.dsac_agent import DistributionalSACAgent
from finrl_pro_ds.training.sac_trainer import SACTrainer


# --------------------------------------------------------------------------- #
# Detection — SACTrainer._assert_finite_metrics
# --------------------------------------------------------------------------- #
_HEALTHY = {"critic_loss": 18.7, "actor_loss": 76.2, "alpha": 0.285,
            "alpha_loss": -0.01, "entropy": 1.2, "q1_mean": -31.7}


def test_finite_metrics_pass_through():
    """The healthy case must not raise — the guard is not allowed to be trigger-happy."""
    SACTrainer._assert_finite_metrics(_HEALTHY, total_steps=13000)


@pytest.mark.parametrize("key", ["actor_loss", "critic_loss", "alpha", "q1_mean"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_halts(key, bad):
    """A NaN/inf metric must HALT, not log. This is the pqwttrqd case."""
    m = dict(_HEALTHY, **{key: bad})
    with pytest.raises(RuntimeError, match="NAN-01"):
        SACTrainer._assert_finite_metrics(m, total_steps=13000)


def test_halt_message_names_the_metric_and_step():
    """The operator must be able to act on the message without re-deriving anything."""
    m = dict(_HEALTHY, actor_loss=float("nan"))
    with pytest.raises(RuntimeError) as ei:
        SACTrainer._assert_finite_metrics(m, total_steps=13000)
    msg = str(ei.value)
    assert "actor_loss" in msg and "13000" in msg


def test_non_numeric_metrics_are_ignored():
    """Only numeric metrics are checkable; a string/None must not crash the checker."""
    SACTrainer._assert_finite_metrics(dict(_HEALTHY, phase="warmup", note=None), total_steps=1)


# --------------------------------------------------------------------------- #
# Containment — SACAgent._guard_alpha_grad
# --------------------------------------------------------------------------- #
@pytest.fixture
def agent():
    return DistributionalSACAgent(
        network_config={"summary_input_dim": 13, "private_dim": 0, "fusion_dim": 64,
                        "n_scales": 1, "window_size": 1, "action_dim": 1,
                        "obs_mode": "summary_stats",
                        "scale_encoder": {"summary_input_dim": 13}},
        n_quantiles=8, cvar_alpha=0.25, quantile_embed_dim=16,
        buffer_size=256, batch_size=8, learning_starts=16, device="cpu",
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_guard_zeroes_non_finite_alpha_grad(agent, bad):
    agent.log_alpha.grad = torch.full_like(agent.log_alpha, bad)
    agent._guard_alpha_grad()
    assert torch.isfinite(agent.log_alpha.grad).all()
    assert float(agent.log_alpha.grad) == 0.0, "a poisoned grad must become a no-op step"


def test_guard_preserves_a_healthy_grad(agent):
    """Containment must not distort ordinary learning."""
    agent.log_alpha.grad = torch.full_like(agent.log_alpha, 0.37)
    agent._guard_alpha_grad()
    assert float(agent.log_alpha.grad) == pytest.approx(0.37)


def test_guard_tolerates_absent_grad(agent):
    agent.log_alpha.grad = None
    agent._guard_alpha_grad()          # must not raise before the first backward()


def test_log_alpha_survives_a_nan_gradient_step(agent):
    """THE REGRESSION: a NaN gradient must not latch log_alpha, because alpha then
    poisons the entropy term of EVERY row's target, permanently."""
    before = float(agent.log_alpha.detach())
    agent.log_alpha.grad = torch.full_like(agent.log_alpha, float("nan"))
    agent._guard_alpha_grad()
    agent.alpha_optimizer.step()
    after = float(agent.log_alpha.detach())
    assert math.isfinite(after), "log_alpha latched to NaN — one bad row poisons the whole run"
    assert after == pytest.approx(before, abs=1e-6), "a poisoned step must be a no-op"
    assert math.isfinite(float(agent.log_alpha.exp())), "alpha = exp(log_alpha) must stay finite"


def test_guard_is_wired_into_both_agents():
    """Declared-but-not-wired is the failure mode this project keeps hitting: assert the
    call actually appears in both update paths, not merely that the helper exists."""
    import inspect

    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    for cls in (SACAgent, DistributionalSACAgent):
        src = inspect.getsource(cls.train_step_mega)
        assert "_guard_alpha_grad()" in src, f"{cls.__name__}.train_step_mega does not guard alpha"
        assert src.index("_guard_alpha_grad()") < src.index("self.alpha_optimizer.step()"), \
            f"{cls.__name__} guards alpha AFTER the optimizer step — too late"


# --------------------------------------------------------------------------- #
# Containment (AMP-off path) — SACAgent._step_if_finite
#
# GradScaler skips an optimizer step when unscale_ finds inf/NaN, but the scaler is
# DISABLED for use_amp:false AND for amp_dtype:bfloat16 — and that branch fell through to a
# bare optimizer.step(). clip_grad_norm_ is not a guard: a non-finite gradient makes
# total_norm non-finite, hence clip_coef non-finite, hence EVERY gradient NaN and the step
# writes NaN into every parameter at once. Worse than the log_alpha case, which corrupts
# one scalar.
# --------------------------------------------------------------------------- #
def _params_finite(mod) -> bool:
    return all(torch.isfinite(p).all() for p in mod.parameters())


def test_clip_grad_norm_does_not_protect_against_nan():
    """Pins the PREMISE. If torch ever made clip_grad_norm_ safe on its own, this fails and
    the guard below can be reconsidered — rather than being cargo-culted forever."""
    lin = torch.nn.Linear(4, 4)
    lin.weight.grad = torch.full_like(lin.weight, float("nan"))
    lin.bias.grad = torch.ones_like(lin.bias)
    total = torch.nn.utils.clip_grad_norm_(lin.parameters(), 10.0)
    assert not torch.isfinite(total), "expected a non-finite total_norm"
    assert not torch.isfinite(lin.bias.grad).all(), \
        "clip_grad_norm_ propagated the non-finite norm into an initially-HEALTHY grad"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_step_if_finite_skips_and_counts(agent, bad):
    before = [p.detach().clone() for p in agent.actor.parameters()]
    for p in agent.actor.parameters():
        p.grad = torch.zeros_like(p)
    skips0 = agent._nonfinite_grad_skips
    agent._step_if_finite(agent.actor_optimizer, torch.tensor(bad))
    assert agent._nonfinite_grad_skips == skips0 + 1, "a skipped step must be counted"
    for b, p in zip(before, agent.actor.parameters()):
        assert torch.equal(b, p.detach()), "parameters moved on a skipped step"


def test_step_if_finite_steps_when_healthy(agent):
    for p in agent.actor.parameters():
        p.grad = torch.ones_like(p)
    skips0 = agent._nonfinite_grad_skips
    agent._step_if_finite(agent.actor_optimizer, torch.tensor(3.0))
    assert agent._nonfinite_grad_skips == skips0, "a healthy step must not be counted as a skip"
    assert _params_finite(agent.actor)


def test_network_survives_a_nan_gradient_with_amp_off(agent):
    """THE REGRESSION: with the scaler disabled, one non-finite gradient must not NaN the
    whole actor. Reproduces the bare-step path taken by use_amp:false and bfloat16."""
    assert not agent.scaler.is_enabled(), "fixture is CPU/AMP-off — the path under test"
    for p in agent.actor.parameters():
        p.grad = torch.zeros_like(p)
    next(agent.actor.parameters()).grad[0][0] = float("nan")
    total = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.gradient_clip)
    agent._step_if_finite(agent.actor_optimizer, total)
    assert _params_finite(agent.actor), "a single NaN grad element NaN'd the entire actor"


def test_amp_off_guard_is_wired_into_both_agents():
    """Declared-but-not-wired check for all four call sites (critic + actor x 2 agents)."""
    import inspect

    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    for cls in (SACAgent, DistributionalSACAgent):
        src = inspect.getsource(cls.train_step_mega)
        assert src.count("_step_if_finite(") == 2, \
            f"{cls.__name__}.train_step_mega must guard BOTH critic and actor on the AMP-off path"
        assert "self.critic_optimizer.step()" not in src, \
            f"{cls.__name__} still has an unguarded bare critic step"
        assert "self.actor_optimizer.step()" not in src, \
            f"{cls.__name__} still has an unguarded bare actor step"


def test_grad_skips_is_exposed_in_metrics():
    """A run that has silently stopped learning must be visible, not look like flat training."""
    import inspect

    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    for cls in (SACAgent, DistributionalSACAgent):
        assert '"grad_skips"' in inspect.getsource(cls.train_step_mega), \
            f"{cls.__name__} does not report grad_skips"
