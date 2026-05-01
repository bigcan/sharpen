"""Unit tests for EnsembleAgent aggregation rules."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from finrl_pro_ds.agents.sac.ensemble_agent import (
    EnsembleAgent,
    _agreement,
    _make_pf_weighted,
    _mean,
    _median,
)


class _StubAgent:
    """Minimal SACAgent stand-in returning a fixed action from .predict."""

    def __init__(self, action_value: float):
        self._action_value = float(action_value)
        self.actor = _StubActor()

    def predict(self, scale_input, private=None, deterministic=True, lob=None, **kw):
        return torch.tensor([[self._action_value]], dtype=torch.float32)


class _StubActor:
    def __init__(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"
        return self

    def train(self, mode: bool = True):
        self.mode = "train" if mode else "eval"
        return self


# --- aggregation primitives -------------------------------------------------

def test_mean_rule():
    acts = np.array([[0.1], [0.3], [-0.2]])
    assert _mean(acts, 0.25) == pytest.approx(np.array([0.0667]), abs=1e-3)


def test_median_rule():
    acts = np.array([[0.1], [0.3], [-0.2]])
    assert _median(acts, 0.25) == pytest.approx(np.array([0.1]))


def test_agreement_two_long():
    # 2 longs >0.25, 1 flat -> long trade sized to mean of the 2 longs
    acts = np.array([[0.4], [0.5], [0.1]])
    out = _agreement(acts, 0.25)
    assert out == pytest.approx(np.array([0.45]))


def test_agreement_two_short():
    acts = np.array([[-0.4], [-0.6], [0.05]])
    out = _agreement(acts, 0.25)
    assert out == pytest.approx(np.array([-0.5]))


def test_agreement_split_stays_flat():
    # 1 long, 1 short, 1 flat -> no majority -> flat
    acts = np.array([[0.5], [-0.4], [0.0]])
    out = _agreement(acts, 0.25)
    assert out == pytest.approx(np.array([0.0]))


def test_agreement_all_flat():
    acts = np.array([[0.1], [0.0], [-0.1]])  # all within deadband 0.25
    out = _agreement(acts, 0.25)
    assert out == pytest.approx(np.array([0.0]))


def test_pf_weighted_equal_weights_fallback():
    f = _make_pf_weighted({})
    acts = np.array([[0.1], [0.3], [-0.2]])
    out = f(acts, 0.0, order=[42, 789, 456])
    assert out == pytest.approx(np.array([0.0667]), abs=1e-3)  # equal weights = mean


def test_pf_weighted_proper_weights():
    f = _make_pf_weighted({42: 3.0, 789: 2.0, 456: 1.0})
    acts = np.array([[0.60], [0.30], [0.00]])
    # weights normalized: [0.5, 0.333, 0.167] -> 0.5*0.6 + 0.333*0.3 + 0.167*0 = 0.4
    out = f(acts, 0.0, order=[42, 789, 456])
    assert out == pytest.approx(np.array([0.40]), abs=1e-3)


# --- EnsembleAgent integration ---------------------------------------------

def test_ensemble_agent_predict_agreement():
    agents = [_StubAgent(0.4), _StubAgent(0.6), _StubAgent(0.1)]
    ens = EnsembleAgent(
        agents=agents, seeds=[42, 2025, 3141],
        aggregation_rule="ens_agreement", deadband=0.25,
    )
    out = ens.predict(None, None)
    assert out.shape == (1, 1)
    # 2 longs (0.4, 0.6), 1 flat (0.1) -> mean of longs = 0.5
    assert float(out[0, 0]) == pytest.approx(0.5)


def test_ensemble_agent_predict_mean():
    agents = [_StubAgent(0.1), _StubAgent(0.3), _StubAgent(-0.2)]
    ens = EnsembleAgent(
        agents=agents, seeds=[42, 2025, 3141],
        aggregation_rule="ens_mean", deadband=0.25,
    )
    out = ens.predict(None, None)
    assert float(out[0, 0]) == pytest.approx(0.0667, abs=1e-3)


def test_ensemble_agent_requires_min_2_agents():
    with pytest.raises(ValueError, match="requires >=2"):
        EnsembleAgent(agents=[_StubAgent(0.1)], seeds=[42])


def test_ensemble_agent_seed_agent_length_mismatch():
    with pytest.raises(ValueError, match="must match"):
        EnsembleAgent(agents=[_StubAgent(0.1), _StubAgent(0.2)], seeds=[42])


def test_ensemble_agent_unknown_rule():
    with pytest.raises(ValueError, match="unknown aggregation_rule"):
        EnsembleAgent(
            agents=[_StubAgent(0.1), _StubAgent(0.2)],
            seeds=[42, 789], aggregation_rule="bogus",
        )


def test_ensemble_agent_actor_proxy_eval_fans_out():
    agents = [_StubAgent(0.1), _StubAgent(0.2), _StubAgent(0.3)]
    ens = EnsembleAgent(
        agents=agents, seeds=[42, 2025, 3141],
        aggregation_rule="ens_agreement",
    )
    ens.actor.eval()
    assert all(a.actor.mode == "eval" for a in agents)
    ens.actor.train(True)
    assert all(a.actor.mode == "train" for a in agents)


def test_ensemble_agent_pf_weighted_uses_config_weights():
    agents = [_StubAgent(0.6), _StubAgent(0.3), _StubAgent(0.0)]
    ens = EnsembleAgent(
        agents=agents, seeds=[42, 2025, 3141],
        aggregation_rule="ens_pf_weighted", deadband=0.0,
        seed_pfs={42: 3.0, 2025: 2.0, 3141: 1.0},
    )
    out = ens.predict(None, None)
    assert float(out[0, 0]) == pytest.approx(0.40, abs=1e-3)
