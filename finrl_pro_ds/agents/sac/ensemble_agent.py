"""Ensemble wrapper over multiple SACAgent instances for live inference.

Implements the same `.predict()` signature as SACAgent so LiveTradingEngine can
swap a solo agent for an ensemble without engine-side changes.

Aggregation rules match scripts/sg1_xauusd_ensemble_eval.py:
  - ens_mean         np.mean of per-agent actions
  - ens_median       np.median
  - ens_agreement    deadband-classify each agent's raw action into
                     {short, flat, long}; trade only if >=2 agents agree on a
                     directional label; size = mean of the agreeing agents
  - ens_pf_weighted  weighted mean using seed->PF weights from config
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _mean(actions: np.ndarray, _deadband: float) -> np.ndarray:
    return actions.mean(axis=0)


def _median(actions: np.ndarray, _deadband: float) -> np.ndarray:
    return np.median(actions, axis=0)


def _agreement(actions: np.ndarray, deadband: float) -> np.ndarray:
    """Majority-direction vote with size = mean of agreeing agents.

    Matches `_agg_agreement` in scripts/sg1_xauusd_ensemble_eval.py so live
    aggregation reproduces the WF-verified behavior bit-for-bit.

    PRECONDITION: single-asset action space. The directional label is taken
    from `actions[:, 0]`; for a multi-asset ensemble the vote would need to
    be computed per action dimension. Current live customer (SG-1 XAUUSD) has
    action_dim=1, so this is safe.
    """
    labels = np.zeros(actions.shape[0], dtype=int)
    first_dim = actions[:, 0]
    labels[first_dim > deadband] = 1
    labels[first_dim < -deadband] = -1
    n_long = int((labels == 1).sum())
    n_short = int((labels == -1).sum())
    if n_long >= 2:
        return actions[labels == 1].mean(axis=0)
    if n_short >= 2:
        return actions[labels == -1].mean(axis=0)
    return np.zeros_like(actions[0])


def _make_pf_weighted(seed_pfs: Dict[int, float]):
    """Factory: returns a weighted-mean aggregator whose weights come from the
    supplied seed->PF map. Falls back to equal weights if any seed missing."""
    def f(actions: np.ndarray, _deadband: float, order: Optional[List[int]] = None) -> np.ndarray:
        if order is None or any(seed_pfs.get(s) is None or seed_pfs.get(s, 0) <= 0 for s in order):
            w = np.ones(actions.shape[0], dtype=float)
        else:
            w = np.array([seed_pfs[s] for s in order], dtype=float)
        w = w / w.sum()
        return (actions * w[:, None]).sum(axis=0)
    return f


_RULES = {
    "ens_mean":     _mean,
    "ens_median":   _median,
    "ens_agreement": _agreement,
}


class EnsembleAgent:
    """Drop-in replacement for SACAgent that aggregates N SAC actors.

    Args:
        agents: ordered list of loaded SACAgent instances (actor.eval() each)
        seeds:  ordered list of seeds matching `agents` (used only for
                pf_weighted routing and diagnostics)
        aggregation_rule: one of ens_mean/ens_median/ens_agreement/ens_pf_weighted
        deadband: classifier threshold for ens_agreement; should match the
                  env.deadband_threshold the agents were trained under
        seed_pfs: seed -> WF/L1 test PF; required for ens_pf_weighted
    """

    def __init__(
        self,
        agents: List[Any],
        seeds: List[int],
        aggregation_rule: str = "ens_agreement",
        deadband: float = 0.25,
        seed_pfs: Optional[Dict[int, float]] = None,
    ):
        if len(agents) != len(seeds):
            raise ValueError(f"agents ({len(agents)}) and seeds ({len(seeds)}) must match")
        if len(agents) < 2:
            raise ValueError(f"ensemble requires >=2 agents, got {len(agents)}")
        self.agents = agents
        self.seeds = list(seeds)
        self.aggregation_rule = aggregation_rule
        self.deadband = float(deadband)
        self.seed_pfs = dict(seed_pfs or {})

        if aggregation_rule == "ens_pf_weighted":
            self._rule_fn = _make_pf_weighted(self.seed_pfs)
            self._rule_needs_order = True
        elif aggregation_rule in _RULES:
            self._rule_fn = _RULES[aggregation_rule]
            self._rule_needs_order = False
        else:
            raise ValueError(
                f"unknown aggregation_rule '{aggregation_rule}'; "
                f"expected one of {list(_RULES) + ['ens_pf_weighted']}"
            )

        # Mirror the single-agent .actor attribute so engine-side code that
        # toggles .actor.eval() (e.g., deterministic inference guard) works.
        self.actor = _ActorProxy([a.actor for a in agents])

    def predict(
        self,
        scale_input,
        private: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        lob: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Aggregate per-seed predictions into a single (1, 1) tensor.

        Matches SACAgent.predict signature. Runs each agent under no_grad; the
        aggregation happens on CPU numpy arrays (trivial cost vs inference)."""
        per_agent = []
        for a in self.agents:
            act = a.predict(scale_input, private,
                            deterministic=deterministic, lob=lob, **kwargs)
            per_agent.append(act.detach().cpu().numpy())
        stacked = np.stack(per_agent, axis=0)  # (N, B, 1) typically (N, 1, 1)
        # flatten batch to pass (N, action_dim) to the rule
        if stacked.ndim == 3:
            batch_size = stacked.shape[1]
            out = np.zeros((batch_size, stacked.shape[2]), dtype=stacked.dtype)
            for b in range(batch_size):
                slice_b = stacked[:, b, :]  # (N, action_dim)
                if self._rule_needs_order:
                    out[b] = self._rule_fn(slice_b, self.deadband, order=self.seeds)
                else:
                    out[b] = self._rule_fn(slice_b, self.deadband)
            return torch.as_tensor(out)
        # (N, action_dim) path
        if self._rule_needs_order:
            agg = self._rule_fn(stacked, self.deadband, order=self.seeds)
        else:
            agg = self._rule_fn(stacked, self.deadband)
        return torch.as_tensor(agg).unsqueeze(0)


class _ActorProxy:
    """Lets engine-side `agent.actor.eval()` / `.train()` fan out to all underlying actors."""

    def __init__(self, actors):
        self._actors = actors

    def eval(self):
        for a in self._actors:
            a.eval()
        return self

    def train(self, mode: bool = True):
        for a in self._actors:
            a.train(mode)
        return self
