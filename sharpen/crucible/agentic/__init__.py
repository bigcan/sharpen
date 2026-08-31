"""Crucible agentic layer (spec §7) — the Hypothesis Author loop (P2, MANUAL).

CR-1 governs everything here: **the agent proposes, the statistics dispose.** The Hypothesis
Author reads ONLY the agent-visible ledger projection (dedup keys + killed families) plus the data
catalog and economic priors; it never sees a verdict, DSR, or holdout outcome. It emits
*pre-registered* :class:`~sharpen.signals.spec.SignalSpec`s (content-hashed before scoring,
CR-2) whose ``candidate_type`` (CR-9) is chosen per substrate — ``overlay`` for low-breadth
macro/positioning feature slots, ``cross_sectional`` for the OHLCV world. Those seeds feed the
UNCHANGED C3 ``evolve`` mine → T0–T5 funnel; survivors become :class:`DiscoveryCard`s.

The LLM is an *injectable seam* (:class:`Proposer`), mirroring the P1b connector ``transport``:
the shipped default (:class:`LibrarySeedProposer`) is deterministic and offline so the whole loop
runs and tests reproducibly (the P2 exit gate is a synthetic ``0 PROMISING`` null-safety run, like
``generate_alphas --mode synthetic``). An LLM-backed proposer is a drop-in for the P3 continuous
orchestrator. Spec: ``docs/research/crucible_agentic_discovery_spec.md`` §7, §8 (P2 row).
"""
from __future__ import annotations

from .card import DiscoveryCard
from .hypothesis import HypothesisAuthor, PreRegisteredSpec
from .llm_proposer import LlmProposer
from .loop import HypothesisLoopResult, run_hypothesis_loop
from .proposer import (
    HypothesisProposal,
    LibrarySeedProposer,
    ProposalContext,
    Proposer,
)
from .scout import DataScout, ScoutFinding, ScoutReport

__all__ = [
    "DataScout",
    "DiscoveryCard",
    "HypothesisAuthor",
    "HypothesisLoopResult",
    "HypothesisProposal",
    "LibrarySeedProposer",
    "LlmProposer",
    "PreRegisteredSpec",
    "ProposalContext",
    "Proposer",
    "ScoutFinding",
    "ScoutReport",
    "run_hypothesis_loop",
]
