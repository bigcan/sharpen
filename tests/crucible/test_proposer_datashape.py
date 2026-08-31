"""P3 data-shape proposer context + orchestrator wiring (spec Part A2, F3/F5 of the cont-117 audit).

The orchestrator now hands every proposer three optional CR-1-legal hints — cross-section width
(``panel_n``), per-feature-slot bar COUNTS (``feature_slot_bars``), and a per-tick entropy token
(``mechanism_nonce``). Load-bearing assertions, in moat-priority order:

  * CRU-1: the offline :class:`LibrarySeedProposer` reads none of them, so its batch — and therefore
    every existing verdict/manifest — is byte-identical whether or not they are populated.
  * CR-1: the new fields carry data SHAPE and entropy only; even with a fully-scored ledger, no
    score/verdict/holdout value can reach a proposer through them (they are sourced from the panel +
    tick timestamp, never the ledger).
  * the orchestrator computes + threads them correctly (finite-bar counts, a deterministic nonce).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from sharpen.crucible import (
    OrchestratorStore,
    Substrate,
    TickBudget,
    TrialLedger,
    TrialRecord,
    run_orchestrator_tick,
)
from sharpen.crucible.agentic import HypothesisAuthor, LibrarySeedProposer, LlmProposer
from sharpen.crucible.agentic.llm_proposer import _render_context
from sharpen.crucible.agentic.proposer import HypothesisProposal, ProposalContext
from sharpen.crucible.orchestrator.orchestrator import _feature_slot_bars, _mechanism_nonce
from sharpen.crucible.orchestrator.substrate import PreparedSubstrate
from sharpen.signals.features import Panel, make_synthetic_panel
from sharpen.signals.generation.fitness import FitnessConfig

_TS = "2026-07-06T00:00:00+00:00"
_FORBIDDEN_VALUES = ("secret prior", "0.42", "3.1", "2.2", "PROMISING", "NO_GO")
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)


def _panel_with_slots() -> Panel:
    """A small panel with two feature slots of DIFFERENT depth: one fully-finite, one half-NaN — so a
    bar-count assertion is meaningful (the counts must differ)."""
    t = 120
    deep = np.sin(np.arange(t) / 5.0).astype(np.float64)          # all finite
    shallow = np.full(t, np.nan)
    shallow[:40] = 1.0                                            # only 40 finite bars
    return make_synthetic_panel(T=t, N=8, seed=1,
                                feature_slots={"fred:DGS10": deep, "cot:gold_comm_net": shallow})


# ============================================================ build_context threading ====

def test_build_context_datashape_defaults_are_empty() -> None:
    """A pre-existing single-arg call site gets empty data-shape fields (byte-identical context)."""
    ctx = HypothesisAuthor(LibrarySeedProposer(), TrialLedger(":memory:")).build_context(
        ("close", "macro:regime"))
    assert ctx.panel_n == 0
    assert ctx.feature_slot_bars == ()
    assert ctx.mechanism_nonce == ""


def test_build_context_populates_datashape() -> None:
    ctx = HypothesisAuthor(LibrarySeedProposer(), TrialLedger(":memory:")).build_context(
        ("close", "fred:DGS10"), panel_n=10,
        feature_slot_bars=(("fred:DGS10", 2750),), mechanism_nonce="deadbeef")
    assert ctx.panel_n == 10
    assert ctx.feature_slot_bars == (("fred:DGS10", 2750),)
    assert ctx.mechanism_nonce == "deadbeef"


# ============================================================ CRU-1: library invariance ====

def test_library_proposer_output_invariant_to_datashape() -> None:
    """The frozen-verdict guarantee (CRU-1): the offline proposer ignores every data-shape hint, so
    the exact same formulas/names/types come out with or without them populated. If a future edit made
    LibrarySeedProposer condition on these fields, this locks the regression."""
    proposer = LibrarySeedProposer()
    terminals = ("close", "volume", "fred:DGS10", "cot:gold_comm_net")
    bare = ProposalContext(available_terminals=terminals, max_proposals=128)
    enriched = ProposalContext(available_terminals=terminals, max_proposals=128, panel_n=10,
                               feature_slot_bars=(("cot:gold_comm_net", 40), ("fred:DGS10", 2750)),
                               mechanism_nonce="deadbeef")
    a = proposer.propose(bare)
    b = proposer.propose(enriched)
    assert [(p.name, p.formula, p.candidate_type, p.expected_sign) for p in a] == \
           [(p.name, p.formula, p.candidate_type, p.expected_sign) for p in b]


# ============================================================ CR-1: no score can leak in ====

def test_datashape_fields_carry_no_scores_even_with_scored_ledger(tmp_path) -> None:
    """CR-1 for the NEW surface: a ledger full of verdicts/DSR/holdout deltas cannot leak into the
    data-shape hints — they are sourced from the panel + tick ts, never the ledger. Build a context
    off a scored ledger WITH panel data and assert none of the forbidden score values appear."""
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="aaaa", crucible_version="v", family="altdata",
                              candidate_type="overlay", verdict="NO_GO", dsr=1.9,
                              delta_sr_oos=0.42, marginal_hlz_t=3.1, formula="secret",
                              economic_rationale="secret prior"))
    ledger.record(TrialRecord(candidate_hash="bbbb", crucible_version="v", family="101alpha",
                              candidate_type="cross_sectional", verdict="PROMISING", dsr=2.2))
    ctx = HypothesisAuthor(LibrarySeedProposer(), ledger).build_context(
        ("close", "fred:DGS10"), panel_n=10,
        feature_slot_bars=(("fred:DGS10", 2750),), mechanism_nonce=_mechanism_nonce(_TS))
    blob = repr((ctx.panel_n, ctx.feature_slot_bars, ctx.mechanism_nonce))
    for leaked in _FORBIDDEN_VALUES:
        assert leaked not in blob
    # the killed family (real, from the ledger) still comes through — proving it is a live context
    assert ctx.killed_families == ("altdata",)


# ============================================================ _render_context surfacing ====

def test_render_context_surfaces_datashape_when_present() -> None:
    ctx = ProposalContext(available_terminals=("close", "fred:DGS10"), panel_n=10,
                          feature_slot_bars=(("fred:DGS10", 2750),), mechanism_nonce="deadbeef")
    text = _render_context(ctx)
    assert "Cross-section width: 10 names" in text
    assert "fred:DGS10(2750)" in text
    assert "deadbeef" in text


def test_render_context_omits_datashape_when_absent() -> None:
    """A context without the hints renders exactly the pre-P3 message (no stray '0 names' / nonce)."""
    ctx = ProposalContext(available_terminals=("close", "macro:regime"))
    text = _render_context(ctx)
    assert "Cross-section width" not in text
    assert "Exploration nonce" not in text
    assert "history depth" not in text


# ============================================================ orchestrator helpers ====

def test_feature_slot_bars_counts_finite_and_sorts() -> None:
    bars = _feature_slot_bars(_panel_with_slots())
    assert bars == (("cot:gold_comm_net", 40), ("fred:DGS10", 120))   # sorted by name; finite counts


def test_feature_slot_bars_empty_when_no_slots() -> None:
    assert _feature_slot_bars(make_synthetic_panel(T=30, N=4, seed=0)) == ()


def test_mechanism_nonce_deterministic_rotates_and_12hex() -> None:
    a = _mechanism_nonce(_TS)
    assert a == _mechanism_nonce(_TS)                               # deterministic (reproduce-stable)
    assert a != _mechanism_nonce("2026-07-07T00:00:00+00:00")       # rotates per tick
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


# ============================================================ orchestrator wiring (CR-1 end-to-end) ==

@dataclass
class _Recorder:
    """Records the exact context the orchestrator built, then proposes nothing (so the tick no-ops
    right after — no expensive mine needed to inspect the wiring)."""

    model_id: str = "recorder"
    seen: list[ProposalContext] = field(default_factory=list)

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        self.seen.append(context)
        return []


def _prepared(panel: Panel) -> PreparedSubstrate:
    base = {"tsmom": np.zeros(panel.T), "rates_carry": np.zeros(panel.T)}
    ts = pd.date_range("2014-01-02", periods=panel.T, freq="B").view("int64").astype(np.float64) / 1e9
    return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                             asset_classes=("macro",), snapshot_hash="snap-x")


def test_orchestrator_threads_datashape_into_the_proposer_context(tmp_path) -> None:
    """End-to-end: the orchestrator computes panel_n / finite-bar counts / a deterministic nonce and
    hands them to the proposer — AND (CR-1) a scored ledger leaks nothing through them."""
    panel = _panel_with_slots()
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="dead", crucible_version="v", family="altdata",
                              candidate_type="overlay", verdict="NO_GO", dsr=1.9, delta_sr_oos=0.42))
    rec = _Recorder()
    sub = Substrate(substrate_id="syn", prepare=lambda: _prepared(panel), ledger=ledger,
                    cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK), proposer=rec)
    store = OrchestratorStore(tmp_path / "orch.db")
    run_orchestrator_tick(substrates=[sub], store=store, gates_path="configs/signal_eval.gates.yaml",
                          tick_ts=_TS)

    ctx = rec.seen[0]
    assert ctx.panel_n == panel.N
    assert ctx.feature_slot_bars == (("cot:gold_comm_net", 40), ("fred:DGS10", 120))
    assert ctx.mechanism_nonce == _mechanism_nonce(_TS)
    for leaked in _FORBIDDEN_VALUES:
        assert leaked not in repr((ctx.panel_n, ctx.feature_slot_bars, ctx.mechanism_nonce))


# ============================================================ LLM proposer through the orchestrator ==

class _CannedTransport:
    def __init__(self, formula: str = "rank(delta(close, 5))") -> None:
        self.formula = formula
        self.calls = 0

    def __call__(self, url: str, headers: dict, body: dict) -> dict:
        self.calls += 1
        return {"content": [{"type": "tool_use", "name": "propose_hypotheses",
                             "input": {"proposals": [{"name": "cs-a", "hypothesis": "h",
                                                      "expected_sign": 1,
                                                      "candidate_type": "cross_sectional",
                                                      "formula": self.formula,
                                                      "economic_rationale": "r"}]}}],
                "usage": {"input_tokens": 500, "output_tokens": 300}}


@pytest.mark.slow
def test_orchestrator_llm_proposer_stamps_model_id_and_charges_est_tokens(tmp_path) -> None:
    """The CR-7 wiring for the LLM path: an LLM-backed Substrate mines, stamps a `llm-…` provenance id
    into the manifest, and charges the STATIC est_tokens_per_tick to both the manifest token_cost and
    the tick budget (the orchestrator budgets on the estimate, deterministically — not live usage)."""
    est = 777
    proposer = LlmProposer(api_key="test-key", transport=_CannedTransport())
    sub = Substrate(substrate_id="syn", prepare=lambda: _prepared(make_synthetic_panel(T=320, N=12,
                    seed=3, feature_slots={"macro:regime": np.sin(np.arange(320) / 7.0)})),
                    ledger=TrialLedger(tmp_path / "l.db"), cfg=FitnessConfig(embargo=10),
                    evolve_kwargs=dict(_EK), proposer=proposer, est_tokens_per_tick=est)
    store = OrchestratorStore(tmp_path / "orch.db")
    budget = TickBudget(max_candidates=256)
    res = run_orchestrator_tick(substrates=[sub], store=store,
                                gates_path="configs/signal_eval.gates.yaml", tick_ts=_TS,
                                budget=budget)
    outcome = res.outcomes[0]
    assert outcome.mined is True
    assert outcome.result is not None
    assert outcome.result.manifest.agent_model_id.startswith("llm-claude-sonnet-5-prompt")
    assert outcome.result.manifest.token_cost == est
    assert budget.tokens_spent == est
