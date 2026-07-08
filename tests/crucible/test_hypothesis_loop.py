"""Crucible P2 — the agentic hypothesis loop (spec §7/§8, P2 row).

Load-bearing assertions, in moat-priority order:
  * CR-1 (anti-oracle): the agent's whole input surface (``ProposalContext``) carries NO score /
    verdict / holdout field, even when the ledger is full of scored rows; and the scorer's signature
    has no channel for the economic rationale.
  * CR-2 (pre-registration): specs are written to the ledger with ``verdict=None`` + a ``proposal_ts``
    BEFORE any mining.
  * Guardrails: killed families are dropped and ledger duplicates are deduped BEFORE compute.
  * P2 EXIT GATE: end-to-end on synthetic noise yields 0 PROMISING (null-safety, and deterministic).

Design: docs/research/crucible_agentic_discovery_spec.md §7 (roles/guardrails), §8 (P2 row).
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crucible import TrialLedger, TrialRecord
from finrl_pro_ds.crucible.agentic import (
    DiscoveryCard,
    HypothesisAuthor,
    HypothesisProposal,
    LibrarySeedProposer,
    ProposalContext,
    run_hypothesis_loop,
)
from finrl_pro_ds.crucible.agentic.card import INCUBATION_PENDING
from finrl_pro_ds.crucible.agentic.hypothesis import candidate_hash
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

T, N = 320, 12
_CFG = FitnessConfig(embargo=10)
_TS = "2026-07-02T00:00:00+00:00"
# Score-derived fields that must NEVER appear in anything the agent can read (CR-1).
_FORBIDDEN = {"verdict", "dsr", "delta_sr_oos", "marginal_hlz_t", "holdout", "passes_gate",
              "fdr_wealth_charged", "spec_json", "economic_rationale", "formula"}


# --------------------------------------------------------------------- fixtures ----

def _panel() -> Panel:
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    regime = (np.sin(2.0 * np.pi * np.arange(T) / 80.0) + 0.2 * rng.standard_normal(T)
              ).astype(np.float64)
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"},
                 feature_slots={"macro:regime": regime})


def _base_ts() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(1)
    base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    ts = pd.date_range("2014-01-02", periods=T, freq="B").view("int64").astype(np.float64) / 1e9
    return base, ts


_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)


@dataclass
class RecordingProposer:
    """A proposer that records the exact context it was handed (for the CR-1 leak test)."""

    model_id: str = "recording-stub"
    to_emit: list[HypothesisProposal] = field(default_factory=list)
    seen: list[ProposalContext] = field(default_factory=list)

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        self.seen.append(context)
        return list(self.to_emit)


# --------------------------------------------------------------------- CR-1 moat ----

def test_agent_view_exposes_no_score_columns(tmp_path) -> None:
    """The ledger agent-view columns are exactly the dedup keys — no score/verdict column."""
    assert set(TrialLedger.agent_view_columns()).isdisjoint(_FORBIDDEN)
    assert set(TrialLedger.agent_view_columns()) == {"candidate_hash", "candidate_type", "family"}


def test_proposal_context_carries_no_scores_even_with_scored_ledger(tmp_path) -> None:
    """CR-1 LOAD-BEARING: with a ledger full of scored rows (verdicts, DSR, holdout deltas), the
    ProposalContext the Author hands the proposer contains ONLY dedup keys + killed families —
    no score field, so a proposer cannot hill-climb a reused holdout (Fable finding #4)."""
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="aaaa", crucible_version="v", family="altdata",
                              candidate_type="overlay", verdict="NO_GO", dsr=1.9,
                              delta_sr_oos=0.42, marginal_hlz_t=3.1, formula="secret",
                              economic_rationale="secret prior"))
    ledger.record(TrialRecord(candidate_hash="bbbb", crucible_version="v", family="101alpha",
                              candidate_type="cross_sectional", verdict="PROMISING", dsr=2.2))
    proposer = RecordingProposer()
    author = HypothesisAuthor(proposer, ledger)
    author.propose(author.build_context(("close", "macro:regime")), proposal_ts=_TS)

    ctx = proposer.seen[0]
    flat = asdict(ctx)
    # no forbidden key names anywhere in the context payload
    assert set(flat).isdisjoint(_FORBIDDEN)
    # the actual values carry no leaked scores: only hashes + family names
    assert flat["existing_candidate_hashes"] == frozenset({"aaaa", "bbbb"})
    assert flat["killed_families"] == ("altdata",)          # NO_GO family, no PROMISING member
    blob = repr(flat)
    for leaked in ("secret prior", "0.42", "3.1", "2.2", "PROMISING", "NO_GO"):
        assert leaked not in blob


def test_scorer_signature_cannot_receive_rationale() -> None:
    """Structural CR-1: the mine/deflation entry points have NO parameter through which the agent's
    economic rationale could reach the scorer."""
    from finrl_pro_ds.signals.generation.evolve import evolve
    from finrl_pro_ds.signals.generation.fitness import combination_fitness

    for fn in (evolve, combination_fitness):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"economic_rationale", "rationale", "hypothesis", "narrative"})


# --------------------------------------------------------------------- guardrails ----

def test_killed_family_dropped_before_compute(tmp_path) -> None:
    """A proposal in a killed family is rejected pre-compute (spec §6/§7.1)."""
    ledger = TrialLedger(tmp_path / "l.db")
    ledger.record(TrialRecord(candidate_hash="dead", crucible_version="v", family="altdata",
                              verdict="NO_GO"))
    proposer = RecordingProposer(to_emit=[
        HypothesisProposal("ov-x", "h", "altdata", 1, "overlay", "macro:regime"),
        HypothesisProposal("cs-y", "h", "101alpha", -1, "cross_sectional", "rank(close)"),
    ])
    author = HypothesisAuthor(proposer, ledger)
    specs = author.propose(author.build_context(("close", "macro:regime")), proposal_ts=_TS)
    fams = {s.spec.family for s in specs}
    assert "altdata" not in fams and "101alpha" in fams
    assert author.last_proposal_stats == {"raw": 2, "accepted": 1, "dropped": 1}


def test_ledger_duplicate_deduped_before_compute(tmp_path) -> None:
    """An exact prior candidate (by canonical-formula hash) is not re-proposed."""
    ledger = TrialLedger(tmp_path / "l.db")
    chash = candidate_hash("rank(close)")
    ledger.record(TrialRecord(candidate_hash=chash, crucible_version="v", family="101alpha",
                              verdict="LOGGED"))
    proposer = RecordingProposer(to_emit=[
        HypothesisProposal("cs-dup", "h", "101alpha", -1, "cross_sectional", "rank(close)"),
        HypothesisProposal("cs-new", "h", "101alpha", 1, "cross_sectional", "rank(volume)"),
    ])
    author = HypothesisAuthor(proposer, ledger)
    specs = author.propose(author.build_context(("close", "volume")), proposal_ts=_TS)
    assert [s.spec.name for s in specs] == ["cs-new"]


def test_invalid_formula_dropped(tmp_path) -> None:
    """A proposal whose formula does not grammar-parse is dropped, not raised."""
    ledger = TrialLedger(tmp_path / "l.db")
    proposer = RecordingProposer(to_emit=[
        HypothesisProposal("bad", "h", "101alpha", 1, "cross_sectional", "this is not dsl ((("),
        HypothesisProposal("ok", "h", "101alpha", 1, "cross_sectional", "rank(close)"),
    ])
    author = HypothesisAuthor(proposer, ledger)
    specs = author.propose(author.build_context(("close",)), proposal_ts=_TS)
    assert [s.spec.name for s in specs] == ["ok"]


# --------------------------------------------------------------------- CR-2 preregister ----

def test_preregistration_writes_rows_with_null_verdict_and_ts(tmp_path) -> None:
    """CR-2: preregister locks the hash into the ledger with proposal_ts and NO verdict yet."""
    ledger = TrialLedger(tmp_path / "l.db")
    author = HypothesisAuthor(LibrarySeedProposer(include_overlay=False), ledger)
    specs = author.propose(author.build_context(("close", "volume", "returns")), proposal_ts=_TS)
    assert specs
    author.preregister(specs, run_id="r0", crucible_version="crucible-vX")
    conn = ledger._conn
    for pr in specs:
        row = conn.execute(
            "SELECT verdict, proposal_ts, spec_json FROM trial_ledger WHERE candidate_hash=?",
            (pr.candidate_hash,)).fetchone()
        assert row["verdict"] is None                       # not scored yet
        assert row["proposal_ts"] == _TS
        assert "spec_content_hash" in row["spec_json"]


# --------------------------------------------------------------------- discovery card ----

def test_discovery_card_roundtrip_and_incubation_pending(tmp_path) -> None:
    card = DiscoveryCard(candidate_hash="abcd", formula="rank(close)",
                         candidate_type="cross_sectional", crucible_version="crucible-vX",
                         gates_hash="ffff", holdout_delta_sr=0.05, n_paths=15, verdict="PROMISING")
    assert card.incubation_status == INCUBATION_PENDING     # CR-8: no card is human-eligible in P2
    assert card.eligible_for_human_gate is False
    again = DiscoveryCard.from_json(card.to_json())
    assert again == card
    path = card.write(tmp_path)
    assert path.exists() and path.name == "card_abcd.json"


# --------------------------------------------------------------------- P2 EXIT GATE ----

@pytest.mark.slow
def test_exit_gate_synthetic_zero_promising_and_deterministic(tmp_path) -> None:
    """P2 EXIT GATE: the full manual loop on a synthetic NOISE panel yields 0 PROMISING (the filter
    is the point), populates the manifest agent fields, and is deterministic (same verdicts twice)."""
    panel = _panel()
    base, ts = _base_ts()

    def _run(db: str):
        ledger = TrialLedger(tmp_path / db)
        author = HypothesisAuthor(LibrarySeedProposer(), ledger)
        return run_hypothesis_loop(
            panel=panel, base_returns=base, timestamps=ts, cfg=_CFG, evolve_kwargs=dict(_EK),
            author=author, run_id="hyp-syn", crucible_version="crucible-vX", gates_hash="ffff",
            proposal_ts=_TS, data_snapshot_hash="snap")

    r1 = _run("l1.db")
    assert r1.n_promising == 0                              # null-safety on noise
    assert {s.spec.candidate_type for s in r1.specs} == {"cross_sectional", "overlay"}
    assert set(r1.reports) == {"cross_sectional", "overlay"}
    # manifest carries the agentic provenance (CR-7) the pre-P2 manifest left null
    assert r1.manifest.agent_model_id == "library-seed-v1"
    assert r1.manifest.token_cost == 0
    assert r1.manifest.proposal_ts == _TS
    assert r1.manifest.file_drawer_N_after > r1.manifest.file_drawer_N_before

    r2 = _run("l2.db")
    assert r1.manifest.verdicts == r2.manifest.verdicts     # deterministic
    assert r1.n_promising == r2.n_promising == 0


# --------------------------------------------------------------------- card linkage ----
# The exit-gate loop yields 0 survivors, so the card-building path (prereg linkage + verbatim
# verdict copy) is never exercised end-to-end. These unit-test _card_for directly on a hand-built
# PROMISING survivor — the property that a survivor's card carries its pre-registered rationale and
# copies the scorer numbers verbatim (CR-1), not paraphrased.

def _fitness_result() -> object:
    from finrl_pro_ds.signals.generation.fitness import FitnessResult
    return FitnessResult(fitness=0.9, delta_sr_oos=0.08, delta_sr_median=0.06,
                         frac_paths_positive=0.8, delta_sr_p05=-0.01, n_paths=15, dsr_aug=1.7,
                         cand_hlz_pass=True, turnover_ann=1.2, n_nodes=3, passes_gate=True,
                         marginal_t=3.3)


def test_card_links_survivor_to_preregistered_spec_verbatim() -> None:
    from finrl_pro_ds.crucible.agentic.hypothesis import PreRegisteredSpec
    from finrl_pro_ds.crucible.agentic.loop import _card_for
    from finrl_pro_ds.signals.generation.evolve import Candidate, GenerationReport
    from finrl_pro_ds.signals.spec import SignalSpec

    from finrl_pro_ds.signals.generation.grammar import parse, to_formula
    formula = to_formula(parse("rank(delta(close, 20))"))    # canonical form (matches the mine's key)
    chash = candidate_hash(formula)
    res = _fitness_result()
    cand = Candidate(formula=formula, fitness=0.9, result=res)               # type: ignore[arg-type]
    report = GenerationReport(hall_of_fame=[cand], gen_n_total=5, gen_n_eff=5.0,
                              holdout_validation=[{"formula": formula, "holdout_delta": 0.05,
                                                   "holdout_passes": True}], promising=[cand])
    spec = SignalSpec(name="cs-rev", hypothesis="reversal", family="101alpha", expected_sign=-1,
                      candidate_type="cross_sectional")
    pr = PreRegisteredSpec(spec=spec, formula=formula, candidate_hash=chash,
                           economic_rationale="pre-registered reversal prior", proposal_ts=_TS)

    card = _card_for(cand, "cross_sectional", report, {chash: pr}, crucible_version="crucible-vX",
                     gates_hash="ffff", data_snapshot_hash="snap")
    # linked to the pre-registration (CR-2): rationale + proposal_ts carried
    assert card.economic_rationale == "pre-registered reversal prior"
    assert card.proposal_ts == _TS
    assert card.spec["spec_content_hash"] == spec.content_hash()
    # verbatim scorer numbers (CR-1) — copied, not paraphrased
    assert card.holdout_delta_sr == 0.05
    # C7-10: combiner_marginal_delta_sr is None until a REAL combiner pass exists — it must NOT be a
    # verbatim alias of holdout_delta_sr (two "independent" numbers that are one mislead a Tier-2 read).
    assert card.combiner_marginal_delta_sr is None
    assert card.holdout_passes is True
    assert card.dsr_aug == 1.7 and card.marginal_t == 3.3 and card.n_paths == 15
    assert card.train_delta_sr_oos == 0.08 and card.delta_sr_median == 0.06
    # CR-8: never human-eligible in P2
    assert card.incubation_status == INCUBATION_PENDING and card.eligible_for_human_gate is False


def test_card_marks_evolved_offspring_when_no_prereg_match() -> None:
    """A survivor with no matching pre-registered seed (an evolved offspring) is labelled as such and
    carries no proposal_ts — it is not a pre-registered hypothesis."""
    from finrl_pro_ds.crucible.agentic.loop import _card_for
    from finrl_pro_ds.signals.generation.evolve import Candidate, GenerationReport
    from finrl_pro_ds.signals.generation.grammar import parse, to_formula

    formula = to_formula(parse("rank(sum(returns, 5))"))
    cand = Candidate(formula=formula, fitness=0.5, result=_fitness_result())  # type: ignore[arg-type]
    report = GenerationReport(hall_of_fame=[cand], gen_n_total=5, gen_n_eff=5.0,
                              holdout_validation=[{"formula": formula, "holdout_delta": 0.04,
                                                   "holdout_passes": True}], promising=[cand])
    card = _card_for(cand, "cross_sectional", report, {}, crucible_version="crucible-vX",
                     gates_hash="ffff", data_snapshot_hash="snap")
    assert card.economic_rationale == "evolved offspring (search-derived)"
    assert card.proposal_ts is None and card.spec == {}
    assert card.holdout_delta_sr == 0.04
