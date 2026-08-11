"""The manual agentic hypothesis loop (spec §8, P2 row) — Author → mine → deflate → card.

This is the P2 deliverable wired end-to-end, reusing every downstream stage UNCHANGED:

    Author.propose (CR-1) → preregister (CR-2) → C3 ``evolve`` (mine + T0–T5 deflate, per
    candidate_type) → record verdicts to the ledger → DiscoveryCard per PROMISING survivor →
    run_manifest (now with ``agent_model_id`` / ``token_cost``, CR-7).

It is *manual* in the P2 sense: one call = one pass over one prepared substrate, no scheduler and no
``substrate_dirty`` gate (that is P3). The exit gate is null-safety: on a synthetic panel the loop
yields 0 PROMISING (the cont-73 lesson — the filter is the point). Nothing here can promote past
PROMISING; every card is ``incubation_status=PENDING_P4`` / not human-eligible (CR-8).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...signals.features import Panel
from ...signals.generation.cohort import CohortConfig
from ...signals.generation.cohort_eval import (
    derive_cohort_seed,
    evaluate_cohort,
    pool_content_hash,
)
from ...signals.generation.evolve import (
    CONTRACT_CORRECTED,
    CONTRACT_SHIPPED,
    Candidate,
    GenerationReport,
    evolve,
)
from ...signals.generation.fitness import FitnessConfig
from ...signals.generation.grammar import available_terminals
from ..ledger import TrialRecord
from ..manifest import RunManifest
from ..search_memory import classify_rejection
from .card import DiscoveryCard
from .cohort_card import CohortCard, card_from_verdict
from .hypothesis import HypothesisAuthor, PreRegisteredSpec, candidate_hash

if TYPE_CHECKING:
    from ...signals.generation.base_sleeves import SleeveComponents
    from ..corrected_contract import CorrectedConfig
    from ..search_memory import SearchMemoryConfig

log = logging.getLogger("crucible.loop")

# Non-killing verdict for a mined genome that was scored but not selected — it is a non-survivor,
# NOT a falsified named family, so it must not enter killed_families (which gates future proposals).
_LOGGED = "LOGGED"
_PROMISING = "PROMISING"
# Terminal verdict for a PRE-REGISTERED spec that was mined but did NOT surface in its type's
# hall-of-fame/promising set — "scored and lost", distinct from "never scored" (NOW-3, C3-05/C6-06/
# C7-07). Like _LOGGED it is NON-killing: a non-survivor must never enter killed_families.
_SCORED_NOT_SELECTED = "SCORED_NOT_SELECTED"


@dataclass(frozen=True, slots=True)
class HypothesisLoopResult:
    """Outcome of one manual loop pass."""

    specs: list[PreRegisteredSpec]
    reports: dict[str, GenerationReport]          # candidate_type -> GenerationReport
    cards: list[DiscoveryCard]
    manifest: RunManifest
    n_promising: int = 0
    dropped: int = 0                              # proposals rejected pre-compute (dedup/killed/parse)
    cohort_cards: list[CohortCard] = field(default_factory=list)   # Phase 4: opt-in cohort verdicts
    extra: dict = field(default_factory=dict)


def _holdout_entry(report: GenerationReport, formula: str) -> dict | None:
    """The binding held-out validation record for ``formula`` (evolve re-scores survivors there)."""
    for hv in report.holdout_validation:
        if hv.get("formula") == formula:
            return hv
    return None


def _card_for(cand: Candidate, ct: str, report: GenerationReport,
              prereg_by_hash: dict[str, PreRegisteredSpec], *, crucible_version: str,
              gates_hash: str, data_snapshot_hash: str | None) -> DiscoveryCard:
    """Build a DiscoveryCard for a PROMISING survivor — verdict fields copied verbatim (CR-1)."""
    chash = candidate_hash(cand.formula)
    res = cand.result
    hv = _holdout_entry(report, cand.formula) or {}
    pr = prereg_by_hash.get(chash)               # matched pre-registered seed, if the survivor is one
    holdout_delta = hv.get("holdout_delta")
    return DiscoveryCard(
        candidate_hash=chash, formula=cand.formula, candidate_type=ct,
        crucible_version=crucible_version, gates_hash=gates_hash,
        proposal_ts=(pr.proposal_ts if pr else None), data_snapshot_hash=data_snapshot_hash,
        spec=(_spec_json(pr) if pr else {}),
        economic_rationale=(pr.economic_rationale if pr else "evolved offspring (search-derived)"),
        verdict=_PROMISING,
        holdout_passes=bool(hv.get("holdout_passes")) if "holdout_passes" in hv else None,
        holdout_delta_sr=(float(holdout_delta) if holdout_delta is not None else None),
        train_delta_sr_oos=(None if res is None else float(res.delta_sr_oos)),
        delta_sr_median=(None if res is None else float(res.delta_sr_median)),
        frac_paths_positive=(None if res is None else float(res.frac_paths_positive)),
        dsr_aug=(None if res is None else float(res.dsr_aug)),
        marginal_t=(None if res is None else float(res.marginal_t)),
        n_paths=(None if res is None else int(res.n_paths)),
        # C7-10: there is no separate combiner pass yet, so this was a verbatim ALIAS of
        # holdout_delta_sr — the SAME number surfaced twice reads to a Tier-2 auditor as two
        # independent confirmations. None it until a real marginal-combiner delta is computed.
        combiner_marginal_delta_sr=None,
    )


def _economic_floor(cfg: FitnessConfig, corrected_cfg: "CorrectedConfig | None") -> float:
    """The smallest marginal ΔSR the ACTIVE decision contract would call a discovery — the yardstick U4
    measures a rejection's decisiveness against. Under the corrected contract that is
    ``guards.uplift_min`` (calibrated against its own null by U7); under the shipped funnel it is the
    equivalent ``min_combination_uplift`` leg. Read from the live config, never hardcoded."""
    return float(corrected_cfg.uplift_min if corrected_cfg is not None
                 else cfg.min_combination_uplift)


def _spec_json(pr: PreRegisteredSpec) -> dict:
    from .hypothesis import _spec_dict
    return {"spec": _spec_dict(pr.spec), "spec_content_hash": pr.spec.content_hash()}


def run_hypothesis_loop(
    *,
    panel: Panel,
    base_returns: dict[str, np.ndarray],
    timestamps: np.ndarray,
    cfg: FitnessConfig,
    evolve_kwargs: dict,
    author: HypothesisAuthor,
    run_id: str,
    crucible_version: str,
    gates_hash: str,
    proposal_ts: str,
    catalog_asset_classes: tuple[str, ...] = (),
    data_snapshot_hash: str | None = None,
    token_cost: int | None = 0,
    pre_proposed: list[PreRegisteredSpec] | None = None,
    cohort_cfg: CohortConfig | None = None,
    cohort_mc_kwargs: dict | None = None,
    cohort_gates_hash: str | None = None,
    base_components: "dict[str, SleeveComponents] | None" = None,
    contract: str = CONTRACT_SHIPPED,
    corrected_cfg: "CorrectedConfig | None" = None,
    corrected_gates_hash: str | None = None,
    lord_level: float | None = None,
    search_memory_cfg: "SearchMemoryConfig | None" = None,
    search_memory_gates_hash: str | None = None,
    substrate_mde: float | None = None,
    cohort_pool: list[tuple[str, str, str]] | None = None,
) -> HypothesisLoopResult:
    """Run one manual pass. ``evolve_kwargs`` is the runner block from ``load_generation_config``
    (rng_seed/pop_size/… — WITHOUT ``candidate_type``, which the loop sets per group).

    ``pre_proposed`` lets a caller (the P3 orchestrator) hand in specs it already obtained from
    ``author.propose`` — so the substrate_dirty check and the mine share ONE proposer call rather
    than paying an LLM proposer's tokens twice (CR-7). When None (the P2 default) the loop proposes
    itself; either way ``author.last_proposal_stats`` carries the drop tally from that single call.

    ``contract`` / ``corrected_cfg`` / ``lord_level`` (crucible-v6.0) select and parameterize the
    decision layer ``evolve`` applies on the embargoed holdout; ``corrected_gates_hash`` is the
    corrected thresholds file's byte hash, pinned into the manifest symmetrically with ``gates_hash``.
    Defaults reproduce the shipped funnel exactly.

    ``search_memory_cfg`` / ``substrate_mde`` (U4, crucible-v10.0) enable the REJECTION CLASSIFICATION
    that makes ``ledger.killed_families()`` able to fire at all: a rejected candidate is stamped
    ``DECISIVE`` (the substrate had the power to resolve an economically interesting edge and the answer
    was no ⇒ terminal, the family dies) or ``UNDERPOWERED`` (it did not ⇒ parked, re-testable when the
    substrate deepens). Both default to None ⇒ no classification, byte-identical to pre-U4: the
    ``verdict`` column and the manifest are untouched either way, since the class lives in its own
    nullable ledger column (:mod:`crucible.search_memory`)."""
    ledger = author.ledger
    n_before = ledger.count()

    # --- Stage 2: HYPOTHESIZE (agent, CR-1) -------------------------------------------------------
    if pre_proposed is None:
        terminals = available_terminals(panel)
        context = author.build_context(terminals, asset_classes=catalog_asset_classes)
        specs = author.propose(context, proposal_ts=proposal_ts)
    else:
        specs = pre_proposed
    dropped = author.last_proposal_stats["dropped"]              # recorded by propose(), no re-call
    author.preregister(specs, run_id=run_id, crucible_version=crucible_version,
                       data_snapshot_hash=data_snapshot_hash)
    prereg_by_hash = {pr.candidate_hash: pr for pr in specs}
    log.info("preregistered %d specs (run %s, ts %s)", len(specs), run_id, proposal_ts)

    # --- Stage 3+4: MINE + DEFLATE, per candidate_type (evolve unchanged) -------------------------
    ek = {k: v for k, v in evolve_kwargs.items() if k != "candidate_type"}
    # NOW-11B (C3-03): derive this tick's generation RNG seed from the pinned proposal_ts, so successive
    # nights explore FRESH trajectories instead of re-walking the same rng_seed=7 every night. It is
    # deterministic in proposal_ts, so `crucible reproduce` (same pinned ts) re-derives the identical
    # seed; the base seed still comes from the gates YAML (not hardcoded). The manifest's
    # rng_seeds["generation"] then records the DERIVED seed automatically. [crucible-v2.9 MINOR]
    _base_seed = int(ek.get("rng_seed", 7))
    ek["rng_seed"] = int(hashlib.sha256(
        f"{_base_seed}|{proposal_ts}".encode("utf-8")).hexdigest(), 16) % (2**32)
    reports: dict[str, GenerationReport] = {}
    cards: list[DiscoveryCard] = []
    verdicts: dict[str, str] = {}
    for ct in ("cross_sectional", "overlay"):
        seeds = [pr.formula for pr in specs if pr.spec.candidate_type == ct]
        if not seeds:
            continue
        log.info("mining %d %s seeds (contract=%s)", len(seeds), ct, contract)
        report = evolve(seeds, panel, base_returns, timestamps, cfg, candidate_type=ct,
                        base_components=base_components, contract=contract,
                        corrected_cfg=corrected_cfg, lord_level=lord_level, **ek)
        reports[ct] = report
        promising_hashes = {candidate_hash(c.formula) for c in report.promising}
        # Record every surfaced genome to the ledger (file-drawer): the hall-of-fame UNION the
        # gate-passers. report.promising is NOT guaranteed a subset of hall_of_fame, so a PROMISING
        # candidate ranked outside the top-K would otherwise get a card but no ledger verdict/manifest
        # entry (the C3-05 latent break). Offspring carry family=None so a non-survivor never pollutes
        # killed_families; the pre-registered seeds keep their family.
        surfaced: dict[str, Candidate] = {}
        for c in (*report.hall_of_fame, *report.promising):
            surfaced.setdefault(candidate_hash(c.formula), c)
        for chash, c in surfaced.items():
            verdict = _PROMISING if chash in promising_hashes else _LOGGED
            verdicts[chash] = verdict
            pr = prereg_by_hash.get(chash)
            res = c.result
            # U4: classify only a candidate that ACTUALLY REACHED the holdout gate and failed it. A
            # train-pre-filter cull (or an offspring blocked by `prereg_only`) was never tested
            # out-of-sample, so calling it a "rejection" — decisive or otherwise — would invent a
            # negative result that no test produced. `_holdout_entry` returns None for those.
            rej_class = None
            if verdict != _PROMISING and search_memory_cfg is not None:
                hv_e = _holdout_entry(report, c.formula)
                if hv_e is not None and hv_e.get("holdout_passes") is False:
                    rej_class = classify_rejection(
                        implied_mde=substrate_mde, economic_floor=_economic_floor(cfg, corrected_cfg),
                        cfg=search_memory_cfg)
            ledger.record(TrialRecord(
                candidate_hash=chash, crucible_version=crucible_version,
                family=(pr.spec.family if pr else None), candidate_type=ct,
                formula=c.formula,
                economic_rationale=(pr.economic_rationale if pr else None),
                first_seen_run=run_id, proposal_ts=(pr.proposal_ts if pr else proposal_ts),
                verdict=verdict,
                # `res` is the TRAIN pre-filter FitnessResult, so all three of these are train-split
                # CPCV values, NOT out-of-sample (the card layer names the same delta
                # `train_delta_sr_oos`; S553-cont-131). Under contract="corrected" `dsr` and
                # `marginal_hlz_t` are DIAGNOSTICS ONLY — the v6.0 decision drops both legs, and the
                # binding statistic is the holdout JKM z recorded in `report.holdout_validation`
                # (`corrected_t` / `p_value`). Read `manifest.contract` before interpreting them.
                dsr=(None if res is None else float(res.dsr_aug)),
                delta_sr_oos=(None if res is None else float(res.delta_sr_oos)),
                marginal_hlz_t=(None if res is None else float(res.marginal_t)),
                data_snapshot_hash=data_snapshot_hash,
                rejection_class=rej_class,
                implied_mde_at_test=(substrate_mde if rej_class is not None else None)))
        for c in report.promising:
            cards.append(_card_for(c, ct, report, prereg_by_hash, crucible_version=crucible_version,
                                   gates_hash=gates_hash, data_snapshot_hash=data_snapshot_hash))

    # NOW-3 (C3-05/C6-06/C7-07): give EVERY pre-registered spec a terminal ledger verdict so "scored
    # and lost" is distinguishable from "never scored" (124/184 taiwan_v2 rows were stuck verdict=None).
    # A prereg seed mined but not surfaced in its type's hall-of-fame/promising set is a non-survivor:
    # SCORED_NOT_SELECTED (NON-killing). LEDGER-ONLY — the run manifest's `verdicts` keeps documenting
    # surfaced genomes, so a synthetic null run (0 promising) stays byte-identical (reproduce contract).
    # The monotone upsert (C7-04) preserves each row's CR-2 spec_json / proposal_ts / family.
    for pr in specs:
        if pr.candidate_hash in verdicts:
            continue
        ledger.record(TrialRecord(
            candidate_hash=pr.candidate_hash, crucible_version=crucible_version,
            family=pr.spec.family, candidate_type=pr.spec.candidate_type, formula=pr.formula,
            economic_rationale=pr.economic_rationale, first_seen_run=run_id,
            proposal_ts=pr.proposal_ts, verdict=_SCORED_NOT_SELECTED,
            data_snapshot_hash=data_snapshot_hash))

    # --- Cohort gate (Phase 4, Doc 1/2): OPT-IN weak-signal ensemble over THIS tick's OVERLAY pool.
    # Reads scored return streams only (post-moat, CR-1). Disabled ⇒ no-op AND manifest byte-identical
    # (cohort_prov stays {} → the cohort manifest fields keep their empty defaults). The provenance is
    # PINNED into the reproduce contract (not the non-gated `extra`) because cohort verdicts + the MC
    # p-value are decision-bearing. Caps at PROMISING (Tier-2 for capital). ------------------------
    cohort_cards, cohort_prov = _evaluate_cohort_gate(
        pool=(cohort_pool if cohort_pool is not None
              else [(pr.candidate_hash, pr.formula, pr.spec.candidate_type) for pr in specs]),
        panel=panel, base_returns=base_returns, timestamps=timestamps, cfg=cfg, ek=ek,
        run_id=run_id, crucible_version=crucible_version, gates_hash=gates_hash,
        proposal_ts=proposal_ts, data_snapshot_hash=data_snapshot_hash, cohort_cfg=cohort_cfg,
        cohort_mc_kwargs=cohort_mc_kwargs, cohort_gates_hash=cohort_gates_hash,
        base_components=base_components)

    n_after = ledger.count()
    manifest = RunManifest(
        run_id=run_id, crucible_version=crucible_version, gates_hash=gates_hash,
        proposal_ts=proposal_ts, rng_seeds={"generation": int(ek.get("rng_seed", 7))},
        file_drawer_N_before=n_before, file_drawer_N_after=n_after,
        data_snapshot_hash=data_snapshot_hash, agent_model_id=author.proposer.model_id,
        token_cost=token_cost, verdicts=verdicts,
        cohort_gates_hash=cohort_prov.get("cohort_gates_hash"),
        cohort_verdicts=cohort_prov.get("cohort_verdicts", {}),
        cohort_card_hashes=cohort_prov.get("cohort_card_hashes", {}),
        # v6.0: pin the verdict FUNCTION. `corrected_gates_hash` is only meaningful under the corrected
        # contract, so a shipped run leaves it None and its manifest differs from a corrected run's.
        contract=contract,
        corrected_gates_hash=(corrected_gates_hash if contract == CONTRACT_CORRECTED else None),
        # U4: pin the search-memory gates ONLY when the search memory is actually active, so a run with
        # U4 detached keeps its pre-v10.0 manifest bytes (same rule as corrected_gates_hash above).
        search_memory_gates_hash=(search_memory_gates_hash if search_memory_cfg is not None else None))

    return HypothesisLoopResult(
        specs=specs, reports=reports, cards=cards, manifest=manifest,
        n_promising=len(cards), dropped=dropped, cohort_cards=cohort_cards)


def run_cohort_only(
    *, panel: Panel, base_returns: dict[str, np.ndarray], timestamps: np.ndarray,
    cfg: FitnessConfig, evolve_kwargs: dict, cohort_pool: list[tuple[str, str, str]],
    run_id: str, crucible_version: str, gates_hash: str, proposal_ts: str,
    data_snapshot_hash: str | None, cohort_cfg: CohortConfig | None,
    cohort_mc_kwargs: dict | None, cohort_gates_hash: str | None,
    base_components: "dict[str, SleeveComponents] | None" = None,
) -> tuple[list[CohortCard], dict]:
    """Run ONLY the cohort gate, over a caller-supplied pool — no mine, no proposer, no per-candidate
    verdicts (crucible-v12.2).

    This exists because the cohort is a distinct adjudication over a SET, and a substrate can be
    "clean" in the per-candidate sense (every hypothesis already individually scored) while a cohort
    test over exactly those hypotheses has never run. `us_equity` was in precisely that state on
    2026-08-11: 8 cross-sectional pre-registrations, each tested and lost, and a mixed-pool cohort
    verdict that had never been rendered on any of them. Mining is the wrong instrument there — it
    would re-score and re-charge hypotheses whose per-candidate answer is already in the ledger.

    ``cohort_pool`` is ``[(candidate_hash, formula, candidate_type), ...]``, normally
    ``TrialLedger.pre_registered_pool(f"tick-{substrate_id}-")`` — ALL of the substrate's
    pre-registrations, never a performance-ranked subset (a Sharpe-selected pool through a gate that
    does not price the selection measured FPR 1.000; the MC null prices only the selection it
    performs itself). Costs exactly ONE LORD++ test (ADR-4), charged by the caller.

    Returns ``(cohort_cards, provenance)`` — the same pair :func:`_evaluate_cohort_gate` returns, so
    the orchestrator folds it into a tick identically to the mined path."""
    ek = {k: v for k, v in evolve_kwargs.items() if k != "candidate_type"}
    return _evaluate_cohort_gate(
        pool=cohort_pool, panel=panel, base_returns=base_returns, timestamps=timestamps, cfg=cfg,
        ek=ek, run_id=run_id, crucible_version=crucible_version, gates_hash=gates_hash,
        proposal_ts=proposal_ts, data_snapshot_hash=data_snapshot_hash, cohort_cfg=cohort_cfg,
        cohort_mc_kwargs=cohort_mc_kwargs, cohort_gates_hash=cohort_gates_hash,
        base_components=base_components)


def _evaluate_cohort_gate(
    *, pool: list[tuple[str, str, str]], panel: Panel, base_returns: dict[str, np.ndarray],
    timestamps: np.ndarray, cfg: FitnessConfig, ek: dict, run_id: str, crucible_version: str,
    gates_hash: str, proposal_ts: str, data_snapshot_hash: str | None,
    cohort_cfg: CohortConfig | None, cohort_mc_kwargs: dict | None, cohort_gates_hash: str | None,
    base_components: "dict[str, SleeveComponents] | None" = None,
) -> tuple[list[CohortCard], dict]:
    """Run the opt-in cohort gate on the tick's OVERLAY specs; return ``(cohort_cards, provenance)``
    where ``provenance`` carries the manifest's pinned cohort fields (``cohort_gates_hash`` /
    ``cohort_verdicts`` / ``cohort_card_hashes``). A no-op (returns ``([], {})``, so the manifest stays
    byte-identical to the pre-cohort path) when the gate is disabled, no cohort config is attached,
    the tick pre-registered no spec of an ADMITTED type (overlay, plus cross_sectional when
    cohort.include_cross_sectional is on — v12.1), or no cohort can form (Doc 2 §5)."""
    if not (cohort_cfg is not None and cohort_mc_kwargs and cohort_mc_kwargs.get("enabled")):
        return [], {}
    # v12.1: the pool is the tick's pre-registered OVERLAY specs, PLUS its cross-sectional specs when
    # `cohort.include_cross_sectional` is on. Pre-v12.1 this line was a hard `== "overlay"` filter,
    # which routed the two halves of the machine to the wrong gates: cross-sectional candidates carry
    # the panel's breadth (the larger per-member δ in `IR_cohort = δ·√K·hit_rate`) and went to the
    # per-candidate gate measured at ~0% power, while overlays — one scalar per day, and structurally
    # correlated with the base book they tilt — were the only thing the ONE gate with measured power
    # ever saw. See `docs/research/crucible_zero_alpha_root_cause_2026-08-09.md` and the
    # `assemble_candidate_pool` docstring. Offspring are NOT admitted (ADR-3 (B) still stands: the
    # pool must be deterministic for `pool_content_hash`); this widens the TYPE, not the search.
    admitted_types = {"overlay"}
    if cohort_cfg.include_cross_sectional:
        admitted_types.add("cross_sectional")
    formulas = {h: f for h, f, ct in pool if ct in admitted_types}
    if not formulas:
        return [], {}
    candidate_types = {h: ct for h, f, ct in pool if ct in admitted_types}
    n_xs = sum(1 for t in candidate_types.values() if t == "cross_sectional")
    pch = pool_content_hash(formulas, candidate_types)
    cgh = cohort_gates_hash or ""
    seed = derive_cohort_seed(gates_hash, cgh, pch, run_id)
    verdict = evaluate_cohort(
        panel, base_returns, timestamps, formulas, cohort_cfg, cfg,
        mc_kwargs=cohort_mc_kwargs, cost_bps=float(ek.get("cost_bps", 0.0010)),
        holdout_frac=float(ek.get("holdout_frac", 0.25)),
        holdout_embargo=int(ek.get("holdout_embargo", 21)), seed=seed,
        base_components=base_components, candidate_types=candidate_types,
        # The cross-sectional sleeve knobs must be the MINE's, not this function's defaults, or the
        # cohort would score a different stream than the per-candidate gate did (same defaults as
        # `orchestrator._incubation_params`).
        hold_horizon=int(ek.get("hold_horizon", 21)),
        ls_min_names=int(ek.get("ls_min_names", 6)))
    if verdict is None:
        return [], {}
    card = card_from_verdict(
        verdict, crucible_version=crucible_version, funnel_gates_hash=gates_hash,
        cohort_gates_hash=cgh, proposal_ts=proposal_ts, data_snapshot_hash=data_snapshot_hash)
    log.info("cohort gate: %s (p=%.4f, holdout ΔSR=%.3f, %d members of %d seen; pool = %d overlay + "
             "%d cross_sectional)", verdict.verdict, verdict.mc_p_value, verdict.holdout_delta_sr,
             verdict.n_members, verdict.n_candidates_seen, len(formulas) - n_xs, n_xs)
    # Pin BOTH the verdict string and the full-card content hash (the latter pins the MC p-value +
    # every other card field) so `crucible reproduce` re-derives them byte-identically (spec §5).
    provenance = {
        "cohort_gates_hash": cgh,
        "cohort_verdicts": {card.cohort_hash: card.verdict},
        "cohort_card_hashes": {card.cohort_hash: card.content_hash()},
    }
    return [card], provenance
