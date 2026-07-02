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

import logging
from dataclasses import dataclass, field

import numpy as np

from ...signals.features import Panel
from ...signals.generation.evolve import Candidate, GenerationReport, evolve
from ...signals.generation.fitness import FitnessConfig
from ...signals.generation.grammar import available_terminals
from ..ledger import TrialRecord
from ..manifest import RunManifest
from .card import DiscoveryCard
from .hypothesis import HypothesisAuthor, PreRegisteredSpec, candidate_hash

log = logging.getLogger("crucible.loop")

# Non-killing verdict for a mined genome that was scored but not selected — it is a non-survivor,
# NOT a falsified named family, so it must not enter killed_families (which gates future proposals).
_LOGGED = "LOGGED"
_PROMISING = "PROMISING"


@dataclass(frozen=True, slots=True)
class HypothesisLoopResult:
    """Outcome of one manual loop pass."""

    specs: list[PreRegisteredSpec]
    reports: dict[str, GenerationReport]          # candidate_type -> GenerationReport
    cards: list[DiscoveryCard]
    manifest: RunManifest
    n_promising: int = 0
    dropped: int = 0                              # proposals rejected pre-compute (dedup/killed/parse)
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
        combiner_marginal_delta_sr=(float(holdout_delta) if holdout_delta is not None else None),
    )


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
) -> HypothesisLoopResult:
    """Run one manual pass. ``evolve_kwargs`` is the runner block from ``load_generation_config``
    (rng_seed/pop_size/… — WITHOUT ``candidate_type``, which the loop sets per group)."""
    ledger = author.ledger
    n_before = ledger.count()

    # --- Stage 2: HYPOTHESIZE (agent, CR-1) -------------------------------------------------------
    terminals = available_terminals(panel)
    context = author.build_context(terminals, asset_classes=catalog_asset_classes)
    specs = author.propose(context, proposal_ts=proposal_ts)
    dropped = author.last_proposal_stats["dropped"]              # recorded by propose(), no re-call
    author.preregister(specs, run_id=run_id, crucible_version=crucible_version,
                       data_snapshot_hash=data_snapshot_hash)
    prereg_by_hash = {pr.candidate_hash: pr for pr in specs}
    log.info("preregistered %d specs (run %s, ts %s)", len(specs), run_id, proposal_ts)

    # --- Stage 3+4: MINE + DEFLATE, per candidate_type (evolve unchanged) -------------------------
    ek = {k: v for k, v in evolve_kwargs.items() if k != "candidate_type"}
    reports: dict[str, GenerationReport] = {}
    cards: list[DiscoveryCard] = []
    verdicts: dict[str, str] = {}
    for ct in ("cross_sectional", "overlay"):
        seeds = [pr.formula for pr in specs if pr.spec.candidate_type == ct]
        if not seeds:
            continue
        log.info("mining %d %s seeds", len(seeds), ct)
        report = evolve(seeds, panel, base_returns, timestamps, cfg, candidate_type=ct, **ek)
        reports[ct] = report
        promising_hashes = {candidate_hash(c.formula) for c in report.promising}
        # Record every surfaced genome to the ledger (file-drawer). Offspring carry family=None so a
        # non-survivor never pollutes killed_families; the pre-registered seeds keep their family.
        for c in report.hall_of_fame:
            chash = candidate_hash(c.formula)
            verdict = _PROMISING if chash in promising_hashes else _LOGGED
            verdicts[chash] = verdict
            pr = prereg_by_hash.get(chash)
            res = c.result
            ledger.record(TrialRecord(
                candidate_hash=chash, crucible_version=crucible_version,
                family=(pr.spec.family if pr else None), candidate_type=ct,
                formula=c.formula,
                economic_rationale=(pr.economic_rationale if pr else None),
                first_seen_run=run_id, proposal_ts=(pr.proposal_ts if pr else proposal_ts),
                verdict=verdict,
                dsr=(None if res is None else float(res.dsr_aug)),
                delta_sr_oos=(None if res is None else float(res.delta_sr_oos)),
                marginal_hlz_t=(None if res is None else float(res.marginal_t)),
                data_snapshot_hash=data_snapshot_hash))
        for c in report.promising:
            cards.append(_card_for(c, ct, report, prereg_by_hash, crucible_version=crucible_version,
                                   gates_hash=gates_hash, data_snapshot_hash=data_snapshot_hash))

    n_after = ledger.count()
    manifest = RunManifest(
        run_id=run_id, crucible_version=crucible_version, gates_hash=gates_hash,
        proposal_ts=proposal_ts, rng_seeds={"generation": int(ek.get("rng_seed", 7))},
        file_drawer_N_before=n_before, file_drawer_N_after=n_after,
        data_snapshot_hash=data_snapshot_hash, agent_model_id=author.proposer.model_id,
        token_cost=token_cost, verdicts=verdicts)

    return HypothesisLoopResult(
        specs=specs, reports=reports, cards=cards, manifest=manifest,
        n_promising=len(cards), dropped=dropped)
