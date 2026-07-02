"""The continuous orchestrator — one nightly tick over a set of substrates (spec §8 P3 row, §3, §7).

``run_orchestrator_tick`` is ONE night: it visits each substrate, applies the ``substrate_dirty``
eligibility gate (§10.1), and — only for substrates with fresh, not-yet-scored hypotheses — runs the
UNCHANGED P2 mine + T0–T5 funnel (``run_hypothesis_loop``) within a per-tick cost budget (CR-7),
charging per-substrate online-FDR wealth per test (§6.1) and recording an auditable tick log. Run it
N nights (the CLI loops it) to satisfy the P3 exit gate: *4 unattended nights, mining only when
dirty, versioned, reproducible, within budget; a clean-panel night correctly no-ops; FDR accounted.*

**What is a "test" for the FDR (spec §6.1).** One pre-registered hypothesis = one test in the
substrate's online-FDR stream. The genetic search's evolved OFFSPRING are search internals *within* a
pre-registered test's exploration, not independent tests, so they are logged to the ledger (as P2
already does) but do NOT each charge FDR wealth. A pre-registered spec is a *discovery* (replenishing
the budget) iff its own canonical formula survives as PROMISING — the conservative reading.

**What is mined (the dedup ↔ dirty coupling).** ``fresh_specs`` = the Author's proposals after
ledger dedup — the not-yet-scored hypotheses. A substrate is mined iff it has fresh specs; a clean
panel (no new data → no new feature slots → no new formulas, and every old formula already scored)
has none, so the night no-ops and spends zero FDR wealth. ``data_changed`` is tracked and reported;
in practice new data means new series → new feature slots → new overlay formulas → fresh specs, so
the two §10.1 legs coincide. A pure row-extension of an existing series yields no new pre-registered
candidate in P3 (re-judging a locked hypothesis on forward data is the P4 lockbox, CR-8), so it
correctly does not re-spend FDR wealth here.

CR-1 is untouched: this layer schedules and accounts AROUND the funnel; it never reads a gate value
or scores a candidate. Nothing promotes past PROMISING.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..agentic.card import DiscoveryCard
from ..agentic.hypothesis import HypothesisAuthor
from ..agentic.loop import HypothesisLoopResult, run_hypothesis_loop
from ..version import CRUCIBLE_VERSION, gates_hash
from ...signals.generation.grammar import available_terminals
from .budget import TickBudget
from .burst import route_burst
from .fdr import OnlineFDR
from .substrate import OrchestratorStore, Substrate, TickRecord, substrate_dirty

log = logging.getLogger("crucible.orchestrator")


@dataclass(frozen=True, slots=True)
class SubstrateTickOutcome:
    """What happened to one substrate on one night."""

    substrate_id: str
    dirty: bool
    mined: bool
    reason: str
    snapshot_hash: str
    n_preregistered: int = 0
    n_scored: int = 0                 # candidates scored (mine hall-of-fame) — the budget spend axis
    n_promising: int = 0
    fdr_charged_total: float = 0.0
    fdr_num_tests: int = 0            # cumulative online-FDR tests on this substrate (post-tick)
    burst_target: str | None = None
    budget_breached: bool = False
    result: HypothesisLoopResult | None = None
    cards: list[DiscoveryCard] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OrchestratorTickResult:
    """The full night: one outcome per substrate + the consumed tick budget."""

    tick_ts: str
    gates_hash: str
    crucible_version: str
    outcomes: list[SubstrateTickOutcome]
    budget: TickBudget

    @property
    def n_mined(self) -> int:
        return sum(1 for o in self.outcomes if o.mined)

    @property
    def n_promising_total(self) -> int:
        return sum(o.n_promising for o in self.outcomes)


def run_orchestrator_tick(
    *,
    substrates: list[Substrate],
    store: OrchestratorStore,
    gates_path: str | Path,
    tick_ts: str,
    crucible_version: str = CRUCIBLE_VERSION,
    budget: TickBudget | None = None,
    out_dir: str | Path | None = None,
) -> OrchestratorTickResult:
    """Run one nightly tick. ``tick_ts`` is the (deterministic, caller-supplied) proposal timestamp
    for CR-2/CR-8 — pass a fixed value for a reproducible run. ``budget`` defaults to a fresh
    :class:`TickBudget`; it is shared across the substrates this tick (CR-7). If ``out_dir`` is
    given, per-substrate manifests + discovery cards are written under it."""
    ghash = gates_hash(gates_path)
    budget = budget or TickBudget()
    outcomes: list[SubstrateTickOutcome] = []

    for sub in substrates:
        prepared = sub.prepare()
        snap = prepared.snapshot_hash
        prev_snap = store.last_snapshot_hash(sub.substrate_id)
        data_changed = prev_snap != snap
        store.set_snapshot_hash(sub.substrate_id, snap)          # observed this tick (mined or not)

        # --- Stage 2 (agent, CR-1): ONE proposer call → fresh (deduped) hypotheses ----------------
        author = HypothesisAuthor(sub.proposer, sub.ledger, max_proposals=sub.max_proposals)
        terminals = available_terminals(prepared.panel)
        context = author.build_context(terminals, asset_classes=prepared.asset_classes)
        fresh_specs = author.propose(context, proposal_ts=tick_ts)
        n_fresh = len(fresh_specs)
        dirty, reason = substrate_dirty(data_changed=data_changed, n_fresh_hypotheses=n_fresh)

        # A substrate is only MINED if it has fresh specs to score; a data-only change with no new
        # candidate is dirty-but-nothing-to-test (conserves FDR wealth — the §10.1 intent).
        if not (dirty and fresh_specs):
            store.record_tick(TickRecord(
                tick_ts=tick_ts, substrate_id=sub.substrate_id, dirty=dirty, mined=False,
                reason=reason, snapshot_hash=snap))
            outcomes.append(SubstrateTickOutcome(
                substrate_id=sub.substrate_id, dirty=dirty, mined=False, reason=reason,
                snapshot_hash=snap, fdr_num_tests=_fdr_tests(store, sub)))
            log.info("substrate %s: NO-OP (%s)", sub.substrate_id, reason)
            continue

        # --- CR-7 cost budget: skip-and-report if this substrate no longer fits the tick ----------
        est_tokens = int(sub.est_tokens_per_tick)
        if not budget.can_afford(candidates=n_fresh, tokens=est_tokens):
            breach = (f"{sub.substrate_id}: needs {n_fresh} candidates / {est_tokens} tokens; "
                      f"tick budget exhausted (cand {budget.candidates_spent}/{budget.max_candidates},"
                      f" tok {budget.tokens_spent}/{budget.max_tokens})")
            budget.mark_breach(breach)
            store.record_tick(TickRecord(
                tick_ts=tick_ts, substrate_id=sub.substrate_id, dirty=True, mined=False,
                reason=f"{reason}; BUDGET BREACH — skipped", snapshot_hash=snap,
                n_preregistered=n_fresh, budget_breached=True))
            outcomes.append(SubstrateTickOutcome(
                substrate_id=sub.substrate_id, dirty=True, mined=False,
                reason=f"{reason}; budget breach", snapshot_hash=snap, n_preregistered=n_fresh,
                budget_breached=True, fdr_num_tests=_fdr_tests(store, sub)))
            log.warning("substrate %s: %s", sub.substrate_id, breach)
            continue

        budget.charge(candidates=n_fresh, tokens=est_tokens)
        burst = route_burst(est_candidates=n_fresh, is_hpo=sub.is_hpo)
        log.info("substrate %s: MINING %d fresh specs (%s) -> %s",
                 sub.substrate_id, n_fresh, reason, burst.target)

        # --- Stage 3+4: UNCHANGED mine + T0–T5 (reuse P2 loop; no 2nd proposer call) ---------------
        run_id = f"tick-{sub.substrate_id}-{tick_ts}"
        result = run_hypothesis_loop(
            panel=prepared.panel, base_returns=prepared.base_returns,
            timestamps=prepared.timestamps, cfg=sub.cfg, evolve_kwargs=sub.evolve_kwargs,
            author=author, run_id=run_id, crucible_version=crucible_version, gates_hash=ghash,
            proposal_ts=tick_ts, catalog_asset_classes=prepared.asset_classes,
            data_snapshot_hash=snap, token_cost=est_tokens, pre_proposed=fresh_specs)

        # --- online-FDR: charge one test per pre-registered spec (deterministic order) ------------
        promising_hashes = {c.candidate_hash for c in result.cards}
        fdr = store.load_fdr(sub.substrate_id, alpha=sub.fdr_alpha, w0=sub.fdr_w0,
                             alpha_floor=sub.fdr_alpha_floor)
        fdr_total = 0.0
        for pr in sorted(fresh_specs, key=lambda s: s.candidate_hash):
            charged = fdr.observe(is_discovery=(pr.candidate_hash in promising_hashes))
            sub.ledger.update_fdr_charge(pr.candidate_hash, charged)
            fdr_total += charged
        store.save_fdr(sub.substrate_id, fdr)

        n_scored = sum(len(r.hall_of_fame) for r in result.reports.values())
        store.record_tick(TickRecord(
            tick_ts=tick_ts, substrate_id=sub.substrate_id, dirty=True, mined=True, reason=reason,
            snapshot_hash=snap, n_preregistered=n_fresh, n_scored=n_scored,
            n_promising=result.n_promising, fdr_charged_total=fdr_total,
            budget_breached=False, burst_target=burst.target,
            manifest_hash=result.manifest.content_hash()))
        outcomes.append(SubstrateTickOutcome(
            substrate_id=sub.substrate_id, dirty=True, mined=True, reason=reason, snapshot_hash=snap,
            n_preregistered=n_fresh, n_scored=n_scored, n_promising=result.n_promising,
            fdr_charged_total=fdr_total, fdr_num_tests=fdr.num_tests, burst_target=burst.target,
            result=result, cards=list(result.cards)))

        if out_dir is not None:
            sub_dir = Path(out_dir) / _safe(sub.substrate_id) / _safe(tick_ts)
            for card in result.cards:
                card.write(sub_dir / "cards")
            result.manifest.write(sub_dir)

    return OrchestratorTickResult(tick_ts=tick_ts, gates_hash=ghash,
                                  crucible_version=crucible_version, outcomes=outcomes,
                                  budget=budget)


def _fdr_tests(store: OrchestratorStore, sub: Substrate) -> int:
    """Cumulative online-FDR test count for a substrate that was NOT mined this tick (for reporting)."""
    fdr: OnlineFDR = store.load_fdr(sub.substrate_id, alpha=sub.fdr_alpha, w0=sub.fdr_w0,
                                    alpha_floor=sub.fdr_alpha_floor)
    return fdr.num_tests


def _safe(s: str) -> str:
    """Filesystem-safe fragment for run/tick directory names (timestamps carry ``:``)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
