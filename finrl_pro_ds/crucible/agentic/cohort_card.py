"""CohortCard — one record per EVALUATED weak-signal cohort (Phase 4, Doc 1/2).

The cohort analogue of :class:`~finrl_pro_ds.crucible.agentic.card.DiscoveryCard`: it bundles the
verbatim cohort verdict (CR-1 — the Triage Analyst may add narrative but never edits a number) a
human needs before spending a Tier-2 deep audit. A cohort is a *select-and-combine of m-of-N weak
candidates* (overlays and, since crucible-v12.1, cross-sectional rank-L/S sleeves), so its fields differ from a single-survivor card (members, the analytic ``SR*_cohort``, the
MC-null p-value, the embargoed-holdout ΔSR) — hence a separate schema rather than overloading
DiscoveryCard.

Caps at PROMISING (CLAUDE.md): a freshly-minted card is ``incubation_status="PENDING_P4"`` /
``eligible_for_human_gate=False``. The CR-8 forward-incubation lockbox is NOT wired for cohorts in v1
(ADR-7): the in-run embargoed holdout guard (Doc 2 §4) is the transfer check; forward incubation of a
cohort is a scoped follow-up. Nothing here promotes to capital — a human-initiated Tier-2 deep
lifecycle audit remains non-negotiable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ...signals.generation.cohort_eval import CohortVerdict
from .card import INCUBATION_PENDING


@dataclass(frozen=True, slots=True)
class CohortCard:
    """A cohort's discovery card. All score fields are copied verbatim from the scorer's
    :class:`CohortVerdict`; ``agent_narrative`` is the only free-text field the agent may author."""

    cohort_hash: str                 # == CohortVerdict.pool_content_hash (deterministic; the FDR key)
    members: tuple[str, ...]         # admitted member candidate ids
    n_members: int
    n_candidates_seen: int           # the deflation N (pool the cohort was selected from)
    crucible_version: str
    funnel_gates_hash: str           # the frozen crucible-v2.0 funnel hash (moat, unchanged)
    cohort_gates_hash: str           # the cohort gate file's own provenance hash
    proposal_ts: str | None = None
    data_snapshot_hash: str | None = None

    # verbatim scorer verdict (CR-1)
    verdict: str = "PROMISING"       # "PROMISING" | "LOGGED"
    sr_star_cohort: float | None = None        # analytic order-statistic benchmark (Doc 1)
    dsr_cohort_book: float | None = None       # deflated cohort-book Sharpe (Doc 1)
    passes_analytic_floor: bool | None = None  # cheap pre-filter verdict (NOT binding)
    mc_p_value: float | None = None            # BINDING selection-aware MC null p-value (Doc 2 §3)
    mc_t_obs: float | None = None              # observed within-panel ΔSR
    mc_n_reps: int | None = None
    mc_n_valid_reps: int | None = None
    mc_block_length: int | None = None
    passes_mc: bool | None = None
    holdout_delta_sr: float | None = None      # embargoed-holdout annualized ΔSR (Doc 2 §4)
    holdout_passes: bool | None = None
    mean_pairwise_corr: float | None = None    # realized diversification of the admitted set
    # v12.1 pool composition — how many of the pool / of the admitted members were CROSS-SECTIONAL
    # rank-L/S sleeves rather than base-book overlays. 0/0 == the pre-v12.1 overlay-only shape. A
    # Tier-2 reader needs this to know WHAT the MC null adjudicated, not just that it ran.
    n_pool_cross_sectional: int | None = None
    n_members_cross_sectional: int | None = None

    # lockbox (CR-8) — NOT wired for cohorts in v1 (ADR-7): a cohort card is never yet human-eligible
    incubation_status: str = INCUBATION_PENDING
    eligible_for_human_gate: bool = False

    # narrative — the ONLY field the agent authors; verdict fields above are untouchable
    agent_narrative: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["members"] = list(self.members)                 # JSON has no tuple; keep it a list
        return d

    def content_hash(self) -> str:
        """Deterministic 12-hex SHA-256 of the whole card — pins the full verdict (MC p-value,
        holdout ΔSR, members, …) into the run manifest so ``crucible reproduce`` re-derives it
        byte-identically (spec §5). A NaN field short-circuited by an early stage serializes to a
        stable ``"NaN"`` token, so the hash stays byte-deterministic across runs (Doc 2 §6)."""
        payload = json.dumps(self.to_json(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def write(self, out_dir: str | Path) -> Path:
        """Write ``cohort_<cohort_hash>.json`` into ``out_dir``; return its path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"cohort_{self.cohort_hash}.json"
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, data: dict) -> "CohortCard":
        fields = set(cls.__dataclass_fields__)            # type: ignore[attr-defined]
        kw = {k: v for k, v in data.items() if k in fields}
        if "members" in kw:
            kw["members"] = tuple(kw["members"])
        return cls(**kw)


def card_from_verdict(
    verdict: CohortVerdict,
    *,
    crucible_version: str,
    funnel_gates_hash: str,
    cohort_gates_hash: str,
    proposal_ts: str | None = None,
    data_snapshot_hash: str | None = None,
    agent_narrative: str = "",
) -> CohortCard:
    """Build a :class:`CohortCard` from a scorer :class:`CohortVerdict`, copying every score field
    VERBATIM (CR-1). ``cohort_hash`` == the verdict's ``pool_content_hash`` (the deterministic FDR-test
    key). The card is always stamped ``PENDING_P4`` / not human-eligible (ADR-7, caps at PROMISING)."""
    return CohortCard(
        cohort_hash=verdict.pool_content_hash, members=tuple(verdict.members),
        n_members=verdict.n_members, n_candidates_seen=verdict.n_candidates_seen,
        crucible_version=crucible_version, funnel_gates_hash=funnel_gates_hash,
        cohort_gates_hash=cohort_gates_hash, proposal_ts=proposal_ts,
        data_snapshot_hash=data_snapshot_hash, verdict=verdict.verdict,
        sr_star_cohort=verdict.sr_star_cohort, dsr_cohort_book=verdict.dsr_cohort_book,
        passes_analytic_floor=verdict.passes_analytic_floor, mc_p_value=verdict.mc_p_value,
        mc_t_obs=verdict.mc_t_obs, mc_n_reps=verdict.mc_n_reps,
        mc_n_valid_reps=verdict.mc_n_valid_reps, mc_block_length=verdict.mc_block_length,
        passes_mc=verdict.passes_mc, holdout_delta_sr=verdict.holdout_delta_sr,
        holdout_passes=verdict.holdout_passes, mean_pairwise_corr=verdict.mean_pairwise_corr,
        n_pool_cross_sectional=verdict.n_pool_cross_sectional,
        n_members_cross_sectional=verdict.n_members_cross_sectional,
        agent_narrative=agent_narrative)
