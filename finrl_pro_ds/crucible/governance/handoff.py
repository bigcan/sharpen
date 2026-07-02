"""Tier-2 handoff packet — the survivor → human gate bridge (spec §3 human gate, §7.1 role 3) — P5.

The lockbox (CR-8) decides *eligibility*: a survivor's card becomes human-gate-eligible ONLY once its
forward-incubation track is CLEARED. This module packages a CLEARED survivor into the artifact a human
needs to decide whether to spend a Tier-2 deep lifecycle audit — and packages the EXACT audit
invocation — WITHOUT ever running it. CLAUDE.md is non-negotiable: promotion to capital requires a
**human-initiated** Tier-2 (`.claude/workflows/deep_strategy_audit.js`); no agent may trigger it. So
:func:`handoff_for` prepares and notifies; it never promotes, and it REFUSES to build a packet for a
non-CLEARED entry (the lockbox is the sole arbiter of eligibility).

The packet copies the scorer verdict verbatim (CR-1 — the Triage Analyst adds narrative, never edits a
number) and pins the four reproducibility layers so the auditor re-derives from an immutable snapshot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agentic.card import DiscoveryCard
from ..lockbox.lockbox import STATUS_CLEARED, LockboxEntry

# The Tier-2 deep lifecycle audit entry point (CLAUDE.md; skill `deep_strategy_audit`). Recorded in the
# packet so the human runs the SAME audit the moat requires — never invoked from here.
_AUDIT_WORKFLOW = ".claude/workflows/deep_strategy_audit.js"


@dataclass(frozen=True, slots=True)
class Tier2Handoff:
    """Everything a human needs to decide on a CLEARED survivor + the exact (human-run) audit call.

    Score fields are copied verbatim from the DiscoveryCard (CR-1); the forward-incubation evidence is
    copied from the CLEARED lockbox entry (CR-8). ``audit_command`` is the invocation the OPERATOR runs
    — this packet never executes it (CLAUDE.md: Tier-2 is human-initiated)."""

    candidate_hash: str
    substrate_id: str
    formula: str
    candidate_type: str
    # reproducibility layers (spec §5) — pinned so the auditor re-derives from an immutable snapshot
    crucible_version: str
    gates_hash: str | None
    proposal_ts: str | None
    data_snapshot_hash: str | None
    # pre-registration + verbatim verdict (CR-1/CR-2)
    spec: dict = field(default_factory=dict)
    economic_rationale: str = ""
    holdout_delta_sr: float | None = None
    dsr_aug: float | None = None
    marginal_t: float | None = None
    delta_sr_median: float | None = None
    frac_paths_positive: float | None = None
    n_paths: int | None = None
    combiner_marginal_delta_sr: float | None = None
    # forward-incubation evidence (CR-8) — the binding forward gate that made this eligible
    incubation_status: str = STATUS_CLEARED
    forward_sharpe: float | None = None
    min_forward_sharpe: float | None = None
    n_forward_bars: int = 0
    min_forward_bars: int | None = None
    forward_start_ts: str | None = None
    forward_end_ts: str | None = None
    # the human-run Tier-2 handoff
    workstream: str = ""
    scope: str = ""
    audit_command: str = ""
    agent_narrative: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["_advisory"] = (
            "NOT PROMOTED. This survivor cleared forward incubation (CR-8) and is ELIGIBLE for the "
            "human Tier-2 deep lifecycle audit. Capital promotion requires a human to run the audit "
            "below; no agent may promote past PROMISING (CLAUDE.md).")
        return d

    def write(self, out_dir: str | Path) -> Path:
        """Write ``handoff_<candidate_hash>.json`` + a human-readable ``.md`` brief into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"handoff_{self.candidate_hash}.json").write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        path = out / f"handoff_{self.candidate_hash}.md"
        path.write_text(self._markdown(), encoding="utf-8")
        return path

    def _markdown(self) -> str:
        return "\n".join([
            f"# Tier-2 handoff — survivor `{self.candidate_hash}`",
            "",
            "> **NOT PROMOTED.** Cleared forward incubation (CR-8); eligible for the human Tier-2 gate.",
            "> Capital promotion requires the human-run audit below — no agent may promote (CLAUDE.md).",
            "",
            f"- substrate: `{self.substrate_id}`  ·  type: `{self.candidate_type}`",
            f"- formula: `{self.formula}`",
            f"- crucible: `{self.crucible_version}`  ·  gates: `{self.gates_hash}`  "
            f"·  data snapshot: `{self.data_snapshot_hash}`  ·  proposal_ts: `{self.proposal_ts}`",
            "",
            "## Forward-incubation evidence (the binding gate, CR-8)",
            f"- forward Sharpe **{_fmt(self.forward_sharpe)}** ≥ floor {_fmt(self.min_forward_sharpe)} "
            f"over {self.n_forward_bars} bars (horizon {self.min_forward_bars})",
            f"- window: {self.forward_start_ts} → {self.forward_end_ts}",
            "",
            "## In-sample verdict (verbatim — CR-1)",
            f"- binding holdout ΔSR: {_fmt(self.holdout_delta_sr)}  ·  DSR(aug): {_fmt(self.dsr_aug)}"
            f"  ·  marginal t: {_fmt(self.marginal_t)}",
            f"- CPCV median ΔSR: {_fmt(self.delta_sr_median)}  ·  frac paths +: "
            f"{_fmt(self.frac_paths_positive)}  ·  n_paths: {self.n_paths}",
            f"- economic rationale: {self.economic_rationale or '(evolved offspring)'}",
            "",
            "## Human action — run the Tier-2 deep lifecycle audit",
            "```",
            self.audit_command,
            "```",
            *( [f"\n_{self.agent_narrative}_"] if self.agent_narrative else [] ),
        ])


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def deep_audit_invocation(workstream: str, scope: str) -> str:
    """The exact Tier-2 audit command the OPERATOR runs (never executed here, CLAUDE.md). Mirrors the
    `deep_strategy_audit` skill contract: ``args:{workstream, scope}`` over
    ``.claude/workflows/deep_strategy_audit.js``."""
    args = json.dumps({"workstream": workstream, "scope": scope}, sort_keys=True)
    return f"/deep_strategy_audit  args:{args}   # {_AUDIT_WORKFLOW}"


def handoff_for(entry: LockboxEntry, *, workstream: str, scope: str,
                card: DiscoveryCard | None = None, agent_narrative: str = "") -> Tier2Handoff:
    """Build a :class:`Tier2Handoff` for a CLEARED lockbox entry (CR-8). RAISES if the entry is not
    CLEARED — only a forward-incubation-cleared survivor may reach the human gate. ``card`` (optional)
    supplies the verbatim in-sample verdict + pre-registered spec; without it the packet still carries
    the entry's binding forward evidence + reproducibility pins."""
    if entry.status != STATUS_CLEARED:
        raise ValueError(
            f"refusing Tier-2 handoff for {entry.candidate_hash}: status is {entry.status!r}, not "
            f"CLEARED — only a forward-incubation-cleared survivor is human-gate-eligible (CR-8)")
    v: dict = {}
    if card is not None:
        v = dict(
            spec=card.spec, economic_rationale=card.economic_rationale,
            holdout_delta_sr=card.holdout_delta_sr, dsr_aug=card.dsr_aug, marginal_t=card.marginal_t,
            delta_sr_median=card.delta_sr_median, frac_paths_positive=card.frac_paths_positive,
            n_paths=card.n_paths, combiner_marginal_delta_sr=card.combiner_marginal_delta_sr,
            agent_narrative=(agent_narrative or card.agent_narrative))
    else:
        v = dict(agent_narrative=agent_narrative)
    return Tier2Handoff(
        candidate_hash=entry.candidate_hash, substrate_id=entry.substrate_id, formula=entry.formula,
        candidate_type=entry.candidate_type, crucible_version=entry.crucible_version,
        gates_hash=entry.gates_hash, proposal_ts=entry.proposal_ts,
        data_snapshot_hash=entry.data_snapshot_hash,
        incubation_status=entry.status, forward_sharpe=entry.forward_sharpe,
        min_forward_sharpe=entry.min_forward_sharpe, n_forward_bars=entry.n_forward_bars,
        min_forward_bars=entry.min_forward_bars, forward_start_ts=entry.forward_start_ts,
        forward_end_ts=entry.forward_end_ts,
        workstream=workstream, scope=scope,
        audit_command=deep_audit_invocation(workstream, scope), **v)
