"""Governance driver — scan the lockbox for cleared survivors and hand them off ONCE (P5).

This is the survivor → notification → Tier-2-handoff pipeline, run AROUND the funnel (deliberately NOT
inside ``run_orchestrator_tick``, so the P3 core stays byte-identical and the CR-1 layer boundary is
clean). It finds lockbox entries that are CLEARED (forward-incubation eligible, CR-8) and not yet
handed off, builds a :class:`Tier2Handoff`, notifies the operator, and records the handoff so a later
scan does not re-notify. It NEVER runs the Tier-2 audit — that is human-initiated (CLAUDE.md).

Typical wiring: the P3 orchestrator CLI calls :func:`scan_and_handoff` after its nights loop, pointing
``card_lookup`` at the discovery cards it already wrote to disk so the packet carries the verbatim
in-sample verdict.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ..agentic.card import DiscoveryCard
from ..lockbox.lockbox import Lockbox
from .handoff import Tier2Handoff, handoff_for
from .notify import Notifier
from .store import GovernanceStore, HandoffRecord

log = logging.getLogger("crucible.governance")

CardLookup = Callable[[str], "DiscoveryCard | None"]


def card_from_dir(cards_dir: str | Path) -> CardLookup:
    """A ``card_lookup`` that reads ``card_<hash>.json`` from a directory the orchestrator wrote (or
    any parent, searched recursively). Returns None if absent — the handoff still carries the entry's
    binding forward evidence + reproducibility pins."""
    root = Path(cards_dir)

    def _lookup(candidate_hash: str) -> DiscoveryCard | None:
        matches = sorted(root.rglob(f"card_{candidate_hash}.json"))
        if not matches:
            return None
        try:
            return DiscoveryCard.from_json(json.loads(matches[0].read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            log.warning("card_from_dir: failed to read %s (%r)", matches[0], exc)
            return None

    return _lookup


def scan_and_handoff(
    lockbox: Lockbox,
    store: GovernanceStore,
    notifier: Notifier,
    *,
    handoff_ts: str,
    workstream: str,
    scope: str,
    substrate_id: str | None = None,
    card_lookup: CardLookup | None = None,
    out_dir: str | Path | None = None,
    agent_narrative: str = "",
) -> list[Tier2Handoff]:
    """Hand off every newly-CLEARED survivor exactly once. Returns the handoffs created THIS scan
    (already-recorded candidates are skipped). ``handoff_ts`` is caller-supplied for reproducibility
    (repo convention: no wall-clock in deterministic paths). If ``out_dir`` is given, each packet's
    JSON + markdown brief is written there."""
    created: list[Tier2Handoff] = []
    for entry in lockbox.eligible(substrate_id):               # CLEARED entries only (CR-8)
        if store.already_handed_off(entry.candidate_hash):
            continue
        card = card_lookup(entry.candidate_hash) if card_lookup is not None else None
        handoff = handoff_for(entry, workstream=workstream, scope=scope, card=card,
                              agent_narrative=agent_narrative)
        notifier.notify(handoff)
        if out_dir is not None:
            handoff.write(out_dir)
        store.record(HandoffRecord(
            candidate_hash=entry.candidate_hash, substrate_id=entry.substrate_id,
            handoff_ts=handoff_ts, forward_sharpe=entry.forward_sharpe,
            audit_command=handoff.audit_command))
        log.info("handed off survivor %s for human Tier-2 (%s)", entry.candidate_hash, workstream)
        created.append(handoff)
    return created
