"""Crucible governance layer (spec §3 human gate, §7.1 role 3) — Crucible P5.

The moat's final wall: a survivor may become *eligible* for capital only through a human-initiated
Tier-2 deep lifecycle audit (CLAUDE.md, non-negotiable). This package turns a lockbox-CLEARED survivor
(CR-8) into an operator notification + a Tier-2 handoff packet that carries the verbatim verdict, the
binding forward-incubation evidence, the four reproducibility pins, and the EXACT audit command — and
records the handoff so it happens exactly once. Nothing here promotes or runs the audit; it only
surfaces the decision to a human.
"""
from __future__ import annotations

from .driver import card_from_dir, scan_and_handoff
from .handoff import Tier2Handoff, deep_audit_invocation, handoff_for
from .notify import FileNotifier, LogNotifier, Notifier
from .store import GovernanceStore, HandoffRecord

__all__ = [
    "FileNotifier",
    "GovernanceStore",
    "HandoffRecord",
    "LogNotifier",
    "Notifier",
    "Tier2Handoff",
    "card_from_dir",
    "deep_audit_invocation",
    "handoff_for",
    "scan_and_handoff",
]
