"""Operator notification seam for a cleared survivor (spec §3 human gate) — Crucible P5.

When a survivor clears forward incubation, the operator must be *told* (so they can choose to spend a
Tier-2 audit). Notification is an injectable seam — mirroring the ``Proposer`` / connector-``transport``
pattern — so the governance driver is hermetic and testable, and a real backend is a drop-in.

The shipped defaults are offline and side-effect-local: :class:`FileNotifier` drops a
``NOTIFY_<hash>.md`` the operator (or a watcher) picks up, and :class:`LogNotifier` logs a structured
line. The live Watchdog Telegram bot (``@<TELEGRAM_BOT>``) is a documented drop-in — a ``TelegramNotifier``
implementing this Protocol — deliberately NOT wired here so no message is sent from a headless/CI run.

A notifier NEVER promotes and NEVER runs the Tier-2 audit (CLAUDE.md); it only surfaces the handoff.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from .handoff import Tier2Handoff

log = logging.getLogger("crucible.governance")


@runtime_checkable
class Notifier(Protocol):
    """Any object that can surface a :class:`Tier2Handoff` to the operator. Must not promote/audit."""

    def notify(self, handoff: Tier2Handoff) -> None: ...


class LogNotifier:
    """Emits a structured log line per cleared survivor — the zero-config default."""

    def notify(self, handoff: Tier2Handoff) -> None:
        log.warning(
            "CRUCIBLE SURVIVOR CLEARED INCUBATION — human Tier-2 eligible: candidate=%s substrate=%s "
            "fwd_sharpe=%s over %d bars. Run: %s",
            handoff.candidate_hash, handoff.substrate_id, handoff.forward_sharpe,
            handoff.n_forward_bars, handoff.audit_command)


class FileNotifier:
    """Writes a ``NOTIFY_<candidate_hash>.md`` into ``out_dir`` — an offline, pickup-able signal."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def notify(self, handoff: Tier2Handoff) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"NOTIFY_{handoff.candidate_hash}.md"
        path.write_text(
            f"# Survivor cleared incubation — Tier-2 eligible\n\n"
            f"- candidate: `{handoff.candidate_hash}`  ·  substrate: `{handoff.substrate_id}`\n"
            f"- forward Sharpe {handoff.forward_sharpe} over {handoff.n_forward_bars} bars\n\n"
            f"Run the human Tier-2 audit (NOT automated):\n\n```\n{handoff.audit_command}\n```\n",
            encoding="utf-8")
        log.info("wrote operator notification %s", path)
