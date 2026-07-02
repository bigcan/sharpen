"""Governance store — once-only handoff bookkeeping (spec §3 human gate) — Crucible P5.

A survivor that clears incubation must be handed off to the operator EXACTLY ONCE — a nightly scan
re-visiting the same CLEARED entry must not re-notify. This tiny SQLite store records which candidates
have been handed off (and the audit command surfaced), so the driver is idempotent across ticks. Per
repo convention the DB file is git-ignored; it holds governance bookkeeping only — never a score, a
verdict, or a gate value (CR-1).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS handoffs (
    candidate_hash TEXT PRIMARY KEY,
    substrate_id   TEXT,
    handoff_ts     TEXT NOT NULL,
    forward_sharpe REAL,
    audit_command  TEXT
);
"""


@dataclass(frozen=True, slots=True)
class HandoffRecord:
    """One recorded operator handoff — the idempotency key + a light audit trail."""

    candidate_hash: str
    substrate_id: str | None
    handoff_ts: str
    forward_sharpe: float | None
    audit_command: str | None


class GovernanceStore:
    """SQLite record of survivors already handed off to the human gate (idempotency)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GovernanceStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def already_handed_off(self, candidate_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM handoffs WHERE candidate_hash = ?", (candidate_hash,)).fetchone()
        return row is not None

    def record(self, rec: HandoffRecord) -> None:
        """Insert a handoff record. Idempotent: a duplicate candidate_hash is ignored (the first
        handoff's timestamp is authoritative)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO handoffs (candidate_hash, substrate_id, handoff_ts, "
            "forward_sharpe, audit_command) VALUES (?, ?, ?, ?, ?)",
            (rec.candidate_hash, rec.substrate_id, rec.handoff_ts, rec.forward_sharpe,
             rec.audit_command))
        self._conn.commit()

    def handed_off(self) -> list[HandoffRecord]:
        rows = self._conn.execute(
            "SELECT * FROM handoffs ORDER BY candidate_hash").fetchall()
        return [HandoffRecord(candidate_hash=r["candidate_hash"], substrate_id=r["substrate_id"],
                              handoff_ts=r["handoff_ts"], forward_sharpe=r["forward_sharpe"],
                              audit_command=r["audit_command"]) for r in rows]
