"""The incubation lockbox (CR-8, spec §6.2) — time is the final arbiter.

A candidate that clears the within-run funnel is **PROMISING**, not "discovered". It is *enrolled*
here and judged on forward data that did not exist when its hypothesis was written. This module owns
the durable enrollment record + the verdict state machine; :mod:`.incubation` owns the forward
measurement it consumes.

Why this is the keystone (all three at once, without touching the funnel/gates/human wall):
  * kills the deflation death spiral — forward data is fresh, so evidence quality does not decay with
    the number of past trials;
  * kills the ledger-oracle loop — you cannot hill-climb a holdout you must wait for;
  * kills LLM-corpus contamination — "economic priors" that are secretly memorized backtests have no
    edge on data that did not exist yet.

**Anti-peeking (the honest-test guard).** The evaluation HORIZON (``min_forward_bars``) is
pre-registered at enrollment and outcome-independent. The verdict is rendered EXACTLY ONCE, at the
first incubation pass whose forward window reaches that horizon, and is then TERMINAL (CLEARED /
REJECTED — never re-judged). A nightly pass BEFORE the horizon leaves the entry INCUBATING and
never human-eligible. So this is a fixed-horizon test, not optional stopping on a running statistic.

Nothing here promotes past PROMISING toward capital: CLEARED only makes a card *eligible* for the
human Tier-2 gate, which (CLAUDE.md) remains non-negotiable.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from ..agentic.card import DiscoveryCard
from .incubation import ForwardEvidence, IncubationCriterion

# Verdict vocabulary. INCUBATING is the only non-terminal state; a card is human-gate-eligible iff
# and only iff its entry is CLEARED. REJECTED is terminal-dead (forward evidence failed the criterion).
STATUS_INCUBATING = "INCUBATING"
STATUS_CLEARED = "CLEARED"
STATUS_REJECTED = "REJECTED"
_TERMINAL = frozenset({STATUS_CLEARED, STATUS_REJECTED})


@dataclass(frozen=True, slots=True)
class LockboxEntry:
    """One enrolled PROMISING survivor. The criterion columns are the pre-registered CR-2 lock (never
    edited after enrollment); the accrual columns are updated each incubation pass until terminal."""

    candidate_hash: str
    substrate_id: str
    formula: str
    candidate_type: str
    crucible_version: str
    gates_hash: str | None
    proposal_ts: str                 # the CR-8 lockbox boundary — only bars strictly after this count
    enrolled_tick_ts: str
    data_snapshot_hash: str | None
    # pre-registered criterion (CR-2) — pinned at enrollment
    min_forward_bars: int
    min_forward_sharpe: float
    # mutable accrual state
    status: str = STATUS_INCUBATING
    forward_sharpe: float | None = None
    n_forward_bars: int = 0
    forward_start_ts: str | None = None
    forward_end_ts: str | None = None
    last_tick_ts: str | None = None
    verdict_tick_ts: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def eligible_for_human_gate(self) -> bool:
        """A card becomes eligible for the human Tier-2 gate ONLY once its lockbox track is CLEARED
        (CR-8). INCUBATING and REJECTED are both not-eligible."""
        return self.status == STATUS_CLEARED


def advance(entry: LockboxEntry, evidence: ForwardEvidence, tick_ts: str) -> LockboxEntry:
    """Pure state transition for one incubation pass (persistence-free, so it is unit-testable).

    Terminal entries are returned UNCHANGED (anti-peeking / anti-re-judge). Otherwise the accrual
    fields are refreshed from ``evidence``; the verdict is rendered iff the forward window has reached
    the pre-registered horizon — CLEARED when forward Sharpe clears the floor, else REJECTED."""
    if entry.is_terminal:
        return entry
    reached = evidence.n_forward_bars >= entry.min_forward_bars
    sharpe = evidence.forward_sharpe
    if reached:
        cleared = sharpe is not None and sharpe == sharpe and sharpe >= entry.min_forward_sharpe
        status = STATUS_CLEARED if cleared else STATUS_REJECTED
        verdict_ts: str | None = tick_ts
    else:
        status = STATUS_INCUBATING
        verdict_ts = None
    return replace(
        entry, status=status, forward_sharpe=sharpe, n_forward_bars=evidence.n_forward_bars,
        forward_start_ts=evidence.forward_start_ts, forward_end_ts=evidence.forward_end_ts,
        last_tick_ts=tick_ts, verdict_tick_ts=verdict_ts)


def updated_card(card: DiscoveryCard, entry: LockboxEntry) -> DiscoveryCard:
    """Reflect a lockbox entry's incubation state onto its DiscoveryCard (verbatim — no re-scoring).
    ``eligible_for_human_gate`` is set ONLY from the entry's CLEARED status (CR-8)."""
    return replace(card, incubation_status=entry.status,
                   incubation_forward_sharpe=entry.forward_sharpe,
                   eligible_for_human_gate=entry.eligible_for_human_gate)


_COLS = (
    "candidate_hash", "substrate_id", "formula", "candidate_type", "crucible_version", "gates_hash",
    "proposal_ts", "enrolled_tick_ts", "data_snapshot_hash", "min_forward_bars", "min_forward_sharpe",
    "status", "forward_sharpe", "n_forward_bars", "forward_start_ts", "forward_end_ts",
    "last_tick_ts", "verdict_tick_ts",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lockbox_entries (
    candidate_hash      TEXT PRIMARY KEY,
    substrate_id        TEXT NOT NULL,
    formula             TEXT NOT NULL,
    candidate_type      TEXT NOT NULL,
    crucible_version    TEXT NOT NULL,
    gates_hash          TEXT,
    proposal_ts         TEXT NOT NULL,       -- CR-8 boundary: only bars strictly after this count
    enrolled_tick_ts    TEXT NOT NULL,
    data_snapshot_hash  TEXT,
    min_forward_bars    INTEGER NOT NULL,    -- pre-registered criterion (CR-2), never edited
    min_forward_sharpe  REAL NOT NULL,
    status              TEXT NOT NULL,       -- INCUBATING | CLEARED | REJECTED
    forward_sharpe      REAL,
    n_forward_bars      INTEGER NOT NULL DEFAULT 0,
    forward_start_ts    TEXT,
    forward_end_ts      TEXT,
    last_tick_ts        TEXT,
    verdict_tick_ts     TEXT
);
CREATE INDEX IF NOT EXISTS ix_lockbox_substrate ON lockbox_entries(substrate_id);
CREATE INDEX IF NOT EXISTS ix_lockbox_status ON lockbox_entries(status);
"""


class Lockbox:
    """SQLite-backed incubation lockbox: enrolled survivors + their forward accrual state. Per repo
    convention the DB file is git-ignored; the store is append-then-update (enroll once, incubate
    until terminal)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Lockbox":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- enrollment (idempotent) ------------------------------------------------------------------
    def enroll(self, card: DiscoveryCard, criterion: IncubationCriterion, *, substrate_id: str,
               tick_ts: str) -> LockboxEntry:
        """Enroll a PROMISING survivor, pinning ``criterion`` per CR-2. Idempotent on
        ``candidate_hash``: a re-mined duplicate does NOT reset an existing entry's accrued state or
        criterion — the first enrollment's proposal_ts / horizon are authoritative. The CR-8 boundary
        is the card's own ``proposal_ts`` when present, else the discovery tick (conservative: data
        after discovery is provably unseen)."""
        existing = self.get(card.candidate_hash)
        if existing is not None:
            return existing
        proposal_ts = card.proposal_ts or tick_ts
        entry = LockboxEntry(
            candidate_hash=card.candidate_hash, substrate_id=substrate_id, formula=card.formula,
            candidate_type=card.candidate_type, crucible_version=card.crucible_version,
            gates_hash=card.gates_hash, proposal_ts=proposal_ts, enrolled_tick_ts=tick_ts,
            data_snapshot_hash=card.data_snapshot_hash,
            min_forward_bars=int(criterion.min_forward_bars),
            min_forward_sharpe=float(criterion.min_forward_sharpe))
        self._write(entry)
        return entry

    def record_incubation(self, candidate_hash: str, evidence: ForwardEvidence,
                          tick_ts: str) -> LockboxEntry | None:
        """Apply one incubation pass to an enrolled candidate and persist the result. Returns the
        updated entry, or None if the candidate is not enrolled. A terminal entry is a no-op
        (anti-re-judge) but still returned so the caller can refresh its card."""
        entry = self.get(candidate_hash)
        if entry is None:
            return None
        advanced = advance(entry, evidence, tick_ts)
        if advanced != entry:
            self._write(advanced)
        return advanced

    # --- reads -------------------------------------------------------------------------------------
    def get(self, candidate_hash: str) -> LockboxEntry | None:
        row = self._conn.execute(
            "SELECT * FROM lockbox_entries WHERE candidate_hash = ?", (candidate_hash,)).fetchone()
        return None if row is None else self._row_to_entry(row)

    def active_entries(self, substrate_id: str | None = None) -> list[LockboxEntry]:
        """Non-terminal (still-INCUBATING) entries — the ones an incubation pass must refresh."""
        if substrate_id is None:
            rows = self._conn.execute(
                "SELECT * FROM lockbox_entries WHERE status = ? ORDER BY candidate_hash",
                (STATUS_INCUBATING,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM lockbox_entries WHERE status = ? AND substrate_id = ? "
                "ORDER BY candidate_hash", (STATUS_INCUBATING, substrate_id)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def entries(self, substrate_id: str | None = None) -> list[LockboxEntry]:
        """All entries (any status), deterministic order — the audit trail."""
        if substrate_id is None:
            rows = self._conn.execute(
                "SELECT * FROM lockbox_entries ORDER BY candidate_hash").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM lockbox_entries WHERE substrate_id = ? ORDER BY candidate_hash",
                (substrate_id,)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def eligible(self, substrate_id: str | None = None) -> list[LockboxEntry]:
        """CLEARED entries — the ONLY candidates a human Tier-2 gate may consider (CR-8)."""
        return [e for e in self.entries(substrate_id) if e.eligible_for_human_gate]

    # --- persistence helpers -----------------------------------------------------------------------
    def _write(self, entry: LockboxEntry) -> None:
        vals = [getattr(entry, c) for c in _COLS]
        placeholders = ", ".join("?" for _ in _COLS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in _COLS if c != "candidate_hash")
        self._conn.execute(
            f"INSERT INTO lockbox_entries ({', '.join(_COLS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(candidate_hash) DO UPDATE SET {updates}", vals)
        self._conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> LockboxEntry:
        return LockboxEntry(**{c: row[c] for c in _COLS})
