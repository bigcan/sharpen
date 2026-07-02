"""Split trial ledger — the anti-oracle moat (spec §6, Fable finding #4).

The full ``trial_ledger`` records every distinct scored candidate WITH its verdict, DSR, and OOS
deltas — which makes it a *fitness oracle*. If the LLM Hypothesis Author could read those score
columns it could sequentially hill-climb on a fixed, reused holdout, laundering p-hacking through
"economic priors". So access is SPLIT (CR-1):

  * ``trial_ledger``     — full, append-only, **agent-BLIND**. Scorer + orchestrator read/write.
  * ``ledger_agent_view``— a SQL VIEW exposing ONLY dedup keys (candidate_hash, candidate_type,
                           family). **No verdict, no DSR, no OOS delta, no holdout.** Plus the
                           ``killed_families()`` aggregate (the ~20 NO-GOs) so the agent does not
                           rediscover dead families. This is the concrete enforcement of CR-1.

SQLite backend per repo convention (``scripts/collect_run.py`` etc.); the DB file is git-ignored.
P0 is the schema + the split; the online-FDR ``fdr_wealth_charged`` column is nullable until P3.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Verdict vocabulary. Terminal-kill verdicts mark a candidate as dead; a family all of whose members
# are killed (and none promising) is a "killed family" the agent must not re-propose.
KILLED_VERDICTS = frozenset({"NO_GO", "NO_ADD", "GATE_FAIL"})
PROMISING_VERDICTS = frozenset({"PROMISING", "ADD_CANDIDATE"})

# Columns the agent-visible view is allowed to expose. Anything score/holdout-derived is FORBIDDEN
# here by construction — the VIEW selects exactly these, and a test asserts nothing else leaks.
_AGENT_VIEW_COLUMNS = ("candidate_hash", "candidate_type", "family")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_ledger (
    candidate_hash      TEXT PRIMARY KEY,
    crucible_version    TEXT NOT NULL,
    family              TEXT,
    candidate_type      TEXT,            -- 'cross_sectional' | 'overlay' (CR-9)
    spec_json           TEXT,
    formula             TEXT,
    economic_rationale  TEXT,            -- stored, NEVER shown to the scorer (CR-1)
    first_seen_run      TEXT,
    proposal_ts         TEXT,
    verdict             TEXT,            -- SCORE column: agent-blind
    dsr                 REAL,            -- SCORE column: agent-blind
    delta_sr_oos        REAL,            -- SCORE column: agent-blind
    marginal_hlz_t      REAL,            -- SCORE column: agent-blind
    data_snapshot_hash  TEXT,
    fdr_wealth_charged  REAL             -- online-FDR spend (P3); nullable in P0
);
CREATE INDEX IF NOT EXISTS ix_trial_family ON trial_ledger(family);
CREATE INDEX IF NOT EXISTS ix_trial_verdict ON trial_ledger(verdict);

-- Agent-VISIBLE projection: dedup keys only. No verdict/dsr/delta/holdout columns — CR-1.
CREATE VIEW IF NOT EXISTS ledger_agent_view AS
    SELECT candidate_hash, candidate_type, family FROM trial_ledger;
"""


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One scored candidate. Score fields are optional so a pre-registration row can be written
    before scoring and filled in later via :meth:`TrialLedger.record` (upsert)."""

    candidate_hash: str
    crucible_version: str
    family: str | None = None
    candidate_type: str | None = None
    spec_json: str | None = None
    formula: str | None = None
    economic_rationale: str | None = None
    first_seen_run: str | None = None
    proposal_ts: str | None = None
    verdict: str | None = None
    dsr: float | None = None
    delta_sr_oos: float | None = None
    marginal_hlz_t: float | None = None
    data_snapshot_hash: str | None = None
    fdr_wealth_charged: float | None = None


class TrialLedger:
    """Append-only trial ledger with a split agent-blind / agent-visible access model (spec §6)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- write (orchestrator / scorer side; agent-blind) ------------------------------------------
    def record(self, rec: TrialRecord) -> None:
        """Upsert a candidate by its hash. Re-recording the same hash updates score fields (a
        pre-registration row scored later) rather than duplicating — the ledger is keyed on the
        immutable content hash, so a genuine re-proposal is a no-op, not a double-count."""
        cols = [
            "candidate_hash", "crucible_version", "family", "candidate_type", "spec_json",
            "formula", "economic_rationale", "first_seen_run", "proposal_ts", "verdict", "dsr",
            "delta_sr_oos", "marginal_hlz_t", "data_snapshot_hash", "fdr_wealth_charged",
        ]
        vals = [getattr(rec, c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "candidate_hash")
        self._conn.execute(
            f"INSERT INTO trial_ledger ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(candidate_hash) DO UPDATE SET {updates}",
            vals,
        )
        self._conn.commit()

    def count(self) -> int:
        """Total distinct candidates in the ledger (the cross-run file-drawer N)."""
        return int(self._conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0])

    def update_fdr_charge(self, candidate_hash: str, fdr_wealth_charged: float) -> None:
        """Stamp the per-substrate online-FDR wealth spent on a scored trial (spec §6.1, P3). The
        candidate must already exist (the orchestrator records it during mining, then charges FDR);
        a missing hash is a no-op UPDATE, which the caller treats as a programming error upstream."""
        self._conn.execute(
            "UPDATE trial_ledger SET fdr_wealth_charged = ? WHERE candidate_hash = ?",
            (float(fdr_wealth_charged), candidate_hash))
        self._conn.commit()

    # --- read (agent-visible; CR-1) ---------------------------------------------------------------
    def is_duplicate(self, candidate_hash: str) -> bool:
        """True if this exact candidate was already scored (dedup before spending compute)."""
        row = self._conn.execute(
            "SELECT 1 FROM trial_ledger WHERE candidate_hash = ? LIMIT 1", (candidate_hash,)
        ).fetchone()
        return row is not None

    def killed_families(self) -> list[str]:
        """Families that are dead (≥1 terminal-kill verdict, none ever PROMISING) — the ~20 NO-GOs
        the agent must not rediscover (spec §6). Family NAMES only; no scores leak."""
        killed = ", ".join("?" for _ in KILLED_VERDICTS)
        promising = ", ".join("?" for _ in PROMISING_VERDICTS)
        rows = self._conn.execute(
            f"SELECT DISTINCT family FROM trial_ledger "
            f"WHERE family IS NOT NULL AND verdict IN ({killed}) "
            f"AND family NOT IN (SELECT family FROM trial_ledger WHERE verdict IN ({promising}))",
            (*KILLED_VERDICTS, *PROMISING_VERDICTS),
        ).fetchall()
        return sorted(r[0] for r in rows)

    def agent_view(self) -> dict:
        """The complete agent-VISIBLE projection (CR-1): dedup rows (candidate_hash / type / family)
        + the killed-family list. Contains NO verdict / DSR / OOS-delta / holdout field."""
        rows = self._conn.execute(
            f"SELECT {', '.join(_AGENT_VIEW_COLUMNS)} FROM ledger_agent_view"
        ).fetchall()
        return {
            "candidates": [dict(r) for r in rows],
            "candidate_hashes": sorted(r["candidate_hash"] for r in rows),
            "killed_families": self.killed_families(),
        }

    @staticmethod
    def agent_view_columns() -> tuple[str, ...]:
        """The exact columns the agent view exposes — used by the CR-1 leak test."""
        return _AGENT_VIEW_COLUMNS
