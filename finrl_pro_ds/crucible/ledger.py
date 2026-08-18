"""Split trial ledger — the anti-oracle moat (spec §6, Fable finding #4).

The full ``trial_ledger`` records every distinct scored candidate WITH its verdict, DSR, and OOS
deltas — which makes it a *fitness oracle*. If the LLM Hypothesis Author could read those score
columns it could sequentially hill-climb on a fixed, reused holdout, laundering p-hacking through
"economic priors". So access is SPLIT (CR-1):

  * ``trial_ledger``     — full, append-only, **agent-BLIND**. Scorer + orchestrator read/write.
  * ``ledger_agent_view``— a SQL VIEW exposing ONLY dedup keys (candidate_hash, semantic_hash,
                           candidate_type, family). **No verdict, no DSR, no OOS delta, no holdout.**
                           Plus the ``killed_families()`` aggregate so the agent does not rediscover
                           dead families. This is the concrete enforcement of CR-1.

**U4 (``crucible-v10.0``) — the moat is no longer inert.** Until U4 no code path wrote a killing
verdict, so ``killed_families()`` returned ``[]`` unconditionally and the proposer prompt always read
"(none)" (audit RC-5). A rejection now carries a ``rejection_class`` — ``DECISIVE`` (the test could
resolve the smallest edge the contract would accept and did not) or ``UNDERPOWERED`` (it could not, so
the negative is uninformative) — computed by :mod:`crucible.search_memory` from the substrate's power
stamp. ``killed_families()`` counts DECISIVE rejections alongside the legacy ``KILLED_VERDICTS``, and
``readmissible()`` hands the orchestrator back the parked ones once the substrate has gained real power.
The three columns this needs (``semantic_hash``, ``rejection_class``, ``implied_mde_at_test``) are all
NULLABLE and back-filled by migration, and the ``verdict`` vocabulary is UNCHANGED — so every historical
row, manifest byte and reproduce verdict is preserved (CRU-1).

SQLite backend per repo convention (``scripts/collect_run.py`` etc.); the DB file is git-ignored.
P0 is the schema + the split; the online-FDR ``fdr_wealth_charged`` column is nullable until P3.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .search_memory import (
    REJECTION_DECISIVE,
    REJECTION_UNDERPOWERED,
    is_readmissible,
    semantic_hash,
)

if TYPE_CHECKING:
    from .search_memory import SearchMemoryConfig

log = logging.getLogger("crucible.ledger")

# Verdict vocabulary. Terminal-kill verdicts mark a candidate as dead; a family all of whose members
# are killed (and none promising) is a "killed family" the agent must not re-propose.
KILLED_VERDICTS = frozenset({"NO_GO", "NO_ADD", "GATE_FAIL"})
PROMISING_VERDICTS = frozenset({"PROMISING", "ADD_CANDIDATE"})

# Columns the agent-visible view is allowed to expose. Anything score/holdout-derived is FORBIDDEN
# here by construction — the VIEW selects exactly these, and a test asserts nothing else leaks.
# ``semantic_hash`` (U4) is a DEDUP KEY like ``candidate_hash`` — a hash of the commutative-canonical
# AST. It is computed from the formula TEXT alone and carries no score/verdict information, so adding it
# does not widen the anti-oracle moat (CRU-2); ``rejection_class`` / ``implied_mde_at_test`` are
# score-adjacent and stay OUT of the view.
_AGENT_VIEW_COLUMNS = ("candidate_hash", "semantic_hash", "candidate_type", "family")

# --- upsert monotonicity policy (C7-04) -----------------------------------------------------------
# Re-recording a candidate by hash must NEVER destroy provenance or downgrade a settled verdict. The
# ledger is "append-only" in spirit, but two legitimate re-records used to CLOBBER it: (a) the loop
# re-records a scored pre-registered seed WITHOUT its spec_json (erasing the CR-2 lock), and (b) an
# evolved offspring can re-derive a formula that was already PROMISING (downgrading it to LOGGED and
# nulling its provenance). The blanket ``SET col=excluded.col`` did both. The policy below fixes it:
#   * a SETTLED verdict (PROMISING/killed) and its score columns are FROZEN against any later re-record;
#   * CR-2 / provenance columns keep their FIRST non-null value (an immutable pre-registration lock);
#   * every other column fills forward — a NULL in the incoming record never clobbers a stored value.
_SETTLED_VERDICTS = PROMISING_VERDICTS | KILLED_VERDICTS
_UPSERT_KEEP_FIRST = frozenset({"first_seen_run", "proposal_ts", "spec_json", "economic_rationale"})
# U4: ``rejection_class`` / ``implied_mde_at_test`` are SCORE columns — they are derived from the
# holdout decision plus the substrate's power stamp, so they freeze with the verdict. Freezing them is
# what makes a DECISIVE kill terminal and keeps the MDE-at-test honest (a later, deeper tick must not
# silently rewrite the power a past test actually had).
_UPSERT_SCORE = frozenset({"verdict", "dsr", "delta_sr_oos", "marginal_hlz_t", "data_snapshot_hash",
                           "implied_mde_at_test"})
# ``rejection_class`` needs its OWN rule (not the generic score one): a DECISIVE kill is TERMINAL, so it
# must survive any later re-record even though the carrying verdict (``LOGGED``) is not in
# ``_SETTLED_VERDICTS``. Without this a re-scored genome that came back UNDERPOWERED on a different
# substrate would resurrect a family the funnel had already decisively falsified. ``implied_mde_at_test``
# deliberately does NOT get this treatment — it tracks the LATEST test's power, which is what
# power-aware re-admission compares against.
_UPSERT_REJECTION_SQL = (
    "rejection_class=CASE "
    f"WHEN trial_ledger.rejection_class = '{REJECTION_DECISIVE}' THEN '{REJECTION_DECISIVE}' "
    "ELSE COALESCE(excluded.rejection_class, trial_ledger.rejection_class) END")
# ``fdr_wealth_charged`` is owned by :meth:`TrialLedger.update_fdr_charge`; a record() carrying None
# must not wipe an already-charged value — it fills forward like provenance.

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
    delta_sr_oos        REAL,            -- SCORE column, agent-blind. TRAIN-split CPCV mean ΔSR at the
                                         -- running gen_n_eff (the pre-filter result). Despite the `_oos`
                                         -- name (kept for schema back-compat), this is NOT out-of-sample:
                                         -- the certified holdout re-score is a separate pass that rarely/
                                         -- never runs, so this column is a train-split value (S553-cont-131).
    marginal_hlz_t      REAL,            -- SCORE column: agent-blind
    data_snapshot_hash  TEXT,
    fdr_wealth_charged  REAL,            -- online-FDR spend (P3); nullable in P0
    semantic_hash       TEXT,            -- U4 DEDUP KEY: hash of the commutative-canonical AST
    rejection_class     TEXT,            -- U4 SCORE column: 'DECISIVE' | 'UNDERPOWERED' | NULL
    implied_mde_at_test REAL             -- U4 SCORE column: substrate MDE when this test ran
);
CREATE INDEX IF NOT EXISTS ix_trial_family ON trial_ledger(family);
CREATE INDEX IF NOT EXISTS ix_trial_verdict ON trial_ledger(verdict);
"""

# Indexes on the U4 columns are created AFTER the ALTER TABLE migration, not inside _SCHEMA: on a pre-U4
# database the columns do not exist yet when _SCHEMA runs, and `CREATE INDEX IF NOT EXISTS` on a missing
# column is a hard OperationalError — which would make every historical ledger un-openable.
_U4_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_trial_semantic ON trial_ledger(semantic_hash);
CREATE INDEX IF NOT EXISTS ix_trial_rejection ON trial_ledger(rejection_class);
"""

# The agent view is created SEPARATELY from _SCHEMA and DROPPED first: `CREATE VIEW IF NOT EXISTS` on an
# existing DB would silently keep the OLD column list, so a migrated ledger would expose the pre-U4
# projection while the code believed otherwise. A view carries no data, so dropping it is free.
# `WHERE verdict IS NOT NULL` — the dedup keys must describe candidates that were actually SCORED.
# A pre-registration writes its row BEFORE the holdout runs (the CR-2 lock on proposal_ts) and the
# verdict is upserted only at scoring, so without this filter an unscored pre-registration becomes a
# permanent self-block: author pre-registers -> row written -> the tick NO-OPs before scoring -> every
# later tick reads the hash out of this view and drops the proposal as a duplicate -> NO-OP -> forever.
# Measured (S553-cont-152): eight WQ101 seeds pre-registered on a cross_asset tick on 2026-07-02 sat
# with verdict/rejection_class/implied_mde_at_test all NULL and blocked themselves on EVERY substrate
# for five weeks; the first adequately-powered substrate this project has built accepted 0/8 proposals
# and mined nothing, which in the logs is almost indistinguishable from "found no alpha".
#
# CRU-2 IS NOT WIDENED: the projected COLUMNS are unchanged and no verdict, DSR, OOS delta or holdout
# value crosses the boundary. This REMOVES rows from the agent's view, it does not add information.
_AGENT_VIEW_SQL = f"""
DROP VIEW IF EXISTS ledger_agent_view;
CREATE VIEW ledger_agent_view AS
    SELECT {', '.join(_AGENT_VIEW_COLUMNS)} FROM trial_ledger WHERE verdict IS NOT NULL;
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
    delta_sr_oos: float | None = None    # TRAIN-split CPCV mean ΔSR (see schema note); not out-of-sample
    marginal_hlz_t: float | None = None
    data_snapshot_hash: str | None = None
    fdr_wealth_charged: float | None = None
    # U4. ``semantic_hash`` defaults to None so every existing construction site keeps working; the
    # ledger derives it from ``formula`` when it is absent, so a caller cannot accidentally write a row
    # that is invisible to semantic dedup.
    semantic_hash: str | None = None
    rejection_class: str | None = None
    implied_mde_at_test: float | None = None


class TrialLedger:
    """Append-only trial ledger with a split agent-blind / agent-visible access model (spec §6)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # U4 migration: CREATE TABLE IF NOT EXISTS never alters an existing table, so a ledger written by
        # a pre-U4 build needs the three new columns added (same pattern as OrchestratorStore's
        # ``_ensure_columns``). All three are NULLABLE, so historical rows migrate without back-fill and
        # keep their exact verdicts. ``semantic_hash`` IS back-filled for rows that carry a formula —
        # without it, every historical genome would be invisible to semantic dedup and the search would
        # cheerfully re-derive all 403 of them.
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(trial_ledger)")}
        for col, decl in (("semantic_hash", "TEXT"), ("rejection_class", "TEXT"),
                          ("implied_mde_at_test", "REAL")):
            if col not in have:
                self._conn.execute(f"ALTER TABLE trial_ledger ADD COLUMN {col} {decl}")
        self._conn.executescript(_U4_INDEX_SQL)
        self._conn.executescript(_AGENT_VIEW_SQL)
        self._conn.commit()
        if "semantic_hash" not in have:
            self._backfill_semantic_hashes()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- write (orchestrator / scorer side; agent-blind) ------------------------------------------
    def record(self, rec: TrialRecord) -> None:
        """Monotone upsert by candidate_hash (C7-04). Re-recording the same hash NEVER destroys
        provenance or downgrades a settled verdict: a PROMISING/killed verdict and its score columns
        are frozen; CR-2 columns (spec_json / proposal_ts / first_seen_run / economic_rationale) keep
        their first non-null value; all other columns fill forward (a NULL in the incoming record can
        never clobber a stored value). A genuine re-proposal of a scored candidate is therefore a
        no-op, not a double-count and not a downgrade (see the ``_UPSERT_*`` policy above)."""
        cols = [
            "candidate_hash", "crucible_version", "family", "candidate_type", "spec_json",
            "formula", "economic_rationale", "first_seen_run", "proposal_ts", "verdict", "dsr",
            "delta_sr_oos", "marginal_hlz_t", "data_snapshot_hash", "fdr_wealth_charged",
            "semantic_hash", "rejection_class", "implied_mde_at_test",
        ]
        vals = [getattr(rec, c) for c in cols]
        if rec.semantic_hash is None and rec.formula:
            # Derive the U4 dedup key here rather than trusting every call site to remember it. An
            # unparseable formula (should not reach the ledger) leaves it NULL rather than raising —
            # recording the trial matters more than indexing it.
            try:
                vals[cols.index("semantic_hash")] = semantic_hash(rec.formula)
            except Exception:                                 # noqa: BLE001
                pass
        placeholders = ", ".join("?" for _ in cols)
        settled = "(" + ", ".join(f"'{v}'" for v in sorted(_SETTLED_VERDICTS)) + ")"
        set_parts: list[str] = []
        for c in cols:
            if c == "candidate_hash":
                continue
            if c == "rejection_class":                        # U4: DECISIVE is terminal (see above)
                set_parts.append(_UPSERT_REJECTION_SQL)
            elif c in _UPSERT_KEEP_FIRST:                     # CR-2 / provenance: first non-null wins
                set_parts.append(f"{c}=COALESCE(trial_ledger.{c}, excluded.{c})")
            elif c in _UPSERT_SCORE:                          # frozen once settled, else fill-forward
                set_parts.append(
                    f"{c}=CASE WHEN trial_ledger.verdict IN {settled} THEN trial_ledger.{c} "
                    f"ELSE COALESCE(excluded.{c}, trial_ledger.{c}) END")
            else:                                             # provenance / fdr charge: never nulled
                set_parts.append(f"{c}=COALESCE(excluded.{c}, trial_ledger.{c})")
        self._conn.execute(
            f"INSERT INTO trial_ledger ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(candidate_hash) DO UPDATE SET {', '.join(set_parts)}",
            vals,
        )
        self._conn.commit()

    def _backfill_semantic_hashes(self) -> None:
        """One-shot U4 migration: compute ``semantic_hash`` for pre-U4 rows that carry a formula. Rows
        whose formula does not parse are left NULL and counted in the log — they simply stay outside
        semantic dedup (exact-hash dedup still covers them)."""
        rows = self._conn.execute(
            "SELECT candidate_hash, formula FROM trial_ledger "
            "WHERE semantic_hash IS NULL AND formula IS NOT NULL").fetchall()
        done = bad = 0
        for r in rows:
            try:
                sh = semantic_hash(r["formula"])
            except Exception:                                 # noqa: BLE001 — unparseable legacy row
                bad += 1
                continue
            self._conn.execute("UPDATE trial_ledger SET semantic_hash = ? WHERE candidate_hash = ?",
                               (sh, r["candidate_hash"]))
            done += 1
        self._conn.commit()
        if rows:
            log.info("U4 ledger migration: back-filled semantic_hash for %d/%d rows (%d unparseable)",
                     done, len(rows), bad)

    def count(self) -> int:
        """Total distinct candidates in the ledger (the cross-run file-drawer N)."""
        return int(self._conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0])

    def pre_registered_pool(self, run_prefix: str) -> list[tuple[str, str, str]]:
        """Every PRE-REGISTERED candidate first seen on runs starting with ``run_prefix``, as
        ``[(candidate_hash, formula, candidate_type), ...]`` sorted by hash (deterministic — the
        cohort's ``pool_content_hash`` and MC seed depend on it).

        This is the COHORT's ledger-sourced pool (crucible-v12.2). The cohort previously drew only
        the CURRENT tick's fresh specs, which coupled its ``K`` to one night's proposals (the two
        cohort verdicts ever rendered ran at K=10 and K=6) and made it unable to adjudicate a
        substrate whose hypotheses were all already scored — the state ``us_equity`` reached on
        2026-08-11, where the dirty gate reported "no fresh hypotheses" about a cohort test that had
        never run. Since ``IR_cohort = δ·√K·hit_rate``, K is a power term, not bookkeeping.

        Scoping is by ``first_seen_run`` prefix because the ledger has no ``substrate_id`` column and
        run ids are formed ``tick-<substrate_id>-<ts>`` by the orchestrator; the caller passes
        ``f"tick-{substrate_id}-"``. Rows are restricted to ``spec_json IS NOT NULL`` — that is what
        distinguishes a written-down PRE-REGISTRATION from an evolved offspring, and admitting
        offspring would make the pool non-deterministic and break the reproduce contract (ADR-3 (B)).

        NOT a selection: this returns ALL pre-registrations for the substrate, never a
        performance-ranked subset. Routing a Sharpe-SELECTED pool into a gate that does not price the
        selection measured FPR 1.000 (2026-08-09); the MC null prices the selection it performs
        ITSELF, on a pool it is handed whole. Verdict columns are not read here."""
        rows = self._conn.execute(
            "SELECT candidate_hash, formula, candidate_type FROM trial_ledger "
            "WHERE first_seen_run LIKE ? AND spec_json IS NOT NULL AND formula IS NOT NULL "
            "ORDER BY candidate_hash",
            (f"{run_prefix}%",)).fetchall()
        return [(str(a), str(b), str(c)) for a, b, c in rows]

    def update_fdr_charge(self, candidate_hash: str, fdr_wealth_charged: float) -> None:
        """ACCUMULATE the per-substrate online-FDR wealth spent on a scored trial (spec §6.1, P3). The
        candidate MUST already exist (the orchestrator records it during mining, then charges FDR).
        A missing hash used to be a silent no-op UPDATE — which lets the FDR audit trail diverge
        undetected (C7-08) — so it now RAISES: a charge with no ledger row is a programming error
        upstream, not something to swallow.

        **Accumulates, does not overwrite (U4 follow-up).** This was a blind ``SET``, which was correct
        while every candidate was tested exactly once. U4's power-aware re-admission breaks that
        assumption: a re-admitted candidate is charged a SECOND LORD++ level, and a blind SET would erase
        the first — the per-candidate trail would under-count the wealth actually spent on that
        hypothesis, which is the same class of silent divergence C7-08 exists to prevent, arriving through
        a different door. ``COALESCE(...,0) + ?`` is byte-identical for a first charge (NULL → 0 + x = x),
        so no historical row or manifest changes."""
        cur = self._conn.execute(
            "UPDATE trial_ledger "
            "SET fdr_wealth_charged = COALESCE(fdr_wealth_charged, 0.0) + ? WHERE candidate_hash = ?",
            (float(fdr_wealth_charged), candidate_hash))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(
                f"update_fdr_charge: no trial_ledger row for candidate_hash {candidate_hash!r} — a "
                "candidate must be recorded before its online-FDR wealth is charged")

    # --- read (agent-visible; CR-1) ---------------------------------------------------------------
    def is_duplicate(self, candidate_hash: str) -> bool:
        """True if this exact candidate was already SCORED (dedup before spending compute).

        ``verdict IS NOT NULL`` is the scored test, and it is load-bearing rather than cosmetic. A
        pre-registration writes its ledger row BEFORE the holdout runs (that is the CR-2 lock on
        ``proposal_ts``), and the verdict is upserted only when the candidate is actually scored. The
        query used to match on ``candidate_hash`` alone, which made an unscored pre-registration
        permanently self-blocking:

            author pre-registers -> row written (verdict NULL) -> tick NO-OPs before scoring
            -> next tick sees the row and drops the proposal as a duplicate -> NO-OP -> forever.

        Measured consequence (S553-cont-152): eight WorldQuant-101 seed specs pre-registered on a
        `cross_asset` tick on 2026-07-02 under crucible-v2.6 carried verdict/rejection_class/
        implied_mde_at_test all NULL, and blocked themselves on EVERY substrate for five weeks. The
        first adequately-powered substrate this project has ever built accepted 0/8 proposals and
        mined nothing at all as a direct result — the failure looks exactly like "found no alpha".

        Nothing is laundered by re-testing them: no LORD++ wealth was ever charged for a trial that
        did not run, and the row keeps its ORIGINAL ``proposal_ts``, so re-admission is the same
        pre-registered hypothesis finally being tested rather than a fresh one.

        CRU-2: still returns one bool about ledger membership and leaks no verdict, DSR, OOS delta or
        holdout value. This narrows which rows count as duplicates; it does not widen the agent view.
        """
        row = self._conn.execute(
            "SELECT 1 FROM trial_ledger WHERE candidate_hash = ? AND verdict IS NOT NULL LIMIT 1",
            (candidate_hash,)).fetchone()
        return row is not None

    def is_semantic_duplicate(self, formula: str) -> bool:
        """U4: True if a genome SEMANTICALLY equal to ``formula`` (same commutative-canonical AST) was
        already scored, even if its exact canonical string differs. ``is_duplicate`` remains the exact
        check; this is the wider one."""
        try:
            sh = semantic_hash(formula)
        except Exception:                                     # noqa: BLE001 — unparseable ⇒ not a dup
            return False
        return self._conn.execute(
            "SELECT 1 FROM trial_ledger WHERE semantic_hash = ? AND verdict IS NOT NULL LIMIT 1",
            (sh,)).fetchone() is not None

    def killed_families(self) -> list[str]:
        """Families that are dead — the NO-GOs the agent must not rediscover (spec §6). Family NAMES
        only; no scores leak.

        A family is dead when it has ≥1 terminal rejection and NEVER a PROMISING member. Terminal means
        either a legacy ``KILLED_VERDICTS`` verdict (kept for compatibility; the funnel never wrote one)
        or, since U4, a ``rejection_class = 'DECISIVE'`` row — a rejection the substrate had the power to
        make meaningful (:mod:`crucible.search_memory`). An ``UNDERPOWERED`` rejection is deliberately
        NOT terminal: at Crucible's measured MDE that is every rejection, and killing families off
        underpowered tests would manufacture NO-GOs out of low power."""
        killed = ", ".join("?" for _ in KILLED_VERDICTS)
        promising = ", ".join("?" for _ in PROMISING_VERDICTS)
        rows = self._conn.execute(
            f"SELECT DISTINCT family FROM trial_ledger "
            f"WHERE family IS NOT NULL AND (verdict IN ({killed}) OR rejection_class = ?) "
            f"AND family NOT IN (SELECT family FROM trial_ledger WHERE verdict IN ({promising}))",
            (*KILLED_VERDICTS, REJECTION_DECISIVE, *PROMISING_VERDICTS),
        ).fetchall()
        return sorted(r[0] for r in rows)

    def readmissible(self, *, current_mde: float, cfg: "SearchMemoryConfig",
                     limit: int = 32) -> list[dict]:
        """U4 power-aware re-admission: parked (``UNDERPOWERED``) candidates whose original test ran at
        a materially WORSE MDE than the substrate now has, so re-testing them can produce information
        the first test could not (:func:`search_memory.is_readmissible`).

        Returned rows carry ``formula`` / ``family`` / ``candidate_type`` / ``spec_json`` — enough for the
        ORCHESTRATOR to re-inject them as seeds. This is a SCORER-side read (the orchestrator is
        agent-blind by design), deliberately NOT surfaced through ``agent_view``: exposing "this exact
        candidate is eligible again" would tell the proposer, at candidate granularity, that the genome
        was not decisively killed — a score-derived bit, and precisely the widening CRU-2 forbids.

        Ordered oldest-MDE-first (the most badly-underpowered original tests, i.e. the ones with the most
        to gain) then by hash for determinism, and capped at ``limit`` so re-admissions cannot crowd the
        per-tick candidate budget (CR-7)."""
        rows = self._conn.execute(
            "SELECT candidate_hash, semantic_hash, family, candidate_type, formula, spec_json, "
            "       economic_rationale, proposal_ts, rejection_class, implied_mde_at_test "
            "FROM trial_ledger WHERE rejection_class = ? AND formula IS NOT NULL "
            "ORDER BY implied_mde_at_test DESC, candidate_hash ASC",
            (REJECTION_UNDERPOWERED,)).fetchall()
        cap = max(0, int(limit))
        out: list[dict] = []
        for r in rows:
            if len(out) >= cap:                               # checked BEFORE appending, so limit=0 → []
                break
            if is_readmissible(rejection_class=r["rejection_class"],
                               mde_at_test=r["implied_mde_at_test"],
                               current_mde=current_mde, cfg=cfg):
                out.append(dict(r))
        return out

    def agent_view(self) -> dict:
        """The complete agent-VISIBLE projection (CR-1): dedup rows (candidate_hash / semantic_hash /
        type / family) + the killed-family list. Contains NO verdict / DSR / OOS-delta / holdout /
        rejection-class field."""
        rows = self._conn.execute(
            f"SELECT {', '.join(_AGENT_VIEW_COLUMNS)} FROM ledger_agent_view"
        ).fetchall()
        return {
            "candidates": [dict(r) for r in rows],
            "candidate_hashes": sorted(r["candidate_hash"] for r in rows),
            "semantic_hashes": sorted({r["semantic_hash"] for r in rows if r["semantic_hash"]}),
            "killed_families": self.killed_families(),
        }

    @staticmethod
    def agent_view_columns() -> tuple[str, ...]:
        """The exact columns the agent view exposes — used by the CR-1 leak test."""
        return _AGENT_VIEW_COLUMNS
