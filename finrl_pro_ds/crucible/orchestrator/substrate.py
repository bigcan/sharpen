"""Substrate descriptor + persistent orchestrator state + the ``substrate_dirty`` gate (§10.1).

A **substrate** is one (data domain + universe) the loop mines over — e.g. the cross-asset ETF panel
with its macro/positioning feature slots, or a synthetic test panel. The orchestrator visits a list
of substrates each nightly tick.

**The nightly-cadence multiplicity hazard (§10.1).** COT publishes weekly, most FRED series monthly;
a *blind* nightly tick would re-mine an UNCHANGED panel every night. Re-mining unchanged data spends
online-FDR wealth (:mod:`.fdr`) with zero chance of new information — needlessly tightening the
substrate's future error budget. Hard rule: **a tick mines a substrate only if EITHER new data has
arrived on it OR a fresh, previously-unscored hypothesis batch exists for it.** Otherwise the tick
no-ops and reports "no new information". :func:`substrate_dirty` is that gate; nightly is only the
*eligibility* cadence.

Persistence is a single central SQLite DB (``OrchestratorStore``): the per-substrate online-FDR
budget (so it survives across nights) and the tick history (for the "4 unattended nights" audit and
for the reproducibility contract). The per-substrate *trial ledger* stays separate (owned by the
substrate) — the FDR budget and tick log are orchestration state, not scored trials.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ...signals.features import Panel
from ...signals.generation.fitness import FitnessConfig
from ..agentic.proposer import LibrarySeedProposer, Proposer
from ..ledger import TrialLedger
from ..lockbox.incubation import IncubationCriterion
from ..lockbox.lockbox import Lockbox
from .fdr import OnlineFDR


@dataclass(frozen=True, slots=True)
class PreparedSubstrate:
    """The concrete mining inputs for one substrate at one point in time (what ``Substrate.prepare``
    returns): the panel (with any PIT-safe feature slots), the base-sleeve books the overlay path
    tilts, the timestamp axis, the catalog asset classes (informational context for the agent), and
    the ``snapshot_hash`` pinning the exact data surface (feeds ``substrate_dirty`` + the manifest)."""

    panel: Panel
    base_returns: dict[str, np.ndarray]
    timestamps: np.ndarray
    asset_classes: tuple[str, ...]
    snapshot_hash: str


@dataclass
class Substrate:
    """A mining substrate the orchestrator visits each tick. ``prepare`` is a thunk that (re)loads
    the current data surface — called once per tick so freshly-arrived data is picked up. ``ledger``
    is the substrate's own append-only trial ledger (agent-blind); ``cfg``/``evolve_kwargs`` are the
    UNCHANGED funnel knobs. ``fdr_alpha`` is the substrate's target FDR level (its total error
    budget); ``is_hpo`` flags an HPO-shaped tick for the burst router (default False: mining is
    non-HPO)."""

    substrate_id: str
    prepare: Callable[[], PreparedSubstrate]
    ledger: TrialLedger
    cfg: FitnessConfig
    evolve_kwargs: dict
    proposer: Proposer = field(default_factory=LibrarySeedProposer)
    max_proposals: int = 32
    fdr_alpha: float = 0.10
    fdr_w0: float | None = None
    fdr_alpha_floor: float = 0.0
    is_hpo: bool = False
    est_tokens_per_tick: int = 0     # proposer token estimate for the CR-7 budget (0 for offline)
    # CR-8 forward-incubation lockbox (P4). OPT-IN: when both are set, PROMISING survivors are
    # enrolled and accrued forward each tick; when ``lockbox`` is None the substrate does not incubate
    # (byte-identical P3 behavior). ``incubation_criterion`` is the pre-registered CR-2 lock pinned at
    # enrollment; require it whenever a lockbox is attached.
    lockbox: Lockbox | None = None
    incubation_criterion: IncubationCriterion | None = None

    def __post_init__(self) -> None:
        if self.lockbox is not None and self.incubation_criterion is None:
            raise ValueError("a Substrate with a lockbox must also carry an incubation_criterion "
                             "(the pre-registered CR-2 criterion pinned at enrollment)")


def substrate_dirty(*, data_changed: bool, n_fresh_hypotheses: int) -> tuple[bool, str]:
    """The §10.1 eligibility gate. Dirty (mine-worthy) iff new data arrived OR ≥1 fresh, not-yet-
    scored hypothesis exists. Returns ``(dirty, reason)`` — the reason is recorded in the tick log so
    the "mined only when dirty" property is auditable across the unattended nights."""
    if data_changed and n_fresh_hypotheses > 0:
        return True, f"new data + {n_fresh_hypotheses} fresh hypotheses"
    if data_changed:
        return True, "new data arrived on substrate"
    if n_fresh_hypotheses > 0:
        return True, f"{n_fresh_hypotheses} fresh (unscored) hypotheses"
    return False, "no new data and no fresh hypotheses - conserving FDR wealth"


@dataclass(frozen=True, slots=True)
class TickRecord:
    """One row of the tick history — one substrate on one night."""

    tick_ts: str
    substrate_id: str
    dirty: bool
    mined: bool
    reason: str
    snapshot_hash: str
    n_preregistered: int = 0
    n_scored: int = 0
    n_promising: int = 0
    fdr_charged_total: float = 0.0
    budget_breached: bool = False
    burst_target: str | None = None
    manifest_hash: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS fdr_state (
    substrate_id  TEXT PRIMARY KEY,
    state_json    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS last_seen (
    substrate_id  TEXT PRIMARY KEY,
    snapshot_hash TEXT
);
CREATE TABLE IF NOT EXISTS ticks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_ts           TEXT NOT NULL,
    substrate_id      TEXT NOT NULL,
    dirty             INTEGER NOT NULL,
    mined             INTEGER NOT NULL,
    reason            TEXT,
    snapshot_hash     TEXT,
    n_preregistered   INTEGER,
    n_scored          INTEGER,
    n_promising       INTEGER,
    fdr_charged_total REAL,
    budget_breached   INTEGER,
    burst_target      TEXT,
    manifest_hash     TEXT
);
CREATE INDEX IF NOT EXISTS ix_ticks_substrate ON ticks(substrate_id);
"""


class OrchestratorStore:
    """SQLite-backed orchestration state: per-substrate online-FDR budgets + the full tick history.
    Single central file (``orchestrator.db``); the trial ledgers stay per-substrate."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "OrchestratorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- online-FDR budget (persist across nights) ------------------------------------------------
    def load_fdr(self, substrate_id: str, *, alpha: float, w0: float | None = None,
                 alpha_floor: float = 0.0) -> OnlineFDR:
        """The substrate's persisted FDR account, or a fresh one seeded with the given params if this
        substrate has never been tested. The stored ``(num_tests, discoveries)`` make the budget
        reproducible; the params come from the Substrate config on first use."""
        row = self._conn.execute(
            "SELECT state_json FROM fdr_state WHERE substrate_id = ?", (substrate_id,)).fetchone()
        if row is None:
            return OnlineFDR(alpha=alpha, w0=w0, alpha_floor=alpha_floor)
        return OnlineFDR.from_json(json.loads(row["state_json"]))

    def save_fdr(self, substrate_id: str, fdr: OnlineFDR) -> None:
        self._conn.execute(
            "INSERT INTO fdr_state (substrate_id, state_json) VALUES (?, ?) "
            "ON CONFLICT(substrate_id) DO UPDATE SET state_json = excluded.state_json",
            (substrate_id, json.dumps(fdr.to_json(), sort_keys=True)))
        self._conn.commit()

    # --- data-change detection (the 'new data arrived' leg of substrate_dirty) --------------------
    def last_snapshot_hash(self, substrate_id: str) -> str | None:
        """The data snapshot_hash last observed for this substrate (None if never). ``!=`` current ⇒
        new data has arrived."""
        row = self._conn.execute(
            "SELECT snapshot_hash FROM last_seen WHERE substrate_id = ?", (substrate_id,)).fetchone()
        return None if row is None else row["snapshot_hash"]

    def set_snapshot_hash(self, substrate_id: str, snapshot_hash: str) -> None:
        """Record the snapshot observed this tick (called every tick, mined or not, so a data change
        on a later night is detected against the most recent observation)."""
        self._conn.execute(
            "INSERT INTO last_seen (substrate_id, snapshot_hash) VALUES (?, ?) "
            "ON CONFLICT(substrate_id) DO UPDATE SET snapshot_hash = excluded.snapshot_hash",
            (substrate_id, snapshot_hash))
        self._conn.commit()

    # --- tick history ------------------------------------------------------------------------------
    def record_tick(self, rec: TickRecord) -> None:
        self._conn.execute(
            "INSERT INTO ticks (tick_ts, substrate_id, dirty, mined, reason, snapshot_hash, "
            "n_preregistered, n_scored, n_promising, fdr_charged_total, budget_breached, "
            "burst_target, manifest_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rec.tick_ts, rec.substrate_id, int(rec.dirty), int(rec.mined), rec.reason,
             rec.snapshot_hash, rec.n_preregistered, rec.n_scored, rec.n_promising,
             rec.fdr_charged_total, int(rec.budget_breached), rec.burst_target, rec.manifest_hash))
        self._conn.commit()

    def ticks(self, substrate_id: str | None = None) -> list[dict]:
        """The tick history (all, or one substrate), oldest first — the unattended-nights audit trail."""
        if substrate_id is None:
            rows = self._conn.execute("SELECT * FROM ticks ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ticks WHERE substrate_id = ? ORDER BY id", (substrate_id,)).fetchall()
        return [dict(r) for r in rows]
