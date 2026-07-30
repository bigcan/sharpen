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

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

from ...signals.features import Panel
from ...signals.generation.cohort import CohortConfig
from ...signals.generation.fitness import FitnessConfig
from ..agentic.proposer import LibrarySeedProposer, Proposer
from ..ledger import TrialLedger
from ..lockbox.incubation import IncubationCriterion
from ..lockbox.lockbox import Lockbox
from .fdr import OnlineFDR

if TYPE_CHECKING:
    from ...signals.generation.base_sleeves import SleeveComponents
    from ..corrected_contract import CorrectedConfig
    from ..search_memory import SearchMemoryConfig


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
    power: "SubstratePower | None" = None    # NOW-5 statistical-power stamp (None ⇒ unstamped/legacy)
    # F14 (Tier B): per-sleeve gross/cost/gross-exposure decomposition of base_returns for the
    # overlay-cost correction. None ⇒ evolve's unit-gross fallback (exact for synthetic/proxy books).
    base_components: "dict[str, SleeveComponents] | None" = None


@dataclass(frozen=True, slots=True)
class SubstratePower:
    """The statistical-power stamp for a prepared substrate (NOW-5, C2-01/C6-07): how deep the panel
    is and the minimum marginal ΔSR the funnel could DETECT there, interpolated from the E1/E2
    calibration sweep. A high ``implied_mde_delta_sr`` means the substrate is underpowered for realistic
    alphas — mining it spends proposer tokens + FDR wealth at ~zero detection probability (this is the
    stamp that would have flagged the flagship cross_asset run mined on a ~504-bar/holdout-126 panel,
    implied MDE ≈ 4.45).

    ``implied_mde_delta_sr`` is ``+inf`` exactly when ``interp_mode`` is an ``unmeasured_*`` mode: the
    substrate sits off the calibration grid, so its power was never measured and the stamp claims none
    (fail-closed — see :func:`interp_mde`). It is a sentinel, not an estimate; do not average, plot, or
    regress it alongside the measured modes."""

    panel_T: int
    holdout_bars: int
    holdout_frac: float
    implied_mde_delta_sr: float      # +inf ⇔ interp_mode is 'unmeasured_*' (sentinel, not an estimate)
    # measured-estimate modes: 'grid' | 'interpolated' | 'extrapolated_low'
    # fail-closed sentinel modes: 'unmeasured_high' (off the top of the grid) | 'unmeasured_degenerate'
    interp_mode: str
    calibration_sweep_hash: str


@dataclass(frozen=True, slots=True)
class PowerGuard:
    """Substrate-power gate config (from configs/crucible_power.gates.yaml). ``ceiling`` is the
    plausible true marginal ΔSR; a substrate whose implied MDE exceeds it is UNDERPOWERED. ``action``
    is 'warn' (log, still mine) or 'refuse' (skip the mine unless ``force``)."""

    enabled: bool
    ceiling: float
    action: str
    force: bool = False              # --force-underpowered: mine anyway under action='refuse'


def _power_holdout_bars(panel_T: int, holdout_frac: float) -> int:
    """Bars in the binding held-out validation window — mirrors the generation split EXACTLY (the
    funnel re-scores survivors on these bars), so the power stamp is measured against the gate's own N.
    Verified against the calibration sweep: T={756,1512,2782,4044} ⇒ holdout={189,378,696,1011}."""
    return int(panel_T) - int(int(panel_T) * (1.0 - float(holdout_frac)))


#: What a sweep JSON is a curve FOR. Files written before the cross-sectional surface existed carry no
#: ``candidate_type`` and are, in fact, overlay curves (they plant ``macro:plant`` and score it through
#: ``_overlay_returns``), so that is the honest default for a legacy file — not a wildcard.
_DEFAULT_SWEEP_CANDIDATE_TYPE = "overlay"


def sweep_candidate_type(sweep: dict) -> str:
    """The candidate type a sweep characterizes (``overlay`` for pre-surface files)."""
    return str(sweep.get("mde_sweep", {}).get("candidate_type", _DEFAULT_SWEEP_CANDIDATE_TYPE))


def _pooled_points(sweep: dict) -> list[tuple[int, float]]:
    """``[(holdout_bars, mde)]`` ascending, POOLED over any extra sweep axis by taking the WORST
    (largest) MDE at each depth.

    The overlay sweep has one row per ``holdout_bars``, so pooling is a no-op there and every existing
    interpolation property is preserved. The cross-sectional surface adds a breadth axis (``n``), giving
    several rows per depth — and the measurement (2026-07-29, N=12→100) found **no systematic
    N-dependence** in MDE, which is what theory says: ΔSR is already risk-adjusted and the standard
    error of a Sharpe DIFFERENCE is set by the number of TIME observations, not by the cross-section.
    Indexing the lookup by ``n`` would therefore claim a resolution the data does not support, so we
    pool and keep the worst measured value at each depth — the fail-safe direction for a guard that
    refuses when MDE is too high.

    Rows with a null MDE (the sweep detected nothing at any beta) are DROPPED rather than read as 0:
    "undetected" is the opposite of "detectable at zero effect"."""
    by_h: dict[int, float] = {}
    for r in sweep.get("mde_sweep", {}).get("rows", []):
        m = r.get("mde_realized_delta_sr")
        if m is None:
            continue
        mf = float(m)
        if not math.isfinite(mf):
            continue
        hb = int(r["holdout_bars"])
        by_h[hb] = max(by_h.get(hb, -math.inf), mf)
    return sorted(by_h.items())


def interp_mde(holdout_bars: int, sweep: dict) -> tuple[float, str]:
    """Minimum-detectable marginal ΔSR (at the sweep's target power) for a holdout of ``holdout_bars``,
    read off the E1/E2 MDE sweep. Returns ``(mde, mode)``.

    BETWEEN grid points we interpolate linearly in x = 1/√(holdout_bars). The interpolant is ANCHORED
    at two MEASURED neighbours, so 1/√N is only the basis it bends between them — never a claim about
    the funnel's true scaling.

    OFF THE TOP of the grid we REFUSE to answer: ``(+inf, 'unmeasured_high')``. Until S553-cont-135
    this branch applied a 1/√N law ("more bars ⇒ smaller MDE ∝ 1/√N"), which the project's OWN
    cont-129 intraday Stage-0 experiment then DIRECTLY MEASURED and FALSIFIED: above N_eff≈1000 the
    deflated funnel's MDE FLATTENS to ~N^-0.21, and the exponent is itself DECAYING (it is already
    ~N^-0.567 across this very grid, 189→1011 — so 1/√N never described this curve, at either end).
    Because 1/√N falls FASTER than the funnel really gains power, that branch UNDER-stated the MDE:
    it reported MORE power than exists and FAILED OPEN — the one direction a guard must never fail.
    It was latent only because the grid's top (1011) is exactly where the daily substrates sit; the
    first deeper or higher-frequency substrate — precisely where one goes LOOKING for power — would
    have been waved through (at holdout 20000 it computed MDE 0.315 vs the measured law's 0.749,
    ALLOWing a mine the evidence refuses; the fail-open crossover was holdout ≈7,955).

    Re-fitting the branch to the measured -0.21 was considered and REJECTED. That exponent is measured
    on a DIFFERENT substrate and axis (intraday N_eff at H=1) and only over 1011→10210, so past that it
    is the same extrapolation wearing a better exponent; the exponent is still decaying, so -0.21 keeps
    under-stating further out; and the cont-131 audit pinned a data-INDEPENDENT ``marginal_t`` floor
    (~0.4), so the true MDE does not decay to zero at all — ANY power law → 0 is asymptotically
    fail-open (the -0.21 law "clears" a 0.50 ceiling at holdout ≈1.4e5, which that floor says is
    unreachable at ANY N). Off the grid the honest statement is not a smaller number; it is "not
    measured". The unblock is to EXTEND the sweep so real substrates INTERPOLATE between measured
    anchors — this branch is the forcing function for that, and ``--force-underpowered`` is the
    operator's explicit, logged override.

    The +inf is a fail-closed SENTINEL, not an estimate: it flows through the caller's
    ``implied_mde_delta_sr > ceiling`` test to REFUSE, and reads in the tick log as "power never
    measured here". Only the ``grid`` / ``interpolated`` / ``extrapolated_low`` modes carry a real MDE
    estimate; consequently MDE is monotone non-increasing in ``holdout_bars`` only ACROSS the measured
    domain (h ≤ h_hi), not across the sentinel.

    The BOTTOM branch keeps 1/√N — NOT because it is conservative there (it is not: at the measured
    -0.567 rate, sqrt under-states by ~3% at holdout 126 rising to ~12% at 30 — the SAME
    anti-conservative direction as the bug above) but because it is verdict-INVARIANT: for h < h_lo it
    is bounded below by ``m_lo`` = 3.63, the LARGEST measured MDE, which already exceeds any plausible
    true-alpha ceiling (ΔSR ~0.3-0.5), so it refuses whatever the exponent. Its value is a provenance
    figure (the flagship's ≈4.45), never a number the verdict turns on — so it is left byte-stable
    rather than perturbed for a cosmetic gain.
    """
    pts = _pooled_points(sweep)
    if not pts:
        return math.inf, "unmeasured_empty"
    h = int(holdout_bars)
    if h <= 0:            # degenerate/empty holdout: nothing is tested, so nothing is detectable. Also
        # guards the divisions below, which used to raise ZeroDivisionError at h=0 and — worse —
        # silently return a COMPLEX mde at h<0, which would blow up the caller's `>` compare.
        return math.inf, "unmeasured_degenerate"
    for hi, mi in pts:
        if hi == h:
            return mi, "grid"
    h_lo, m_lo = pts[0]
    h_hi, m_hi = pts[-1]
    if h < h_lo:
        return m_lo * (h_lo / h) ** 0.5, "extrapolated_low"     # ≥ m_lo ⇒ refuses whatever the exponent
    if h > h_hi:
        return math.inf, "unmeasured_high"                      # off-grid ⇒ unmeasured ⇒ claim NO power
    for (a_h, a_m), (b_h, b_m) in zip(pts, pts[1:]):
        if a_h <= h <= b_h:
            x, xa, xb = h ** -0.5, a_h ** -0.5, b_h ** -0.5
            return a_m + (b_m - a_m) * (x - xa) / (xb - xa), "interpolated"
    return m_hi, "grid"                                          # unreachable (guarded above)


def stamp_substrate_power(panel_T: int, holdout_frac: float, sweep: "dict | Mapping[str, dict]",
                          sweep_hash: str,
                          candidate_types: "tuple[str, ...] | None" = None) -> SubstratePower:
    """Build the :class:`SubstratePower` stamp for a panel of ``panel_T`` bars.

    ``sweep`` is either ONE sweep dict (the legacy form — treated as the curve for its own declared
    ``candidate_type``, i.e. ``overlay`` for pre-surface files) or a mapping ``{candidate_type: sweep}``.

    ``candidate_types`` names the paths this substrate will actually MINE. An MDE curve characterizes a
    GATE, and a tick mines cross_sectional AND overlay genomes through structurally different scoring
    paths whose curves are numerically different — at matched depth the cross-sectional path measures
    EQUAL-OR-WORSE than the overlay one (holdout 696: overlay 1.23 vs cross-sectional 1.33–1.85). So
    judging a cross-sectional mine by the overlay curve UNDER-states its MDE: the guard claims more
    power than the substrate has, which is the fail-OPEN direction crucible-v5.0 exists to close.

    The stamp therefore takes the WORST (largest) MDE across the types being mined, and a type with no
    measured curve yields ``(+inf, 'unmeasured_candidate_type')`` — refuse — rather than silently
    borrowing another type's curve. ``None`` keeps the historical single-curve behaviour."""
    hb = _power_holdout_bars(panel_T, holdout_frac)
    if not isinstance(sweep, Mapping) or "mde_sweep" in sweep:
        by_type: dict[str, dict] = {sweep_candidate_type(sweep): sweep}    # type: ignore[arg-type]
    else:
        by_type = dict(sweep)                                             # type: ignore[arg-type]
    wanted = tuple(candidate_types) if candidate_types else tuple(by_type)
    if not wanted:
        # No curve was consulted, so no power is claimed. Initialising the fold at -inf and returning
        # it would pass ANY ceiling — a fail-OPEN on the empty set, and precisely the shape of bug
        # crucible-v5.0 was written to close. An empty type set is UNMEASURED.
        return SubstratePower(panel_T=int(panel_T), holdout_bars=hb, holdout_frac=float(holdout_frac),
                              implied_mde_delta_sr=math.inf, interp_mode="unmeasured_empty",
                              calibration_sweep_hash=sweep_hash)
    worst_mde, worst_mode = -math.inf, "unmeasured_empty"
    for ct in wanted:
        sw = by_type.get(ct)
        if sw is None:
            worst_mde, worst_mode = math.inf, "unmeasured_candidate_type"
            break
        mde, mode = interp_mde(hb, sw)
        if mde > worst_mde:
            worst_mde, worst_mode = mde, (mode if len(wanted) == 1 else f"{mode}:{ct}")
    return SubstratePower(panel_T=int(panel_T), holdout_bars=hb, holdout_frac=float(holdout_frac),
                          implied_mde_delta_sr=float(worst_mde), interp_mode=worst_mode,
                          calibration_sweep_hash=sweep_hash)


def panel_content_hash(panel: Panel) -> str:
    """12-hex SHA-256 over a Panel's window + OHLCV/membership/feature-slot content (NOW-6, C2-07).
    Provenance ONLY — never read as a per-bar feature (no LEAK-2 surface) and never seeds any RNG. It
    lets a snapshot_hash cover the PRIMARY mining surface (price bars), which the alt-data catalog hash
    alone missed, so a price-bar arrival or a revision flips the substrate dirty."""
    h = hashlib.sha256()
    h.update(f"{panel.dates[0]}|{panel.dates[-1]}|{panel.T}x{panel.N}|".encode())
    h.update("\x1f".join(panel.tickers).encode())
    h.update(np.ascontiguousarray(panel.dates.view(np.int64)).tobytes())
    for arr in (panel.open, panel.high, panel.low, panel.close, panel.volume, panel.adv_usd):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(panel.active, dtype=bool).tobytes())
    h.update(np.ascontiguousarray(panel.sector_id).tobytes())
    for name in sorted(panel.feature_slots):
        h.update(f"{name}=".encode())
        h.update(np.ascontiguousarray(panel.feature_slots[name], dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def folded_snapshot_hash(catalog_hash: str, panel: Panel) -> str:
    """Combine the alt-data catalog metadata hash with the panel content hash into the single
    ``snapshot_hash`` consumed by ``substrate_dirty`` AND the manifest data pin (NOW-6/C2-07). Keeps
    the catalog hash as an input (alt-data metadata still matters) while adding price-bar coverage."""
    return hashlib.sha256(f"{catalog_hash}|{panel_content_hash(panel)}".encode()).hexdigest()[:12]


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
    # Phase 4 weak-signal COHORT gate. OPT-IN: when cohort_cfg is set AND cohort_mc_kwargs['enabled'],
    # the loop evaluates a cohort over this tick's overlay pool (Doc 1/2). None ⇒ no cohort path
    # (byte-identical pre-cohort behavior). cohort_gates_hash is the cohort gate file's provenance hash
    # (computed by the runner, symmetric with the funnel gates_hash) — pinned into the manifest + the
    # deterministic MC seed.
    cohort_cfg: CohortConfig | None = None
    cohort_mc_kwargs: dict | None = None
    cohort_gates_hash: str | None = None
    # crucible-v6.0 DECISION CONTRACT. "shipped" (default) = the historical 6-way AND re-scored on the
    # holdout. "corrected" = the audit §5 contract (one JKM Sharpe-difference z + a BINDING LORD++
    # p-gate + the three cheap guards; the F1-sealed marginal_t and F2-sealed dsr_aug legs dropped).
    # Its thresholds live in their OWN file (configs/crucible_corrected_contract.gates.yaml), so the
    # frozen funnel gates_hash moat is untouched — the same ADR-1 separation the lockbox/cohort/power
    # gates use. Selecting "corrected" changes the verdict FUNCTION; it is a per-substrate operator
    # decision, never an agent one (CR-1).
    #
    # DELIBERATE ASYMMETRY WITH THE CLI (crucible-v8.0). `crucible_orchestrator.py --contract` defaults
    # to "corrected" — that is the OPERATOR surface, and flipping it is the point of v8.0. This
    # dataclass default stays "shipped" because it is the LIBRARY surface: a programmatic caller that
    # builds a Substrate directly should state its contract, and a conservative default keeps every
    # existing fixture and embedding byte-identical rather than silently re-gating it. If you are
    # wiring a new programmatic path, pass `contract` EXPLICITLY — this default is a back-compat
    # anchor, not a recommendation.
    contract: str = "shipped"
    corrected_cfg: "CorrectedConfig | None" = None
    corrected_gates_hash: str | None = None
    # U4 SEARCH MEMORY (crucible-v10.0). Classifies each holdout rejection as DECISIVE (terminal — the
    # family enters killed_families and the proposer stops re-deriving it) or UNDERPOWERED (parked,
    # re-admitted once the substrate gains real power). Thresholds live in their own file
    # (configs/crucible_search_memory.gates.yaml), whose hash is pinned into the tick alongside the other
    # parallel gates. None ⇒ no classification, byte-identical to pre-U4.
    search_memory_cfg: "SearchMemoryConfig | None" = None
    search_memory_gates_hash: str | None = None
    # Cap on how many PARKED candidates a single tick may re-admit (CR-7: re-admissions consume the same
    # per-tick candidate budget as fresh hypotheses, so they must not be able to starve discovery).
    max_readmissions: int = 8

    def __post_init__(self) -> None:
        if self.lockbox is not None and self.incubation_criterion is None:
            raise ValueError("a Substrate with a lockbox must also carry an incubation_criterion "
                             "(the pre-registered CR-2 criterion pinned at enrollment)")
        if self.contract not in ("shipped", "corrected"):
            raise ValueError(f"contract must be 'shipped' or 'corrected'; got {self.contract!r}")
        if self.contract == "corrected" and self.corrected_cfg is None:
            raise ValueError("a Substrate with contract='corrected' must carry corrected_cfg "
                             "(CorrectedConfig from configs/crucible_corrected_contract.gates.yaml)")


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
    status: str = "OK"                # "OK" | "ERROR" (C9-11: distinguish a crashed tick from a ran one)
    error: str | None = None          # exception repr when status == "ERROR"
    # NOW-5 substrate-power stamp (C2-01/C6-07): the panel depth + interpolated MDE this tick mined at.
    panel_T: int | None = None
    holdout_bars: int | None = None
    implied_mde_delta_sr: float | None = None
    power_interp_mode: str | None = None


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
    manifest_hash     TEXT,
    status            TEXT,
    error             TEXT,
    panel_T           INTEGER,
    holdout_bars      INTEGER,
    implied_mde_delta_sr REAL,
    power_interp_mode TEXT
);
CREATE INDEX IF NOT EXISTS ix_ticks_substrate ON ticks(substrate_id);
"""


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: list[tuple[str, str]]) -> None:
    """Idempotently ALTER TABLE to add any missing ``(name, ddl)`` columns. ``CREATE TABLE IF NOT
    EXISTS`` never adds a column to an EXISTING table, so a schema that grew across versions needs
    this to migrate on-disk DBs (the tick table persists across runs)."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    conn.commit()


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
        # Migrate on-disk tick tables written by an earlier schema (status/error for C9-11; the
        # substrate-power columns for NOW-5) — CREATE-IF-NOT-EXISTS never alters an existing table.
        _ensure_columns(self._conn, "ticks", [
            ("status", "TEXT"), ("error", "TEXT"),
            ("panel_T", "INTEGER"), ("holdout_bars", "INTEGER"),
            ("implied_mde_delta_sr", "REAL"), ("power_interp_mode", "TEXT")])

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

    def clear_snapshot(self, substrate_id: str) -> None:
        """Forget the last observed snapshot for a substrate — roll back a tentative observation when
        a tick crashes mid-work, so the NEXT tick re-detects the data as new and retries it (C9-01).
        A no-op when the substrate was never seen."""
        self._conn.execute("DELETE FROM last_seen WHERE substrate_id = ?", (substrate_id,))
        self._conn.commit()

    # --- tick history ------------------------------------------------------------------------------
    def record_tick(self, rec: TickRecord) -> None:
        self._conn.execute(
            "INSERT INTO ticks (tick_ts, substrate_id, dirty, mined, reason, snapshot_hash, "
            "n_preregistered, n_scored, n_promising, fdr_charged_total, budget_breached, "
            "burst_target, manifest_hash, status, error, panel_T, holdout_bars, "
            "implied_mde_delta_sr, power_interp_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rec.tick_ts, rec.substrate_id, int(rec.dirty), int(rec.mined), rec.reason,
             rec.snapshot_hash, rec.n_preregistered, rec.n_scored, rec.n_promising,
             rec.fdr_charged_total, int(rec.budget_breached), rec.burst_target, rec.manifest_hash,
             rec.status, rec.error, rec.panel_T, rec.holdout_bars, rec.implied_mde_delta_sr,
             rec.power_interp_mode))
        self._conn.commit()

    def ticks(self, substrate_id: str | None = None) -> list[dict]:
        """The tick history (all, or one substrate), oldest first — the unattended-nights audit trail."""
        if substrate_id is None:
            rows = self._conn.execute("SELECT * FROM ticks ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ticks WHERE substrate_id = ? ORDER BY id", (substrate_id,)).fetchall()
        return [dict(r) for r in rows]
