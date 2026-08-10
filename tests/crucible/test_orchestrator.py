"""Crucible P3 — the continuous orchestrator (spec §8 P3 row, §6.1, §7, §10.1).

Exit-gate assertions (the P3 "gate to next"):
  * mining happens ONLY when a substrate is dirty (new data OR fresh hypotheses);
  * a clean-panel night correctly no-ops (spends zero FDR wealth);
  * a new-data night re-mines and the online-FDR budget is accounted per test;
  * the per-tick cost budget halts-and-reports (CR-7) rather than overspending;
  * four unattended nights are versioned and reproducible (bit-identical tick history).

Plus unit coverage of the LORD++ recurrence (§6.1), the ``substrate_dirty`` gate, the budgeter, and
the burst router. Design: docs/research/crucible_agentic_discovery_spec.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crucible import (
    CRUCIBLE_VERSION,
    OnlineFDR,
    OrchestratorStore,
    Substrate,
    TickBudget,
    TrialLedger,
    route_burst,
    run_orchestrator_tick,
    substrate_dirty,
)
from finrl_pro_ds.crucible.orchestrator.burst import GPUHUB_1, GPUHUB_2, LOCAL
from finrl_pro_ds.crucible.orchestrator.fdr import gamma
from finrl_pro_ds.crucible.orchestrator.substrate import PreparedSubstrate
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

GATES = "configs/signal_eval.gates.yaml"
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
T, N = 320, 12


# ============================================================ online-FDR (LORD++) ====

def test_gamma_sums_to_at_most_one_and_is_non_increasing() -> None:
    """The spending sequence must sum to ≤1 (FDR-validity) and decay (non-increasing) — spec §6.1."""
    vals = [gamma(k) for k in range(1, 5000)]
    assert all(a >= b > 0 for a, b in zip(vals, vals[1:]))       # strictly positive, non-increasing
    total = sum(gamma(k) for k in range(1, 500_000))
    assert 0.999 < total <= 1.0                                  # normalized, conservative (≤1)


def test_lordpp_recurrence_matches_closed_form() -> None:
    """α_t = γ_t·W0 + (α−W0)·γ_{t−τ₁} + α·Σ_{j≥2}γ_{t−τⱼ}, discoveries strictly before t (§6.1)."""
    alpha, w0 = 0.10, 0.05
    f = OnlineFDR(alpha=alpha, w0=w0)
    assert f.next_level() == pytest.approx(gamma(1) * w0)        # t=1, no prior discovery
    f.observe(is_discovery=False)                               # t=1 spent, no discovery
    assert f.next_level() == pytest.approx(gamma(2) * w0)        # t=2, still no τ
    f.observe(is_discovery=True)                               # t=2 is a discovery → τ₁=2
    # t=3: base decay + the first-discovery replenishment term (α−W0)·γ(3−2)
    assert f.next_level() == pytest.approx(gamma(3) * w0 + (alpha - w0) * gamma(1))


def test_fdr_discovery_replenishes_budget() -> None:
    """A discovery lifts the very next test's level above where pure decay would have left it."""
    decayed = OnlineFDR(alpha=0.10)
    for _ in range(2):
        decayed.observe(is_discovery=False)
    pure_decay_t3 = decayed.next_level()

    replenished = OnlineFDR(alpha=0.10)
    replenished.observe(is_discovery=False)
    replenished.observe(is_discovery=True)                     # τ₁=2
    assert replenished.next_level() > pure_decay_t3


def test_fdr_serialization_roundtrip() -> None:
    f = OnlineFDR(alpha=0.08, w0=0.03, alpha_floor=1e-4)
    for disc in (False, True, False):
        f.observe(is_discovery=disc)
    g = OnlineFDR.from_json(f.to_json())
    assert (g.alpha, g.w0, g.alpha_floor, g.num_tests, g.discoveries) == (
        f.alpha, f.w0, f.alpha_floor, f.num_tests, f.discoveries)
    assert g.next_level() == pytest.approx(f.next_level())      # budget resumes identically


def test_fdr_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        OnlineFDR(alpha=0.0)
    with pytest.raises(ValueError):
        OnlineFDR(alpha=0.1, w0=0.2)                            # w0 > alpha


# ============================================================ substrate_dirty gate ====

def test_substrate_dirty_truth_table() -> None:
    assert substrate_dirty(data_changed=False, n_fresh_hypotheses=0)[0] is False   # clean → no-op
    assert substrate_dirty(data_changed=True, n_fresh_hypotheses=0)[0] is True
    assert substrate_dirty(data_changed=False, n_fresh_hypotheses=3)[0] is True
    assert substrate_dirty(data_changed=True, n_fresh_hypotheses=3)[0] is True


# ============================================================ budgeter (CR-7) ====

def test_budget_affordability_and_breach() -> None:
    b = TickBudget(max_candidates=10, max_tokens=100)
    assert b.can_afford(candidates=10, tokens=100)
    assert not b.can_afford(candidates=11, tokens=0)
    assert not b.can_afford(candidates=0, tokens=101)
    b.charge(candidates=6, tokens=40)
    assert not b.can_afford(candidates=5, tokens=0)             # 6+5 > 10
    with pytest.raises(ValueError):
        b.charge(candidates=5, tokens=0)                       # charging past a cap is an error
    b.mark_breach("substrate-x skipped")
    assert b.breached and b.breach_reason == "substrate-x skipped"


def test_budget_unbounded_axis() -> None:
    b = TickBudget(max_candidates=None, max_tokens=None)
    assert b.can_afford(candidates=10_000, tokens=10**9)


# ============================================================ burst router ====

def test_burst_router_fleet_rule() -> None:
    assert route_burst(est_candidates=5, is_hpo=True).target == GPUHUB_1        # HPO → gpuhub-1
    assert route_burst(est_candidates=5, is_hpo=False).target == LOCAL          # small non-HPO local
    assert route_burst(est_candidates=500, is_hpo=False,
                       local_capacity=128).target == GPUHUB_2                   # big non-HPO bursts


# ============================================================ substrate fixtures ====

def _panel(seed: int, *, extra_slot: bool = False) -> Panel:
    """A NOISE OHLCV panel carrying a ``macro:regime`` feature slot (drives the overlay path). With
    ``extra_slot`` a SECOND macro series is present — a new data domain → new overlay hypotheses."""
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    slots = {"macro:regime": (np.sin(2 * np.pi * np.arange(T) / 80.0)
                              + 0.2 * rng.standard_normal(T)).astype(np.float64)}
    if extra_slot:
        slots["macro:regime2"] = (np.cos(2 * np.pi * np.arange(T) / 55.0)).astype(np.float64)
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots=slots)


def _base(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    ts = pd.date_range("2014-01-02", periods=T, freq="B").view("int64").astype(np.float64) / 1e9
    return base, ts


def _make_substrate(tmp_path, name: str, state: dict) -> Substrate:
    """A substrate whose ``prepare`` reads a mutable ``state`` dict — so a test can simulate new data
    arriving between nights by flipping ``state['extra_slot']``. ``snapshot_hash`` reflects the slots
    present (new slot ⇒ new snapshot ⇒ ``data_changed``)."""
    def prepare() -> PreparedSubstrate:
        panel = _panel(0, extra_slot=state["extra_slot"])
        base, ts = _base(0)
        snap = "snap-" + "-".join(sorted(panel.feature_slots))
        return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash=snap)
    ledger = TrialLedger(tmp_path / f"{name}_ledger.db")
    return Substrate(substrate_id=name, prepare=prepare, ledger=ledger,
                     cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK))


# ============================================================ orchestrator: exit gate ====

@pytest.mark.slow
def test_clean_panel_night_noops_and_conserves_fdr(tmp_path) -> None:
    """Night 1 mines (fresh hypotheses); an identical night 2 NO-OPS and spends no FDR wealth."""
    store = OrchestratorStore(tmp_path / "orch.db")
    sub = _make_substrate(tmp_path, "syn", {"extra_slot": False})

    r1 = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                               tick_ts="2026-07-02T00:00:00+00:00")
    o1 = r1.outcomes[0]
    assert o1.mined and o1.n_preregistered > 0
    assert o1.fdr_num_tests == o1.n_preregistered              # one FDR test per pre-registered spec
    assert o1.n_promising == 0                                 # noise → the filter holds

    r2 = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                               tick_ts="2026-07-03T00:00:00+00:00")
    o2 = r2.outcomes[0]
    assert not o2.mined and not o2.dirty                       # clean-panel night no-ops
    assert o2.fdr_num_tests == o1.fdr_num_tests                # FDR wealth conserved (no new test)


@pytest.mark.slow
def test_new_data_night_remines_and_charges_fdr(tmp_path) -> None:
    """A new macro series (new feature slot) → new overlay hypotheses → the substrate re-mines and
    the online-FDR test count advances; every mined spec's ledger row carries its fdr charge."""
    store = OrchestratorStore(tmp_path / "orch.db")
    state = {"extra_slot": False}
    sub = _make_substrate(tmp_path, "syn", state)

    r1 = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                               tick_ts="2026-07-02T00:00:00+00:00")
    tests_after_1 = r1.outcomes[0].fdr_num_tests

    state["extra_slot"] = True                                 # new data domain arrives
    r2 = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                               tick_ts="2026-07-03T00:00:00+00:00")
    o2 = r2.outcomes[0]
    assert o2.mined and o2.dirty
    assert o2.fdr_num_tests > tests_after_1                    # new tests charged on new data
    assert o2.fdr_charged_total > 0.0

    # every pre-registered spec mined this night has a non-null fdr_wealth_charged in the ledger
    charged = sub.ledger._conn.execute(
        "SELECT COUNT(*) FROM trial_ledger WHERE fdr_wealth_charged IS NOT NULL").fetchone()[0]
    assert charged >= o2.n_preregistered


@pytest.mark.slow
def test_budget_breach_skips_substrate(tmp_path) -> None:
    """A per-tick candidate cap below the fresh-spec count halts-and-reports: the substrate is NOT
    mined, the breach is recorded, and no FDR wealth is spent (CR-7)."""
    store = OrchestratorStore(tmp_path / "orch.db")
    sub = _make_substrate(tmp_path, "syn", {"extra_slot": False})
    budget = TickBudget(max_candidates=1, max_tokens=None)     # too small for the seed bank

    r = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                              tick_ts="2026-07-02T00:00:00+00:00", budget=budget)
    o = r.outcomes[0]
    assert o.dirty and not o.mined and o.budget_breached
    assert r.budget.breached and o.fdr_num_tests == 0          # nothing scored, no FDR spent


@pytest.mark.slow
def test_four_nights_versioned_and_reproducible(tmp_path) -> None:
    """The P3 exit gate: 4 unattended nights, mining ONLY when dirty (night 1), versioned, and
    bit-identical across two independent runs from clean state."""
    nights = [f"2026-07-0{d}T00:00:00+00:00" for d in (2, 3, 4, 5)]

    def four_nights(root) -> list[dict]:
        store = OrchestratorStore(root / "orch.db")
        sub = _make_substrate(root, "syn", {"extra_slot": False})
        for ts in nights:
            res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES, tick_ts=ts)
            assert res.crucible_version == CRUCIBLE_VERSION     # versioned
        return store.ticks("syn")

    a = four_nights(tmp_path / "runA")
    b = four_nights(tmp_path / "runB")

    # mined only when dirty: exactly night 1 mines; nights 2–4 no-op on the unchanged panel
    mined_flags = [row["mined"] for row in a]
    assert mined_flags == [1, 0, 0, 0]
    # reproducible: identical tick history (mined flags, FDR charge, promising count, manifest hash)
    cols = ("mined", "dirty", "n_preregistered", "n_scored", "n_promising", "fdr_charged_total",
            "manifest_hash", "reason")
    assert [{c: row[c] for c in cols} for row in a] == [{c: row[c] for c in cols} for row in b]


# ============================================================ NOW-2: tick resilience (C9-01/C9-11) ====

def _boom() -> PreparedSubstrate:
    raise RuntimeError("prepare exploded")


def test_tick_isolates_a_failing_substrate(tmp_path) -> None:
    """A substrate whose prepare() raises does NOT abort the night: it records an ERROR tick, its
    snapshot is NOT advanced (never observed → next tick retries), and later substrates still run."""
    store = OrchestratorStore(tmp_path / "orch.db")
    bad_a = Substrate(substrate_id="bad_a", prepare=_boom, ledger=TrialLedger(tmp_path / "a.db"),
                      cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK))
    bad_b = Substrate(substrate_id="bad_b", prepare=_boom, ledger=TrialLedger(tmp_path / "b.db"),
                      cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK))

    res = run_orchestrator_tick(substrates=[bad_a, bad_b], store=store, gates_path=GATES,
                                tick_ts="2026-07-02T00:00:00+00:00")
    # both substrates produced an outcome — the first failure did not abort the second
    assert [o.substrate_id for o in res.outcomes] == ["bad_a", "bad_b"]
    assert all(o.status == "ERROR" and not o.mined for o in res.outcomes)
    # ERROR ticks recorded; snapshots NOT advanced (a failed observation must be retried next tick)
    assert [row["status"] for row in store.ticks()] == ["ERROR", "ERROR"]
    assert store.last_snapshot_hash("bad_a") is None and store.last_snapshot_hash("bad_b") is None


def test_tick_error_rolls_back_snapshot_to_prev(tmp_path) -> None:
    """A substrate that fails AFTER being observed once rolls its snapshot BACK to the last good
    value (not a tentative new one), so the next tick still detects the pending change (C9-01)."""
    store = OrchestratorStore(tmp_path / "orch.db")
    store.set_snapshot_hash("s", "prev-good")               # a prior successful observation
    sub = Substrate(substrate_id="s", prepare=_boom, ledger=TrialLedger(tmp_path / "s.db"),
                    cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK))
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                          tick_ts="2026-07-03T00:00:00+00:00")
    assert store.last_snapshot_hash("s") == "prev-good"     # rolled back, not advanced


def test_orchestrator_store_migrates_status_columns(tmp_path) -> None:
    """C9-11: status/error are added by an idempotent migration to a pre-existing ticks table
    (CREATE-IF-NOT-EXISTS never adds columns); opening twice does not error."""
    import sqlite3
    db = tmp_path / "orch.db"
    conn = sqlite3.connect(str(db))                          # simulate an OLD schema without status/error
    conn.execute("CREATE TABLE ticks (id INTEGER PRIMARY KEY AUTOINCREMENT, tick_ts TEXT, "
                 "substrate_id TEXT, dirty INTEGER, mined INTEGER, reason TEXT, snapshot_hash TEXT)")
    conn.commit()
    conn.close()
    store = OrchestratorStore(db)                            # migrates on open
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(ticks)")}
    assert {"status", "error"} <= cols
    store.close()
    OrchestratorStore(db).close()                           # idempotent second open


def test_tick_record_persists_the_holdout_denominator(tmp_path) -> None:
    """crucible-v12.0: `n_promising` is unreadable without the number of candidates the BINDING
    holdout gate adjudicated. The column must migrate onto a pre-v12 ticks table (as NULL — UNKNOWN,
    never 0) and must round-trip on a fresh write."""
    import sqlite3

    from finrl_pro_ds.crucible.orchestrator.substrate import TickRecord

    db = tmp_path / "orch.db"
    conn = sqlite3.connect(str(db))       # the REAL pre-v12 schema, minus n_holdout_tested only
    conn.execute("CREATE TABLE ticks (id INTEGER PRIMARY KEY AUTOINCREMENT, tick_ts TEXT NOT NULL, "
                 "substrate_id TEXT NOT NULL, dirty INTEGER NOT NULL, mined INTEGER NOT NULL, "
                 "reason TEXT, snapshot_hash TEXT, n_preregistered INTEGER, n_scored INTEGER, "
                 "n_promising INTEGER, fdr_charged_total REAL, budget_breached INTEGER, "
                 "burst_target TEXT, manifest_hash TEXT, status TEXT, error TEXT, panel_T INTEGER, "
                 "holdout_bars INTEGER, implied_mde_delta_sr REAL, power_interp_mode TEXT)")
    conn.execute("INSERT INTO ticks (tick_ts, substrate_id, dirty, mined, reason, snapshot_hash, "
                 "n_preregistered, n_scored, n_promising) VALUES "
                 "('t0', 's', 1, 1, 'legacy', 'h0', 8, 10, 0)")
    conn.commit()
    conn.close()

    store = OrchestratorStore(db)
    assert "n_holdout_tested" in {r["name"] for r in store._conn.execute("PRAGMA table_info(ticks)")}
    legacy = store.ticks("s")[0]
    assert legacy["n_holdout_tested"] is None, (
        "a pre-v12 tick must read UNKNOWN, not 0 — back-filling a zero would assert that a run we "
        "cannot inspect tested nothing")

    store.record_tick(TickRecord(tick_ts="t1", substrate_id="s", dirty=True, mined=True,
                                 reason="mined", snapshot_hash="h1", n_preregistered=8, n_scored=10,
                                 n_holdout_tested=8, n_promising=0))
    fresh = store.ticks("s")[-1]
    assert fresh["n_holdout_tested"] == 8 and fresh["n_promising"] == 0
    store.close()


# ============================================================ NOW-5: substrate-power guard (C2-01) ====

def test_power_guard_refuses_underpowered_mine(tmp_path) -> None:
    """NOW-5: under action='refuse' an underpowered substrate (implied MDE > ceiling) is NOT mined —
    zero ledger rows, zero FDR spend — but its tick row carries the power stamp. This is the guard that
    would have caught the 504-bar flagship (mined at implied MDE ≈ 4.45)."""
    from dataclasses import replace

    from finrl_pro_ds.crucible.orchestrator.substrate import PowerGuard, stamp_substrate_power
    sweep = {"mde_sweep": {"rows": [{"holdout_bars": 189, "mde_realized_delta_sr": 3.63},
                                    {"holdout_bars": 1011, "mde_realized_delta_sr": 1.40}]}}
    sub = _make_substrate(tmp_path, "syn", {"extra_slot": False})
    base_prepare = sub.prepare

    def prepare_underpowered() -> PreparedSubstrate:
        p = base_prepare()
        return replace(p, power=stamp_substrate_power(p.panel.T, 0.25, sweep, "h"))

    sub.prepare = prepare_underpowered
    store = OrchestratorStore(tmp_path / "orch.db")
    r = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                              tick_ts="2026-07-02T00:00:00+00:00",
                              power_gate=PowerGuard(enabled=True, ceiling=0.5, action="refuse"))
    o = r.outcomes[0]
    assert not o.mined and "UNDERPOWERED" in o.reason
    assert sub.ledger.count() == 0 and o.fdr_num_tests == 0    # no proposer/FDR spend on a dead substrate
    row = store.ticks("syn")[0]                                # power stamp recorded on the tick row
    assert row["panel_T"] == T and row["holdout_bars"] == T - int(T * 0.75)
    assert row["implied_mde_delta_sr"] > 0.5 and row["mined"] == 0
