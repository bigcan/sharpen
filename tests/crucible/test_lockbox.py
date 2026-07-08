"""Crucible P4 — the forward-incubation lockbox (spec §8 P4 row, §6.2, CR-8).

Exit-gate assertions (the P4 "gate to next"):
  * a seeded survivor ENROLLS and accrues forward OOS evidence on bars strictly after proposal_ts;
  * NO card is eligible for the human gate before its incubation criterion clears.

Plus the CR-8 keystone tripwires (only post-proposal bars ever count), the fixed-horizon /
anti-peeking verdict state machine, idempotent enrollment, and the P3-preserving no-op when a
substrate has no lockbox. Design: docs/research/crucible_agentic_discovery_spec.md §6.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crucible import (
    ForwardEvidence,
    IncubationCriterion,
    Lockbox,
    OrchestratorStore,
    STATUS_CLEARED,
    STATUS_INCUBATING,
    STATUS_REJECTED,
    Substrate,
    TrialLedger,
    forward_evidence,
    load_incubation_criterion,
    run_orchestrator_tick,
    updated_card,
)
from finrl_pro_ds.crucible.agentic.card import DiscoveryCard
from finrl_pro_ds.crucible.lockbox.lockbox import LockboxEntry, advance
from finrl_pro_ds.crucible.orchestrator.substrate import PreparedSubstrate
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.fitness import _CAND, FitnessConfig, _combined_book
from finrl_pro_ds.signals.generation.evolve import _overlay_returns
from finrl_pro_ds.signals.eval_harness import _ann_sharpe

GATES = "configs/signal_eval.gates.yaml"
LOCKBOX_GATES = "configs/crucible_lockbox.gates.yaml"
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
N = 12
OVERLAY_FORMULA = "macro:regime"        # a valid overlay DSL formula on the macro feature slot


# ============================================================ fixtures ====

def _panel(T: int, seed: int = 0) -> Panel:
    """A noise OHLCV panel carrying a ``macro:regime`` feature slot (drives the overlay path)."""
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2020-01-01") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    slots = {"macro:regime": (np.sin(2 * np.pi * np.arange(T) / 80.0)
                              + 0.2 * rng.standard_normal(T)).astype(np.float64)}
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots=slots)


def _base(T: int, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    ts = (pd.date_range("2020-01-01", periods=T, freq="B").view("int64").astype(np.float64) / 1e9)
    return base, ts


def _iso(ts_epoch_seconds: float) -> str:
    return pd.Timestamp(ts_epoch_seconds, unit="s", tz="UTC").isoformat()


def _make_card(proposal_ts: str, chash: str = "cand-abc") -> DiscoveryCard:
    return DiscoveryCard(candidate_hash=chash, formula=OVERLAY_FORMULA, candidate_type="overlay",
                         crucible_version="crucible-v2.5", gates_hash="deadbeef0000",
                         proposal_ts=proposal_ts, data_snapshot_hash="snap-1", verdict="PROMISING")


# ============================================================ CR-8 keystone: forward window ====

def test_forward_mask_strictly_excludes_pre_proposal_bars() -> None:
    """Only bars STRICTLY after proposal_ts are in the forward window — the CR-8 boundary."""
    from finrl_pro_ds.crucible.lockbox.incubation import forward_mask
    _, ts = _base(300)
    p_idx = 200
    proposal = _iso(float(ts[p_idx]))
    mask = forward_mask(ts, proposal)
    assert not mask[:p_idx + 1].any()                    # nothing at/before the proposal bar
    assert mask[p_idx + 1:].all()                        # every later bar is forward
    assert int(mask.sum()) == len(ts) - (p_idx + 1)


def test_forward_mask_monotonic_shrink_as_proposal_moves_later() -> None:
    """A later proposal_ts yields a strictly smaller forward window (never re-includes an earlier bar)."""
    from finrl_pro_ds.crucible.lockbox.incubation import forward_mask
    _, ts = _base(300)
    counts = [int(forward_mask(ts, _iso(float(ts[p]))).sum()) for p in (100, 150, 200, 250)]
    assert counts == sorted(counts, reverse=True)        # strictly non-increasing as proposal moves later
    assert counts[0] > counts[-1]


def test_forward_mask_accepts_datetime64_and_epoch_axes() -> None:
    """The mask handles both a datetime64 panel axis and a numeric epoch-seconds axis identically."""
    from finrl_pro_ds.crucible.lockbox.incubation import forward_mask
    _, ts = _base(120)
    proposal = _iso(float(ts[60]))
    dt = pd.to_datetime(ts, unit="s").values.astype("datetime64[ns]")
    np.testing.assert_array_equal(forward_mask(ts, proposal), forward_mask(dt, proposal))


# ============================================================ forward_evidence ====

def _forward_evidence(panel, base, ts, proposal, cfg):
    return forward_evidence(formula=OVERLAY_FORMULA, candidate_type="overlay", panel=panel,
                            base_returns=base, timestamps=ts, proposal_ts=proposal, cfg=cfg,
                            hold_horizon=21, cost_bps=0.001, ls_min_names=6)


def test_forward_evidence_measures_only_forward_marginal_sharpe() -> None:
    """White-box: forward_evidence returns exactly the annualized Sharpe of the marginal book stream
    (b_aug − b_base) restricted to bars strictly after proposal_ts — no pre-proposal bar contributes."""
    from finrl_pro_ds.crucible.lockbox.incubation import forward_mask
    T = 260
    panel, (base, ts) = _panel(T), _base(T)
    cfg = FitnessConfig(embargo=10)
    proposal = _iso(float(ts[180]))
    ev = _forward_evidence(panel, base, ts, proposal, cfg)
    assert ev is not None and ev.n_forward_bars > 0

    # recompute the marginal stream via the SAME funnel builders, slice forward, Sharpe — must match
    base_book = _combined_book({k: np.asarray(v) for k, v in base.items()}, ts, cfg)
    cand = _overlay_returns(OVERLAY_FORMULA, panel, base_book, cost_bps=0.001)[0]
    b_aug = _combined_book({**{k: np.asarray(v) for k, v in base.items()}, _CAND: cand}, ts, cfg)
    marg = b_aug - base_book
    fwd = forward_mask(ts, proposal) & np.isfinite(marg)
    assert ev.n_forward_bars == int(fwd.sum())
    assert ev.forward_sharpe == pytest.approx(_ann_sharpe(marg[fwd], cfg.periods_per_year))
    # the reported forward window is entirely post-proposal
    assert pd.Timestamp(ev.forward_start_ts) > pd.Timestamp(proposal)


def test_forward_evidence_none_or_empty_before_any_forward_data() -> None:
    """A proposal at the very end of the panel has no forward bar yet — zero-evidence, not a crash."""
    T = 200
    panel, (base, ts) = _panel(T), _base(T)
    proposal = _iso(float(ts[-1]))                       # nothing strictly after the last bar
    ev = _forward_evidence(panel, base, ts, proposal, FitnessConfig(embargo=10))
    assert ev is not None and ev.n_forward_bars == 0


# ============================================================ criterion loader (no hardcoded gates) ====

def test_load_incubation_criterion_from_yaml() -> None:
    crit = load_incubation_criterion(LOCKBOX_GATES)
    assert crit.min_forward_bars == 63 and crit.min_forward_sharpe == pytest.approx(0.30)


def test_incubation_criterion_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        IncubationCriterion(min_forward_bars=1, min_forward_sharpe=0.0)
    with pytest.raises(ValueError):
        IncubationCriterion(min_forward_bars=10, min_forward_sharpe=float("nan"))


# ============================================================ verdict state machine (advance) ====

def _entry(**kw) -> LockboxEntry:
    d = dict(candidate_hash="c1", substrate_id="syn", formula=OVERLAY_FORMULA,
             candidate_type="overlay", crucible_version="crucible-v2.5", gates_hash="g",
             proposal_ts="2020-01-01T00:00:00+00:00", enrolled_tick_ts="2020-01-01T00:00:00+00:00",
             data_snapshot_hash="s", min_forward_bars=10, min_forward_sharpe=0.30)
    d.update(kw)
    return LockboxEntry(**d)


def test_advance_incubating_before_horizon_is_not_eligible() -> None:
    e = _entry()
    out = advance(e, ForwardEvidence(n_forward_bars=4, forward_sharpe=99.0,
                                     forward_start_ts="a", forward_end_ts="b"), "t2")
    assert out.status == STATUS_INCUBATING and not out.eligible_for_human_gate
    assert out.n_forward_bars == 4 and out.verdict_tick_ts is None     # no verdict below horizon


def test_advance_clears_at_horizon_when_sharpe_passes() -> None:
    e = _entry()
    out = advance(e, ForwardEvidence(10, 0.40, "a", "b"), "t9")        # >= 10 bars, >= 0.30 sharpe
    assert out.status == STATUS_CLEARED and out.eligible_for_human_gate
    assert out.verdict_tick_ts == "t9"


def test_advance_rejects_at_horizon_when_sharpe_fails() -> None:
    e = _entry()
    out = advance(e, ForwardEvidence(12, 0.10, "a", "b"), "t9")        # horizon reached, sharpe short
    assert out.status == STATUS_REJECTED and not out.eligible_for_human_gate
    assert out.verdict_tick_ts == "t9"


def test_advance_terminal_entry_is_never_rejudged() -> None:
    """Anti-peeking: once terminal, a later (even glowing) evidence pass cannot flip the verdict."""
    cleared = _entry(status=STATUS_CLEARED, forward_sharpe=0.5, n_forward_bars=10,
                     verdict_tick_ts="t9")
    assert advance(cleared, ForwardEvidence(500, -5.0, "a", "b"), "t50") == cleared
    rejected = _entry(status=STATUS_REJECTED, forward_sharpe=-0.5, n_forward_bars=10,
                      verdict_tick_ts="t9")
    assert advance(rejected, ForwardEvidence(500, 5.0, "a", "b"), "t50") == rejected


# ============================================================ NOW-8: incubation seams (C8-04/05/06) ====

def test_advance_stamps_verdict_snapshot_hash_at_terminal() -> None:
    """C8-06: the panel snapshot the TERMINAL forward Sharpe used is stamped at the verdict; a pass
    below the horizon does not stamp one (nothing terminal to reproduce yet)."""
    cleared = advance(_entry(), ForwardEvidence(10, 0.40, "a", "b"), "t9", snapshot_hash="snap-term")
    assert cleared.status == STATUS_CLEARED and cleared.verdict_snapshot_hash == "snap-term"
    below = advance(_entry(), ForwardEvidence(4, 9.0, "a", "b"), "t2", snapshot_hash="snap-x")
    assert below.status == STATUS_INCUBATING and below.verdict_snapshot_hash is None


def test_advance_snapshot_hash_is_optional_backcompat() -> None:
    """advance() without a snapshot_hash still works (default None) — keeps every existing caller valid."""
    out = advance(_entry(), ForwardEvidence(10, 0.40, "a", "b"), "t9")
    assert out.status == STATUS_CLEARED and out.verdict_snapshot_hash is None


def test_lockbox_record_stall_and_error_counts(tmp_path) -> None:
    """C8-04/05: durable per-entry stall/error counters distinguish a silently-stalled entry from
    healthy accrual across nights; a terminal entry is a no-op."""
    lb = Lockbox(tmp_path / "lb.db")
    lb._write(_entry(candidate_hash="c1"))
    lb.record_stall("c1", "t2")
    lb.record_stall("c1", "t3")
    lb.record_error("c1", "t4")
    e = lb.get("c1")
    assert e.n_stalled_passes == 2 and e.n_error_passes == 1 and e.last_tick_ts == "t4"
    assert e.status == STATUS_INCUBATING                       # counting a stall/error never renders a verdict
    lb._write(_entry(candidate_hash="done", status=STATUS_CLEARED, verdict_tick_ts="t1"))
    lb.record_stall("done", "t9")                              # terminal → no-op
    assert lb.get("done").n_stalled_passes == 0
    lb.close()


def test_lockbox_migrates_old_schema(tmp_path) -> None:
    """C8-04/05/06: the new columns are added by an idempotent migration to a pre-existing lockbox
    table (CREATE-IF-NOT-EXISTS never alters an existing table); a new-schema entry round-trips."""
    import sqlite3
    db = tmp_path / "lb.db"
    conn = sqlite3.connect(str(db))                            # OLD schema: no NOW-8 columns
    conn.execute("CREATE TABLE lockbox_entries (candidate_hash TEXT PRIMARY KEY, substrate_id TEXT, "
                 "formula TEXT, candidate_type TEXT, crucible_version TEXT, gates_hash TEXT, "
                 "proposal_ts TEXT, enrolled_tick_ts TEXT, data_snapshot_hash TEXT, "
                 "min_forward_bars INTEGER, min_forward_sharpe REAL, status TEXT, forward_sharpe REAL, "
                 "n_forward_bars INTEGER, forward_start_ts TEXT, forward_end_ts TEXT, "
                 "last_tick_ts TEXT, verdict_tick_ts TEXT)")
    conn.commit()
    conn.close()
    lb = Lockbox(db)                                           # migrates on open
    cols = {r["name"] for r in lb._conn.execute("PRAGMA table_info(lockbox_entries)")}
    assert {"verdict_snapshot_hash", "n_stalled_passes", "n_error_passes"} <= cols
    lb._write(_entry(candidate_hash="c1"))                     # a full new-schema entry round-trips
    assert lb.get("c1").n_stalled_passes == 0
    lb.close()
    Lockbox(db).close()                                        # idempotent second open


def test_incubate_active_isolates_poison_and_counts(tmp_path, monkeypatch) -> None:
    """C8-04/05: _incubate_active never propagates a forward_evidence failure — a raise is counted as
    an error pass (entry stays INCUBATING), a None return is counted as a stall. The tick survives a
    poison-pill formula on a drifted panel instead of crashing every future tick for the substrate."""
    from types import SimpleNamespace

    from finrl_pro_ds.crucible.orchestrator import orchestrator as orch
    lb = Lockbox(tmp_path / "lb.db")
    lb._write(_entry(candidate_hash="boom"))          # active_entries orders by hash: boom < stall
    lb._write(_entry(candidate_hash="stall"))
    sub = SimpleNamespace(lockbox=lb, substrate_id="syn", cfg=None,
                          evolve_kwargs=dict(hold_horizon=21, cost_bps=0.001, ls_min_names=6))
    prepared = SimpleNamespace(panel=None, base_returns=None, timestamps=None, snapshot_hash="snap-x")

    calls = {"n": 0}

    def fake_fe(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("poison on a drifted panel")   # first (boom) → error
        return None                                            # second (stall) → degenerate

    monkeypatch.setattr(orch, "forward_evidence", fake_fe)
    result = orch._incubate_active(sub, prepared, "t1")        # MUST NOT raise
    assert result.n_errors == 1 and result.n_stalled == 1 and result.touched == []
    assert lb.get("boom").n_error_passes == 1 and lb.get("boom").status == STATUS_INCUBATING
    assert lb.get("stall").n_stalled_passes == 1
    lb.close()


# ============================================================ Lockbox store ====

def test_lockbox_enroll_idempotent_and_pins_criterion(tmp_path) -> None:
    lb = Lockbox(tmp_path / "lock.db")
    crit = IncubationCriterion(min_forward_bars=10, min_forward_sharpe=0.30)
    card = _make_card("2020-06-01T00:00:00+00:00")
    e1 = lb.enroll(card, crit, substrate_id="syn", tick_ts="2020-06-01T00:00:00+00:00")
    # accrue some state, then re-enroll the SAME candidate — must NOT reset the entry (idempotent)
    lb.record_incubation(card.candidate_hash, ForwardEvidence(4, 0.1, "a", "b"), "t2")
    e2 = lb.enroll(card, IncubationCriterion(min_forward_bars=999, min_forward_sharpe=9.9),
                   substrate_id="syn", tick_ts="t2")
    assert e2.min_forward_bars == 10 and e2.n_forward_bars == 4     # first criterion + accrual survive
    assert e1.proposal_ts == "2020-06-01T00:00:00+00:00"


def test_lockbox_eligible_lists_only_cleared(tmp_path) -> None:
    lb = Lockbox(tmp_path / "lock.db")
    crit = IncubationCriterion(min_forward_bars=5, min_forward_sharpe=0.0)
    for h, bars, sharpe in [("clear", 6, 1.0), ("reject", 6, -1.0), ("incub", 2, 5.0)]:
        lb.enroll(_make_card("2020-06-01T00:00:00+00:00", h), crit, substrate_id="syn", tick_ts="t0")
        lb.record_incubation(h, ForwardEvidence(bars, sharpe, "a", "b"), "t1")
    assert [e.candidate_hash for e in lb.eligible("syn")] == ["clear"]
    assert {e.candidate_hash for e in lb.active_entries("syn")} == {"incub"}


# ============================================================ orchestrator wiring (EXIT GATE) ====

def _grow_substrate(tmp_path, lockbox, criterion, *, seed=0):
    """A substrate whose panel GROWS each tick (forward data arriving); reads a mutable ``state``."""
    state = {"n_bars": 0}
    T_full = 320
    panel_full, (base_full, ts_full) = _panel(T_full, seed), _base(T_full, seed)

    def prepare() -> PreparedSubstrate:
        k = state["n_bars"]
        panel = panel_full.truncated(k - 1)
        base = {kk: vv[:k] for kk, vv in base_full.items()}
        ts = ts_full[:k]
        return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash=f"snap-{k}")

    sub = Substrate(substrate_id="syn", prepare=prepare, ledger=TrialLedger(tmp_path / "led.db"),
                    cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK), lockbox=lockbox,
                    incubation_criterion=criterion)
    return sub, state, ts_full


def _seed_survivor(lockbox, ts_full, p_idx, criterion):
    """Seed the lockbox with a PROMISING survivor whose proposal sits at an interior bar."""
    proposal = _iso(float(ts_full[p_idx]))
    card = _make_card(proposal, "seeded-survivor")
    lockbox.enroll(card, criterion, substrate_id="syn", tick_ts=proposal)
    return card, proposal


@pytest.mark.slow
def test_orchestrator_incubates_forward_and_clears_at_horizon(tmp_path) -> None:
    """EXIT GATE: a seeded survivor enrolls, accrues forward OOS evidence as the panel extends past
    its proposal, is NOT human-eligible before the horizon, and CLEARS at the horizon."""
    store = OrchestratorStore(tmp_path / "orch.db")
    lb = Lockbox(tmp_path / "lock.db")
    P_IDX = 250
    crit = IncubationCriterion(min_forward_bars=20, min_forward_sharpe=-1e9)   # clears once at horizon
    sub, state, ts_full = _grow_substrate(tmp_path, lb, crit)
    card, proposal = _seed_survivor(lb, ts_full, P_IDX, crit)

    # night 1: panel reaches only a few bars past the proposal -> still INCUBATING, NOT eligible
    state["n_bars"] = P_IDX + 8
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES, tick_ts="2026-07-02T00:00:00+00:00")
    e = lb.get(card.candidate_hash)
    assert e.status == STATUS_INCUBATING and not e.eligible_for_human_gate
    assert 0 < e.n_forward_bars < crit.min_forward_bars
    assert lb.eligible("syn") == []                                  # no human gate before clearing

    # night 2: panel extends well past the horizon -> verdict rendered, CLEARED, now eligible
    state["n_bars"] = 320
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES, tick_ts="2026-07-20T00:00:00+00:00")
    e = lb.get(card.candidate_hash)
    assert e.status == STATUS_CLEARED and e.eligible_for_human_gate
    assert e.n_forward_bars >= crit.min_forward_bars
    assert [x.candidate_hash for x in lb.eligible("syn")] == ["seeded-survivor"]
    # the survivor's card reflects the cleared state verbatim
    assert updated_card(card, e).eligible_for_human_gate is True


@pytest.mark.slow
def test_orchestrator_rejects_survivor_that_fails_forward(tmp_path) -> None:
    """A survivor whose forward Sharpe cannot clear an (impossibly high) floor is REJECTED at the
    horizon — terminal-dead, never human-eligible."""
    store = OrchestratorStore(tmp_path / "orch.db")
    lb = Lockbox(tmp_path / "lock.db")
    P_IDX = 250
    crit = IncubationCriterion(min_forward_bars=20, min_forward_sharpe=1e9)    # nothing can clear this
    sub, state, ts_full = _grow_substrate(tmp_path, lb, crit)
    card, _ = _seed_survivor(lb, ts_full, P_IDX, crit)

    state["n_bars"] = 320
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES, tick_ts="2026-07-20T00:00:00+00:00")
    e = lb.get(card.candidate_hash)
    assert e.status == STATUS_REJECTED and not e.eligible_for_human_gate
    assert lb.eligible("syn") == []


@pytest.mark.slow
def test_substrate_without_lockbox_is_p3_noop(tmp_path) -> None:
    """Backward-compat: a substrate with no lockbox does not incubate and reports empty lockbox
    fields — the P3 behavior is byte-identical (the reproducibility gate is protected)."""
    store = OrchestratorStore(tmp_path / "orch.db")
    T = 200
    panel, (base, ts) = _panel(T), _base(T)

    def prepare() -> PreparedSubstrate:
        return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash="snap-x")

    sub = Substrate(substrate_id="syn", prepare=prepare, ledger=TrialLedger(tmp_path / "led.db"),
                    cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK))    # no lockbox
    r = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                              tick_ts="2026-07-02T00:00:00+00:00")
    o = r.outcomes[0]
    assert o.n_enrolled == 0 and o.n_incubating == 0 and o.lockbox_entries == []
