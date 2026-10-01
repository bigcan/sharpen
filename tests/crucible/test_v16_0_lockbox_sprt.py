"""crucible-v16.0: the lockbox decides on the forward Sharpe DIFFERENCE with Wald's SPRT.

The pre-v16 rule looked once, at 63 bars, at the Sharpe of ``b_aug − b_base`` (the Tier-C-sealed
substitution residual, negative for a genuine diversifier) against 0.30 — a coin flip (P(clear) 0.44 at
zero edge). Each test below FAILS on that rule.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from sharpen.crucible import IncubationCriterion, Lockbox, load_incubation_criterion
from sharpen.crucible.agentic.card import DiscoveryCard
from sharpen.crucible.lockbox.incubation import ForwardEvidence, _sprt_evidence
from sharpen.crucible.lockbox.lockbox import (
    STATUS_CLEARED,
    STATUS_INCONCLUSIVE,
    STATUS_INCUBATING,
    STATUS_REJECTED,
    LockboxEntry,
    advance,
    updated_card,
)
from sharpen.signals.eval_harness import _ann_sharpe
from sharpen.signals.generation.fitness import FitnessConfig, augmented_book

LOCKBOX_GATES = "configs/crucible_lockbox.gates.yaml"
SPRT = IncubationCriterion(min_forward_bars=63, min_forward_sharpe=0.30, test="sprt",
                           max_forward_bars=756, block_bars=21, calib_bars=504, delta_sr_h1=0.30,
                           alpha=0.10, beta=0.20)


def _streams(t: int, sr_cand: float, *, corr: float = 0.0, seed: int = 3):
    """Two base sleeves at annual SR 0.5 and a candidate at annual ``sr_cand``, ``corr`` to sleeve 1."""
    rng = np.random.default_rng(seed)
    vol = 0.01
    z1, z2, z3 = rng.standard_normal((3, t))
    s1 = vol * z1 + vol * 0.5 / math.sqrt(252)
    s2 = vol * z2 + vol * 0.5 / math.sqrt(252)
    zc = corr * z1 + math.sqrt(1 - corr * corr) * z3
    cand = vol * zc + vol * sr_cand / math.sqrt(252)
    ts = pd.date_range("2000-01-03", periods=t, freq="B").view("int64").astype(np.float64) / 1e9
    return {"s1": s1, "s2": s2}, cand, ts


# ------------------------------------------------------------------ the shipped gates
def test_shipped_lockbox_gates_select_the_sprt_rule() -> None:
    crit = load_incubation_criterion(LOCKBOX_GATES)
    assert crit.test == "sprt"
    assert (crit.alpha, crit.beta, crit.delta_sr_h1) == (0.10, 0.20, 0.30)
    assert crit.max_forward_bars >= crit.min_forward_bars


# ------------------------------------------------------------------ the statistic
def test_sprt_evidence_is_the_forward_sharpe_difference_of_the_augmented_book() -> None:
    """White-box: forward ΔSR is SR(b_aug) − SR(b_base) on the forward bars of the SAME augmented book
    the corrected contract scores, and the LLR is Wald's on block sums of the vol-standardized
    difference, with vols and noise pinned from pre-proposal bars."""
    base, cand, ts = _streams(1800, 0.8)
    cfg = FitnessConfig(embargo=10)
    fwd = np.arange(ts.size) >= 1200
    ev = _sprt_evidence(base, cand, ts, cfg, fwd, SPRT)
    b_base, b_aug, _ = augmented_book(base, cand, ts, cfg)
    ok = np.isfinite(b_base) & np.isfinite(b_aug)
    post = np.flatnonzero(ok & fwd)
    assert ev["forward_delta_sr"] == pytest.approx(
        _ann_sharpe(b_aug[post], cfg.periods_per_year) - _ann_sharpe(b_base[post], cfg.periods_per_year))
    pre = np.flatnonzero(ok & ~fwd)[-SPRT.calib_bars:]
    d = b_aug / b_aug[pre].std(ddof=1) - b_base / b_base[pre].std(ddof=1)
    s = d[pre].reshape(-1, 21).sum(axis=1).std(ddof=1)
    fb = d[post[: (post.size // 21) * 21]].reshape(-1, 21).sum(axis=1)
    theta = 0.30 / math.sqrt(cfg.periods_per_year) * 21 / s
    assert ev["n_blocks"] == fb.size
    assert ev["sprt_llr"] == pytest.approx(float(np.sum(theta * fb / s - theta * theta / 2)))


def test_a_genuine_diversifier_accumulates_evidence_for_the_alternative() -> None:
    """A candidate that raises the book's Sharpe well past H1 (independent, SR 1.5 beside two SR-0.5
    sleeves) builds LLR past the CLEARED boundary, while a candidate that lowers it builds LLR below the
    REJECTED one — the sign the gate certifies. The pre-v16 statistic (the marginal-residual Sharpe) is
    SMALLER than the ΔSR it was standing in for. (A diversifier at ΔSR ~0.2, inside the indifference
    zone below H1 = 0.30, correctly accumulates almost nothing either way.)"""
    cfg = FitnessConfig(embargo=10)
    base, good, ts = _streams(504 + 2520, 1.5, seed=11)
    fwd = np.arange(ts.size) >= 504
    ev_good = _sprt_evidence(base, good, ts, cfg, fwd, SPRT)
    assert ev_good["forward_delta_sr"] > 0.30 and ev_good["sprt_llr"] > SPRT.log_upper
    b_base, b_aug, _ = augmented_book(base, good, ts, cfg)
    marg = (b_aug - b_base)[fwd & np.isfinite(b_aug) & np.isfinite(b_base)]
    assert _ann_sharpe(marg, cfg.periods_per_year) < ev_good["forward_delta_sr"]
    base, bad, ts = _streams(504 + 2520, -0.5, corr=0.3, seed=12)
    ev_bad = _sprt_evidence(base, bad, ts, cfg, fwd, SPRT)
    assert ev_bad["forward_delta_sr"] < 0 and ev_bad["sprt_llr"] < SPRT.log_lower


def test_too_short_a_pre_proposal_window_yields_no_sprt_evidence() -> None:
    base, cand, ts = _streams(300, 0.8)
    assert _sprt_evidence(base, cand, ts, FitnessConfig(embargo=10), np.arange(300) >= 30, SPRT) == {}


# ------------------------------------------------------------------ the state machine
def _entry(**kw) -> LockboxEntry:
    d = dict(candidate_hash="c1", substrate_id="syn", formula="macro:regime", candidate_type="overlay",
             crucible_version="crucible-v16.0", gates_hash="g", proposal_ts="2020-01-01T00:00:00+00:00",
             enrolled_tick_ts="2020-01-01T00:00:00+00:00", data_snapshot_hash="s",
             min_forward_bars=63, min_forward_sharpe=0.30, test="sprt", max_forward_bars=756,
             block_bars=21, calib_bars=504, delta_sr_h1=0.30, alpha_sprt=0.10, beta_sprt=0.20)
    d.update(kw)
    return LockboxEntry(**d)


def _ev(bars: int, llr: float, sharpe: float = 0.0) -> ForwardEvidence:
    return ForwardEvidence(bars, sharpe, "a", "b", forward_delta_sr=0.1, sprt_llr=llr,
                           n_blocks=bars // 21)


def test_sprt_verdicts_follow_the_wald_boundaries() -> None:
    up, lo = SPRT.log_upper, SPRT.log_lower
    assert advance(_entry(), _ev(42, up + 5), "t").status == STATUS_INCUBATING      # before min bars
    cleared = advance(_entry(), _ev(84, up + 0.01, sharpe=-2.0), "t")               # marginal SR < 0.30
    assert cleared.status == STATUS_CLEARED and cleared.eligible_for_human_gate
    assert advance(_entry(), _ev(84, lo - 0.01, sharpe=5.0), "t").status == STATUS_REJECTED
    assert advance(_entry(), _ev(84, 0.0), "t").status == STATUS_INCUBATING
    capped = advance(_entry(), _ev(756, 0.0), "t", snapshot_hash="snap")
    assert capped.status == STATUS_INCONCLUSIVE and not capped.eligible_for_human_gate
    assert capped.verdict_snapshot_hash == "snap" and capped.is_terminal
    assert advance(capped, _ev(900, up + 50), "t2") == capped                       # never re-judged


def test_a_fixed_entry_keeps_the_rule_it_was_enrolled_under() -> None:
    old = _entry(test="fixed", max_forward_bars=0)
    assert advance(old, _ev(84, -99.0, sharpe=0.5), "t").status == STATUS_CLEARED   # marginal SR rule


@pytest.mark.slow
def test_orchestrator_accrues_sprt_evidence_for_an_sprt_entry(tmp_path) -> None:
    from test_lockbox import GATES, _grow_substrate, _seed_survivor

    from sharpen.crucible import OrchestratorStore, run_orchestrator_tick

    crit = IncubationCriterion(min_forward_bars=21, min_forward_sharpe=0.30, test="sprt",
                               max_forward_bars=756, block_bars=21, calib_bars=210,
                               delta_sr_h1=0.30, alpha=0.10, beta=0.20)
    lb = Lockbox(tmp_path / "lock.db")
    sub, state, ts_full = _grow_substrate(tmp_path, lb, crit)
    card, _ = _seed_survivor(lb, ts_full, 250, crit)
    state["n_bars"] = 320
    run_orchestrator_tick(substrates=[sub], store=OrchestratorStore(tmp_path / "orch.db"),
                          gates_path=GATES, tick_ts="2026-07-20T00:00:00+00:00")
    e = lb.get(card.candidate_hash)
    assert e.test == "sprt" and e.n_blocks >= 3
    assert e.sprt_llr is not None and math.isfinite(e.sprt_llr)
    assert e.forward_delta_sr is not None


def test_enrollment_pins_the_sprt_criterion_and_old_stores_migrate_to_fixed(tmp_path) -> None:
    card = DiscoveryCard(candidate_hash="h1", formula="macro:regime", candidate_type="overlay",
                         crucible_version="crucible-v16.0", gates_hash="g",
                         proposal_ts="2020-01-01T00:00:00+00:00")
    with Lockbox(tmp_path / "lb.db") as lb:
        e = lb.enroll(card, SPRT, substrate_id="syn", tick_ts="2020-01-01T00:00:00+00:00")
        assert e.criterion == SPRT
        got = lb.record_incubation("h1", _ev(84, SPRT.log_upper + 1), "t1")
        assert got.status == STATUS_CLEARED and lb.get("h1").sprt_llr == pytest.approx(got.sprt_llr)
        assert updated_card(card, got).incubation_forward_delta_sr == pytest.approx(0.1)
    # a pre-v16 store (no v16 columns) migrates to the pre-v16 rule
    con = sqlite3.connect(tmp_path / "old.db")
    con.execute("CREATE TABLE lockbox_entries (candidate_hash TEXT PRIMARY KEY, substrate_id TEXT NOT NULL,"
                " formula TEXT NOT NULL, candidate_type TEXT NOT NULL, crucible_version TEXT NOT NULL,"
                " gates_hash TEXT, proposal_ts TEXT NOT NULL, enrolled_tick_ts TEXT NOT NULL,"
                " data_snapshot_hash TEXT, min_forward_bars INTEGER NOT NULL, min_forward_sharpe REAL NOT NULL,"
                " status TEXT NOT NULL, forward_sharpe REAL, n_forward_bars INTEGER NOT NULL DEFAULT 0,"
                " forward_start_ts TEXT, forward_end_ts TEXT, last_tick_ts TEXT, verdict_tick_ts TEXT)")
    con.execute("INSERT INTO lockbox_entries (candidate_hash, substrate_id, formula, candidate_type,"
                " crucible_version, proposal_ts, enrolled_tick_ts, min_forward_bars, min_forward_sharpe,"
                " status) VALUES ('old', 'syn', 'macro:regime', 'overlay', 'crucible-v14.0',"
                " '2020-01-01', '2020-01-01', 63, 0.3, 'INCUBATING')")
    con.commit()
    con.close()
    with Lockbox(tmp_path / "old.db") as lb:
        assert lb.get("old").test == "fixed" and lb.get("old").criterion.test == "fixed"
