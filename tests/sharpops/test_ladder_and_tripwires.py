"""SharpOps promotion-ladder statistics and artifact tripwires (sharpen/sharpops/).

Each test pins BOTH directions, because a check that never fires and a check that always fires
both read as "green" (feedback: one-directional testing). Nulls must stay quiet at their nominal
level; planted edges and planted artifacts must be caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.sharpops import ladder as L
from sharpen.sharpops import tripwires as T


def _series(r, start="2020-01-01"):
    return pd.Series(r, index=pd.bdate_range(start, periods=len(r)))


# --------------------------------------------------------------------------- pooled OOS
def test_pooled_oos_rejects_overlapping_folds():
    a = _series(np.zeros(30) + 0.001, "2020-01-01")
    b = _series(np.zeros(30) + 0.001, "2020-01-20")
    with pytest.raises(ValueError, match="overlap"):
        L.pooled_oos_sharpe([a, b])


def test_pooled_oos_pools_disjoint_folds():
    rng = np.random.default_rng(0)
    folds = [_series(rng.normal(0.0008, 0.006, 250), f"{2010 + 2 * k}-01-01") for k in range(4)]
    out = L.pooled_oos_sharpe(folds, n_boot=500)
    assert out["n_folds"] == 4 and out["n_days"] == 1000
    assert len(out["fold_sharpes_diagnostic"]) == 4
    lo, hi = out["ci95"]
    assert lo < out["sharpe_ann"] < hi
    assert 0.0 <= out["psr_vs_0"] <= 1.0


def test_rung_test_fails_closed():
    assert L.rung_test_passes(0.81, 0.20)
    assert not L.rung_test_passes(0.79, 0.20)
    assert not L.rung_test_passes(None, 0.20)
    assert not L.rung_test_passes(float("nan"), 0.20)


def test_pooled_psr_is_calibrated_under_the_null():
    """At true SR 0 the paper rung (alpha 0.20) must pass ~20% of the time, not more."""
    oc = L.operating_characteristic(
        lambda r: L.rung_test_passes(L.pooled_oos_sharpe([_series(r)], n_boot=10)["psr_vs_0"], 0.20),
        n_days=500, sharpes=[], n_sims=400, target_sharpe=1.0)
    lo, hi = oc["false_pass_at_sr0"]["wilson95"]
    assert lo <= 0.20 <= hi + 0.03
    assert oc["power_at_target"]["pass_rate"] > oc["false_pass_at_sr0"]["pass_rate"]


# --------------------------------------------------------------------------- e-process
def test_eprocess_is_valid_under_the_null():
    """Ville: P(ever crossing 1/alpha) <= alpha under a zero-mean record, checked daily."""
    rng = np.random.default_rng(3)
    hits = sum(L.betting_eprocess(rng.normal(0.0, 0.01, 750), bound=0.10, alpha=0.10)["rejected_h0"]
               for _ in range(300))
    lo, _ = T.wilson(hits, 300)
    assert lo <= 0.10


def test_eprocess_rejects_a_real_edge():
    r = np.random.default_rng(4).normal(0.004, 0.01, 750)      # SR ~6: must reject quickly
    out = L.betting_eprocess(r, bound=0.10, alpha=0.10)
    assert out["rejected_h0"] and out["first_rejection_day"] < 750


def test_eprocess_power_at_the_registered_bound():
    """Reachability pin (Math check 2026-09-30): at the registered 5% bound a SR-2 book is
    rejected within 3 years most of the time; the old 10% bound halved the bet."""
    sims = np.clip(L.planted_returns(2.0, 756, 120, seed=41), -0.0499, None)
    p05 = np.mean([L.betting_eprocess(x, bound=0.05, alpha=0.10)["rejected_h0"] for x in sims])
    p10 = np.mean([L.betting_eprocess(x, bound=0.10, alpha=0.10)["rejected_h0"] for x in sims])
    assert p05 > 0.7 and p05 > p10


def test_eprocess_never_rejects_a_losing_book():
    r = np.random.default_rng(5).normal(-0.001, 0.01, 750)
    assert not L.betting_eprocess(r, bound=0.10, alpha=0.10)["rejected_h0"]


def test_eprocess_fails_closed_on_a_loss_beyond_the_bound():
    with pytest.raises(ValueError, match="void"):
        L.betting_eprocess([0.01, -0.2, 0.01], bound=0.10, alpha=0.10)


def test_eprocess_clips_gains_conservatively():
    big = L.betting_eprocess([0.01] * 5 + [0.5] + [0.01] * 5, bound=0.10, alpha=0.10)
    capped = L.betting_eprocess([0.01] * 5 + [0.10] + [0.01] * 5, bound=0.10, alpha=0.10)
    assert big["n_clipped_gains"] == 1 and big["e_value"] == pytest.approx(capped["e_value"])


def test_eprocess_bet_is_predictable():
    """Day t's bet may use only days < t: changing the last return cannot change the path
    before it."""
    # noisy book: the bet stays BELOW its cap, so a look-ahead bet would visibly differ
    base = list(np.clip(np.random.default_rng(14).normal(0.001, 0.03, 60), -0.095, 0.095))
    a = L.betting_eprocess(base[:-1] + [-0.09], bound=0.1, alpha=0.1, return_path=True)
    b = L.betting_eprocess(base[:-1] + [0.09], bound=0.1, alpha=0.1, return_path=True)
    assert 0.0 < a["bets"][-1] < 0.5                      # last bet unsaturated: the test has teeth
    assert a["bets"] == b["bets"]                         # every bet, including day 50's
    assert a["path"][:-1] == b["path"][:-1] and a["path"][-1] != b["path"][-1]
    assert all(0.0 <= lam <= 0.5 for lam in a["bets"]) and a["bets"][:2] == [0.0, 0.0]


# --------------------------------------------------------------------------- overlay gate
def test_non_inferiority_both_directions():
    rng = np.random.default_rng(7)
    base = rng.normal(0.0004, 0.006, 2500)
    same = base + rng.normal(0, 0.0003, 2500)                # ~identical book
    worse = base - 0.0006                                     # ~1.6 SR worse
    assert L.non_inferiority(same, base, margin=0.10, alpha=0.05, n_boot=800)["non_inferior"]
    assert not L.non_inferiority(worse, base, margin=0.10, alpha=0.05, n_boot=800)["non_inferior"]


def test_superiority_on_a_lower_is_better_secondary():
    rng = np.random.default_rng(8)
    base = rng.normal(0.0004, 0.006, 1500)
    calmer = base * 0.5                                       # same Sharpe, half the drawdown
    mdd = lambda r: -float(np.min(np.cumsum(r) - np.maximum.accumulate(np.cumsum(r))))  # noqa: E731
    assert L.superiority(calmer, base, mdd, alpha=0.05, lower_is_better=True, n_boot=500)["superior"]
    assert not L.superiority(base, calmer, mdd, alpha=0.05, lower_is_better=True, n_boot=500)["superior"]


def test_paired_bootstrap_needs_aligned_series():
    with pytest.raises(ValueError):
        L.paired_block_bootstrap(np.zeros(100), np.zeros(99), np.mean)


# --------------------------------------------------------------------------- planted signals
def test_planted_returns_have_the_declared_sharpe():
    sims = L.planted_returns(1.0, 5000, 200, vol_ann=0.10, ar1=0.2)
    sr = sims.mean(axis=1) / sims.std(axis=1, ddof=1) * np.sqrt(252)
    assert np.mean(sr) == pytest.approx(1.0, abs=0.08)
    assert np.mean(sims.std(axis=1) * np.sqrt(252)) == pytest.approx(0.10, rel=0.03)


def test_operating_characteristic_exposes_a_bar_pf_floor():
    """The audit's point, executable: a PF >= 1.10 floor on 4 months of DAILY returns has
    almost no power at SR 1 while a PSR gate at the same data has some."""
    def pf(r):
        pos, neg = r[r > 0].sum(), -r[r < 0].sum()
        return pos / neg if neg > 0 else np.inf
    oc = L.operating_characteristic(lambda r: pf(r) >= 1.10, n_days=84, sharpes=[], n_sims=300,
                                    target_sharpe=1.0)
    assert oc["false_pass_at_sr0"]["pass_rate"] < 0.5
    assert set(oc["grid"]) == {"0", "1"}


# --------------------------------------------------------------------------- tripwires
def test_plausibility_ceiling():
    assert T.plausibility_ceiling(1.2, 3.0)["pass"]
    assert not T.plausibility_ceiling(4.1, 3.0)["pass"]
    assert not T.plausibility_ceiling(float("nan"), 3.0)["pass"]


def test_cost_sign():
    assert T.cost_sign(0.8, 0.6)["pass"]
    assert not T.cost_sign(0.6, 0.8)["pass"]


def test_seed_tripwire_catches_seed01_and_nonreproduction():
    same = [0.01, -0.02, 0.03]
    assert not T.seed_tripwire({1: same, 2: list(same)})["pass"]                 # seeds never reached env
    assert T.seed_tripwire({1: same, 2: [0.02, 0.0, 0.01]}, rerun=(same, list(same)))["pass"]
    assert not T.seed_tripwire({1: same, 2: [0.02, 0.0, 0.01]}, rerun=(same, [0.01, -0.02, 0.031]))["pass"]
    assert not T.seed_tripwire({1: same})["pass"]


def _momentum(r, lag):                   # position from past returns only (lag >= 1: causal)
    sig = np.sign(pd.Series(r).rolling(20).sum().shift(lag).fillna(0).to_numpy())
    return sig * r


def test_shuffle_placebo_catches_look_ahead():
    r = np.random.default_rng(9).normal(0, 0.01, 1000)
    peek = lambda x: np.sign(x) * x                                            # noqa: E731 reads r_t
    out = T.shuffle_placebo(peek, r, n_perm=50, max_null_sharpe=0.3, alpha=0.05)
    assert not out["pass"] and not out["quiet"]                                # earns on noise
    honest = T.shuffle_placebo(lambda x: _momentum(x, 1), r, n_perm=50, max_null_sharpe=0.3, alpha=0.05)
    assert honest["quiet"]                                                     # no edge, but no artifact


def test_circular_shift_null_both_directions():
    rng = np.random.default_rng(10)
    n = 2000
    sig = np.sign(rng.standard_normal(n))
    edge = 0.002 * sig + rng.normal(0, 0.01, n)          # returns follow the signal's timing
    assert T.circular_shift_null(sig, edge, n_shifts=200, min_shift=21, alpha=0.05)["pass"]
    # no timing edge: the false-pass RATE over many noise records stays near alpha
    hits = sum(T.circular_shift_null(sig, rng.normal(0, 0.01, n), n_shifts=100, min_shift=21,
                                     alpha=0.05, seed=k)["pass"] for k in range(200))
    assert T.wilson(hits, 200)[0] <= 0.05


def test_selection_embargo_reproduces_the_sg1_btc_breach():
    hpo_val = [("2025-07-01", "2025-10-01")]
    wf_tests = [("2025-08-01", "2025-09-01"), ("2025-09-01", "2025-10-01"), ("2025-10-01", "2025-11-01")]
    out = T.selection_embargo(hpo_val, wf_tests)
    assert not out["pass"] and [h["test_fold"] for h in out["overlaps"]] == [0, 1]   # half-open: fold 2 clear
    assert T.selection_embargo(hpo_val, wf_tests[2:])["pass"]
    assert not T.selection_embargo(hpo_val, wf_tests[2:], embargo_days=5)["pass"]    # embargo widens


def test_wilson_matches_known_interval():
    lo, hi = T.wilson(14, 21)
    assert (round(lo, 3), round(hi, 3)) == (0.454, 0.828)
    assert T.wilson(0, 0) == (0.0, 1.0)
