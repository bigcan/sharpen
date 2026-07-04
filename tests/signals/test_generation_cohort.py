"""Phase-1 tests for the cohort deflation statistic (novel statistic #1) + de-correlated admission.

Design + math audit: ``docs/research/crucible_weak_signal_ensemble_spec.md`` Part 1. The binding
gate (the MC null) is tested separately in ``test_generation_cohort_mc.py``; these tests pin the
ANALYTIC ``SR*_cohort`` floor and the greedy admission that the whole cohort path rests on.

The load-bearing test is :func:`test_noise_cohort_tripwire_m9_n100` — it reproduces the ⛔ MATH
AUDIT falsification (naive ``n_trials=N`` DSR admits pure-noise cohorts at ~100% for m=9/N=100)
and asserts the corrected order-statistic ``SR*_cohort`` rejects them. If that test ever flips, the
noise-admission bug is back.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from finrl_pro_ds.signals.generation.cohort import (
    CohortConfig,
    blep_sr_star,
    cohort_dsr,
    cohort_sr_star,
    ensemble_multiplier,
    evaluate_cohort_analytic,
    expected_topm_order_stat_sum,
    greedy_decorrelated_admission,
    mean_offdiagonal_corr,
    measure_pool_diversity,
    pairwise_corr_matrix,
)
from finrl_pro_ds.signals.generation.cohort import _with_redundancy
from finrl_pro_ds.signals.generation.fitness import (
    FitnessConfig,
    _combined_book,
    _per_period_sharpe,
)


# ------------------------------------------------------------------------------------------
# Calibration anchor: SR*_cohort at m=1 == the audited BLdP SR* (MC-validated <2%)
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [50, 100, 500, 1000])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_m1_reduces_to_bldp_sr_star(n: int, sigma: float) -> None:
    """At m=1 the order-statistic SR*_cohort must equal the audited BLdP benchmark to <2% (the
    reduction that guards the single-candidate path)."""
    cohort = cohort_sr_star(sigma, m=1, n_candidates_seen=n, mean_pairwise_corr=0.0)
    bldp = blep_sr_star(sigma, n_trials=n)
    assert abs(cohort - bldp) / bldp < 0.02


@pytest.mark.parametrize("n", [50, 100, 500])
def test_m1_matches_audited_dsr_sr_star(n: int) -> None:
    """The audited ``deflated_sharpe_ratio`` computes its own ``sr_star`` from the trial pool; at
    m=1 our benchmark must agree with it to <2% on the same dispersion (σ_trials = std of pool)."""
    rng = np.random.default_rng(20260703 + n)
    pool = rng.normal(0.0, 1.0, size=n).tolist()
    sigma = float(np.std(pool, ddof=1))
    d = deflated_sharpe_ratio(
        0.5, pool, n_obs=500, skew=0.0, excess_kurt=0.0, n_trials=n, periods_per_year=1)
    assert d is not None
    cohort = cohort_sr_star(sigma, m=1, n_candidates_seen=n, mean_pairwise_corr=0.0)
    assert abs(cohort - d["sr_star"]) / d["sr_star"] < 0.02


# ------------------------------------------------------------------------------------------
# The benchmark is monotone in N (a bigger search ⇒ a higher bar) and √m at ρ̄=0
# ------------------------------------------------------------------------------------------
def test_sr_star_monotone_in_n() -> None:
    vals = [cohort_sr_star(1.0, m=4, n_candidates_seen=n, mean_pairwise_corr=0.0)
            for n in (20, 50, 100, 250, 500, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_ensemble_multiplier_endpoints() -> None:
    # uncorrelated → √m
    for m in (2, 5, 9, 12):
        assert ensemble_multiplier(m, 0.0) == pytest.approx(np.sqrt(m))
    # perfectly correlated → 1 (no diversification benefit)
    assert ensemble_multiplier(9, 1.0) == pytest.approx(1.0, abs=1e-4)
    # single member → 1
    assert ensemble_multiplier(1, 0.3) == 1.0


def test_order_stat_sum_grows_with_pool() -> None:
    a = expected_topm_order_stat_sum(4, 50)
    b = expected_topm_order_stat_sum(4, 500)
    assert b > a > 0.0


# ------------------------------------------------------------------------------------------
# THE TRIPWIRE — naive DSR admits noise at ~100% for m=9/N=100; SR*_cohort must reject it
# ------------------------------------------------------------------------------------------
def _noise_cohort_book_sharpe(rng, *, n_pool: int, m: int, t: int):
    """Simulate the real select-and-combine: draw n_pool i.i.d. noise streams, take the top-m by
    per-period Sharpe, equal-weight combine, return (book_sr, all_pool_sharpes, rho_bar, book)."""
    R = rng.normal(0.0, 1.0, size=(t, n_pool))
    srs = R.mean(axis=0) / R.std(axis=0, ddof=1)
    top = np.argsort(srs)[::-1][:m]
    book = R[:, top].mean(axis=1)
    book_sr = _per_period_sharpe(book)
    # rho_bar of the admitted m
    _, corr = pairwise_corr_matrix({str(i): R[:, j] for i, j in enumerate(top)})
    return book_sr, srs.tolist(), mean_offdiagonal_corr(corr), book


@pytest.mark.parametrize("m", [4, 9])
def test_noise_cohort_tripwire_m9_n100(m: int) -> None:
    """Reproduce the audit: over seeded pure-noise pools (N=100, T=500), the NAIVE ``n_trials=N``
    DSR passes the 0.90 gate at a high rate, while the corrected order-statistic ``SR*_cohort``
    rejects at ≈α. Wide margins (this is the qualitative fix, not an exact-% match)."""
    rng = np.random.default_rng(4242)
    n_pool, t, reps = 100, 500, 200
    naive_pass = corrected_pass = 0
    for _ in range(reps):
        book_sr, pool_srs, rho_bar, book = _noise_cohort_book_sharpe(rng, n_pool=n_pool, m=m, t=t)
        sk, ek = skewness(book.tolist()), excess_kurtosis(book.tolist())
        # NAIVE: feed n_trials=N into the audited max-of-N DSR (the blocked construction)
        d = deflated_sharpe_ratio(book_sr, pool_srs, n_obs=t, skew=sk, excess_kurt=ek,
                                  n_trials=n_pool, periods_per_year=1)
        if d is not None and d["dsr"] >= 0.90:
            naive_pass += 1
        # CORRECTED: order-statistic SR*_cohort for a top-m-of-N combination
        sigma = float(np.std(pool_srs, ddof=1))
        sr_star = cohort_sr_star(sigma, m=m, n_candidates_seen=n_pool, mean_pairwise_corr=rho_bar)
        if cohort_dsr(book_sr, sr_star, n_obs=t, skew=sk, excess_kurt=ek) >= 0.90:
            corrected_pass += 1
    naive_rate, corrected_rate = naive_pass / reps, corrected_pass / reps
    # naive is badly anti-conservative; corrected is conservative (a floor)
    assert naive_rate > 0.5, f"naive DSR should over-admit noise (got {naive_rate:.2f})"
    assert corrected_rate < 0.20, f"SR*_cohort should reject noise cohorts (got {corrected_rate:.2f})"
    assert corrected_rate < naive_rate - 0.3


# ------------------------------------------------------------------------------------------
# Greedy de-correlated admission (#2a)
# ------------------------------------------------------------------------------------------
def test_admission_collapses_duplicates_to_one() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=400)
    returns = {f"dup{i}": base.copy() for i in range(5)}   # 5 identical streams
    scores = {k: 1.0 for k in returns}
    admitted = greedy_decorrelated_admission(
        returns, scores, max_pairwise_corr=0.70, max_cohort_size=12)
    assert len(admitted) == 1


def test_admission_correlation_threshold_two_sided() -> None:
    """|corr| just ABOVE the cap rejects the second stream; well BELOW admits both. Boundary is
    inclusive per the pre-registered spec (reject iff |corr| > cap)."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=4000)
    # ~0.8 correlated > cap 0.70 ⇒ second rejected
    b_hi = 0.8 * a + np.sqrt(1 - 0.8**2) * rng.normal(size=4000)
    assert greedy_decorrelated_admission(
        {"a": a, "b": b_hi}, {"a": 2.0, "b": 1.0},
        max_pairwise_corr=0.70, max_cohort_size=12) == ("a",)
    # ~0.3 correlated < cap 0.70 ⇒ both admitted
    b_lo = 0.3 * a + np.sqrt(1 - 0.3**2) * rng.normal(size=4000)
    assert set(greedy_decorrelated_admission(
        {"a": a, "b": b_lo}, {"a": 2.0, "b": 1.0},
        max_pairwise_corr=0.70, max_cohort_size=12)) == {"a", "b"}


def test_admission_nan_score_sorts_last_deterministically() -> None:
    """A candidate with a NaN rank score (degenerate zero-variance stream) must sort LAST and not
    corrupt the deterministic ordering (A-1 regression)."""
    rng = np.random.default_rng(9)
    returns = {"good": rng.normal(size=1000), "flat": np.zeros(1000),
               "ok": rng.normal(size=1000)}
    scores = {"good": 0.5, "flat": float("nan"), "ok": 0.2}
    admitted = greedy_decorrelated_admission(
        returns, scores, max_pairwise_corr=0.30, max_cohort_size=3)
    # deterministic across repeated calls; the NaN-scored degenerate stream is not ranked ahead
    again = greedy_decorrelated_admission(
        returns, scores, max_pairwise_corr=0.30, max_cohort_size=3)
    assert admitted == again
    assert admitted[0] == "good"        # highest finite score first
    if "flat" in admitted:
        assert admitted.index("flat") == len(admitted) - 1   # NaN sorts last


def test_admission_keeps_uncorrelated_and_respects_cap() -> None:
    rng = np.random.default_rng(3)
    returns = {f"x{i}": rng.normal(size=1500) for i in range(8)}   # ~independent
    scores = {k: rng.random() for k in returns}
    admitted = greedy_decorrelated_admission(
        returns, scores, max_pairwise_corr=0.30, max_cohort_size=5)
    assert len(admitted) == 5                                       # capped
    # admitted set is genuinely de-correlated
    _, corr = pairwise_corr_matrix({n: returns[n] for n in admitted})
    off = np.abs(corr[np.triu_indices(len(admitted), 1)])
    assert (off < 0.30).all()


def test_admission_ranks_by_score() -> None:
    rng = np.random.default_rng(5)
    returns = {f"x{i}": rng.normal(size=1200) for i in range(4)}
    scores = {"x0": 0.1, "x1": 0.9, "x2": 0.2, "x3": 0.5}
    admitted = greedy_decorrelated_admission(
        returns, scores, max_pairwise_corr=0.30, max_cohort_size=1)
    assert admitted == ("x1",)   # highest score admitted first


# ------------------------------------------------------------------------------------------
# Combiner redundancy wiring (Doc 1 Part 3) — λ_r=0 must NOT perturb the live book
# ------------------------------------------------------------------------------------------
def test_cohort_combiner_lambda_r_zero_is_byte_identical_live_book() -> None:
    """The cohort path turns on the combiner redundancy down-weight; with λ_r=0 it MUST reproduce
    the shipped inverse-vol book bit-for-bit (protects tailwind_v1 / every live config)."""
    rng = np.random.default_rng(0)
    t = 400
    returns = {"tsmom": rng.normal(0.02, 1, t), "rates": rng.normal(0.01, 1, t),
               "cand": rng.normal(0.0, 1, t)}
    ts = np.int64(1_500_000_000) + np.arange(t) * 86400
    fcfg = FitnessConfig(combiner_window=63, combiner_min_periods=20)
    live = _combined_book(returns, ts, fcfg)
    cohort0 = _combined_book(returns, ts, _with_redundancy(fcfg, 0.0))
    cohorton = _combined_book(returns, ts, _with_redundancy(fcfg, 0.8))
    assert np.array_equal(live, cohort0)               # byte-identical
    assert not np.allclose(live, cohorton)             # λ_r>0 genuinely re-weights


# ------------------------------------------------------------------------------------------
# The analytic-floor wrapper — must reject noise cohorts even WITH a base book present
# (Math-audit units fix: the floor deflates the COHORT-ONLY book, not the base-dominated aug book)
# ------------------------------------------------------------------------------------------
def _ccfg(**kw) -> CohortConfig:
    base = dict(max_cohort_size=8, min_cohort_size=3, max_pairwise_corr=0.35,
                promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                combiner_redundancy_strength=0.0)
    base.update(kw)
    return CohortConfig(**base)


def test_analytic_floor_rejects_noise_cohort_with_base_present() -> None:
    """Over seeded pure-noise pools (with a real-drift base book present), the analytic pre-filter
    must reject at a low rate — proving the units fix: a base-dominated aug-book Sharpe cannot
    smuggle a noise cohort past the cohort-only SR*_cohort benchmark."""
    fcfg = FitnessConfig(n_groups=5, k_test=2, embargo=3, periods_per_year=1.0,
                         combiner_window=63, combiner_min_periods=20,
                         perf_window=63, perf_min_periods=20)
    passes = 0
    reps, t = 40, 320
    for r in range(reps):
        rng = np.random.default_rng(3300 + r)
        base = {"tsmom": rng.normal(0.03, 1, t), "rates": rng.normal(0.02, 1, t)}  # real drift
        pool = {f"c{i}": rng.normal(0.0, 1, t) for i in range(16)}                  # pure noise
        ts = np.int64(1_500_000_000) + np.arange(t) * 86400
        ev = evaluate_cohort_analytic(pool, base, ts, _ccfg(), fcfg)
        if ev is not None and ev.passes_analytic_floor:
            passes += 1
    assert passes / reps <= 0.15, f"analytic floor over-admits noise cohorts ({passes}/{reps})"


# ------------------------------------------------------------------------------------------
# Pool-diversity measurement (Doc 3 Part C — "measure data breadth first")
# ------------------------------------------------------------------------------------------
def test_measure_pool_diversity_diverse_vs_concentrated() -> None:
    """A diverse (independent) pool has low ρ̄_pool and many de-correlated admits; a concentrated
    pool (one idea in many costumes — the cont-107 failure) has ρ̄≈1 and admits ≈1."""
    rng = np.random.default_rng(0)
    t = 1500
    diverse = {f"x{i}": rng.normal(size=t) for i in range(10)}
    rep = measure_pool_diversity(diverse, max_pairwise_corr=0.35)
    assert rep.mean_abs_pairwise_corr < 0.15
    assert rep.n_decorrelated_admits >= 8

    seed = rng.normal(size=t)
    concentrated = {f"dup{i}": seed + 0.02 * rng.normal(size=t) for i in range(10)}
    rep2 = measure_pool_diversity(concentrated, max_pairwise_corr=0.35)
    assert rep2.mean_abs_pairwise_corr > 0.9
    assert rep2.n_decorrelated_admits == 1


def test_measure_pool_diversity_within_vs_cross_source() -> None:
    """With source labels, within-source |corr| (near-duplicate views of one asset — the COT-gold
    concentration) exceeds cross-source |corr| (orthogonal underlyings — what breadth buys)."""
    rng = np.random.default_rng(1)
    t = 1500
    g = rng.normal(size=t)
    e = rng.normal(size=t)
    returns = {
        "cot:gold_comm_net": g, "cot:gold_noncomm_net": -g + 0.05 * rng.normal(size=t),  # same asset
        "cot:corn_comm_net": e,                                                            # different asset
        "fred:DGS10": rng.normal(size=t),
    }
    sources = {"cot:gold_comm_net": "cot:metal", "cot:gold_noncomm_net": "cot:metal",
               "cot:corn_comm_net": "cot:ag", "fred:DGS10": "fred"}
    rep = measure_pool_diversity(returns, max_pairwise_corr=0.35, sources=sources)
    assert rep.mean_within_source_abs_corr > rep.mean_cross_source_abs_corr


def test_analytic_floor_returns_none_when_pool_too_correlated() -> None:
    """A pool that is all one idea (near-duplicates) admits < min_cohort_size ⇒ no verdict."""
    rng = np.random.default_rng(1)
    t = 300
    seed = rng.normal(size=t)
    pool = {f"dup{i}": seed + 0.01 * rng.normal(size=t) for i in range(10)}   # ~collinear
    base = {"tsmom": rng.normal(0.02, 1, t), "rates": rng.normal(0.02, 1, t)}
    ts = np.int64(1_500_000_000) + np.arange(t) * 86400
    fcfg = FitnessConfig(combiner_window=63, combiner_min_periods=20)
    ev = evaluate_cohort_analytic(pool, base, ts, _ccfg(max_pairwise_corr=0.30), fcfg)
    assert ev is None
