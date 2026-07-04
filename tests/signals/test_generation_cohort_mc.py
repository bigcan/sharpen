"""Phase-2 tests for the selection-aware MC null (novel statistic #2 — the BINDING cohort gate).

Design + math audit: ``docs/research/crucible_mc_null_spec.md`` §7 (the validated targets ARE the
acceptance criteria). Dims are kept small (the combiner runs inside every replicate); the tests pin
the QUALITATIVE properties the audit established, with generous bounds:

  * size ≈ α under the correct JOINT resample, and STRICTLY separated from the anti-conservative
    base-fixed variant (BLOCKER-1 regression stays caught — a collapse of the two rates fails);
  * the observed t_obs IS the documented per-period ΔSR functional, numerically pinned so a convention
    drift (``_ann_sharpe`` / a different base-vs-cohort config) cannot silently rescale it (BLOCKER-3);
  * genuine weak diversifiers are DETECTED via the ΔSR statistic (the gate is not vacuous);
  * pure-noise cohorts are rejected — both a small pool and the production m≥9/N≥100 config;
  * the p-value is deterministic under a fixed seed (reproducibility invariant).
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.signals.eval_harness import _ann_sharpe
from finrl_pro_ds.signals.generation.cohort import (
    CohortConfig,
    _with_redundancy,
    greedy_decorrelated_admission,
    standalone_sharpe_scores,
)
from finrl_pro_ds.signals.generation.cohort_mc import (
    auto_block_length,
    mc_null_pvalue,
    stationary_bootstrap_indices,
)
from finrl_pro_ds.signals.generation.fitness import (
    FitnessConfig,
    _combined_book,
    _per_period_sharpe,
)

_ALPHA = 0.05


def _fcfg() -> FitnessConfig:
    # small combiner windows so the book warms up fast on short test panels
    return FitnessConfig(n_groups=5, k_test=2, embargo=3, periods_per_year=1.0,
                         combiner_window=63, combiner_min_periods=20,
                         perf_window=63, perf_min_periods=20)


def _ccfg(**kw) -> CohortConfig:
    base = dict(max_cohort_size=8, min_cohort_size=3, max_pairwise_corr=0.35,
                promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                combiner_redundancy_strength=0.0)
    base.update(kw)
    return CohortConfig(**base)


def _timestamps(t: int) -> np.ndarray:
    return (np.int64(1_500_000_000) + np.arange(t, dtype=np.int64) * 86_400)


def _noise_base(rng, t: int) -> dict[str, np.ndarray]:
    # base sleeves keep a real (small positive) drift — H0 concerns candidates only
    return {"tsmom": rng.normal(0.02, 1.0, t), "rates": rng.normal(0.015, 1.0, t)}


def _noise_pool(rng, t: int, n: int) -> dict[str, np.ndarray]:
    return {f"c{i}": rng.normal(0.0, 1.0, t) for i in range(n)}


def _signal_pool(rng, t: int, n_sig: int, n_noise: int, s: float) -> dict[str, np.ndarray]:
    pool = {f"sig{i}": rng.normal(s, 1.0, t) for i in range(n_sig)}   # weak positive-Sharpe, iid
    pool.update({f"n{i}": rng.normal(0.0, 1.0, t) for i in range(n_noise)})
    return pool


# ------------------------------------------------------------------------------------------
# Bootstrap primitive
# ------------------------------------------------------------------------------------------
def test_stationary_bootstrap_is_a_valid_resample() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 21, rng)
    assert idx.shape == (500,)
    assert idx.min() >= 0 and idx.max() < 500


def test_bootstrap_deterministic_under_seed() -> None:
    a = stationary_bootstrap_indices(300, 21, np.random.default_rng(123))
    b = stationary_bootstrap_indices(300, 21, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_auto_block_length_reasonable() -> None:
    """The auto-ℓ fallback path (block_length=None) returns a sensible ℓ: larger for a persistent
    AR(1) series than for white noise, clamped into range."""
    rng = np.random.default_rng(0)
    t = 600
    white = rng.normal(size=(t, 3))
    ar = np.zeros((t, 3))
    for j in range(3):
        e = rng.normal(size=t)
        for i in range(1, t):
            ar[i, j] = 0.6 * ar[i - 1, j] + e[i]
    ell_white = auto_block_length(white)
    ell_ar = auto_block_length(ar)
    assert 1 <= ell_white <= t // 4 and 1 <= ell_ar <= t // 4
    assert ell_ar > ell_white     # persistent series ⇒ longer blocks


# ------------------------------------------------------------------------------------------
# Determinism of the gate
# ------------------------------------------------------------------------------------------
def test_pvalue_deterministic_under_seed() -> None:
    rng = np.random.default_rng(11)
    t = 300
    base, pool, ts = _noise_base(rng, t), _noise_pool(rng, t, 12), _timestamps(t)
    r1 = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=80, alpha_cohort=_ALPHA,
                        seed=999, block_length=21)
    r2 = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=80, alpha_cohort=_ALPHA,
                        seed=999, block_length=21)
    assert r1.p_value == r2.p_value and r1.t_obs == r2.t_obs


# ------------------------------------------------------------------------------------------
# BLOCKER-3 numeric tripwire — t_obs IS the documented per-period ΔSR functional
# ------------------------------------------------------------------------------------------
def test_tobs_is_per_period_delta_sr_functional() -> None:
    """MC-03: pin ``t_obs`` to the EXACT within-panel statistic the null replicates use, reconstructed
    here from the same helpers — ``_per_period_sharpe(combine(base∪members, cohort_cfg))
    − _per_period_sharpe(combine(base, fcfg))``. This locks three conventions that a refactor could
    silently break while every OTHER test (all self-vs-self / rate-based) stayed green, which would
    rescale ``T_obs`` against the null distribution and make ``p`` meaningless:

      1. the observed statistic is the PER-PERIOD Sharpe, not ``_ann_sharpe`` (here ``periods_per_year
         != 1`` so the two functionals differ numerically — the negative check below proves it);
      2. the base book is combined with ``fcfg``; the augmented book with
         ``cohort_cfg = _with_redundancy(fcfg, combiner_redundancy_strength)`` (redundancy set > 0 so
         ``fcfg`` and ``cohort_cfg`` genuinely differ — an aug-book built on ``fcfg`` would mismatch);
      3. admission is the standalone-Sharpe greedy de-dup (``members_obs`` is pinned too).
    """
    rng = np.random.default_rng(2024)
    t = 260
    base = _noise_base(rng, t)
    # small, ~independent weak-drift pool ⇒ deterministic admission of all 5 (≥ min_cohort_size ⇒
    # t_obs finite, not -inf).
    pool = {f"c{i}": rng.normal(0.03, 1.0, t) for i in range(5)}
    ts = _timestamps(t)
    # periods_per_year != 1 ⇒ _ann_sharpe != _per_period_sharpe; redundancy != 0 ⇒ cohort_cfg != fcfg
    fcfg = FitnessConfig(n_groups=5, k_test=2, embargo=3, periods_per_year=252.0,
                         combiner_window=63, combiner_min_periods=20,
                         perf_window=63, perf_min_periods=20)
    ccfg = _ccfg(combiner_redundancy_strength=0.5)

    r = mc_null_pvalue(pool, base, ts, ccfg, fcfg, n_reps=16, alpha_cohort=_ALPHA,
                       seed=7, block_length=21)

    # independent reconstruction of the observed statistic from the documented functional
    base_np = {k: np.asarray(v, dtype=np.float64) for k, v in base.items()}
    cand_np = {k: np.asarray(v, dtype=np.float64) for k, v in pool.items()}
    members = greedy_decorrelated_admission(
        cand_np, standalone_sharpe_scores(cand_np),
        max_pairwise_corr=ccfg.max_pairwise_corr, max_cohort_size=ccfg.max_cohort_size)
    assert len(members) >= ccfg.min_cohort_size
    cohort_cfg = _with_redundancy(fcfg, ccfg.combiner_redundancy_strength)
    sr_base = _per_period_sharpe(_combined_book(base_np, ts, fcfg))
    aug = {**base_np, **{m: cand_np[m] for m in members}}
    sr_aug = _per_period_sharpe(_combined_book(aug, ts, cohort_cfg))
    expected = sr_aug - sr_base

    assert np.isfinite(r.t_obs)
    assert r.t_obs == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert tuple(r.members_obs) == tuple(members)                 # admission convention pinned too

    # the fixture genuinely distinguishes the two Sharpe conventions: the _ann_sharpe reconstruction
    # is numerically different, so assertion (1) above has teeth (guards "per-period, NOT annualized").
    ann_expected = (_ann_sharpe(_combined_book(aug, ts, cohort_cfg), fcfg.periods_per_year)
                    - _ann_sharpe(_combined_book(base_np, ts, fcfg), fcfg.periods_per_year))
    assert abs(ann_expected - expected) > 1e-6
    assert r.t_obs != pytest.approx(ann_expected, rel=1e-6)


# ------------------------------------------------------------------------------------------
# Size calibration (joint resample) + BLOCKER-1 regression (base-fixed is anti-conservative)
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_size_calibration_joint_vs_base_fixed() -> None:
    """Over pure-noise pools the JOINT resample false-passes at ≈α; the base-fixed variant
    (BLOCKER 1) over-rejects H0 less conservatively — its size must exceed the joint size."""
    meta, t, n, reps = 40, 300, 12, 100
    joint_pass = fixed_pass = 0
    for mrep in range(meta):
        rng = np.random.default_rng(7000 + mrep)
        base, pool, ts = _noise_base(rng, t), _noise_pool(rng, t, n), _timestamps(t)
        rj = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=reps, alpha_cohort=_ALPHA,
                            seed=100 + mrep, block_length=21, resample_base=True)
        rf = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=reps, alpha_cohort=_ALPHA,
                            seed=100 + mrep, block_length=21, resample_base=False)
        joint_pass += rj.passes_mc
        fixed_pass += rf.passes_mc
    joint_rate, fixed_rate = joint_pass / meta, fixed_pass / meta
    # joint size sits near α — a generous upper bound for the small meta count (measured 0.075 here).
    # This ALSO catches the BLOCKER-1 mutation: if line-187's joint resample is dropped, joint_rate
    # collapses onto the inflated base-fixed rate (~0.125) and breaches this ceiling.
    assert joint_rate <= 0.10, f"joint size {joint_rate:.3f} should be ~α={_ALPHA}"
    # BLOCKER-1 is anti-conservative: holding the base fixed inflates the false-pass rate above α.
    assert fixed_rate >= 0.09, f"base-fixed size {fixed_rate:.3f} should inflate above α"
    # STRICT separation (the load-bearing MC-01 tripwire). The joint and base-fixed paths MUST NOT
    # collapse to the same rate. Mutating cohort_mc.py:187 to hold the base fixed (base_b = base) makes
    # both branches identical ⇒ gap == 0, so this fails and the anti-conservative bug cannot ship
    # green. The honest gap here is ~0.050; the 0.03 floor leaves headroom for MC noise.
    assert fixed_rate - joint_rate >= 0.03, (
        f"joint {joint_rate:.3f} and base-fixed {fixed_rate:.3f} collapsed — BLOCKER-1 "
        f"separation lost (line-187 joint resample dropped?)")


# ------------------------------------------------------------------------------------------
# Power — genuine weak diversifiers are detected (the gate is not vacuous)  [BLOCKER-3: ΔSR]
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_power_detects_weak_diversifiers() -> None:
    """Inject m genuinely-uncorrelated weak positive-Sharpe streams; the ΔSR MC must PASS at a rate
    well above the noise size — proving the statistic choice (ΔSR, not marginal-stream) is not blind
    to variance-reduction diversifiers."""
    meta, t = 20, 420
    power_pass = 0
    for mrep in range(meta):
        rng = np.random.default_rng(9000 + mrep)
        base = _noise_base(rng, t)
        pool = _signal_pool(rng, t, n_sig=8, n_noise=4, s=0.11)
        ts = _timestamps(t)
        r = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=120, alpha_cohort=_ALPHA,
                           seed=500 + mrep, block_length=21)
        power_pass += r.passes_mc
    power_rate = power_pass / meta
    assert power_rate >= 0.4, f"MC power on true diversifiers too low ({power_rate:.2f}) — vacuous gate?"


# ------------------------------------------------------------------------------------------
# Tripwire — a pure-noise cohort is rejected
# ------------------------------------------------------------------------------------------
def test_noise_cohort_rejected() -> None:
    rng = np.random.default_rng(4242)
    t = 320
    base, pool, ts = _noise_base(rng, t), _noise_pool(rng, t, 20), _timestamps(t)
    r = mc_null_pvalue(pool, base, ts, _ccfg(), _fcfg(), n_reps=150, alpha_cohort=_ALPHA,
                       seed=1, block_length=21)
    assert not r.passes_mc, f"pure-noise cohort must not pass (p={r.p_value:.3f})"
    assert 1.0 / 151 <= r.p_value <= 1.0


# ------------------------------------------------------------------------------------------
# MC-02 — noise rejection through the MC path at the PRODUCTION cohort config
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_noise_cohort_rejected_at_production_config() -> None:
    """MC-02: exercise the BINDING ``mc_null_pvalue`` path at the shipped PRODUCTION knobs
    (``max_cohort_size=12``, ``combiner_redundancy_strength=0.5`` — configs/crucible_cohort.gates.yaml)
    on the m≥9/N≥100 regime. The analytic tripwire
    (``test_generation_cohort.py::test_noise_cohort_tripwire_m9_n100``) only proves ``SR*_cohort``
    rejects that cell; this proves the MC null itself does too, at the config the gate ships with.
    Over seeded pure-noise pools nothing passes and every p-value sits clearly above α (measured
    p_min≈0.14)."""
    meta, t, n, reps = 5, 300, 100, 100
    ccfg = _ccfg(max_cohort_size=12, combiner_redundancy_strength=0.5)
    passes = 0
    p_min = 1.0
    for mrep in range(meta):
        rng = np.random.default_rng(20260704 + mrep)
        base, pool, ts = _noise_base(rng, t), _noise_pool(rng, t, n), _timestamps(t)
        r = mc_null_pvalue(pool, base, ts, ccfg, _fcfg(), n_reps=reps, alpha_cohort=_ALPHA,
                           seed=700 + mrep, block_length=21)
        passes += r.passes_mc
        p_min = min(p_min, r.p_value)
    assert passes == 0, f"production-config noise cohorts must not pass the MC null ({passes}/{meta})"
    # p bounded well away from α (measured min 0.139; 0.10 floor keeps margin over α=0.05)
    assert p_min >= 0.10, f"min MC p {p_min:.3f} should sit clearly above α={_ALPHA}"
