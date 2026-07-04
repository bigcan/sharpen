"""Phase-2 tests for the selection-aware MC null (novel statistic #2 — the BINDING cohort gate).

Design + math audit: ``docs/research/crucible_mc_null_spec.md`` §7 (the validated targets ARE the
acceptance criteria). Dims are kept small (the combiner runs inside every replicate); the tests pin
the QUALITATIVE properties the audit established, with generous bounds:

  * size ≈ α under the correct JOINT resample (BLOCKER 1 fixed);
  * the base-fixed variant is anti-conservative (BLOCKER-1 regression stays caught);
  * genuine weak diversifiers are DETECTED via the ΔSR statistic (the gate is not vacuous);
  * pure-noise cohorts (the m=9/N=100 cell) are rejected;
  * the p-value is deterministic under a fixed seed (reproducibility invariant).
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.signals.generation.cohort import CohortConfig
from finrl_pro_ds.signals.generation.cohort_mc import (
    auto_block_length,
    mc_null_pvalue,
    stationary_bootstrap_indices,
)
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

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
    # joint size near α (generous upper bound for the small meta count)
    assert joint_rate <= 0.15, f"joint size {joint_rate:.2f} should be ~α={_ALPHA}"
    # BLOCKER-1: dropping joint base resampling inflates size
    assert fixed_rate >= joint_rate, f"base-fixed {fixed_rate:.2f} should be ≥ joint {joint_rate:.2f}"


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
