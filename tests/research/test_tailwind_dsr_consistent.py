"""Tripwires for the TAILWIND consistent-DSR recomputation (scripts/research/tailwind_dsr_consistent.py).

The recomputation's load-bearing choice is the trial count. ``deflated_sharpe_ratio`` takes the
EMPIRICAL cross-sectional SR variance, and the script keeps the raw/declared N rather than
discounting it by the participation-ratio n_eff (the discount FORMULAS.md RS01 and the 2026-09-23
audit S-7 recommend). The empirical variance already shrinks with trial correlation, so the
discount double-counts it. These pin the measured basis for that choice, data-free:

  * iid trials: sigma_cs * e(K) reproduces the null E[max] (BLdP's own setting)
  * common factor: raw K is ~exact, while n_eff understates the null max -> fail-open
  * pure clusters: raw K overstates the null max -> fail-closed

and that ``e_max_factor`` is the SAME order-statistic factor ``deflated_sharpe_ratio`` uses, so
the calibration measures the estimator the gate actually runs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sharpen.crypto.eval.statistics import deflated_sharpe_ratio

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research" / "tailwind_dsr_consistent.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("tailwind_dsr_consistent", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _participation_ratio(corr: np.ndarray) -> float:
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def test_e_max_factor_is_the_library_factor(mod):
    trials = [0.01, -0.02, 0.03, 0.0, 0.015]
    sd = float(np.std(trials, ddof=1))
    for n in (2, 7, 24, 100):
        d = deflated_sharpe_ratio(0.05, trials, n_obs=500, skew=0.0, excess_kurt=0.0, n_trials=n)
        assert d["sr_star"] == pytest.approx(sd * mod.e_max_factor(n), rel=1e-12)


def test_iid_trials_raw_count_is_calibrated(mod):
    cal = mod.null_max_calibration(np.eye(18), n_sims=60_000)
    assert 0.95 < mod.estimator_ratio(cal, 18) < 1.07
    assert cal["n_star"] == pytest.approx(18, rel=0.25)


def test_common_factor_neff_fails_open_raw_count_does_not(mod):
    k, rho = 18, 0.6
    corr = np.full((k, k), rho)
    np.fill_diagonal(corr, 1.0)
    cal = mod.null_max_calibration(corr, n_sims=60_000)
    neff = _participation_ratio(corr)                        # ~2.5
    assert mod.estimator_ratio(cal, k) >= 0.97                # raw count: ~exact
    assert mod.estimator_ratio(cal, neff) < 0.80              # discounted count: fail-open


def test_pure_clusters_raw_count_is_fail_closed(mod):
    k, size = 18, 3
    corr = np.eye(k)
    for g in range(k // size):
        block = slice(g * size, (g + 1) * size)
        corr[block, block] = 0.999
    np.fill_diagonal(corr, 1.0)
    cal = mod.null_max_calibration(corr, n_sims=60_000)
    assert mod.estimator_ratio(cal, k) > 1.10                 # overstates the null max
