"""Unit test: the daily-IC series feeds statistics.deflated_sharpe_ratio (ADR-1).

Validates the central reuse claim — IC-IR is the per-period Sharpe of the daily IC
series, so DSR deflates it directly, and a larger ``n_trials`` (bigger candidate batch)
lowers the DSR (a higher bar for a best-of-N winner).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from finrl_pro_ds.signals import cross_sectional_ic


def _ic_series() -> tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(11)
    fwd = rng.standard_normal((400, 50))
    sig = 0.3 * fwd + rng.standard_normal((400, 50))  # positive but noisy IC
    r = cross_sectional_ic(sig, fwd)
    return r.ic_series, r.ic_ir, r.n_days


def test_dsr_on_ic_series_hardens_with_more_trials() -> None:
    ic, ic_ir, n_days = _ic_series()
    assert ic_ir > 0
    # Fix the across-trial SR variance pool; vary ONLY n_trials to isolate the
    # order-statistic (selection-bias) effect on the deflation benchmark SR*.
    trials = list(np.random.default_rng(0).normal(0.0, 0.05, size=64))
    g1, g2 = skewness(ic.tolist()), excess_kurtosis(ic.tolist())
    d_few = deflated_sharpe_ratio(ic_ir, trials, n_obs=n_days, skew=g1,
                                  excess_kurt=g2, n_trials=3)
    d_many = deflated_sharpe_ratio(ic_ir, trials, n_obs=n_days, skew=g1,
                                   excess_kurt=g2, n_trials=300)
    assert d_few is not None and d_many is not None
    assert 0.0 <= d_few["dsr"] <= 1.0
    assert 0.0 <= d_many["dsr"] <= 1.0
    assert d_many["dsr"] <= d_few["dsr"] + 1e-9          # more trials -> lower DSR
    assert d_many["sr_star"] >= d_few["sr_star"] - 1e-12  # higher deflation benchmark


def test_dsr_undefined_guards() -> None:
    # n_trials < 2 -> None (no multiplicity to deflate against)
    assert deflated_sharpe_ratio(0.5, [0.1], n_obs=100, skew=0.0,
                                 excess_kurt=0.0, n_trials=1) is None
    # n_obs < 3 -> None (too short a track to estimate variance bracket)
    assert deflated_sharpe_ratio(0.5, [0.1, 0.2, 0.3], n_obs=2, skew=0.0,
                                 excess_kurt=0.0, n_trials=3) is None
