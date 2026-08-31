"""Probabilistic Sharpe Ratio + Minimum Track Record Length (Bailey & Lopez de Prado).

``min_track_record_length`` is the analytic inverse of ``probabilistic_sharpe_ratio``
(same skew/kurtosis-adjusted SE), wired into the paper-soak performance backstop (P11-05,
S553-cont-57). These tests pin (a) the inverse relationship exactly and (b) the
have-enough-track monotonicity the gate relies on.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from sharpen.crypto.eval.statistics import (
    excess_kurtosis,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    skewness,
)

ANN = 252


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _series(mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def test_mintrl_is_exact_inverse_of_psr():
    """At a track of length MinTRL the skew/kurtosis-adjusted PSR equals the target
    confidence — re-derive the PSR z-stat with n := MinTRL from the public moments and
    assert Phi(z) == prob (the inverse property the gate's MinTRL report rests on)."""
    r = _series(0.0006, 0.012, 800, seed=5)
    prob = 0.95
    sr = sharpe_ratio(r, periods_per_year=ANN)
    g1, g2 = skewness(r), excess_kurtosis(r)
    bracket = 1.0 - g1 * sr + ((g2 + 2.0) / 4.0) * sr ** 2
    mintrl = min_track_record_length(r, sr_benchmark=0.0, prob=prob, periods_per_year=ANN)
    assert np.isfinite(mintrl) and mintrl > 1.0
    # PSR z-stat evaluated AT n = MinTRL must reproduce the target confidence.
    z_at_mintrl = (sr - 0.0) * sqrt(mintrl - 1.0) / sqrt(bracket)
    assert abs(_normal_cdf(z_at_mintrl) - prob) < 1e-6


def test_mintrl_infinite_when_sr_below_benchmark():
    """An edge that does not beat the benchmark can never be confirmed → MinTRL = inf."""
    r = _series(-0.0004, 0.012, 500, seed=9)            # negative drift ⇒ SR < 0
    assert not np.isfinite(min_track_record_length(r, sr_benchmark=0.0, periods_per_year=ANN))
    # And vs a positive benchmark a weak-positive edge is also unreachable.
    weak = _series(0.00005, 0.012, 500, seed=10)
    assert not np.isfinite(
        min_track_record_length(weak, sr_benchmark=5.0, periods_per_year=ANN))


def test_have_enough_track_iff_psr_above_confidence():
    """The gate uses: PSR(actual) >= prob  ⇔  MinTRL <= actual length. Pin both branches."""
    prob = 0.95
    strong = _series(0.0009, 0.011, 600, seed=1)        # high SR, long track
    psr_s = probabilistic_sharpe_ratio(strong, sr_benchmark=0.0, periods_per_year=ANN)
    mintrl_s = min_track_record_length(strong, sr_benchmark=0.0, prob=prob, periods_per_year=ANN)
    assert psr_s >= prob and mintrl_s <= len(strong)

    weak = _series(0.0003, 0.013, 120, seed=2)          # modest SR, short track
    psr_w = probabilistic_sharpe_ratio(weak, sr_benchmark=0.0, periods_per_year=ANN)
    mintrl_w = min_track_record_length(weak, sr_benchmark=0.0, prob=prob, periods_per_year=ANN)
    if psr_w < prob:
        assert mintrl_w > len(weak)


def test_mintrl_short_sample_is_infinite():
    assert not np.isfinite(min_track_record_length([0.01, 0.02], periods_per_year=ANN))
