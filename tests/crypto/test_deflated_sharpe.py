"""Deflated Sharpe Ratio + circular block-bootstrap CI (Bailey & Lopez de Prado 2014).

Promoted from the options-VRP falsification sleeve into ``crypto/eval/statistics`` so the
cross_asset_momentum pre-capital audit can haircut its best-of-N headline (Tier-2 N1,
P11-01/P11-02/P11-09). DSR is the multiple-testing control PSR@0 is not: it deflates the
benchmark to the EXPECTED MAX Sharpe of ``n_trials`` draws.

The golden test re-derives the BLdP formula from primitives (an INDEPENDENT code path) and
asserts the library reproduces it to 1e-9 — a mutation tripwire: a wrong sign, dropped
``sqrt(n-1)``, or a mangled variance bracket breaks the match.
"""
from __future__ import annotations

from math import e, erf, sqrt
from statistics import NormalDist

import numpy as np

from sharpen.crypto.eval.statistics import (
    block_bootstrap_sharpe_ci,
    block_bootstrap_sortino_ci,
    deflated_sharpe_ratio,
    excess_kurtosis,
    probabilistic_sharpe_ratio,
    skewness,
    sortino_ratio,
    std,
)

ANN = 252
_GAMMA_E = 0.5772156649015329


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _dsr_reference(obs_sr, trials, n_obs, g1, g2, n_trials):
    """Independent re-derivation of the BLdP deflated Sharpe (the golden)."""
    v_sr = std(trials, ddof=1) ** 2
    nd = NormalDist()
    sr_star = sqrt(v_sr) * (
        (1.0 - _GAMMA_E) * nd.inv_cdf(1.0 - 1.0 / n_trials)
        + _GAMMA_E * nd.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )
    bracket = 1.0 - g1 * obs_sr + ((g2 + 2.0) / 4.0) * obs_sr ** 2
    z = (obs_sr - sr_star) * sqrt(n_obs - 1) / sqrt(bracket)
    return _phi(z), sr_star


def test_dsr_matches_reference_golden():
    """Mutation tripwire: library DSR == independent re-derivation to 1e-9, and the value
    sits at the hand-computed magnitude (~0.9407) so both paths can't silently co-mutate."""
    trials = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]   # daily trial Sharpes
    obs, n_obs, g1, g2, n = 0.08, 1000, -0.5, 3.0, 10
    got = deflated_sharpe_ratio(obs, trials, n_obs=n_obs, skew=g1, excess_kurt=g2,
                                n_trials=n, periods_per_year=ANN)
    ref_dsr, ref_srstar = _dsr_reference(obs, trials, n_obs, g1, g2, n)
    assert got is not None
    assert abs(got["dsr"] - ref_dsr) < 1e-9
    assert abs(got["sr_star"] - ref_srstar) < 1e-12
    assert abs(got["sr_star_ann"] - ref_srstar * sqrt(ANN)) < 1e-9
    assert abs(got["dsr"] - 0.9407) < 5e-3          # coarse magnitude pin


def test_dsr_strictly_decreases_with_more_trials():
    """The whole point: more configs searched ⇒ higher expected-max bar ⇒ lower DSR."""
    trials = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    vals = [deflated_sharpe_ratio(0.06, trials, n_obs=2000, skew=0.0, excess_kurt=0.0,
                                  n_trials=n, periods_per_year=ANN)["dsr"]
            for n in (2, 5, 20, 100)]
    assert vals == sorted(vals, reverse=True)        # monotonically non-increasing
    assert vals[0] > vals[-1]                        # and strictly lower at 100 vs 2


def test_dsr_is_a_haircut_below_psr_under_multiplicity():
    """At n_trials > 1 the DSR must be <= the un-deflated PSR (same track), because PSR
    tests SR>0 while DSR tests SR > expected-max-of-N. The 0.601-style headline is exactly
    this best-of-N case (P11-02: PSR@0 is not a multiplicity control)."""
    r = np.random.default_rng(3).normal(0.0005, 0.012, 1500)
    sr_daily = float(r.mean() / r.std(ddof=1))
    g1, g2 = skewness(r.tolist()), excess_kurtosis(r.tolist())
    trials = [sr_daily * k for k in (0.6, 0.8, 1.0, 1.1, 1.2)]   # a plausible search spread
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods_per_year=1)
    dsr = deflated_sharpe_ratio(sr_daily, trials, n_obs=len(r), skew=g1, excess_kurt=g2,
                                n_trials=12, periods_per_year=ANN)["dsr"]
    assert dsr <= psr + 1e-9


def test_dsr_none_on_degenerate_inputs():
    assert deflated_sharpe_ratio(0.05, [0.04, 0.05], n_obs=500, skew=0.0, excess_kurt=0.0,
                                 n_trials=1, periods_per_year=ANN) is None          # <2 trials
    assert deflated_sharpe_ratio(0.05, [0.04, 0.04, 0.04], n_obs=500, skew=0.0,
                                 excess_kurt=0.0, n_trials=8, periods_per_year=ANN) is None  # 0 var
    assert deflated_sharpe_ratio(0.05, [0.04, 0.06], n_obs=2, skew=0.0, excess_kurt=0.0,
                                 n_trials=8, periods_per_year=ANN) is None          # <3 obs


def test_block_bootstrap_is_deterministic_and_ordered():
    r = np.random.default_rng(11).normal(0.0006, 0.011, 800)
    a = block_bootstrap_sharpe_ci(r, block=21, n_boot=500, seed=7, periods_per_year=ANN)
    b = block_bootstrap_sharpe_ci(r, block=21, n_boot=500, seed=7, periods_per_year=ANN)
    assert a == b                                    # deterministic given seed
    assert a["ci_low"] <= a["ci_high"]
    assert 0.0 <= a["p_sharpe_lt_0"] <= 1.0
    # a clearly positive-drift series should rarely bootstrap to SR<0
    assert a["p_sharpe_lt_0"] < 0.5


def test_block_bootstrap_none_on_short_series():
    assert block_bootstrap_sharpe_ci([0.01] * 10, block=21, periods_per_year=ANN) is None


# --- block_bootstrap_sortino_ci (the tail-adjusted A/B selection CI) ----------
def test_block_bootstrap_sortino_deterministic_and_ordered():
    r = np.random.default_rng(11).normal(0.0006, 0.011, 800)
    a = block_bootstrap_sortino_ci(r, block=21, n_boot=500, seed=7, periods_per_year=ANN)
    b = block_bootstrap_sortino_ci(r, block=21, n_boot=500, seed=7, periods_per_year=ANN)
    assert a == b                                    # deterministic given seed
    assert a["ci_low"] <= a["ci_high"]
    assert 0.0 <= a["p_sortino_lt_0"] <= 1.0
    assert a["p_sortino_lt_0"] < 0.5                 # positive-drift ⇒ rarely Sortino<0
    # the point Sortino sits inside the bootstrap CI
    point = sortino_ratio(r.tolist(), periods_per_year=ANN)
    assert a["ci_low"] <= point <= a["ci_high"]


def test_block_bootstrap_sortino_uses_downside_only():
    """A series with NO returns below target has zero downside deviation ⇒ every draw is
    +inf Sortino ⇒ p_sortino_lt_0 == 0 (proves it keys off DOWNSIDE, not symmetric vol)."""
    r = [0.0, 0.01, 0.02, 0.0, 0.015] * 20           # all >= target 0.0
    out = block_bootstrap_sortino_ci(r, block=21, n_boot=200, seed=7, periods_per_year=ANN)
    assert out is not None
    assert out["p_sortino_lt_0"] == 0.0
    assert out["ci_high"] == float("inf")            # honest: no-downside draws → +inf


def test_block_bootstrap_sortino_none_on_short_series():
    assert block_bootstrap_sortino_ci([0.01] * 10, block=21, periods_per_year=ANN) is None


# --------------------------------------------------------------------------- #
# PBO / CSCV — Probability of Backtest Overfitting (GP7-03)
# --------------------------------------------------------------------------- #
def test_pbo_skill_free_population_near_half():
    """A skill-free population (pure noise) ⇒ the IS-best is random OOS ⇒ PBO ≈ 0.5."""
    import numpy as np
    from sharpen.crypto.eval.statistics import probability_of_backtest_overfitting
    rng = np.random.default_rng(0)
    out = probability_of_backtest_overfitting(rng.standard_normal((1200, 40)), n_splits=10)
    assert out is not None and 0.30 <= out["pbo"] <= 0.70
    assert out["n_strategies"] == 40 and out["n_combos"] == 252   # C(10,5)


def test_pbo_dominant_config_is_low():
    """One config consistently superior across all blocks ⇒ IS-best is also OOS-best ⇒ PBO low."""
    import numpy as np
    from sharpen.crypto.eval.statistics import probability_of_backtest_overfitting
    rng = np.random.default_rng(1)
    m = rng.standard_normal((1200, 40))
    m[:, 0] += 0.25                                       # a genuine, persistent edge
    out = probability_of_backtest_overfitting(m, n_splits=10)
    assert out is not None and out["pbo"] <= 0.15


def test_pbo_degenerate_returns_none():
    import numpy as np
    from sharpen.crypto.eval.statistics import probability_of_backtest_overfitting
    assert probability_of_backtest_overfitting(np.zeros((100, 1)), n_splits=10) is None   # N<2
    assert probability_of_backtest_overfitting(np.zeros((5, 10)), n_splits=10) is None     # T<n_splits
    assert probability_of_backtest_overfitting(np.zeros((100, 10)), n_splits=7) is None    # odd
