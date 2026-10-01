"""CSCV-PBO on a Sharpe metric (TAILWIND Tier-2 N9 / T3-02).

The mean metric ranks books by raw per-block mean return, so on a grid whose books run at
different leverage the most-levered book is in-sample best on scale alone. TAILWIND's recorded
PBO 0.0009 ("not overfit") was that artifact. The Sharpe metric must be leverage-invariant, sit
near 0.5 on a skill-free grid, and still go low on real skill; the default stays "mean" so
Crucible verdicts are byte-stable (CRU-1).
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crypto.eval.statistics import probability_of_backtest_overfitting as pbo
from sharpen.crypto.eval.statistics import strip_leading_warmup


def _grid(T=2000, N=12, seed=0, vol=None, mu=0.0003):
    rng = np.random.default_rng(seed)
    vol = np.full(N, 0.01) if vol is None else np.asarray(vol, float)
    z = rng.standard_normal((T, N))
    return mu * (vol / 0.01) + z * vol          # identical true Sharpe across columns


def test_sharpe_pbo_is_leverage_invariant_and_mean_pbo_is_not():
    M = _grid()
    M10 = M.copy()
    M10[:, 3] *= 10.0
    s, s10 = pbo(M, metric="sharpe", n_splits=10), pbo(M10, metric="sharpe", n_splits=10)
    assert s10["pbo"] == pytest.approx(s["pbo"], abs=1e-12)
    m, m10 = pbo(M, n_splits=10), pbo(M10, n_splits=10)
    assert m10["pbo"] != m["pbo"]                 # the mean metric sees the scale


def _correlated(T, N, seed, vols, rho, sr_d=0.03):
    """Books sharing a common factor (TAILWIND's grid variants are highly correlated), identical
    true Sharpe, leverage set by ``vols``."""
    rng = np.random.default_rng(seed)
    z = np.sqrt(rho) * rng.standard_normal((T, 1)) + np.sqrt(1 - rho) * rng.standard_normal((T, N))
    return (sr_d + z) * np.asarray(vols)


def test_equal_sharpe_books_of_unequal_vol_reproduce_the_artifact():
    """Same true Sharpe, vol 0.5%..5%, correlated: the mean metric crowns the most-levered book and
    reports a LOW PBO ("not overfit"); the Sharpe metric reports ~0.5-0.6 (no skill). Measured
    2026-09-30: mean 0.15 / sharpe 0.60 at rho 0.8."""
    vols = np.linspace(0.005, 0.05, 12)
    rows = [(pbo(_correlated(3000, 12, k, vols, 0.8), n_splits=10)["pbo"],
             pbo(_correlated(3000, 12, k, vols, 0.8), n_splits=10, metric="sharpe")["pbo"])
            for k in range(10)]
    mean_p, sh_p = np.median([r[0] for r in rows]), np.median([r[1] for r in rows])
    assert mean_p < 0.3 and 0.4 < sh_p < 0.7


def test_sharpe_pbo_still_detects_real_skill():
    M = _grid(seed=3, mu=0.0)
    M[:, 0] += 0.002                              # one genuinely better book (SR ~3)
    out = pbo(M, n_splits=10, metric="sharpe", track=0)
    assert out["pbo"] < 0.1
    assert out["tracked_is_best_frac"] > 0.9 and out["tracked_below_median_frac"] < 0.05


def test_track_reports_the_deployed_books_conditional_rank():
    M = _grid(seed=4, mu=0.0)
    out = pbo(M, n_splits=10, metric="sharpe", track=5)
    assert 0.0 <= out["tracked_below_median_frac"] <= 1.0
    assert out["metric"] == "sharpe"
    with pytest.raises(ValueError):
        pbo(M, n_splits=10, track=99)
    with pytest.raises(ValueError):
        pbo(M, n_splits=10, metric="sortino")


def test_default_metric_is_unchanged():
    M = _grid(seed=5)
    a = pbo(M, n_splits=10)
    b = pbo(M, n_splits=10, metric="mean")
    assert a == b and a["metric"] == "mean"


def test_strip_leading_warmup():
    M = _grid(T=100, N=3, seed=6)
    M[:10, 0] = 0.0
    M[:25, 2] = 0.0                                # the slowest book goes live at row 25
    trimmed, dropped = strip_leading_warmup(M)
    assert dropped == 25 and trimmed.shape == (75, 3)
    with pytest.raises(ValueError, match="never live"):
        strip_leading_warmup(np.zeros((10, 2)))
