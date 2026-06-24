"""Unit tests: cross-sectional IC core + FDR + block bootstrap.

Synthetic panels only (no external data). Establishes the golden behaviours the harness
relies on: signal==fwd -> IC≈1, signal==-fwd -> IC≈-1, independent -> IC≈0 and
insignificant, and that the active mask / degenerate-day guards work.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.signals import (
    bh_fdr,
    block_bootstrap_mean,
    cross_sectional_ic,
    one_sided_p,
    spearman_ic,
)


def _fwd(t: int = 300, n: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, n))


def test_strong_positive_ic() -> None:
    rng = np.random.default_rng(0)
    fwd = rng.standard_normal((300, 40))
    sig = fwd + 0.05 * rng.standard_normal((300, 40))  # near-perfect but VARYING daily IC
    r = cross_sectional_ic(sig, fwd)
    assert r.ic_mean > 0.95
    assert r.ic_ir > 5.0
    assert r.n_days == fwd.shape[0]


def test_constant_ic_series_yields_nan_ir() -> None:
    # signal == fwd -> IC is exactly 1.0 every day -> zero-variance IC series.
    # IC-IR (mean/std) is then undefined; the harness must see NaN, not +inf.
    fwd = _fwd()
    r = cross_sectional_ic(fwd.copy(), fwd)
    assert r.ic_mean == 1.0
    assert np.isnan(r.ic_ir)
    assert np.isnan(r.ic_tstat)


def test_perfect_negative_ic() -> None:
    fwd = _fwd(seed=3)
    r = cross_sectional_ic(-fwd, fwd)
    assert r.ic_mean < -0.99


def test_zero_ic_on_independent() -> None:
    r = cross_sectional_ic(_fwd(seed=1), _fwd(seed=2))
    assert abs(r.ic_mean) < 0.1
    assert abs(r.ic_tstat) < 3.0  # not significant


def test_positive_but_noisy_ic_is_significant() -> None:
    rng = np.random.default_rng(7)
    fwd = rng.standard_normal((500, 60))
    sig = 0.3 * fwd + rng.standard_normal((500, 60))  # weak true correlation
    r = cross_sectional_ic(sig, fwd)
    assert 0.05 < r.ic_mean < 0.5
    assert r.ic_tstat > 3.0  # many days -> detectable


def test_cross_sectional_ic_matches_scipy_reference() -> None:
    # the vectorized IC must equal a per-day scipy.spearmanr loop (parity guard for the opt)
    from scipy.stats import spearmanr

    rng = np.random.default_rng(123)
    t, n = 200, 50
    sig = rng.standard_normal((t, n))
    fwd = 0.2 * sig + rng.standard_normal((t, n))
    active = rng.random((t, n)) > 0.1                 # ~10% inactive at random
    ref: list[float] = []
    for i in range(t):
        m = np.isfinite(sig[i]) & np.isfinite(fwd[i]) & active[i]
        if int(m.sum()) < 4:
            continue
        rho, _ = spearmanr(sig[i][m], fwd[i][m])
        if np.isfinite(rho):
            ref.append(float(rho))
    r = cross_sectional_ic(sig, fwd, active=active)
    assert r.n_days == len(ref)
    assert np.allclose(r.ic_series, ref, atol=1e-9)
    assert abs(r.ic_mean - float(np.mean(ref))) < 1e-9


def test_active_mask_skips_thin_days() -> None:
    fwd = _fwd(t=50, n=10)
    active = np.ones((50, 10), dtype=bool)
    active[:, :8] = False  # only 2 names active -> below min_names=4
    r = cross_sectional_ic(fwd.copy(), fwd, active=active, min_names=4)
    assert r.n_days == 0
    assert np.isnan(r.ic_mean)


def test_shape_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        cross_sectional_ic(np.zeros((10, 5)), np.zeros((10, 6)))
    with pytest.raises(ValueError):
        cross_sectional_ic(np.zeros(10), np.zeros(10))


def test_spearman_ic_guards() -> None:
    assert np.isnan(spearman_ic(np.ones(100), np.arange(100)))  # constant x
    assert np.isnan(spearman_ic(np.arange(5), np.arange(5)))    # too few obs
    assert spearman_ic(np.arange(100), np.arange(100)) > 0.99   # monotone


def test_bh_fdr_monotone_and_bounded() -> None:
    q = bh_fdr([0.001, 0.01, 0.5, 0.9])
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[0] <= q[-1]
    assert bh_fdr([]) == []


def test_one_sided_p_directions() -> None:
    assert one_sided_p(0.5, 0.1) < 0.05   # strong positive -> small p
    assert one_sided_p(-0.5, 0.1) > 0.95  # negative -> large p
    assert one_sided_p(0.1, 0.0) == 1.0   # degenerate SE


def test_block_bootstrap_mean() -> None:
    s = np.random.default_rng(0).standard_normal(500) + 0.1
    d = block_bootstrap_mean(s, n_boot=500)
    assert d is not None
    assert d["ci_low"] < d["mean"] < d["ci_high"]
    assert block_bootstrap_mean(np.zeros(5)) is None  # too short
