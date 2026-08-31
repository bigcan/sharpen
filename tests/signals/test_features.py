"""Unit tests: Panel mechanics, OHLC sanity, and cross-sectional neutralization."""
from __future__ import annotations

import dataclasses

import numpy as np

from sharpen.signals import cross_sectional_ic
from sharpen.signals.features import (
    make_synthetic_panel,
    neutralize,
    ohlc_violations,
)


def test_panel_shapes_and_truncation_equivalence() -> None:
    p = make_synthetic_panel(T=100, N=20, seed=1)
    assert p.T == 100 and p.N == 20
    pt = p.truncated(49)
    assert pt.T == 50 and pt.N == 20
    assert pt.tickers == p.tickers
    # the retained rows must be byte-identical (the basis of the causality tripwire)
    assert np.array_equal(pt.close, p.close[:50], equal_nan=True)
    assert np.array_equal(pt.active, p.active[:50])


def test_forward_returns_label() -> None:
    p = make_synthetic_panel(T=50, N=10, seed=2)
    fwd = p.forward_returns(1)
    assert np.all(np.isnan(fwd[-1]))            # last row has no future
    expected = p.close[6] / p.close[5] - 1.0     # manual cell check at t=5, h=1
    assert np.allclose(fwd[5], expected, equal_nan=True)
    fwd5 = p.forward_returns(5)
    assert np.all(np.isnan(fwd5[-5:]))


def test_ohlc_sanity_clean_and_violation() -> None:
    p = make_synthetic_panel(T=30, N=8, seed=3)
    assert ohlc_violations(p)["total"] == 0
    bad_high = p.high.copy()
    bad_high[10, 3] = p.close[10, 3] * 0.5       # high below close -> violation
    p_bad = dataclasses.replace(p, high=bad_high)
    v = ohlc_violations(p_bad)
    assert v["high_lt_max_oc"] >= 1 and v["total"] >= 1


def test_neutralize_removes_pure_sector_signal() -> None:
    p = make_synthetic_panel(T=200, N=60, n_sectors=6, seed=4)
    # a signal that IS the sector id -> fully spanned by sector dummies -> residual ~0
    sig = np.tile(p.sector_id.astype(np.float64), (p.T, 1))
    neu = neutralize(sig, p, steps=("zscore", "sector"))
    finite = neu[np.isfinite(neu)]
    assert finite.size > 0
    assert np.max(np.abs(finite)) < 1e-6          # sector fully neutralized out


def test_neutralize_preserves_within_sector_variation() -> None:
    p = make_synthetic_panel(T=200, N=60, n_sectors=6, seed=5)
    rng = np.random.default_rng(0)
    randsig = rng.standard_normal((p.T, p.N))
    neu = neutralize(randsig, p, steps=("zscore", "sector"))
    finite = neu[np.isfinite(neu)]
    assert finite.std() > 0.1                     # idiosyncratic signal survives


def test_neutralize_shape_guard() -> None:
    import pytest

    p = make_synthetic_panel(T=20, N=5, seed=6)
    with pytest.raises(ValueError):
        neutralize(np.zeros((20, 6)), p)


def test_synthetic_panel_is_ic_neutral() -> None:
    # sanity: a random signal on the random panel has ~0 cross-sectional IC
    p = make_synthetic_panel(T=300, N=40, seed=7)
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((p.T, p.N))
    r = cross_sectional_ic(sig, p.forward_returns(1), active=p.active)
    assert abs(r.ic_mean) < 0.1
