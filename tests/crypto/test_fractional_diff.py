"""Regression tests for fixed-window fractional differentiation (FFD).

Pins the 2026-07-08 feature-engineering audit finding FE-08: the vectorized
``_fractional_diff`` fed ``_get_ffd_weights``'s REVERSED weight array straight
into ``np.convolve``. Convolution time-reverses its kernel internally, so the
double reversal applied the FFD filter mirror-imaged — the unit weight w_0
landed on the OLDEST bar of the window and the fractional tail on the newest,
i.e. a (window-1)-bar-stale mirrored filter instead of Lopez de Prado's causal
FFD y[t] = sum_k w_k x[t-k].

The impulse-response test below fails on the mirrored variant and passes on
the correct one; the loop-reference test pins exact equivalence to the
classic AFML dot-with-window implementation the convolve version replaced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.features.crypto_features import (
    _fractional_diff,
    _get_ffd_weights,
)


def _ffd_loop_reference(series: pd.Series, d: float, window: int) -> pd.Series:
    """Classic AFML fixed-width FFD: dot(reversed weights, trailing window)."""
    w_rev = _get_ffd_weights(d, window).flatten()  # [w_{K-1} ... w_0]
    k = len(w_rev)
    vals = series.ffill().fillna(0.0).values
    res = np.zeros(len(vals), dtype=np.float64)
    for t in range(k - 1, len(vals)):
        res[t] = float(np.dot(w_rev, vals[t - k + 1 : t + 1]))
    return pd.Series(res, index=series.index)


def test_ffd_weight_recursion():
    """w_0=1, w_k = -w_{k-1}(d-k+1)/k — spot-check d=0.4."""
    w = _get_ffd_weights(0.4, 4).flatten()[::-1]  # natural order
    np.testing.assert_allclose(w, [1.0, -0.4, -0.12, -0.064], atol=1e-12)


def test_ffd_impulse_response_is_causal_natural_order():
    """Unit impulse at t must produce w_0 at t, w_1 at t+1, ... (never mirrored)."""
    n = 12
    s = pd.Series(np.zeros(n))
    s.iloc[5] = 1.0
    out = _fractional_diff(s, d=0.4, window=4).values
    # y[5]=w_0=1, y[6]=w_1=-0.4, y[7]=w_2=-0.12, y[8]=w_3=-0.064
    np.testing.assert_allclose(out[5:9], [1.0, -0.4, -0.12, -0.064], atol=1e-12)
    # Nothing before the impulse (causality) and nothing after the tail.
    assert np.all(out[:5] == 0.0)
    np.testing.assert_allclose(out[9:], 0.0, atol=1e-12)


def test_ffd_matches_loop_reference():
    """Vectorized convolve output == classic loop implementation, bit-tight."""
    rng = np.random.RandomState(7)
    s = pd.Series(np.cumsum(rng.normal(0, 1.0, 500)))
    for d, window in ((0.4, 100), (0.35, 100), (0.7, 30)):
        fast = _fractional_diff(s, d=d, window=window)
        ref = _ffd_loop_reference(s, d=d, window=window)
        np.testing.assert_allclose(fast.values, ref.values, rtol=1e-10, atol=1e-10)


def test_ffd_no_lookahead():
    """Perturbing the future must never change past outputs."""
    rng = np.random.RandomState(11)
    base = pd.Series(np.cumsum(rng.normal(0, 1.0, 300)))
    pert = base.copy()
    pert.iloc[200:] += 100.0
    out_base = _fractional_diff(base, d=0.4, window=50).values
    out_pert = _fractional_diff(pert, d=0.4, window=50).values
    np.testing.assert_array_equal(out_base[:200], out_pert[:200])
