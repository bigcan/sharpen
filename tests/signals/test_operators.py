"""Unit tests for the WorldQuant operator vocabulary: correctness + causality."""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.signals.library import operators as op


def test_delay_and_delta() -> None:
    x = np.arange(10, dtype=float).reshape(10, 1)
    d = op.delay(x, 2)
    assert np.isnan(d[:2]).all()
    assert np.allclose(d[2:], x[:-2])
    assert np.allclose(op.delta(x, 1)[1:], 1.0)        # arange → first diff is 1


def test_ts_sum_stddev_minmax() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert np.isnan(op.ts_sum(x, 3)[:2, 0]).all()
    assert np.allclose(op.ts_sum(x, 3)[2:, 0], [6, 9, 12])
    assert np.allclose(op.ts_max(x, 3)[2:, 0], [3, 4, 5])
    assert np.allclose(op.ts_min(x, 3)[2:, 0], [1, 2, 3])
    assert np.isclose(op.stddev(x, 3)[2, 0], np.std([1, 2, 3], ddof=1))


def test_rank_cross_sectional() -> None:
    x = np.array([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]])
    r = op.rank(x)
    assert np.allclose(r[0], [1 / 3, 2 / 3, 1.0])
    assert np.allclose(r[1], [1.0, 2 / 3, 1 / 3])


def test_scale_unit_l1() -> None:
    x = np.array([[1.0, -1.0, 2.0]])
    s = op.scale(x, 1.0)
    assert np.isclose(np.abs(s[0]).sum(), 1.0)
    assert np.allclose(s[0], [0.25, -0.25, 0.5])


def test_indneutralize_zeros_group_means() -> None:
    x = np.array([[1.0, 3.0, 10.0, 20.0, 30.0]])     # groups: {0,1}=A, {2,3,4}=B
    groups = np.array([0, 0, 1, 1, 1])
    out = op.indneutralize(x, groups)
    assert np.isclose(out[0, :2].mean(), 0.0)
    assert np.isclose(out[0, 2:].mean(), 0.0)


def test_signedpower() -> None:
    x = np.array([[-4.0, 9.0]])
    assert np.allclose(op.signedpower(x, 0.5), [[-2.0, 3.0]])


def test_correlation_perfect_linear() -> None:
    x = np.arange(8, dtype=float).reshape(8, 1) + np.array([[0.0]])
    x = np.tile(np.arange(8, dtype=float), (1, 1)).T
    y = 2.0 * x + 1.0
    c = op.correlation(x, y, 5)
    assert np.isclose(c[4, 0], 1.0, atol=1e-9)        # y is an affine fn of x → corr = 1


def test_ts_rank_and_argmax_on_increasing() -> None:
    x = np.arange(12, dtype=float).reshape(12, 1)
    assert np.isclose(op.ts_rank(x, 5)[6, 0], 1.0)    # today is the largest in its window
    assert np.isclose(op.ts_argmax(x, 5)[6, 0], 4.0)  # max at the most-recent slot (d-1)


def test_decay_linear_constant() -> None:
    x = np.full((10, 1), 5.0)
    out = op.decay_linear(x, 4)
    assert np.isclose(out[5, 0], 5.0)                 # weighted avg of a constant is the constant


def test_returns() -> None:
    c = np.array([[100.0], [110.0], [99.0]])
    r = op.returns(c)
    assert np.isnan(r[0, 0])
    assert np.isclose(r[1, 0], 0.10)
    assert np.isclose(r[2, 0], -0.10)


@pytest.mark.parametrize("fn", [
    lambda x: op.delay(x, 3),
    lambda x: op.delta(x, 2),
    lambda x: op.ts_sum(x, 5),
    lambda x: op.stddev(x, 5),
    lambda x: op.ts_rank(x, 5),
    lambda x: op.ts_argmax(x, 5),
    lambda x: op.decay_linear(x, 5),
    lambda x: op.correlation(x, np.roll(x, 1, axis=0), 5),
])
def test_timeseries_operators_are_causal(fn) -> None:
    """Truncation equivalence: value at row t must not depend on data after t."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 4))
    full = fn(x)
    for t in (12, 20, 27):
        trunc = fn(x[: t + 1])
        a, b = full[t], trunc[t]
        fin = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN pattern differs at t={t}"
        assert np.allclose(a[fin], b[fin], atol=1e-9), f"value differs at t={t} (look-ahead)"
