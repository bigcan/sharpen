"""The O(T log M) rewrite of the P1b as-of-join causality gate (assert_asof_join_causal).

The exhaustive O(T^2) version re-ran the whole as-of join once per bar on the truncated calendar;
this replaces it with a single vectorized comparison of ``asof_join`` against an independent per-bar
release-time reference (:func:`_canonical_asof_values`). These tests are the safety proof for that
swap on a LEAK-2 tripwire:

  * EQUIVALENCE — the fast canonical reference reproduces the OLD per-bar truncated value
    (``asof_join(series, bars[:t+1])[t]``) EXACTLY, across revisions / ties / gaps / edge shapes.
  * TEETH — with ``asof_join`` monkeypatched to a leaky variant (reference-period bind, or a
    one-bar look-ahead), the gate RAISES. Losing the teeth would let a data-layer leak through.
  * PRESERVATION — the gate still passes silently for the genuine release-time join.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crucible.data import quality_gate as qg
from sharpen.crucible.data.connector import SeriesData, SeriesRef
from sharpen.crucible.data.quality_gate import (
    _canonical_asof_values,
    asof_join,
    assert_asof_join_causal,
    naive_reference_period_join,
)

_BASE = np.datetime64("2020-01-01")


def _days(offsets) -> np.ndarray:
    return (_BASE + np.asarray(offsets) * np.timedelta64(1, "D")).astype("datetime64[ns]")


def _series(ref_off, rel_off, vals, asset_class="positioning") -> SeriesData:
    return SeriesData(
        ref=SeriesRef(source_id="src", series_id="s", asset_class=asset_class),
        reference_period=_days(ref_off), value=np.asarray(vals, dtype=np.float64),
        release_timestamp=_days(rel_off))


def _old_reference(series: SeriesData, bars: np.ndarray) -> np.ndarray:
    """The value the OLD O(T^2) gate compared against: bar t's as-of value when the calendar is
    TRUNCATED at t (``asof_join(series, bars[:t+1])[t]``). This IS the semantics being preserved, so
    the fast reference must match it bar-for-bar."""
    return np.array([asof_join(series, bars[: t + 1])[t] for t in range(bars.shape[0])])


def _random_series(rng: np.random.Generator, *, with_revisions: bool) -> SeriesData:
    m = int(rng.integers(1, 40))
    ref = np.sort(rng.integers(0, 300, size=m))
    lag = rng.integers(0, 10, size=m)                 # non-negative release lag (PIT-valid)
    rel = ref + lag
    val = rng.standard_normal(m).round(6)
    if with_revisions and m >= 2:                     # re-release the latest period later, new value
        ref = np.concatenate([ref, ref[-1:]])
        rel = np.concatenate([rel, rel[-1:] + rng.integers(1, 20, size=1)])
        val = np.concatenate([val, rng.standard_normal(1).round(6)])
    return _series(ref, rel, val)


def _nan_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.nan_to_num(a, nan=-9.9e18), np.nan_to_num(b, nan=-9.9e18)))


# =========================================================== EQUIVALENCE vs the old O(T^2) value ===

@pytest.mark.parametrize("seed", range(25))
def test_canonical_matches_old_truncation_value_random(seed: int) -> None:
    """The fast per-bar reference == the old truncated-calendar value at EVERY bar (the exact
    semantics the O(T^2) loop enforced), over random series with revisions, ties, and gaps."""
    rng = np.random.default_rng(seed)
    series = _random_series(rng, with_revisions=bool(seed % 2))
    span = int(series.reference_period.max().astype("datetime64[D]").astype(int)
              - _BASE.astype("datetime64[D]").astype(int)) + 30
    bars = _days(np.arange(-3, span))
    assert _nan_equal(_canonical_asof_values(series, bars), _old_reference(series, bars))
    # and asof_join itself agrees with the reference (so the gate passes for the real join)
    assert _nan_equal(asof_join(series, bars), _canonical_asof_values(series, bars))


def test_canonical_edge_shapes() -> None:
    bars = _days(np.arange(0, 20))
    # empty series → all NaN
    empty = _series([], [], [])
    assert np.all(np.isnan(_canonical_asof_values(empty, bars)))
    # single obs, released mid-window
    one = _series([5], [8], [3.0])
    got = _canonical_asof_values(one, bars)
    assert np.all(np.isnan(got[:8])) and np.all(got[8:] == 3.0)
    # exact (ref, rel) duplicate with different values → later-in-order supersedes (matches old value)
    dup = _series([5, 5], [8, 8], [3.0, 7.0])
    assert _nan_equal(_canonical_asof_values(dup, bars), _old_reference(dup, bars))
    # all releases AFTER every bar → all NaN (nothing public yet)
    future = _series([1, 2], [999, 1000], [1.0, 2.0])
    assert np.all(np.isnan(_canonical_asof_values(future, bars)))


# =========================================================== PRESERVATION (passes for real join) ===

def test_gate_passes_for_real_join_with_revisions() -> None:
    series = _series([0, 10, 20, 20], [2, 12, 22, 40], [1.0, 2.0, 3.0, 3.5])   # last = revision
    bars = _days(np.arange(-2, 60))
    assert_asof_join_causal(series, bars)             # must not raise


# =========================================================== TEETH (LEAK-2 regression guard) ===

def test_gate_raises_when_join_binds_by_reference_period(monkeypatch) -> None:
    """If asof_join regressed to a reference-period bind (the exact look-ahead the gate exists to
    catch — a value used before it was released), the canonical comparison must FIRE."""
    series = _series([0, 7, 14], [5, 12, 19], [10.0, 20.0, 30.0])   # +5d release lag
    bars = _days(np.arange(0, 30))
    # sanity: the two joins genuinely differ on the post-reference/pre-release window
    assert not _nan_equal(naive_reference_period_join(series, bars), asof_join(series, bars))
    monkeypatch.setattr(qg, "asof_join", naive_reference_period_join)
    with pytest.raises(AssertionError, match="not causal"):
        assert_asof_join_causal(series, bars)


def test_gate_raises_on_one_bar_look_ahead(monkeypatch) -> None:
    """A join that peeks ONE bar ahead (out[t] uses bar t+1's data) is non-per-bar; the canonical
    per-bar reference diverges and the gate fires — even for a single contaminated bar."""
    series = _series([0, 30], [1, 31], [1.0, 2.0])
    bars = _days(np.arange(0, 40))

    def _peek(s, b):                                  # correct join, then shift one bar's value earlier
        out = asof_join(s, b)
        if out.shape[0] > 25:
            out = out.copy()
            out[25] = out[35]                         # bar 25 sees bar 35's (future) value
        return out

    monkeypatch.setattr(qg, "asof_join", _peek)
    with pytest.raises(AssertionError, match="not causal"):
        assert_asof_join_causal(series, bars)


def test_gate_does_not_raise_on_clean_join_control(monkeypatch) -> None:
    """Control for the teeth tests: with asof_join left as the genuine release-time join, the SAME
    inputs pass — proving the two tests above fail because of the leak, not the fixture."""
    series = _series([0, 30], [1, 31], [1.0, 2.0])
    bars = _days(np.arange(0, 40))
    assert_asof_join_causal(series, bars)             # unpatched → no raise
