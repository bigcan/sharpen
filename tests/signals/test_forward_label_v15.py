"""crucible-v15.0 LEAK-2 tripwire: the forward-return LABEL may not be selected on FUTURE membership.

``Panel.active`` is the point-in-time MEMBERSHIP mask (us_equity top-300 by ADV, taiwan_smallcap's
cap-rank band). The label used to require ``active[t+h]``, so a name that left the universe inside
``(t, t+h]`` — disproportionately a name that fell — lost its realized return: survivorship built from
information the model could not have had at ``t``. These tests fail on the pre-v15 label.
"""
from __future__ import annotations

import numpy as np

from sharpen.signals.features import Panel


def _panel(close: np.ndarray, active: np.ndarray) -> Panel:
    t, n = close.shape
    dates = (np.datetime64("2020-01-02") + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"N{i}" for i in range(n)), close, close, close, close,
                 np.ones((t, n)), active, close, np.zeros(n, int), {})


def test_exit_from_membership_keeps_the_realized_return() -> None:
    close = np.array([[10.0, 10.0], [9.0, 11.0], [8.0, 12.0]])
    active = np.array([[True, True], [True, True], [False, True]])    # name 0 LEAVES the universe at t=2
    fwd = _panel(close, active).forward_returns(2)
    assert fwd[0, 0] == 8.0 / 10.0 - 1.0          # formed at t=0 while a member: the loss is realized
    assert fwd[0, 1] == 12.0 / 10.0 - 1.0


def test_no_formation_outside_membership_and_no_price_no_label() -> None:
    close = np.array([[10.0, 10.0, 10.0], [9.0, np.nan, 11.0], [8.0, 5.0, 0.0]])
    active = np.array([[False, True, True], [True, True, True], [True, True, True]])
    fwd = _panel(close, active).forward_returns(1)
    assert np.isnan(fwd[0, 0])                    # not a member at formation → no label
    assert np.isnan(fwd[0, 1])                    # no price at t+1 → no label (delisting unknowable)
    assert np.isnan(fwd[1, 2])                    # non-positive price at t+1 → no label


def test_selection_on_future_exit_no_longer_biases_the_label_mean() -> None:
    """Names that exit after falling: the old label dropped exactly those, inflating the mean."""
    rng = np.random.default_rng(0)
    t, n, h = 300, 200, 21
    rets = 0.01 * rng.standard_normal((t, n))
    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    trailing = np.full((t, n), np.nan)
    trailing[h:] = close[h:] / close[:-h] - 1.0
    active = ~(trailing < -0.08)                  # a name "exits the band" after a large fall
    fwd = _panel(close, active).forward_returns(h)
    formed = np.zeros((t, n), bool)
    formed[:-h] = active[:-h] & np.isfinite(fwd[:-h])
    true_mean = np.nanmean(np.where(active[:-h], close[h:] / close[:-h] - 1.0, np.nan))
    assert abs(np.nanmean(fwd[:-h][formed[:-h]]) - true_mean) < 1e-12
