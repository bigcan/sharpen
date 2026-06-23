"""TRIPWIRE (LEAK-2): the forward path never lets future data leak into the past.

Perturbing bars strictly AFTER index ``m`` must not change any weight, return, or
equity value at a step ``< m`` — every such step's inputs (causal conviction, causal
vol, prices ``<= m``) lie at or before ``m``. If anyone reintroduces look-ahead
(future-looking signal, cross-window normalization, a fill that peeks ahead), this
fails. Mirrors the env-level ``tests/envs/test_allocator_causality.py`` for the NEW
paper forward path the env tests cannot reach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.paper import ParityHarness

from .conftest import allocator_arrays


def _mid_month_cutoffs(timestamps: np.ndarray) -> list[int]:
    """Indices ~mid-month (last month-end + 10 bdays), spread across the series. A
    forward leak that forward-fills from the NEXT month-end IS absorbed at a month-end
    cutoff but NOT mid-month — so the tripwire must perturb from a mid-month index to
    catch a reintroduced intra-month leak (P2-04)."""
    ts = np.asarray(timestamps, dtype=np.int64)
    months = pd.to_datetime(ts, unit="s").to_period("M")
    me = pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()
    cands = [int(m + 10) for m in me if 50 < m + 10 < len(ts) - 2]
    return cands[::3][:4]


def test_future_perturbation_does_not_change_the_past(cfg, arrays_18):
    """Perturb conviction/price/vol/volume/tech at indices > m (m mid-month); assert the
    entire forward path over steps [0, m) is byte-unchanged. Mid-month cutoffs ensure a
    reintroduced intra-month forward leak would be caught here, not masked by a boundary."""
    base = arrays_18
    live_base, _ = ParityHarness(cfg).run(base)
    cutoffs = _mid_month_cutoffs(base["timestamps"])
    assert cutoffs, "no mid-month cutoffs derived"

    for m in cutoffs:
        pert = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        fut = slice(m + 1, None)                              # strictly future of step m
        pert["conviction_ary"][fut] = -base["conviction_ary"][fut]      # flip the signal
        pert["price_ary"][fut] = base["price_ary"][fut] * 1.4           # shock prices
        pert["vol_ary"][fut] = base["vol_ary"][fut] * 2.0               # shock vol
        pert["volume_ary"][fut] = base["volume_ary"][fut] * 0.25        # shock liquidity
        pert["tech_ary"][fut] = base["tech_ary"][fut] + 3.0
        live_pert, _ = ParityHarness(cfg).run(pert)

        # Steps < m depend only on data <= m; everything there must be identical.
        np.testing.assert_array_equal(live_base.weights[:m], live_pert.weights[:m])
        np.testing.assert_array_equal(live_base.step_returns[:m], live_pert.step_returns[:m])
        np.testing.assert_allclose(live_base.equity_curve[:m + 1], live_pert.equity_curve[:m + 1],
                                   atol=1e-9, rtol=0)


def test_single_future_bar_perturbation_is_causal(cfg, arrays_18):
    """Perturbing a SINGLE future bar (m+1, mid-month) must not change steps [0, m) — a
    one-bar leak an all-future perturbation could mask at a boundary (P2-04)."""
    base = arrays_18
    live_base, _ = ParityHarness(cfg).run(base)
    me = pd.Series(np.arange(len(base["timestamps"]))).groupby(
        pd.to_datetime(base["timestamps"], unit="s").to_period("M").values).max().to_numpy()
    m = int(me[len(me) // 2] + 8)                            # mid-month
    j = m + 1
    pert = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    pert["conviction_ary"][j] = -base["conviction_ary"][j]
    pert["price_ary"][j] = base["price_ary"][j] * 1.5
    pert["vol_ary"][j] = base["vol_ary"][j] * 3.0
    live_pert, _ = ParityHarness(cfg).run(pert)

    np.testing.assert_array_equal(live_base.weights[:m], live_pert.weights[:m])
    np.testing.assert_array_equal(live_base.step_returns[:m], live_pert.step_returns[:m])


def test_current_bar_perturbation_is_causal(cfg, arrays_18):
    """Current-bar (P2-01): perturb the SIGNAL inputs (conviction, vol) at index m ITSELF —
    not m+1 — and assert steps [0, m) are byte-unchanged. The future sweep starts at m+1, so a
    decision at m-1 that peeked ONE bar ahead (read index m) is INVISIBLE to it; this closes
    that boundary blind spot (the X2 failure mode on the forward path). Each weights[k<m] is
    decided from conviction/vol at its own bar (< m), so flipping index m must not move it."""
    base = arrays_18
    live_base, _ = ParityHarness(cfg).run(base)
    me = pd.Series(np.arange(len(base["timestamps"]))).groupby(
        pd.to_datetime(base["timestamps"], unit="s").to_period("M").values).max().to_numpy()
    ms = [int(me[len(me) // 3] + 7), int(me[2 * len(me) // 3] + 7)]
    tested = 0
    for m in ms:
        if not (50 < m < len(base["timestamps"]) - 2):
            continue
        pert = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        pert["conviction_ary"][m] = -base["conviction_ary"][m]      # flip the signal AT m
        pert["vol_ary"][m] = base["vol_ary"][m] * 3.0               # shock vol AT m
        live_pert, _ = ParityHarness(cfg).run(pert)
        np.testing.assert_array_equal(live_base.weights[:m], live_pert.weights[:m])
        np.testing.assert_array_equal(live_base.step_returns[:m], live_pert.step_returns[:m])
        tested += 1
    assert tested, "no in-range mid-month index derived"


def test_weights_are_price_independent(cfg):
    """The frozen-core weight VALUES are vol-scaled conviction — a pure price shock
    (signal/vol untouched) must leave every weight identical (only returns/equity move)."""
    base = allocator_arrays(T=400, n=6, seed=21, volume=1e7)
    live_base, _ = ParityHarness(cfg).run(base)
    pert = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    pert["price_ary"][1:] = base["price_ary"][1:] * 1.25     # global price shock (keeps sign/availability)
    live_pert, _ = ParityHarness(cfg).run(pert)
    np.testing.assert_array_equal(live_base.weights, live_pert.weights)
