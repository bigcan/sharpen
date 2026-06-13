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
import pytest

from finrl_pro_ds.paper import ParityHarness

from .conftest import allocator_arrays


@pytest.mark.parametrize("m", [260, 500, 778])
def test_future_perturbation_does_not_change_the_past(cfg, arrays_18, m):
    """Perturb conviction/price/vol/volume/tech at indices > m; assert the entire
    forward path over steps [0, m) is byte-unchanged."""
    base = arrays_18
    live_base, _ = ParityHarness(cfg).run(base)

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


def test_weights_are_price_independent(cfg):
    """The frozen-core weight VALUES are vol-scaled conviction — a pure price shock
    (signal/vol untouched) must leave every weight identical (only returns/equity move)."""
    base = allocator_arrays(T=400, n=6, seed=21, volume=1e7)
    live_base, _ = ParityHarness(cfg).run(base)
    pert = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    pert["price_ary"][1:] = base["price_ary"][1:] * 1.25     # global price shock (keeps sign/availability)
    live_pert, _ = ParityHarness(cfg).run(pert)
    np.testing.assert_array_equal(live_base.weights, live_pert.weights)
