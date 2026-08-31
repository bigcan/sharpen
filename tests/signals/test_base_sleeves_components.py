"""Tier B (Crucible independent-audit F14) tripwires — the base-sleeve gross/cost/gross-exposure
decomposition the overlay-cost correction consumes.

Locks: (A) ``with_components=True`` leaves ``net`` bit-identical to the default path (the live TSMOM
edge's book must not move); (B) the ``net == gross - cost`` invariant is exact; (C) ``cost >= 0`` and
``gross_exposure >= 0``; (D) ``unit_components`` is the exact unit-gross cost-free decomposition; (E)
a leveraged (gross > 1) target-weight book reports ``gross_exposure > 1`` (so the overlay cost is
charged at the true gross, not unit).
"""
from __future__ import annotations

import numpy as np

from sharpen.signals.generation.base_sleeves import (
    SleeveComponents,
    _book_from_target_weights,
    unit_components,
)

T, N = 500, 6


def _weights_fwd(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((T, N)) * 0.8                 # leveraged: Σ|w| typically > 1
    fwd = rng.standard_normal((T, N)) * 0.01
    fwd[rng.random((T, N)) < 0.03] = np.nan               # scattered NaNs (inactive cells)
    return w, fwd


def test_with_components_net_is_byte_identical_to_default() -> None:
    """The live-edge safety net: turning components ON must not perturb the net book by one ULP."""
    w, fwd = _weights_fwd(1)
    net = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.001)
    comp = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.001, with_components=True)
    assert isinstance(comp, SleeveComponents)
    assert np.array_equal(net, comp.net, equal_nan=True)


def test_net_equals_gross_minus_cost_exactly() -> None:
    w, fwd = _weights_fwd(2)
    c = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.0025, with_components=True)
    m = np.isfinite(c.net)
    assert np.max(np.abs(c.net[m] - (c.gross[m] - c.cost[m]))) == 0.0


def test_cost_and_gross_exposure_nonnegative() -> None:
    w, fwd = _weights_fwd(3)
    c = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.001, with_components=True)
    assert bool(np.all(c.cost >= 0.0))
    assert bool(np.all(c.gross_exposure >= 0.0))


def test_unit_components_is_exact_unit_gross_decomposition() -> None:
    w, fwd = _weights_fwd(4)
    net = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.001)
    uc = unit_components(net)
    assert np.array_equal(uc.gross, net, equal_nan=True)     # gross == net
    assert bool(np.all(uc.cost == 0.0))                      # no embedded cost
    assert bool(np.all(uc.gross_exposure == 1.0))            # unit gross


def test_leveraged_book_reports_gross_exposure_above_one() -> None:
    """A vol-scaled directional book runs gross > 1; the exposure stream must reflect it (else the
    overlay cost keeps under-charging at unit gross — the F14 defect)."""
    w, fwd = _weights_fwd(5)
    c = _book_from_target_weights(w, fwd, hold_horizon=21, cost_bps=0.001, with_components=True)
    # Σ|w| for N=6 standard-normal×0.8 weights averages well above 1.
    assert float(np.nanmean(c.gross_exposure)) > 1.0
