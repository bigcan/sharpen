"""crucible-v7.0: the substrate-power stamp is CANDIDATE-TYPE aware (audit V4, 2026-07-29).

An MDE curve characterizes a GATE, not a dataset. A tick mines cross_sectional AND overlay genomes
through structurally different scoring paths, and the measured curves differ: at matched depth the
cross-sectional path is EQUAL-OR-WORSE than the overlay one (holdout 1011: overlay 1.281 vs
cross-sectional 1.833). Every curve before this change was measured on the overlay path yet applied to
both, so a cross-sectional mine was judged by a curve that UNDER-states its MDE — the guard claiming
more power than the substrate has, which is the fail-OPEN direction crucible-v5.0 exists to close.

These tests pin the three properties that repair depends on: worst-across-types, fail-closed on an
unmeasured type, and pooling over a surface's extra axis without disturbing the single-curve path.
"""
from __future__ import annotations

import math

import pytest

from sharpen.crucible.orchestrator.substrate import (
    interp_mde,
    stamp_substrate_power,
    sweep_candidate_type,
)


def _sweep(rows, candidate_type=None):
    inner = {"rows": rows}
    if candidate_type is not None:
        inner["candidate_type"] = candidate_type
    return {"mde_sweep": inner}


_OVERLAY = _sweep([{"holdout_bars": 200, "mde_realized_delta_sr": 1.0},
                   {"holdout_bars": 800, "mde_realized_delta_sr": 0.40}], "overlay")
# a SURFACE: several breadths per depth
_XSEC = _sweep([{"holdout_bars": 200, "n": 12, "mde_realized_delta_sr": 1.5},
                {"holdout_bars": 200, "n": 100, "mde_realized_delta_sr": 1.2},
                {"holdout_bars": 800, "n": 12, "mde_realized_delta_sr": 0.90},
                {"holdout_bars": 800, "n": 100, "mde_realized_delta_sr": 0.70}],
               "cross_sectional")


def test_legacy_sweep_without_candidate_type_reads_as_overlay():
    """Files written before the surface existed carry no `candidate_type` — and they ARE overlay
    curves (they plant macro:plant through _overlay_returns). Defaulting them to a wildcard would let
    one silently judge a cross-sectional mine, which is the bug this change closes."""
    assert sweep_candidate_type(_sweep([{"holdout_bars": 1, "mde_realized_delta_sr": 1.0}])) == "overlay"
    assert sweep_candidate_type(_XSEC) == "cross_sectional"


def test_surface_pools_over_breadth_taking_the_WORST_mde():
    """MDE showed no systematic N-dependence when measured (N=12->100), so indexing by n would claim a
    resolution the data lacks. Pool and keep the worst value at each depth — fail-safe for a refuse-gate."""
    assert interp_mde(200, _XSEC) == (1.5, "grid")      # not 1.2
    assert interp_mde(800, _XSEC) == (0.90, "grid")     # not 0.70


def test_pooling_is_a_noop_for_a_single_curve():
    assert interp_mde(200, _OVERLAY) == (1.0, "grid")
    assert interp_mde(800, _OVERLAY) == (0.40, "grid")


def test_stamp_takes_the_WORST_mde_across_the_types_being_mined():
    """The repair itself. A substrate mining both types must be judged by the harder of the two."""
    both = stamp_substrate_power(3200, 0.25, {"overlay": _OVERLAY, "cross_sectional": _XSEC}, "h",
                                 candidate_types=("overlay", "cross_sectional"))
    assert both.holdout_bars == 800
    assert both.implied_mde_delta_sr == pytest.approx(0.90)     # xsec, not the overlay 0.40
    assert both.interp_mode.endswith(":cross_sectional")

    overlay_only = stamp_substrate_power(3200, 0.25, {"overlay": _OVERLAY}, "h",
                                         candidate_types=("overlay",))
    assert overlay_only.implied_mde_delta_sr == pytest.approx(0.40)
    assert overlay_only.interp_mode == "grid"                   # unsuffixed when a single type


def test_unmeasured_candidate_type_fails_CLOSED():
    """A type with no measured curve must REFUSE, never borrow another type's. Borrowing is exactly
    how the overlay curve came to judge cross-sectional mines in the first place."""
    st = stamp_substrate_power(3200, 0.25, {"overlay": _OVERLAY}, "h",
                               candidate_types=("overlay", "cross_sectional"))
    assert st.implied_mde_delta_sr == math.inf
    assert st.interp_mode == "unmeasured_candidate_type"


def test_empty_type_set_is_UNMEASURED_not_a_free_pass():
    """crucible-v8.0. The worst-across-types fold starts at -inf; returning that when NO type was
    consulted would pass ANY ceiling — a fail-OPEN on the empty set. An empty set means no curve was
    read, so no power is claimed. This is the stamp half of the missing-curve repair: the loader hands
    back `{}` when a configured guard cannot find its curve, and it must land on REFUSE."""
    st = stamp_substrate_power(3200, 0.25, {}, "", candidate_types=())
    assert st.implied_mde_delta_sr == math.inf
    assert st.interp_mode == "unmeasured_empty"
    assert st.holdout_bars == 800          # still stamps the depth it could measure


def test_legacy_single_sweep_positional_form_still_works():
    """Backward compatibility: the pre-V4 call passed ONE sweep dict and no candidate_types."""
    st = stamp_substrate_power(3200, 0.25, _OVERLAY, "h")
    assert st.implied_mde_delta_sr == pytest.approx(0.40)
    assert st.interp_mode == "grid"


def test_null_and_nonfinite_mde_rows_are_dropped_not_read_as_zero():
    """`undetected` is the opposite of `detectable at zero effect`. A None row must not become a 0.0
    MDE that waves a substrate through."""
    sw = _sweep([{"holdout_bars": 200, "mde_realized_delta_sr": None},
                 {"holdout_bars": 800, "mde_realized_delta_sr": 0.40}], "overlay")
    mde, mode = interp_mde(200, sw)
    assert mde > 0.40 and mode == "extrapolated_low"     # 200 now sits BELOW the only measured anchor
    assert interp_mde(800, sw) == (0.40, "grid")

    empty = _sweep([{"holdout_bars": 200, "mde_realized_delta_sr": None}], "overlay")
    assert interp_mde(200, empty) == (math.inf, "unmeasured_empty")
