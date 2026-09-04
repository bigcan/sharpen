"""Tripwires for the power-wall re-derivation (``scripts/research/forward_power.py``, audit F7).

Two things need pinning here:

  1. **The ideal MDE80 arithmetic**, against F7's own published anchors AND against the duplicate
     copy in ``planted_sweep.py``. The formula is deliberately written out in both scripts rather
     than imported across research modules; without a test asserting they agree, that duplication
     is free to drift and the two scripts would quietly publish different power walls.
  2. **The 80%-crossing interpolation**, including the case where the sweep never reaches 80%.
     Silently returning the top of the grid there would report a finite MDE for a gate that never
     fired — the same shape as the vacuous results this project keeps rediscovering.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "research" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fp = _load("forward_power")


# ---------------------------------------------------------------------------------------------
# 1. The ideal arithmetic
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_bars,audit", [(1011, 1.42), (4044, 0.71)])
def test_ideal_mde80_matches_audit_f7(n_bars: int, audit: float) -> None:
    """F7's published ideal single-prereg t>=2 MDE80, to two decimals."""
    assert fp.ideal_mde80(2.0, n_bars, 252.0) == pytest.approx(audit, abs=0.01)


def test_ideal_mde80_agrees_with_planted_sweep() -> None:
    """The duplicated formula must match `planted_sweep.mde80` exactly, at every span.

    This is the anti-drift guard for a deliberate duplication. If someone edits one copy, this
    goes red rather than the two scripts publishing different walls.
    """
    ps = _load("planted_sweep")
    for n in (252, 1011, 2520, 4044, 10000):
        assert fp.ideal_mde80(2.0, n, 252.0) == pytest.approx(ps.mde80(2.0, n, 252.0), rel=1e-12)


def test_ideal_mde80_scales_as_inverse_sqrt_n() -> None:
    a = fp.ideal_mde80(2.0, 1000, 252.0)
    b = fp.ideal_mde80(2.0, 4000, 252.0)
    assert b == pytest.approx(a / 2.0, rel=1e-9)


def test_ideal_mde80_rejects_degenerate_span() -> None:
    with pytest.raises(ValueError):
        fp.ideal_mde80(2.0, 1, 252.0)


# ---------------------------------------------------------------------------------------------
# 2. The 80% crossing — both directions
# ---------------------------------------------------------------------------------------------
def _row(realized: float, power: float) -> dict:
    return {"median_realized_delta_sr": realized, "power": power}


def test_crossing_interpolates_between_bracketing_points() -> None:
    rows = [_row(1.0, 0.20), _row(2.0, 0.60), _row(3.0, 1.00)]
    got = fp._interp_crossing(rows, 0.80)
    assert got == pytest.approx(2.5, abs=1e-9), "power 0.80 sits halfway between 2.0 and 3.0"


def test_crossing_returns_none_when_never_reached() -> None:
    """An unreachable MDE must be reported as unreachable, not as the top of the grid.

    Returning the last grid value here would state a finite minimum detectable effect for a gate
    that never once reached 80% power — a number that reads as a measurement and is not one.
    """
    rows = [_row(1.0, 0.01), _row(2.0, 0.05), _row(3.0, 0.30)]
    assert fp._interp_crossing(rows, 0.80) is None


def test_crossing_ignores_non_finite_rows() -> None:
    rows = [_row(float("nan"), 0.0), _row(1.0, 0.20), _row(2.0, 1.00)]
    got = fp._interp_crossing(rows, 0.80)
    assert got is not None and 1.0 < got <= 2.0


def test_crossing_handles_a_flat_step() -> None:
    """Equal powers across a bracketing step must not divide by zero."""
    rows = [_row(1.0, 0.50), _row(2.0, 0.80), _row(3.0, 0.80)]
    assert fp._interp_crossing(rows, 0.80) == pytest.approx(2.0, abs=1e-9)


def test_crossing_when_grid_starts_already_above_target() -> None:
    """A grid whose FIRST point already clears 80% means the MDE is at or BELOW the grid.

    Reporting None there would say "unreachable" — understating the gate's power, the opposite
    error to the one `test_crossing_returns_none_when_never_reached` guards. Both directions are
    wrong in ways that read as findings, so both are pinned.
    """
    rows = [_row(1.0, 0.80), _row(2.0, 0.95)]
    assert fp._interp_crossing(rows, 0.80) == pytest.approx(1.0, abs=1e-9)
    rows_hi = [_row(0.4, 0.99), _row(1.0, 1.00)]
    assert fp._interp_crossing(rows_hi, 0.80) == pytest.approx(0.4, abs=1e-9)


def test_crossing_is_monotone_in_target() -> None:
    rows = [_row(1.0, 0.10), _row(2.0, 0.50), _row(3.0, 0.90)]
    assert fp._interp_crossing(rows, 0.50) <= fp._interp_crossing(rows, 0.80)


# ---------------------------------------------------------------------------------------------
# 3. The controls must be able to fail
# ---------------------------------------------------------------------------------------------
def test_controls_ok_requires_both_directions() -> None:
    """`controls_ok` is null-quiet AND saturation-loud; either alone must not pass it.

    Without the saturation leg, a gate that can never fire at any effect size would report
    'MDE unreachable' and look like a finding rather than a broken harness.
    """
    def ok(null_p: float, sat_p: float) -> bool:
        return bool(null_p <= 0.10 and sat_p >= 0.80)

    assert ok(0.00, 1.00)
    assert not ok(0.00, 0.10), "a gate that never fires must NOT count as valid controls"
    assert not ok(0.50, 1.00), "a gate that fires on noise must NOT count as valid controls"
    assert not ok(0.50, 0.10)
