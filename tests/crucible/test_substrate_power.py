"""NOW-5 substrate-power stamp (audit C2-01/C6-07) — the MDE interpolation + the power guard.

Locks the stamp to the funnel's own holdout split and to the E1/E2 calibration sweep, and verifies the
1/√N MDE interpolation (a t-statistic's SE ∝ 1/√N, so at fixed power the minimum-detectable ΔSR ∝
1/√(holdout_bars)). The flagship regression: a ~504-bar / holdout-126 substrate interpolates to an
implied MDE ≈ 4.45 ΔSR — far above any plausible alpha, exactly the underpowered mine the guard flags.
"""
from __future__ import annotations

import pytest

from finrl_pro_ds.crucible.orchestrator.substrate import (
    PowerGuard,
    SubstratePower,
    _power_holdout_bars,
    interp_mde,
    stamp_substrate_power,
)

# The E1/E2 MDE sweep grid (holdout_bars -> MDE at target power), matching calibration_mde_sweep.json.
_SWEEP = {"mde_sweep": {"rows": [
    {"holdout_bars": 189, "mde_realized_delta_sr": 3.6325},
    {"holdout_bars": 378, "mde_realized_delta_sr": 2.8467},
    {"holdout_bars": 696, "mde_realized_delta_sr": 1.9366},
    {"holdout_bars": 1011, "mde_realized_delta_sr": 1.4025},
]}}


def test_holdout_bars_matches_calibration_grid() -> None:
    """The stamp's holdout split must equal the funnel's binding split (else the power is mis-measured)."""
    assert _power_holdout_bars(756, 0.25) == 189
    assert _power_holdout_bars(1512, 0.25) == 378
    assert _power_holdout_bars(2782, 0.25) == 696
    assert _power_holdout_bars(4044, 0.25) == 1011


def test_interp_mde_returns_grid_nodes_exactly() -> None:
    for row in _SWEEP["mde_sweep"]["rows"]:
        mde, mode = interp_mde(row["holdout_bars"], _SWEEP)
        assert mde == pytest.approx(row["mde_realized_delta_sr"]) and mode == "grid"


def test_interp_mde_flagship_extrapolates_low() -> None:
    """The C2-01 case: holdout 126 (below the smallest grid point 189) → 1/√N extrapolation ≈ 4.45."""
    mde, mode = interp_mde(126, _SWEEP)
    assert mde == pytest.approx(3.6325 * (189 / 126) ** 0.5) and mode == "extrapolated_low"
    assert mde > 4.0                                            # unambiguously underpowered


def test_interp_mde_extrapolates_high_below_grid_floor() -> None:
    mde, mode = interp_mde(4044, _SWEEP)                        # deeper than the deepest grid point
    assert mode == "extrapolated_high" and mde == pytest.approx(1.4025 * (1011 / 4044) ** 0.5)
    assert mde < 1.4025                                         # more bars ⇒ smaller MDE


def test_interp_mde_between_nodes_is_interpolated_and_bracketed() -> None:
    mde, mode = interp_mde(500, _SWEEP)                         # between 378 and 696
    assert mode == "interpolated" and 1.9366 < mde < 2.8467


def test_interp_mde_is_monotone_decreasing_in_bars() -> None:
    vals = [interp_mde(h, _SWEEP)[0] for h in range(120, 4000, 40)]
    assert all(a > b for a, b in zip(vals, vals[1:]))          # deeper panel ⇒ strictly lower MDE


def test_stamp_substrate_power_assembles_the_record() -> None:
    p = stamp_substrate_power(504, 0.25, _SWEEP, sweep_hash="abc123")
    assert isinstance(p, SubstratePower)
    assert p.panel_T == 504 and p.holdout_bars == _power_holdout_bars(504, 0.25)
    assert p.calibration_sweep_hash == "abc123" and p.interp_mode == "extrapolated_low"
    assert p.implied_mde_delta_sr > 4.0                        # T=504 is underpowered by the calibration


def test_power_guard_is_a_plain_config_record() -> None:
    g = PowerGuard(enabled=True, ceiling=0.5, action="warn")
    assert g.enabled and g.ceiling == 0.5 and g.action == "warn" and g.force is False


# ============================================================ NOW-6: panel content hash (C2-07) ====

def _tiny_panel(*, extra_bar: bool = False, mutate: bool = False, feat=None):
    import numpy as np

    from finrl_pro_ds.signals.features import Panel
    rng = np.random.default_rng(0)
    tt = 40 + (1 if extra_bar else 0)
    nn = 3
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((tt, nn)), axis=0) + 4.0)
    if mutate:
        close[5, 1] *= 1.01                        # a single revised cell
    dates = (np.datetime64("2015-01-02") + np.arange(tt)).astype("datetime64[ns]")
    slots = {} if feat is None else {"macro:x": feat[:tt].astype(np.float64)}
    return Panel(dates, tuple(f"E{i}" for i in range(nn)), close, close, close, close,
                 close, np.ones((tt, nn), bool), close, rng.integers(0, 2, size=nn),
                 {"source": "t"}, feature_slots=slots)


def test_panel_content_hash_pins_bars_window_and_slots() -> None:
    from finrl_pro_ds.crucible.orchestrator.substrate import folded_snapshot_hash, panel_content_hash
    base = panel_content_hash(_tiny_panel())
    assert base == panel_content_hash(_tiny_panel())                 # deterministic
    assert base != panel_content_hash(_tiny_panel(extra_bar=True))   # a new price bar flips it
    assert base != panel_content_hash(_tiny_panel(mutate=True))      # a revised cell flips it
    import numpy as np
    assert base != panel_content_hash(_tiny_panel(feat=np.arange(41.0)))  # a feature slot flips it
    # folded hash changes if EITHER the catalog hash or the panel changes; stable when both fixed
    p = _tiny_panel()
    assert folded_snapshot_hash("cat1", p) == folded_snapshot_hash("cat1", p)
    assert folded_snapshot_hash("cat1", p) != folded_snapshot_hash("cat2", p)
    assert folded_snapshot_hash("cat1", p) != folded_snapshot_hash("cat1", _tiny_panel(extra_bar=True))
