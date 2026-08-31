"""NOW-5 substrate-power stamp (audit C2-01/C6-07) — the MDE interpolation + the power guard.

Locks the stamp to the funnel's own holdout split and to the E1/E2 calibration sweep. BETWEEN measured
grid points the MDE is interpolated linearly in 1/√(holdout_bars) — anchored at two measured
neighbours. OFF THE TOP of the grid it is NOT extrapolated at all: the stamp returns ``+inf`` /
``unmeasured_high`` and the guard refuses (S553-cont-135). The flagship regression: a ~504-bar /
holdout-126 substrate reads an implied MDE ≈ 4.45 ΔSR — far above any plausible alpha, exactly the
underpowered mine the guard flags.
"""
from __future__ import annotations

import math

import pytest

from sharpen.crucible.orchestrator.substrate import (
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
_H_HI, _M_HI = 1011, 1.4025                 # deepest MEASURED grid point (where the daily substrates sit)
_H_LO, _M_LO = 189, 3.6325                  # shallowest measured grid point
_CEILING = 0.50                             # configs/crucible_power.gates.yaml: plausible_delta_sr_max

# The measured off-grid law, from the cont-129 intraday Stage-0 experiment: above N_eff≈1000 the
# deflated funnel's MDE flattens to ~N^-0.21 (it reproduces that run's measured 0.86 at N_eff=10210 to
# 2dp). Used here ONLY as the yardstick a fix must not fall below — never as an implementation.
def _measured_law(h: float) -> float:
    return _M_HI * (_H_HI / h) ** 0.21


def _falsified_sqrt_law(h: float) -> float:
    """The pre-fix ``extrapolated_high`` branch, kept to prove the repair is monotone-STRICTER."""
    return _M_HI * (_H_HI / h) ** 0.5


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


def test_interp_mde_between_nodes_is_interpolated_and_bracketed() -> None:
    mde, mode = interp_mde(500, _SWEEP)                         # between 378 and 696
    assert mode == "interpolated" and 1.9366 < mde < 2.8467


def test_interp_mde_is_monotone_decreasing_across_the_MEASURED_domain() -> None:
    """Deeper panel ⇒ strictly lower MDE — but only where the sweep actually measured. Above ``h_hi``
    the return is a fail-closed sentinel, not an estimate, so monotonicity does not extend there (this
    test used to sweep to 4000 and so silently asserted the falsified extrapolation was well-formed)."""
    vals = [interp_mde(h, _SWEEP)[0] for h in range(120, _H_HI + 1, 40)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert all(math.isfinite(v) for v in vals)


# ===================================== S553-cont-135: the anti-conservative extrapolation repair ====
# The pre-fix branch extrapolated ABOVE the grid by 1/√N ("more bars ⇒ smaller MDE"). cont-129's
# intraday Stage-0 experiment DIRECTLY MEASURED that scaling and falsified it: the funnel's MDE
# FLATTENS to ~N^-0.21 above N_eff≈1000. 1/√N therefore UNDER-stated the MDE off-grid — the guard
# claimed more power than exists and FAILED OPEN, which for a refuse-action gate is the one
# unacceptable direction. The repair refuses to extrapolate at all.

def test_off_grid_high_is_unmeasured_not_extrapolated() -> None:
    """THE regression. Off the top of the grid the stamp must claim NO power rather than invent it."""
    for h in (_H_HI + 1, 2028, 4044, 10210, 20000, 52826):
        mde, mode = interp_mde(h, _SWEEP)
        assert mode == "unmeasured_high", f"holdout {h} must not be extrapolated"
        assert math.isinf(mde) and mde > 0, f"holdout {h} must fail CLOSED (+inf), got {mde}"


def test_the_worked_fail_open_case_now_refuses() -> None:
    """holdout=20000 is the brief's worked example: the sqrt branch computed MDE 0.315 < ceiling 0.50
    and ALLOWED the mine; the measured N^-0.21 law says 0.749 — REFUSE. Pin both the old fail-open and
    the new verdict, so a regression is unambiguous rather than a number nobody recognises."""
    assert _falsified_sqrt_law(20000) == pytest.approx(0.315, abs=1e-3)   # what it used to compute
    assert _falsified_sqrt_law(20000) < _CEILING                          # ...and it ALLOWED the mine
    assert _measured_law(20000) == pytest.approx(0.749, abs=1e-3)         # what the evidence says
    assert _measured_law(20000) > _CEILING                                # ...which is REFUSE
    mde, mode = interp_mde(20000, _SWEEP)
    assert mode == "unmeasured_high" and mde > _CEILING                   # the guard now refuses


def test_repair_never_understates_the_measured_law_off_grid() -> None:
    """The defect in one line: the returned MDE must never sit BELOW what the measured law implies.
    The old branch violated this at every h > h_hi; the sentinel satisfies it everywhere."""
    for h in (1012, 2028, 4044, 7955, 10210, 20000, 52826, 1_215_000):
        mde, _ = interp_mde(h, _SWEEP)
        assert mde >= _measured_law(h), f"holdout {h}: {mde} under-states measured {_measured_law(h)}"
        assert _falsified_sqrt_law(h) < _measured_law(h)      # the old branch DID under-state, at every h


def test_repair_is_monotone_stricter_everywhere_so_CRU_1_holds() -> None:
    """CRU-1: a gate repair may only ever make the verdict STRICTER, so no recorded verdict can move.
    Verified as a property over the whole domain, not asserted: for every holdout, the repaired MDE is
    >= the pre-fix MDE (higher MDE ⇒ more likely to exceed the ceiling ⇒ refuse). Nothing that was
    refused can become allowed."""
    for h in list(range(1, 1200, 7)) + [2028, 4044, 10210, 20000, 100_000]:
        new, _ = interp_mde(h, _SWEEP)
        old = _falsified_sqrt_law(h) if h > _H_HI else new   # at/below h_hi the repair is a no-op...
        assert new >= old, f"holdout {h} got LOOSER: {new} < {old}"


def test_measured_domain_is_byte_stable_so_recorded_stamps_survive() -> None:
    """The repair must not perturb any value the record already contains. Every power-stamped tick ever
    recorded sits at holdout=1011 / mode 'grid' / MDE 1.4025 (verified against the live orchestrator
    DBs, 2026-07-16) — the exact top grid point, which is why the fail-open branch was still LATENT.
    The flagship's extrapolated_low ≈4.45 provenance figure is likewise untouched."""
    assert interp_mde(_H_HI, _SWEEP) == (_M_HI, "grid")                   # the whole recorded record
    mde, mode = interp_mde(126, _SWEEP)                                   # the flagship stamp
    assert mode == "extrapolated_low" and mde == pytest.approx(_M_LO * (_H_LO / 126) ** 0.5)


def test_extrapolated_low_refuses_regardless_of_the_exponent() -> None:
    """The bottom branch keeps 1/√N — and is NOT conservative (at the measured -0.567 in-grid rate,
    sqrt under-states there too, the same direction as the bug). It is safe for a different reason,
    pinned here: it is bounded below by ``m_lo`` = 3.63, the LARGEST measured MDE, so it exceeds any
    plausible ΔSR ceiling and refuses whatever the true exponent. Its value is provenance, not a
    number the verdict turns on. If a future ceiling ever rises above 3.63, THIS test is the tripwire."""
    for h in (1, 30, 63, 100, 126, _H_LO - 1):
        mde, mode = interp_mde(h, _SWEEP)
        assert mode == "extrapolated_low" and mde >= _M_LO
        assert mde > _CEILING                                   # ⇒ refuse, independent of the exponent
    assert _M_LO > _CEILING, "the bottom branch's safety argument has expired — see the docstring"


def test_no_holdout_whatsoever_can_be_ALLOWED_by_this_calibration() -> None:
    """The invariant that makes every residual approximation error in this function harmless, and the
    honest summary of the funnel's state: with the shipped sweep the guard refuses EVERY substrate.
    Below/at the grid the MDE is floored by ``m_hi`` = 1.4025 (the DEEPEST measured point, 2.8x the
    ceiling); above it the answer is the +inf sentinel. So the chord's worst under-statement (-0.31%,
    in the concave 189->378 segment) and the low branch's (~3-12%) cannot flip a verdict — they move
    numbers that are 2.8x-7x clear of the ceiling. The falsified ``extrapolated_high`` branch was the
    ONLY path to an ALLOW, and the power it reported was fictitious. If this test ever goes red, the
    calibration has genuinely changed and the per-branch approximation arguments must be re-derived."""
    worst = min(interp_mde(h, _SWEEP)[0] for h in range(1, _H_HI + 1))
    assert worst == pytest.approx(_M_HI), "the measured domain is floored by the deepest grid point"
    assert worst > _CEILING, "an approximation error could now flip a verdict — re-derive the branches"


def test_degenerate_holdout_fails_closed_instead_of_crashing() -> None:
    """A zero/negative holdout used to raise ZeroDivisionError (h=0) or — worse — silently return a
    COMPLEX MDE (h<0) that would explode the guard's `> ceiling` compare. No holdout ⇒ nothing is
    tested ⇒ nothing is detectable."""
    for h in (0, -5):
        mde, mode = interp_mde(h, _SWEEP)
        assert mode == "unmeasured_degenerate"
        assert isinstance(mde, float) and math.isinf(mde) and mde > _CEILING


def test_unmeasured_stamp_drives_the_guard_to_refuse() -> None:
    """End-to-end: the sentinel must actually reach a REFUSE through the caller's real comparison
    (``implied_mde_delta_sr > ceiling``, orchestrator.py) — the fix is worthless if it stops at the
    stamp. T=80000 @ holdout_frac 0.25 ⇒ holdout 20000, the worked fail-open case."""
    guard = PowerGuard(enabled=True, ceiling=_CEILING, action="refuse")
    p = stamp_substrate_power(80_000, 0.25, _SWEEP, sweep_hash="abc123")
    assert p.holdout_bars == 20_000 and p.interp_mode == "unmeasured_high"
    assert p.implied_mde_delta_sr > guard.ceiling               # the guard's own test ⇒ mine refused
    assert not (guard.action == "refuse" and guard.force), "sanity: this config does refuse"


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

    from sharpen.signals.features import Panel
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
    from sharpen.crucible.orchestrator.substrate import folded_snapshot_hash, panel_content_hash
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
