"""POWER-LORD-01 — the power guard must be able to stamp at the LORD++ level production RUNS.

The substrate-power curve used to be scored only at ``fresh_lord_level`` — the level a first-ever
test spends — while production runs a PERSISTENT per-substrate account whose level decays over a
barren stream. So the guard stamped power at a threshold production does not use; the mismatch is
one-directional (levels only decay), grows with every test the substrate has ever run, and was
invisible in the stamp. Measured on us_equity: MDE 1.00x fresh -> 1.33x after 8 tests.

These tests pin the four properties the fix depends on:

1. **The legacy path is byte-identical.** Everything below defaults to the fresh reading, so a
   caller that knows nothing about account depth behaves exactly as before.
2. **Depth selection is conservative.** A deeper account has a TIGHTER level and a HIGHER MDE, so
   the grid lookup rounds UP. Rounding down would hand back more power than exists — fail-open.
3. **A missing family degrades LOUDLY.** Surfaces built before the schema (the deep sweep) carry no
   family; the stamp falls back to fresh and says so in ``interp_mode``. Falling back is the status
   quo, so it is not a regression — but a fresh-level stamp must never masquerade as a live one.
4. **Emitting the family is RNG-neutral.** The extra levels are post-processing of one scored draw,
   so the fresh-level power and MDE are unchanged to the bit. This is what lets a re-run of the
   sweep be compared against the pre-family run as a determinism check.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finrl_pro_ds.crucible.orchestrator.substrate import (  # noqa: E402
    _pooled_points,
    interp_mde,
    lord_depth_for,
    stamp_substrate_power,
)

CT = "cross_sectional"


def _sweep(*, with_family: bool = True) -> dict:
    """A two-depth cross-sectional surface. MDEs rise with account depth (the measured direction)."""
    def row(t, n, hb, fresh, fam):
        r = {"t": t, "n": n, "holdout_bars": hb, "mde_realized_delta_sr": fresh}
        if with_family:
            r["mde_by_lord_tests"] = {str(k): v for k, v in fam.items()}
        return r

    sw = {
        "candidate_type": CT, "contract": "corrected",
        "rows": [
            row(4044, 12, 1011, 1.50, {0: 1.50, 8: 1.90, 32: 2.10}),
            row(4044, 100, 1011, 1.10, {0: 1.10, 8: 1.40, 32: 1.60}),
            row(8064, 12, 2016, 0.83, {0: 0.83, 8: 1.05, 32: 1.20}),
            row(8064, 100, 2016, 0.89, {0: 0.89, 8: 1.12, 32: 1.30}),
        ],
    }
    if with_family:
        sw["lord_tests_grid"] = [0, 8, 32]
        sw["lord_levels"] = {"0": 2.1875e-2, "8": 6.5035e-4, "32": 4.4663e-5}
    return {"mde_sweep": sw}


# --------------------------------------------------------------------------- #
# 1. the legacy path does not move
# --------------------------------------------------------------------------- #
def test_pooled_points_default_is_the_fresh_field():
    """Pooling takes the WORST MDE per depth, read from the legacy field."""
    assert _pooled_points(_sweep()) == [(1011, 1.50), (2016, 0.89)]


def test_family_presence_does_not_change_the_default_reading():
    """Adding the family to a surface must not perturb what a legacy caller sees."""
    assert _pooled_points(_sweep(with_family=True)) == _pooled_points(_sweep(with_family=False))
    assert interp_mde(1726, _sweep(with_family=True)) == interp_mde(1726, _sweep(with_family=False))


def test_stamp_without_count_is_the_fresh_stamp():
    sw = {CT: _sweep()}
    a = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,))
    b = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,), lord_tests_already=None)
    assert a.implied_mde_delta_sr == b.implied_mde_delta_sr
    assert a.interp_mode == b.interp_mode == "interpolated"


# --------------------------------------------------------------------------- #
# 2. depth selection is conservative (rounds UP)
# --------------------------------------------------------------------------- #
def test_lord_depth_rounds_up_never_down():
    sw = _sweep()
    assert lord_depth_for(sw, 0) == 0
    assert lord_depth_for(sw, 1) == 8, "an account with 1 test must not read the FRESH curve"
    assert lord_depth_for(sw, 8) == 8
    assert lord_depth_for(sw, 9) == 32
    assert lord_depth_for(sw, 32) == 32


def test_lord_depth_clamps_past_the_grid():
    """Past the deepest measured depth, the closest measured statement is the deepest one."""
    assert lord_depth_for(_sweep(), 10_000) == 32


def test_lord_depth_is_none_without_a_family():
    assert lord_depth_for(_sweep(with_family=False), 8) is None


# --------------------------------------------------------------------------- #
# 3. the family read, and the loud fallback
# --------------------------------------------------------------------------- #
def test_deeper_account_never_reports_more_power():
    """The whole point: a depleted account must stamp an MDE >= the fresh one."""
    sw = {CT: _sweep()}
    fresh = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,))
    prev = fresh.implied_mde_delta_sr
    for k in (0, 8, 32):
        s = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,), lord_tests_already=k)
        assert s.implied_mde_delta_sr >= prev - 1e-12, f"MDE fell at depth {k} — fail-OPEN"
        assert s.interp_mode.endswith(f":lord{k}")
        prev = s.implied_mde_delta_sr
    assert prev > fresh.implied_mde_delta_sr, "test fixture is vacuous — no depletion at all"


def test_missing_family_degrades_to_fresh_and_says_so():
    sw = {CT: _sweep(with_family=False)}
    fresh = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,))
    degraded = stamp_substrate_power(4930, 0.35, sw, "h", candidate_types=(CT,),
                                     lord_tests_already=8)
    assert degraded.implied_mde_delta_sr == fresh.implied_mde_delta_sr
    assert degraded.interp_mode.endswith(":lord_unmeasured"), (
        "a fresh-level stamp must never masquerade as a live-level one")


def test_null_family_entry_is_unmeasured_not_zero():
    """A depth that never reached target power is DROPPED, not read as a 0.0 MDE."""
    sw = _sweep()
    for r in sw["mde_sweep"]["rows"]:
        r["mde_by_lord_tests"]["32"] = None
    assert _pooled_points(sw, 32) == []
    assert interp_mde(1726, sw, 32) == (math.inf, "unmeasured_empty")


def test_interp_mde_family_does_not_silently_retry_at_fresh():
    """`interp_mde` itself must not paper over a missing depth — only the caller may degrade."""
    sw = _sweep(with_family=False)
    assert interp_mde(1726, sw, 8) == (math.inf, "unmeasured_empty")


# --------------------------------------------------------------------------- #
# 4. emitting the family is RNG-neutral (the determinism claim in the docstring)
# --------------------------------------------------------------------------- #
def test_family_emission_does_not_perturb_the_fresh_curve():
    """Scoring once and thresholding per level must reproduce the fresh-level curve exactly.

    This is what lets the refined-grid sweep be re-run with the family and compared bit-for-bit
    against the pre-family run. Deliberately tiny (t=1512, n=12, 3 betas, 3 seeds) so it stays a
    unit test.
    """
    from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig
    from research.crucible_calibration import _xsec_power_curve, load_calib

    cc = load_calib(ROOT / "configs" / "crucible_calibration.gates.yaml",
                    ROOT / "configs" / "us_equity_signal_eval.gates.yaml")
    corrected = CorrectedConfig.from_yaml(ROOT / "configs" / "us_equity_corrected_contract.gates.yaml")
    kw = dict(t=1512, n=12, betas=[0.0, 0.0005, 0.0015], n_seeds=3,
              cost_bps=cc.ek["cost_bps"], corrected=corrected, holdout_frac=0.25,
              hold_horizon=21, min_names=6)

    plain = _xsec_power_curve(cc, **kw)                                  # fresh only
    withfam = _xsec_power_curve(cc, **kw, lord_tests_grid=[0, 8, 32])    # family emitted

    for a, b in zip(plain, withfam, strict=True):
        assert a["beta"] == b["beta"]
        assert a["power"] == b["power"], "fresh-level power moved when the family was emitted"
        assert a["detections"] == b["detections"]
        assert a["lord_level"] == b["lord_level"]
        assert (math.isnan(a["mean_realized_delta_sr"]) and math.isnan(b["mean_realized_delta_sr"])) \
            or a["mean_realized_delta_sr"] == b["mean_realized_delta_sr"]
        # depth 0 IS the fresh level, so it must reproduce `power` exactly
        assert b["power_by_lord_tests"]["0"] == b["power"]
        # and deeper depths can only detect LESS
        assert b["power_by_lord_tests"]["8"] <= b["power_by_lord_tests"]["0"]
        assert b["power_by_lord_tests"]["32"] <= b["power_by_lord_tests"]["8"]
