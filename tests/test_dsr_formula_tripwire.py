"""Numeric tripwire for the Differential Sharpe Ratio formula (MATH-R01).

Closes finding P4-07 of the 2026-05-29 sg1-btc deep lifecycle audit: DSR had only
plumbing/regime tests, no assertion that pins the *formula*. A wrong exponent or a
dropped factor silently corrupts the training signal with NO failing test — the
most insidious bug class (a "leakage"-tier fault for the reward).

Canonical formula (Moody & Saffell, 2001 — ~/.claude/skills/math/FORMULAS.md MATH-R01):

    delta_A = R_t - A_{t-1}
    delta_B = R_t^2 - B_{t-1}
    DSR_t   = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / (B_{t-1} - A_{t-1}^2)^{3/2}
    A_t = A_{t-1} + eta * delta_A     (updated AFTER DSR uses the pre-update values)
    B_t = B_{t-1} + eta * delta_B

Hand-derivation of the golden value (eta=0.1, scale=1.0):
  step 1: R=0.02 -> warmup, returns 0.0; A=0.002, B=0.00004
  step 2: R=-0.01
    delta_A = -0.01 - 0.002          = -0.012
    delta_B = 0.0001 - 0.00004       =  0.00006
    prev_var = 0.00004 - 0.002^2     =  0.000036
    denom    = 0.000036 ** 1.5       =  2.16e-7
    numer    = 0.00004*(-0.012) - 0.5*0.002*0.00006 = -5.4e-7
    DSR      = -5.4e-7 / 2.16e-7     = -2.5   <-- golden

This single value is a mutation tripwire — it changes under every common corruption:
  * exponent 1.5 -> 0.5            -> denom 0.006   -> DSR ~ -9.0e-5   (FAIL)
  * drop the 0.5 factor           -> numer -6.0e-7  -> DSR ~ -2.778    (FAIL)
  * flip the 0.5 term sign (+)    -> numer -4.2e-7  -> DSR ~ -1.944    (FAIL)
  * use post-update A/B           -> different prev_var/numer          (FAIL)
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.envs.dsr import DSRCalculator

GOLDEN_STEP2_DSR = -2.5


def test_dsr_matches_hand_computed_golden():
    dsr = DSRCalculator(eta=0.1, scale=1.0)
    out1 = dsr.compute(0.02)
    assert out1 == 0.0, f"first step must be warmup (0.0), got {out1}"
    out2 = dsr.compute(-0.01)
    assert abs(out2 - GOLDEN_STEP2_DSR) < 1e-9, (
        f"DSR formula drift: expected {GOLDEN_STEP2_DSR}, got {out2}. "
        "A mismatch means the exponent (3/2), the 0.5 factor, the pre-update A/B "
        "convention, or the delta definitions were altered — see MATH-R01."
    )


def test_dsr_uses_pre_update_variance_not_post():
    """The denominator must use pre-update (B_{t-1}-A_{t-1}^2). If post-update
    values were used the golden would not reproduce; assert the exact value."""
    dsr = DSRCalculator(eta=0.1, scale=1.0)
    dsr.compute(0.02)
    assert abs(dsr.compute(-0.01) - GOLDEN_STEP2_DSR) < 1e-9


def test_dsr_scale_multiplies_output():
    """scale is applied to the raw DSR before clipping (golden * scale)."""
    base = DSRCalculator(eta=0.1, scale=1.0)
    base.compute(0.02)
    raw = base.compute(-0.01)

    scaled = DSRCalculator(eta=0.1, scale=3.0)
    scaled.compute(0.02)
    out = scaled.compute(-0.01)
    assert abs(out - raw * 3.0) < 1e-9, f"scale not applied: {out} != {raw}*3"


def test_dsr_clipped_to_pm10():
    """Output is clipped to [-10, 10]; never exceed the bound on any sequence."""
    dsr = DSRCalculator(eta=0.001, scale=1000.0)  # large scale forces clipping
    rng = np.random.default_rng(0)
    seen_clip = False
    for _ in range(500):
        r = float(rng.normal(0.0, 0.02))
        out = dsr.compute(r)
        assert -10.0 <= out <= 10.0, f"DSR out of clip bound: {out}"
        if abs(out) == 10.0:
            seen_clip = True
    assert seen_clip, "clip never engaged — large-scale path not exercised"


def test_dsr_reset_restores_warmup():
    """After reset(), the next compute() is a warmup step returning 0.0."""
    dsr = DSRCalculator(eta=0.1, scale=1.0)
    dsr.compute(0.02)
    dsr.compute(-0.01)
    dsr.reset()
    assert dsr.compute(0.02) == 0.0, "first compute after reset must be warmup (0.0)"
