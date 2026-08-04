"""What breadth would a Crucible substrate need to be worth mining? (S553-cont-151)

Two measurements this session pin the problem from both ends, and together they give a NUMBER rather
than the standing slogan "breadth binds":

  * DETECTION FLOOR. `crucible_frequency_invariance_probe.py` measured MDE(calendar ΔSR) ~ 1.39-1.47
    at a 3.39-year holdout, and confirmed it is set by the holdout's CALENDAR SPAN alone —
    SE(annualized Sharpe) = 1/√(calendar years), invariant to bar spacing. No amount of resampling
    moves it. So the funnel can only ever promote an alpha whose TRUE marginal ΔSR clears ~1.4.
  * ACHIEVABLE EFFECT. The fundamental law of active management: IR = IC · √BR, with BR the number of
    INDEPENDENT bets per year = (effective independent instruments) x (rebalances per year). This is
    the only term a substrate designer controls once the history length is fixed.

Setting achievable IR >= detection floor and solving for effective breadth turns "get more breadth"
into a target that can be checked against a candidate instrument list BEFORE any data is fetched.

The measured shortfall on the current 12-instrument hourly panel is the point of this script: its
correlation matrix has a participation ratio of 5.01, not 12 (nine of the twelve are USD crosses, and
EURGBP/EURJPY are triangular combinations of the others at rho ~0.999). Adding more USD crosses adds
columns and almost no breadth.

Usage:
    python scripts/research/crucible_breadth_requirement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Measured this session.
MDE_CALENDAR = 1.39          # frequency_invariance_probe, hourly arm, 3.39y holdout
BARS_PER_YEAR = 5694.0
HOLD_BARS = 21               # candidate rebalance cadence on the intraday substrate
N_EFF_MEASURED = 5.01        # participation ratio of the 12-instrument return correlation matrix

#: Plausible per-bet cross-sectional IC. 0.02-0.03 is the standard range for a real, surviving
#: cross-sectional signal; 0.05 is generous. Quoted as a RANGE on purpose — the answer is a
#: sensitivity, not a point estimate, and the honest output is "how much breadth for each IC you
#: are willing to assume".
IC_GRID = (0.015, 0.02, 0.025, 0.03, 0.05)


def effective_breadth(n_eff: float, rebalances_per_year: float) -> float:
    return float(n_eff) * float(rebalances_per_year)


def main() -> int:
    rpy = BARS_PER_YEAR / HOLD_BARS
    print(f"substrate: {BARS_PER_YEAR:,.0f} bars/yr, hold {HOLD_BARS} bars "
          f"=> {rpy:.0f} rebalances/yr")
    print(f"detection floor (measured): true marginal calendar dSR must exceed {MDE_CALENDAR:.2f}\n")

    print(f"{'IC':>7}{'IR now (n_eff=5.01)':>22}{'n_eff NEEDED':>15}{'x more breadth':>17}")
    rows = []
    for ic in IC_GRID:
        ir_now = ic * np.sqrt(effective_breadth(N_EFF_MEASURED, rpy))
        # IR = IC*sqrt(n_eff*rpy) >= MDE  =>  n_eff >= (MDE/IC)^2 / rpy
        n_needed = (MDE_CALENDAR / ic) ** 2 / rpy
        rows.append((ic, ir_now, n_needed, n_needed / N_EFF_MEASURED))
        print(f"{ic:>7.3f}{ir_now:>22.2f}{n_needed:>15.1f}{n_needed / N_EFF_MEASURED:>16.1f}x")

    print("\nREADING. At the IC range a real cross-sectional signal actually delivers (0.02-0.03),")
    print("the current panel's achievable IR is BELOW the funnel's own detection floor — the")
    print("substrate cannot produce a detectable alpha even if a genuine signal is present. The")
    print("shortfall is in EFFECTIVE instruments, and it is the one term still under our control:")
    print("history length is fixed by what is free, and bar frequency is measured not to matter.")
    print("\nTarget for any future substrate: count PARTICIPATION RATIO, not tickers. Nine USD")
    print("crosses are ~1 factor; the decorrelated axes are asset CLASSES (rates, credit, equity")
    print("indices, commodities, crypto) and, within a class, genuinely separate names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
