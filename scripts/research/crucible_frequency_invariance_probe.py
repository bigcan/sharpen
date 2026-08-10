"""Does sampling FREQUENCY buy the Crucible funnel detection power? (pre-registered, S553-cont-151)

WHY THIS EXISTS
---------------
S553-cont-150 concluded that Crucible's depth blocker was "solved by measurement": extending the
cross-sectional MDE grid gave interpolated MDE **0.401** at holdout 19,325 on the 12-instrument
hourly Dukascopy panel, below the 0.50 ceiling, hence ALLOW. The 12-instrument panel was built as
"the first substrate to clear BOTH power constraints".

That reading has a UNIT problem, and this probe is the falsification test for it.

  * ``load_generation_config`` hardcodes ``periods_per_year=252.0`` (config.py:80, "the book is
    DAILY-marked"). Every MDE on the calibration curve is therefore an annualized ΔSR **in units of
    252 bars = 1 year**.
  * The calibration's synthetic panel stamps one bar per CALENDAR DAY, so on that substrate the
    252-bar year IS a calendar year and the curve reads correctly.
  * The hourly panel puts **5,694 bars in a calendar year** (measured: T=77,298 over 13.58y). Its
    holdout of 19,325 bars is **3.39 calendar years**, not 19,325 days. Read through a 252-bar-year
    convention, a calendar ΔSR of δ is REPORTED as δ·√(252/5694) = δ/4.75.

So the guard compares a 252-bar-year MDE against ``plausible_delta_sr_max: 0.50``, a ceiling whose
justification ("realistic single-signal marginal alpha ΔSR ~0.3-0.5", and the project's own TSMOM
anchor net SR 0.60) is stated in CALENDAR-annualized Sharpe. Two different units, one comparison.

PRE-REGISTERED PREDICTIONS (written before running; the file is the record)
--------------------------------------------------------------------------
P1. **Frequency-invariance.** At a MATCHED calendar span and a MATCHED calendar-annualized ΔSR, the
    detection rate of the shipped statistic (``corrected_contract._sharpe_diff_z`` at ``t_min``) is
    the SAME at 252 bars/yr and at 5,694 bars/yr. Equivalently the calendar-unit MDE depends on the
    holdout's calendar SPAN and not on how finely it is sampled. If P1 holds, sampling faster buys
    ZERO detection power and the cont-150 unblock is a relabelling of the axis, not a gain.
P2. **The 252-unit MDE falls with frequency** (this is the number cont-150 read): the SAME
    experiment reported in 252-bar-year units gives an MDE ~4.75x smaller at hourly than at daily.
    P1 and P2 are not in conflict — that ratio is exactly the unit conversion √(5694/252).
P3. **Calendar MDE tracks 3.17/√(holdout years)** — the textbook SE(SR_ann)=1/√years times the
    (z_{0.99}+z_{0.80}) two-error constant — within the sim's noise.

FALSIFIER: if the hourly arm detects a matched calendar ΔSR at a materially HIGHER rate than the
daily arm (beyond MC error), P1 is wrong, frequency does buy power, and cont-150 stands as written.

WHAT IS AND IS NOT MODELLED. Books are iid Gaussian per bar; ``aug`` is the equal-weight convex
blend of base and candidate, which reproduces the ρ≈0.99 base/aug correlation the real combiner
produces and which the Memmel paired variance depends on. Real books carry mild weight-persistence
autocorrelation; that is handled by the SHIPPED ``n_eff_mode='ar1'`` deflation, which is used here
rather than reimplemented. No claim is made about real-market return distributions: the question is
about the ESTIMATOR's resolution, which is a property of the statistic and the sample, not of the
alpha.

Usage:
    python scripts/research/crucible_frequency_invariance_probe.py [--seeds 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible.corrected_contract import _sharpe_diff_z  # noqa: E402
from finrl_pro_ds.signals.eval_harness import _ann_sharpe  # noqa: E402

# The two sampling conventions under test, at a MATCHED calendar span.
DAILY_PPY = 252.0
HOURLY_PPY = 5694.0        # measured on the 12-instrument Dukascopy panel (77,298 bars / 13.58y)
HOLDOUT_YEARS = 3.39       # the intraday panel's 19,325-bar holdout, in calendar years
BASE_SR_CAL = 0.60         # the project's validated TSMOM anchor, calendar-annualized
T_MIN = 2.33               # corrected-contract significance threshold (configs/..._contract.gates.yaml)
POWER_TARGET = 0.80


def _books(rng: np.random.Generator, *, n_bars: int, ppy: float, base_sr_cal: float,
           cand_sr_cal: float) -> tuple[np.ndarray, np.ndarray]:
    """One (base, aug) realization at ``ppy`` bars/year.

    A stream whose CALENDAR-annualized Sharpe is ``sr_cal`` has per-bar Sharpe ``sr_cal/√ppy``; that
    single line is the whole frequency mapping. Unit per-bar vol throughout — Sharpe is scale-free,
    and fixing vol keeps the two arms comparable. ``aug`` is the equal-weight blend, the shape the C1
    combiner produces, so ρ(base, aug) lands where the Memmel paired variance expects it.
    """
    base = rng.standard_normal(n_bars) + base_sr_cal / np.sqrt(ppy)
    cand = rng.standard_normal(n_bars) + cand_sr_cal / np.sqrt(ppy)
    return base, 0.5 * (base + cand)


def _detection_rate(*, ppy: float, cand_sr_cal: float, seeds: int, seed0: int
                    ) -> tuple[float, float, float]:
    """Fraction of seeds where the SHIPPED z clears ``T_MIN``, plus the mean realized ΔSR in
    calendar units and in the shipped 252-bar-year units."""
    n_bars = int(round(HOLDOUT_YEARS * ppy))
    hits, d_cal, d_252 = 0, [], []
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        base, aug = _books(rng, n_bars=n_bars, ppy=ppy, base_sr_cal=BASE_SR_CAL,
                           cand_sr_cal=cand_sr_cal)
        z, _sa, _sb, _n, _neff = _sharpe_diff_z(base, aug, n_eff_mode="ar1")
        if np.isfinite(z) and z >= T_MIN:
            hits += 1
        # The SAME realization read through both conventions: the true unit of the ΔSR the gate sees.
        d_cal.append(_ann_sharpe(aug, ppy) - _ann_sharpe(base, ppy))
        d_252.append(_ann_sharpe(aug, DAILY_PPY) - _ann_sharpe(base, DAILY_PPY))
    return hits / seeds, float(np.mean(d_cal)), float(np.mean(d_252))


def _mde(*, ppy: float, seeds: int, seed0: int, grid: np.ndarray) -> dict:
    """Smallest gridded candidate calendar-SR whose detection rate reaches ``POWER_TARGET``."""
    rows = []
    for cand_sr in grid:
        rate, dc, d252 = _detection_rate(ppy=ppy, cand_sr_cal=float(cand_sr), seeds=seeds,
                                         seed0=seed0)
        rows.append({"cand_sr_cal": float(cand_sr), "power": rate,
                     "realized_delta_sr_calendar": dc, "realized_delta_sr_252units": d252})
        if rate >= POWER_TARGET:
            break
    hit = next((r for r in rows if r["power"] >= POWER_TARGET), None)
    return {"ppy": ppy, "n_bars": int(round(HOLDOUT_YEARS * ppy)), "rows": rows,
            "mde_delta_sr_calendar": None if hit is None else hit["realized_delta_sr_calendar"],
            "mde_delta_sr_252units": None if hit is None else hit["realized_delta_sr_252units"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--out", default="results/crucible_power_units/frequency_invariance.json")
    args = ap.parse_args()

    grid = np.arange(0.2, 6.01, 0.2)
    daily = _mde(ppy=DAILY_PPY, seeds=args.seeds, seed0=1000, grid=grid)
    hourly = _mde(ppy=HOURLY_PPY, seeds=args.seeds, seed0=9000, grid=grid)

    law = 3.17 / np.sqrt(HOLDOUT_YEARS)     # P3: (z_.99 + z_.80)/√years

    print(f"\nHoldout span held FIXED at {HOLDOUT_YEARS} calendar years; base SR {BASE_SR_CAL} cal.")
    print(f"{'arm':<10}{'bars/yr':>10}{'n_bars':>10}{'MDE cal':>12}{'MDE 252u':>12}")
    for tag, r in (("daily", daily), ("hourly", hourly)):
        mc = r["mde_delta_sr_calendar"]
        m2 = r["mde_delta_sr_252units"]
        print(f"{tag:<10}{r['ppy']:>10,.0f}{r['n_bars']:>10,}"
              f"{'n/a' if mc is None else f'{mc:>12.3f}'}"
              f"{'n/a' if m2 is None else f'{m2:>12.3f}'}")

    verdict: dict = {"holdout_years": HOLDOUT_YEARS, "base_sr_calendar": BASE_SR_CAL,
                     "t_min": T_MIN, "power_target": POWER_TARGET, "seeds": args.seeds,
                     "textbook_law_3.17_over_sqrt_years": float(law),
                     "daily": daily, "hourly": hourly}
    mc_d, mc_h = daily["mde_delta_sr_calendar"], hourly["mde_delta_sr_calendar"]
    if mc_d is not None and mc_h is not None:
        ratio = mc_h / mc_d
        verdict["P1_frequency_invariance_calendar_mde_ratio"] = ratio
        verdict["P1_holds"] = bool(0.75 <= ratio <= 1.33)   # one grid step either way
        m2_d, m2_h = daily["mde_delta_sr_252units"], hourly["mde_delta_sr_252units"]
        verdict["P2_252unit_mde_ratio"] = m2_h / m2_d
        verdict["P2_expected_unit_conversion"] = float(np.sqrt(DAILY_PPY / HOURLY_PPY))
        print(f"\nP1  calendar MDE ratio hourly/daily = {ratio:.3f}   "
              f"=> frequency-invariance {'HOLDS' if verdict['P1_holds'] else 'REFUTED'}")
        print(f"P2  252-unit MDE ratio hourly/daily = {m2_h / m2_d:.3f}  "
              f"(pure unit conversion would be {np.sqrt(DAILY_PPY / HOURLY_PPY):.3f})")
        print(f"P3  textbook 3.17/sqrt({HOLDOUT_YEARS}) = {law:.3f} vs measured daily "
              f"{mc_d:.3f} / hourly {mc_h:.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
