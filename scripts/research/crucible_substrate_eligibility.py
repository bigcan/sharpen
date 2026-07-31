"""Which free substrates on disk can actually clear Crucible's power guard?

The recurring answer to "why can't Crucible discover" has been "breadth" or "U8". The measured
answer is sharper and it is a MISMATCH:

  * where the data is DEEP (TX intraday, N_eff 10^4-10^5) the ECONOMICS are closed — TX
    single-market intraday is cost-killed at the ~1bp floor and its whole workstream is a NO-GO
    (0/10 rescue filters), so power there buys nothing;
  * where the ECONOMICS work (daily Taiwan small/mid-cap, daily cross-asset ETFs) the data is
    SHALLOW — N_eff ~10^3, where the measured MDE is 1.1-1.8 against a 0.50 ceiling.

So the binding constraint is not breadth (measured NOT to lower MDE: 0.420 at N=12 vs 0.644 at
N=100, same depth) and not the gate. It is the absence of a substrate that is simultaneously deep
ENOUGH and cheap ENOUGH. This script makes that concrete: for every substrate actually on disk it
reports the holdout depth, the effective N at a range of holding periods, and the MDE read off the
MEASURED corrected-contract curve — refusing (as the production guard does) to extrapolate off the
top of the grid.

It is a READ-ONLY eligibility calculator. It mines nothing, changes no gate, and issues no verdict
about any strategy — a substrate clearing the power guard means the funnel COULD detect a marginal
edge there, not that one exists.

Usage:
    python scripts/research/crucible_substrate_eligibility.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Measured corrected-contract MDE-vs-N_eff curves. Populated from the intraday sweeps; the deep
# run extends the top anchor. Loaded at runtime so this never hardcodes a measurement.
CURVE_FILES = (
    ROOT / "results" / "crucible_intraday_power_deep" / "intraday_power_corrected.json",
    ROOT / "results" / "crucible_intraday_power" / "intraday_power_corrected.json",
)
CEILING = 0.50          # the funnel's MDE ceiling — NEVER moved to manufacture an ALLOW (v8.0)
# Cost-drag assumptions, stated explicitly so they can be argued with. Back-solved from the MEASURED
# P1@63d point (turnover_ann 2.95, cost_wall 0.190 @0.21%) => book_vol 3.3%.
# Book vol is PER-SUBSTRATE — a single constant is wrong. The 3.3% below is back-solved from the
# ~200-name Taiwan L/S book; a single-instrument futures overlay carries the instrument's own vol
# (~6x higher), so reusing 3.3% there would inflate its cost drag ~6x and manufacture a false
# NO-GO. Each entry names its basis.
TURNOVER_PER_REBAL = 0.70    # fraction of book traded each rebalance (P1@63d implies ~0.74)
HOLDOUT_FRAC = 0.25


def load_curve() -> tuple[list[tuple[float, float]], str]:
    """Merge every available measured corrected-contract H=1 curve (deepest file wins on ties)."""
    pts: dict[float, float] = {}
    used: list[str] = []
    for f in reversed(CURVE_FILES):          # shallow first, deep overwrites
        if not f.exists():
            continue
        used.append(f.name)
        for c in json.loads(f.read_text(encoding="utf-8")).get("primary_h1", []):
            m = c.get("mde_realized_delta_sr")
            if m:
                pts[float(c["n_eff"])] = float(m)
    return sorted(pts.items()), ", ".join(used) or "NONE"


def interp_mde(n_eff: float, pts: list[tuple[float, float]]) -> tuple[float | None, str]:
    """Log-log interpolation between MEASURED anchors. Off the top => refuse, mirroring
    `finrl_pro_ds/crucible/orchestrator/substrate.interp_mde`, which returns +inf there because a
    power law extrapolated past the grid UNDER-states MDE and fails OPEN."""
    if not pts:
        return None, "no_curve"
    lo, hi = pts[0][0], pts[-1][0]
    if n_eff < lo:
        return None, "below_grid"
    if n_eff > hi:
        return None, "unmeasured_high(REFUSE)"
    xs = [math.log(a) for a, _ in pts]
    ys = [math.log(b) for _, b in pts]
    x = math.log(n_eff)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            s = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
            return math.exp(ys[i - 1] + s * (x - xs[i - 1])), "interp"
    return math.exp(ys[-1]), "interp"


def cost_drag_sharpe(rebalances_per_year: float, cost_rt: float, book_vol: float,
                     turnover_per_rebalance: float) -> float:
    """Annualized Sharpe lost to costs.

        drag = turnover_ann * cost_rt / book_vol,   turnover_ann = turnover_per_rebal * rebals/yr

    Calibrated against a MEASURED point rather than asserted: P1 at 63-day holding recorded
    `turnover_ann = 2.95` and `cost_wall = 0.190` at the 0.21% standard cost model
    (`results/taiwan_smallcap_altdata_lowturn/scorecard.json`), which back-solves to a long-short
    book vol of 2.95*0.0021/0.190 = 3.3% — the right order for a dollar-neutral cross-sectional
    book. Indicative, not a substitute for running the real cost model.

    THE POINT of this function: N_eff ∝ 1/(2H−1) but rebalances/yr ∝ 1/H, so POWER wants SHORT
    holding and ECONOMICS wants LONG holding. They pull in opposite directions on the same axis.
    """
    turnover_ann = turnover_per_rebalance * rebalances_per_year
    return turnover_ann * cost_rt / max(book_vol, 1e-9)


# (label, bars_per_name, n_names, bar_minutes, round-trip cost bps, book_vol, cost source)
SUBSTRATES = [
    ("Taiwan smallcap (daily)",   5292,  612, 1440, 21.0, 0.033,
     "0.30% sell tax + comm; vol MEASURED back-solve from P1@63d"),
    ("Cross-asset ETF (daily)",   5133,   18, 1440,  2.0, 0.069,
     "TSMOM 2bp; vol = tailwind combined native 6.92% (measured)"),
    ("Crypto perp (hourly)",     37000,   10,   60,  4.0, 0.250,
     "majors taker ~2-5bp RT, no tax; 10-name dollar-neutral ~25%"),
    ("TX futures (15-min)",     125543,    1,   15, 10.0, 0.200,
     "CLOSED: ~1bp/trade floor, 0/10 rescues; single-instrument ~20%"),
    ("TX futures (3-min)",      625679,    1,    3, 10.0, 0.200, "CLOSED (same)"),
    ("TX futures (1-min)",     1859954,    1,    1, 10.0, 0.200, "CLOSED (same)"),
    # REFERENCE ROW — not on disk. The profile that WOULD open a cell: an ultra-liquid future
    # (ES/NQ) at hourly bars over ~15y. Cost ~0.3bp RT (1 tick on a $50 multiplier) is 10-30x
    # cheaper than TX or crypto, and 15y of hourly bars is ~24.6k bars. Shown to make the
    # requirement concrete rather than leaving "we need better data" as a vague wish.
    ("[ref] ES-class 1h, 15y",   24570,    1,   60,  0.3, 0.150,
     "NOT ON DISK — the profile the constraint implies"),
]
HOLDINGS_BARS = (1, 2, 4, 8, 21, 63)


def main() -> int:
    pts, src = load_curve()
    if not pts:
        print("No measured corrected-contract curve found. Run:\n"
              "  python scripts/research/crucible_intraday_power.py --contract corrected")
        return 1
    lo, hi = pts[0][0], pts[-1][0]
    print("=" * 100)
    print("CRUCIBLE SUBSTRATE ELIGIBILITY — corrected contract, MEASURED curve only")
    print(f"curve source: {src}   measured N_eff span [{lo:,.0f}, {hi:,.0f}]   ceiling {CEILING}")
    print("=" * 100)
    print("A substrate is ELIGIBLE where MDE <= ceiling by INTERPOLATION. Off the top of the grid the")
    print("production guard REFUSES (fail-closed) — those cells are not eligibility, they are ignorance.\n")

    rows = []
    for label, bars, n_names, bar_min, cost_bps, book_vol, cost_src in SUBSTRATES:
        holdout = int(bars * HOLDOUT_FRAC)
        print(f"{label}   T={bars:,} bars/name  N={n_names}  holdout={holdout:,}  "
              f"cost~{cost_bps:.1f}bp RT  bookVol~{book_vol:.1%}  [{cost_src}]")
        best = None
        # bars available per year for this substrate (calendar-aware: crypto is 24/7)
        bars_per_year = (365 * 1440 / bar_min) if "Crypto" in label else (
            252 * (1440 / bar_min if bar_min >= 1440 else 300 / bar_min))
        for h in HOLDINGS_BARS:
            n_eff = holdout / (2 * h - 1)
            mde, mode = interp_mde(n_eff, pts)
            hold_lbl = (f"{h*bar_min/1440:.0f}d" if h * bar_min >= 1440 else f"{h*bar_min:.0f}m")
            drag = cost_drag_sharpe(bars_per_year / h, cost_bps / 1e4, book_vol, TURNOVER_PER_REBAL)
            if mde is None:
                print(f"     hold {hold_lbl:>5}  N_eff={n_eff:>10,.0f}   MDE=  --    "
                      f"costDrag={drag:>7.2f}  {mode}")
            else:
                powered = mde <= CEILING
                # A cell is VIABLE only if the funnel can detect an edge AND that edge survives cost.
                viable = powered and (mde > drag)
                if viable and best is None:
                    best = (hold_lbl, n_eff, mde, drag)
                tag = ("VIABLE" if viable else
                       ("cost-killed" if powered else "under-powered"))
                print(f"     hold {hold_lbl:>5}  N_eff={n_eff:>10,.0f}   MDE={mde:>5.3f}  "
                      f"costDrag={drag:>7.2f}  {tag}")
        rows.append((label, best, cost_bps, cost_src))
        print()

    print("-" * 100)
    print("SUMMARY — eligible (deep enough) vs tradeable (cheap enough) are DIFFERENT gates:\n")
    for label, best, cost_bps, cost_src in rows:
        if best is None:
            print(f"  {label:<28} NO viable cell — powered and cost-survivable never coincide")
        else:
            hold_lbl, n_eff, mde, drag = best
            print(f"  {label:<28} VIABLE from {hold_lbl:>5} hold "
                  f"(N_eff {n_eff:,.0f}, MDE {mde:.3f}, drag {drag:.2f})")
    print("\nEligible != an edge exists. It means the funnel could DETECT a marginal ΔSR at that")
    print("depth. Finding one still requires a pre-registered hypothesis or a mining run.")
    print("-" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
