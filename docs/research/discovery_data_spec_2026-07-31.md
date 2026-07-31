# What data would actually unblock discovery — solved, and a correction to my own spec

**Date:** 2026-07-31 (S553-cont-146). Closing artifact for the session's 14-probe arc.

## Why this exists

After fourteen negatives I concluded the binding constraint was data and stated a spec: *"~8,000+
holdout bars at ≤0.5bp round-trip."* I then asserted such data is "a purchase, not a search"
**without testing it**. Both halves needed checking. Testing them produced a correction.

## 1. Free intraday depth — MEASURED, not assumed

`yfinance` hourly bars available for liquid futures (all six queried):

| symbol | 1h/730d | 1h/max | 1d/max |
|---|---|---|---|
| ES=F | 13,682 | 11,430 | 6,531 |
| NQ=F | 13,680 | 11,430 | 6,531 |
| CL=F | 13,515 | 11,306 | 6,512 |
| GC=F | 13,723 | 11,461 | 6,503 |
| ZN=F | 13,645 | 11,383 | 6,493 |
| 6E=F | 13,721 | 11,459 | 6,533 |

**The free intraday ceiling is ~13,700 hourly bars — a hard 730-day API cap, not a rate limit.**
Daily history is long (~26 years) but daily is far too shallow for the funnel.

## 2. The spec, solved JOINTLY — and my earlier version was wrong

The error: "8,000 holdout bars at ≤0.5bp" optimises **depth alone**. But holding period `H` drives
**both** gates in opposite directions:

- `N_eff = holdout / (2H − 1)` — power wants **short** holding;
- `drag = turnover · (bars_yr / H) · bps / vol` — economics wants **long** holding.

Solving both at once for ES-class parameters (6,000 hourly bars/yr, 0.3bp round trip, 15% book vol,
0.70 turnover per rebalance), against the measured corrected-contract MDE curve:

| holding | cost drag | min years | total bars | MDE at that depth | viable |
|---|---|---|---|---|---|
| 1 hr | **0.84** | — | — | — | **impossible at ANY depth** |
| **2 hr** | **0.42** | **10.8** | **64,800** | 0.497 | ✓ **minimum** |
| 3 hr | 0.28 | 18.0 | 108,000 | 0.497 | ✓ |
| 4 hr | 0.21 | 25.2 | 151,200 | 0.497 | ✓ |
| 8 hr | 0.11 | 54.0 | 324,000 | 0.497 | ✓ |

**Minimum viable spec: ~11 years of hourly data on an ES-class instrument (64,800 bars), traded at
2-hour holding, ~0.3bp round trip.**

Two consequences that were not in the earlier statement:

- **Sub-2-hour holding is structurally excluded at any depth.** At hourly rebalancing the cost drag
  (0.84) exceeds every MDE that could clear the 0.50 ceiling. No amount of data fixes it.
- **Longer holding needs MORE total data, not less** — `N_eff` shrinks as `2H−1`, so the total-bars
  requirement rises faster than the drag falls. H=2 is the minimum-data corner.

## 3. The verdict on "free data first"

| source | bars | holdout | MDE | drag | outcome |
|---|---|---|---|---|---|
| yfinance ES=F hourly | 13,682 | 3,420 | 0.654 | 0.96 | too shallow **and** cost-killed at H=1 |
| yfinance CL=F hourly | 13,515 | 3,378 | 0.663 | 0.95 | same |
| crypto perp hourly (on disk) | 37,000 | 9,250 | **0.401** | **2.04** | **depth clears, economics kills** |
| **required** | **64,800** | — | 0.497 | 0.42 | ✓ |

**Shortfall: 4.7×**, against a hard API cap.

This is the lock stated exactly: **crypto has the depth and not the economics; liquid futures have
the economics and not the depth.** Free data provides each half separately and neither together.

## 4. What this changes

- The session's conclusion — discovery is data-blocked — is now **measured on both sides** rather
  than inferred from probe failures.
- The requirement is a **purchasable, checkable spec**: ~11y of hourly ES/NQ/CL-class bars. That is
  ordinary vendor data (CME via Databento/Polygon/IQFeed etc.), not exotic.
- **A prior claim of mine is corrected.** "~8,000 holdout bars at ≤0.5bp" understated the
  requirement ~4.7× by optimising one gate. Anyone acting on the old number would have bought too
  little data and re-run into the same wall.

## Reproduce

The measured yfinance ceilings and the joint solve are both reproducible from the commands in this
session's log; the MDE curve is `results/crucible_intraday_power/intraday_power_corrected.json`.
