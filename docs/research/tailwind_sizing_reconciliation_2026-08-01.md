# TAILWIND sizing reconciliation — capital-gate item (2), measured

**Date:** 2026-08-01 · Closes the open item from
`docs/research/tailwind_v1_forward_path_render_2026-07-31.md` finding A2.

## The defect, restated precisely

The render established that the challenge's sizing is **certified by evidence that cannot see it**:
the research basis (`portfolio_frontier.risk_parity`) renormalises each sleeve's *return series* to
10% vol with a full-sample constant, so `target_vol_asset` / `lev_cap` are a numerical **no-op**
there (max deviation 1.7e-17). The paper executor takes a different route — it combines sleeve
*weights* via `combine_sleeve_weights` and reads the levers straight from config
(`cross_asset_loader.py:458-459`) — so on that path they **bind**.

Per-sleeve weight construction is in fact **identical** in both paths:
`w = clip(conviction · min(target_vol_asset/σ, lev_cap), ±lev_cap)`
(`multi_asset_allocator_env.py:391` vs `xsec_momentum_falsification.vol_scaled_weights`).
**The divergence is entirely in the combine.**

## Measured (not calculated) — executor realised vol vs levers

Driven through the real `TwoSleeveExecutor` on the repo's synthetic tailwind bundle
(`tests/paper/test_tailwind_executor._tailwind_bundle`, T=1200, 18 assets):

| target_vol_asset | lev_cap | max_gross | **realised book vol** | mean gross |
|---|---|---|---|---|
| 0.10 | 2.0 | 3.0 *(own-capital cfg)* | **8.49%** | 1.70 |
| **0.15** | **3.0** | **4.5** *(challenge cfg)* | **12.75%** | **2.54** |
| 0.0324 | 2.0 | 3.0 | 7.03% | 1.41 |
| 0.10 | 2.0 | *cap off* | 24.00% | 4.77 |
| 0.15 | 3.0 | *cap off* | 36.36% | 7.16 |

**Findings:**

1. **`target_vol_asset` is the effective lever, and it is linear.** 0.10 → 8.49%, 0.15 → 12.75%;
   ratio **1.50**, exactly the lever ratio. The config's stated mechanism is correct on this path.
2. **The gross cap does NOT bind** at either setting — mean gross 1.70 and 2.54 against caps of 3.0
   and 4.5.
3. **The research basis is pinned at 6.92% regardless.** So the executor at challenge levers runs
   **~1.8× the vol of the book whose DSR, PBO and P(pass) were certified**, and no amount of
   research-basis evidence can constrain it.

### Two of my own analytical predictions were WRONG, and measuring caught both

Before running this I calculated, from measured sleeve vols and a static inverse-vol α, that the
executor would run ~30.9% vol with **gross ≈16.5 so the cap would BIND hard**, and concluded that
`max_gross_exposure` — not `target_vol_asset` — was the real vol lever. **Both are wrong.** Measured
gross is 1.70/2.54 (cap clears comfortably) and `target_vol_asset` is exactly the lever the config
says it is. The static-α approximation ignored that the executor's α are *trailing* and that
conviction is bounded, which together keep gross far below the closed-form estimate.

This is the session's recurring lesson applied to my own arithmetic: **a calculation that runs is
not a calculation that is right.**

## What reconciliation actually requires

The two mechanisms cannot be made to agree by editing a config, because the research basis
**deliberately** discards scale (that is what makes its Sharpe/DSR leverage-invariant). The honest
reconciliation is a **measured mapping**, not an equation:

> `target_vol_asset` → executor realised book vol, measured on the **real** bundle, then set so the
> realised vol equals the intended effective vol.

The numbers above are on the **synthetic** bundle, whose per-asset vol (~15.9% ann) is a modelling
choice, so they establish the *mechanics and linearity* — not the production calibration. **The
production step is to re-run this table on the real 18-ETF bundle** and read `target_vol_asset` off
it for the 10% target the forward-path render cleared.

## Status of capital-gate item (2)

**Advanced, not closed.** The mechanism is now understood and measured, the direction and linearity
are established, and the false "max_gross is the real lever" hypothesis is eliminated. Remaining:
the same table on the real bundle. Items (3) Tier-2 audit, (4) six protocol-v2 wiring gaps, and (5)
operator go-ahead are untouched.

**A challenge attempt remains BLOCKED.**
