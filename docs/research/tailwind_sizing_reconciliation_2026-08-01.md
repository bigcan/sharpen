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


---

# PART 2 — RESOLVED on the REAL bundle: `max_gross_exposure` is the lever

Part 1's table was measured on the **synthetic** bundle and concluded `target_vol_asset` was the
effective, linear lever. **On the real 18-ETF bundle that is wrong.** Re-measured through the same
`TwoSleeveExecutor`, real data, 5,176 steps 2006-01-03 → 2026-07-31, both sleeves live
(`momentum` + `defensive`), data-integrity gate OK, `assert_causal` PASS on both signal stacks.

## The three levers, isolated

**`target_vol_asset` — saturates, barely matters** (lev_cap 2.0, max_gross 3.0):

| target_vol_asset | 0.05 | 0.075 | 0.10 |
|---|---|---|---|
| realised vol | 9.08% | 9.66% | 9.83% |

Doubling it moves realised vol by **8%**. Real ETF vols are low and heterogeneous (bonds ~5%), so
`target_vol/σ` saturates against the cap for most assets.

**`lev_cap` — saturates, barely matters** (target_vol 0.10, gross cap OFF):

| lev_cap | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| realised vol | 24.55% | 26.12% | 26.47% | 26.56% | 26.57% |

**`max_gross_exposure` — LINEAR, and it is the binding lever** (target_vol 0.10, lev_cap 2.0):

| max_gross | 1.0 | 1.5 | 2.0 | 2.5 | **3.0** | **4.5** | 6.0 |
|---|---|---|---|---|---|---|---|
| realised vol | 3.44% | 5.07% | 6.68% | 8.26% | **9.83%** | **14.13%** | 17.81% |
| mean gross | 0.72 | 1.08 | 1.44 | 1.80 | 2.16 | 3.21 | 4.24 |

vol / max_gross = 0.0344, 0.0338, 0.0334, 0.0330, 0.0328 — **spread 1.049×, essentially constant.**

## The reconciliation (this closes capital-gate item 2)

> **executor realised vol ≈ 0.0334 × `max_gross_exposure`**, on the real bundle.
> `target_vol_asset` and `lev_cap` are effectively free parameters — both saturate.

| configuration | max_gross | **executor vol** |
|---|---|---|
| own-capital `tailwind_v1.yaml` | 3.0 | **9.83%** |
| challenge `tailwind_v1_challenge.yaml` | 4.5 | **14.13-14.42%** |
| **for the render-cleared 10% target** | **≈2.99** | **10.0%** |
| research basis (certifies DSR/PBO/P(pass)) | *n/a* | **6.92% regardless** |

**Two consequences:**

1. **The own-capital config already runs at 9.83%** — inside the ≤10% band the forward-path render
   cleared. Reaching the render's target needs `max_gross_exposure ≈ 3.0`, i.e. the own-capital
   value, **not** a `target_vol_asset` change.
2. **The config's stated mechanism is right by accident.** It claims scaling all three levers 1.5×
   delivers 1.5× vol. It roughly does (9.83% → 14.13%, ratio 1.44) — but *only* because
   `max_gross_exposure` is one of the three. The other two contribute nothing.

The research basis still sits at 6.92% whatever the levers say — it discards scale by design, which
is what makes its Sharpe/DSR leverage-invariant. **That gap is not closable; it is now merely
KNOWN**, and the mapping above lets the executor be set to any intended vol deliberately.

## Method note — the synthetic bundle gave the WRONG mechanism

Part 1 measured `target_vol_asset` as a clean linear lever (0.10→8.49%, 0.15→12.75%, ratio exactly
1.50) because the synthetic bundle draws every asset from one distribution at ~15.9% ann vol, so
nothing saturates. Real ETFs span ~5-25% vol, the cap binds on most of them, and the mechanism
inverts. **A synthetic fixture validated the plumbing and falsified the physics.** Use it for
wiring; never read a calibration off it.

Sequence for the record: analytic guess said `max_gross` binds → synthetic said `target_vol_asset`
→ real data said `max_gross` after all. The first guess was right and the synthetic measurement
talked me out of it. Two of three answers here were wrong, and only the real bundle settled it.

## Status

**Capital-gate item (2): CLOSED.** The mapping is measured on the production bundle. Items (3)
Tier-2 audit, (4) six protocol-v2 wiring gaps, (5) operator go-ahead remain open. **A challenge
attempt remains BLOCKED.**
