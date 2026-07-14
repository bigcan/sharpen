# RESULT — Crucible Cross-Market Pooling, Stage-0 Part A (power)

**Date:** 2026-07-13 · **Session:** S553-cont-130 · **Verdict: A1 = FAIL → Stage-0 NO-GO**
**Pre-registration:** `crucible_crossmarket_pooling_stage0_preregistration_2026-07-13.md`

---

## Bottom line

Cross-market pooling — the one power lever the E1/E2 calibration never considered — **cannot** drag the
deflated funnel's per-market MDE (1.403) below the 0.50 realistic-alpha ceiling with the number of
independent gold-positioning markets that actually exist. Per the pre-registration's execution order
(**A-FAIL → stop before the US COT / price pull**), Part B was **not run**. Pooling is a **NO-GO** — a
Math-verified, anchor-validated falsification, not an unresolved gap. **All three power levers are now
closed.**

## What was computed (analytic + Monte-Carlo, tripwires green first)

Pooling `k` markets of a common mechanism, each with per-market MDE `m` and equicorrelated pairwise PnL
correlation `ρ`: `MDE_pool = m·√((1+(k−1)ρ)/k)`, so the markets needed to reach ceiling `c=0.50` is
`k_needed(ρ) = (1−ρ)/(f−ρ)` with `f=(c/m)²=0.127`, and pooling **can never** reach `c` once `ρ ≥ 0.127`
(the correlation floor `m·√ρ`).

**Tripwires (all PASS before the verdict was read):** TC-1 Monte-Carlo `Var(mean)` matches the closed form
to 0.56% across the `(k,ρ)` grid; TC-2 the harness reproduces the funnel's own anchor **MDE 1.4025 at
holdout 1011** (= the E1/E2 1.403); TC-3 degenerate limits `k=1` and `ρ=1` both leave MDE = m exactly.

### Markets needed vs markets available

| ρ | k_needed (m=1.403, the gate) | k_needed (m=1.20) | k_needed (m=1.10, optimistic single-test) |
|---|---|---|---|
| 0.00 (independent, optimistic) | **7.9** | 5.8 | 4.8 |
| 0.05 | 12.3 | 7.7 | 6.1 |
| 0.10 | 33.3 | 12.2 | 8.4 |
| 0.127 (floor) | ∞ | 18.7 | 11.0 |

**k_available for the same gold-positioning mechanism = 2** (US CFTC COT `088691` + Taiwan T86 `00635U`
dealer net); a generous upper bound admitting related metals/venues = 4.

### Pooled MDE at the counts that exist (measured m=1.403)

| markets | ρ=0.0 | ρ=0.05 | ρ=0.10 | ρ=0.127 |
|---|---|---|---|---|
| **k=2** (US + TW) | 0.99 | 1.02 | 1.04 | 1.05 |
| k=4 (generous) | 0.70 | 0.75 | 0.80 | 0.82 |

Even the generous, optimistically-independent `k=4` lands at **0.70 > 0.50**. The realistic `k=2` case sits
at **~1.0**, barely better than a single market.

## The finding: it's the same √N wall, and N (markets) is even scarcer than years

- To close 1.40→0.50 you need the pooled variance cut to `f=0.127`, i.e. **~8 independent markets** at the
  optimistic `ρ=0`. Gold institutional-positioning data exists for **~2** (US, Taiwan). The lever is short
  by 4× on the one axis it was supposed to buy.
- Gold is a **global asset**, so `ρ=0` is unrealistically optimistic — the shared gold-return component
  correlates the markets' PnLs. Any `ρ ≥ 0.127` makes 0.50 **unreachable at any k**.
- The conclusion is robust to the miner-vs-single-test distinction: even the optimistic single-pre-registered
  test (`m=1.10`, no 37-way multiplicity) still needs **k≥5** at `ρ=0` — more than the 2 that exist.
- This **closes the third and last power lever** the calibration named. With extend-history (dead, ~130 yr)
  and higher-frequency (closed, intraday MDE 0.86) already gone, only a **deliberately less-deflated gate**
  remains — documented but not recommended (it trades away E1's proven zero-false-positive strength).

## Honesty caveats (Tier-1 self-audit, PASS WITH NOTES)

1. **The pooling law is idealized** (equicorrelation, equal per-market SE). Inverse-variance weighting on
   unequal panels helps marginally but cannot change the `√k` order — the verdict's 4× shortfall absorbs any
   such slack. TC-1's MC confirms the closed form; TC-3 confirms the boundaries.
2. **`m=1.403` is the miner's MDE** (embeds multiplicity + full deflation). The `m∈{1.10,1.20}` columns show
   even a multiplicity-free single test doesn't rescue it at `k=2`. So the NO-GO holds whether you pool the
   miner's outputs or pool independent pre-registered single tests.
3. **The candidate was an operator overfit anyway.** The cont-129 "winner" `decay_linear³(stddev(stddev(
   delta(dealer_net,60),3),30),30)` had an **empty rationale**; its true replicable effect is almost
   certainly ≪ 0.66. So even if pooling had power, there is likely nothing to detect — a double NO-GO.
4. **`ρ=0` is a gift to the lever, not the base case.** Reporting it and still failing makes the verdict
   airtight; the realistic `ρ>0` for a global asset only makes it worse.

## Reproduce

```
python scripts/research/crucible_crossmarket_power.py
```
Config: `configs/crucible_calibration_crossmarket.gates.yaml` (validates, changes no verdict — not part of
the frozen funnel hash). Result (gitignored): `results/crucible_crossmarket_power/crossmarket_power.json`.
Funnel `gates_hash 519158fa1450` untouched.

## Consequence for the roadmap

Crucible-Taiwan discovery is **power-exhausted across all three levers**: extend history (dead), higher
frequency (closed, cont-129), and **cross-market pooling (this — closed for want of independent markets)**.
The only remaining lever is a deliberately less-deflated gate — **not recommended**. **Recommended
direction: return to the deployable TSMOM → paper path** (net SR ~0.60, sitting at the paper gate
DSR 0.896 < 0.95). Crucible cycles are better spent moving that to capital than mining a power-exhausted
substrate.

**Related:** `project_crucible_calibration_e1e2_s553` (the anchor + the two prior levers),
`project_crucible_llm_proposer_via_cli_s553` (the candidate), `crucible_intraday_stage0_partA_result_2026-07-13.md`
(the sibling lever), `project_commodity_carry_sleeve_nogo_s553` (broaden-to-many-commodities is a different,
already-NO-GO mechanism).
