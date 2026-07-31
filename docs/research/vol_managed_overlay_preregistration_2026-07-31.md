# PRE-REGISTRATION — Volatility-managed OVERLAY on the existing TSMOM book (V1)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the overlay was built or any Sharpe was seen.**

## 0. Why this is a different class of test

Twelve probes today all asked the same shape of question: *is there a new SIGNAL to add?* Four became
sleeve candidates and all four failed the admission rule `s2 > s1·(√(2+2ρ) − 1)`, which is a statement
about **adding** things.

V1 is not a sleeve. It is an **overlay on the book that already exists** — it adds no instrument, no
new data, and no new return stream, so the admission rule does not apply and there is no correlation
bar to clear. It re-times the exposure of the validated cross-asset TSMOM sleeve.

**It is also not already in the book.** `configs/tailwind_v1.yaml` sets
`risk_parity.target_portfolio_vol: null` **deliberately** ("null = convex inverse-vol; leverage is
applied per-asset"). The book vol-scales **per asset**; it has never applied **portfolio-level**
volatility timing. So this is a genuine gap, not a re-run.

Moreira-Muir (2017) is the reference result: scaling a strategy's exposure by the inverse of its own
recent realised variance raises Sharpe across many documented factors. If it holds here it improves
the strategy that is about to be deployed — which is a more useful outcome than a marginal new sleeve.

## 1. What is measured (LOCKED)

Base series: the existing cross-asset TSMOM net sleeve (`portfolio_frontier.build_momentum_net`),
unchanged.

Overlay: `w[t] = clip(target_var / realised_var[t−1], 0, cap)` applied to the base return, where
`realised_var[t−1]` is the trailing variance over a **pre-registered** window and `target_var` is the
full-sample variance (a constant, so it sets scale only — Sharpe is leverage-invariant and cannot be
manufactured by the constant).

**Pre-registered parameters, fixed now:**
- vol windows: **21 and 63 days** — both reported, neither chosen after the fact. Two windows are the
  entire grid; there is no sweep.
- leverage cap: **3.0** (mirrors `env.lev_cap` in the challenge config).
- `realised_var[t−1]` is **lagged one day** — the overlay may only use variance known before the
  return it scales (LEAK-2).

**Costs:** the overlay changes exposure daily, which is real turnover. Cost is charged on
`|w[t] − w[t−1]|` at **2bp and 5bp**. An overlay that only works gross is a FAIL.

## 2. Pre-committed pass conditions

1. **Net Sharpe improvement ≥ +0.10** over the unmanaged base at 5bp, on the full sample.
   A smaller gain is not a pass — it is inside the noise this session has repeatedly demonstrated
   (the multi-sleeve run produced a +0.016 that flipped sign with the window).
2. **The improvement must hold in ≥3 of 4 subperiods.** A vol-managed result driven entirely by one
   crisis (2008 or 2020) is a timing artifact, not an overlay that works.
3. **Both vol windows (21 and 63) must show a positive gain.** If only one does, the result is
   window-specific and is recorded as such rather than as a finding.

All three must hold.

## 3. Honest priors, before the run

**For:** the effect is well documented across factors; the mechanism (volatility is persistent and
forecastable while returns are not) is sound and applies to trend books; it needs no new data.

**Against, and these are serious:** Moreira-Muir has been **actively contested** — Cederburg,
O'Doherty, Wang and Yan (2020) find vol-managed strategies frequently fail out-of-sample and that the
in-sample gains are fragile. The base book is *already per-asset vol-scaled*, so much of the
volatility variation this overlay targets may already be removed, leaving little to harvest. And
daily exposure changes on a monthly-rebalanced book add turnover that did not exist before. **The
most likely outcome is a small gross gain that dies on costs**, which is why costs are charged and
the bar is +0.10 net, not any positive number.

## 4. Stop rule

One overlay, two pre-registered windows, one cap. **No V2.** If it fails, portfolio-level vol
management on this book is recorded closed and is not re-probed with other windows, other caps,
a vol-of-vol term, or a downside-only variance estimate.

## 5. Promotion ceiling

A pass earns a **recommendation to test in the TAILWIND re-sizing work**, not an automatic change.
It would still require the forward-path render and Tier-2 gates that bind every capital decision.

## 6. Results — **NO-GO**: the gain is one subperiod

Run 2026-07-31, `results/vol_managed_overlay/vol_managed_overlay.json`. Base cross-asset TSMOM:
5,133 days (2006-01→2026-05), Sharpe **0.601** (matches the known headline — sanity check passed).

| variant | base | managed | gain | exposure turnover/yr |
|---|---|---|---|---|
| w=21 @2bp | 0.602 | 0.679 | +0.077 | 19.6 |
| **w=21 @5bp** | 0.602 | 0.673 | **+0.071** | 19.6 |
| w=63 @2bp | 0.605 | 0.629 | +0.025 | 7.6 |
| **w=63 @5bp** | 0.605 | 0.627 | **+0.022** | 7.6 |

Subperiod gains (managed − base):

| window | 2006-09 | 2010-15 | 2016-20 | 2021-26 | positive |
|---|---|---|---|---|---|
| w=21 | **−0.223** | **−0.053** | **+0.599** | **−0.150** | 1/4 |
| w=63 | **−0.110** | **−0.192** | **+0.526** | **−0.159** | 1/4 |

### Verdict against the pre-committed bars

| bar (§2) | w=21 | w=63 | |
|---|---|---|---|
| net gain ≥ +0.10 @5bp | +0.071 | +0.022 | ✗ **FAIL** |
| gain positive in ≥3 of 4 subperiods | 1/4 | 1/4 | ✗ **FAIL** |
| both windows positive | +0.071 | +0.022 | ✓ pass |

**NO-GO.** The headline gain is small, and condition 2 shows *why* it exists at all: **the entire
effect is one subperiod (2016-20, which contains the 2020 crash). The overlay LOSES in the other
three, in both windows.** On the headline alone this would have read as "small but promising"; the
subperiod split shows a single-episode timing artifact, not an overlay that works.

This is a direct empirical confirmation of the prior recorded in §3 — Cederburg, O'Doherty, Wang and
Yan's finding that vol-managed gains are fragile. The §3 prediction ("most likely a small gross gain
that dies on costs") was close but not quite right: the gain survives costs at w=21 and dies on
**regime instability** instead.

### Ledger (durable)

- **Portfolio-level volatility management on the cross-asset TSMOM book: NO-GO.** Gain +0.071 (w21)
  / +0.022 (w63) at 5bp, and positive in only 1 of 4 subperiods for both windows. Per §4 there is no
  V2 — no other windows, caps, vol-of-vol terms, or downside-only variance.
- **Actionable for the TAILWIND re-sizing work: do NOT add a portfolio vol-timing overlay.**
  `target_portfolio_vol: null` should stay null. This closes a question the re-sizing work would
  otherwise have had to ask.
- **Method note that generalises beyond this probe:** a headline improvement of +0.07 on a 20-year
  sample looked plausible and was entirely one regime. **Any overlay or timing result must be shown
  per-subperiod before it is believed** — the aggregate number cannot distinguish "works" from
  "worked once".
