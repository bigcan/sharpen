# Multi-sleeve frontier — do the measured sleeves help TOGETHER? (NO)

**Date:** 2026-07-31 (S553-cont-146) · **Artifact:** `results/multisleeve_frontier/multisleeve_frontier.json`
**Script:** `scripts/research/multisleeve_frontier_2026.py`

## The question the pairwise tests could not answer

T1/T2/T3 each tested a candidate sleeve **pairwise** against the existing cross-asset TSMOM book, and
the admission rule derived from T3 — `s2 > s1·(√(2+2ρ) − 1)` — is a **two-asset** result. It says
nothing about whether the candidates are correlated **to each other**. If T1 and T3 were mutually
independent, the three-sleeve optimum would not be determined by the pairwise tests and the sleeve
arc's conclusion would have been incomplete.

**No new signal was computed.** This is a portfolio question on fixed, already-pre-registered,
already-run return series, so it incurs no multiplicity.

## The hypothesis was partly right

Correlation matrix on the common overlap (1,083 days, 2022-01→2026-04):

| | base | T1 country | T3 commodity | T2 crypto |
|---|---|---|---|---|
| **base** | 1.000 | 0.768 | 0.413 | −0.063 |
| **T1** | 0.768 | 1.000 | **0.136** | −0.089 |
| **T3** | 0.413 | **0.136** | 1.000 | −0.019 |
| **T2** | −0.063 | −0.089 | −0.019 | 1.000 |

**T1 and T3 are nearly independent of each other (ρ 0.136)** — as hypothesised. The candidates are not
redundant with one another, only with the base.

## And it produces no deployable gain

| subset | equal-risk (deployable) | max-Sharpe (in-sample ceiling) |
|---|---|---|
| base alone | **0.833** | 0.833 |
| base + T3 | 0.849 (+0.016) | 0.877 |
| base + T1 | 0.703 | 0.865 |
| base + T1 + T3 | **0.808** | 0.891 |
| base + T1 + T3 + T2 | 0.664 | **0.916** |

At **deployable (equal-risk) weights — the combine TAILWIND actually uses — adding both sleeves makes
the book WORSE** (0.808 < 0.833). Only base+T3 edges up, by +0.016.

**The 0.916 max-Sharpe figure is not an edge and must not be cited as one.** It is an unconstrained,
in-sample optimum fitted on four assets over 1,083 days, requiring exact weights and a short position
in a negative-Sharpe sleeve. It is reported solely to *bound* what any allocation could achieve — and
the bound being +0.083 in-sample means the honest out-of-sample expectation is near zero.

## The decisive tell: the effect flips sign with the window

- base+T3 over **4,995 days** (the T3 pairwise run): **0.568 vs 0.609** — the sleeve **hurts**.
- base+T3 over **1,083 days** (this overlap): **0.849 vs 0.833** — the sleeve **helps**.

Same two series, opposite conclusions, purely from the window. The overlap is also unrepresentative:
the base sleeve earns **0.833** on it versus **0.601** full-sample, so this window flatters trend
generally. A +0.016 gain that changes sign with the sample is noise.

## Verdict

**NO-GO. The sleeve arc is closed.** None of T1/T2/T3 is admissible alone, and they are not
admissible together at any deployable weighting. The pairwise admission rule stands, and this run
shows it was not missing a multi-sleeve escape.

### Durable ledger

- **Multi-sleeve combination of {country TSMOM, commodity TSMOM, crypto TSMOM} on the cross-asset
  book: NO-GO.** Best deployable gain +0.016, sign-unstable across windows.
- **Mutual independence is necessary but nowhere near sufficient.** T1 and T3 are nearly uncorrelated
  to *each other* (0.136) and still cannot help, because both are too correlated to the base and
  neither carries enough standalone Sharpe. Low candidate-to-candidate correlation is not a
  substitute for clearing the pairwise bar.
- **Method rule:** when a combine is evaluated on a window shorter than the sleeves' own histories,
  report the base sleeve's Sharpe *on that same window* before concluding anything. Here it was
  0.833 vs 0.601 full-sample, which alone explains the apparent improvement.

---

## ADDENDUM — re-run on the LONG window (crypto dropped). Conclusion strengthened.

The run above used the common overlap of **all four** sleeves, which crypto's short history
constrained to **1,083 days**. That was flagged as a limitation at the time and is now fixed: crypto
is dropped (its standalone Sharpe is negative anyway) and base/T1/T3 are combined on their own
**4,995-day** overlap (2006-07-21 → 2026-05-29).

Standalone on this window: base **0.609** · T1 country **0.289** · T3 commodity **0.370**.

Correlations: T1↔base **0.709** · T3↔base **0.485** · **T1↔T3 0.230**.

| subset | equal-risk (deployable) | max-Sharpe (in-sample ceiling) |
|---|---|---|
| **base alone** | **0.609** | 0.609 |
| base + T1 | 0.486 | 0.642 |
| base + T3 | 0.568 | 0.615 |
| base + T1 + T3 | 0.525 | 0.644 |

**Every equal-risk combine is WORSE than the base alone** (0.486 / 0.568 / 0.525 vs 0.609). The
max-Sharpe ceilings gain at most **+0.035** in-sample and unconstrained — a number that would not
survive out-of-sample with fitted weights, on the same reasoning applied to the 0.916 figure above.

**This confirms the window artifact the original run identified.** base+T3 measured 0.849 vs 0.833 on
the 1,083-day window and **0.568 vs 0.609** on the full 4,995 days. Same two series, opposite
conclusions, purely from the sample — exactly the failure the "report the base's Sharpe on the same
window" rule was written to catch. The long window is the honest one.

T1 and T3 remain near-independent of each other (ρ 0.230) and it still does not rescue them, which
re-confirms the durable lesson: **mutual independence is necessary but nowhere near sufficient** when
both candidates are too correlated to the base and neither clears the pairwise admission bar.

**The sleeve arc is closed on the honest window, not just the short one.**
