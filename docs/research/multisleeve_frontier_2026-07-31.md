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
