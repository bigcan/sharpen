# ATL × Jev: can an AI that reads SEC filings build a tradeable strategy?

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investing), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan).
> Video: link added when Ep05 is published.

We had an AI model read about 25,000 SEC 8-K filings from S&P 500 companies (2012 to 2024) and turned its answers into five trading signals. Every rule and pass bar was written into git before any filing was scored. Then the signals went through [Sharpen](https://github.com/bigcan/sharpen), our falsification-first evaluation framework.

**Verdict:** one of the five signals, `jev-surprise-63`, passed all three screening tests. It is still not a tradeable strategy. Its long-short portfolio earns a Sharpe ratio of **0.02 before costs and −0.28 after standard costs**, and it breaks even only if each trade costs under about 0.7 basis points. The best signal in the study lost money once trading costs were charged. Historical simulation, 2012 to 2024, on a price panel that leaves out delisted companies, so every number is an upper bound.

## What was run

- **The model:** Jev (TypeSafe, "System One"), asked six fixed questions about each filing: was guidance raised, was guidance lowered, what is the business trajectory, is the tone more positive than the numbers justify, are the results flattered by one-offs, is there a material adverse event. It answers with probabilities and cannot write text. It reads facts that are in the filing; it is never asked to forecast prices.
- **The data:** 98,119 8-Ks listed for index members, 584 companies. 24,676 filings were scored (24,654 with text) for 140,614 answers, all from one model version (`jev-1.13.0`), at a cost of $6.57.
- **The tests (all fixed before any score existed):** P1, Sharpen's frozen evaluation funnel. P2, a placebo that shuffles scores across firms reporting in the same week, 200 times. P3, a check that Jev adds information beyond two non-AI text baselines (Loughran–McDonald tone and Lazy-Prices similarity). P4, one look at clean data after 2025-02. P5, a forward lockbox.
- **Cost:** three calendar days and about $7 of Jev credits. The price and filing data were free (SEC EDGAR, Yahoo Finance).

## Results

| Signal | Verdict | 5-day IC | IC t | Deflated Sharpe | FDR q | Sharpe before costs |
|---|---|---|---|---|---|---|
| **jev-surprise-63** | **PROMISING** | 0.0090 | 3.21 | 0.946 | 0.025 | 0.02 |
| jev-toneinfl-63 | logged | 0.0058 | 2.28 | 0.764 | 0.10 | −0.04 |
| jev-comp-63 | logged | 0.0072 | 2.16 | 0.723 | 0.10 | −0.13 |
| jev-comp-21 | logged | 0.0032 | 0.71 | 0.251 | 0.45 | −0.12 |
| jev-quality-63 | logged | 0.0015 | 0.66 | 0.178 | 0.45 | −0.15 |

- **P2 passed:** none of 200 placebo shuffles reached the real IC (p = 0.005).
- **P3 passed:** Jev's coefficient had a Newey–West t of 3.18 after controlling for both baselines.
- **Costs:** Sharpe 0.02 with no costs, −0.28 at standard costs, −0.73 at harsh costs, at about 8 turnovers a year. Paying for itself at standard costs needs a Sharpe of about 0.30 before costs.

## Why the passing signal is weaker than it looks

- **The screening years fall inside Jev's memory.** In a probe of 10,207 dated questions, Jev's monthly lean matched what markets actually did through 2023 (2020: +0.69) and stopped matching from 2024 (2026: −0.72). Masking names, tickers and dates does not hide a filing: Jev still picked the filer from 10 same-sector candidates 93.5% of the time (chance: 10%). So anything scored before 2025 can screen a signal but never certify it. The clean window starts 2025-02-01.
- **The t-statistic depends on the estimator.** It clears the 3.0 bar under the funnel's own overlap correction; stricter estimators give 2.6 to 2.9.
- **The effect fades.** It sits in 2012 to 2017 and is statistically zero in 2018 to 2024 (IC 0.0045, t 1.1). About 40% is shared with ordinary price signals.
- **Rank skill is not money.** The same portfolio weights earn a Sharpe of 0.76 against return ranks but 0.02 against actual returns.
- **The clean test was not run.** The one pre-registered look at post-2025 data has about a 20 to 25% chance of passing even if the effect is real, so it was left unopened. A pass would show the effect is alive, not that it pays.

## What worked: the machinery

The signal did not survive. The process did what it is built to do:

- Every rule was committed before the data it judged, and the multiplicity count (8 trials, not 5) was charged up front.
- A deep audit by 15 AI agents, a finder and a skeptic for each area, ran before the one clean look could be spent. It confirmed 111 findings and refuted 3, and its verdict was "not ready". The weak evidence was exposed before any money was at risk.
- Four independent reconstructions reproduced the P1, P2 and P3 results bit for bit, and the checks in this folder reproduce the quoted numbers from the saved files.
- The whole study, from first probe to final report, took three days and about $7.

This is the point of [Sharpen](https://github.com/bigcan/sharpen): it is built to kill weak ideas cheaply and early, in a way you can inspect. [Crucible](https://github.com/bigcan/sharpen), Sharpen's agentic alpha miner, is the part that goes looking for ideas worth testing (see the KISI video "I Built an AI to Mine for Alpha").

## Check it yourself (CPU, no keys, no downloads, a few seconds)

```bash
git clone https://github.com/bigcan/sharpen && cd sharpen
python studies/atl-jev/verify.py
```

It reads the saved screening results in `results/atl_jev/` and asserts the screening, placebo, baseline, run and Phase 0 numbers quoted above, ending with `0 mismatches`. A few figures are quoted from the final report only and are not asserted by the script: the 0.7 bp break-even, the 2.6 to 2.9 range of stricter t-statistics, the 2018 to 2024 IC and t, the 0.76 rank Sharpe, the 40% overlap with price signals, the 20 to 25% chance for the clean look, the deep-audit counts, and the three-day duration. Expected output: [`expected/verify-output.txt`](expected/verify-output.txt). Figures and where each comes from, with the report-only ones marked: [figures.md](figures.md).

Not rerunnable from a fresh clone: re-scoring the filings (needs a Jev API key and about $7) and rebuilding the price panel.

## Read the full report

- [`docs/research/atl_jev_final_report_2026-09-25.md`](../../docs/research/atl_jev_final_report_2026-09-25.md): the final report, including the researcher's own assessment
- [`docs/research/atl_jev_prereg.md`](../../docs/research/atl_jev_prereg.md): the pre-registration, written before any filing was scored
- [`docs/research/atl_jev_strategy_plan_2026-09-23.md`](../../docs/research/atl_jev_strategy_plan_2026-09-23.md) · [`atl_jev_phase1_research.md`](../../docs/research/atl_jev_phase1_research.md) · [`atl_jev_deep_lifecycle_audit_2026-09-24.md`](../../docs/research/atl_jev_deep_lifecycle_audit_2026-09-24.md)
- Code: [`scripts/research/atl_jev_*.py`](../../scripts/research/) and [`sharpen/jev/`](../../sharpen/jev/)

This study judges one signal from one model on one arena (S&P 500 earnings releases). It says nothing about other models, other markets, or other ways of using language models. Nothing in it evaluates the Agentic Trading Lab or Jev as software.
