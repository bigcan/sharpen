# ATL × Jev — Final Report

## Can an AI agent reading SEC filings produce a tradeable strategy?

**Date:** 2026-09-25 · **Branch:** `agentic-trading-lab` · **Workstream:** `atl-jev`

**Status:** Screening complete. The one-look clean-window test (P4) has not been run, and the clean window is
unopened. Recommendation: close the study without spending the look. That decision is the operator's.

**Sources:**
- plan: `docs/research/atl_jev_strategy_plan_2026-09-23.md`;
- pre-registration: `docs/research/atl_jev_prereg.md`;
- literature review: `docs/research/atl_jev_phase1_research.md`;
- Tier-2 audit: `docs/research/atl_jev_deep_lifecycle_audit_2026-09-24.md`;
- artifacts: `results/atl_jev/`.

## Summary

- **The question.** Can an AI agent, TypeSafe's Jev model, read every S&P 500 company's SEC 8-K filings and produce a
  trading strategy that passes Sharpen, this project's falsification-first evaluation framework?
- **The answer: no tradeable strategy.**
  - One of five pre-registered signals, `jev-surprise-63`, passed all three screening tests on 2012–2024 data.
  - It is a small ranking effect (5-day IC 0.009) that does not survive trading costs.
  - Its long-short portfolio earns a Sharpe ratio of 0.02 before costs and −0.28 after.
  - It breaks even only if each trade costs under about 0.7 basis points.
- **Why the result is weaker than it looks:**
  - the screening years fall inside Jev's memory;
  - the price data leaves out delisted companies;
  - the t-statistic clears the 3.0 bar only under the funnel's own overlap correction, and stricter estimators give
    2.6–2.9;
  - the effect sits in 2012–2017 and is statistically zero in 2018–2024;
  - about 40% of it is shared with ordinary price signals.
- **Why it cannot be proven.** Only data from after the model's memory can certify a result, and that means from
  2025-02 on. That is 411 trading days, which gives the single pre-registered look a 20–25% chance of passing even if
  the effect is real.
- **What worked: the machinery.**
  - Every screening number reproduces bit-for-bit from committed code.
  - Every rule was committed before any score existed.
  - A deep audit exposed the weak evidence before the one clean look was spent and before any money was at risk.
- **Would a more capable model fix it?** Very unlikely. Claude Opus 5.5 would read the filings better, but its
  knowledge runs to June 2026, so no clean history would be left to test it on.
- **Cost:** three calendar days (2026-09-23 → 09-25) and about $7 of Jev credits.

## 1. The setup

**Goal.** Build a strategy on the Agentic Trading Lab (ATL, `Open-Finance-Lab/AgenticTrading`) with Jev as the
agent, and have it pass Sharpen (github.com/bigcan/sharpen).

**Jev** is TypeSafe's "System One" model, released 2026-09-15.
- It answers typed questions only (yes/no, a score, or a choice) with probabilities. It cannot write text.
- It is cheap: $0.042 per million input tokens, with output free (list price during the study).
- Its training cutoff is undisclosed.

**Design choice: Jev reads, it does not forecast.** Asking a language model "will this stock go up?" invites it to
answer from memory of what actually happened. Instead:
- Jev answered fixed *extraction* questions about each filing, about facts that are in the text.
- A pre-registered linear rule turned the answers into a daily cross-sectional score.
- Technical indicators were excluded, because price signals on liquid large caps are a closed family in this
  project's record.

```
SEC EDGAR 8-K ─► mask names, tickers, dates ─► Jev answers 6 fixed questions ─► fixed scoring rule
                                                                                     │
   Sharpen: P1 funnel ─► P2 placebo ─► P3 baselines ─► [P4 one clean look] ─► [P5 lockbox]
```

**ATL's role.** ATL was installed and smoke-tested as the execution and paper-trading harness. The strategy never
reached it, because nothing cleared the bar to be traded. Nothing in this report evaluates ATL itself.

**What "pass Sharpen" means.** There are five legs, all fixed in git before any filing was scored:

| Leg | Test | Bar |
|---|---|---|
| P1 | Sharpen's frozen evaluation funnel, 2012–2024 | `PROMISING`: IC t ≥ 3, deflated Sharpe ≥ 0.90, FDR q ≤ 0.10, no negative subperiod |
| P2 | Placebo: shuffle scores across firms that reported in the same week, 200 times | The real IC beats the placebos at p ≤ 0.05 |
| P3 | Beat two non-LLM text baselines: Loughran–McDonald tone and Lazy-Prices similarity | Jev's coefficient t ≥ 2 after controlling for both |
| P4 | One look at clean data, 2025-02-01 → 2026-09-22 | 5-day IC t ≥ 2 and a Sharpe before costs above 0 |
| P5 | Forward lockbox, at least 63 new trading days | Forward marginal Sharpe ≥ 0.30 |

Even a P5 pass earns only a human deep audit before any capital. The funnel's best possible verdict is
`PROMISING`, never GO.

## 2. Phase 0 — Can a language model be tested honestly?

Three questions came before building anything:
- does Jev remember the market;
- can it recognize masked filings;
- is the test strong enough to detect a realistic effect?

**It remembers market regimes.**
- *Method:* 10,207 dated questions ("did X close higher or lower in month m?") across 101 securities, 2018-01 →
  2026-08, with the wording randomized.
- *The registered test failed to see it.* The accuracy test found no cutoff, but that test is fooled by a model that
  simply believes "stocks usually rise".
- *The statistic that works:* the monthly correlation between Jev's average implied P(up) and the share of
  securities that actually rose. A fixed prior cannot produce it.

| Year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| Correlation | +0.18 | +0.21 | **+0.69** | +0.28 | **+0.54** | **+0.58** | −0.11 | −0.40 | −0.72 |

In March 2020, Jev's average implied P(up) was 0.19; 15% of names actually rose. The clean window was set to start
on 2025-02-01.

**Masking does not anonymize a filing.**
- *Method:* 200 earnings releases with names, tickers, dates, URLs and phone numbers stripped.
- *Result:* Jev still picked the filer from 10 same-sector candidates 93.5% of the time (chance: 10%), and the year
  49.5% of the time (chance: 14%).
- *Why:* brands, products, business segments and executive names give the filer away.
- *Consequence:* anything scored before 2025 can only screen, never certify.

**Power.** Planted-signal tests on the frozen funnel:
- On the S&P 500, the smallest effect detected 80% of the time is a 5-day IC of about 0.02. There were 0 false
  positives in 672 null signals.
- The Dow 30 can never pass, because the funnel requires at least 50 names a day. Even with that rule waived, it
  would need an IC of about 0.05.

## 3. Phase 1 — Pre-registration

The literature review expected a large-cap 5-day IC of about 0.01–0.02, at or below what the test can detect. It
concluded: "The expected outcome is still a clean NO-GO." The test ran anyway because it was cheap and decisive
either way.

Six extraction questions were frozen under questionnaire hash `3fc01888e31f`:

| ID | Question (gist) | Sign | Mechanism |
|---|---|---|---|
| E1 | Guidance raised | + | Text surprise beyond the headline number (PEAD.txt) |
| E2 | Guidance lowered | − | Same |
| E3 | Forward business trajectory | + | Same |
| E4 | Tone more positive than the numbers justify | − | Tone management reverses (Huang, Teoh & Zhang 2014) |
| E5 | Results flattered by one-off items | − | Earnings quality |
| M1 | Material adverse event | − | Negative drift after adverse 8-Ks (Lerman & Livnat 2010) |

- **Signals.** Five signals combine these blocks: surprise (E1–E3), tone (E4) and quality (E5 + M1). Each is held 21
  or 63 trading days.
- **Deflation** is charged for 8 trials, not 5.
- **Timing.** A filing enters on the trading day it was accepted if that was before 15:30 New York time, and
  otherwise on the next trading day. The test therefore measures only drift after the market's first reaction.

## 4. Phases 2–3 — Build and screening

**Corpus.**
- 98,119 8-Ks were listed for index members. 24,676 were in scope: 23,214 earnings releases and 1,462 event filings,
  from 584 companies, 2012–2024.
- The SEC blocked the first download attempt at 4 requests per second. The pull then ran at 2 per second for 304
  minutes, with no failures.
- **A detail that matters:** 11% of earnings releases are not labelled `EX-99.1` (they use `EX-99` or `EX-99.01`).
  A literal match would have fed Jev a one-paragraph stub instead of the release.
- 61% of releases exceed the 24,000-character limit, so Jev read their first 24,000 characters.

**Scoring.**
- 24,654 filings produced 140,614 answers, all from one model version (`jev-1.13.0`), for $6.57.
- Spot checks matched known events: Apple's January 2019 guidance cut, and Kraft Heinz's impairments and restatement.

**Engineering.** Each component shipped with tests, plus deliberately planted bugs to prove the tests fire. For
example, the corpus builder's tests caught 19 of 19 planted bugs.

**P1 results.** All are upper bounds: the era is contaminated and the panel is survivorship-biased.

| Signal | Verdict | 5d IC | IC t | DSR | FDR q | Sharpe before costs |
|---|---|---|---|---|---|---|
| **jev-surprise-63** | **PROMISING** | 0.0090 | 3.21 | 0.946 | 0.025 | 0.02 |
| jev-toneinfl-63 | logged | 0.0058 | 2.28 | 0.764 | 0.10 | −0.04 |
| jev-comp-63 (primary) | logged | 0.0072 | 2.16 | 0.723 | 0.10 | −0.13 |
| jev-comp-21 | logged | 0.0032 | 0.71 | 0.251 | 0.45 | −0.12 |
| jev-quality-63 | logged | 0.0015 | 0.66 | 0.178 | 0.45 | −0.15 |

`jev-surprise-63` was positive at every horizon from 1 to 63 days and in all four subperiods. The funnel still
flagged it as cost-blocked, and as fragile across resampled train/test paths (47% of paths positive).

- **P2 passed:** none of 200 placebo shuffles reached the real IC (p = 0.005).
- **P3 passed:** controlling for both baselines, Jev's coefficient had a Newey–West t of 3.18. The bar was low,
  though: on their own the baselines carry no information in large caps (IC +0.0002 and +0.0010).
- **Decision K3: proceed to P4** with `jev-surprise-63`.

## 5. The deep audit — why P4 was not run

The clean window can be opened only once, so a Tier-2 deep lifecycle audit ran before the look. It used 15 AI agents,
a finder and a skeptic for each area, and re-derived everything from data to live trading. It confirmed 111
findings and refuted 3. **Verdict: not ready.**

**What held up.**
- Four independent reconstructions reproduced P1, P2 and P3 bit-for-bit.
- The clean-window firewall held.
- Every rule was in git before any full-corpus score existed, and the multiplicity count was honest.
- There is also evidence that the clean window really is clean for this model: `jev-1.13.0` names 96% of masked
  filers but dated masked 2025 releases correctly only once in 27 tries.

**The dominant risk: nothing pinned the look to what was screened.** Any of these could change and still pass every
existing check:
- the model, requested through the alias `jev-latest`, which could silently move to a newer version;
- the text-extraction code;
- the 2025–26 price source;
- the universe files;
- the evaluation code;
- the one-look lock file.

**The statistics that licensed the look were weaker than they read.**
- **The t-statistic.** The funnel corrects for overlapping 5-day returns with a kernel that under-weights the
  overlap. Stricter estimators all fall below the 3.0 bar:
  - Hansen–Hodrick: 2.66;
  - Newey–West with automatic bandwidth: 2.90;
  - non-overlapping subsamples: 2.60 on average.

  The verdict still follows the registered rules. The underlying funnel defect affects every multi-day verdict and
  is recorded for a future funnel version.
- **Power.** Replaying the exact P4 recipe on 137 screening windows passes 24–25% of the time. Windows starting in
  2018 or later pass 3 of 65. A fail would be nearly uninformative, and waiting does not help: 50% power needs about
  five years of clean data.
- **Dollars.** The same portfolio weights earn a Sharpe of 0.76 against return *ranks* but 0.02 against actual
  returns. Break-even cost is about 0.7 bp one-way at about 8 turnovers a year, before any cost of borrowing shares
  for the short side.

**The mechanism is weaker than registered.** This looks more like a persistent company trait than news in the
filing:
- last quarter's score keeps 90% of the IC;
- the IC is near zero at the release and shows up 21–62 days later;
- momentum, reversal and announcement-return controls cut the P3 coefficient by about 42%;
- 84% of the standalone IC comes from E3, the most judgment-like question.

**A timing slip of the AI researcher's own.** A test written in Phase 2 treated a filing accepted at 15:29 on 2024-07-03 as
same-day, but NYSE closed at 13:00 that day. It changed the screening IC by two millionths (0.009037 → 0.009039).
Still, a test with the wrong clock must not be carried into the only look.

## 6. Why this is not a tradeable strategy

1. **Too weak to pay trading costs.** The Sharpe is 0.02 before costs (standard error 0.28) and −0.28 after.
   Break-even is about 0.7 bp per trade, below realistic costs even for S&P 500 stocks.
2. **Probably overstated.** The screening era is contaminated, the price panel has survivorship bias, and the
   t-statistic depends on the estimator.
3. **Fading, and mostly not coming from Jev.** 2012–2017 carries the effect, and the 2018–2024 IC is 0.0045 (t 1.1).
   About 40% is shared with price signals, and it behaves like a sticky company trait, not news.
4. **Cannot be proven, and the test is not about money.** 411 clean days give a 20–25% chance of passing even if the
   effect is real, and about 5% at its post-2018 strength. P4 measures ranking skill, not profit.

One nuance cuts the other way: a P4 pass would not be worthless.
- A pass would multiply the odds that the effect is real by about 6–10. A fail would barely move them.
- In every screening sub-window where the IC t reached 2, the portfolio after costs was positive (median Sharpe
  0.60).
- Even so, a pass would say the effect is alive, not that it pays. Trading it would still need a new pre-registered
  study built around portfolio construction and costs.

**The root cause is the arena.** S&P 500 earnings releases are among the most-read documents in the market. The
literature predicted an effect this small, and the edge from LLM-read news shows up mainly in small caps.

## 7. Would a more capable model fix it?

The natural follow-up is to replace Jev with a frontier model such as Claude Opus 5.5, the model that ran this study.

**What would improve: reading quality.** Jev answers fixed typed questions. A frontier model can reason over the whole
release and compare it with earlier quarters. Published work finds that larger language models predict returns
better (Lopez-Lira & Tang, 2023–24), so the IC would plausibly rise.

**What would not improve:**
- **Contamination gets worse.** Claude Opus 5.5's knowledge runs to June 2026. It knows, for example, how the market
  moved through the April 2025 tariff selloff. The 2025–26 window that is clean for Jev is inside its training data,
  so only filings released from now on would be clean.
- **Proof would take years.** At Jev's effect size, a forward test needs about five years for even odds of detecting
  it. At double the effect it still needs 1.5–3 years, possibly longer than one model version stays in service.
- **The cost gap is about 15×.** Breaking even at standard costs needs a Sharpe before costs of about 0.30, and the
  portfolio makes 0.02.
- **It is not an edge anyone would own.** Every fund can run the same models on the same filings minutes after
  release. The literature finds LLM-news returns concentrated in small caps and shrinking as adoption rises.
- **Letting the model trade directly is worse still.** No honest backtest is possible, because it remembers what
  prices did.

**Cost is not the obstacle.**
- At Opus 5.5 list prices ($4 / $20 per million input / output tokens), re-scoring the 24,676 filings would cost
  roughly $1–2k, or about half that through the batch API. The result would be contaminated anyway.
- Scoring new filings as they arrive (about 2,100 a year) would cost roughly $50–150 a year.
- *Assumptions:* about 20k characters per filing, which is about 7k input tokens, plus 1–2k output tokens.

**Where a stronger model could matter:** less-covered markets and harder documents, such as small caps and
non-English filings. Fewer funds use LLMs there, and an effect might be large enough to detect in a year or two.

## 8. Honest assessment by the AI researcher

*This section is Claude's own assessment, written at the operator's request. Claude (Opus 5.5) is the agent that
designed, built, ran and audited this study.*

**Am I confident I can build a tradeable strategy? No.**
- I can fix the audit's blockers; that is engineering.
- But the reasons this signal is not tradeable are not bugs. They are properties of the signal and the market, and
  no fix of mine changes them.
- Nor am I confident I can find a genuinely new edge in liquid markets.

**The evidence.**
- This project has tested dozens of ideas, and one edge survived: cross-asset time-series momentum (TSMOM). That is
  a published premium with a net Sharpe of about 0.6, not a discovery, and its deployable version is still blocked by
  its own audit.
- The project's alpha-mining platform, Crucible, has never produced a signal that survived its lockbox.
- The bottleneck is not engineering. I work from the same public data and the same papers as everyone else, and
  that is exactly what the market has already priced. Working harder does not create an information advantage.
- The usual rescues have not worked here either:
  - none of 54 risk-overlay variants turned a losing strategy profitable;
  - the project's reinforcement-learning stack could not learn a planted signal five times stronger than Jev's
    (a re-test with a fix started 2026-09-24).

**Where I went wrong on this study.**
- **Too generous a verdict.** My Phase 1 review said the likely outcome was NO-GO, yet I still labelled it a
  "conditional go". I have done that repeatedly in this project, and it has been too generous.
- **A forecast off by about 50×.** I predicted that an IC of 0.01–0.02 would be worth a Sharpe of 1.0–1.8 before
  costs. The real IC of 0.009 was worth 0.02. The audit measured two causes:
  - my planning signals were about 1.8× steadier per unit of IC than the real one;
  - the real signal ranks stocks without picking the ones whose moves matter in dollars.
- **A pass rule that could not mean tradeable.** I pre-registered "pass" as a Sharpe above zero before costs, so even
  a pass could never have meant tradeable. I should have said so on day one.
- **Two slips.** The audits caught the half-day timing error in a test and a commit that landed on another session's
  branch.

**What I am confident about.**
- Building and testing quickly and honestly, and saying "no" before money is at risk.
- Turning TSMOM into a clean, low-cost strategy. My guess is a live Sharpe of 0.3–0.5, with long flat stretches. That
  is real but modest, and it has to beat the cheap trend-following funds that already sell this exposure.

**What would improve the odds:** a niche where being small helps, such as small caps or less-covered markets, or data
few others have. The odds would be better, but there is still no promise.

**Bottom line.**
- My honest guess is under 5% that this S&P 500 design becomes tradeable with a frontier model doing the reading.
- If an LLM bet is worth making, it is a pre-registered forward test in a less-covered market, with the model version
  pinned and a one-to-two-year wait accepted.

## 9. Lessons worth keeping

1. **Measure a language model's market memory by month-level breadth, not item accuracy.** A "stocks rise" prior
   passes accuracy tests. The correlation between its monthly lean and what actually happened exposes real memory.
2. **Masking does not anonymize large-company documents.** Jev identified 93.5% of filers after names, tickers and
   dates were removed.
3. **Before a model's cutoff, backtests can screen but never certify.** What the model remembers defines the clean
   window, not the calendar.
4. **Compute power before building.** The Dow 30 could never have passed this test, and knowing that on day one
   saved the effort.
5. **Rank IC is not money.** Report the dollar Sharpe, the net Sharpe and the break-even cost next to every IC.
6. **Overlapping returns inflate t-statistics** unless corrected properly. Report a Hansen–Hodrick t beside any
   multi-day result.
7. **Pin the model version, not an alias.** `jev-latest` could have changed under a one-shot test without any check
   firing.
8. **Check document labels.** 11% of earnings releases were not labelled `EX-99.1`.
9. **Pre-register and keep one look.** Every rule here was committed before the data it judged. That is why the
   numbers can be trusted even though the edge cannot.
10. **Audit before the irreversible step.** Routine checks passed. Only the deep audit found the half-day timing error
    and the estimator dependence.

## 10. Status, costs and next steps

- **Status.** P1–P3 are done and P4 has not been run. The clean window (2025-02-01 → 2026-09-22) is unopened. The
  operator has two open decisions:
  - BP-1: does the frozen-funnel t-statistic still license the look?
  - BP-2: spend the look at about 20% power, or close?
- **Recommendation: close without spending the look.** The look has only a 20–25% chance of passing. Even a pass
  would show that the effect is alive, not that it pays, and trading it would still need a new study.
- **Spent:** three calendar days and about $7 of Jev credits (Phase 0 $0.13, scoring $6.57). The data was free: SEC
  EDGAR, yfinance and Alpaca paper.
- **If pursued anyway:**
  - fix blockers N1–N10 from the audit, about a day of work;
  - confirm `jev-1.13.0` is still served;
  - then look once (about $1.2 of Jev scoring), under reading rules declared in advance in a prereg §10 addendum.
- **Better uses of the next effort:**
  - the TSMOM / TAILWIND path;
  - or a pre-registered forward test of an LLM text signal in a less-covered market.

## Appendix A — Glossary

- **Basis point (bp):** 0.01%.
- **IC (information coefficient):** the rank correlation, across stocks on one day, between a signal and the next h
  days' returns, averaged over days. 0 means no information, and useful equity signals typically run a few
  hundredths.
- **IC t-statistic:** how many standard errors the average IC sits from zero. Sharpen's promotion bar is 3.
- **IC-IR:** mean IC divided by its standard deviation, a measure of consistency.
- **Sharpe ratio:** the long-short portfolio's annualized return divided by its volatility. "Before costs"
  (frictionless) ignores trading costs; "net" includes them.
- **Deflated Sharpe ratio (DSR):** the probability that a Sharpe is real after accounting for how many strategies were
  tried (here, 8).
- **FDR q:** a p-value adjusted for testing several signals at once.
- **Point-in-time (PIT) universe:** index membership as it was on each date, so the test never trades stocks that only
  joined later.
- **Survivorship bias:** leaving out companies that later disappeared, which flatters results.
- **Pre-registration:** fixing hypotheses, data, rules and pass bars in writing (here, in git) before seeing results.
- **One-look clean window:** a period the model cannot have memorized, opened exactly once so the result cannot be
  tuned.
- **Placebo test:** re-running with the signal shuffled. A real effect should beat almost every shuffle.
- **Loughran–McDonald tone:** sentiment measured with finance-specific word lists.
- **Lazy Prices:** the finding that firms which change their filing language tend to underperform, measured here as
  similarity to the previous release.
- **Fama–MacBeth regression:** a daily cross-sectional regression, averaged over days. Here it tests whether Jev adds
  information beyond the baselines.
- **Newey–West / Hansen–Hodrick:** standard-error corrections for autocorrelation, which overlapping multi-day returns
  require.
- **Power / MDE80:** power is the chance a test detects an effect of a given size, and MDE80 is the smallest effect
  detected 80% of the time.
- **CPCV:** combinatorial purged cross-validation, meaning many train/test splits run to check robustness.

## Appendix B — Commits (branch `agentic-trading-lab`)

| Commit | Date | Step |
|---|---|---|
| `0bc1d581` | 2026-09-23 | Phase 0: Jev client, EDGAR filing text, contamination and power probes |
| `9c7f6df1` | 2026-09-23 | Phase 1 pre-registration frozen (questionnaire `3fc01888e31f`) |
| `e67a01b2` | 2026-09-23 | Phase 2 design, the five registered signals |
| `147424d3` | 2026-09-24 | Corpus builder, scorer, baselines, screening panel, P1–P3 legs |
| `094159bd` | 2026-09-24 | Tier-1 audit follow-ups |
| `166edc07` | 2026-09-24 | Scorer survives Jev HTTP 529 overloads |
| `c3755135` | 2026-09-24 | P3 pass; K3 decision: proceed to P4 |
| `ba06333c` | 2026-09-25 | Tier-2 deep lifecycle audit: P4 not ready |
