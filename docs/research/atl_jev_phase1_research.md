# ATL × Jev Phase 1: which filing questions, with which signs?

> **Created:** 2026-09-23 | **Workstream:** `atl-jev` (branch `agentic-trading-lab`)
> **Question:** Which extraction-only Jev questionnaire over SEC 8-K filings has the best-documented mechanism for
> cross-sectional S&P 500 return predictability 5–63 trading days after the filing? What effect size is realistic,
> and what are the non-LLM baselines?
> **Verdict:** **CONDITIONAL GO.** Conditions are in the Recommendation below.

## Executive summary

One new channel survives the record: the *qualitative* content of earnings releases. Guidance changes, forward
language, tone inflated beyond the numbers, and earnings flattered by one-offs are all things an LLM reads and a
numeric surprise misses. The strongest support is PEAD.txt, a surprise built from earnings-call language alone. It
drifts almost twice as far as numeric PEAD (2010–2019) and "is far from disappearing" in its second half.

Everything else in the record points to a small large-cap effect:
- the project's own numeric-PEAD test is dead in liquid names;
- Lazy Prices is dead on the S&P 100 (t = −0.5);
- LLM-news predictability lives mostly in small stocks and is decaying with adoption.

The realistic 5-day IC for an S&P 500 text signal is **~0.01–0.02**. That is at or below the frozen funnel's MDE80
of 0.02. The test is cheap (~$20 of Jev) and decisive either way, so it is worth running on a questionnaire with
documented mechanisms and a priori signs. The expected outcome is still a clean NO-GO.

## Prior art (internal)

| Record | Result | Bearing on this design |
|---|---|---|
| PEAD post-pandemic (S553-cont-62, 2026-06-22; `.agent/artifacts/pead_post_pandemic_research.md`) | NO-GO. 1,688 events in liquid US names: IC@20 ≈ −0.001, standardized SUE +0.016 (p = 0.51), and a 1-day **reversal** in large/mega caps (t = −2.29). | The *numeric* surprise is priced. Only content beyond the number is still open. |
| XLG / mega-cap overlay (S553-cont-74) | IC collapses on the top-50: 0/100 FDR survivors, against 40/100 on the broad universe. | Large caps are the hardest universe. The breadth of S&P 500 PIT is needed. |
| Liquid large-cap cross-section closed (cont-73…78) | Reversal and momentum premia are arbitraged to about cost. | Do not re-propose price signals. This design adds a new information source (text). |
| Crucible spec review (S553-cont-97) | "LLM economic priors are partly memorized backtests." | Confirmed by Phase 0 C1/C3. Certify only on post-cutoff data. |
| Phase 0 (`0bc1d581`) | Jev regime memory runs through 2023. Masked filings are 93.5% identifiable. S&P 500 MDE80 ≈ IC 0.02; DJIA-30 unreachable. | Defines the screening/certification split and the verdict universe. |

No text or LLM signal has ever been scored through the funnel. WandB has no runs for this technique; the PEAD study
ran as tmp scripts.

**Stale guidance noted.** The researcher reference's "no daily timeframes (sample collapse)" rule and its oracle-PF
ceilings come from the single-asset intraday RL era. They do not apply to a daily cross-sectional funnel, whose
breadth was measured directly in Phase 0.

## Literature review

- **PEAD.txt** (Meursault, Liang, Routledge & Scanlon, *JFQA* 2023; figures read from the paper).
  - *Method:* a regularized text regression on earnings-call transcripts, re-fit quarterly on the prior 2 years and
    used out of sample. Sample 2010–2019.
  - *Results:* the quintile spread from the day after the call reaches 2.87% / 4.61% / 6.51% / 8.01% at 63 / 126 /
    189 / 252 days. Numeric PEAD reaches 1.54% / 2.70% / 3.87% / 4.63%. Both shrink in the second half; PEAD.txt
    less so. No size split is reported.
  - *Relevance:* the mechanism for a +1 "text surprise" block (guidance and forward details behind the number).
  - *Caveats:* call transcripts rather than press releases, and a broad rather than large-cap universe.
- **Tone management** (Huang, Teoh & Zhang, *TAR* 2014). Abnormally positive tone in earnings press releases
  relative to the numbers brings a positive announcement return, then a *delayed negative* reaction over the next
  one to two quarters. It also predicts weaker future earnings. This is the mechanism for a **−1 "tone beyond
  results"** question, and it is the reason "tone" must be asked *relative to the numbers*, not in absolute terms.
- **8-K items** (Lerman & Livnat, *RAST* 2010; 2005–2006 sample). Drift after the filing is overwhelmingly
  *negative*: bankruptcy −15% to −19% over 30–90 days; change in control and accelerated obligations −1.5% to −3.7%,
  significant only at 30–90 days. "Very few" items drift positive. This is the mechanism for a −1 adverse-event
  question. These events are rare in the S&P 500 and act slowly.
- **GPT and news** (Lopez-Lira & Tang, 2023–24). On post-cutoff headlines, GPT-4 scores predict drift "especially for
  small stocks and negative news," and "strategy returns decline as LLM adoption rises." Expect a smaller and
  shrinking effect in large caps.
- **Look-ahead and distraction** (Glasserman & Lin, 2023). Inside the training window, anonymized headlines
  *outperform* named ones, and the "distraction effect" is strongest for large firms. Masking is still worth doing
  against distraction, but Phase 0 C3 shows it cannot remove identity for large-cap filings.
- **LLM statement analysis** (Kim, Muhn & Nikolaev, 2024). GPT-4 beats analysts on the direction of earnings changes
  from anonymized statements. That defence rests on anonymization; our C3 result (93.5% identifiable) suggests it is
  weaker for large caps than assumed.
- **Lazy Prices on the S&P 100** (free-data replication; figures read from the raw README). Over 2009-03 → 2026-09:
  FF5+MOM alpha −0.9% (t = −0.5), Sharpe −0.11 [−0.56, 0.38], rank correlation 0.03 (t = 1.1). The best of 54
  variants has a deflated Sharpe ratio of 0.13. The original 1994–2014 effect does not survive among the
  most-watched firms. The B1 baseline is therefore expected to be about 0 here.
- **Newer earnings-call LLM work** (Zhang & Zhou, EAR-AI, 2009–2024, chronologically consistent LLMs). This is
  reported in the abstract as robust out-of-sample Sharpe. It was not verifiable (SSRN returned 403), so it carries
  no weight here.

## Codebase alignment

| Dimension | Assessment | Notes |
|---|---|---|
| Interface | Pass | `Signal` = `SignalSpec` + `compute(panel)`. The event score is held over a window as a (T, N) matrix. |
| Data | Partial | EDGAR text + S&P 500 PIT panel exist. **LEAK-2 release-row mapping (A10) must be built first.** |
| Architecture | Pass | `sharpen/jev/{client,anonymize}`, `edgar_filings` exist. `questionnaire.py` (B2) is new. |
| Compute | Pass | ~70k filings × 6.7k tokens ≈ $20 of Jev. EDGAR fetch ~10 h. No GPU. |
| Prior failures | Partial | Numeric PEAD is dead. Text-beyond-number is untested here, and it is the one new channel. |
| Invariants | Pass | Frozen funnel gates are untouched (CRU-1). Extraction-only questions. No outcome reaches Jev. |

## Assessment matrix

| Factor | Score (1–5) | Evidence |
|---|---|---|
| Theoretical soundness | 4 | Three documented mechanisms: text surprise (PEAD.txt), tone management, adverse-event drift. |
| Project fit | 4 | Linear core, frozen funnel, free data, ATL as runtime. RL has no role (signal, not execution). |
| Implementation effort | 3 | LEAK-2 mapping, corpus pipeline, B2 module, signals, baselines: about a week. |
| Risk of failure | 2 | Large caps are efficient, numeric PEAD is dead, Lazy Prices is dead in the S&P 100, and LLM edge decays with adoption. |
| Expected improvement | 2 | Plausible IC 0.01–0.02 ⇒ frictionless L/S Sharpe ~1.0–1.8 on the planted structure, if real. |
| Opportunity cost | 4 | ~$20 and a week. The pipeline is reusable for a later mid-cap pre-registration. |

## Recommendation

**Verdict: CONDITIONAL GO.** Pre-register and run. The conditions:
1. **Instrument check before freezing.** Run the draft questionnaire on screening-era filings only, with no returns
   looked at. Each question must be non-degenerate and internally consistent (rules in the pre-registration).
2. **Build the LEAK-2 mapping and its negative test before any filing is scored** (A10).
3. **The verdict universe is the S&P 500 only.** Mid-caps, where the literature finds larger effects, would be a
   separate future pre-registration with its own multiplicity.
4. **Short-hold "reversal" variants are excluded.** A 5-day hold covers ~8% of names per day, below the frozen
   50-name floor.

## Implementation sketch (frozen in the pre-registration)

**Earnings-release questions** (Item 2.02 EX-99.1, masked). Every instruction says to answer only from the document.

| ID | Type | Question (short form) | Sign | Mechanism |
|---|---|---|---|---|
| E1 | noul | Raises its outlook versus previously given guidance | +1 | Text surprise (PEAD.txt) |
| E2 | noul | Lowers or withdraws its outlook | −1 | Text surprise |
| E3 | score (5) | Management's description of forward demand/momentum | +1 | Details behind the number |
| E4 | noul | Tone noticeably more optimistic than the reported results justify | −1 | Tone management |
| E5 | noul | Headline results flattered by self-identified one-off items | −1 | Earnings quality |

**All-8-K question.** M1 is asked of every in-scope filing.

| ID | Type | Question (short form) | Sign | Mechanism |
|---|---|---|---|---|
| M1 | noul | Discloses a material adverse development (impairment, restatement, lost contract, covenant/going-concern, accelerated obligation) | −1 | Adverse-event drift |

A draft **M2** (unplanned CEO/CFO departure) was **dropped by the pre-freeze instrument check**. All 17 sampled Item
5.02 filings were routine, so its answers never varied (std 0.037 < 0.10). Item 5.02 left scope with it.

**Scoring.**
- *Answer mapping:* `noul` → p, the evidence that the event occurs, so a "no" contributes about 0; `score` →
  2s/(L − 1) − 1, centered on "stable, or not discussed". The draft 2p − 1 noul mapping was rejected in the Math
  audit: it made a filing's neutral level depend on how many negatively signed questions it was asked.
- *Filing score:* the mean of sign × mapped answer over the questions asked.
- *Daily signal:* each name carries its latest filing score from its LEAK-2 release row for the hold window, and is
  NaN otherwise.

**Signals** (5 registered; deflation charged at **n = 8**, the budget pre-announced in the plan). Question signs are
applied inside the filing score, so a higher score always means better news. Every signal is therefore registered
with `expected_sign = +1`.

| Signal | Questions | Hold | Role |
|---|---|---|---|
| COMP-63 | all | 63 d | Primary |
| COMP-21 | all | 21 d | |
| SURPRISE-63 | E1–E3 | 63 d | |
| TONEINFL-63 | E4 (enters as −E4) | 63 d | |
| QUALITY-63 | E5, M1 (each enters negated) | 63 d | |

**Baselines (outside the batch).**
- B1: Loughran–McDonald net tone, and similarity to the prior release (Lazy-Prices-style). Jev must add marginal
  information (P3).
- B2: a shuffled-firm placebo.

Hand-off: the Architect designs Phase 2 (B2 questionnaire module, LEAK-2 release row, B5 signals, baselines, B8
lockbox seam).

## References

1. Meursault, Liang, Routledge & Scanlon (2023), "PEAD.txt: Post-Earnings-Announcement Drift Using Text," *JFQA*. https://doi.org/10.1017/S0022109022001181
2. Huang, Teoh & Zhang (2014), "Tone Management," *The Accounting Review* 89(3). https://publications.aaahq.org/accounting-review/article/89/3/1083/3532/Tone-Management
3. Lerman & Livnat (2010), "The new Form 8-K disclosures," *Review of Accounting Studies* 15. https://link.springer.com/article/10.1007/s11142-009-9114-7
4. Lopez-Lira & Tang (2023–24), "Can ChatGPT Forecast Stock Price Movements?" https://arxiv.org/abs/2304.07619
5. Glasserman & Lin (2023), "Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis." https://arxiv.org/abs/2309.17322
6. Kim, Muhn & Nikolaev (2024), "Financial Statement Analysis with Large Language Models." https://arxiv.org/abs/2407.17866
7. Cohen, Malloy & Nguyen (2020), "Lazy Prices," *Journal of Finance*; S&P 100 replication: https://github.com/iqueipopg/lazy-prices
8. Internal: `.agent/artifacts/pead_post_pandemic_research.md`; `docs/research/atl_jev_strategy_plan_2026-09-23.md` §8.
