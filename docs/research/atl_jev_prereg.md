# ATL × Jev — pre-registration (Phase 1)

**Status:** FROZEN at the commit that adds this file. No filing is scored for the strategy before that commit.
**Questionnaire:** `sharpen/jev/questionnaire.py`, version `v1`, hash **`3fc01888e31f`** (see §8).
**Rules file:** `configs/atl_jev.gates.yaml` (`phase1`). **Research basis:** `docs/research/atl_jev_phase1_research.md`.
**Plan:** `docs/research/atl_jev_strategy_plan_2026-09-23.md`.

Anything here changes only through a new pre-registration: a new questionnaire hash or a new signal, with the
multiplicity raised to match. Rewording a question, adding a signal, moving a window or taking a second look at
the clean window after seeing any result is not a revision. It is a new study.

## 1. Hypotheses

The qualitative content of 8-K filings carries information about S&P 500 returns in the weeks after the market's
first reaction. The market does not fully price it. The information is:
- guidance changes and forward language;
- tone inflated beyond the reported numbers;
- earnings flattered by one-offs;
- adverse events.

Mechanisms: text surprise beyond the number (PEAD.txt), tone management, and negative post-filing drift.

| Signal | Questions (blocks) | Hold | Role |
|---|---|---|---|
| `jev-comp-63` | surprise + tone + quality | 63 trading days | **primary** |
| `jev-comp-21` | surprise + tone + quality | 21 | secondary |
| `jev-surprise-63` | surprise (E1–E3) | 63 | secondary |
| `jev-toneinfl-63` | tone (E4, negated) | 63 | secondary |
| `jev-quality-63` | quality (E5, M1, negated) | 63 | secondary |

All five are registered with `expected_sign = +1`. Question signs are applied inside the score, so a higher score is
always better news. Deflation is charged at **n = 8**, the trial budget announced in the plan, not the 5 used.

## 2. Data

- **Universe:** S&P 500 point-in-time membership (`data/raw/equity_panel/sp500_pit_members.csv`, as-of join).
- **Prices:** the PIT-union panel. 270 unpriceable delisted names are missing, so screening results carry the
  funnel's UPPER BOUND caveat.
- **Windows:**
  - *Screening* 2012-01-01 → 2024-12-31. This era is contaminated: Phase 0 found regime memory through 2023 and
    93.5% filer identifiability. P1–P3 can therefore screen but never certify.
  - *Clean* 2025-02-01 → 2026-09-22: the adopted T_c of 2025-01 plus a 1-month embargo. It gets **one look** (P4).
- **Filings:** EDGAR 8-Ks for index members, in scope by item.
  - Item 2.02 (earnings release) → E1–E5 and M1, read from the EX-99.1 press release (the 8-K body if absent).
  - Items 1.02, 1.03, 2.04, 2.05, 2.06, 3.01, 4.01, 4.02 → M1. Read from the 8-K body with the cover page
    stripped, followed by any EX-99.1. (Item 5.02 left scope with the dropped M2; see §8.)
  - All other 8-Ks are out of scope.
- **Text handling:** masked with `sharpen.jev.anonymize`, and truncated to 24,000 characters.
- **Model:** Jev, requested as `jev-latest`, with the served version recorded on every answer. A version change
  mid-study is a new model, and scoring stops (plan C6).

## 3. Timestamps (LEAK-2) — the release row

The funnel pairs row *t* with the return `close[t+h]/close[t] − 1`. A filing with EDGAR `acceptanceDateTime` τ
therefore enters on **row t = the trading day τ falls on, if τ is before 15:30 ET; otherwise the next trading day**.
ET means America/New_York local time, DST-aware. Weekends and holidays roll forward to the next trading day. The zone
and cutoff are read from `phase1.release_row` (`sharpen.jev.release`), not written into code.

Consequences:
- A pre-open release never earns that day's opening reaction.
- An after-close release skips the entire next-day reaction.
- The test therefore measures only drift after the first reaction.

This rule is implemented in Phase 2 with a **negative test** before any filing is scored. The test fails if a filing
accepted at 15:31 ET, or after the close, can reach the same day's row. It includes a date on each side of a DST
change, since the UTC offset differs.

## 4. Signal construction

1. **Filing block score.** For each in-scope filing and block, the mean of `sign × mapped answer` over the answered
   questions (`filing_score`). Mapping: `noul` → p, the evidence that the event occurs, so a "no" contributes
   about 0; `score` → 2s/(L − 1) − 1, centered on the neutral level. The mapping is part of the hashed
   definition (`SCORING`).
2. **Daily signal** (`phase1.construction`). For each name and day, only in-scope filings whose release rows lie in
   the last `hold_days` rows count:
   - *surprise* and *tone:* the block score of the most recent such earnings release;
   - *quality:* the **minimum** block score over all such filings, so a later routine filing cannot erase an
     adverse flag (§8);
   - several filings released on one row are averaged first;
   - a composite signal is the mean of the available block scores.

   With no such filing the value is NaN, and the name drops out of that day's cross-section, never scored as 0.
3. **Window edges.** Filings released up to `hold_days` rows before a window's first row count toward that window's
   early rows (warm-up). Evaluation uses only the window's own rows. **P4 is stricter:** only filings accepted on
   or after `p4_clean_window.min_filing_accepted` (2025-01-01, the adopted T_c) feed it, so no December-2024 score
   is held into February-2025 rows.
4. **Evaluation.** The frozen funnel (`configs/signal_eval.gates.yaml`, hash `519158fa1450`) applies its default
   neutralization (winsorize, z-score, sector).

## 5. Evaluation plan and pass bar

| Leg | Test | Bar |
|---|---|---|
| **P1** | The five signals in one batch, screening window, frozen funnel, `Multiplicity.preregistered(8)` | `PROMISING` under the frozen gates |
| **P2** | Shuffled-firm placebo: filing scores permuted across filers whose release rows share a calendar week, 200 times, with each IC computed by the funnel's own function. Plus the funnel's own cost legs. | The real 5-day IC beats the placebo distribution at p ≤ 0.05 |
| **P3** | B1 baselines on the same documents and release rows: Loughran–McDonald net tone, and cosine similarity to the name's previous in-scope release (Lazy-Prices-style). Daily Fama–MacBeth regression of 5-day forward-return ranks on the Jev signal and both baselines. | Newey–West (5 lags) t of the Jev coefficient ≥ 2.0 |
| **P4** | The single top-ranked `PROMISING` signal (funnel `rank_key`), clean window (411 NYSE days), one look. The IC t is the funnel's own `tier1_gross_power` `ic_tstat` at the primary horizon; the Sharpe is `tier2_capturability` `frictionless_sharpe`. | 5-day IC t ≥ 2.0 **and** frictionless long-short Sharpe > 0 |
| **P5** | Crucible lockbox via the B8 seam | ≥ 63 forward bars, forward marginal Sharpe ≥ 0.30 (`crucible_lockbox.gates.yaml`) |

**Kill rules:**
- **K3:** no signal is `PROMISING` on the screening window, or none clears P3. The result is NO-GO, and the clean
  window is never opened.
- **K4:** P4 fails. NO-GO.
- **K5:** P5 fails. REJECTED.

A pass through P5 makes the signal eligible for the human Tier-2 audit and nothing more.

**Power, measured in advance** (`scripts/research/atl_jev_prereg_power.py`; `results/atl_jev/phase1/prereg_power_h*.json`).
This replaces Phase 0's uniform-timing, 21-day model and its √days extrapolation, which Math audit findings M-2 and
M-3 showed did not describe the registered signals.

*Method:*
- Each name reports once a quarter, 18–45 days after quarter end (25–60 for Q4), at a stable position in the
  reporting season.
- The planted score is held for the registered hold, or until the next release, and correlates with the post-release
  drift.
- 8 batches per level, each 1 planted + 7 nulls, charged at n = 8, on the frozen funnel.
- P4 is computed with the funnel's own IC and capturability functions on a window of the clean window's exact length
  (411 NYSE days).

| Hold | Measured 5d IC | P1 detection | P4 power (mean t) | Scored names/day (median) · days < 50 names |
|---|---|---|---|---|
| 63 (4 signals, incl. primary) | 0.000 | 0/8 | 0/8 (−0.04) | 408 · 1% |
| 63 | 0.010 | **7/8** | **5/8 (2.48)** | |
| 63 | 0.020 | 8/8 | 8/8 (4.86) | |
| 63 | 0.030 | 8/8 | 8/8 (7.05) | |
| 63 | 0.050 | 8/8 | 8/8 (10.71) | |
| 21 (`jev-comp-21`) | 0.000 | 0/8 | 0/8 (−0.17) | 102 · **41%** |
| 21 | 0.010 | 3/8 | 2/8 (1.31) | |
| 21 | 0.021 | 8/8 | 7/8 (2.75) | |
| 21 | 0.031 | 8/8 | 8/8 (4.15) | |
| 21 | 0.051 | 8/8 | 8/8 (6.77) | |

- **The primary signal can be confirmed from an IC of about 0.01–0.02.** Screening detects 0.010 in 7/8 batches, and
  the clean window confirms it 5/8 of the time. At 0.020 both legs are certain.
- **`jev-comp-21` needs about 0.02.** Seasonal timing leaves 41% of its days below the frozen 50-name floor.
- **No false positives:** 0 of 280 nulls per hold scored `PROMISING`, and no zero-strength plant passed either leg.

## 6. Contamination statement

Screening-era answers come from a model that remembers 2018–2023 market regimes and can name 93.5% of masked
filers. Every P1–P3 number on 2012–2024 is therefore an UPPER BOUND.

The design limits the leak in three ways:
- questions are extraction-only, with the answer in the document;
- no question asks about returns or outcomes;
- nothing Jev sees contains a return, price or verdict.

Only P4 and P5 run on data the model cannot have seen.

## 7. Budget

About 70k in-scope filings (≈ 500 names × 13.7 years × ~10 8-Ks per year) at ~6.6k input tokens each ⇒ **≈ $20 of
Jev** calls. The EDGAR pull takes ~10 h. No GPU.

## 8. Instrument check (before freezing; no returns read)

**Rules** were fixed in `phase1.instrument_check` before the first run:
- Every question's mapped answers must vary (std ≥ 0.10, applied once n ≥ 10).
- An outlook must not be both raised and lowered on more than 10% of releases.

**Sample:** screening era 2019–2023 only, with per-company seeding: 40 Item 2.02 releases and 20 in-scope event
8-Ks. No return was read at any point. Artifacts are in `results/atl_jev/phase1/`.

**Run 1** (draft questionnaire, hash `14787b6c917d`): **M2 failed.**
- M2 (unplanned CEO/CFO exit) had std 0.037 over 17 answers. Every sampled Item 5.02 filing was routine
  (compensation arrangements, elections, planned transitions such as GE's), and Jev correctly answered "no" to
  all of them.
- The rule does not distinguish "never varies" from "rare". Loosening it after seeing that would be post hoc, so
  **M2 was dropped** and Item 5.02 left scope with it.

**Math audit (M-1), implemented while Run 1 was still running, before its result was read.** The noul mapping
changed from 2p − 1 to p. With 2p − 1, a clean event filing outranked a news-free earnings release purely by
question count. The mapping is now inside the hash.

**Run 2** (final questionnaire, hash **`3fc01888e31f`**): **PASS.**

| Q | n | mean | std | decisive share |
|---|---|---|---|---|
| E1 raise outlook | 40 | 0.231 | 0.356 | 0.175 |
| E2 lower/withdraw | 40 | 0.087 | 0.171 | 0.050 |
| E3 forward momentum | 40 | 0.412 | 0.385 | 0.400 |
| E4 tone beyond results | 40 | 0.417 | 0.144 | 0.275 |
| E5 one-off flattered | 40 | 0.444 | 0.248 | 0.400 |
| M1 adverse development | 60 | 0.368 | 0.360 | 0.367 |

- Raise/lower conflicts: 0.
- **Spot checks read correctly against the source text.** Axon's "Raising Full-Year Revenue Outlook" scored E1 =
  0.99. Boston Scientific's "withdrew its Q1 2020 sales and EPS guidance" scored E2 = 0.96.
- E4 mostly hedges (only 27.5% of answers are decisive). It passes the rule, and its weakness is noted here in
  advance.

**Construction change made *after* Run 2, from its sample's item mix and answers, with no returns read.** 11 of the
20 in-scope event filings were routine credit-facility refinancings (Items 1.01 + 1.02 + 2.03), and M1 correctly
scored them as non-adverse. Under a "latest filing wins" rule, such a filing would erase an earlier impairment or
one-off flag from the quality block. So the quality block takes the **minimum** in-window filing score
(`construction.quality: min_in_window`). The adverse-event mechanism is a lasting drift, which a later routine filing
does not undo. Surprise and tone keep the latest earnings release.
- The two runs cost $0.014 of Jev calls. Every answer came from `jev-1.13.0`.

**Stamps:** both runs, and the §5 power study, read the pre-freeze gates file. The frozen file adds only
`questionnaire.hash`, `construction`, `release_row` and `p4_clean_window.min_filing_accepted`. Every section those
runs read is unchanged.
