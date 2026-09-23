# ATL × Jev — plan for an agentic strategy that passes Sharpen

**Date:** 2026-09-23 · **Branch:** `agentic-trading-lab` · **Workstream id:** `atl-jev` · **Status:** PLAN (nothing built, nothing scored)

Goal: build a trading strategy on the Agentic Trading Lab (ATL, `Open-Finance-Lab/AgenticTrading`) with
TypeSafe's **Jev** as the agent's decision model, and have it clear Sharpen's falsification bar.

## 0. Summary

- **Jev's role: reading filings, not forecasting prices.** Jev reads each SEC 8-K filing (with its EX-99 press release) at its
  EDGAR acceptance time. It answers a fixed, pre-registered set of *extraction* questions. A pre-registered linear rule
  maps those answers to a cross-sectional score. It does not read ATL's RSI/MACD/SMA/Bollinger features: technical
  signals on liquid large caps are a closed family in this repo's record.
- **Main risk: pretraining look-ahead.** Jev was released 2026-09-15 and its training cutoff is undisclosed. It may
  "know" what happened after any historical filing. Only data after Jev's *measured* knowledge cutoff, plus the forward
  lockbox, can certify a result. Everything before that cutoff is screening, labelled UPPER BOUND.
- **Verdict universe: S&P 500 PIT daily.** The frozen funnel requires ≥ 50 names/day, so DJIA-30 fails its first
  hygiene tier. ATL's DJIA-30 hourly environment is the agent runtime, the execution-parity check, and the paper
  venue.
- **Jev must beat a non-LLM text baseline.** It is measured against Loughran–McDonald tone and Lazy-Prices similarity on
  the same filings at matched turnover. This is the same rule the project applies to RL: beat the linear baseline out
  of sample.
- **Honest prior: low.** The closest relatives in the record (post-pandemic PEAD, liquid large-cap cross-section) are
  NO-GO. Published LLM-news alpha concentrates in small caps and in the short (negative-news) leg. ATL's environment
  cannot short, but the Alpaca paper account can, so the long-short book can still be paper-traded directly. The plan is built so that a NO-GO costs about two weeks and under $100, before the forward clock starts.

## 1. What "passes Sharpen" means here

All five legs are pre-registered before any filing is scored.

| Leg | Test | Bar (source) |
|---|---|---|
| P1 Funnel | `eval_signals.py` on S&P 500 PIT, with a declared multiplicity ledger (`--n-hypotheses`, `--multiplicity-ledger`, `--prereg`) | `PROMISING` under the frozen `configs/signal_eval.gates.yaml`: DSR ≥ 0.90, IC-IR ≥ 0.05, IC t ≥ 3, BH q ≤ 0.10, no negative subperiod, frictionless book > floor. Multiplicity must be declared, not batch-shaped. |
| P2 Falsification | Circular-shift timing null (`hma_cross_falsification.py` pattern), cost sweep (`sharpen/signals/costs.py`), planted-signal power (`planted_sweep.py`, `forward_power.py`) | Timing null p ≤ 0.05. Net Sharpe > 0 at standard cost. Reported MDE below the observed effect. |
| P3 Beats text baseline | Jev score vs LM tone and Lazy-Prices similarity, same filings and timestamps | Jev's marginal IC over the baseline is > 0 with t ≥ 2 (threshold in `configs/atl_jev.gates.yaml`) |
| P4 Clean window | One pre-registered look at [measured cutoff T_c + embargo, 2026-09-22] | Same sign as screening, frictionless book > 0, power stated |
| P5 Lockbox | Crucible lockbox forward incubation | ≥ 63 post-enrollment bars, forward marginal Sharpe ≥ 0.30 (`configs/crucible_lockbox.gates.yaml`) |

After P5, a Tier-2 deep lifecycle audit (`deep_strategy_audit`) is still required before any capital. The funnel's
verdict tops out at `PROMISING`, never GO.

## 2. What we are working with (verified 2026-09-23)

**Jev.** A System One model: typed answers only (`choice`, `score`, `noul`), each with probabilities and confidence. It
cannot generate text, accepts text input only, and costs $0.042 per million input tokens (output free). The smoke
test from this repo's `.env` key returned HTTP 200. The model served under the `jev-latest` alias was
**`jev-1.13.0`**, so the alias can move and the served version must be pinned in provenance. The call took 21.6 s and
used 375 input tokens for 3 questions. The repo already has an outcome-blind `JevRanker`
(`sharpen/crucible/agentic/jev_ranker.py`); we reuse its transport and fail-closed pattern.

**ATL environment (`us-equity-hourly-v1`).**
- Universe and action space: DJIA-30, hourly, long-only (`supports_shorting: false`), max weight 0.25, ≤ 10 market
  orders per step.
- Capital: `initial_cash` 0–3000 through the run API, so whole-share rounding is material.
- Observations: RSI, MACD, SMA, Bollinger bands and turnover, plus optional market cap and SIC code.
- Runtime: the SDK's `AgentRunner` only needs a `decide(observation)` method, with a 60 s decision deadline.
- Data and metrics: bars come from Alpaca SIP, and the reported Sharpe is undeflated.
- News: FinSearch news starts around 2026-07 and needs a bearer token.
- License: OpenMDW-1.0 (permissive; keep notices).
- **Alpaca:** paper keys were added to `.env` on 2026-09-23 under ATL's variable names and verified read-only. The
  account is ACTIVE with **shorting enabled** (ATL's environment still cannot short). Hourly `sip` bars came back for
  both a 2026 week and a 2016 week; the counts include extended-hours bars.

**Sharpen.** A signal is a `SignalSpec` (content-hashed pre-registration) plus `compute(panel)`. The repo has:
- local S&P 500 PIT membership (1996 onward) and a PIT-union price panel (2007 onward, yfinance). Survivorship bias
  gets the UPPER BOUND caveat, and the frozen gates require survivorship-free data for promotion. The certifying legs
  P4 and P5 are recent or forward, so they are nearly survivorship-free.
- `EdgarConnector` (XBRL facts), with `SEC_EDGAR_UA` configured.

EDGAR's submissions API was verified to return per-filing `acceptanceDateTime` (UTC), 8-K `items` (for example `2.02`
earnings, `5.02` officer change) and `primaryDocument`. The lockbox's `forward_evidence()` takes a DSL **formula**, not
a precomputed signal; see build item B8.

## 3. Design

```
EDGAR submissions + filing docs ──► anonymizer ──► Jev questionnaire ──► answer cache (content-addressed,
 (acceptanceDateTime, items)        (mask names,     (typed, hashed,        key = served model + prompt hash
                                     tickers, dates)   pre-registered)        + state hash)
                                                                                │
                     ┌──────────────────────────────────────────────────────────┤
                     ▼                                                          ▼
   Sharpen: compute(panel) → funnel (P1) → battery (P2)          ATL AgentRunner.decide(obs) reads the same cache
            → baseline (P3) → clean window (P4) → lockbox (P5)   → long-only DJIA-30 orders (parity + paper)
```

Roles follow FinAgents' pool decomposition (data / alpha / risk / portfolio / cost). The v1 build does not import
ATL's `orchestration/` MCP/A2A stack; the strategy does not need it.

### 3.1 Jev asks extraction questions; a fixed rule trades the answers

Illustrative question set (the final one, at most about 10 items, is pre-registered in Phase 1):

- `noul`: guidance raised / guidance cut versus prior guidance
- `score` (5 levels): results versus the prior-year period *as described in the document*
- `score`: management's characterization of business conditions
- `noul`: headline results driven by self-identified one-off items
- `noul`: material adverse development (impairment, restatement, lost contract, covenant or going-concern stress)
- `noul`: CEO/CFO departure without a named successor
- `choice`: primary event type

**Mapping.** Signs are fixed a priori from economic rationale. Answers are equal-weighted after z-scoring. The event
score decays with half-life H, and the frozen gates' neutralization (sector and size) is applied.

**Why this design.**
- **Extraction is checkable.** The answer is in the text, so a sample can be audited for accuracy. A forecasting answer
  can only come from the model's own knowledge, which is where memorization leaks in.
- **Typed answers are auditable.** They are hashable like a `SignalSpec`, reproducible from the cache, and need no
  output parsing.
- **Jev is cheap enough for breadth.** The full S&P 500 corpus is roughly $40–100 at list price, and breadth is what
  the funnel's power needs.

### 3.2 Contamination protocol (the load-bearing part)

| ID | Step | Decision it drives |
|---|---|---|
| C1 | **Knowledge-cutoff probe.** Dated market-fact `noul` questions: monthly direction of SPY and each DJIA name, 2018-01 → 2026-08, about 3k questions, well under $1. Plot accuracy by month. | T_c = the changepoint where accuracy falls to chance. The clean window is [T_c + embargo, now]. |
| C2 | **Anonymizing state renderer.** Masks registrant name, ticker, CIK and dates. Tripwire test: no identifier, price, return or verdict field reaches the request body (mirrors the ranker's CR-1 tripwire). | Required for every Jev call |
| C3 | **Identification probe.** Jev picks the filer from 10 same-sector candidates given the anonymized text (chance = 10%). | If identifiable above the pre-registered threshold, pre-T_c results are screening only (they are anyway) and H1 relies on P4 and P5 alone. |
| C4 | **Named-vs-anonymized A/B** in the screening era (Glasserman–Lin design) | If named beats anonymized, that is a knowledge leak. Reported. |
| C5 | **Deliberate leak detector.** A named, dated *forecast* question ("will it outperform over 5 days?"). This is the naive LLM-agent strategy. | It should "work" before T_c and die after. If it does not die, the clean window is not clean and P4 is void. |
| C6 | **Version pin.** Record the served `jev-x.y.z` on every answer. | A version change is a new trial and triggers a C1 re-run. |

### 3.3 Controls and baselines

- **B1 — non-LLM text baseline (P3).** Loughran–McDonald tone plus Lazy-Prices similarity on the same documents.
- **B2 — shuffled-firm placebo.** Each date's Jev scores are permuted across firms; IC should collapse.
- **B3 — ATL-native technical control.** Jev reads only ATL's anonymized technical observation. This is expected NO-GO
  (closed family). It doubles as the end-to-end ATL integration test.
- **B4 — matched-exposure equal-weight DJIA benchmark.** ATL's long-only book is mostly beta, so its alpha is measured
  against this, not against zero.

## 4. Phases

| Phase | Work | Output | Kill if |
|---|---|---|---|
| **0 Feasibility** (2–3 d) | Warm Jev throughput with 64-question batches and concurrency. C1 cutoff probe. C3 on about 200 filings. Planted-signal power (MDE) on S&P 500 PIT *and* DJIA-30. Clone ATL to `C:/FinRL/AgenticTrading`, install the SDK, run the rule-based example. | `results/atl_jev/phase0/*.json`, measured T_c, cost estimate | K0: corpus cost or runtime infeasible. K2: MDE above any plausible text-signal IC on both universes (gate unreachable; ask this before building). |
| **1 Pre-registration** (1–2 d) | Researcher skill (NotebookLM KB first). Freeze the questionnaire, signs, H (primary 5 d per frozen gates, secondary 21 d), trial budget (≤ 8 variants) and C/B thresholds. | `docs/research/atl_jev_prereg.md` + `configs/atl_jev.gates.yaml`, **committed before any scoring** | — |
| **2 Build** (~1 wk) | B1–B9 below. Audit skill after code; Math skill on the score, decay and mapping. | Tests green, ruff clean | — |
| **3 Screening** (pre-T_c, 3–4 d) | Score the corpus. Run the funnel with the declared ledger, B1–B4, C4 and C5. | Funnel cards, baseline table | K3: no variant `PROMISING` **or** none beats B1 ⇒ NO-GO, stop. |
| **4 Clean window** (1–2 d) | Single pre-registered look (P4) plus P2 battery. ATL replay of the DJIA-30 slice for execution parity (ATL fills and costs vs the Sharpen book; >30% divergence halts, as in PF-XCHECK). | Verdict artifact | K4: sign flip or non-positive frictionless book in the clean window. |
| **5 Forward lockbox** (≥ 63 trading days; earliest ≈ mid-Jan 2027) | Live EDGAR polling → Jev → cache. `AgentRunner` on ATL paper (long-only DJIA-30), plus the long-short verdict book traded directly on Alpaca paper. Lockbox enrollment. | Lockbox entry and verdict | K5: forward marginal Sharpe < 0.30 ⇒ REJECTED. |
| **6 Tier-2 audit** | `deep_strategy_audit` (finder + skeptic per pillar) | Audit report | Operator decision |

No model is trained, so Protocol v2's training stages do not apply. Each phase still writes one decision artifact.
All thresholds live in `configs/atl_jev.gates.yaml`, never in code.

## 5. Build list

These files stay inside the repo boundary (`sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`).

| # | Path | What |
|---|---|---|
| B1 | `sharpen/jev/client.py` | Typed System One client (`choice`/`score`/`noul`): injectable transport, fail-closed on missing key, retries, concurrency, disk cache keyed on served-model + prompt hash + state hash, token and cost ledger |
| B2 | `sharpen/jev/questionnaire.py` | Frozen question sets with a content hash (the strategy's pre-registration) |
| B3 | `sharpen/jev/anonymize.py` | Entity and date masking |
| B4 | `sharpen/crucible/data/edgar_filings.py` | Submissions API plus filing-document fetcher and text extraction. `release_ts` = first bar strictly after `acceptanceDateTime` (next session on the daily panel). Rate-limited to SEC fair access. |
| B5 | `sharpen/signals/library/jev_filings.py` | `Signal` whose `compute(panel)` turns cached answers into a decayed event-score panel. Also the B1 baseline signals. |
| B6 | `scripts/research/jev_{cutoff_probe,identification_probe,score_filings}.py` | Phase 0 probes and the resumable batch scorer |
| B7 | `scripts/research/atl_jev_agent.py` | ATL `AgentRunner` agent (`decide()` reads the cache → long-only top-k weights within ATL constraints), with replay and live modes. Also a direct Alpaca-paper mode for the long-short verdict book, since the account can short and ATL's environment cannot. |
| B8 | Lockbox seam (Architect decision) | `forward_evidence()` accepts only DSL formulas. Either add a per-name alt-data terminal (`jev:filing_score`) or add a `forward_evidence` variant over a precomputed return stream. Must be additive (CRU-1: no existing verdict changes). |
| B9 | `tests/jev/…`, `tests/test_edgar_filings_causality.py` | Anonymizer tripwire; cache determinism; version pin; transport-failure degrade; **negative LEAK-2 test** (an after-close filing cannot move the same-day bar) |

## 6. Budget

Measured in Phase 0 (`results/atl_jev/phase0/phase0_summary.json`):

- **Jev:** Phase 0 cost $0.13 in live calls. The full S&P 500 8-K corpus (about 100k filings at ~6.7k input
  tokens each, including a ~10-question questionnaire) is estimated at **$28**.
- **Throughput:** warm latency is ~0.6 s per request, whether it carries 64 short questions or one full filing; the
  21.6 s first call was a cold start. At 8-way concurrency the corpus takes ~2 h of Jev time. The **EDGAR pull
  dominates at ~14 h** (2 requests per filing at 4 req/s).
- **Other inputs:** EDGAR, ATL (run locally) and Alpaca paper are free. No GPU needed.

## 7. Decisions and actions needed

1. **ATL access — resolved 2026-09-23.** Alpaca paper keys are in `.env` and verified, so ATL runs locally. When ATL
   is cloned, its `dashboard/.env` gets the same three variables.
2. **Universe — settled by Phase 0's power check.** The verdict runs on S&P 500 PIT, and ATL DJIA-30 is used for
   execution and paper. Under the frozen gates DJIA-30 can never pass: it fails the breadth floor. Even with that floor
   waived it needs an IC of about 0.05 to be detected (see §8), above anything plausible for this signal.
3. **Phase 0 — done 2026-09-23.** Verdict: proceed to Phase 1 (see §8).
4. **Optional.** Re-authenticate NotebookLM (`notebooklm-mcp-auth`) for the Phase 1 literature pass.

## 8. Progress

**2026-09-23 — ATL clone and smoke test (Phase 0, partial).**
- **Clone:** `C:/FinRL/AgenticTrading` at `52a2b596`, outside this repo, 109 MB.
- **Python environment:** `.venv` (Python 3.13, matching ATL's pin, 587 MB) with the pinned `requirements.txt` and
  the SDK (`agentictrading` 0.2.0, editable install).
- **Config:** `dashboard/.env` holds the Alpaca paper keys plus
  `DATABASE_PATH=C:/FinRL/FinRL-Pro_DS/results/atl_jev/atl_runs.db`. **Gotcha:** without that override, ATL writes
  run rows into its git-tracked `dashboard/storage/data/backtest.db`, which dirties the clone and blocks `git pull`.
- **Smoke test:** `HourlyBacktester(..., use_llm=False)`, the rule-based reference agent (no LLM, no account). It ran
  AAPL/MSFT/JPM/KO over 2026-09-08 → 09-11 in 2.5 s. Data came from `sip`, with no IEX fallback and no clamp. There
  were 28 hourly decision bars, and the agent made 0 trades. Equal-weight buy-and-hold returned +1.76%. A re-run
  from the bar cache was identical.
- **Parity note:** ATL decides hourly but executes and values on **5-minute** bars. The Phase 4 parity check must
  mirror that rather than assume hourly fills.
- **Disk:** drive C: has about 13 GB free. Store filings as cleaned, compressed text (parquet/zstd) plus answer
  caches, not raw HTML.

**2026-09-23 — Phase 0 complete: PROCEED to Phase 1.** The rules were pre-registered in `configs/atl_jev.gates.yaml`
before any probe ran. Artifacts are in `results/atl_jev/phase0/`. Every answer came from `jev-1.13.0`, and live Jev
spend was $0.13.

- **C1, knowledge cutoff: Jev remembers market regimes.**
  - *Method:* 10,207 dated questions ("did X close higher or lower in month m?") over 101 securities, 2018-01 →
    2026-08, with the wording randomized between higher and lower.
  - *Pre-registered rule:* the item-accuracy changepoint found **no** cutoff.
  - *Why that rule is misleading:* it is confounded, because a model that merely assumes "stocks usually rise"
    scores each period's share of up-months. The docstring now records this limitation.
  - *The statistic that works:* the correlation between Jev's average implied P(up) in a month and the share of
    securities that actually rose. A constant prior cannot produce this correlation.
  - *Result:* +0.18 / +0.21 / **+0.69** / +0.28 / **+0.54** / **+0.58** for 2018–2023, then −0.11 (2024), −0.40
    (2025) and −0.72 (2026). The 2020 COVID and 2022 selloff months are recalled clearly: March 2020 averaged an
    implied P(up) of 0.19 when 15% of names rose.
  - **Adopted: T_c = 2025-01, so the clean window runs from 2025-02-01.** This deviates from the pre-registered rule,
    in the conservative direction only, and the rule is written into `atl_jev_phase0_summary.py`.
- **C3, identification: masking does not hide the filer.**
  - *Method:* 200 Item 2.02 earnings releases (30 from DJIA names), 2019–2025, with names, tickers, datelines, dates,
    URLs and phone numbers masked. Zero residual identifiers survived the masking.
  - *Result:* Jev still picked the filer from 10 same-sector candidates **93.5%** of the time (95% CI 89–96%;
    chance is 10%; the unmasked control scored 100%). It picked the year 49.5% of the time against 14% chance.
  - *Why:* brands, products, segments and executive names give the filer away.
  - *Consequence:* anything scored before 2025 can only screen (UPPER BOUND). **C4 is dropped**: the named-vs-masked
    A/B compares two near-identical conditions. The real protections are extraction-only questions plus P4 and P5.
- **K2, power under the frozen funnel.** Setup: a planted filing-event signal (quarterly events, each score held 21
  days, so a third of names are active), 8 batches per level, each batch 1 planted + 7 null signals with 8 hypotheses
  declared.
  - *False positives:* **none** — 0 of 672 null signals scored PROMISING, and 0 of 16 zero-strength plants (S&P 500
    and the DJIA what-if combined).
  - *S&P 500 PIT (2012–2024, median 422 active names):* measured IC 0.011 → detected 62%; IC 0.021 → **100%**, so
    **MDE80 ≈ 0.02**, which is reachable.
  - *DJIA-30 under the frozen gates:* unreachable by construction (breadth floor).
  - *DJIA-30 with the floor waived (to 8 names):* IC 0.031 → 12%; IC 0.057 → 88%, so MDE80 ≈ 0.05, above the
    plausible 0.03.
  - *Clean-window power (planning estimate, t ∝ √days, ~415 days):* on the S&P 500, IC 0.02 → t ≈ 2.9 and IC 0.03
    → t ≈ 4.2. On DJIA-30, IC 0.03 → t ≈ 1.1. P4 has power on the S&P 500 and none on DJIA-30.
- **K0, feasibility:** $28 and ~16 h for the full corpus. Does not fire.
- **Tier-1 audit of the Phase 0 code:** 12 findings — 9 fixed, 2 accepted, 1 open.
  - The 2 accepted findings are hardcoded design constants and an in-memory OHLC repair in the power study (counts
    recorded).
  - Tests: 1,379 pass on the floor, Crucible, signals and Jev suites. All 7 injected mutations of the new safeguards
    were caught.
  - CRU-1 holds: `signal_eval.gates.yaml` is untouched at `519158fa1450`.
  - **Open, and blocking Phase 3:** the LEAK-2 mapping from `acceptanceDateTime` to the first tradable row does not
    exist yet, and it needs its negative test (B9).

**2026-09-23 — Phase 1 complete: pre-registration FROZEN** (`docs/research/atl_jev_prereg.md`; questionnaire `v1`
hash `3fc01888e31f`; rules in `configs/atl_jev.gates.yaml` `phase1`). No filing has been scored for the strategy.
- **Research** (`docs/research/atl_jev_phase1_research.md`): **CONDITIONAL GO**.
  - *For:* PEAD.txt text drift persists where numeric PEAD died (2010–2019).
  - *Against:* Lazy Prices is dead on the S&P 100 (t = −0.5), and LLM-news edge is small-cap and decaying.
  - *Expectation:* a large-cap 5-day IC of ~0.01–0.02.
- **Questionnaire:** six extraction-only questions — E1 raise (+), E2 lower (−), E3 forward momentum (+), E4 tone
  beyond results (−), E5 one-off flattered (−), M1 adverse (−). Five signals (`jev-comp-63` primary), deflated at
  n = 8.
- **Math audit:** 5 findings.
  - *M-1, fixed before the freeze:* yes/no answers now map to evidence p instead of 2p − 1.
  - *M-2 and M-3:* the power claims are now measured rather than assumed.
  - *M-4 and M-5:* the DST-aware ET rule and pinning every leg to the funnel's own functions, both written into the
    pre-registration.
- **Instrument check** (no returns read): Run 2 passed. M2 (unplanned CEO/CFO exit) was dropped by the pre-set rule.
  The quality block takes the in-window minimum. Every change is logged in pre-registration §8 with its exact
  timing.
- **Measured power** (seasonal earnings timing, frozen funnel, n = 8; clean window = 411 NYSE days): pre-registration
  §5. The primary 63-day hold detects a 5-day IC of 0.010 in 88% of screening batches, with 62% clean-window power.
  At 0.020 both are 100%. The 21-day hold needs ~0.02, because 41% of its days fall under the 50-name floor.
- **Phase 2 wiring obligations** (declared in `phase1`, not yet read by code): `universe`, `screening_window`,
  `construction.*`, `p2_placebo.*`, `p3_baseline.*`.

**Not in v1:**
- ATL FinSearch news: its history starts around 2026-07 and it needs a token. It could be a forward-only add-on.
- Low-confidence escalation to a generative System Two model: that adds a trial and a contamination vector.
- Jev scores as a Crucible mining substrate: that needs B8 first.
