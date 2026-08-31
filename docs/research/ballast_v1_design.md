# BALLAST v1 — RL-managed long-only S&P 500 core

**Status:** DESIGN (not built). Pre-registration document.
**Date:** 2026-08-14
**Objective:** a long-only portfolio of 50–100 S&P 500 names that beats SPY total return on
**risk-adjusted** terms with a **lower** maximum drawdown, net of realistic cost, measured once on
2015-2025 out-of-sample.
**Codename:** `ballast-v1` (nautical family with `tailwind-v1`; ballast = what keeps you upright).

---

## 0. Why this is not a re-proposal of a closed family

`MEMORY.md` records **liquid large-cap equity cross-section CLOSED** (momentum, XLG timing, distress
reversal, PEAD, RL DCA, LETF). That closure is real and it constrains this design, but it does not
cover this objective. The distinction is load-bearing:

| Closed work | This design |
|---|---|
| **Market-neutral** L/S alpha — must generate return from zero | **Long-only, benchmark-relative** — inherits the equity risk premium, must only *improve* on it |
| Success = positive net Sharpe from the spread | Success = higher Sharpe / lower maxDD **than SPY**, i.e. information ratio > 0 |
| Killed by: costs eat the spread; reversal leg is a survivorship mirage | Turnover is 10–30×/yr lower; no short leg, so no borrow, no crasher-buying leg |
| Signal families: short-horizon reversal, 12-1 momentum | Signal families: **low-risk, quality, value** — the premia that specifically survive a long-only constraint |

The honest counter-evidence: session cont-74 tested a **long-only tilt** (archetype A) and rejected it
as a survivorship artifact — but that test ran on the *current-constituent* panel. Its PIT counterpart
(A.1) **passed**. So the long-only cell was rejected on a data defect, not on economics. This design
re-opens it **only** with a point-in-time, delisting-complete spine from day one (§2), and pre-commits
to the delisting-injection stress (§7 G1) as the first gate, not the last.

**Expectation reset, binding.** A realistic win is SPY Sharpe ~0.55–0.65 → **0.75–0.90**, maxDD
−55% → **−38/−45%**, information ratio **0.25–0.50**, turnover 60–120%/yr. Anything reporting IR > 1.0
or maxDD < −25% on this universe is a bug or a leak, and will be treated as such.

---

## 1. Design principle: RL does allocation, not stock-picking

The project's standing doctrine is *linear core first, RL as a thin overlay behind a beat-linear
gate*, and the reason is empirical: single-asset directional RL is falsified, and the KB consult
(`cross_asset_allocator_kb_consult.md`) validated residual-over-baseline as the architecture that
works. This design applies that literally.

A 500-name action head is the wrong shape for SAC anyway — it is high-dimensional, it breaks whenever
the index reconstitutes, and it forces the agent to relearn cross-sectional ranking that a
deterministic, already-audited signal library computes exactly. So:

- **Stock selection is deterministic**, from `sharpen/signals/` (causal, T0-tripwired, audited).
- **RL controls the meta-parameters of portfolio construction** — a ~10-dimensional continuous action.
  This is a control problem RL is actually suited to: state-dependent, path-dependent, with a
  turnover/signal tradeoff no static linear rule can express.

This also makes the two things the user asked the strategy to *decide* — the number of holdings and
the rebalance period — first-class actions rather than hyperparameters (§4).

**Corollary: the agent is universe-size invariant.** Its observation and action dimensions do not
depend on N. Index churn changes which names the deterministic layer ranks; it never changes the
agent's interface. That is the design's answer to "S&P 500 stocks change from time to time" — the
problem is solved structurally, not patched.

---

## 2. Layer 0 — the data spine (this is the project's critical path)

### 2.1 Point-in-time membership — SOLVED, with a floor at 1996

Free `fja05680` S&P 500 membership history, already on disk at
`C:\tmp\sp500_pit_members.csv` (also `data/raw/equity_panel/sp500_pit_members.csv`):

- **2712 snapshots, 1996-01-02 → 2026-06-02**, 1202 ever-members.
- Load pattern already written: `scripts/research/xlg_pit_validation.py:51-110` (`load_membership`,
  `searchsorted`-based as-of membership matrix `M[t,i]`).

**This is a hard floor: 1990 is not reachable on free data.** The requested 1990 training start
requires CRSP (WRDS) or an equivalent vendor. Training start becomes **1996** on the free/Sharadar
path (§2.2), which still gives 19 years of training vs 11 of OOS.

### 2.2 Delisting-complete prices — THE BLOCKER. **P0 measured it and it is worse than estimated.**

The full 1996-2025 panel is built (`scripts/research/ballast_build_panel.py`,
`results/ballast_v1/p0_panel_report.json`). Headline: of **1202 names ever in the S&P 500 since
1996, yfinance can price 774 — 428 (35.6%) are unpriceable.**

But the aggregate understates the problem. The coverage *curve* is the real finding:

| Year | Index members | Priceable | Coverage |
|---|---|---|---|
| 1996 | 487 | 189 | **38.8%** |
| 2000 | 492 | 238 | 48.4% |
| 2005 | 496 | 275 | 55.5% |
| 2010 | 498 | 337 | 67.7% |
| 2014 | 498 | 368 | 73.9% |
| 2019 | 505 | 430 | 85.1% |
| 2025 | 503 | 493 | **97.9%** |

**Mean coverage: training 1996-2014 = 57.2%, OOS 2015-2025 = 87.9%.**

This is monotone, and that is what makes it dangerous rather than merely large:

1. **The 1996-2005 "S&P 500" in this panel is a portfolio of 30-year survivors.** At 39-55% coverage,
   the names present are largely those still listed in 2026. Any quality / low-vol / momentum result
   measured there is close to guaranteed to look good for reasons unrelated to the signal.
2. **Train and test are drawn from systematically different universes** (57% vs 88% complete). This
   is not the classic uniform survivorship bias that inflates both halves equally — it is a
   *gradient*, so a policy is fit on a much more heavily filtered world than it is scored in.
3. The missing names are not random: they are the bankruptcies and distressed deletions (SIVB, FRC,
   SBNY, Lehman, Circuit City). **For a long-only book the bias is upward** — the backtest never
   holds what went to zero. This is what killed cont-74's long-only tilt; `xlg_b2_stress.py` was
   built to expose it (injecting just SIVB+FRC cut that sleeve's OOS Sharpe from +0.69 to +0.25).

**Design consequences, adopted:**

- **Training starts 2007, not 1996.** Below ~60% coverage the panel is not a usable approximation of
  the index. 2007-2014 gives 8 years of training that still contains the GFC. The 1996-2006 block is
  retained in the panel but used only for feature warmup and for the inflation measurement below.
- **New required measurement — the coverage-matched inflation test (G1b),** promoted ahead of P3
  after the P2 result. It runs entirely inside the **training** window, so it costs nothing from the
  single-shot OOS budget. Implementation: `scripts/research/ballast_g1b_inflation.py`.

  **The obvious construction is not available, and that null is worth keeping.** The natural ladder
  — progressively require longer *survival* — cannot be built: only **18 of the panel's 774 names**
  have a price series ending anywhere in 2007-2025. yfinance retains essentially nothing that died,
  which is precisely why the 428 are missing rather than merely truncated. There is no internal
  survival gradient to exploit, and any method that assumes one will silently measure nothing.

  **What is available: an index-exit ladder on real prices.** 402 names traded in 2007-2014 and 86
  of them are no longer S&P 500 members by 2026, yet remain priceable. So the ladder requires
  membership at successively later vintages `V ∈ {2015, 2018, 2022, 2026}`; higher `V` is a stricter
  survival requirement and therefore a more biased panel. Regressing the core's Sharpe on the
  resulting coverage gives the inflation slope.

  **Direction of the bound.** Index deletion is a *milder* event than the bankruptcies and
  distressed acquisitions that make up the 428 — deletions include mergers, which are neutral to
  positive. The measured slope is therefore a **LOWER BOUND** on the true inflation, and the
  extrapolation from the measured coverage span to full coverage is reported as an estimate with its
  extrapolation ratio stated, never as a measurement.

**Verdict unchanged: BALLAST cannot produce a GO on free price data** — but it can now say *by how
much* it is wrong, which is the most valuable thing free data can buy here.

### 2.3 Fundamentals — PIT-correct connector exists; coverage starts ~2009 on free data

`sharpen/crucible/data/edgar.py` is a real, tested, PIT-correct SEC XBRL connector: it reads
`data.sec.gov/api/xbrl/companyconcept`, carries both the fiscal period `end` and the **`filed`** date,
uses `release_timestamp = filed + 1 calendar day` (C1-02, closing the after-close same-day look-ahead),
and replays amendments as revisions through `quality_gate.asof_join`. Keyless; needs `SEC_EDGAR_UA`
(present in `.env`).

**MEASURED (P1, `results/ballast_v1/p1_fundamentals_report.json`).** 774 names, 707 CIK-mapped.
Coverage — the fraction of names with any fundamental known — is:

| Year | ≤2008 | 2009 | 2010 | 2011 | 2014 | 2020 | 2025 |
|---|---|---|---|---|---|---|---|
| Coverage | **0.0%** | 13.7% | 45.7% | 64.6% | 73.1% | 81.5% | 87.7% |

So the XBRL mandate's phase-in is visible to the day, and **fundamentals are unusable before 2010
and only become broad in 2011.** Per-concept cell coverage varies a lot (`assets` 40.7%,
`gross_profit` 20.0% — many filers, banks especially, never tag `GrossProfit`), which is why the
quality sleeve blends four ratios and tolerates partial availability per name rather than requiring
all of them.

Consequence for the training window: 2007-2014 training has fundamentals for roughly its last four
years. The design already handles this — availability masks in the observation, and P2 reports the
market-only and full sleeve sets **separately** so the fundamentals' contribution is measured rather
than assumed. It does mean the quality/value sleeves are effectively a 2011+ phenomenon in this
build, and that is stated wherever their numbers appear.

Implementation note: the provider uses the bulk **`companyfacts`** endpoint (one request per filer,
~800 total) rather than the per-concept `companyconcept` endpoint that `crucible/data/edgar.py`
uses (~8k requests). PIT semantics are identical; only the payload shape differs. Responses are
pruned to the modelled tags before caching — a full payload is ~4 MB of ~500 tags, so the universe
would otherwise cost >3 GB of cache for the ~10 tags the sleeves read (measured: 144 MB after
pruning). The cache records which tags it kept, so adding a concept re-fetches rather than silently
serving an incomplete payload.

Design response: a `FundamentalProvider` interface with two backends — `edgar` (free, 2009+) and
`sharadar` (paid, 1998+, `datekey` = PIT). Every fundamental feature carries an **availability mask**;
the model is trained with fundamentals masked-off before their coverage start, so a fundamentals-free
regime is a state the agent has seen rather than a distribution shift at deployment.

### 2.4 Panel contract

Reuse `sharpen.signals.features.Panel` unchanged — `(dates, tickers, o,h,l,c,v, active, adv,
sector_id, meta)`. `active[t,i]` = PIT member **and** priced **and** passes a liquidity floor. All
OHLCV passes `scripts/clean_ohlcv.py` (DATA-CLEAN). `meta.survivorship_free` and
`meta.unpriceable_dropped` are stamped and **printed in every result artifact** — no result is quoted
without them.

---

## 3. Layer 1 — deterministic per-name scoring (the linear core)

Six sleeves, each computed causally, then winsorized → cross-sectional z-score → **GICS
sector-neutralized** → size-neutralized. All reuse `sharpen/signals/features.py` primitives.

| # | Sleeve | Inputs | Data | Rationale |
|---|---|---|---|---|
| S1 | **Low-risk** | 252d realized vol, 60d downside vol, market beta, beta stability | market | The one large-cap long-only premium with the strongest long-horizon evidence; directly serves the "lower risk" objective |
| S2 | **Quality** | gross profitability (GP/assets), ROE, ROE stability, accruals, leverage, net issuance | fundamental | Survives long-only; low turnover (quarterly refresh) |
| S3 | **Value** | earnings yield, FCF yield, B/P, EV/EBITDA — all sector-relative | fundamental | Diversifies S2; sector-neutral form avoids the "cheap sectors" trap |
| S4 | **Momentum 12-1** | 12m return skipping last month, path-smoothness (Frog-in-the-Pan) | market | Included with eyes open: cont-78 measured it at ≈cost in a **L/S** form. Long-only, sector-neutral, blended at low weight it is a diversifier, not the engine |
| S5 | **Trend / participation** | price vs 200d MA, 50/200 slope, distance-from-52w-high | market | Defensive: reduces exposure to names already breaking down |
| S6 | **Liquidity/size within index** | ADV rank, turnover ratio, market cap rank | market | Capacity control + the small-within-large tilt |

**Baseline (frozen linear core) = equal-weight blend of available sleeves → top-K by score, K=75,
inverse-vol weighted, monthly rebalance, 5% name cap, ±5% sector deviation vs index.**
This is the object the RL must beat (§7 G3) and the object the RL is a residual over (§4).

Sleeves that fail the harness's own T0 causality tripwire or T1 breadth floor are dropped before any
RL work. That check costs an afternoon and can end the project cheaply.

---

## 4. Layer 2 — the RL meta-allocator

### Observation (~90–130 dims, N-invariant)

| Block | Contents |
|---|---|
| **Sleeve state** (6 × ~5) | trailing 63d/252d cross-sectional IC per sleeve, IC dispersion, score dispersion, sleeve autocorrelation, availability mask |
| **Market state** (~15) | index trend (multi-horizon), realized vol + vol-of-vol, drawdown from peak, breadth (% above 200d), term spread, credit spread (via `treasury_curve_loader.py` + ETF proxies), cross-sectional return dispersion |
| **Portfolio state** (~12) | current K, active share, tracking error (63d), turnover used vs budget, days since last rebalance, sector deviation L1, weight concentration (HHI), current cash/defensive weight |
| **Baseline block** (~8) | the frozen linear core's current target — **ADR-3, residual learning**: the agent sees what the baseline would do, so a null action reproduces the core and any gain is measurably incremental |

EMA-Z normalization, **reset at every split boundary (LEAK-1)**.

### Action — `Box(-1, 1, (10,))`

| Index | Meaning | Mapping |
|---|---|---|
| `a[0:6]` | sleeve blend weights | softmax over 6, temperature-annealed |
| `a[6]` | **number of holdings K** | → `[50, 100]`, quantized to 5, hysteresis band to prevent churn |
| `a[7]` | **rebalance trigger** | → no-trade band `[0, 1]` on the score-drift statistic; the *realized* rebalance period is an emergent output, reported, not assumed |
| `a[8]` | concentration | equal-weight ↔ score-tilted, interpolation parameter |
| `a[9]` | defensive tilt | 0–20% into cash/lowest-beta decile — the drawdown lever |

Both user requirements — "determine the optimal number of stocks" and "determine the optimal
rebalance period" — are `a[6]` and `a[7]`. They are **learned, state-dependent, and reported as
distributions over the OOS period**, not tuned constants.

### Portfolio construction layer (deterministic, auditable, between action and env)

blended score → drop non-`active` → sector-capped top-K → weights `w ∝ score_tilt / vol`, then:
`w_i ≤ 5%`, sector deviation ≤ ±5% vs index, per-name trade ≤ 10% of 20d ADV, `Σw ≤ 1`, `w ≥ 0`
(**long-only, no leverage**). Every constraint is a config value in `configs/ballast_v1.yaml`.

### Reward — benchmark-relative, drawdown-asymmetric

```
r_t = (log R_port,t − log R_spx,t)                 # active return: the objective, literally
    − λ · turnover_cost_t                          # realized, from the cost model — not a proxy
    − μ · max(0, TE_63d − TE_target)²              # keeps it a core holding, not a sector bet
    − ν · max(0, DD_t − DD_spx,t)                  # asymmetric: only penalize being WORSE than SPY
```

Wrapped in the existing `DSRCalculator` (`sharpen/envs/dsr.py`) applied to the **active** return
series, so the shaped objective is a differential *information* ratio. λ, μ, ν are HPO'd once and then
**locked** (BUG-01). `hindsight_weight = 0.0` in every backtest config (BUG-03).

The `ν` term is the mechanism for "lower risk": the agent is rewarded for tracking on the way up and
diverging on the way down, which is exactly what a low-vol/quality tilt plus a defensive lever can
deliver and what a return-only reward would never find.

### Env

New class `sharpen/envs/sp500_core_allocator_env.py`, generalizing `MultiAssetAllocatorEnv`
(reuse its fixed-entry-notional PnL, cost stack, gross-cap — SHORT-ACCT-safe paths kept verbatim; the
short leg is simply unreachable at `w ≥ 0`). Raw numpy dict returns preserved. **Do not modify
`MultiAssetAllocatorEnv`** — TAILWIND depends on it.

**Keystone test (blocking):** a zero-RL policy — action fixed at the baseline encoding — must
reproduce the frozen linear core's backtest Sharpe to within ±0.05. Until that passes, no RL number
means anything. This is the same keystone that made the TAILWIND allocator trustworthy.

---

## 5. Layer 3 — the ensemble

Seed-only ensembles average away variance but not bias. Diversity here is **by construction**:

| Axis | Variants |
|---|---|
| Reward | DSR-on-active · Sortino-on-active · CVaR₅-penalized |
| Algorithm | SAC · TQC (distributional; recommended by the KB consult, deferred then — this is its use case) |
| Training block | 5 block-bootstrap resamples of the training window |
| Seed | 3 per cell |

3 × 2 × 5 × 3 = 90 candidate members → prune to the **top 12 by validation information ratio**
(2011-2014), a cut made *before* OOS is touched and recorded in the manifest.

**Aggregation — average the target portfolios, not the actions.** Each member emits a target weight
vector `w^(m)`; the book holds `w̄ = normalize(mean_m w^(m))`. Averaging in weight space is stable
(nonlinear action→weight mapping makes action-averaging incoherent) and keeps every constraint in §4
satisfiable by re-projection.

**Disagreement gate — the second risk mechanism.** Let `D_t = mean pairwise L1 distance` between
member target weights. When `D_t` exceeds its trailing 90th percentile, shrink toward the frozen
linear core: `w = (1−κ)·w̄ + κ·w_core`, `κ = f(D_t)`. Rationale: member disagreement is a measurable
proxy for state-space regions the ensemble has not learned; falling back to the audited core there is
strictly safer than averaging confused policies. `κ`'s schedule is a gate config value, and the
ablation "ensemble with vs without the disagreement gate" is a **required** reported result — if it
does not help, it ships off.

---

## 6. Training / evaluation protocol

Protocol v2, six stages, one stage = one WandB run = one decision artifact.
`python scripts/validate_config.py --config configs/ballast_v1.yaml --stage <stage>` must exit 0
before any launch. The validator will need a `ballast` env-type branch (the same env-type-gated
extension pattern used for `multi_asset_allocator`).

| Split | Window | Use |
|---|---|---|
| **Train** | 1996-01 → 2010-12 (1990-01 if vendor data is bought) | HPO + policy training |
| **Validation** | 2011-01 → 2014-12 | member pruning, gate calibration, all model selection |
| **OOS** | **2015-01 → 2025-12** | **touched exactly once**, single-shot, no re-spins |

Walk-forward (5 expanding folds) runs **inside** train+val only. The OOS decision is pre-registered:
the gates in §7 are frozen in `configs/ballast_v1.gates.yaml` and hashed before the OOS run.

### Benchmarks

| Benchmark | Why |
|---|---|
| **SPY total return** | the user's stated benchmark — primary |
| **RSP equal-weight S&P** | diagnostic. A 75-name book is closer to EW than to cap-weight; if BALLAST beats SPY but not RSP, the "alpha" is an equal-weighting tilt and must be reported as such |
| **Frozen linear core** | the beat-linear gate (G3) |
| **60/40 SPY/IEF** | risk-adjusted reference for the "lower risk" claim |

### Costs

5 bps one-way commission + half-spread by liquidity bucket + square-root market impact on ADV
participation, via the model in `scripts/research/xlg_sleeve_realistic_cost.py`. Reported at **$10M /
$100M / $1B** book sizes. Steady-state fees from step 0 (`decision_steady_state_fees_pattern`).

---

## 7. Gates — `configs/ballast_v1.gates.yaml` (never hardcoded)

| ID | Gate | Threshold |
|---|---|---|
| **G1** | **Delisting-injection stress** (reuse `xlg_b2_stress.py` harness) | verdict must not flip when the measured-missing delisting cohort is injected at realistic terminal losses. **Run first.** |
| G2 | Beats benchmark on **both** axes | OOS net Sharpe > SPY Sharpe **AND** OOS maxDD < SPY maxDD. Both required — one alone is not the objective |
| G3 | Beats the linear core | OOS information-ratio uplift ≥ 0.10 vs frozen core, **else ship the linear core** |
| G4 | Deflation | DSR on the active-return series, `n_trials` = the honest full search count (sleeves × HPO trials × ensemble cells) ≥ 0.95 |
| G5 | Capacity | net Sharpe at $100M ≥ 0.80 × net Sharpe at $10M |
| G6 | Factor attribution | FF5+UMD regression on active returns reported. Alpha t-stat is **disclosed, not gated** — if the outperformance is entirely low-vol + quality loadings, that is an honest finding about a cheap factor portfolio, and it gets said |
| G7 | Turnover realism | realized turnover ≤ 250%/yr; realized average holding period reported |

**Pre-registered kill criteria (no re-spins, no reframing):**
- G1 fails ⇒ **NO-GO, project closed**, logged to the ledger as a data verdict.
- OOS IR < 0.20 ⇒ NO-GO.
- OOS maxDD ≥ SPY maxDD ⇒ NO-GO on the stated objective regardless of return.

**Tier-2 deep lifecycle audit is mandatory** before the OOS verdict is read as deploy-gating
(`.claude/workflows/deep_strategy_audit.js`, finder + skeptic per pillar). Stakes trigger it, not the
diff size. Routine `/audit` is blind to exactly the class of bug that has erased two edges in this
project.

---

## 8. Build plan

| Phase | Deliverable | Gate to exit | GPU |
|---|---|---|---|
| **P0** | PIT panel builder `sharpen/data/sp500_pit_panel.py` (membership + delisting-complete prices + DATA-CLEAN + manifest) | panel meta stamps coverage; unpriceable fraction measured and **acceptable** | no |
| **P1** | `FundamentalProvider` (`edgar` + `sharadar` backends), availability masks, as-of join | PIT negative test: no feature at `t` uses a filing with `filed + 1d > t` (LEAK-2) | no |
| **P2** | Six sleeves in `sharpen/signals/library/ballast.py` + frozen linear core backtest | T0 causality tripwire green; core beats SPY Sharpe on **validation** (if the linear core cannot, RL will not either — cheap early kill) | no |
| **P3** | `Sp500CoreAllocatorEnv` + portfolio construction layer + config + `validate_config` branch | **keystone baseline-parity ±0.05**; negative tests for LEAK-1/LEAK-2; long-only invariant test | no |
| **P4** | SAC/TQC integration, HPO, walk-forward | Protocol-v2 manifests PASS at each stage | yes |
| **P5** | Ensemble build, pruning, disagreement gate + ablation | validation IR beats single best member | yes |
| **P6** | **Single-shot OOS 2015-2025**, benchmark comparison, G1-G7, Tier-2 audit | verdict | no |

**P2 is the real decision point.** If the deterministic six-sleeve core does not beat SPY risk-adjusted
on validation with fundamentals masked appropriately, the RL layer is not worth building and the
project should stop there. It is the cheapest possible way to find out.

### P2 methodology trap, found and fixed on the first run (keep this)

The first P2 pass scored `sleeve_quality` at Sharpe **0.961** with a max drawdown of only **−18.9%**
over 2007-2014 — apparently the best sleeve by a wide margin. It was an artifact of the coverage
curve in §2.3: fundamentals are 0% available before 2009, so the quality book **held nothing through
the GFC**, and the backtest recorded that flat stretch as a shallow drawdown it never earned. The
tell was `avg_holdings = 50.5` against a target `k = 75` — an average across ~3 years at zero
holdings and ~5 years at 75.

Two permanent guards, now in the runner:

1. **`exposure_frac` is reported for every strategy** — the fraction of days the book actually held
   anything. Anything below ~0.99 means the risk metrics are partly a record of sitting in cash, and
   the drawdown in particular is not comparable to a fully-invested benchmark.
2. **Fundamental-dependent strategies are scored from 2011** (where coverage reaches 64.6%),
   **against a benchmark measured over the same window**, with a market-only run on that same window
   as a like-for-like control. Comparing a 2011-2014 book to a 2007-2014 benchmark would swap one
   artifact for another.

This generalizes: on any panel with a data-availability ramp, *a strategy that cannot trade looks
low-risk*. The guard is to report exposure alongside every risk number, never a risk number alone.

### P2 RESULT — CONDITIONAL PASS. Read it as "not yet falsified", not as evidence the core works.

`results/ballast_v1/p2_linear_core.json`, `exposure_frac` ≥ 0.999 throughout (cash artifact gone).
Costs 5 bps one-way, k=75, monthly rebalance, benchmark SPY total return over the matching window.

| Strategy | Window | Sharpe | vs SPY | maxDD | vs SPY | IR | turnover 1-way |
|---|---|---|---|---|---|---|---|
| core_market_only | 2007-2014 | 0.519 | 0.415 | −52.3% | −55.2% | 0.228 | 2.53 |
| core_full (6 sleeves) | 2011-2014 | 1.165 | 1.014 | −18.1% | −18.6% | 0.645 | 2.65 |
| **core_market_only** | **2011-2014** | **1.220** | 1.014 | −18.5% | −18.6% | **0.847** | 2.62 |

Per-sleeve information ratios (2007-2014, or 2011-2014 for fundamentals):
liquidity **0.817** · quality 0.652 · momentum 0.257 · trend 0.139 · low_risk 0.039 · value 0.022.

All three variants clear G2 on both legs. Three findings matter more than the pass:

1. **Fundamentals are DILUTIVE.** On the identical 2011-2014 window the 4-sleeve market-only core
   (1.220) beats the 6-sleeve full core (1.165). The like-for-like control is what exposes this;
   without it, `core_full`'s 1.165-vs-1.014 would have read as "the fundamentals earned their keep".
   Quality does carry real selection information (IR 0.652) but not enough to pay for value, whose
   IR is **0.022** — value's high Sharpe is market exposure, not selection. Same for low_risk
   (IR 0.039). **This does not remove fundamentals from the design** (they are in the observation
   and the RL layer may find state-dependent uses for them that a fixed equal blend cannot), but the
   static-blend evidence says they do not belong at equal weight.
2. **The strongest sleeve is the one most exposed to the survivorship hole.** `liquidity` is a
   small-within-large tilt at IR 0.817, and the 428 missing names are disproportionately those that
   were small *at the moment they failed*. This is not a reason to drop the sleeve; it is the reason
   G1b must run before any GPU spend.
3. **The margin is thin where it counts.** On the full 2007-2014 window the uplift is **+0.104**
   against a gate threshold of +0.10. That is inside any plausible survivorship inflation. The
   2011-2014 numbers look far better but are four years of a strong bull market (SPY Sharpe 1.014)
   and should not be quoted as the headline.

**Consequent change to the plan: G1b is promoted ahead of P3.** It runs entirely inside the training
window, so it costs nothing from the OOS budget. If inflation accounts for more than ~0.10 Sharpe,
the entire measured margin is an artifact and BALLAST should close before any RL work rather than
after it.

---

## 8b. G1b RESULT — the P2 margin does not survive. **Free-data BALLAST is a NO-GO.**

`results/ballast_v1/g1b_inflation.json`. Index-exit ladder, 2007-2014, market-only core, k=75,
5 bps. Every rung is real prices.

| Universe | Names | Coverage | Core Sharpe | Core IR | Core maxDD |
|---|---|---|---|---|---|
| member_at_2026 (most biased) | 504 | 0.554 | **0.615** | 0.504 | −47.6% |
| member_at_2022 | 461 | 0.568 | 0.606 | 0.473 | −47.0% |
| member_at_2018 | 428 | 0.612 | 0.601 | 0.458 | −47.0% |
| member_at_2015 | 390 | 0.641 | 0.559 | 0.337 | −49.7% |
| baseline (all priceable) | 774 | 0.672 | **0.519** | 0.228 | −52.3% |

**The relationship is monotone across every rung: as coverage improves, performance falls.**
Sharpe slope **−0.764 per unit coverage, r = −0.936**; IR slope −2.203, r = −0.941. Drawdown
deepens too. This is the survivorship signature in its cleanest form — the more of the real universe
you admit, the worse the strategy looks.

Extrapolating from the baseline's 0.672 coverage to 1.0:

- **Implied Sharpe inflation: +0.237.**
- **The P2 margin over SPY was +0.104.** The inflation over-explains it by **2.3×**.
- Extrapolated core Sharpe at full coverage: **0.282, against SPY's 0.415** — i.e. at full coverage
  the core would *underperform* the benchmark it was built to beat. Extrapolated core IR: **−0.46**.

**And this is a lower bound.** The ladder is built from index *deletions*, which include mergers and
are milder than the bankruptcies and distressed acquisitions making up the 428 fully-missing names.
The true inflation is larger than 0.237.

### The one sleeve I was wrong about

I flagged `liquidity` (small-within-large) in §8's P2 read as the sleeve most likely inflated,
reasoning that the missing names were small when they failed. **The ladder says the opposite.**
Per-sleeve IR against coverage:

| Sleeve | IR at cov 0.554 | IR at cov 0.672 | slope | r |
|---|---|---|---|---|
| low_risk | 0.255 | 0.039 | −1.793 | **−0.997** |
| momentum | 0.490 | 0.257 | −1.875 | −0.927 |
| trend | 0.259 | 0.139 | −0.798 | −0.828 |
| **liquidity** | 0.942 | 0.817 | −0.477 | **−0.259** |

`low_risk`, `momentum` and `trend` collapse as coverage improves — they are the inflated ones, and
`low_risk` is almost perfectly explained by coverage alone (r = −0.997). `liquidity` shows no
reliable relationship (r = −0.26, non-monotone across rungs) and is the most robust of the four on
this axis.

The caveat that keeps this from being a green light for a size tilt: the ladder tests the
*index-deletion* cohort, which is still listed. It structurally cannot test the fully-missing
bankruptcy cohort — and that is precisely the cohort a small-cap tilt would have bought. So
`liquidity` is robust *on the axis that could be measured*, not robust in general.

### Verdict

The pre-registered gate `g1b_coverage_matched_inflation.max_sharpe_inflation = 0.25` returns PASS at
0.237 — **on a technicality that does not survive contact with the actual numbers.** That threshold
was fixed before P2 revealed the margin was only +0.104, so it is not the binding comparison. The
binding comparison is inflation (0.237) versus margin (0.104), and the margin loses. The gate was
mis-specified, not satisfied. It is **not** being retroactively lowered — the historical number
stands as written, with §11 recording the corrected specification for any future re-run.

**BALLAST on free data: NO-GO.** The measured edge is fully accounted for by survivorship bias. No
GPU work is justified. (§8c revises the *mechanism* behind this — pool quality, not loser-holding —
and shows the inflation estimate probably overstates the damage. The verdict stands; the confidence
in its magnitude does not.) This is the negative screen operating exactly as designed and at the intended
cost — the project was killed by measurement, before any RL was built.

---

## 8c. Exit-exposure test — the sleeves DO avoid deteriorating names. This corrects §8b's mechanism.

`scripts/research/ballast_exit_exposure.py`, `results/ballast_v1/exit_exposure.json`.

### Why this had to be run

G1b establishes that the measured margin is inflated. It does **not** establish *why*, and the two
candidate mechanisms have opposite implications for whether paid data is worth buying:

* **Loser-holding** — the sleeves buy deteriorating names and the free panel hides the damage. If
  so, complete data makes everything strictly worse and BALLAST is dead.
* **Pool quality** — the sleeves avoid deteriorating names fine, but the *selection pool* is
  pre-filtered to companies that stayed in the index for another decade, so the top-K is drawn from
  guaranteed long-run winners. If so, there is an offsetting effect complete data would reveal: a
  long-only book is scored against the **actual** index, which held Lehman/SIVB/FRC all the way
  down, and **you get no credit for avoiding a name that is not in your data**.

### Method

For every index EXIT among priceable names in 2007-2014, look back 252/126/63/21 trading days and
record where the exiting name sat in that day's cross-section. `score_percentile` is the rank of the
blended core score among active names — self-normalizing, so no matched control is needed: under
"no avoidance skill and no adverse selection" it averages 0.50. Events are split by the name's
126-day market-relative return before exit (`< −20%` ⇒ "decline", the failure proxy; the rest are
mostly mergers, which pay a premium and are *good* to hold).

### Result — 74 exit events, 33 unique names, 14 decline

| Cohort | Days before exit | Score percentile | CI95 | Held | Base rate |
|---|---|---|---|---|---|
| **decline** | 252 | **0.294** | [0.17, 0.41] | **0.000** | 0.241 |
| decline | 126 | 0.253 | [0.11, 0.39] | 0.071 | 0.236 |
| decline | 63 | 0.194 | [0.09, 0.30] | 0.000 | 0.235 |
| decline | 21 | **0.085** | [0.04, 0.13] | 0.000 | 0.233 |
| other | 252 | 0.538 | [0.37, 0.71] | 0.333 | 0.224 |
| other | 126 | 0.499 | [0.34, 0.66] | 0.235 | 0.219 |
| other | 63 | 0.462 | [0.30, 0.63] | 0.316 | 0.220 |
| other | 21 | 0.443 | [0.28, 0.61] | 0.263 | 0.219 |

Every decline-cohort CI excludes 0.50, the percentile falls monotonically as exit approaches, and
the book held **zero** of them at three of the four horizons against a ~23% base rate. The merger
cohort sits at roughly the base rate, which is the desired behaviour.

### The circularity caveat — read the h=252 row, not the h=21 row

The "decline" label is a 126-day market-relative return, and `momentum` (12-1 return) and `trend`
(price vs 200d MA) are built from almost exactly that. So **the h=21 and h=63 readings are close to
tautological** — the cohort is partly defined by the signal being tested.

The h=252 reading is the informative one: there the score date precedes the classification window
entirely, so nothing about the score can be a restatement of the label. It still shows percentile
**0.294 with zero holdings**. A full year before exit, before the measured decline even begins, the
sleeves already ranked these names in the bottom 30% and owned none of them.

Generalizes: when classifying a cohort by a variable that is also (a component of) the signal under
test, only the horizons where the score strictly precedes the classification window carry
information. Report them separately rather than averaging across horizons.

### What this changes, and what it does not

**Mechanism: pool quality, not loser-holding.** The §8b hypothesis that the sleeves were partly
buying the exit cohort is **refuted**. G1b's degradation comes from the selection pool being
pre-enriched with future winners.

**It does not overturn the NO-GO,** for one reason: **SPY is cap-weighted.** A company on its way to
zero has already shrunk to a trivial index weight by the time it fails — so the benchmark barely
suffers from the failures the strategy so cleanly avoids. Avoidance credit is large against an
equal-weight benchmark and small against SPY specifically. (The book is closer to equal-weight, so
per name held it is *more* exposed to a failure than SPY is — it simply does not hold them.)

**It does weaken the extrapolation.** G1b measured the mild deletion cohort and extended that slope
to the severe missing cohort. This test shows the strategy treats the two very differently (0.294 vs
0.538 at h=252), so adding the severe cohort to the pool is likely closer to neutral than the
deletion cohort was, and the −0.237 inflation probably **overstates** the damage.

**Standing judgement: ~30-35% that delisting-complete data flips this to a genuine GO** (revised up
from ~20% before this test). Below even odds, and n=14 decline names is thin. Recommendation
unchanged but better supported: buy one month of vendor data, rebuild 2007-2014, re-run P2 + G1b
only — days of CPU, no GPU, OOS budget intact. That resolves the exact quantity §8b and §8c
disagree about, rather than arguing it from two partial measurements.

## 9. Open decision: the data spine (operator call — involves spending)

Everything above is buildable. Only §2.2 gates whether the *result* is trustworthy.

| Option | Cost | Training start | Delisting-complete | Fundamentals | Verdict |
|---|---|---|---|---|---|
| **A. Free (yfinance + EDGAR)** | $0 | 1996 | **No — ~28% of the universe missing** | 2009+ | Backtest is an **upper bound only**; G1 will most likely fail. Usable to build and debug the whole pipeline, not to produce a verdict |
| **B. Sharadar (Nasdaq Data Link)** SEP + SF1 + TICKERS | ~$50–100/mo | 1998 | **Yes** (delisted names retained) | **1998+, PIT via `datekey`** | **Recommended.** Solves prices and fundamentals in one subscription; the 1998 start costs 2 of the requested 8 pre-2006 years |
| **C. CRSP + Compustat (WRDS)** | academic/institutional | **1990** ✓ | Yes (with delisting returns) | 1990+ | The only option that meets the literal 1990 spec. Needs an institutional affiliation |
| **D. EODHD / FMP** | ~$20–80/mo | ~1998 | Partial | varies | Cheaper, coverage less certain than Sharadar |

The `signals` harness was originally specified against Sharadar and the key has been pending since
S553-cont-71; this project is the strongest reason yet to close that purchase.

### DECIDED 2026-08-14 (operator): **Option A — free data only, upper-bound status accepted.**

No vendor purchase. This locks the following, and they are not negotiable later in the project:

1. **Training window is 1996-2014**, not 1990-2014. The free PIT membership floor is 1996-01-02.
2. **BALLAST cannot produce a GO.** Every verdict is capped at **PROMISING / upper bound**, exactly
   the convention `equity_panel_loader.py` already stamps (`survivorship_free=False` ⇒ scorecard caps
   at PROMISING). Promotion to paper or capital would require a clean re-validation on vendor data,
   and that re-validation is a separate decision, not an extension of this one.
3. **The project's value is as a NEGATIVE SCREEN.** The logic is asymmetric and it is the whole
   justification for running it: a long-only book measured on a panel that omits ~28% of the
   historical universe — disproportionately the failures — is being scored on *optimistic* data. If
   it cannot beat SPY here, it is dead for certain and we have spent no money finding out. If it
   does beat SPY here, we have learned only that the question is worth paying to answer properly.
4. **G1 (delisting injection) is re-scoped accordingly.** It cannot certify the strategy on this
   data. Its job is to measure *how much* of any outperformance the missing cohort accounts for, by
   injecting the known-unpriceable names at realistic terminal losses. A result that survives a full
   injection sweep is a materially stronger PROMISING than one that does not; a result that dies
   under injection is a NO-GO on the spot.
5. **Fundamentals are EDGAR-only, so they begin ~2009.** Sleeves S2 (quality) and S3 (value) are
   masked-off for 1996-2008 and the agent trains with the availability mask in its observation, so
   the fundamentals-absent regime is a learned state rather than a deployment-time surprise. The
   1996-2008 block trains an effectively market-data-only policy; 2009+ trains the full one. This is
   disclosed in every result table, and the P2 core is reported **both** ways (4-sleeve full-history
   and 6-sleeve 2009+) so the fundamentals' contribution is measured rather than assumed.

Build order stands: **P0 → P1 → P2**, CPU only, stopping at the P2 kill point (§8).

---

## 10. What would make me wrong

Recorded now, so it is not rationalized later:

1. **G1 fails and there is no vendor data** ⇒ the project produces no verdict. That is the most
   likely single outcome on the free path, and it is a data verdict, not a strategy verdict.
2. **The linear core (P2) already beats SPY and the RL adds nothing** ⇒ ship the linear core. Per
   doctrine this is a *success*, not a failure, and G3 is written to produce exactly that outcome.
3. **G6 shows the outperformance is 100% low-vol + quality loadings** ⇒ BALLAST is a cheap factor
   portfolio with an RL turnover controller. Still potentially worth holding, but it must be described
   that way and benchmarked against buying USMV/QUAL rather than against SPY alone.
4. **1996-2014 contains two once-in-a-generation crashes and 2015-2025 contains one** ⇒ the OOS regime
   is materially calmer than training. A defensive tilt learned on 2000 and 2008 may simply cost
   performance in the OOS. This is a real risk of the requested split and is not fixable by
   engineering; it is disclosed, and G2's dual requirement (Sharpe **and** drawdown) is what keeps it
   from being papered over.
