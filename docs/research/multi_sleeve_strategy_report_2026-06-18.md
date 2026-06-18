# New-Strategy Campaign — Multi-Sleeve Diversified Book + Honest Frontier toward 100%/yr

> **Session:** 553-cont-53 | **Date:** 2026-06-18 | **Author:** autonomous quant-RL agent (Opus 4.8)
> **Mandate:** design a NEW RL trading strategy aiming at ~100%/yr / minimal-DD; deliver the *honest* max-achievable risk-adjusted frontier.
> **Verdict:** **CONDITIONAL-GO on the LINEAR multi-sleeve book (momentum + rates-carry) to paper; NO-GO on a NEW RL agent (not justified).** 100%/yr is **not honestly reachable**; the credible frontier is ~6–19 %/yr at 14–26 % maxDD (2–3 uncorrelated sleeves).

---

## 0. How to read this report

Every claim is tagged **[PROVEN]** (measured this session on leak-safe, net-of-cost, causal backtests, reproducible scripts) or **[SPECULATED]** (literature prior / analytical extrapolation not yet measured here). The mandate explicitly values an honest "this does not reach the target" answer with numbers — that is the outcome.

---

## 1. Strategy design + economic rationale

**Thesis.** ~500 prior sessions falsified every *single-signal* corner (single-asset directional/multiscale RL, PRISM regimes, R0 price/vol regime-gating, R1 illiquidity, naive carry). The deployable survivor is a **linear cross-asset time-series momentum (TSMOM)** book. The north-star (high return / bounded DD) is therefore gated by the **number of genuinely-uncorrelated, cost-survivable sleeves**, not by any new single signal — because portfolio Sharpe ≈ s·√N for N uncorrelated sleeves of Sharpe s, and only a higher *portfolio* Sharpe lets you lever toward high return at bounded DD.

**The new strategy = a multi-sleeve diversified book**, assembled from validated linear premia, each on an economic basis (risk-premium / behavioural), combined by **static risk parity**, vol-targeted to a chosen leverage. RL is admitted only as a *sleeve-allocation overlay* and only if it beats static OOS.

**Sleeves and their economic basis:**
| Sleeve | Basis | Status |
|---|---|---|
| **TSMOM** (18-ETF cross-asset trend) | behavioural under-reaction / CTA trend premium (MOP 2012) | **[PROVEN] deployable**, honest net Sharpe 0.39–0.55 |
| **Rates-carry** (Treasury curve carry+roll) | term-premium / roll-down | **[PROVEN] uncorrelated diversifier**, net Sharpe 0.467, corr→momentum 0.014 |
| Options-VRP (BTC short-straddle) | variance-risk premium | [PROVEN elsewhere, cont-39/41] honest band 0.6–1.1, separate substrate, short-vol tail |
| Value (5y reversal cross-asset) | mean-reversion / valuation | **[PROVEN] FALSIFIED this session** (decayed post-2011) |

---

## 2. Cheap-falsification verdicts (this session)

### 2a. Cross-asset VALUE sleeve — **NO-GO** [PROVEN]
Pre-registered spec `value_falsification_spec_2026-06-18.md` (committed before data, gates locked). Price-only AMP "value" leg = 5y→1y long-horizon reversal, cross-sectional within asset class, on the 18-ETF momentum universe, common window **2011–2026** (deliberately spanning the post-2015 decay regime). Script `scripts/research/value_falsification.py`; verdict `results/value_falsification/verdict.json`.

| Metric (net 2 bps) | Value |
|---|---|
| XS-VALUE pooled net Sharpe | **−0.364** (all 4 classes negative) |
| rank-IC vs shuffle-null95 | **−0.0589** escapes 0.0338 → real long-horizon **continuation, not reversal** |
| corr(value, momentum) | **+0.039** (NOT AMP's −0.5: no premium *and* no diversification here) |
| combined (risk-parity) Sharpe vs momentum-alone | **0.126 ≪ 0.545** (uplift −0.419); combined DD −38 % vs −23 % |
| leak tripwire | PASS (same-day −0.313 *less* negative than causal — no look-ahead flatters value) |

**Why:** value is the **2nd textbook additive factor to decay in the modern liquid-ETF regime** (after naive carry, cont-45) — equity book-value value (HML −0.86 %/yr 2008–25) is the cautionary parallel; here the price-reversal cross-asset proxy also fails because the 2010s rewarded continuation. Economically coherent, robust, and consistent with the project's own carry result.

### 2b. Sleeve-timing / RL-allocator pre-gate — **RL NOT JUSTIFIED** [PROVEN]
Project doctrine: cheap-falsify before RL. Test whether a *linear* dynamic sleeve-allocation beats **static risk-parity** OOS with a **frictionless** meta-layer (maximally generous; RL only adds cost). Script `scripts/research/sleeve_timing_falsification.py`; verdict `results/portfolio_frontier/sleeve_timing_verdict.json`.

| Strategy | full | in-sample | **OOS (2018–26)** | OOS maxDD@10vol |
|---|---|---|---|---|
| **static risk-parity (baseline)** | 0.751 | 0.879 | **0.517** | −15.1 % |
| sleeve-momentum (factor momentum) | 0.434 | 0.511 | 0.345 | −20.3 % |
| drawdown-control | 0.753 | 0.953 | 0.351 | −18.3 % |

Both dynamic challengers are **worse OOS** (uplift −0.166). **Decision: ship static; RL sleeve-allocator NOT justified** — the 4th independent "ship-linear/static" in this project (after cross-asset WF cont-34, options-VRP WF cont-39, cost-lever retry cont-43). RL-as-allocator over-trades and the gross edge is inseparable from churn; here even a *frictionless* linear timer can't beat static.

---

## 3. Honest return / DD / Sharpe / leverage / capacity frontier  [PROVEN empirical + SPECULATED extension]

Empirical 2-sleeve book = **momentum + rates-carry**, static risk-parity, common window 2006–2026. Script `scripts/research/portfolio_frontier.py`; `results/portfolio_frontier/frontier.json`.

**Combined Sharpe:** curated **0.75**; **Fable-honest 0.601** (momentum haircut 0.601→0.389, rates 0.467, corr 0.014). Rates-carry lifts the *honest* book **+0.21 Sharpe** (0.389 → 0.601) purely via zero correlation — the core actionable finding. **OOS (2018–26) combined Sharpe 0.517** (curated) confirms the diversification holds out-of-sample.

**Frontier (Fable-honest combined book, Sharpe 0.601), vol-target = leverage:**

| Vol target | Leverage | Ann. return | maxDD | Calmar |
|---|---|---|---|---|
| 7.5 % | 0.8× | 4.5 % | −10.6 % | 0.43 |
| 10 % | 1.0× | 6.0 % | −13.9 % | 0.43 |
| 15 % | 1.5× | 9.0 % | −20.3 % | 0.44 |
| **20 %** | **2.0×** | **12.0 %** | **−26.4 %** | 0.46 |
| 30 % | 3.0× | 18.0 % | −38.1 % | 0.47 |
| 50 % | 5.0× | 30.1 % | −61.1 % | 0.49 |
| 100 % | 10× | 60.1 % | −97.3 % | 0.62 |

**[SPECULATED] 3-sleeve extension (+ options-VRP @ Sharpe 0.8, corr ~0, separate BTC substrate):** risk-parity portfolio Sharpe → **~0.95 honest / ~1.08 curated**. At Sharpe 0.95 the same frontier shifts up ~1.6× (e.g. ~19 %/yr at −25 % DD).

### How far toward 100 %/yr — the honest answer
- **100 %/yr is NOT honestly reachable.** At the honest combined Sharpe 0.601 it requires **~166 % vol (16.6× leverage) → −100 % maxDD (ruin).** Even at the speculative 3-sleeve 0.95 Sharpe it needs ~105 % vol → catastrophic DD.
- **Honest sweet spot:** **~11 %/yr at −25 % maxDD** (≈19 % vol) for the 2-sleeve book; **~6 %/yr at −14 % DD** run conservatively (10 % vol); **~19 %/yr at −25 % DD** if the VRP 3rd sleeve is added.
- **Sleeves needed for 100 %/yr at a tolerable −25 % DD:** Sharpe ~5.3 → **~113 uncorrelated sleeves** of this quality (s≈0.5). There are not ~100 genuinely-uncorrelated cost-survivable premia in liquid markets → the target is structurally out of reach for an honest book. **Capacity** is high (liquid ETFs/futures, monthly cadence, turnover ~22–65/yr) — capacity is not the binding constraint; *Sharpe* is.

---

## 4. Protocol-v2 / RL training results
**None — and correctly so.** The cheap-falsification gate (§2b) stopped before any RL env/GPU run, per the mandatory "falsify cheap before you build RL" discipline. Building an RL allocator would have violated the project's own kill criterion ("RL fails to beat the linear core OOS → not justified"). The linear baseline *is* the recommended artifact; there is no RL number to report because RL was never justified to train.

## 5. Tier-2 deep-lifecycle audit
**Not run this session — and not yet required.** No capital was promoted and no deploy-gating WF/OOS verdict was *read* to gate capital. The Tier-2 `deep_strategy_audit` is the **mandatory gate before promoting the momentum+rates-carry book to paper capital** (it would re-derive the rates-carry sleeve's leak-safety end-to-end, the curve-data provenance, and the combined-book accounting — the same class of latent-leak check that caught the X2 leak). This is the explicit STOP before any capital step.

---

## 6. Recommendation

**CONDITIONAL-GO — ship the LINEAR multi-sleeve book (momentum + rates-carry) to paper; do NOT build a new RL agent.**

- **PROVEN, actionable:** add **rates-carry** as a 2nd sleeve to the momentum book. It lifts the *honest* portfolio Sharpe 0.39 → 0.60 (+0.21) via zero correlation, holds OOS (0.517, 2018–26), is cheap (turnover 4.9/yr), and is a genuine uncorrelated diversifier. This is a strictly better deployable than momentum-alone.
- **PROVEN NO-GO:** the cross-asset **value** sleeve (decayed) and a **new RL allocator** (frictionless linear timing already fails to beat static).
- **Honest ceiling:** ~6–12 %/yr at 14–26 % DD (2 sleeves); ~19 %/yr at −25 % DD if options-VRP is added as a 3rd sleeve. **100 %/yr is not honestly attainable** without ruinous leverage or ~100 uncorrelated sleeves.

### Concrete next steps (ranked)
1. **[GATE] Tier-2 deep-lifecycle audit** of the momentum+rates-carry book before any paper-capital promotion (re-derive rates-carry leak-safety + curve provenance + combined accounting). **STOP here for operator go-ahead.**
2. **Wire rates-carry into the existing paper executor** (the rung-1 cross-asset executor already shadows the momentum core; rates-carry is the documented `carry_ary`-style additive sleeve — the env slot exists). Re-run the rung-1 forward-path audit on the 2-sleeve book.
3. **[SPECULATED, future] Add genuinely-NEW uncorrelated sleeves** to raise the portfolio Sharpe — the only honest lever toward higher return. Untested-here candidates worth a cheap pre-registered falsification: **EM-FX carry** (R1 weak-go, needs EM rate data), **valuation-adjusted FX carry** (Macrosynergy), **commodity term-structure carry** (needs front+2nd futures), **trend at faster/intraday horizons on a broad liquid futures universe**. Each must clear the same net-of-cost + additivity gate.
4. **Re-open the RL sleeve-allocator ONLY when ≥4 genuinely-uncorrelated sleeves exist** (enough breadth for rotation to add value), and only behind the pre-registered beat-static-OOS gate (≥+0.05 Sharpe, no fold worse than −0.10).

---

## 7. Artifacts (reproducible)
- Research: `.agent/artifacts/cross_asset_value_sleeve_research.md`
- Pre-registered spec: `docs/research/value_falsification_spec_2026-06-18.md` (commit `dcb99a94`, before data)
- Value falsification: `scripts/research/value_falsification.py` → `results/value_falsification/verdict.json` (commit `ce494d81`)
- Frontier + sleeve-timing: `scripts/research/{portfolio_frontier,sleeve_timing_falsification}.py` → `results/portfolio_frontier/{frontier,sleeve_timing_verdict}.json` (commit `1dee8813`)

## 8. References
1. Asness, Moskowitz, Pedersen (2013), *Value and Momentum Everywhere*, J. Finance.
2. Moskowitz, Ooi, Pedersen (2012), *Time Series Momentum*, JFE.
3. Ehsani, Linnainmaa (2022), *Factor Momentum and the Momentum Factor*.
4. Project internal: `xsec_momentum` (cont-31), `carry_falsification` (cont-45), Fable verdict (2026-06-11), gmgp1-btc canary (cont-33/40), options-VRP audit (cont-41).
