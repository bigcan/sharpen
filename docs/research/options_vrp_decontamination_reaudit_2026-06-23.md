# Options-VRP Sleeve — De-Contamination Re-Audit (Tier-2)

**Date:** 2026-06-23 · **Session:** S553-cont-70
**Audit object:** the crypto options-VRP short-vol sleeve, wired **flag-OFF** into the paper executor (`sharpen/paper/portfolio_executor.py`, `configs/live_multi_sleeve_paper.yaml` `sleeves.vrp.enabled=false`). Two data paths kept distinct: **PATH A** the DVOL-synthetic core (`results/options_vrp/verdict.json`, the shipped object) and **PATH B** the real-chain skew falsification (`skew_verdict.json` + `instrument_ab_verdict.json`).
**Why this re-audit:** the deploy-gating real-chain **0.61** anchor — the "only real-price corroboration" the 2026-06-11 Tier-2 audit leaned on, and the literal floor of the deployed "honest band 0.6–0.9" — was **USDC-linear data contamination**. De-contaminated, the real-chain edge does not survive (straddle **−0.25**, strangle **+0.05**, instrument A/B **NO-GO across all 8 BTC/ETH cells**).
**Method / provenance:** the `options_vrp_decontamination_reaudit` workflow (8 pillars × finder+skeptic + synthesis) was launched but **terminated on the account monthly spend limit** before any finder returned (1.09M subagent tokens in 320s, `report:null`) — the same failure mode that killed the 2026-06-11 audit. Completed **inline on Opus 4.8** by direct code/data verification: every headline number was independently reproduced read-only (`C:\tmp\vrp_reaudit_repro.py`), and the contamination-completeness pillar was verified with a dedicated probe over the full 63-month cached chain (`C:\tmp\vrp_v1_contamination_probe.py`).

---

## Executive Summary

**DEPLOY VERDICT: BLOCKED — paper-gate clearance RESCINDED. The sleeve STAYS FLAG-OFF.** Revival is gated on **paid-data re-validation** (a true 21d-roll real-chain reconciliation on the daily Tardis chain) clearing a pre-registered bar; **absent that funded re-validation, RETIRE.** No clean real-execution evidence currently supports deployment, and the one real-execution test we have is NO-GO.

**The dominant cross-cutting fault is corroboration-collapse.** The 2026-06-11 audit found "no S1" and cleared the sleeve for a paper sleeve in large part because a *second, independent* number — the real-chain monthly straddle at **0.61** — appeared to corroborate the DVOL-synthetic 1.06/0.77 and gave the honest band a real-price floor. That second number was the **same class of error** (a mixed-unit / duplicate-instrument data bug), not independent confirmation. When the corroborating number is de-contaminated it does not merely weaken — it **inverts** to −0.25 and drags every tradeable structure to NO-GO. A headline justified by a second figure that turns out to be the same mistake is the exact silent-failure class a Tier-2 exists to catch, and the prior audit missed it because it read `skew_verdict.json`'s number at face value rather than auditing the chain symbols underneath it.

**What is *sound* (verified this session):**
- **The fix is complete and correct.** Over the full 134,590-row / 63-month cached chain, the `INVERSE_ONLY` filter (`~symbol.str.contains("_")`, `options_vrp_skew_falsification.py:147,170-171`) separates Deribit inverse (coin) from USDC-linear cleanly: 89,492 inverse rows (bases BTC/ETH/SOL, none containing `_`) vs 45,098 linear rows (**all** `_USDC`), and **0** BTC rows carry a USD price without an underscore — i.e. zero filter-evading linear rows. *(V1-01, S4)*
- **The contamination is exactly characterised.** The ATM short-strike argmin flips pre→post-fix in **exactly 5 months** — 2025-12, 2026-02, 2026-03, 2026-04, 2026-06 — each time the pre-fix pick is a `BTC_USDC-…` row priced ~$2,360–4,580 (USD) read as a coin premium ~0.034–0.056, mis-sizing `n` by ~10⁵ and booking the spurious ~+5%/cycle. *(V1-02, confirmed)*
- **PATH A is structurally immune to this bug.** `deribit_options_loader` ingests only `get_volatility_index_data` (DVOL) + `get_tradingview_chart_data` (perp OHLC) + funding (`deribit_options_loader.py:134,336`); it returns no per-option-symbol rows and no `USDC` reference (`RawOptionsData.perp[BTC]`=OHLCV, `dvol[BTC]`=index series). The mixed-unit-per-symbol contamination cannot occur there. *(V1-04, S4)*
- **The signed-leg refactor preserved accounting.** With the filter OFF the refactored signed engine reproduces the committed contaminated anchor **0.6115 / 1.0046 to 4 dp** — that *is* the parity proof that signed multi-leg accounting reduces exactly to the prior unsigned `select_legs` for all-short structures; the +27.4pp swing is purely de-contamination. `leg_vega`/`leg_gamma` additivity is asserted at import (`options_pricing.py:168-179`). *(V3-01, S4)*

**The problem is what the clean data now says, not a remaining code defect.** On real Deribit prices, at the only cadence free data supports (monthly sell-and-hold-to-expiry, 63 cycles), short-vol on BTC/ETH does not clear any gate — and the **contaminated cycles were all in the recent OOS window** (2025-12…2026-06), so the contamination specifically faked the recent-period performance that made the sleeve look alive. The DVOL-synthetic 0.77/1.06 (PATH A) is **untouched and still reproduces**, but it has now **lost its only real-price corroboration**, and the same BTC short straddle reads **+1.06 synthetic vs −0.25 real** — a 1.32-Sharpe gap that free data cannot resolve.

**No S1 *code* defect remains.** The S1-equivalent is at the verdict/decision layer: **the paper-gate clearance is no longer supported by clean evidence.** Treat the sleeve as a research candidate blocked on paid real-execution data, not a paper-ready sleeve.

---

## Scope Answers

### (1) Contamination completeness — is the fix complete, and are there other vectors?

**Fix: COMPLETE and CORRECT** (V1, probe-verified). Clean underscore separation across all 63 months; the only linear quote token is `USDC`; zero inverse symbols contain `_`; zero filter-evading linear BTC rows. The 5 contaminated months reproduce exactly. The filter keys on `_`, which both Deribit `BTC_USDC` and `BTC_USDT` linear conventions carry, so it generalises beyond the USDC seen here.

**Other vectors in PATH B: none found.** In the inverse-only set (89,492 rows): 0 rows with bid≤0 / ask≤0 / mark_iv≤0, **0 crossed quotes** (bid>ask), 0 NaN deltas; the 20,258 NaN-bid rows are excluded by the existing `bid_price>0` filter in `_pick_expiry` (`options_vrp_skew_falsification.py:128`). The strike/delta argmins therefore operate on a clean inverse set.

**Other vectors in PATH A: structurally impossible** for this bug class (no per-symbol option rows ingested — V1-04). **Residual caveat:** "immune to *this* contamination" is not "audited clean end-to-end" — PATH A's DVOL-proxy and daily-MTM remain modelling choices (carried over from the 2026-06-11 audit, still valid), they are simply not affected by the USDC mix.

### (2) PATH-A soundness & honest-band re-derivation

PATH A reproduces (`verdict.json` portfolio 0.7703 / BTC 1.0609, GO) and was never touched by the bug. **But its deploy justification is gone.** The 2026-06-11 honest band `[0.6, 0.9]` had its **floor = the contaminated real-chain anchor 0.61** (`verdict.json:313`, written by `options_vrp_falsification.py:481`). With that removed, the only things anchoring PATH A are its own in-sample, multiple-comparison-winning statistics:

| PATH A metric (clean, from `verdict.json`) | Portfolio | BTC |
|---|---|---|
| Net Sharpe (best-of-≥16-config grid) | 0.77 | 1.06 |
| Deflated Sharpe DSR (N=16 / N=28) | **0.37 / 0.27** | 0.61 / 0.51 |
| Block-bootstrap P(SR<0.5) | **0.29** | 0.12 |
| Real-execution corroboration (PATH B, clean) | **−0.55 (NO-GO)** | **−0.25 (NO-GO)** |

**Re-derived honest expectation:** there is **no real-execution-validated edge**. Any deployable number is bounded *above* by the multiplicity-deflated DSR (~0.3 portfolio / ~0.5 BTC) and contradicted *below* by the only real-execution test (negative across all structures). The defensible statement is **"deflated ~0.3–0.5, unconfirmed by real execution, real-chain NO-GO"** — explicitly **not** 0.6–0.9. The headline 1.06 is a best-of-2-asset, best-of-grid figure on a synthetic proxy.

### (3) Stale 0.61 citations to correct

| Location | Current (stale) | Should say |
|---|---|---|
| `configs/options_vol_harvest.gates.yaml:10-15` (rl_beats_linear comment) | "HONEST deploy-sizing band ~0.6-0.9 (real-chain monthly anchor 0.61; multiplicity-deflated ~0.5-0.65)" | The real-chain anchor was USDC-linear contamination; clean real-chain is NO-GO. No real-execution-validated band; PATH A deflated DSR ~0.3–0.5, unconfirmed. Sleeve BLOCKED pending paid-data re-validation. |
| `results/options_vrp/verdict.json:313,321` + writer `scripts/research/options_vrp_falsification.py:481,486-491` | `real_chain_anchor_net_sharpe: 0.61` + honest_band note quoting it | Set anchor to the clean value (−0.25) or null with a `contaminated_anchor_voided` flag; rewrite the note to "real-chain NO-GO; no real-execution corroboration." |
| memory `project_options_vrp_paper_gate_cleared_s553` | "Honest band 0.6–0.9 (deflated 0.37–0.61, real-chain anchor 0.61)"; "PAPER GATE CLEARED unconditional" | Clearance RESCINDED; anchor contaminated; clean real-chain NO-GO. |
| memory `project_options_vrp_paper_sleeve_wired_s553` | gate-cleared sleeve wired flag-off | Wiring stands (flag-off, byte-identical when off); the gate it cleared is rescinded; enabling now requires paid-data re-validation, not just a combined-book audit. |
| `docs/research/options_vrp_paper_promotion_2026-06-22.md:26,76` | "real-chain monthly anchor 0.61" | Same correction; add a void banner pointing here. |
| `.agent/memory/core.md:47,49` (S553-cont-36/37 banners) | "strangle 1.00 > straddle 0.61", "0.92 combined" | Historical snapshots — correct via a new boot banner, not by rewriting history. |
| `docs/research/options_vrp_linear_core_deep_lifecycle_audit_2026-06-11.md` (V1-03, V2-09, V8) | "only real-price anchor = 0.61", "strangle 0.92 > straddle 0.61" | Add a prominent VOID banner at top pointing to this re-audit; its PATH-A reproduction/BS/margin/DSR work remains valid. |

### (4) The decision

**STAY GATED (flag-OFF) + RESCIND the paper-gate clearance; revival gated on paid-data RE-VALIDATION, else RETIRE.**

- **Not "stay gated as paper-ready":** the cleared status rested partly on a contaminated number; the clean real-execution test is NO-GO. Carrying it as "paper-ready / honest band 0.6–0.9" would repeat the prior error.
- **Not outright "retire" today:** PATH A is methodologically *different* from PATH B (21d-roll daily-hedge over 1892 daily bars vs monthly hold-to-expiry over 63 cycles, `bootstrap_sortino_ci` spanning [−1.29, +0.89] — too few cycles to reject zero), so PATH B's −0.25 removes the corroboration and adds contrary-but-noisy evidence; it does not *prove* PATH A's specific cadence has no edge. That question is unanswerable on free data (free Tardis = monthly snapshots only).
- **Revival path (pre-register before spending):** buy the daily Tardis chain (~$700–2,700), run the **true 21d-roll daily-hedged real-chain** reconciliation, and require **net Sharpe ≥ the PATH-A multiplicity-deflated floor (~0.5) AND > 0 out-of-sample on inverse-only real prices AND the diversification gate on realised returns**. If the operator will not fund this, the default is **RETIRE** — by the project's own "premium must be capturable net of real microstructure" doctrine, the free real-execution evidence already points to failure, and this joins the campaign's "structure without capture" outcomes.

---

## Per-Pillar Findings

Severity: **S1** = live loss / invalidates verdict · **S2** = materially weakens robustness · **S3** = rigor/hygiene · **S4** = minor or POSITIVE. Effort: **Now** (config/test/doc) · **Next** (re-run / paid data) · **Research**.

### V1 — Contamination completeness & data-substrate integrity *(the pillar)*

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V1-01 | S4 ✅ | `INVERSE_ONLY` fix complete/correct: clean `_` separation over 63 months (89,492 inverse vs 45,098 `_USDC`; 0 filter-evading BTC rows) | probe over `data/processed/deribit_chain/*.parquet`; `options_vrp_skew_falsification.py:147,170-171` | Lock the symbol-taxonomy check into `tardis_options_chain_loader` as a DATA-CLEAN assert | Now |
| V1-02 | S2→S4 | Contamination exactly characterised: ATM argmin flips in **5 months** (2025-12…2026-06), each a USD-priced `BTC_USDC` row (~$2.4–4.6k) mis-read as coin | probe Q2; reproduces the +27.4pp swing | Already fixed; add a regression test asserting inverse-only selection per month | Now |
| V1-03 | **S2** | The contaminated cycles are **all in the recent OOS window** — the contamination faked recent-period survival, not ancient history | probe Q2 (all 5 flips ≥ 2025-12); clean `recent_12m` straddle −0.11 / strangle −0.80 (`skew_verdict.json:23,36`) | Note in the verdict that the recent-12m anchor was the most contaminated | Now |
| V1-04 | S4 ✅ | PATH A structurally immune: loader serves only DVOL + perp + funding, no per-symbol rows, no `USDC` | `deribit_options_loader.py:134,336`; `RawOptionsData` columns | Document the path-separation explicitly in the arch doc | Now |
| V1-05 | S3 | No other PATH-B garbage vector: 0 crossed/zero/neg quotes in the inverse set; NaN bids excluded by `bid_price>0` | probe Q3; `_pick_expiry:128` | Keep the `>0` guard; add a crossed-quote assert for future data | Next |
| V1-06 | S3 | `manifest.json` rewritten today (10:40) by a `load()`; provenance still clobbered per load (carryover V1-07 from 2026-06-11, unfixed) | `ls data/processed/deribit/manifest.json`; prior audit V1-07 | Per-fetch timestamped manifest + SHA256 (still open) | Now |

### V2 — Signal & feature causality

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V2-01 | S4 ✅ | PATH B is unconditional short-vol (sell first-of-month, hold to expiry); intrinsic settlement looks up the perp at the *actual* expiry date — a settlement, not a leak | `simulate_real_chain:298-342`, `spot_by_date.get(expd.normalize())` | None | — |
| V2-02 | S3 | Monthly cadence + `target_tenor_days=30` choice is fixed, not swept — minimal multiplicity on PATH B, but the *whole* sample is graded (no holdout) | `SkewConfig:113-121`; single config | If revived on paid data, hold out a period | Next |

### V3 — Pricing & P&L + the signed-leg refactor

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V3-01 | S4 ✅ | Refactor parity proven: filter-OFF signed engine = committed 0.6115/1.0046 to 4dp; all-short reduces exactly to prior unsigned engine | `C:\tmp\vrp_reaudit_repro.py` (2); `simulate_real_chain` docstring:225-234 | Keep the filter-toggle parity as a regression anchor | Now |
| V3-02 | S4 ✅ | `leg_vega`/`leg_gamma` correct: call==put at r=0, sum to straddle vega/gamma, 0 at expiry — asserted at import | `options_pricing.py:134-179` (bs_self_test) | None | — |
| V3-03 | S3 | Signed iron-structure accounting (wings net against shorts; `n` off short premium gross of wings) is internally consistent but exercised only by the NO-GO A/B cells | `simulate_real_chain:256-339`; `instrument_ab_verdict.json` | If a capped-tail structure is ever revived, add a dedicated wing-settlement test | Research |

### V4 — Execution & cost realism

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V4-01 | **S2** | Cost is **not** the gap: measured real half-spread 0.59 vol-pt (BTC) / 0.67 (ETH) is *lower* than PATH A's modelled 1.0 — so PATH B's loss is methodology/path (monthly hold-to-expiry, 63 cycles), and PATH A's edge has no real-execution confirmation | `skew_verdict.json:120-123`; `verdict.json` config `option_spread_vol_pts:1.0` | The synthetic→real gap is cadence/path, resolvable only on paid daily chain | Next |
| V4-02 | S4 ✅ | Fees charged both sides; 4-leg irons pay ~2×; perp taker on rehedge; funding applied | `simulate_real_chain:262-269,320-322` | None | — |

### V5 — Margin, liquidation & capital realism

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V5-01 | S3 | Margin still unmodelled in PATH B (only `n` sizing); lower stakes now (no edge to deploy) | `simulate_real_chain:256-259` | Defer until/unless re-validated | Research |
| V5-02 | S4 ✅ | Iron structures genuinely cap the tail (iron_fly net vega 9.6k vs straddle 16k, bounded settlement) — a real capital-efficiency property, but with negative edge to protect | `instrument_ab_verdict.json` BTC iron_fly `max_net_vega_per_100k:9622` | Note: tail-capping ≠ edge | — |

### V6 — Tail risk & adversarial stress

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V6-01 | **S2** | The short-vol left tail is now the *dominant* reason not to deploy PATH A: skew −2.72, kurt 20.1, Sortino 0.96 < Sharpe 1.06, CVaR99 −1.47%/day — i.e. you are paid a thin, **unconfirmed** premium to be short a fat left tail | `verdict.json:103-109` risk_stats | Size to the left tail; do not lever; treat as candidate not sleeve | Now |
| V6-02 | S2 | There is **no tail-control variant that keeps an edge**: the iron structures truncate the tail but are NO-GO; the strangle (prior "tail fix") is +0.05 NO-GO | `instrument_ab_verdict.json` (8/8 NO-GO) | The 2026-06-11 "trade the strangle" tail control is void (see V8) | Now |

### V7 — Methodology, multiplicity, reproduction & the core tension *(decisive)*

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V7-01 | S4 ✅ | Everything reproduces bit-exactly: clean skew −0.2546/+0.0476 NO-GO; A/B 8/8 NO-GO; PATH A 0.7703/1.0609 GO; contamination 0.6115/1.0046 | `C:\tmp\vrp_reaudit_repro.py` vs on-disk | Freeze the harness as the re-audit fingerprint | Now |
| V7-02 | **S1-eq** | **Corroboration-collapse:** the deploy case rested on the real-chain 0.61 as independent confirmation; it was the same error class and inverts to −0.25 ⇒ paper-gate clearance unsupported | `skew_verdict.json` (clean) vs `verdict.json:313`; prior audit V1-03/V2-09 | Rescind clearance; correct citations; re-validate on paid data or retire | Now |
| V7-03 | **S2** | Honest band floor (0.6) was literally the contaminated anchor; clean PATH A deflates to DSR 0.37 (N=16)/0.27 (N=28), P(SR<0.5)=0.29 | `verdict.json:66-79` deflated_sharpe, bootstrap | Replace the band with the deflated/unconfirmed read (scope answer 2) | Now |
| V7-04 | NEEDS-DATA | The 1.32-Sharpe synthetic-vs-real gap cannot be resolved on free data (monthly-only); needs paid daily chain for a true 21d-roll real-chain run | free Tardis cadence; `tardis_options_chain_loader` | Paid Tardis daily chain re-validation | Next |

### V8 — External SOTA, the voided strangle recommendation & honest-band

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V8-01 | **S2** | **The 2026-06-11 top tail-control recommendation is VOID:** "trade the strangle not the straddle (real-chain strangle 0.92 > straddle 0.61)" — both were contaminated; clean strangle +0.05, NO-GO | prior audit §V8/External-Benchmark; `skew_verdict.json:31-46` | Strike the recommendation; the skew leg is not a free tail fix | Now |
| V8-02 | S3 | Crypto VRP is a documented premium, but the clean real-execution test is evidence it is **not capturable** at monthly hold-to-expiry net of real microstructure on BTC/ETH; 0.77 synthetic looks rich against ~0 real | `skew_verdict.json`; literature (carryover) | Only paid-data 21d-roll can separate "premium exists" from "we can harvest it" | Research |

---

## Prioritized Roadmap

### NOW — config / doc / memory, no re-run (do immediately)
1. **Rescind the paper-gate clearance** and keep `sleeves.vrp.enabled=false`. *(V7-02)*
2. **Correct the 0.61 citations** per scope answer (3): gates.yaml comment, `verdict.json` honest_band + writer `options_vrp_falsification.py:481`, the two memories, the promotion doc; void-banner the 2026-06-11 audit and the historical core.md banners. *(V7-02/03)*
3. **Strike the "trade the strangle" tail-control recommendation** wherever it propagated. *(V8-01)*
4. **Record the re-derived honest read** (deflated ~0.3–0.5, unconfirmed, real-chain NO-GO) in the verdict + memory. *(V7-03)*
5. **Lock the symbol-taxonomy / crossed-quote / inverse-only-selection checks** into `tardis_options_chain_loader` as DATA-CLEAN asserts + a per-month regression test. *(V1-01/02/05)*

### NEXT — requires paid data (the only revival path)
6. **Buy the daily Tardis chain and run the true 21d-roll daily-hedged real-chain reconciliation** (inverse-only), pre-registering the pass bar: net Sharpe ≥ ~0.5 AND > 0 OOS AND diversification gate PASS on realised returns. Pass ⇒ reconsider; fail or unfunded ⇒ retire. *(V4-01, V7-04)*

### RESEARCH — open
7. Whether crypto VRP is harvestable at all net of real microstructure, or is another "structure without capture." *(V8-02)*

---

## Verification Log

**Provenance:** workflow `wf_02595c0b-b8e` (`options_vrp_decontamination_reaudit`) terminated on the account monthly spend limit before any finder returned; all pillars + adjudication completed inline on Opus 4.8 via direct code/data verification. Reproduction harness `C:\tmp\vrp_reaudit_repro.py` (all paths, read-only) and contamination probe `C:\tmp\vrp_v1_contamination_probe.py` (full 63-month chain) reproduce every number above.

| Pillar | Confirmed | Needs-Data | Positives (S4) | Independent reproduction |
|--------|-----------|------------|----------------|--------------------------|
| V1 Contamination | V1-02,03,06 | — | V1-01,04,05 | full-chain symbol probe; 5 months exact |
| V2 Causality | V2-02 | — | V2-01 | settlement-date read |
| V3 Pricing/refactor | V3-03 | — | V3-01,02 | filter-toggle parity 4dp; bs_self_test |
| V4 Cost | V4-01 | — | V4-02 | measured spread 0.59 < modelled 1.0 |
| V5 Margin | V5-01 | — | V5-02 | A/B vega caps |
| V6 Tail | V6-01,02 | — | — | risk_stats; 8/8 A/B NO-GO |
| V7 Method | V7-02,03 | V7-04 | V7-01 | full reproduction harness |
| V8 SOTA | V8-01,02 | — | — | clean strangle +0.05 vs voided 0.92 |

**Counts:** Confirmed ≈ 13 · Needs-Data 1 · Positives 8. **No remaining S1 code defect;** the S1-equivalent (V7-02) is a verdict-layer corroboration-collapse: the paper-gate clearance is no longer supported by clean evidence.

**Kill-criteria status (clean):** real-chain straddle −0.25 < 0.50 ✗ · strangle +0.05 < 0.50 ✗ · instrument A/B 0/8 GO ✗ · PATH-A real-execution corroboration absent/negative ✗. **The deployable VRP edge does not survive de-contamination on real prices.** PATH A's synthetic 0.77/1.06 survives only as an unconfirmed, multiplicity-deflated candidate pending paid-data re-validation.
