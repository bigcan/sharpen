# Options-VRP Linear Short-Straddle Core — Deep Lifecycle Audit (Tier-2)

> ⚠ **PARTIALLY SUPERSEDED 2026-06-23 — see `options_vrp_decontamination_reaudit_2026-06-23.md`.** This audit's real-chain claims are VOID: the "only real-price anchor = **0.61**" (V1-03/V2-09) and "real-chain strangle **0.92** > straddle 0.61, the skew leg adds edge and truncates the tail" (V8 / External-Benchmark) were **USDC-linear data contamination** in the free Tardis chain. De-contaminated, the real-chain is **NO-GO** (straddle −0.25, strangle +0.05, instrument A/B 0/8), so the "trade the strangle" tail-control recommendation is **struck** and the honest band's 0.6 floor is removed. The PATH-A (DVOL-synthetic) reproduction, BS/accounting checks, margin probe, and DSR work below remain valid — PATH A was never contaminated — but its deploy justification (real-chain corroboration) is gone, and the sleeve's paper-gate clearance is **RESCINDED**.

**Date:** 2026-06-11 · **Session:** S553-cont-41
**Audit object:** the STATIC LINEAR rolling delta-hedged short ATM straddle on BTC, built on FREE Deribit data (DVOL as IV proxy, perp chart, funding) — the `ship_linear_core` artifact from S553-cont-39. **NOT** the RL overlay (already falsified: 1/5 WF windows).
**Headline claim under audit:** `results/options_vrp/verdict.json` — net Sharpe **1.10** / PF **1.19** / max DD **5.38%** / total return **30.5%** on BTC, net of Deribit fees + funding + modelled spread; turnover-gated (14–30d roll GO, 7d NO-GO).
**Method:** Tier-2 finder + skeptic per pillar. *Provenance note:* the `options_vrp_tail_audit` workflow (8 pillars × finder+skeptic + synthesis) was launched but **terminated on a monthly spend limit** after 3 of 8 finders completed (V1, V2, V5). The remaining pillars (V3 pricing, V4 cost, **V6 tail**, V7 methodology, V8 SOTA) and all adjudication were **completed inline on Opus 4.8** by recovering the killed finders' surviving probe scripts (`C:\tmp\v{3,4,6,8}_*.py`) and re-running them, plus direct code verification. Every headline number below was independently reproduced read-only (parity max|Δeq| = **0.0**).

---

## Executive Summary

**DEPLOY VERDICT: CLEARED-FOR-PAPER-SLEEVE — conditional on the NOW bucket; NOT cleared for real capital until NEXT clears.**

No S1 was found. The edge is **real and survives adversarial tail treatment at the shipped sizing**: the verdict reproduces bit-for-bit; a severe instantaneous **−30% spot / +40 vol-pt** overnight shock caps at **23% of equity** (under the 40% kill and ~25% tail gate) with worst margin-usage **0.31** (no forced liquidation); the VRP premium dominates (sum of positive variance-spread windows +8.86 vs −1.79 negative). The strategy is **not** a frictionless artifact, not leaky (it is unconditional short-vol — no signal to leak), and not single-event-fragile.

**The dominant cross-cutting fault is over-optimistic headline framing, not a hidden breach.** The shipped 1.10 Sharpe is the most-flattering reading on three independent axes, all of which point the same way (down):

1. **Cost:** the honest fully-costed number is **1.06** — the sim never charges the perp-hedge establishment/roll-jump fee ($1,129 over 5.2y ≈ −0.039 Sharpe). *(V2-05/V4)*
2. **Statistical fragility (the killed-pillar gap, now filled):** returns are sharply left-skewed (**skew −2.73**, kurt 20.2, Sortino 0.99 < Sharpe 1.10 — Sharpe overstates). Block-bootstrap Sharpe CI95 is **[0.18, 2.17]** with **P(SR<0.5)=11%**; the **deflated Sharpe** (multiplicity-corrected for the ~16–28 graded configs) is **0.54–0.65** — the 1.10 barely clears its own corrected hurdle. *(V6)*
3. **Mark fidelity / selection:** the only real-price cross-check (Tardis monthly real-chain) is **0.61**, and the BTC 1.10 is the better of 2 assets graded on the same sample while the *pre-registered* gate was the BTC+ETH portfolio (**0.81**). The honest expectation band is **~0.6–1.1**, not a 1.10 point. *(V1-03/V1-09/V2-09/V2-10)*

   *Adjudication note:* the related worry that "DVOL sits systematically rich above tradeable ATM IV" (which would inflate the synthetic) is **REFUTED** — measured DVOL−ATM basis median **+1.5** vol-pt, mean −2.2, n=34. The synthetic-vs-real gap is **cadence + spread + monthly-hold path**, not a proxy mark bias. *(V8)*

**Top S2 themes (all NOW-fixable, none retrain-blocking):**
- **Declared-but-not-wired safeguards** — the exact silent-failure class the audit discipline exists to catch: DATA-CLEAN / `clean_ohlcv` has **zero consumers** in the options stack; the pre-registered tail gates (`cvar95_floor_pct`, `max_net_vega_per_100k`) have **no Python consumer**; gate thresholds are **hardcoded** in the script (mirroring, not reading, the yaml). All currently benign on today's clean data, but unguarded on the next `--refresh`. *(V1-06, V5-03, V1-11)*
- **Headline DD is close-to-close and ~1.6× optimistic** — intraday-trough DD reconstructed from the already-cached perp/DVOL high-low is **8.42%** vs the quoted 5.38% (worst 2025-10-10). Still well within gates, but any kill-switch keyed to "5.4%" inherits the flattery. *(V1-12, V5-04)*
- **In-sample cadence multiplicity** — the 21d-primary / "14–30d GO" conclusion was both chosen and graded on the full 2021–2026 sample; no holdout validates the cadence. *(V2-03)*

**Bottom line:** ship the paper sleeve after the NOW bucket (≈1 day, no retrain), **sized to the 0.6–0.9 honest band, not 1.10**, with the statistical-fragility caveat logged so paper out-performance is never mistaken for confirmation. Real-capital promotion additionally requires the NEXT bucket (holdout cadence grade, real-chain 21d reconciliation, a wired margin model). Options/vol remains a north-star uncorrelated-sleeve candidate (corr-to-BTC ≈ 0.0), **not** a prop-firm product.

---

## Per-Pillar Findings

Severity: **S1** = live loss / invalidates verdict · **S2** = materially weakens robustness · **S3** = rigor/hygiene · **S4** = minor or POSITIVE. Effort: **Now** (config/test/doc) · **Next** (re-run / measurement) · **Research** (open question).

### V1 — Data substrate integrity (free Deribit)

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V1-01/02 | S4 ✅ | Substrate complete (1893 bars, 0 gaps/NaN/OHLC-viol, event-coherent); headline reproduces to 4 dp | recompute vs `verdict.json:49-56`; parity 0.0 | Lock the ad-hoc checks into the loader | Now |
| V1-03 | **S2** | Headline is DVOL-synthetic; only real-price anchor (Tardis monthly) = **0.61**, 3/6 yrs+, DD 10.9% | `skew_verdict.json:12-28`; `cost_robustness.md:52` | Quote 0.6–1.1 band; re-run real-chain at 21d cadence (paid) | Next |
| V1-06 | **S2** | **DATA-CLEAN declared-but-not-wired** — zero `clean_ohlcv` consumers; `_validate` omits OHLC-invariant/gap/stale checks; funding frame never validated; issues never gate | grep `clean_ohlcv` = 0 hits; `deribit_options_loader.py:194-213,268-271,291-295` | Wire checks into `_validate`; `load()` raises on issues; fix the ✅ in arch doc | Now |
| V1-09 | **S2** | Best-of-2 asset selection in the shipped headline (BTC 1.10 vs pre-reg portfolio gate 0.81; ETH 0.40 would fail) | `verdict.json:2,27-29,49-56`; gates.yaml:12-13 | Anchor deployable expectation on portfolio/band; re-register BTC-only gate on unseen data | Next |
| V1-04 | S3 | 8h cross-series misalignment (DVOL 00:00 vs perp 08:00, both bar-open; proven by lead-lag corr −0.150/−0.112/+0.016). Causal (IV *stale* not future); docstring "exact for daily cadence" overclaims | `options_vol_features.py:13-16,31-36`; lead-lag empirics | Fix docstring; add stamp-convention + lead-lag tripwire | Now |
| V1-05 | S3 | Funding applied ~1–1.5d stale; aligning gives 1.086 vs 1.100 (−0.015 flatter) | `options_vol_features.py:50-58`; `falsification.py:178` | Shift funding to the holding window or document the +0.015 bias | Now |
| V1-07/16 | S3 | Provenance fragile: manifest clobbered each `load()` (current one says assets=BTC-only, raw-ms end_date); no file hashes; results/ + data/ gitignored, no DVC; `--refresh` overwrites graded parquets with no `.bak` | `manifest.json:8-13`; `deribit_options_loader.py:254-266,289-290`; `.gitignore:89,102` | Per-fetch timestamped manifest + SHA256 into verdict.json; `.bak` before refresh | Now |
| V1-08 | S3 | Arg-swap bug `falsification.py:144` (see V3) | — | (V3) | Now |
| V1-10 | S3 | 30d-roll row holds *through* the tenor-floor (gamma-explosive final week the 21d comment says it avoids); tau sits at 1/365 floor, also freezing theta — a modeling artifact unique to that row | `falsification.py:69-74,174,189` | Don't lean on 30d for headline support, or model expiry settlement | Now |
| V1-13 | S3 | DVOL point-in-time publication unverified at series start (ETH DVOL backfill vs live launch unknown) | `deribit_options_loader.py:57-58`; both parquets start 2021-04-01 | One-time check of Deribit DVOL launch/backfill policy | Research |
| V1-14/15 | S4 ✅ | Units/conventions correct end-to-end (DVOL/100, funding sign, ANN=365); LEAK-2 feature tripwires green (4/4) | `options_vol_features.py:28,76`; `test_options_vol_causality.py` | Credit | — |

### V2 — Signal & feature causality (LEAK-2)

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V2-02/08 | S4 ✅ | Causality genuinely clean **and the leak surface is structurally minimal**: the verdict strategy is **unconditional short-vol** — `simulate_asset` never reads `rv_*`/`iv_rv_spread`, so no RV-feature leak is even possible; no full-series normalization anywhere (LEAK-1 vacuous); decisions are pure time-cadence | `falsification.py:102,251,258-263`; `test_options_vol_causality.py` 15/15 | Add positive control to env tripwire; note "skip≥1" must be wired before any *signal-conditioned* variant | Now |
| V2-03 | **S2** | Turnover-cadence conclusion selected AND graded on the same full sample (in-sample multiplicity); "14–30d GO" overstates the grid (14d@0.10 is NO-GO, PF 1.094) | `falsification.py:418-435`; `verdict.json:194-202` | Pick cadence on 2021–23, grade 2024–26 untouched (script already parameterized); correct claim to "21–30d GO; 14d marginal" | Now |
| V2-09 | **S2** | Deploy anchor quotes synthetic 1.10 while real-chain straddle = 0.61; gates.yaml/randd anchor 1.10 with no band | `skew_verdict.json:13-14`; gates.yaml:10-13 | Quote a band; reconcile cadence (V8 already refutes the DVOL-rich mechanism) | Next |
| V2-05 | S3 | **Frictionless leg:** perp-hedge fee on *establishment* (every open) and *roll-jump* never charged — $1,129/5.2y, moves 1.100→1.061 | `falsification.py:137,139-148,168-170`; recompute | Book `perp_taker_fee·|q_new−q_prev|·S` at open + final unwind in BOTH engine copies; re-run parity + gate | Now |
| V2-06 | S3 | Arg-swap (see V3); env twin uses *correct* order → "parity by construction" holds only because the bug is inert | `falsification.py:144` vs `options_pricing.py:70` vs `options_vol_harvest_env.py:253` | (V3) | Now |
| V2-04 | S3 | Roll-phase single-draw: the shipped k=0 phase scores 1.10, *below* the 21-phase median 1.22 (range 0.83–1.46, all clear gate) — not lucky, but ±0.3 undisclosed phase variance on the number gates.yaml anchors | `falsification.py:152-159`; phase sweep | Report phase-ensemble band; trade 2–4 staggered sub-books | Now |
| V2-10 | S3 | Asset selection-after-peeking (mitigated: pre-reg gate was portfolio; BTC/ETH split independently reproduced on Tardis) | `verdict.json:27-34,93-98`; `skew_verdict.json:176-183` | Disclose 1-of-2; haircut toward 0.81 | Next |
| V2-11 | S3 | Pre-registration claimed but not commit-stamped: gates.yaml landed **1.6h after** the GO-announcing commit; verdict.json gitignored | git log `cb2a23e0` (13:03) vs `12c8b455` (14:41); `.gitignore:102` | Commit gates+spec BEFORE the verdict run going forward; retro-commit verdict.json + data hash now | Now |
| V2-12 | S4 ✅ | Adversarial scaffolding real & wired: failing 7d cells published, frictionless cost-gap surfaced, PF-XCHECK Model-B ran clean (BTC +0.112, 78% +), BS self-test at import | `falsification.py:344-368,420-435,249` | Keep; extend PF-XCHECK to net once V2-05 lands | Now |

### V3 — Pricing & P&L mechanics (Black-Scholes + accounting)

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V3-01 | S4 ✅ | **All BS formulas correct** (verified by inspection): d1 at r=0, call/put, straddle Δ=2N(d1)−1, vega=2S·n(d1)√τ, gamma, theta=−S·n(d1)σ/√τ; self-test asserts put-call parity + ATM 0.7979·S·σ√τ | `options_pricing.py:61-143` | Credit | — |
| V3-02 | S4 ✅ | **P&L accounting correct & SHORT-ACCT honored:** `option_pnl=n(V_t−V_n)` (short gains when vol falls), `hedge_pnl=q_t(S_n−S_t)` (long perp), `funding_pnl=−q_t·S_t·funding[t]` (pays when +); pure MTM, no `notional_debt`; close books only transaction cost (buyback is implicit in daily MTM) | `falsification.py:176-180,139-148` | Credit | — |
| V3-03 | S3 | **Arg-swap** `falsification.py:144`: `straddle_price(S,K,max(tau,tau_step),sigma if False else sigma)` passes τ→σ slot, σ→τ slot, + dead ternary. Feeds **only** the fee premium-cap `min()`, which the underlying-fee branch always wins (0/90 closes bind @21d) ⇒ **$0 impact @21d, $11 @30d** — inert but a latent unit landmine; env `_close` differs in source | `falsification.py:144` vs `options_pricing.py:70`; recompute (cap binds 0/90) | One-line fix `straddle_price(S,K,sigma,max(tau,tau_step))` + delete dead code + regression test that makes the cap bind; assert verdict unchanged | Now |

### V4 — Execution & cost realism

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V4-01 | S3 | Honest fully-costed number = **1.06** (21d) / 1.29 (30d) once the omitted roll-hedge fee is charged. Total modelled cost ≈ $22k/5.2y on $100k: open_fee 2.5k + open_spr 4.8k + close_fee 2.5k + close_spr 2.6k + rehedge 3.9k + funding −5.7k + **omitted 1.1k** | recompute cost decomposition | Charge the omitted fee (V2-05); re-anchor to 1.06 | Now |
| V4-02 | **S2** | **Granularity blindness** (shared w/ V6): sim marks once daily at close; the cached high/low (which a real book lives through) has **no consumer**; spread blows out exactly when the short straddle must trade, unmodelled (1.0 vol-pt flat vs measured 0.6 calm / blows out in stress) | `falsification.py:161-194`; `cost_robustness.md` | Intrabar stress pass using cached high/low; stress-widen spread in vol-spike regimes | Next |
| V4-03 | S3 | Roll-before-expiry fidelity needs paid Tardis; free substrate is monthly-snapshot only — quantified ceiling already documented | `cost_robustness.md:9-20` | Hold or buy chain data ($700–2700) | Research |

### V5 — Margin, liquidation & capital realism

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V5-01 | **S2** | **Margin entirely unmodeled** (only `equity>0` guard) — but adversarial recompute shows it **survives**: Deribit standard-margin IM-at-open median 15.0% / **max 41.2%**; intraday MM-usage peaks **34.7%**, **0 days >50%** incl. all crash days; worst MM/eq under −30%/+40 = **0.31**. Verdict survives; the safeguard is absent and headroom rests on this probe | `falsification.py:191-192`; `options_vol_harvest_env.py:356`; arch doc open-Q5 vs ✅ MARGIN-CFG; recompute | Add a Deribit margin model to sim+env; assert MM/eq < threshold each bar; emit max-usage to verdict, gate it | Now |
| V5-03 | **S2** | **Tail gates declared-but-not-wired:** `cvar95_floor_pct=-8.0` and `max_net_vega_per_100k=50000` have **no consumer**; `max_worst_window_dd_pct` is graded only on the *discarded RL leg*, never the linear core; `compute_stress_subreport` never invoked for options (retro-check: core would PASS — DDs 2.5–3.9%, vega/eq 0.142<0.50) | gates.yaml:41-48; `options_vol_pipeline.py:460,588`; grep no consumers | Wire the tail block against the linear core (fail loud); delete/implement `max_net_vega_per_100k` | Now |
| V5-04 | S3 | Headline DD 5.38% is close-basis; intraday-trough = **8.42%** (worst 2025-10-10), 16.95% at the 0.10 grid sizing — quoted DD ~1.6× optimistic as capital-at-risk | recompute; `verdict.json:53` | Report both; key any DD budget to the intraday number | Next |
| V5-05 | S3 | The 0.10-sizing grid rows are presented GO with no margin caveat but are margin-tight (intraday MM-usage 71.4%, 13 days >50%) — capital feasibility silently differs from 0.05 | recompute (pf=0.10) | Annotate grid with max-margin-usage; cap documented sizing | Next |
| V5-02 | **S2** | Collateral denomination undecided/unmodeled: sim is USD-linear; Deribit flagship BTC options are **coin-settled inverse** (wrong-way margin spiral — collateral devalues as short put goes ITM). Severity depends on the venue/contract choice, which is itself undecided | `options_pricing.py:12-15`; all-USD accounting | **NEEDS-DATA:** pick contract (coin-settled vs USDC-linear) + collateral policy; re-run margin probe in chosen frame | Research |
| V5-07 | S3 | Undocumented IV-inverse sizing: notional/equity ∝ 1/IV → largest in *low-vol* regimes (worst margin days were calm Oct–Nov 2023, not crashes — the classic short-vol trap shape); linear sim has zero exposure caps (env-only) | `falsification.py:127-129`; `options_vol_harvest_env.py:209-223` | Document; port env vega/premium caps to the linear sim (no-op at 5%, assert) | Next |
| V5-08/09/10 | S4 ✅ | POSITIVES: full reproducibility (parity 0.0); capital base **conservative** (4× margin headroom, anti-martingale auto-deleverage, portfolio = 2 independent 50k sleeves, no hidden 2× leverage); funding complete & correctly signed; SHORT-ACCT honored | recompute; `falsification.py:128,176-180,293-299` | Emit IM-headroom to verdict when V5-01 lands; add funding to `_validate` | Next |

### V6 — Tail risk & adversarial stress *(the pillar this audit exists for)*

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V6-01 | S4 ✅ | **Overnight shock survives at shipped sizing** — instantaneous loss as % of equity: **−15% spot/+20pt: median 3.0 / p90 5.3 / max 7.7**; **−30%/+40pt: median 8.0 / p90 14.5 / max 23.1**; +20%/+15pt: max 11.1. All < 40% kill & ~25% tail gate. Book is short only ~0.057 vega/eq per 100pt, median notional/eq 0.39 | recompute (D) | Credit; gate the deploy on the intraday/shock bound, not close DD | Now |
| V6-02 | **S2** | **Left tail is real and recurring:** daily skew **−2.73**, kurt 20.2, **Sortino 0.99 < Sharpe 1.10** (Sharpe overstates a short-vol book); CVaR95 −0.73%/day, CVaR99 −1.47%, worst −2.36%. 10 worst days span 2021–2025 (not single-event) — diversified but recurring | recompute (B) | Report Sortino/CVaR alongside Sharpe; size to the left-tail, not the mean | Now |
| V6-03 | **S2** | **Statistical fragility (the deploy-critical caveat):** block-bootstrap Sharpe **CI95 [0.18, 2.17]**, P(SR<0.5)=**10.7%**, P(SR<0)=1.2%; **deflated Sharpe 0.77 (N=8) → 0.65 (N=16) → 0.54 (N=28 trials)** — the 1.10 barely clears its multiplicity-corrected hurdle (SR* 1.05 ann at N=28) | recompute (F) | Treat as a *candidate* not a proven sleeve; log DSR so paper out-perf ≠ confirmation | Now |
| V6-04 | S3 | Intrabar DD 8.42% vs 5.38% close; 2025-10-10 alone had a 6.2%-of-peak intraday excursion below the close mark | recompute (C) | (V5-04) | Next |
| V6-05 | S4 ✅ | VRP premium dominates the negative windows: 14/63 (22%) negative, **not heavily clustered**; sum negatives −1.79 vs positives +8.86; worst window −0.275 | recompute (G) | Credit | — |

### V7 — Methodology, multiplicity & verdict reproduction

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V7-01 | S4 ✅ | **Verdict reproduces exactly** (3 independent re-derivations + this one): Sharpe 1.1002 / PF 1.1927 / DD 5.3789% / ret 30.5171%, parity max|Δeq| = 0.0 over 1892 bars | recompute (A) | Credit; freeze the parquets as the data fingerprint | Now |
| V7-02 | **S2** | Multiplicity not corrected in the headline: ≥16 (8 grid × 2 asset) + ~12 spread-grid configs graded; DSR at N=16–28 = 0.54–0.65 (see V6-03); no held-out period the cadence/asset choice never touched | recompute; `cost_robustness.md` grid | Holdout cadence grade (V2-03); quote DSR | Now |
| V7-03 | S3 | Pre-registration / provenance not tamper-evident (see V2-11, V1-07) | git log | Commit gates before run; hash data into verdict | Now |

### V8 — External SOTA benchmark (short-vol best practice)

| ID | Sev | Finding | Evidence | Fix | Effort |
|----|-----|---------|----------|-----|--------|
| V8-01 | S4 ✅ | **REFUTES the "DVOL rich" worry:** measured DVOL−ATM-IV basis median **+1.5** vol-pt, mean −2.2, 62% positive (n=34 free Tardis snapshots) — the proxy is ~unbiased at the median; the synthetic→real gap is **cadence + spread + monthly-hold path**, not a systematic mark bias | recompute (V8 probe) | Quote the band, but stop attributing the gap to DVOL richness | — |
| V8-02 | S2 | 1.10 gross Sharpe is plausible vs published systematic crypto short-vol (typ. 0.5–1.0 net before tail events), but standard tail overlays are **absent**: no long-wing (short strangle vs straddle), no vol-regime entry filter, no DVOL-level gate. Single highest-value missing control = **a long-wing tail hedge or a DVOL-level entry gate** | `architecture.md`; literature | Add a regime/level entry gate or wing as the v1.1 tail control | Research |
| V8-03 | S3 | DVOL is a 30d constant-maturity *variance-style* synthetic; the sim trades a 30d ATM straddle rolled at 21d — basis dispersion IQR [−3.97, +3.10] is wide even if median-unbiased; recent 2025–26 regime shows ATM IV ≫ DVOL | recompute (V8) | Quantify per-snapshot wedge (free); prefer real ATM IV when chain is bought | Research |

---

## Prioritized Roadmap

### NOW — config / test / doc, no retrain (≈1 day; do before the paper sleeve)
1. **Charge the omitted perp-hedge fee** in both engine copies; re-run the gate + parity keystone — expect 1.10→**1.06**. *(V2-05/V4-01)*
2. **Fix the arg-swap** `falsification.py:144` + delete dead ternary + add a cap-binding regression test; assert verdict byte-identical. *(V3-03)*
3. **Wire the declared safeguards:** DATA-CLEAN checks into `_validate` (raise on issues, incl. funding frame); tail gates (`cvar95`, `max_net_vega_per_100k`, per-window DD) against the **linear core**; load gate thresholds **from** the yaml. *(V1-06, V5-03, V1-11)*
4. **Re-anchor the headline to a band (0.6–0.9 honest / 1.10 best-case)** in gates.yaml/randd/arch doc; add Sortino, CVaR95/99, intraday DD (8.4%), and **DSR (0.54–0.65)** to verdict.json. *(V6-02/03, V5-04, V7-02)*
5. **Provenance:** retro-commit verdict.json + parquet SHA256; `.bak` before `--refresh`; fix the false ✅ rows in the arch compliance table; fix the 8h-misalignment docstring + add a stamp-convention tripwire. *(V1-07, V2-11, V1-04)*
6. **Holdout cadence grade:** pick cadence on 2021–23, grade 2024–26 untouched (script already parameterized) — closes the in-sample multiplicity. *(V2-03/V7-02)*

### NEXT — requires a re-run or paid data (before REAL capital)
7. **Intrabar stress pass** using the cached perp/DVOL high-low; gate the deploy on the intraday-trough DD and a margin model, not close-to-close. *(V4-02, V5-01/04)*
8. **Real-chain 21d-roll reconciliation** on paid Tardis to isolate cadence from mark fidelity and confirm the 0.6–1.1 band. *(V1-03, V2-09)*
9. **Phase-ensemble + sizing:** trade 2–4 staggered roll-phase sub-books; cap documented sizing where intraday margin usage stays <50%. *(V2-04, V5-05)*

### RESEARCH — open questions
10. **Collateral denomination decision** (coin-settled inverse vs USDC-linear) + re-run margin in the chosen frame — wrong-way spiral risk. *(V5-02)*
11. **Tail overlay:** long-wing (short strangle) or DVOL-level/regime entry gate as the v1.1 highest-value tail control. *(V8-02)*
12. DVOL launch/backfill PIT check; per-snapshot DVOL-ATM wedge quantification (free). *(V1-13, V8-03)*

---

## External Benchmark Gap Analysis (V8)

Systematic short-vol on crypto is a documented, **un-decayed** premium (BTC IV>RV ~70%+ of days since 2019; unlike S&P VRP which decayed post-2010). A net Sharpe in the 0.5–1.0 range before tail events is consistent with published practice, so **1.10 gross is plausible, not a red flag** — but the literature standard is to pair the short premium with a tail control, which this core lacks. The single highest-value missing control is a **long-wing hedge (trade the strangle instead of the naked straddle) or a vol-level/regime entry gate**; Phase-0b already showed the real-chain *strangle* (0.92) beat the straddle (0.61), i.e. the skew leg both adds edge and truncates the tail. The DVOL-vs-tradeable-ATM basis — the obvious proxy risk — was measured and is **median-unbiased (+1.5 vol-pt)**, so the proxy is acceptable for a candidate but should be replaced with real ATM IV before real capital.

---

## Appendix — Verification Log

**Provenance:** workflow `wf_84638d3a-4d5` terminated on spend limit after finders V1/V2/V5; V3/V4/V6/V7/V8 + all adjudication completed inline (Opus 4.8) by recovering and re-running the killed finders' probe scripts and direct code verification. The two gating recompute scripts are persisted in-repo: **`scripts/research/options_vrp_tail_recompute.py`** (V6 tail: parity, distribution/skew/CVaR, intrabar-trough DD, overnight shock, margin/liquidation, block-bootstrap + DSR, VRP-negative conditioning) and **`scripts/research/options_vrp_cost_recompute.py`** (V4 cost decomposition + arg-swap A/B + omitted-fee). Both eval-only/CPU; reproduce all numbers above against the cached `data/processed/deribit/*.parquet`.

| Pillar | Confirmed | Refuted | Needs-Data | Positives (S4) | Independent reproduction |
|--------|-----------|---------|------------|----------------|--------------------------|
| V1 Data | V1-03,04,05,06,07,09,10,16 | — | V1-13 | V1-01,02,14,15 | Sharpe/PF/DD/ret to 4dp |
| V2 Causality | V2-03,04,05,06,09,10,11 | — | — | V2-02,08,12 | parity, 15/15 tests |
| V3 Pricing | V3-03 (inert) | — | — | V3-01,02 | BS verified by inspection |
| V4 Cost | V4-01,02,03 | — | — | — | cost decomposition reproduced |
| V5 Margin | V5-01,03,04,05,07 | — | V5-02 | V5-08,09,10 | parity 0.0, margin probe |
| V6 Tail | V6-02,03,04 | — | — | V6-01,05 | shock/bootstrap/DSR recomputed |
| V7 Method | V7-02,03 | — | — | V7-01 | 4th independent repro |
| V8 SOTA | V8-02,03 | **V8-01 (DVOL-rich worry)** | — | V8-01 | DVOL-ATM basis measured |

**Counts:** Confirmed ≈ 27 · Refuted 1 (the DVOL-rich mechanism) · Needs-Data 2 · Positives 14. **No S1.** Dominant S2 cluster: over-optimistic headline (cost/fragility/selection) + declared-but-not-wired safeguards.

**Kill-criteria status (pre-registered):** net Sharpe 1.10 > 0.50 ✅ · net PF 1.19 > 1.10 ✅ · multi-subperiod ✅ · recent-12m Sharpe > 0 ✅ · worst DD 5.38% (8.4% intraday) < 40% ✅. **Verdict survives all kill criteria** at shipped sizing; the audit's contribution is the *honest-band* re-framing and the wiring debt, not a falsification.
