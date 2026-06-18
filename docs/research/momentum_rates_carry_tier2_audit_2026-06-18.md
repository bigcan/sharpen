# Tier-2 Deep Lifecycle Audit — momentum + rates-carry 2-sleeve book

> **Session:** 553-cont-53 | **Date:** 2026-06-18 | **Gate:** pre-paper-capital stakes gate (CLAUDE.md mandatory)
> **Verdict:** **PROCEED-WITH-CAVEATS to PAPER — 0 S1, 2 S2, 4 S3.** Capital (paper) is **not blocked on correctness**; the book is leak-free, causal, cost-honest, and the diversification is real. But forward expectations must be set at **honest net Sharpe ~0.4–0.5**, NOT the curated 0.60–0.75 — rates-carry's standalone edge is ZIRP-era-inflated and its forward value is mostly *decorrelation*, not return.

---

## 0. Provenance note — why this is a focused inline audit

The mandated `deep_strategy_audit` workflow was launched (run `wf_e304b81d-9b8`, 23 finder+skeptic agents). It **mis-parsed the scope** (`workstream: UNSPECIFIED, scope: all`) and the finders audited the **legacy V7 multiscale infrastructure** (e.g. the P2 finder went after `close_z` train/serve skew in the multiscale handler) — code this linear book **does not use**. The final synthesis also died on the account monthly spend limit (same failure mode as the cont-41 options-VRP audit). Rather than rely on a mis-targeted sweep, I ran a **focused, correctly-scoped 7-attack adversarial re-derivation** of the four capital-gating claims directly against this book. Script: `scripts/research/audit_two_sleeve_book.py`; results `results/portfolio_frontier/audit_2sleeve.json`. The workflow's findings about *shared* infra (clean_ohlcv, data provenance) remain available but are largely N/A here.

## 1. Scope

The deploy-gating artifact = the **static linear momentum + rates-carry book** (`xsec_momentum_falsification.py` + `carry_falsification.py` rates sleeve + `portfolio_frontier.py`). No RL / training / multiseed / walk-forward / ensemble → pillars P4–P7, P9 are N/A. Relevant pillars: **P1 data, P2 causality/leak, P3 execution/cost, P8 verdict-gates, P10 sim-to-live.**

## 2. The 7 adversarial attacks (each tries to BREAK a claim)

| # | Attack | Result | Pass |
|---|---|---|---|
| **A** | rates-carry **leak** (same-day vs causal T+1) | causal 0.467 vs same-day **0.430** (gap −0.037 — same-day *worse* ⇒ no look-ahead inflates it) | ✅ |
| **B** | rates-carry **regime front-loading** (subperiod Sharpe) | 2006-09 **0.349** / 2010-15 **0.963** / 2016-20 **0.629** / 2021-26 **−0.374** (3/4 positive) | ✅ gate, ⚠ S2 |
| **C** | **corr stability** incl. stress | worst-subperiod corr **0.189**; 2021-26 **−0.193** — never spikes; diversification holds when needed | ✅ (strong) |
| **D** | **combined** front-loading / OOS | positive **4/4** subperiods; OOS-2018 **0.386** (curated) | ✅ |
| **E** | **cost** survival (harsh 10 bps) | combined 0.75 → **0.688** at 10 bps | ✅ |
| **F** | rates-carry **concentration** (drop-one ETF) | min drop-one **0.428** (SHY/IEF/TLT/LQD all ~0.43–0.47) — not one-instrument-driven | ✅ |
| **G** | **meta-layer cost** (frictionless combine optimism) | 2nd-order; sleeves' internal cost already charged; residual est <2 bps/yr | ✅ (S3) |

**No S1 found.** The book is leak-free (A), causal (A), cost-survivable (E), the diversification is real and stress-robust (C), and the edge is not concentration-driven (F).

## 3. Adjudicated findings (skeptic pass — beyond the mechanical gates)

### S2-1 — rates-carry standalone is regime-dependent (ZIRP-era inflated) [the dominant caveat]
rates-carry passed the ≥3/4-subperiod gate but is **−0.374 in 2021-26** (the rate-hiking / curve-inversion regime). The +0.467 full-sample number is carried by the **2010-2020 steep-curve/ZIRP era** (0.96 / 0.63). Curve-carry is short-vol / crash-exposed (BNP); it whipsawed through the 2022-2024 inversion and bond round-trip. **Forward expectation for the rates-carry sleeve's STANDALONE return should be discounted toward ~0 in the current regime.** Its justification in the book is **decorrelation** (corr −0.193 even in 2021-26), which DOES still hold — it lowers book variance even when its own return is flat.

### S2-2 — deploy expectation is ~0.4–0.5 net Sharpe, not 0.60–0.75
The headline combined Sharpe 0.75 is **curated** (18-ETF momentum). Honest-haircut (momentum 0.389) → combined **0.601**; **OOS-2018 0.386 (curated)** → honest-haircut OOS ≈ **0.30–0.45**. Combined with S2-1 (rates-carry weak forward), the **honest forward expectation is net Sharpe ~0.4–0.5**. Size and report on that basis. This is consistent with the Fable honest band and the rung-1 paper executor's ~0.44 render.

### S3 caveats (hygiene / expectation-setting; do not block paper)
- **S3-1 frontier vol-scalar:** `portfolio_frontier.py` scales the combined series by a **full-sample** constant to hit each vol target (Sharpe-invariant, fine for illustration) — a *deployed* book must vol-target on **trailing** realized vol. The reported frontier DD is indicative, not a live guarantee.
- **S3-2 Yahoo curve provenance:** the Treasury curve (`^IRX/^FVX/^TNX/^TYX`) is **not run through `clean_ohlcv`** (it is a yield index, not OHLCV). Residual bad-print risk is low (drop-one and leak tests passed) but a stale-print/continuity check should wrap the curve loader before live capital.
- **S3-3 prop-firm DD mismatch:** at a deployable 2× (20 % vol) the honest maxDD is ~−26 %, far beyond an FTMO ~10 % limit. This book is the **north-star uncorrelated sleeve**, NOT the prop-firm product; for a prop-firm sleeve it must be sized to ~8 % vol (~5 %/yr, ~−14 % DD).
- **S3-4 meta-layer cost** uncharged (immaterial, <2 bps/yr).

## 4. Verdict & deploy guidance

**PROCEED-WITH-CAVEATS to PAPER (0 S1).** The momentum + rates-carry book is correctness-sound for a **paper soak**; no leak, no cost artifact, real diversification. Promotion to **real capital** remains gated on the paper soak + the S2/S3 items being reflected in sizing and monitoring.

**Deploy expectation (honest):** net Sharpe **~0.4–0.5**; at 10 % vol ≈ **5–6 %/yr at ~−14 % DD**; rates-carry contributes **primarily decorrelation**, expect ~0 standalone return from it in the current rate regime. Monitor the rates-carry sleeve's live contribution separately — if the curve re-steepens, its standalone return should recover; if not, the book is ~momentum-alone + a variance reducer.

## 5. Concrete next steps
1. Wire rates-carry into the rung-1 paper executor as the documented `carry_ary` additive sleeve (`finrl_pro_ds/envs/multi_asset_allocator_env.py:511`); re-run the rung-1 **forward-path** audit on the 2-sleeve executor (the cont-47 step-4 capital-gate items still apply).
2. Add a stale-print/continuity wrap on the Yahoo curve loader before any real-capital step (S3-2).
3. Set monitoring to the honest ~0.4–0.5 Sharpe expectation and attribute per-sleeve P&L (watch rates-carry's regime contribution).
4. Keep building uncorrelated sleeves — the only honest lever toward higher portfolio Sharpe.

## 6. Artifacts
- Focused audit: `scripts/research/audit_two_sleeve_book.py` → `results/portfolio_frontier/audit_2sleeve.json`
- Mis-scoped workflow run: `wf_e304b81d-9b8` (transcripts retained; legacy-infra scope)
- Strategy report: `docs/research/multi_sleeve_strategy_report_2026-06-18.md`
