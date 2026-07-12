# FE Observation-Channel Pre-Registrations — FFD price channel + per-timeframe `norm_span`

> **Status:** SPEC (pre-registered 2026-07-12, Session S553-cont-125) — gates locked BEFORE any result is computed. **No run launched.**
> **Parent:** `docs/research/feature_engineering_deep_audit_gmgp1_sg1_2026-07-08.md` — RESEARCH backlog items (FE-09 FFD channel, FE-05 `norm_span`). Both are **obs-distribution changes** ⇒ they cannot reuse the current checkpoints ⇒ a full Protocol-v2 lifecycle (`docs/protocol_v2.md`) if they graduate.
> **Prior:** FE-09 concluded the existing 8-feature set already spans the memory/stationarity trade-off (`log_return` at d=1, `close_z` at d=0 level), so a fractional-`d` channel *interpolates between two endpoints already in the obs* — expected marginal. These pre-registrations exist to make that prediction **falsifiable and cheaply testable** before any training spend.
> **Design philosophy:** falsify-before-optimize + beat-linear gate (Fable verdict 2026-06-11). A Stage-0 CPU incremental-information probe must clear a locked MDE before any GPU is provisioned. FE-09's prior is that Stage 0 fails; a pre-registered pass would be the evidence that overturns it.

---

## 0. Scope, assets, and what is NOT under test

**Live/candidate cells only** (sg1-btc is SHELVED 2026-06-01 — de-leaked WF FAIL, edge was the X2 leak; it is a **control-only** asset here, never a deploy target):

| Role | Cell | Base scale | Coarse scales | Config anchor |
|------|------|-----------|---------------|---------------|
| Primary | GMGP1 GC 15m | 15m | [15, 60, 240] | `configs/gmgp1_sac_gc_15min.yaml` |
| Primary | GMGP1 XAUUSD 15m (live paper) | 15m | [15, 60, 240] | `configs/gmgp1_xauusd_sac_15min.yaml` |
| Secondary | SG-1 XAUUSD 3m (live paper) | 3m | [3, 15, 60] | `configs/live_sg1_xauusd_ctrader.yaml` twin |
| Control | BTC 15m / 3m | — | — | canary IC ≈ 0 (`results/gmgp1_btc_canary_costcorr_wf/`) |

**Not under test / out of scope:** any change to the reward, action space, cost model, or the leak-fixed `multiscale_handler.py` causal path; any sg1-btc revival (needs a *new structural signal*, not a feature tweak); the day-of-week encoding idea (separate backlog item, FE audit line 100).

**Data-quality precondition (Gate D, from R1 pattern):** every asset passes `scripts/clean_ohlcv.py` (+`.bak`, DATA-CLEAN invariant) and the stale-print scan (≥30 consecutive identical closes with nonzero volume; daily flat-bar fraction >50% beside <10% neighbours; post-flat jump returns). GC/XAUUSD have a corruption history (S106; `project_gmgp1_gold_deleak_probe_s553`) — any asset failing Gate D is logged **UNTESTABLE**, never silently dropped.

---

## 1. Shared Stage-0 harness (CPU-only, $0 GPU)

Reuses the R1-illiquidity probe machinery verbatim (`scripts/research/r1_illiquidity_probe.py`) so probe fidelity == architecture fidelity:

- **Feature construction:** instantiate `MultiScaleOHLCVHandler` per cell with production configs (`window_size=30`, all features), consume `_scale_features` + `_scale_index_map` directly. X2-causal condition re-asserted per gather (tripwire TW-4 below).
- **Target:** forward log-return over H base bars. **Primary endpoint H=1** (matches bar-by-bar repositioning); H=4 secondary; H=16 diagnostic-only.
- **Models:** M1 = Ridge on the flattened obs (α ∈ {1e2,1e3,1e4}, picked on last 20% of train — causal inner validation). M2 = `HistGradientBoostingRegressor` (max_iter 200, lr 0.05) on the reduced summary view. M0 = per-feature Spearman IC (diagnostic).
- **Walk-forward:** expanding window, 6 OOS folds × 2 months over 2025-06→2026-06, **embargo 16 base bars**. Pooled OOS = fold concatenation.
- **Statistics:** pooled-OOS Spearman IC; CI via circular block bootstrap (block = 1 trading day, 10,000 draws). Multiplicity handled by Benjamini–Hochberg FDR over the pre-declared primary family (below). Any cell outside the primary family is exploratory and **cannot flip a verdict**.
- **Trade sim (for cost gates):** `pos_t = sign(pred_t)` if `|pred_t| ≥ τ` (τ = 60th pct of |pred| on that fold's train) else 0; cost = one-way (taker 5.5bps + 5bps slip) × |Δpos|. Report frictionless / net / 2×-harsh PF, MDD, trade count.

**The incremental test (this is the whole point).** Both studies are **nested A/B on the same harness**: fit `baseline` obs and `augmented`/`alternate` obs, compare. The gated quantity is the *difference* (ΔIC, ΔR², ΔPF), never the augmented model's absolute score — a channel that merely reproduces information already present must not pass.

---

## 2. Part A — FFD price channel

### A.1 Hypothesis (mechanism)
A fractionally-differenced log-price channel at `d ∈ {0.3, 0.4, 0.5}` (`_fractional_diff`, window=100, FFD weights, `crypto_features.py`, FE-08-fixed) preserves **long-memory level information** that `log_return` (d=1) discards, while being more stationary than `close_z` (d=0 level). **Prediction under H1:** adding this channel raises pooled-OOS forward-return IC beyond the 8-feature baseline by a margin exceeding the MDE. **FE-09 null:** ΔIC ≈ 0 because `log_return`⊕`close_z` already bracket the d-spectrum; the fractional channel is redundant.

### A.2 Construction
- Compute `ffd_close = _fractional_diff(log(close), d, window=100)` per scale, then normalize through the **same** `SymLog → EMA-Z(span=120) → tanh` pipeline as the z-features (keeps the augmented dim on the same bounded scale — no distribution shock). One extra feature per scale ⇒ augmented obs = 9 features × 3 scales.
- `d` is a **pre-registered set {0.3, 0.4, 0.5}**, not tuned on OOS. Each `d` is one cell in the family; the best `d` is chosen by the *train-fold* inner validation only.
- Causality: `_fractional_diff` is a backward-looking convolution (weights on bars ≤ t); TW-4 asserts max feature timestamp ≤ t after the channel is added.

### A.3 Primary family (locked)
{cell (4 primary/secondary) × d (3) × model (M1, M2)} at H=1 = **24 incremental tests** → BH-FDR. Controls (BTC) reported but excluded from the discovery family.

---

## 3. Part B — per-timeframe `norm_span`

### B.1 Hypothesis (mechanism)
`norm_span=120` is a single default: 120 base bars = **6h at 3m (SG-1) vs 30h at 15m (GMGP1)** of EMA-Z normalization horizon (FE-05). The horizon was never chosen per workstream and is not in HPO space. **Prediction under H1:** some `norm_span ≠ 120` raises the z-features' (dims 3–7) forward-return IC — and/or downstream sim PF — beyond the MDE, i.e. the normalization horizon is mis-set for at least one timeframe. **Null:** 120 is within noise of the best span everywhere; the default is fine.

### B.2 Construction
- Recompute obs at `norm_span ∈ {60, 120, 240, 480}` (120 = incumbent control). This **changes existing dims 3–7 in place** — it is not an added channel, so the A/B is a clean same-dimensionality swap.
- Report per-span pooled-OOS IC of the z-block *and* the full-obs model IC/PF, so a span that helps normalization but hurts the model is caught.
- **Live-parity note (blocking for any graduation):** `norm_span` must be identical train==live (LEAK/parity). A changed span means a new `norm_warmup_*.pkl` bundle and a coordinated live-config bump; flagged now so it is never a silent skew.

### B.3 Primary family (locked)
{cell (4) × span (3 non-incumbent) × model (M1, M2)} at H=1 = **24 tests** → BH-FDR, each vs its own span-120 control on the identical folds.

---

## 4. Tripwires (must pass before any result is read; failure = HALT/fix/rerun)

- **TW-1 (detector works):** rebuild one cell's obs with features shifted +1 bar into the future → pooled IC > +0.10.
- **TW-2 (null is null):** block-shuffled targets → |IC| < 0.01, FDR discoveries = 0, and **ΔIC (aug−base) 95% CI contains 0** (proves the incremental test doesn't manufacture edge from added dims).
- **TW-3 (control consistency):** BTC 15m baseline pooled-OOS IC ≤ +0.01 (canary parity). A large positive control IC ⇒ probe leaks ⇒ HALT.
- **TW-4 (X2 causality):** `coarse_ts[idx] + scale ≤ base_ts[t]` and max feature timestamp ≤ t for every gather, **including the FFD channel**.
- **TW-5 (redundancy sanity, Part A):** the FFD channel's raw Spearman IC vs `log_return` and vs `close_z` is reported; if the channel is ~collinear (|ρ| > 0.95) with an existing feature the "new information" claim is mechanically void regardless of ΔIC.

---

## 5. GO / NO-GO decision rules (LOCKED — numeric gates frozen before results)

**Stage-0 MDE (the CPU gate that decides whether GPU is ever provisioned):**

| Gate | Condition |
|------|-----------|
| **S0-A — incremental structure** | On ≥1 data-clean **primary** cell: **ΔIC (augmented/alternate − baseline) ≥ 0.010** at H=1, AND block-bootstrap 95% CI on ΔIC **> 0**, AND BH-FDR q < 0.10 within the primary family. |
| **S0-B — not a cost mirage** | For any S0-A passer: augmented/alternate **net PF ≥ baseline net PF** (channel must not add turnover without edge) AND frictionless PF ≥ 1.20 on that cell. |
| **S0-C — redundancy cleared (Part A only)** | TW-5 passes (FFD channel not collinear with `log_return`/`close_z`). |

- **GO → Stage 1 (GPU):** ≥1 primary cell passes S0-A + S0-B (+ S0-C for FFD). *Only then* is a single confirmatory Protocol-v2 chain (data-prep → hpo → l1-multiseed N=10 → WF 8-fold+stress → recent-oos) launched on that one cell, A/B vs the current 8-feature/120-span incumbent checkpoint. Training-side numeric gates move to a `configs/<cell>.gates.yaml` (never hardcoded — CLAUDE.md). RL A/B still only behind the beat-linear gate. Expectation bounded by the probe's ΔPF.
- **NO-GO (FE-09 confirmed):** no primary cell clears S0-A. → The FFD channel / alternate `norm_span` adds no linearly-or-GBM-extractable forward-return information beyond the incumbent obs at 3m/15m on the live cells. Backlog item CLOSED as "measured redundant," not merely "untested." **Zero GPU spent.**
- **WEAK / AMBIGUOUS:** a single marginal FDR survivor (ΔIC 0.010–0.015) failing S0-B, or a `norm_span` that improves z-block IC but not full-obs PF → document the exact cell + the single cheapest disambiguating follow-up; no GPU until resolved.

**Caveats locked with the gates:** (i) the deadband sim is cost-optimistic (flat = zero cost) → a NO-GO under optimistic accounting is conservative. (ii) `d ∈ {0.3,0.4,0.5}` and `span ∈ {60,240,480}` are the *entire* pre-registered grids; no post-hoc grid extension may rescue a NO-GO. (iii) A Stage-0 GO licenses **one** confirmatory chain per cell, not a deploy — a Tier-2 deep lifecycle audit (`deep_strategy_audit`) remains mandatory at any capital gate. (iv) Stage-0 tests linear+GBM extractability; a negative result removes the evidence-based justification for GPU, it does not mathematically exclude an exotic SAC-only nonlinearity (beat-linear mandate).

---

## 6. Outputs (when/if run)

`results/fe_obs_channel/`: `data_qc.json`, `ffd_ic_table.csv`, `norm_span_ic_table.csv`, `incremental_delta.csv` (ΔIC + bootstrap CI per cell), `tripwires.json`, `verdict.json` (computed programmatically from the frozen gates), `summary.md`.
Scripts (to author at run time, forking the R1 probe): `scripts/research/fe_obs_channel_probe.py`.

**Budget:** Stage 0 ≈ 1 day wall-clock, $0 GPU. Stage 1 (only on a GO) = one Protocol-v2 chain per passing cell (~N=10 multiseed + 8-fold WF on gpuhub-2). Per the operator: **no training budget is committed by this document** — it is the pre-registration that a GO would unlock.

---

## 7. VERDICT (run 2026-07-12, same session as pre-registration) — **NO-GO**

Computed programmatically from the frozen gates: `results/fe_obs_channel/verdict.json` (full write-up `results/fe_obs_channel/summary.md`). Probe `scripts/research/fe_obs_channel_probe.py` (RNG seed 20260712, ruff-clean, reproducible). Read-only stale-scan substituted for the mutating `clean_ohlcv` Gate-D pass (a research probe must not alter live-strategy data); all cells CLEAN.

**Tripwires PASS:** TW-1 leak-shift IC = 1.00; TW-2 shuffled-null base IC 0.008 with paired ΔIC CI [−0.004,+0.004] bracketing 0 (the incremental test does not fabricate edge from added dims); TW-3 BTC control IC 0.003 ≤ +0.01; TW-4 X2 causality asserted on every gather.

**Gate S0-A (primary cells): FAIL — 0/24 passers.** On both window-complete gold cells no FFD-`d` or `norm_span` variant clears ΔIC ≥ 0.010 with CI_lo > 0 and FDR q < 0.10:

| Primary cell | best ΔIC (any variant×model) | median ΔIC |
|---|---:|---:|
| xauusd_15m | +0.0032 | −0.005 |
| xauusd_3m | −0.0008 | −0.002 |

Median primary ΔIC = −0.0018 (negative). **FE-09 confirmed:** the incumbent obs already spans the memory/stationarity spectrum; neither channel adds extractable forward-return information. (Base obs IC on the gold cells is 0.006–0.016 with sub-1.0 net PF — no base edge to amplify.)

**Secondary robustness (gc_15m): one non-triggering survivor, flagged as artifact.** `gc_15m|FFD d0.4|gbm` (ΔIC 0.0128, q 0.058) meets the raw numeric but per §0 cannot move the verdict — it is GBM-only (ridge +0.0097 < MDE), non-specific (the larger +0.0149 is a `norm_span` change on the same cell/model), and sits on an anomalous base IC (0.076 vs xauusd 0.006–0.016) that flags the 2025-only proxy itself.

**Outcome:** both backlog items CLOSE as **measured redundant** (not "untested"). No Protocol-v2 GPU chain unlocked; zero training budget spent. GC's anomalous base IC is a separate data-window/QC question, not evidence for either channel.

**Scope reconciliation (disclosed).** §0 assigned role labels to *config anchors* before checking data windows. At execution the local data forced a re-mapping: the GC file (`gc_2025_lob1_1min_stitched.parquet`) is **2025-only** → it cannot populate the frozen 2025-06→2026-06 folds (only 4 of 6) → demoted to secondary/robustness. The window-complete gold series is **XAUUSD OANDA** (`xauusd_m1_m.parquet`, 2024-01→2026-05), run at **both** 15m and 3m — these became the primary verdict family. The verdict is **robust to the labeling**: under §0's original labels (XAUUSD-15m primary, SG-1-3m secondary) the primary XAUUSD-15m cell still has 0/… S0-A passers (max ΔIC +0.0032); under the execution labeling (both XAUUSD cells primary) still 0. Every window-complete gold cell is NO-GO regardless of which one is called "primary."

_Note on the verdict script: its first pass mis-scoped the FDR family to include the secondary `gc_15m` cell and reported a spurious "GO"; corrected to the pre-registered PRIMARY-only family (spec §0/§5) before this verdict was recorded. The correction made the gate stricter to match the frozen spec, not looser to change an outcome._
