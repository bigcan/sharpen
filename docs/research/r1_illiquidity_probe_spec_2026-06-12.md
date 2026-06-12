# R1-illiquidity — Liquidity Dose-Response Probe (SG-1 / GMGP1 feature set on less-efficient assets)

> **Status:** SPEC (pre-registered 2026-06-12, Session 553-cont-44) — gates locked BEFORE any result is computed.
> **Operator hypothesis under test:** *"BTC / EURUSD / XAUUSD are too efficient, too liquid, too popular. Less-liquid assets might make SG-1 and GMGP1 work."*
> **Parent decisions:** `docs/research/redesign_pivot_2026-06-02.md` (falsify-before-optimize), Fable verdict 2026-06-11 (RL only behind a beat-linear gate).
> **Protocol class:** cheap CPU-only falsification (R0-regime pattern). No HPO, no training, no GPU, no deploy.

---

## 1. The question (decisive)

The clean-canary falsification (`results/gmgp1_btc_canary_costcorr_wf/`, re-verified 2026-06-09) established that on BTC the de-leaked multiscale feature set has **no gross edge** (frictionless best-solo PF ~1.07–1.09 < 1.2; realized IC < 0 in 25/25 checks). That is the *informational-efficiency* signature, which is **consistent with the operator's theory**: the assets tested so far are among the most heavily arbitraged instruments on earth at 3m–15m horizons.

The theory's testable prediction: **forward-return information extractable from the SG-1/GMGP1 feature set increases as asset liquidity/popularity decreases** (a dose-response), and on some tradeable less-liquid asset that information survives realistic costs.

Formally falsified so far: BTC only (n=1 clean asset — gold died of data corruption, EURUSD was never re-tested de-leaked). The theory is **technically untested**. This probe tests it across a 23-asset liquidity spectrum in one CPU run.

## 2. Why this is the right cheap test

- Reuses the **actual de-leaked feature pipeline**: `finrl_pro_ds/data/multiscale_handler.py` (`_resample_ohlcv`, `_compute_scale_features`, and the X2-causal `_scale_index_map` searchsorted fix at `multiscale_handler.py:360-370`). Zero feature reimplementation ⇒ no fidelity gap between probe and architecture.
- A regularized linear + gradient-boosted probe on the exact observation content is the **beat-linear gate floor** mandated by the Fable verdict: if the linear/GBM baseline finds nothing OOS on an asset, there is no sanctioned reason to point SAC at it. (Canary precedent: on BTC, trained SAC realized-IC matched the ~0 linear conclusion — RL did not find what linear missed.)
- Cost: ~1 day wall-clock (data fetch + CPU probe), $0 GPU. A NO-GO kills the lever pre-GPU; a GO names the asset and bounds expectations before any training spend.

**Scope limitation (stated up front):** the probe tests linear + GBM extractability of the feature set. A negative result does not *mathematically* exclude an exotic nonlinear signal only SAC could find; it removes any evidence-based justification for spending GPU weeks looking, per the beat-linear mandate.

## 3. Asset basket — liquidity spectrum (23 assets, 2 venues)

### Crypto venue (Binance USDT-margined perps, 1m klines via ccxt) — dose-response core

| Tier | Assets | Liquidity class | One-way slippage assumption |
|------|--------|-----------------|------------------------------|
| T0 (control) | BTCUSDT, ETHUSDT | mega-cap, max efficiency | 5.0 bps (canary parity) |
| T1 (Velotrade alts) | SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT | top-10, very liquid | 7.5 bps |
| T2 (mid-cap) | NEARUSDT, ATOMUSDT, FILUSDT, ARBUSDT, OPUSDT, INJUSDT | rank ~30–100 | 12.5 bps |
| T3 (small-cap) | GALAUSDT, CHZUSDT, SANDUSDT, ALGOUSDT | rank ~100–300 | 20.0 bps |

Taker fee 5.5 bps everywhere (Bybit/Binance parity with canary cost layer). All 16 listed on Binance futures before 2024-09-01. Realized liquidity is **measured from the data** (median daily dollar volume), not assumed from tier labels; tiers only set cost assumptions.

### OANDA venue (v20 practice REST, M1 mid) — FTMO-style CFD / exotic-FX branch

| Role | Instruments | One-way cost assumption (spread/2 + commission) |
|------|-------------|--------------------------------------------------|
| Control | EUR_USD | 1.0 bp |
| Exotic FX | USD_MXN, USD_ZAR, USD_TRY | 8 / 10 / 30 bps |
| CFD commodities/index | XCU_USD (copper), JP225_USD, WTICO_USD | 5 / 3 / 4 bps |

Cost numbers are broker-typical **estimates** (FTMO/OANDA retail-CFD class); sensitivity reported at 0.5× / 1× / 2×. Instruments OANDA does not serve are logged **UNTESTABLE** (no silent drops). OANDA `volume` = tick count — same convention the live XAUUSD/EURUSD strategies trained on.

**Window:** 2024-09-01 → 2026-06-01 (21 months). OOS region: 2025-06-01 → 2026-06-01.

## 4. Data-quality gate (Gate D) — runs BEFORE the probe

The gold retirement was data-blocked (stale +5% flat-OHLC prints ≈ 21% of best-fold P&L), and stale prints are exactly what illiquid series have more of. Therefore per asset:

1. `scripts/clean_ohlcv.py` pass (decimal-shift/spike repair, `.bak` mandatory — DATA-CLEAN invariant).
2. **Stale-print scan:** (a) runs ≥ 30 consecutive identical closes **with nonzero volume** (the gold corruption mode); (b) daily flat-bar fraction > 50% while adjacent days < 10% (frozen-feed signature); (c) post-flat-run jump returns (> 5× bar-σ immediately after a ≥ 30-bar flat run). Zero-volume flat minutes on small perps are genuine no-trade bars, NOT corruption — counted separately.
3. **Thresholds:** stale-suspect bars > 2% of series, or any single stale-suspect run contributing > 5% of the asset's total |log-return|, ⇒ asset = **UNTESTABLE** (excluded from inference, reported as data-blocked — the R0 Chronos convention: absence of evidence ≠ negative evidence).

## 5. Feature construction (architecture fidelity)

- Instantiate `MultiScaleOHLCVHandler` per asset per base scale with the production configs: **GMGP1 cell** scales `[15, 60, 240]` (base 15m), **SG-1 cell** scales `[3, 15, 60]` (base 3m); `window_size=30`, `norm_span=120`, all 8 features.
- Consume `_scale_features` + `_scale_index_map` directly (vectorized gather of the same windows `step()` would emit). The X2-causal condition is re-asserted inside the probe (tripwire TW-4): for every base bar t and coarse scale s, `coarse_ts[idx] + s ≤ base_ts[t]`.
- EMA-Z normalization in the handler is causal (`ewm().shift(1)`); the probe uses a single full-history pass (no `norm_cutoff_date`) which is valid for expanding-window WF because every feature at t is a pure function of bars ≤ t. LEAK-1's split-reset exists for RL train/test distribution hygiene, not causality, and does not apply to a causal expanding probe.

**Observation vector per base bar:** flattened 3 scales × 30 window × 8 features = **720 dims** (ridge), plus a reduced 69-dim view (summary-stats mean/std/last of features {0,1,2,6,7} × 3 scales = 45, + last-bar 8×3 = 24) for the GBM.

## 6. Probe design

- **Targets:** forward log-return over H ∈ {1, 4, 16} base bars. **Primary endpoint: H=1** (matches the envs' bar-by-bar repositioning). H=4 secondary, H=16 diagnostic-only.
- **Models:** M1 = Ridge on 720-dim obs (standardized on train only; α ∈ {1e2, 1e3, 1e4} picked on the last 20% of train — causal inner validation). M2 = `HistGradientBoostingRegressor` (max_iter 200, lr 0.05) on the 69-dim view; train rows uniformly strided to ≤ 100k per fit. M0 = per-feature Spearman IC of the 24 last-bar features (diagnostic, not gated).
- **Walk-forward:** expanding window, 6 OOS folds of 2 months each covering 2025-06 → 2026-06; **embargo of 16 base bars** between train end and fold start (kills target-overlap leakage). Pooled OOS = concatenation of folds.
- **Trade simulation (for Gate C):** position decided at close of bar t, filled at close t (env taker convention): `pos_t = sign(pred_t)` if `|pred_t| ≥ τ` else 0, where τ = 60th percentile of |pred| on that fold's train (causal deadband ≈ the envs' deadband 0.25 in spirit). Cost = one-way cost × |Δpos| per bar. Report frictionless PF, tier-cost PF, harsh (2× slippage) PF, plus return, MDD, time-in-market, trade count.

## 7. Statistics

- **IC:** pooled-OOS Spearman(pred, fwd_ret_H1). CI via circular block bootstrap (block = 1 day of base bars, 10,000 draws — canary pattern).
- **Multiplicity:** the **primary family** = {asset × base-scale × model(M1, M2)} at H=1 ≈ 92 tests → Benjamini-Hochberg FDR. Any cell outside the primary family (other horizons, M0 features, post-hoc slices) is **exploratory and cannot flip the verdict** — pre-registered against the "best cell out of 276" temptation.
- **Dose-response (Gate B):** within the crypto venue (16 assets, same fee structure, same venue), Spearman ρ between log(median daily dollar volume) and pooled-OOS IC, per scale × model. Theory predicts **ρ < 0**. Significance: permutation p < 0.05. OANDA branch (7 assets) reported descriptively.

## 8. Tripwires (must pass before results are read; any failure = HALT, fix, rerun)

- **TW-1 (leak detector works):** rebuild one asset's obs with features shifted +1 bar into the future → pooled IC must exceed +0.10. Proves the harness lights up on leak-level signal.
- **TW-2 (null is null):** block-shuffled targets → |IC| < 0.01 and FDR discoveries = 0.
- **TW-3 (control consistency):** BTC 15m pooled-OOS IC must be ≤ +0.01 (canary found realized IC < 0). BTC showing large positive IC ⇒ the probe itself leaks ⇒ HALT.
- **TW-4 (X2 causality):** assert `coarse_ts[idx] + scale ≤ base_ts[t]` for every gathered window; assert max feature timestamp ≤ t.

## 9. GO / NO-GO decision rules (locked)

| Gate | Condition |
|------|-----------|
| **A — structure exists** | pooled-OOS IC ≥ **0.015** AND block-bootstrap 95% CI > 0 AND BH-FDR q < 0.10 within the primary family, for ≥ 1 (asset, scale, model) cell on a **data-clean** asset |
| **B — dose-response** | crypto-venue Spearman(log liquidity, IC) < 0 with permutation p < 0.05 |
| **C — survives costs** | for Gate-A assets: pooled-OOS net PF ≥ **1.10** at tier cost AND frictionless PF ≥ **1.20** (canary skeptic floor) AND net-positive in ≥ 4/6 folds |

- **GO (theory supported, asset named):** ≥ 1 asset passes A + C. → Next step is a single confirmatory de-leaked Protocol-v2 chain on the top asset, expectations bounded by the probe's net PF; RL still only behind the beat-linear gate.
- **WEAK-GO (structure without capture):** A passes somewhere but C fails everywhere, **or** A fails everywhere while B is significantly negative. → The inefficiency mechanism is real but not capturable by intraday taker-style SG/GMGP at realistic costs; actionable expression = extend the cross-sectional/TSMOM core universe with less-efficient sleeves (low-turnover capture). No new intraday RL workstream.
- **NO-GO (theory falsified on the testable universe):** no Gate-A pass anywhere AND Gate B flat/absent. → "Less-liquid assets rescue SG-1/GMGP1" is rejected at 3m/15m horizons across 4 orders of magnitude of crypto liquidity + exotic FX/CFD; the redesign-pivot conclusion (the signal family, not the venue, is the problem) stands fully general.
- **AMBIGUOUS:** anything borderline (e.g., single FDR survivor with IC 0.015–0.02 failing C marginally) → document the exact cell and the single cheapest follow-up that would resolve it; no GPU until resolved.

**Caveats locked with the gates:** (i) the deadband sim is cost-optimistic for the gate (flat = zero cost, no funding/financing); a NO-GO under optimistic accounting is conservative. (ii) Tier cost assumptions for T2/T3 are estimates; Gate C is therefore also reported at 0.5×/2× cost for sensitivity, but the verdict uses the tier numbers. (iii) 2025-06→2026-06 OOS is one macro year; a GO is a license for one confirmatory chain, not a deploy verdict (Tier-2 audit still required at any capital gate).

## 10. Outputs

`results/r1_illiquidity/`: `data_qc.json`, `liquidity.json`, `ic_table.csv`, `wf_probe.csv`, `dose_response.json`, `tripwires.json`, `verdict.json` (computed programmatically), `summary.md`.
Scripts: `scripts/research/r1_illiquidity_fetch.py`, `scripts/research/r1_illiquidity_probe.py`.

---

## 11. VERDICT

*(to be appended after the run — gates above are frozen as of pre-registration commit)*
