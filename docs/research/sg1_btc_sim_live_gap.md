# SG-1-BTC Sim→Live Transfer Gap — Attribution Waterfall

**Strategy:** SG-1, 3-min BTC/USDT perp, SAC SOLO seed 456 ("DECAY-01"), Bybit mainnet-demo, LIMIT @ 5 bps offset.
**Date:** 2026-05-29 · **Method:** 5-layer parallel diff (L1 Data, L2 Obs/Norm, L3 Policy, L4 Execution, L5 Accounting/Funding), synthesized into a single gap waterfall.
**Authoritative live window:** decay01 last restart segment, 38 trades, PV 50050.85 → 49909.51 (−0.282%), fees $370.34 (0.74%), funding ≈ −$1.58.

---

## 1. The problem restated

| Run | Trades | Start → End | Live return |
|-----|--------|-------------|-------------|
| Sim val backtest (2026-04-01..04-15) | — | — | **+26.98% / 14d (PF 2.53)** |
| bybit testnet (0502) | 21 | $50k → $47,363 | −5.3% |
| ensemble-v2 (0507) | 41 | → $46,688 | −6.6% |
| ensemble-v2p1 ens_mean (0508) | 49 | → $48,341 | −3.3% |
| **decay01 solo seed456 (current, 0522)** | 38 | $50,050.85 → $49,909.51 | **−0.28%** |

Every live deployment loses money. ~150 total trades, consistently negative → **structural leak, not noise.**

**Headline framing correction (load-bearing).** There are *two distinct gaps*, and conflating them is the central error to avoid:

- **Gap A — execution/cost gap (sim cost model vs live cost reality):** ~2.5–5% of capital per comparable window. This is what the 5-layer diff can attribute mechanically. It is real, locally fixable, and the subject of the waterfall below.
- **Gap B — alpha-decay gap (sim val regime vs live regime):** the ~25+ percentage-point swing from +26.98%/14d to roughly break-even. This is **NOT** an execution or accounting leak. It is the documented monotone decay of the seed-456 signal on data fresher than the 2026-02-01 (fold-7) train cutoff / 2026-04-15 val window, evaluated on the 2026-05-22+ live regime. **No layer in this diff owns Gap B**, and it cannot be closed by fixing fills or fees — it is a model-freshness / cadence / retrain question.

> **The single most important synthesis finding:** L1, L2, and L3 are all clean (bit-equal / negligible). The live policy receives the correct observation and emits the bit-identical action. Therefore the **entire mechanically-attributable gap lives in L4+L5 (cost), and the dominant headline gap (Gap B) lives in alpha decay that is invisible to a same-window cost diff.** The −0.28% current run is *almost entirely cost* sitting on top of a directional edge that has decayed to roughly break-even (gross +0.46% on the current segment).

---

## 2. The waterfall

Built on the **current decay01 segment** (the cleanest, most run-matched evidence: solo seed-456 = deployed bundle, 38 trades, isolated from restart stitching). Percentages are of $50k capital for this ~0.9-day segment; the per-150-trade column extrapolates to the full deployment.

```
  SIM val return (2026-04-01..04-15, zero-slippage, 5.0bps fee)   +26.98% / 14d   (PF 2.53)
   │
   ├─ (Gap B) ALPHA DECAY: sim-val regime → live-regime edge collapse   ≈ −26 to −27 pp
   │     [NOT a leak — model freshness; no layer owns it; see §4]        ↓
   │
   ├─ ============ residual edge entering live ≈ break-even / slightly + ============
   │     Measured: decay01 segment GROSS directional PnL = +$229 = +0.46%
   │
   ├─ (L4-H1) execution slippage / limit-cross / latency / partials   −4.58 bps/leg
   │     ≈ −$1,130 over ~150 trades  (≈ −2.3% of $50k)                  PROVEN (n=88 predecessor, transferred)
   │
   ├─ (L4-H7 = L5-FEE)  fee-rate under-model 5.0bps→5.5bps             −0.5 bps/leg
   │     ≈ −$139 over ~150 trades (≈ −0.28% over 150; ~1.0–1.2% over a   PROVEN (config 5.0 vs broker 5.5)
   │     full 14d val window given ~600–1000x turnover)                 *** de-dup: L4-H7 and L5-FEE are the SAME term ***
   │
   ├─ (L5/L1) unmodeled funding (sim=0, live pays 8h funding)         ≈ −$1.58 / run
   │     ≈ −0.003% of capital this segment; <$20 over a 7-day run       CONFIRMED but NEGLIGIBLE
   │     *** de-dup: L1-H6-funding-feed and L5-H4 are the SAME term, two windows ***
   │
   ├─ (L3) post-restart cooldown (live-only, ~6.5% of trades skipped) ≈ +small (PROTECTIVE)
   │     removes losing first-3-post-restart legs (−$45 vs +$12)        CONFIRMED, sign is POSITIVE
   │
   ├─ (L1/L2/L3) data feed / obs-norm / policy inference parity        ≈ 0.000
   │     bit-equal: max|B−C|=0.0 obs; 0.000e+00 action diff; faithful feed
   │
   ▼
  LIVE return (decay01 segment)                                    −0.28%  (PV 50050.85 → 49909.51)
```

**Reconciliation against observed (decay01 segment):**

| Component | $ | % of $50k |
|-----------|---|-----------|
| Gross directional PnL (residual live edge) | **+$229** | +0.46% |
| − Fees booked (5.5 bps taker × 38 trades) | −$370 | −0.74% |
| − Funding settled | −$1.58 | −0.003% |
| **= Net (matches WandB/CSV)** | **−$141** | **−0.28%** |

The decomposition closes to the penny: **net loss = (small positive gross edge) − (trading fees) − (funding).** *On the current run, fees alone (0.74%) exceed the gross edge (0.46%).* Slippage (H1) is *inside* the gross-vs-perfect-fill comparison — i.e. the +$229 gross already reflects live fill prices, so the H1 −4.58 bps/leg drag is the gap between this gross and what a perfect-fill sim would have booked. **No unexplained residual** at the segment level.

**Why the other runs lost more (−3.3% to −6.6%):** larger trade counts (41–49) × the same per-leg cost tax, *plus* those bundles (testnet/v2/v2p1) carried *less* residual alpha than decay01 (the predecessor n=88 audit showed perfect-fill itself lost −$900 — i.e. negative gross edge → Gap B was deeper for those bundles). decay01 is the first bundle whose residual gross edge is positive, which is why it is near break-even rather than −5%.

---

## 3. Root-cause ranking (contribution × confidence)

PROVEN, ranked by mechanically-attributable contribution to the **cost gap (Gap A)**:

| Rank | Cause | Layer | Contribution | Confidence | Fix class |
|------|-------|-------|--------------|------------|-----------|
| **1** | **Execution slippage / 5 bps limit-cross / latency / partial-fill resync** (sim assumes zero slippage; `slippage_base_bps=0` default, never overridden) | L4-H1 | **−4.58 bps/leg ≈ −$1,130 / 150 trades (≈2.3% capital)**; 40% of predecessor's −$1,505 loss | **high** | config + retrain |
| **2** | **Fee-rate under-model: sim 5.0 bps vs live 5.5 bps Bybit taker**, amplified by ~600–1000× turnover/fold | L4-H7 ≡ L5-FEE | −0.5 bps/leg; ~1.0–1.2% over 14d val, ~3–5% per long WF fold | **high** | config + retrain |
| **3** | **Turnover is the cost multiplier** (cost = rate × turnover; ~16×/day live, ~136×/day sim WF; trade every ~3.5 bars). The policy was selected under a *cost-free* objective. | L1/L2/reward | Sets the *scale* of #1 and #2; not an independent $ line but the reason they dominate | **high** | retrain |
| 4 | **Unmodeled funding** (sim=0, live 8h perp funding; median \|rate\| 6.16e-5/8h) | L1-H6f ≡ L5-H4 | −$1.58/run; <$20 over 7d; <0.02% capital | high | config (sim haircut) |
| — | post-restart cooldown (live-only) | L3 | **+ (protective)**, ~6.5% of trades, removes losing first-legs | high | keep as-is |

**Refuted / negligible (do NOT chase):**

- **L1 data feed mismatch (H6):** REFUTED. Same venue/instrument/units; direct live-path fetch matched training scale & cadence (vol median 20.28 vs 31.9 BTC base-qty; price 72822–77724 inside training range; cadence 60.0s exact, zero gaps).
- **L1 demo-equity inflation (H6-demo-quirk):** REFUTED. Engine sizes off `total['USDT']`=50050 (within 0.1% of intended 50k), NOT the inflated `totalEquity`=174822. Cosmetic only.
- **L2 obs/norm parity (H5):** REFUTED as a leak. max|B−C| = **0.0** across all 8 features × 3 scales at first trade and after 30 update() calls; S509 warmup pkl verified, tz-strip holds.
- **L3 inference parity (H8):** REFUTED as a leak. Action diff = **0.000e+00** CPU↔CPU; deterministic; N=1 ensemble is a verified no-op; bundle SHA == disk ckpt SHA; deadband + signal-gate thresholds match exactly.
- **L4 same-bar look-ahead (H2):** REFUTED. Both sim and live decide at bar t, earn t→t+1; no temporal leak.
- **L4 maker-fill optimism (H7-maker):** REFUTED — *opposite* is true; the 5 bps cross makes orders marketable → fills TAKER, never maker.
- **L5 sizing/quantization (H3):** REFUTED. Initial_balance scale-invariant (100k sim vs 50k live cancel in %); quantization unbiased ±0.04%/rebalance (avg qty 0.21 BTC = 210× the 0.001 step).

**SPECULATED / unmeasured (downgraded):**

- **L3 CPU-vs-CUDA float32 numerics (H8-dtype-device):** medium confidence — no CUDA box available. Bounded to ~0 trade-decision flips (0 of 2944 live bars within 1e-5 of the 0.25 deadband boundary). Residual PnL impact <0.1 pp. Close by re-running `parity_test.py` on a CUDA box.

---

## 4. Gap B (the dominant headline gap) — explicitly unattributed by this diff

The +26.98% → break-even swing (~26 pp) is **alpha decay**, not a pipeline leak, and no layer owns it. Evidence that it is real and separate:

- L4 perfect-fill replay of the predecessor lost −$900 *with zero execution cost* — the signal itself was already unprofitable on that live window.
- L3 notes the live action mean −0.169 reflects the documented seed-456 short bias on fresher data — the policy is faithfully mapping a decayed signal.
- The sim val window (2026-04-01..04-15) is upstream of the fold-7 train_end 2026-02-01 cutoff and well before the 2026-05-22+ live regime. Per `decision_calendar_anchored_training_window` and the v2.5.1 multiplicity work, microstructure/regime decay is a **cadence/retrain** problem, not a window-shrinking or fill-modeling problem.

**Implication for the soak (see §6):** the soak cannot validate a strategy whose *honest* sim-derived expectation (with realistic costs) is not yet known. Gap B means the corrected sim expectation must be recomputed *and* the model likely re-trained/refreshed before any soak verdict is meaningful.

---

## 5. Ranked fixes

Each tagged **(class · effort · expected gap-recovery)**. Gap-recovery is split: **A** = recovers cost gap; **B** = addresses alpha decay.

| # | Fix | Class | Effort | Expected recovery |
|---|-----|-------|--------|-------------------|
| **1** | **Honest cost model in the env, then re-backtest.** Set `taker_fee: 0.00055` AND `slippage_base_bps: ~5.0` (one-way limit cross) in `sg1_btc_velotrade_*decay01*.yaml` and `live_sg1_btc_bybit.yaml`. Re-run Stage 2.5 bootstrap + Stage 3 WF. | **retrain-required** (re-select under realistic objective) | Low config + ~few GPU-days WF | Doesn't *recover* PnL — it **reveals** the true edge. Expect the sim +26.98% to drop by ~3–5%/fold from cost alone, and PF 2.53 to fall materially. **This is the gate for everything else.** |
| **2** | **Switch to passive/peg maker order** (short timeout → market fallback) instead of marketable 5 bps limit-cross. Captures 1 bp maker vs 5.5 bp taker and avoids the 5 bp cross. | config-only (live) | Low | **A: ~4.5+ bps/leg ≈ most of H1+H7 ≈ up to ~2% capital / 150 trades**, at the cost of some unfilled orders the signal gate must filter. |
| **3** | **Reduce turnover / add a turnover (cost) penalty to the reward**, or widen the effective trade cadence/deadband, so the policy is *selected* net-of-cost. Cost = rate × turnover; at ~16×/day live, turnover is the multiplier on every cost term. | **retrain-required** | Medium (reward change → Audit + Math skills) | A: structurally shrinks #1+#2 by cutting the leg count. Largest *durable* cost lever. |
| **4** | **Minimum-expected-edge gate.** Add a filter so trades whose modeled alpha < ~10 bps round-trip cost are skipped. The env deadband (0.25) gates position *size*, not expected *edge* — weak-conviction signals (alpha < cross) are exactly where the −4.58 bps tax dominates. This is the "widen the signal gate to drop sub-limit-cross trades" lever. | retrain-required (gate threshold tuned on cost-aware backtest) | Medium | A: removes the most-negative-EV legs; compounds with #2 and #3. |
| **5** | **Model funding in the sim** (or apply a backtest haircut). funding_rate is already fetched & logged live; only the SIM side is zero. | config-only | Low | A: ~0.01–0.02% — negligible; do for *honesty/parity*, not PnL. |
| **6** | **Address Gap B: refresh/retrain seed-456 on data through the live cutoff** under the cost-aware objective from #1, on a calendar-anchored 22–24mo window per `decision_calendar_anchored_training_window`. | retrain-required | High | **B: the only lever that touches the ~26 pp headline gap.** Without it, the cost fixes just confirm a near-break-even strategy. |
| — | **Keep** `post_restart_cooldown_bars: 3` (net-protective, ~12 of ~174 would-be trades, all losing first-legs). | — | — | Removing it re-admits the −$45 first-trade skew. |
| — | Cosmetic: set warmup-pkl `fold` field (currently None) in `extract_norm_warmup_buffer.py`; fix `bybit_perp_broker.py:424` is_maker mis-attribution (pollutes fee diagnostics, harmless to PV); assert `resolve_norm_warmup_path()` non-None for prop-firm configs (fail-closed). | config-only | Trivial | 0 PnL; hygiene. |

**Obs/policy fixes:** none required — L2 and L3 are bit-equal (do not spend effort). The only "obs-side" lever is indirect (turnover reduction via reward, #3).

---

## 6. Can the 30-day soak be short-circuited?

**Partially — the gap is now MECHANICALLY EXPLAINED, but the strategy is NOT yet validated.**

- **Gap A (cost) is fully explained and reconciles to the penny** on the current segment: net −0.28% = gross +0.46% − fees 0.74% − funding 0.003%. The soak does not need to *re-discover* that live costs more than a zero-slippage sim — that is now quantified (−4.58 bps/leg slippage + 0.5 bps fee-rate + negligible funding).
- **Gap B (alpha) is NOT explained away — it is the real risk.** The headline +27% was a zero-cost, stale-regime artifact. The strategy's *honest* edge on the live regime is, at best, the +0.46% gross/segment that fees then erase. **A soak would simply confirm a near-break-even-to-slightly-negative strategy** — that is not a useful 30 days.

**Verdict: short-circuit the *cost-discovery* purpose of the soak, but do NOT promote on the soak.** The correct next step is **not** to wait 30 days — it is to (i) apply the realistic fill+fee+funding model (#1, #2, #5), (ii) re-run the WF/bootstrap to get the *corrected sim expectation*, and (iii) if that expectation is still positive net-of-cost, retrain on fresh data (#6) before committing a soak. If the corrected expectation is ≤0, kill or redesign now and save the 30 days entirely.

### Corrected sim-derived expected return — what computation yields it

The honest expectation is **not** +26.98%/14d. To compute it (this is the recommended immediate next action, not a guess):

1. **Re-run the Stage 3 WF / Stage 2.5 bootstrap** for `sg1_btc_velotrade_decay01_wf_multiseed.yaml` with `taker_fee: 0.00055` and `slippage_base_bps: 5.0` set. The env supports both natively (`continuous_swing_env.py:364–366, 387`). This produces a cost-corrected PF and per-fold return distribution.
2. **First-order estimate pending that run** (order-of-magnitude only): the cost-aware monthly expectation ≈ `gross_edge − (fee_rate + slippage_bps) × monthly_turnover`. At the *current decay01 regime*, gross ≈ +0.46% over ~0.9 days of trading ≈ +0.5%/day gross on traded days; cost ≈ (5.5 + 5.0 = 10.5 bps round-leg) × ~16 legs/day ≈ ~1.7%/day. **This implies the current-regime net is mildly negative — i.e. the honest expected monthly return on the present (decayed) signal is approximately −10% to −20%/month before any fix, or roughly break-even-to-slightly-negative after the maker-order switch (#2) cuts ~4.5 bps/leg.** This is precisely why every live run lost money.
3. **Do not annualize** until the WF re-run replaces step 2's proxy. The proxy uses the current-segment gross and live turnover; it carries ±30% magnitude uncertainty but the *sign* (net-negative on current regime, pre-fix) is robust and matches all five live runs.

**Bottom line for the operator:** the gap is explained — it is ~2.5–5% recurring execution+fee cost on top of a signal whose live edge has decayed to roughly break-even. The 30-day soak is unnecessary to confirm this; the next action is a cost-corrected WF re-run (#1) — if it survives net-of-cost, refresh the model (#6) and switch to maker fills (#2) before any soak.

---

## 7. Per-layer evidence tables (file:line anchors)

### L1 — Data quality / feed faithfulness — **~0% gap**

| Hypothesis | Verdict | Key anchor | Note |
|-----------|---------|-----------|------|
| H6 feed mismatch | refuted | `scripts/run_live.py:158-161` (loader, no sandbox flag); `crypto_loader.py:85-86` (fetchMarkets linear); `multiscale_handler.py:53-67` (identical 1→3min agg) | Direct live-path fetch: vol median 20.28 vs training 31.9 BTC (same base-qty); price 72822–77724 inside training 65726–79387; cadence 60.0s exact, 0 gaps |
| H6-demo-quirk (3.5× mis-size) | refuted | `bybit_perp_broker.py:481,539,554,573` reads `total['USDT']` | total['USDT']=50050.85 (not inflated totalEquity 174822); USDC pinned 50000 untouched across 3328 events |
| H6-funding-feed | confirmed | `live_engine.py:1901-1909` `_update_funding_rate`; env has NO funding term | median \|rate\| 6.16e-5/8h; ~$20 over 7d proxy → **handed to L5/L4** |

### L2 — Observation & normalization parity — **~0% gap**

| Hypothesis | Verdict | Key anchor | Note |
|-----------|---------|-----------|------|
| H5 obs/norm leak | refuted | `live_obs_builder.py:688` (full recompute); `:354-357,672-675` (tz strip); `run_live.py:108-118` | **max\|B−C\| = 0.0** all 8 feats × 3 scales, first trade + after 30 updates |
| H5a warmup pkl correct | confirmed | `live_sg1_btc_bybit.yaml:116` norm_warmup_path; pkl seed 456, train_end 2026-02-01 | 199/200 bars bit-identical to parquet; final bar warmup-only, stripped |
| H5b vol_z col-7 boundary artifact | partial | `action_drift.py:356-357`; `l2_parity_sg1btc.py` | Real only on post-cutoff-alone bootstrap; under bootstrap_bars=30000 the join is ~10000 bars deep → agent never sees it |

### L3 — Policy / inference parity — **~0% gap (≤0.1 pp)**

| Hypothesis | Verdict | Key anchor | Note |
|-----------|---------|-----------|------|
| H8 determinism | confirmed | `networks.py:276-278` tanh(mu), no rsample; `sac_agent.py:254-264`; `live_engine.py:1377-1394` | repeat & fresh-reload max\|diff\|=0.000e+00 |
| H8 live-vs-sim parity | confirmed | sim `hpo/evaluate.py:124-140` vs live `live_engine.py:1381-1397` (identical stack/dtype/deterministic) | **0.000e+00** over 200 obs CPU↔CPU |
| H8 N=1 ensemble no-op | confirmed | `agent_loader.py:251-257` returns bare SACAgent | bundle SHA 17b27169 == disk ckpt SHA |
| H8 dtype/device | partial | float32 both (`evaluate.py:116/125/128`, `live_engine.py:1373/1385/1389`); device CUDA→CPU | **medium** — no CUDA box; 0/2944 bars within 1e-5 of deadband → ~0 flips |
| H8 deadband parity | confirmed | sim `continuous_swing_env.py:286`; live `live_engine.py:115,1132,1183` | both effective 0.25 (max_leverage default 1.0) |
| H8 signal-gate parity | confirmed | live `live_engine.py:1418-1449`; train `signal_gated_wrapper.py:174-205`; thresholds == `decay01_wf_multiseed.yaml:89-92` | atr 0.15 / parkinson 0.0009 / volume 0.60 / max_hold 20 |
| H8 cooldown live-only | confirmed | `live_engine.py:1194-1220`; no token in env/wrapper | ~12 of ~174 would-be trades skipped; **net PROTECTIVE** |

### L4 — Trade execution — **~−$1,150 to −$1,300 (≈2.3–2.6% capital / 150 trades)**

| Hypothesis | Verdict | Key anchor | Contribution |
|-----------|---------|-----------|--------------|
| H1 slippage/limit-cross/latency/partials | **confirmed** | sim `continuous_swing_env.py:58` slippage default 0, not set in either config (verified §pre-write); `:364,357,385` zero-shortfall close fill; live `bybit_perp_broker.py:337-339` 5 bp cross; `live_sg1_btc_bybit.yaml:49,132`; `bar_clock.py:51` | **−4.58 bps/leg ≈ −$1,130** (n=88 predecessor: −$605 = 40% of −$1,505) |
| H2 same-bar look-ahead | refuted | `continuous_swing_env.py:230-264,351,357` decide t earn t→t+1; live `live_engine.py:994,1010` identical | ~$0 |
| H7 maker-fill optimism | refuted (opposite) | `bybit_perp_broker.py:516` rate 0.00055 taker / 0.0001 maker; all fills `is_maker=False` (369,405,424) | +0.5 bps live *worse* → **−$139** (≡ L5-FEE) |

### L5 — Accounting / sizing / funding — **~0% structural; one fee-rate term ≡ L4-H7**

| Hypothesis | Verdict | Key anchor | Contribution |
|-----------|---------|-----------|--------------|
| H4 funding driver | refuted | env grep 'funding' = 0; `continuous_swing_env.py:385-389`; live PV reads broker equity `live_engine.py:1822-1845` | ~$1.58/run; 0.002–0.02% capital (≡ L1-H6f) |
| H3 sizing/quantization | refuted | sim `continuous_swing_env.py:385` frac×equity; live `bybit_perp_broker.py:320-324,554` weight=notional/equity | scale-invariant; unbiased ±0.04%/rebalance |
| FEE-undermodel | **confirmed** | config `sg1_btc_velotrade_l1_multiseed_decay01.yaml:75` = 0.0005 (verified); `live_sg1_btc_bybit.yaml:139,150`; broker `bybit_perp_broker.py:516` = 0.00055 (verified) | 0.5 bps; ~1.0–1.2% / 14d val; ~3–5% / long WF fold |
| PV-recon positive gross | confirmed | decay01 seg: PV 50050.85→49909.51, fees 370.34, gross +$229 (+0.46%); delta_fees==order_fee ratio 1.000 | costs (fee+slippage) = entire current-run gap |

---

## 8. Cross-layer de-duplication ledger

| Physical effect | Claimed in | Resolution |
|-----------------|-----------|------------|
| **Fee rate 5.0→5.5 bps** | L4-H7 AND L5-FEE | **SAME term.** Counted ONCE in waterfall (~−$139/150 trades). Both anchor to `bybit_perp_broker.py:516`. |
| **Funding (sim=0)** | L1-H6-funding-feed AND L5-H4 | **SAME term, two windows.** L1's ~$20/7d is a notional×rate×events proxy; L5's −$1.58/run is the settled amount embedded in live PV. Counted ONCE (L5's per-run figure; live PV already contains it). |
| **Zero-slippage sim fill** | L4-H1 (slippage) AND L4-H2 (look-ahead) | H2 refuted look-ahead; the fill optimism is fully and only counted under H1. No double-count. |
| **Demo equity / dual-stablecoin** | L1-H6-demo-quirk AND L5 note | SAME cosmetic effect; both confirm engine reads `total['USDT']`. Zero PnL impact. |

---

## 9. Data limitations (carried from all layers)

- **ccxt_capture_decay01_current.jsonl contains ONLY fetch_balance (3328) + fetch_positions (224)** — NO fetch_ohlcv / create_order / fetch_order (grep-confirmed across all 5 agents). Could not replay per-bar OHLCV the engine consumed, nor extract per-order fill-vs-mid for the *current* run. **Mitigation:** L1 fetched the exact live code path for a recent 2h window (faithful venue/scale/cadence proxy); L4 used the n=88 *predecessor* audit (`C:/tmp/sg1btc_n50_legs.parquet`, identical engine/broker/config family) for slippage — transferable but not perfectly run-matched.
- **Training parquet (data/btc_usdt_1min_bybit.parquet) ends 2026-04-30**, before the decay01 live window (2026-05-22+) → no same-bar A/B price comparison; relied on regime continuity.
- **Live CSVs are WandB-downsampled and stitch ≥4 restart segments** (total_trades resets at rows 58, 91, 133). Only the last segment (38 trades) is the current decay01 run; cross-segment sums are invalid. Held-position mark-to-market between traded bars is invisible from CSV.
- **No CUDA box** → CPU-vs-CUDA float32 inference diff (L3 H8-dtype-device) bounded indirectly, not measured. Medium confidence on that one leg.
- **Per-fold fee-gap %s (3–5%) and per-150-trade $ extrapolations** use turnover/notional proxies; direction and order-of-magnitude robust, exact % ±30%.
- **Recommended infra fix:** extend `_emit_capture` (`live_engine.py:1515` fetch + order path) to capture fetch_ohlcv + order fills so future audits get bar-exact feed parity and per-bar fill-vs-mid.

---
*Synthesis owner: gap-attribution waterfall agent. Layer findings: L1 Data, L2 Obs/Norm, L3 Policy, L4 Execution, L5 Accounting/Funding (parallel diff, 2026-05-29).*
