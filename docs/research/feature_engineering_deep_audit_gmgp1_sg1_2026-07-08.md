# Feature-Engineering Deep Audit — GMGP1 + SG-1 (XAUUSD/GC + BTC)

**Date:** 2026-07-08 · **Session:** S553-cont-125 · **Tier:** 2 (deep, not diff-scoped)
**Scope:** fractional differentiation, normalization/standardization (EMA-Z, rolling stats), LEAK-1 split isolation, LEAK-2 multi-scale causality, warmup handling, sim↔live feature parity, per-config consistency.
**Workstreams:** GMGP1 15m (GC futures, XAUUSD CFD, BTC) + SG-1 3m (XAUUSD, BTC). Gold/XAUUSD ensembles are LIVE (paper) — live-path findings graded accordingly.
**Method note:** the 13-agent finder+skeptic workflow (`wf_ca589bdb-ad2`) died on the Claude subscription session limit with 0 completed agents; all six pillars were ground-truthed directly in the main loop (same fallback as cont-124), with numeric verification scripts and the full tripwire-test suite executed locally.

---

## Executive summary

**The training-side feature pipeline is clean.** Every deliberate audit angle on the sim path — EMA-Z causality, the X2 coarse-bar mapping, LEAK-1 cutoff wiring across all six entry points, warmup carry semantics, PRISM lookup, earn-timing — re-derived correct, and the tripwire tests guarding them are real negative tests (15/15 pass, including a non-vacuity witness). The residual defects live where the 2026-05-29 audit's lesson predicts: **in the live serving path and in shared code no active workstream exercises.**

The dominant cross-cutting fault is **train↔serve skew introduced by the live engine's buffer/private-state handling**, not by the feature math itself:

1. **FE-01 (S1, live-active, FIXED):** the live 1-min buffer duplicated re-fetched minutes every bar (declared dedup tracker `_last_1min_ts_ms` was never wired), inflating resampled volume ≈2× on every closed base bar *except* the newest — so `volume_z` on exactly the decision bar was persistently biased low vs training, and the SG-1 signal gate reads `volume_z`.
2. **FE-08 (S2, shared code, FIXED):** `_fractional_diff` applied the FFD weights time-reversed (double-reversal through `np.convolve`) — numerically proven by impulse response. Not in the GMGP1/SG-1 path (no fracdiff in V7 features), but it is *the* fractional-differentiation implementation in the repo.
3. **FE-02 (S2 latent, GUARDED):** live private state skipped the B1 leverage normalization the training env applies. Inert at today's `leverage: 1` configs; would have silently broken the first leverage-trained bundle. Live now mirrors the division and the engine refuses `max_leverage ≠ 1` until the execution path is leverage-wired.

On the user's three named concerns: **fractional differentiation** — one real bug found and corrected in shared code; its *absence* from V7 features is not a defect (evidence below). **Normalization** — verified causal and healthy; no correction needed on the math, one boundary off-by-one corrected in the live warmup extractor (FE-03). **Standardization** — consistent across sim and live by shared code; one config-plumbing gap closed (FE-04).

---

## Findings

Severity: S1 = live loss/leakage/wrong training signal · S2 = materially weakens robustness · S3 = rigor/hygiene · S4 = minor or positive. All fixes in this session are **checkpoint-preserving**: none changes any observation a trained model sees in sim/backtest.

### F1 — Stationarity & fractional differentiation

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-08 | S2 | **CONFIRMED → FIXED** | `_fractional_diff` (`finrl_pro_ds/crypto/features/crypto_features.py:369`) fed `_get_ffd_weights`' REVERSED array into `np.convolve`, which time-reverses its kernel internally. Net effect: weights applied mirror-imaged — unit weight w₀ on the *oldest* bar of the window, fractional tail on the newest. Impulse response measured `[-0.064, -0.12, -0.4, +1.0]` where correct FFD gives `[+1.0, -0.4, -0.12, -0.064]`. Causal (no look-ahead) but wrong math: with production `window=100` the output is a ~99-bar-stale mirrored filter. Consumers all in retired/shelved workstreams (Sync-1H/2H SigBoost `funding_cumsum_ffd`, FundingArb `ffd_basis/oi`, macro `dxy_ffd`) — no live impact, but any revival would have inherited it. **Fix:** natural-order weights into convolve; pinned by `tests/crypto/test_fractional_diff.py` (impulse response, loop-reference equivalence, no-look-ahead). |
| FE-09 | S4 | POSITIVE | V7's absence of fracdiff is **not a correctness gap**. Measured on real GC data (338k 1-min bars, 2025): `log_return` is memoryless (lag-1 −0.30/−0.40 microstructure bounce at 15m/3m, ~0 beyond), while `close_z` (SymLog→EMA-Z(120)→tanh) retains autocorr 0.95→0.31 out to 60 bars with stable σ≈0.41 at both timeframes — the feature set already spans the memory–stationarity spectrum FFD interpolates (fully-differenced channel + bounded locally-stationary level channel). Introducing a d≈0.4 FFD price channel is a **RESEARCH** upgrade (obs-distribution change → full protocol-v2 lifecycle), not a correction. |
| FE-10 | S4 | POSITIVE | Obs vector is bounded/stationary end-to-end: `log_ret` clipped ±0.1 (clip never binds on GC 3m/15m), `atr_norm` = ATR14/EMA50−1 clipped ±2, `parkinson` clipped [0, 0.1], z-features tanh-bounded, private dims [pos, pnl_proxy, time sin/cos, atr_ratio] all normalized (`continuous_swing_env.py:466-522`). Tanh saturation healthy: 0.08–1.5% of samples |out|>0.96 across z-features (worst: volume_z 15m at 1.5%) — no information-destroying saturation. |

### F2 — Normalization & standardization math

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-11 | S4 | REFUTED (positive) | Suspected 2-bar look-ahead in `_ema_zscore_tanh`'s pre-warmup backfill (`multiscale_handler.py:39-48`) is **not real**: t=0 output is identically 0 (numerator zero by construction) and perturbing later values changes no earlier output (verified numerically). The normalizer is strictly causal: `ewm(span).mean/std().shift(1)` — stats through t−1 normalize bar t. Constant series → no NaN, z=0. Pandas EWM variance is the proper non-negative estimator (no EMA-of-squares pitfall). |
| FE-12 | S4 | POSITIVE | No duplicated-implementation drift in the active path: live reuses the *same* `_compute_scale_features/_ema_zscore_tanh/_symlog/_resample_ohlcv` functions by import (`live_obs_builder.py:27-31`). The `feature_engineering.py` copy (`:152`) belongs to V5/V6 legacy envs (DeepScalper/SwingScalper) — not in the GMGP1/SG-1 path (verified via `env_factory.make_env` v7 branch). |
| FE-05 | S3 | CONFIRMED (no code change) | `norm_span=120` is a single default across timeframes: 120 base bars = **6h at 3m (SG-1) vs 30h at 15m (GMGP1)** of normalization horizon. Train==live (both 120 ✓) so no skew, but the horizon was never chosen per workstream and is not in HPO space. RESEARCH note, not a defect. |

### F3 — LEAK-1 split isolation & warmup

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-13 | S4 | POSITIVE | LEAK-1 wiring verified at **every** feature-building entry point: train env = no cutoff over train-only range; HPO eval env = `norm_cutoff_date=val_start_date` (`hpo/objective.py:395-400`); pipeline backtests = val/test starts (`run_full_pipeline.py:905,918`); WF/ensemble evals = `val_end`/`train_end` (`sg1_xauusd_ensemble_eval.py:200,1028,1057`; `sg1_btc_velotrade_ensemble_eval.py:195`); vector envs forward it (`env_factory.py:210-214`). Warmup carry is past→future only (200 pre-cutoff bars seed the post-cutoff EMA restart, `multiscale_handler.py:161-175`), and the freeze-and-restart design makes val/test/live share one normalization regime by construction. |
| FE-03 | S3 | **CONFIRMED → FIXED** | Warmup-extractor boundary off-by-one: `extract_norm_warmup_buffer.py:73` took bars `<= train_end` while training's cutoff mask (`>= cutoff`, `multiscale_handler.py:301-303`) makes the implicit warmup strictly-before-cutoff. Midnight is a bar boundary on every scale grid, so the shipped live warmup was one bar late — a small EMA-Z transient decaying over ~span bars, but it broke the "bit-equivalent" contract and no test covered the extractor's convention. **Fix:** strict `<`. The deployed gmgp1-xauusd pkl predates the fix; its transient decayed long ago, regenerate at next bundle roll. |
| FE-14 | S4 | NOTE | `RollingWindowSplitter` defaults `buffer_days=0` (no embargo between train/val/test). In this RL setting obs-window overlap into the prior split is causal and labels don't overlap, so this is hygiene, not leakage; the knob exists for operators who want purge gaps. |

### F4 — LEAK-2 multi-scale causality

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-15 | S4 | POSITIVE | X2 coarse→base mapping re-derived correct: coarse bars stamped label='left' close at `t+scale`; the map (`multiscale_handler.py:361-371`, searchsorted on `base_ts − scale_ns`) admits only coarse bars closed **at or before the base bar's START** — conservative by up to one base bar (a coarse bar closing simultaneously with the decision bar's close is excluded), i.e. zero leak with ≤1-bar staleness, and **live mirrors it bit-identically** (`live_obs_builder.py:470-502`, same int64-ns searchsorted). PRISM daily lookup serves strictly-prior day (`multiscale_handler.py:443-460`), disabled in all audited configs anyway. `get_lookahead_volatility` (`:543`) is consumed only by V5/V6 aux-loss paths (`swing_scalper_env.py:384`, `deep_scalper_env.py:938`) — never reaches V7 obs. |
| FE-16 | S4 | POSITIVE | The tripwire suite is genuinely adversarial, not hollow: `tests/data/test_multiscale_causality.py` asserts closed-bar causality per scale AND self-checks non-vacuity (`leak_would_differ > 0`); `tests/test_continuous_swing_causality.py` pins base-scale earn-timing with an echo policy (~50% vs ~100% separation); `tests/data/test_compute_features_with_warmup.py` pins warmup bit-equivalence. All 15 pass. |
| FE-17 | S4 | NOTE | `multiscale_crypto_handler.py` (Sync-2H, retired) imports the shared `_compute_scale_features/_resample_ohlcv` — the X2-fixed code — rather than forking it; its per-asset index mapping was not re-derived here (retired workstream, out of scope). |

### F5 — Sim↔live feature parity

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-01 | **S1** | **CONFIRMED → FIXED** | Live 1-min buffer duplication: `_fetch_new_bars` re-fetches from the last BASE-bar start +1min (`live_engine.py:1566-1571`), overlapping the buffer by up to base_scale−1 already-held minutes every bar; `LiveObsBuilder.update()` concatenated with **no dedup** (`live_obs_builder.py:719-721` pre-fix), and the declared dedup tracker `_last_1min_ts_ms` (`live_engine.py:474`, "FIX AUD-ENG-05") was **never wired** — declared-but-not-wired class. `_resample_ohlcv` sums volume per bin → every closed base bar except the newest accumulated ≈2× volume (1.93× at 15m base, 1.67× at 3m) while the decision bar stayed 1× → after SymLog→EMA-Z the mean adapts to the inflated level and the **newest bar's `volume_z` is persistently biased low** — on exactly the bar the agent acts on, and `volume_z` (idx 7) also feeds the SG-1 signal gate (`live_engine.py:1463,1478`). OHLC/ATR unaffected (duplicate values collapse under first/max/min/last). Invisible to the variance drift detector (a one-of-eight-features bias doesn't trip 0.01×/100× ratios). Affects all V7 live stacks (gmgp1-gold IB, gmgp1-xauusd cTrader — live paper now; BTC/DXTrade when running). Bias magnitude estimated (≈ −log2/σ_symlog-vol post-EMA-Z), mechanism proven by code + regression test. **Fix:** timestamp dedup keep='last' + stable sort in `update()` (and bootstrap path); overlap now degrades into harmless gap-healing, and re-fetched finalized candles replace partial versions. Pinned by `tests/crypto/test_live_obs_builder_buffer.py` (engine-pattern overlap replay → unique buffer, clean volumes, end-state feature parity with a fresh builder). Dead tracker removed. |
| FE-02 | S2 | **CONFIRMED → FIXED (guarded)** | Live private state skipped B1 leverage normalization: training divides position and pnl_proxy by `max_leverage` (`continuous_swing_env.py:477-487`) and scales actions by it (`:267`); live passed raw position (`live_obs_builder.py:844-851` pre-fix) and `_predict` clips to ±1 unscaled (`live_engine.py:1457`). Inert today (every live config runs `leverage: 1`) but leverage HPO configs exist (`sg1_xauusd_leverage_narrow_hpo.yaml`, `dhpo_sg1_xauusd_leverage.yaml`) — the first leverage bundle would have shipped with out-of-distribution private dims AND silently under-sized execution. **Fix:** `LiveObsBuilder(max_leverage=…)` mirrors the division (no-op at 1.0 — parity test proves both L=1.0 and L=2.0 match the env formula), and the engine **fails closed** on `max_leverage ≠ 1.0` until action scaling + position accounting are leverage-wired. |
| FE-18 | S4 | POSITIVE | Everything else in the live path holds parity: same feature functions by import; S509 norm-warmup pkl reproduces training's freeze-and-restart EMA-Z with an equivalence test; scale-suffixed pkl naming disambiguates shared checkpoint dirs (`resolve_norm_warmup_path`); OANDA loader drops incomplete candles (`oanda_data_loader.py:246`) and dedups per fetch; time encoding uses the base-bar timestamp, not wall clock (AUD-C04); window construction + summary stats mirror the handler exactly; S546 calendar expansion handles weekend-closed assets; per-base-bar cadence via BarClock keeps ATR-buffer semantics aligned. |
| FE-19 | S3 | NOTE (no change) | Legacy fallback when no warmup pkl resolves is **fail-open**: `resolve_norm_warmup_path` logs a warning and the builder silently uses rolling EMA-Z — the exact S509 skew the pkl exists to prevent. gmgp1-xauusd wires the pkl explicitly; `live_gmgp1_btc_bybit.yaml` shows no `norm_warmup_path` (falsified workstream, paper only). Operator decision whether to hard-fail; recommend `require_normalizers`-style gate at next live-stack touch. Also noted: env seeds its ATR buffer empty each episode while live pre-seeds 200 bars (mild first-200-bar `atr_ratio` distribution difference favoring live stability), and gmgp1-btc trains on Bitfinex data while live feeds Bybit (cross-venue level shifts largely absorbed by EMA-Z; workstream shelved). |

### F6 — Per-workstream config & obs-space consistency

| ID | Sev | Verdict | Finding |
|----|-----|---------|---------|
| FE-04 | S3 | **CONFIRMED → FIXED** | `features_per_scale` was declared in every config but **never consumed by the handler** — `_load_data` called `_compute_scale_features` without `n_features` (hardcoded default 8), while the env defaulted to 7 (`continuous_swing_env.py:47`) and the SAC trainer to 7 (`sac_trainer.py:54`): three independent defaults agreeing only by convention, no cross-check. All five audited configs declare 8/8/8 consistently (GC, XAUUSD 15m, BTC 15m, SG-1 XAUUSD, SG-1 BTC — wiring map verified through `env_factory` v7 branch). **Fix:** handler now consumes `features.features_per_scale`; env falls back to the forwarded features block before the legacy 7; `make_env` v7 fails fast when `features:`/`env:`/`network.scale_encoder.input_size` disagree (guard verified to fire). Zero behavior change for every existing config. |
| FE-20 | S4 | NOTE | Cross-workstream deltas are deliberate where they matter (scales 3/15/60 vs 15/60/240; `gap_detection: true` on SG-1 XAUUSD and live gold configs, off for 24/7 BTC) with one inconsistency: training config `gmgp1_xauusd_sac_15min.yaml` doesn't set `gap_detection` while its live twin does (S468) — env-mechanics scope, flagged for the next GMGP1-xauusd retrain config roll. `features.asset_class` is inert at train time (consumed only by LiveObsBuilder calendar expansion) — harmless declaration. |

---

## Roadmap

**NOW — done this session (all checkpoint-preserving, live-side or shared-code only):**

| # | Fix | Files |
|---|-----|-------|
| 1 | FE-01 buffer dedup (S1) | `live_obs_builder.py` (`update` + `_init_from_dataframe`), `live_engine.py` (dead tracker note), `tests/crypto/test_live_obs_builder_buffer.py` |
| 2 | FE-08 FFD weight order (S2) | `crypto_features.py`, `tests/crypto/test_fractional_diff.py` |
| 3 | FE-02 leverage parity + fail-closed guard (S2) | `live_obs_builder.py`, `live_engine.py` |
| 4 | FE-03 warmup extractor strict-before-cutoff (S3) | `scripts/extract_norm_warmup_buffer.py` |
| 5 | FE-04 features_per_scale plumbing + tri-declaration guard (S3) | `multiscale_handler.py`, `continuous_swing_env.py`, `env_factory.py` |

Deploy note: fixes 1–3 reach the live stacks at the **next image rebuild** (baked Dockerfiles — Docker Desktop bind mounts don't propagate). Until then the running gold/xauusd paper containers carry the FE-01 volume_z bias; it has been present since inception, so current paper P&L already reflects it — rebuilding upgrades feature fidelity, it does not invalidate the paper experiment.

**NEXT (needs an operator/stage action, not a code defect):**
- Regenerate `norm_warmup_*.pkl` at the next bundle roll (FE-03 boundary now exact).
- Decide fail-open vs fail-closed for missing warmup pkl (FE-19) at next live-stack touch.
- Align `gap_detection` between `gmgp1_xauusd_sac_15min.yaml` and its live twin at next retrain (FE-20).

**RESEARCH (obs-distribution changes → full protocol-v2 lifecycle; queue behind stronger evidence):**
- FFD price channel (d≈0.3–0.5) as a 9th/replacement feature — FE-09 shows the current set already spans the memory spectrum, so expect marginal gain; pre-register before testing. **→ CLOSED 2026-07-12 (measured redundant): pre-registered + Stage-0 CPU probe ran same day (`docs/research/fe_obs_channel_preregistration_2026-07-12.md` §7). NO-GO — 0/24 primary passers; max ΔIC +0.0032 « MDE 0.010 on window-complete gold; median negative. FE-09 confirmed. Zero GPU spent.**
- Per-timeframe `norm_span` study (FE-05): 6h vs 30h normalization horizons were never chosen deliberately. **→ CLOSED 2026-07-12 (measured redundant): same probe Part B — no `norm_span∈{60,240,480}` beats 120 by the MDE on any primary gold cell.**
- Day-of-week encoding for session assets (time sin/cos is minute-of-day only).

## Verification log

- Numeric: FFD impulse-response + loop-reference equivalence (bug proven pre-fix, correctness proven post-fix); EMA-Z causality perturbation test (leak refuted); constant-series NaN check; GC-2025 saturation/autocorr measurements (scratchpad `verify_fe_math.py`).
- Tests: 15/15 pre-existing tripwires green before AND after fixes (`test_multiscale_causality`, `test_compute_features_with_warmup`, `test_continuous_swing_causality`, `test_live_obs_parity`); 7 new regression tests green; full touched-surface run (tests/data, tests/crypto, continuous-swing suites, summary-stats, sensitivity-audit) — 208 tests, 0 failures; ruff clean on all touched files; `make_env` mismatch guard fire verified.
- Counts: 18 findings — 6 confirmed defects (1 S1, 2 S2, 3 S3): 4 fixed outright, 1 fixed + fail-closed-guarded (FE-02), 1 accepted as a research note (FE-05 norm_span); 1 suspicion refuted with evidence (FE-11); 11 positives/notes credited.
