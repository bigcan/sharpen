# TAILWIND-v1 — Deep Lifecycle Audit (Tier-2)

**Date:** 2026-08-18 · **Workstream:** `tailwind_v1` / `tailwind_v1_challenge` (linear cross-asset
TSMOM + BAB, 18 free-data ETFs, monthly rebalance) · **Method:** finder-per-pillar (subagents,
TAILWIND-scoped) + operator adjudication and independent re-verification of every load-bearing
claim · **Trigger:** operator request, ahead of any prop-firm challenge attempt.

**Supersedes nothing.** Extends `docs/research/tailwind_v1_deep_lifecycle_audit_2026-07-01.md`,
whose dominant finding (P7-01, "gates grade the WRONG BOOK") is now **closed on the book axis**
and **re-opened, larger, on the mechanics axis**.

---

## VERDICT: BLOCK

A fee-paying challenge attempt is not authorised. The block is **not** a doubt about the edge —
the signal layer is clean, causality is genuinely guarded, and the selection is not overfit
(PBO 0.0009). The block is that **the book that was certified is not the book that would trade**,
and the machinery that would catch that does not run.

Capital gate after this audit:

| # | Item | State |
|---|---|---|
| 1 | Forward-path render | ⚠ CLEAR **by exactly zero margin**, and it was reading FLAGS until today (P10-01) |
| 2 | Sizing reconciliation | ✅ measured — but see P3-01a, which re-explains what it measured |
| 3 | Tier-2 audit | ❌ **this document — BLOCK** |
| 4 | Protocol-v2 wiring | ✅ 6 FAILs → 0 — but 33 of 48 declared gates are inert (P10-06) |
| 5 | Operator go-ahead | ⏳ open |

---

## 1. The finding that decides it

### P3-01 · The certifying evidence and the wired executor are different books · **S1**

Every deploy-facing number — DSR 0.896, PBO 0.0009, P(pass) 0.746, RENDER_CLEAR — is computed on
the **research basis**: `xsec_momentum_falsification.backtest()`, monthly ffilled weights, 1-bar
lag, 2 bps one-way turnover cost, **no gross cap, no slippage, no borrow**. The executor
(`MultiAssetAllocatorEnv` via `TwoSleeveExecutor`) applies four mechanisms the research basis does
not, and they are not scale factors — no sizing lever can reconcile them.

| Configuration | net SR |
|---|---|
| Research basis (certifies everything) | **0.601** |
| + gross cap 3.0 + daily vol-rescale + `min_trade_pct` | 0.538 |
| + the env's actual 2-bar execution lag | 0.433 |
| + executor slippage (1 bp base; impact ≈ 0 at $100k) | **0.422** |

**≈ −0.18 SR, −30%.** The finder's reconstruction reproduces the research leg exactly (0.601 ==
the pinned `LINEAR_CORE_NET_SHARPE`), which validates the method; the executor leg omits
fixed-entry-notional accounting and the 2-sleeve α combine, so treat the magnitude as indicative
and the direction and rank order as solid.

DSR 0.896 already fails the 0.95 own-capital gate. P(pass) and the render are **monotone in the
book's Sharpe**. So all four certifying numbers move *against* the book once the executor's
mechanics apply, and **none has ever been recomputed on the executor path.**

### P3-01a · The gross cap binds ~always, so the executor is CONSTANT-GROSS, not vol-targeted · **S1**

**Independently verified by the operator on the real 18-ETF panel** (not finder-reported):

```
config: target_vol_asset=0.10  lev_cap=2.0  max_gross_exposure=3.0
pre-cap gross: mean 11.96x  median 12.03x  min 2.87x  max 21.52x
bars where pre-cap gross > 3.0 : 99.13%  (228/230 rebalances)
per-asset lev_cap binds on      :  5.0%  of (bar, asset) cells
```

The daily scalar `3.0 / gross_t` divides out the per-asset vol target **and** the breadth
modulation. The wired book therefore normalises gross to exactly 3.0 every month regardless of
market volatility.

**Failure scenario.** In a 2008-type regime, per-asset vol targeting is supposed to de-lever.
Here it cannot: pre-cap gross must fall ~4× (from ~12 to 3) before the cap releases, which
happened on 2 of 230 rebalances in 20 years. The risk profile the certifying numbers describe —
a vol-targeted book — is not the risk profile that would trade.

**This also re-explains `docs/research/tailwind_sizing_reconciliation_2026-08-01.md`.** That
document measured `target_vol_asset` doubling → +8% vol, `lev_cap` 1→3 → ~nothing,
`max_gross` → linear, and attributed it to "real ETF vols saturating the ratio." The measured
cause is different: `lev_cap` binds on only 5% of cells, while the **gross cap binds ~always**.
The reconciliation's *mapping* (executor vol ≈ 0.0334 × `max_gross_exposure`) is unaffected and
still correct; its *mechanism* prose is not.

### P3-02 · The keystone parity test disables exactly the four mechanisms that differ · **S1**

**Verified by direct read** — `tests/integration/test_allocator_baseline_parity.py:80-95`:

```python
vol_ary=np.full((T, n), 0.10),        # identity vol-scaling -> no daily rescale
volume_ary=np.full((T, n), 1e15),     # ~0 impact slippage
max_gross_exposure=1e9,               # cap DISABLED
slippage_base_bps=0.0, slippage_impact_bps=0.0, min_trade_pct=0.0,
...
idx = min(k + 1, T - 1)               # execution lag hand-compensated
```

The claim "the env reproduces the validated linear core at SR 0.60" holds **only** in a
configuration that switches off production. It is not evidence about the wired book. The
end-to-end TAILWIND executor tests (`tests/paper/test_tailwind_executor.py`) run on synthetic GBM
with parity 0 by construction — correctly documented, but it means **there is no real-data
end-to-end test of the executor with production mechanisms on.**

### P3-03 · Borrow / financing / cash rate modelled nowhere · **S2**

`carry_ary` is hard-zeroed on all three tailwind paths (`cross_asset_loader.py:407, :573, :627`),
so `env._apply_carry` is a no-op; the research basis has no carry term at all. Measured executor
momentum sleeve: mean long 1.93×, mean short 1.07×, net +0.86× — a persistent short notional and
~0.93× of financed leverage, both free.

| Short-leg borrow | Ann. drag | SR drag @ challenge sizing (10% vol) |
|---|---|---|
| 25 bp | 0.27% | 0.027 |
| 50 bp | 0.53% | 0.053 |
| 100 bp | 1.07% | 0.107 |

USO, SLV, DBA, FXY and EEM have each exceeded 100 bp within the sample. Collateral interest and
short rebate are also unmodelled and cut the other way, so the **sign is ambiguous** — but the
magnitude is material at challenge sizing and it is absent from *both* mechanisms, hence
invisible to any research-vs-executor reconciliation.

### P3-04 · Cost asymmetry — real, but the smallest of the three wedges · **S3**

Provenance verified end to end: `COST_MODELS["standard_2bps"]` → `run_book._net_standard` →
`build_momentum_net` / `build_defensive_net` → `pf.risk_parity` → consumed by the DSR/PBO audit,
the challenge simulator, and the forward-path render. Every certifying number is
**frictionless + 2 bps turnover**. The executor charges 2 bps fee + 1 bp base + 5 bps ×
participation.

The impact term is **inert at prop-firm size**: at $100k a 0.02 weight delta is $2,000 against
ETF dollar volumes of $10M–$30B ⇒ participation 1e-7…2e-4 ⇒ impact ≤ 0.001 bps. So the
"conservative impact model" buys no conservatism. Base slippage is the whole gap: ≈ −0.010 SR.

**On P3-07 from the July audit (BAB gross-cap dilutes momentum): NOT CONFIRMED.** Clipping is
proportional within a sleeve and over the union, and the α's are computed from post-cap sleeve
returns, so risk-parity stays self-consistent. There is no momentum-vs-BAB asymmetry. The real
defect is larger and different — the cap costs ~0.063 SR by destroying the risk-targeting signal
on **both** sleeves equally.

---

## 2. Data integrity (P1)

### P1-01 · DATA-CLEAN is absent from the path that produced every capital-facing number · **S2**

`mom.get_prices()` (`xsec_momentum_falsification.py:51-58`) is `yf.download → raw["Close"] →
parquet`. No `clean_ohlcv`, no manifest, no `.bak`, no hash. Grep for
`clean_ohlcv|fetch_and_clean|manifest` across all six tailwind research scripts: **zero hits**.

**The July audit's entire P1 pillar (P1-01…P1-09) was written against `cross_asset_loader.py`** —
the executor's path, which has gates — and never looked at `mom.get_prices()`. So the pillar that
was supposed to certify data integrity described a path that produced none of the numbers.

Structural aggravator: the research panel is **close-only**, so `detect_outliers` (needs OHLC) and
the flat-OHLC-spike branch are **not runnable on it at all**. DATA-CLEAN is not merely un-run
here; it cannot be run without a refetch, and a refetch changes the vintage (P1-03).

**Severity capped at S2 because the data was checked and is clean** (finder-measured on the frozen
panel): 0 mid-series NaN, 0 zero/negative prices, 0 flat-close runs ≥5 bars out of 91,162
observations, and the 3 extreme moves (USO 2020-03-09/-04-21, SLV 2026-01-30) all real events that
persist rather than revert. What is missing is the **gate**, not the correctness — but nothing on
this path would report it if that stopped being true, and the project has already been burned once
by exactly that (S106, gold).

### P1-02 · Survivorship is silently inherited and never acknowledged · **S2**

`xsec_momentum_falsification.py:30-35` hardcodes 18 ETFs; `breadth_expansion.py:40-46` a 32-ETF
superset. Every ticker in both is alive as of 2026-08. Grep for
`survivorship|survivor|delist|selection bias|inception` across all tailwind docs, scripts and
configs: **zero hits**.

This is the time gradient that killed BALLAST. Mitigating and worth stating: cross-asset TSMOM on
broad-beta index wrappers is far less exposed than an equity panel — the mechanism is time-series
trend on an asset class, and a dead wrapper is typically replaced by a live one tracking the same
exposure. The effect is likely small. But it is **undeclared and unquantified**, and it belongs as
a stated caveat on the DSR rather than an unexamined assumption.

### P1-03 · Zero provenance on either cache · **S2**

Neither `get_prices()` nor `extended_prices()` records fetch time, raw sha256, universe, or the
`START`/`END` under which the pull was made. `auto_adjust=True` is a point-in-time transform, so
the same (ticker, date) returns different values after any later dividend or split — a published
number is not reproducible from the artifact.

Two specifics: `get_prices()` is **unconditionally cache-first** and ignores `START`, `END` **and
the universe**, so *adding a ticker silently yields the old panel*; and `extended_prices()`
defaults to `refetch=True`, so every Stage-4 re-run pulls a new vintage under identical metadata.

### P1-04 · Two caches, one claimed basis, unenforced · **S3**

The frozen `results/xsec_momentum/prices_daily.parquet` (ends 2026-05-29) feeds DSR/PBO/render;
`results/tailwind_v1/prices_daily_extended.parquet` (to 2026-08-14) feeds Stage-4 OOS. Measured max
relative difference on the 5,133 overlapping rows: **≤ 1.9e-06** — immaterial, because both were
fetched ~3 hours apart. The claim happens to be true; it is not enforced, and the gap widens
monotonically with every dividend.

---

## 3. Leakage (P2)

**The signal layer is clean.** No LEAK-2 violation in `cross_asset_signals.compute`,
`defensive_signals.defensive_conviction`, the loader array builders, the env, or the **production**
risk-parity combine. Every causality claim was verified by direct inspection rather than trusted
from a docstring, and the tripwires are wired at five real production call sites and provably bite
(11 tests executed, including mutation negatives).

### P2-01 · The full-sample `risk_parity` constant — mechanism real, magnitude refuted · **S3**

The finder claimed the full-sample vol constants contaminate the DSR headline by +0.094 SR, on the
grounds that in a two-sleeve combine the ratio `k_mom/k_bab` is not a scale factor but **the
allocation**, chosen ex post. **The mechanism is correct. The magnitude is an artifact.**

**Operator re-verification** — the finder's causal arm used an *uncapped* scalar, so early in the
sample `0.10/trailing_vol` explodes and the "causally-constructible book" runs unbounded leverage:

| causal `lev_cap` | causal SR | inflation vs shipped |
|---|---|---|
| 1.0 | 0.7614 | **−0.020** (causal better) |
| 2.0 (the config's value) | 0.7432 | −0.002 |
| 3.0 | 0.7417 | −0.000 |
| 5.0 | 0.7338 | +0.008 |
| *uncapped* | *0.639* | *+0.094 ← finder's figure* |

At any realistic cap the inflation is ~0.00. **DSR 0.896 stands.**

**But the operator's earlier sweep conclusion was right for the wrong reason.** That sweep
justified 24 clean call sites with "Sharpe is leverage-invariant" — a *per-series* property that
does **not** cover a two-sleeve combine. The correct justification is empirical: the vol ratio
between these two sleeves is stable over time, so the causal ratio reproduces the full-sample one
to within 0.002. That is a fact about this data, not a theorem. **On a sleeve pair with an unstable
vol ratio the same code leaks materially.**

### P2-02 · The audit's only BAB leak probe tests decay, not look-ahead · **S2**

`audit_tailwind_book.py:96-108` (ATTACK 1) compares `_net_lag(w_bab, rets, 1)` against `lag=0` and
asserts same-day must not beat causal. But `defensive_conviction` at `t` reads `close ≤ t-1` and
`vol_scaled_weights` at `t` reads `rets ≤ t-1`, so **`lag=0` is itself causally executable** —
decide at close(t−1), earn `rets[t]`. The probe measures one-day signal decay. A genuine
look-ahead probe needs `lag=-1`. Green tripwire that never touches the line where a bug would live.

### P2-03 · DSR is fed a padded sample length · **S3**

`backtest()` builds the net series over the full index from 2006 while `rebal` only begins after
warmup: the momentum net series carries **331 leading exact-zero days out of 5,133**. Sharpe is
*deflated* by this (conservative), but the `T` handed to `deflated_sharpe_ratio` and
`block_bootstrap_sharpe_ci` is 5,133 when only 4,802 days are live. DSR is increasing in T, so the
confidence attached to the observed Sharpe is overstated by ~7% of sample.

### P2-04 · The keystone parity test silently SKIPS on a fresh checkout · **S2**

`tests/features/test_cross_asset_signals.py:137` is `@pytest.mark.skipif(not PRICE_CACHE.exists())`
and `results/` is gitignored. `test_baseline_weight_reproduces_linear_core_sharpe` — the only test
tying production `baseline_weight` to the validated 0.601 — reports **skipped, not failed**, in CI
or on any clone. A drift in the production signal lands green.

### P2-05 · `assert_causal` asserts only the first half of the sample · **S3**

`perturb_frac=0.5`, with all comparisons restricted to `date <= perturb_date`, and no caller varies
it. **No row in the second half is ever asserted.** A uniform-in-`t` leak is caught at the boundary,
but a *conditional* leak — fires only near a month-end, only with an interior NaN, only where
`min_periods` fallbacks engage — can sit in the unasserted tail indefinitely. A loop over 3–5
`perturb_frac` values including ~0.95 costs almost nothing.

### P2-06 · `ParityHarness._safe_monthend_conviction` truncation is a no-op · **S3**

`conv_window[:t+1][-1]` ≡ `conviction_ary[t]`. **Covered:** the rebalance calendar, hold/ffill
assembly, index-offset bugs — with real executing negatives. **Not covered:** anything inside
`compute` / `defensive_conviction`, because the conviction array is a fixed precomputed input. A
full-sample normalisation, a `bfill`, a forward-reaching window or an X2-class mis-map would all
yield drift exactly 0. `two_sleeve.py:31-33` claims the recompute catches "a look-ahead bug in
either sleeve's forward assembly" — true for assembly, false for signal construction.

---

## 4. Gate wiring and live consistency (P10)

### P10-01 · The capital-gating render was reading RENDER_FLAGS · **S1 · FIXED `0d75f2fa`**

**Operator-introduced regression.** The render's A2 check reads `prop_firm.vol_multiplier` against
the book's **native** 6.92% vol and requires `|native × mult − effective_vol_ann| < 0.005`:

```
1.5  -> 0.1038  |d| 0.0038  PASS (by 0.0012)
1.0  -> 0.0692  |d| 0.0308  FAIL          <- state on disk for ~20h
1.45 -> 0.1003  |d| 0.0003  PASS          <- restored
```

In `41fbec8b` it was changed 1.5 → 1.0 to match the *simulator's* frame (grid swept on an
already-normalised 10% base). Both numbers are true of different denominators; A2 forces one. The
render was last run **before** that edit, so `RENDER_CLEAR` was asserted in the config header, in
`MEMORY.md`, in the parent memory and in PR #8 while the wired config actually failed. The comment
added to justify the mismatch was itself reconciliation-by-prose — precisely what A2 exists to
forbid. Same orphan class `41fbec8b` was opened to close: `resolve_gates_path()` de-orphaned the
*gates pointer*, but `prop_firm.vol_multiplier` is a **second unguarded input to the same verdict**.

Fixed, re-rendered `RENDER_CLEAR`, and guarded by
`test_vol_multiplier_agrees_between_config_and_gates`. **Still open:** no test executes A2
end-to-end, so a change to `native_vol` or the 0.005 tolerance remains unguarded.

### P10-02 · `paper_soak` has never once been evaluated for this book · **S1**

`evaluate_paper_soak_gates` raises `KeyError('rolling_sharpe_floor')` on **both** challenge gates
files — `soak_metrics.py:242` and `:255` dereference it on both branches of the
`months_observed < min_months` fork, and neither challenge gates file has a `performance:` block
(the own-capital file does). So `run_cross_asset_paper_validation.py:156` — the only runner that
could score this book — crashes, and 21 numeric gates (parity ×5, risk ×4, drift ×5, horizon ×2)
are **un-evaluatable**, not merely unwired. The green test suite misses it because
`tests/paper/test_tailwind_executor.py` loads the **own-capital** gates file, not the challenge one
the config points at.

### P10-03 · Every §8.2/§8.3 live control is declared-only · **S2 today, S1 at promotion**

`ActionDriftTracker` is instantiated only at `crypto/live/live_engine.py:140`; the TAILWIND
executor has zero references to drift, tracker or kill. `safety.kill_file` /
`flatten_on_kill_file` have no reader on this path — the watchdog reads a hardcoded
`/tmp/finrl_live_kill` from inside a container, and TAILWIND has no container. Yet
`validate_config --stage paper-deploy` returns green on all of it, and
`test_paper_deploy_stage_has_no_failures` now pins that green as a regression assertion.

**A green gate over a control with no runtime consumer is a false attestation, which is worse than
an absent one, because it terminates the search.**

### P10-04 · The drift baseline's shape is unreadable by the only existing loader · **S2**

`tailwind_drift_baseline.py` emits `eval_distribution_by_sleeve`; `live_engine._init_drift_tracker`
recognises `ensemble_eval_distribution`, `eval_distribution_by_seed`, `eval_distribution` and then
falls through to LOG_ONLY. A naive wiring logs one startup WARNING and **no CRIT can ever fire**.
Secondary: `tsmom_signal` takes only 4 values, so `deadband_frac` is structurally 0.000 on the
momentum sleeve and 4 of 8 KL bins are empty — `deadband_frac_*` and `action_kl_*` are not
meaningful quantities there even once a consumer exists.

### P10-05 · No runner exits non-zero on a blocking verdict · **S2**

`run_cross_asset_paper_validation.py:200` returns 0 unconditionally after logging
`overall_status`; the render's `__main__` calls `main()` bare, so `RENDER_FLAGS` exits 0. Any CI
job or `&&` chain treats both capital-gating artifacts as green. **This is how the stale
RENDER_CLEAR survived.**

### P10-06 · Gate inventory · **15 wired · 33 inert (21 un-evaluatable)**

Of 48 declared gates in `configs/tailwind_v1_challenge_v2.gates.yaml`, the wired 15 are the
forward-path-render block (6) and the parts of `challenge_pass_gate` the render consumes (9). The
`gates:` block (4) is inert — it targets the RL WF pipeline and the dynamic combiner, neither of
which TAILWIND uses. `paper_soak` (21) is inert behind P10-02. `overfitting_informational` is
inert by design. `paper_soak.consumed_by: [paper_executor, live_monitor]` names a `live_monitor`
that **does not exist**.

### P10-07 · `calibrated_for_sleeves` is a name match, not a calibration check · **S3**

`soak_metrics.py:164-170` compares two sorted lists of **strings**. It answers "did the executor
run the same sleeve names?" — a real guard against the +VRP mis-scoping case — but it cannot
answer "were 8.0/4.0 derived from a render of this book." Any pair of numbers passes. Compounding:
`risk.derivation`, the field a reader treats as authoritative, still cites a render **at 15% vol**
— the sizing that was rejected on three legs.

### P10-08 · Five more validator checks share the inline-vs-overlay defect · **S3**

`check_drift_safemode_gates:884` reads `cfg["gates"]` directly while `check_gates_block:420`
resolves `ensemble.gates_file`. The same inline-only pattern appears in `check_retrain_gate`,
`check_v23_agreement_decay_gates`, `check_wf`, `check_l1_multiseed` and `check_hpo`. Two
structural notes on the helper itself: `_load_ensemble_gates_overlay` reads **only** the standalone
file's top-level `gates:` key — so `paper_soak`, `challenge_pass_gate` and `forward_path_render`
are invisible to *every* validator check, which is why P10-02 is invisible to `validate_config` —
and it resolves a relative `gates_file` against `Path.cwd()`, silently returning inline gates on a
miss.

---

## 5. Recent-OOS (P8) — reclassified

X4 is built (`4e51cb27`); **finding P8-10 from the July audit is closed**. The artifact is
leak-free — risk-parity scalars fitted on 5,133 pre-cutoff bars through 2026-05-29 and applied
forward, mutation-tested — and declares `deploy_gating: false` honestly.

**But the pillar is not "pending", it is STRUCTURALLY UNDERPOWERED.** The holdout is 53 days = 2
rebalances: `HOLD` at PF 1.1335 vs a 1.169 baseline, `power_sufficient: false`,
p(Sharpe<0) = 0.32.

Confirming a 0.73-Sharpe book out of sample at 95% requires **≈9.1 years** of clean holdout:

| clean holdout | rebalances | t on OOS Sharpe |
|---|---|---|
| 0.2y (today) | 2 | 0.30 |
| 0.5y (Dec 2026) | 6 | 0.46 |
| 3y | 36 | 1.13 |
| **9.1y** | 109 | **1.96** |

**Waiting does not fix this**, and an earlier recommendation in this workstream to defer the Tier-2
until ~2026-12 was wrong on that basis. The promotion decision must rest on the other pillars or
not be taken. The FTMO compliance leg is equally unsettleable at this horizon: it returns
`passes: null` because a monthly book cannot produce 4 active trading days in 2 rebalances — a
window artifact, correctly reported as indeterminate.

---

## 6. Positives — verified, not assumed

- **One-basis discipline holds.** `audit_tailwind_book._mom_frame` is the shared basis for the
  DSR/PBO audit, the render, the simulator and the drift baseline. July's P7-01 is closed on the
  book axis.
- **Signal-layer causality is genuinely correct** under direct inspection at every claim, with
  `assert_causal` wired at five production call sites and mutation negatives that provably bite.
- **The current-bar perturbation leg exists at every layer** — the exact blind spot that let the X2
  leak survive.
- **SHORT-ACCT is clean and golden-trajectory tested on the REAL env**, not a mock: no
  `notional_debt` anywhere on the allocator path, long/short symmetry, close-realisation and
  position flip all covered.
- **The env is one bar MORE conservative than the research backtest** — a realism positive, even
  though it is also an unbooked haircut (P3-01).
- **P2-01 (partial trailing month) is correctly handled on the deployed path**, documented at the
  point of danger and covered by an executing test.
- **Structural de-orphaning works** — `resolve_gates_path()` refuses to guess and raises.
- **`causal_vol_lev_cap` held at 3.0 for the right reason**, with the category error documented
  rather than resolved by picking the number that yields CLEAR.
- **Zero/halted volume ⇒ maximum impact**, not free.
- **The data gate fails closed with exit 2** — the one place a non-zero exit exists.
- **The panel is empirically clean:** 0 zeros, 0 mid-NaN, 0 flat runs ≥5 across 91,162 obs.

---

## 7. Minimum before any capital

1. **Recompute DSR / PBO / P(pass) / render on the executor's mechanics** — or state explicitly, in
   every artifact, that the certifying numbers describe a different book (P3-01).
2. **Decide whether constant-gross is intended.** If not, `max_gross_exposure` or
   `target_vol_asset` needs re-deriving so vol targeting actually binds (P3-01a).
3. **Model borrow / financing / cash rate** on both mechanisms (P3-03).
4. **A real-data end-to-end executor test with production mechanisms ON** (P3-02).
5. Add a `performance:` block to both challenge gates files, then actually run the paper
   validation (P10-02).
6. Non-zero exits on `RENDER_FLAGS` and `overall_status != PASS` (P10-05).
7. Wire or explicitly retire the drift/kill controls — do not let `validate_config` attest to them
   (P10-03/04).
8. `lag=-1` leak probe for BAB (P2-02); un-skip the keystone parity test (P2-04); vary
   `perturb_frac` (P2-05); un-pad `T` in the DSR call (P2-03).
9. Route the six inline-only validator checks through the overlay, and extend the overlay past the
   top-level `gates:` key (P10-08).

---

## 8. Coverage and limitations of THIS audit

**Pillars run:** P1, P2, P3, P10 as finders; P8 adjudicated directly.

**Structurally N/A, not unaudited:** P5 (Training & HPO), P6 (L1 multiseed), P7 (walk-forward),
P9 (ensemble formation) — TAILWIND is a frozen linear rule that is never trained, has no seeds and
no folds. P4 (reward design) is near-N/A: the deployed path is the linear core, and the DSR reward
matters only if the RL overlay is ever enabled.

**Dropped:** P11 (external SOTA benchmark). Its highest-value controls — deflated Sharpe,
PBO/CSCV, purged CPCV — are already implemented and produced DSR 0.896 / PBO 0.0009, so it is
advisory rather than decisive for a go/no-go.

**Method caveat.** Finder magnitudes were re-verified by the operator wherever they were
load-bearing, and **two finder claims did not survive**: P2's +0.094 DSR contamination (an
uncapped-leverage artifact) and July's P3-07 BAB dilution concern (clipping is proportional). P3's
SR decomposition is a faithful pandas transcription that omits fixed-entry-notional accounting;
its direction and rank order are solid, its magnitude indicative.

**Operator errors corrected in the record during this audit:** the A2 regression (P10-01); the
caller-sweep justification (P2-01); and the recommendation to defer this audit to December (§5).

**The first-run failure mode is worth recording:** three of four finders died on a session limit
before emitting anything. Relaunching with explicit budget discipline — read named files in
priority order, *report even if partial* — recovered all three. A finder that dies silent is worse
than one that reports half.
