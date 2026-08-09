# Why Crucible has never mined a promising alpha — root-cause investigation

**Date:** 2026-08-09 · **Scope:** the full lifetime record, every ledger and manifest on disk
**Code state:** worktree `crucible-alpha-mining-investigation-005ae1`, `CRUCIBLE_VERSION = crucible-v11.0`
**Method:** measured, not inferred — ledger queries, code derivation, and a planted-alpha sweep driven
through the production scorer.

---

## Verdict

**Crucible has never mined an alpha because its binding gate cannot detect one at the sample sizes it
mines.** The gate's significance leg requires an orthogonal information ratio of **1.2–1.8** for a
coin-flip chance of promotion on the substrates actually used. The project's own only validated edge —
cross-asset TSMOM at net SR 0.60 — would be promoted **5–7%** of the time. Nothing in the plausible
alpha range is reachable, so *zero discoveries is the modal outcome and the record carries almost no
information about whether alpha was present*.

Two further causes compound it: the hypothesis space is public WQ101 formulas on daily OHLCV (with 39%
of the holdout budget spent on statistical duplicates), and for long stretches the machine reported
`promising=0` while testing nothing at all.

This is a **machine artifact, not a market verdict**. That conclusion agrees with the 2026-07-07 design
audit; what this investigation adds is the exact mechanism, derived from the code and confirmed
empirically, plus the finding that the gate's two economic legs are calibrated **~1,000 years apart**.

> ⭐ **A follow-up sweep (§5b) changes the outlook.** A gate with genuinely useful power — the
> selection-aware cohort MC null — **already exists in-tree and is unreachable**, sitting behind a
> "cheap pre-filter" built from the two seals that have never passed. Measured at a realistic 15% hit
> rate it reaches **25% power at IR 0.30 and 50% at IR 0.50**, versus ~0% for the per-candidate path.
> The decisive quantity is then the **hit rate of the hypothesis bank**, in which power is *linear* —
> not panel depth, in which it is only √. Crucible is not out of road; it has been driving with the
> handbrake on.

---

## 1. The record, measured

Every `orchestrator.db` and `trial_ledger.db` on disk (9 stores):

| quantity | value |
|---|---|
| ticks | 32 (16 mined) |
| substrates | 6 — `cross_asset`, `taiwan`, `us_equity`, `intraday`, `intraday_fx`, overlay variants |
| ledger rows | 559 (445 distinct formulas) |
| rows with holdout metrics | 250 (train diagnostics) |
| rows never scored | **176** — the dedup livelock |
| `rejection_class` populated | **0 / 559** |
| `implied_mde_at_test` populated | **0 / 559** |
| **PROMISING, lifetime** | **0** |
| distinct hypothesis families | **3** (`101alpha`, `altdata`, offspring) |

### The binding statistic has never come within 30% of firing

The ledger's `dsr` / `delta_sr_oos` / `marginal_hlz_t` columns are **train-split diagnostics**, not the
decision — [`loop.py:248-253`](finrl_pro_ds/crucible/agentic/loop.py:248) says so explicitly, and
[`ledger.py:154`](finrl_pro_ds/crucible/ledger.py:154) documents `delta_sr_oos` as "TRAIN-split CPCV
mean ΔSR … **not out-of-sample**". The gate is decided by `corrected_t` in the holdout validation
block. Pooling every holdout score on disk (n=18 from leg forensics, plus the 8 `us_equity` seeds):

| leg | pass rate on the **holdout** |
|---|---|
| `collinearity_pass` | 18/18 (100%) |
| `fragility_pass` | 14/18 (78%) |
| `uplift_pass` (ΔSR ≥ 0.10) | 12/18 (67%) |
| **`t_pass` (t ≥ 2.33)** | **0/18** |
| **`lord_pass`** | **0/18** |

* Max `corrected_t` ever observed, any substrate: **+1.644**. Threshold 2.33. Never once above 1.65.
* On `us_equity` — the best-powered substrate ever built (19.6y × 688 names, holdout 1,726 bars =
  6.85y) — best of 8 pre-registered seeds was **+0.442**, with **7 of 8 negative** holdout ΔSR.

**Significance is the sole binding leg.** Every economic and robustness leg passes routinely. This
corrects a widely-cited framing in the standing record: "uplift passes 170/170, so it is inert" is a
*train-split* statistic conditioned on rows that passed the train screen. On the holdout the uplift leg
rejects a third of candidates and is not inert. The problem was never a broken economic filter.

---

## 2. Dominant cause — the gate's own arithmetic

### 2.1 Derivation

From [`corrected_contract.py:123-153`](finrl_pro_ds/crucible/corrected_contract.py:123), the
Jobson–Korkie–Memmel z is

```
z = (sa − sb) / sqrt(var),   var = [2(1−ρ) + 0.5(sa² + sb² − 2·sa·sb·ρ²)] / n_eff
```

with `sa`, `sb` **per-period** Sharpes (~0.03), so the second bracket term is negligible. Writing the
augmented book as `b_aug = (1−w)·b_base + w·c`, both `(sa − sb)` and `sqrt(2(1−ρ))` are `O(w)`, so the
candidate's weight cancels and

> **z ≈ IR_orth · √years**, hence **MDE@80% power = (2.33 + 0.84)/√years = 3.17/√years**

This **exactly reproduces the project's own independently measured calibration constant** (`3.17/√yrs`,
recorded in `project_crucible_power_units_error_2026_08_03`). The derivation and the empirical sweep
agree, so the mechanism is established rather than asserted.

### 2.2 The two legs of the same AND-gate are ~1,000 years apart

`uplift_min: 0.10` and `t_min: 2.33` sit in the same conjunction in
`configs/crucible_corrected_contract.gates.yaml`. They are equally binding only when
`3.17/√years = 0.10` — that is at **1,006 years of holdout**:

| holdout | significance leg needs | 80%-power MDE | uplift leg needs | ratio |
|---|---|---|---|---|
| 2.8y (`intraday`) | 1.39 | 1.91 | 0.10 | **19×** |
| 3.6y (`cross_asset`) | 1.24 | 1.68 | 0.10 | **17×** |
| 5.1y (`taiwan`) | 1.03 | 1.40 | 0.10 | **14×** |
| 6.9y (`us_equity`) | 0.89 | 1.21 | 0.10 | **12×** |
| 1,006y | 0.073 | 0.100 | 0.10 | 1.0× |

The system's *stated* hypothesis is "find something that adds 0.10 Sharpe." Its *operative* test is
"find something with IR above ~1.2." Everything in between is undiscoverable by construction.

### 2.3 Empirical confirmation — planting alphas of known size

I synthesised candidates with a **known** annualized orthogonal IR against a 3-sleeve base book at
SR 0.50 and pushed them through the unmodified production scorer
(`corrected_contract_fitness`, fresh LORD++ level 0.02187, 40 seeds per cell).

**P(promoted) vs the candidate's true orthogonal IR:**

| substrate (holdout) | IR 0.30 | 0.50 | 0.60 | 0.75 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|
| `intraday_fx` (2.6y) | 0.00 | 0.00 | 0.05 | 0.07 | 0.12 | 0.40 | 0.68 |
| `intraday` (2.8y) | 0.00 | 0.05 | 0.03 | 0.07 | 0.23 | 0.35 | 0.57 |
| `cross_asset` (3.6y) | 0.00 | 0.07 | 0.05 | 0.03 | 0.17 | 0.40 | 0.85 |
| `taiwan` (5.1y) | 0.00 | 0.07 | 0.05 | 0.12 | 0.15 | 0.70 | 0.88 |
| `us_equity` (6.9y) | 0.00 | 0.05 | 0.07 | 0.12 | 0.30 | 0.88 | 1.00 |
| *hypothetical 20y* | 0.00 | 0.03 | 0.28 | 0.42 | 0.72 | 1.00 | 1.00 |

**IR required for a 50% / 80% chance of promotion:**

| substrate | IR@50% | IR@80% |
|---|---|---|
| `intraday_fx` | 1.68 | unreachable |
| `intraday` | 1.83 | unreachable |
| `cross_asset` | 1.61 | 1.94 |
| `taiwan` | 1.36 | 1.79 |
| `us_equity` | **1.18** | **1.44** |

Reference points: TSMOM cross-asset (the project's only validated edge) = **0.60**. Crucible's own
configured "plausible true alpha" ceiling = **0.30–0.50**. Institutional single-signal orthogonal IR ≈
0.5–1.0.

**Crucible is calibrated to find alphas roughly 2–3× better than the best alpha the project has ever
validated — and would miss even that one ~93% of the time.**

The sweep also shows the realized statistic runs at **≈0.75–0.84 × IR·√years**, i.e. ~20% worse than the
ideal, because the dynamic combiner down-weights noisy candidates. The gate is therefore slightly
*less* powerful than its own calibration curve implies.

### 2.4 Expected lifetime discoveries

Taking the machine's own stamped MDEs and 250 scored candidates, assuming a generous 1-in-50 hit rate at
a true ΔSR of 0.30:

```
true alphas among 250 tests = 5.0    P(detect each) = 0.049
E[lifetime discoveries]     = 0.25
```

Under-counted if anything — the 250 tests are heavily correlated (measured realized `n_eff` is 21–43),
which lowers the effective count further. **Observing zero is the expected result.** The record cannot
distinguish a good proposer from a broken one, which is why eight rounds of "0 alpha" carry far less
evidentiary weight than they appear to.

### 2.5 One thing checked and *cleared*

[`corrected_contract.py:128-130`](finrl_pro_ds/crucible/corrected_contract.py:128) justifies the design
with "ρ≈0.99 … so the MARGINAL comparison is precise." Measured ρ in production is **0.800** (range
0.506–0.823), and 0.867 in my synthetic — the docstring is simply wrong, because a 3–5 sleeve base book
necessarily gives a new candidate a large weight. I tested whether this is a power leak: **it is not.**
The ρ term cancels against the numerator (§2.1), and the planted sweep confirms weight-invariance. This
is a documentation error, not a defect. Worth correcting; not worth re-litigating.

---

## 3. Second cause — the hypothesis space

Even a perfectly powered machine would struggle with what is actually being searched.

* **Terminals are price and volume only.** [`grammar.py:65`](finrl_pro_ds/signals/generation/grammar.py:65):
  `open, high, low, close, volume, returns, vwap, adv20, adv30, adv60`. Ten OHLCV-derived series.
* **The default seed bank is 8 formulas.** `_CS_SEED_BANK` in
  [`proposer.py`](finrl_pro_ds/crucible/agentic/proposer.py) carries 8 entries (1 skipped) drawn from
  WorldQuant's 101 Alphas — public since 2015, and the single most heavily mined formula set in
  quantitative finance. 92 of the 101 remain unused.
* **39% of the holdout budget was spent on statistical duplicates.** Because the book is
  scale-invariant, scalar multiples and idempotent re-ranks are the *same hypothesis*:

  ```
  rank(adv60)  ·  rank(scale(adv60))  ·  rank(rank(adv60))          -> corrected_t = +1.4388 (identical)
  ((-1)*ts_rank(rank(adv60),3)) · ((-0.5)*...) · ((-2)*...) · ((-3)*...)  -> +0.2561 (identical)
  ```

  Dedup is on the formula string, so these are ledgered, LORD++-charged, and counted in the file drawer
  as four distinct tests. 18 holdout records reduce to **11 distinct fingerprints**.
* **The surviving candidates are not alpha hypotheses.** The formulas that actually reached the holdout
  gate include `rank(adv60)` and `rank(sum(volume, 60))` — ranking average dollar volume, i.e. a size
  and liquidity proxy.
* **Alt-data can only act as market timing.** The overlay path collapses the cross-section to one scalar
  per day, so a non-price dataset yields `T` observations instead of `T×N`
  ([`grammar.py:93-113`](finrl_pro_ds/signals/generation/grammar.py:93) documents this and the partial
  U3 repair). Notably, the only PROMISING ever recorded in this project came through a per-name
  cross-sectional path — **not** through the miner.

---

## 4. Third cause — the machine frequently tested nothing

Verified in the ledger, and consistent with the last two sessions' findings:

* **176 of 559 rows were never scored.** Pre-registrations wrote a ledger row before the holdout, and
  dedup matched on `candidate_hash` alone, so unscored rows became permanent dedup keys — a 5-week
  livelock in which ticks reported `promising=0 status=OK exit 0` while accepting 0/8 proposals.
* **On `us_equity`, 8/8 pre-registered seeds were culled by the train pre-filter**
  ([`evolve.py:262-285`](finrl_pro_ds/signals/generation/evolve.py:262)), so the binding holdout gate
  adjudicated zero hypotheses while 8 LORD++ tests were charged. `promising=0` was vacuous. (Re-scoring
  the 8 through the real gate gives 0/8 anyway — best t 0.442 — so nothing is retracted.)
* **`rejection_class` is NULL on all 559 rows.** The killed-families learning loop has **never fired in
  the machine's entire operating life**; `killed_families()` has always returned `[]`.

These are fixed or fixable (`crucible-v12.0` addresses the train-cull; note it is **not present in this
worktree**, which is at v11.0). They matter less than power, but they are why the historical record is
partly unreadable: *a tick that tests nothing and a tick that finds nothing produce identical logs.*

---

## 5. What is **not** the cause

Worth stating, because each has been proposed at some point:

* **Not a proposer-quality problem.** With per-test power of 3–5%, no proposer could have succeeded. The
  LLM-proposer work was correctly found not to be the bottleneck.
* **Not the FDR/LORD++ accounting.** `lord_pass` is 0/18, but it never binds independently — at every
  depth the `t_min` leg is stricter than the LORD++ leg (e.g. at 5.8y: IR 0.964 vs 0.834).
* **Not the CRU-2 anti-oracle moat.** Verified leak-free by three prior audits; the view fix removed rows
  rather than adding columns.
* **Not "the market has no alpha."** The record has no power to support that claim in either direction.

---

## 5b. Follow-up: can the cohort path rescue it? — MEASURED

After the above, the obvious lever was the built-but-disabled cohort/ensemble path
(`configs/crucible_cohort.gates.yaml`, `enabled: false`), which amplifies a weak signal by
`ensemble_multiplier(m, ρ̄) = √(m/(1+(m−1)ρ̄))`. Two sweeps were run through the real
`greedy_decorrelated_admission` and the real scorer.

### Finding 1 — routing a *selected* cohort through the corrected contract is invalid

**Measured null FPR = 1.000.** Selecting the top-30 of a 100-candidate pool by standalone Sharpe and
scoring the combination through `corrected_contract_fitness` passes on **pure noise**, with mean
`corrected_t` **4.88**. The corrected contract is a *single-hypothesis* test with no file-drawer
deflation (it dropped `dsr_aug`, the funnel's only such term), so select-then-combine is pure
multiplicity laundering.

> This **falsifies my own prior recommendation** to "route the cohort through the corrected contract."
> It is exactly what the cohort's order-statistic `SR*_cohort` benchmark and selection-aware MC null
> exist to prevent, and why `offspring_policy: prereg_only` is the shipped default.

### Finding 2 — with pre-registered members and no selection, the null is clean

FPR **0.000** at K = 1, 12, 30 on both panel lengths. The defensible configuration is: declare the
members in advance, combine all of them, test once.

### Finding 3 — the decisive quantity is the HIT RATE, not the panel or the gate

For K pre-registered members of which `n_true` carry a real edge of IR δ:

> **IR_cohort = δ · n_true / √K** — so combining *dilutes* below a single signal when the hit rate is low.

P(pass), K=30, measured (30 seeds/cell):

| panel | per-signal IR | 3/30 | 6/30 | 10/30 | 15/30 | 30/30 |
|---|---|---|---|---|---|---|
| holdout 6.85y | 0.30 | 0.00 | 0.00 | 0.03 | 0.10 | 0.73 |
| **FULL 19.56y** | 0.20 | 0.00 | 0.03 | 0.07 | 0.10 | **0.83** |
| **FULL 19.56y** | 0.30 | 0.00 | 0.03 | 0.10 | 0.60 | **1.00** |
| **FULL 19.56y** | 0.50 | 0.03 | 0.13 | 0.70 | 1.00 | 1.00 |

Head-to-head at δ=0.30 on the full panel: **single signal 0.00 · cohort 6/30 0.00 · cohort 30/30 1.00.**

**Break-even hit rate is ~40–50%.** Above it the cohort is transformative; below it, it is worse than
useless. The one real measurement available — the 8 `us_equity` WQ101 seeds, 7 of 8 with negative
holdout ΔSR — implies a hit rate near 0.1 on that bank, so **the cohort does not rescue WQ101.**

### The strategic inversion this implies

Since `IR_cohort = δ · √K · (hit rate)`, power is **linear in hit rate** and only **√** in everything
else. Doubling the hit rate is worth as much as quadrupling the data. The binding constraint was never
statistics or panel depth — it is **the quality of the hypothesis bank**. That inverts the framing that
has driven the last several sessions of Crucible work.

### Finding 4 — the selection-aware MC null DOES work, and it is the one path with real power

Selection is not the enemy; *unpriced* selection is. `cohort_mc.mc_null_pvalue` re-runs the identical
admission inside every bootstrap replicate (H0 by demeaning candidates only, base resampled jointly),
so selection is paid for. Size is already green in-repo
(`tests/signals/test_generation_cohort_mc.py`, 9/9, incl. the BLOCKER-1 anti-conservative regression),
so all compute went to power.

Measured — pool 40, K≤20, ρ≤0.10, 10y panel, B=49, α=0.05, 8 seeds:

| scenario | hit rate | **pass rate** | med p | real signals admitted |
|---|---|---|---|---|
| NULL (0 real) | 0.00 | **0.00** | 0.450 | 0.0 |
| 6/40 real, IR 0.30 | 0.15 | **0.25** | 0.120 | 5.0 of 6 |
| 6/40 real, IR 0.50 | 0.15 | **0.50** | 0.050 | 6.0 of 6 |
| 20/40 real, IR 0.30 | 0.50 | **0.75** | 0.020 | 14 of 20 |

**At a realistic 15% hit rate the MC null reaches 25% power at IR 0.30 and 50% at IR 0.50** — against
~0.00 for the per-candidate path and 0.03 for the no-selection prereg cohort at the same hit rate. The
mechanism is visible in the last column: the admission actually *finds* the planted signals (5 of 6,
then 6 of 6). Selection recovers most of what a low hit rate costs, precisely because the MC null lets
you select without invalidating the test.

Caveats: B=49 (the config wants ≥1000), 8 seeds so ±0.15-ish, and planted candidates are mutually
independent Gaussians — real alphas correlate with each other and with the base, which will reduce
effective K. Indicative, not precise.

### ⭐ Finding 5 — that gate is UNREACHABLE: a third instance of the project's signature defect

[`cohort_eval.py:320`](finrl_pro_ds/signals/generation/cohort_eval.py:320):

```python
if not ev.passes_analytic_floor:
    return _verdict(ev, None, (float("nan"), False), pch, n_culled, "LOGGED")   # hard early return
mc = mc_null_pvalue(...)                                                        # line 327 — never reached
```

`passes_analytic_floor` ([`cohort.py:441-444`](finrl_pro_ds/signals/generation/cohort.py:441)) requires
`dsr_book >= promising_dsr (0.90)` **and** `cohort_hlz_t >= cohort_hlz_t_min (3.0)` — **the same two
seals that pass 0/170 across the lifetime record.** It is documented as a "CHEAP PRE-FILTER — NOT the
binding gate" in three separate places, but structurally it is an absolute gate.

This is the **third occurrence of one architectural bug**: *a cheap pre-filter that is stricter than
the gate it protects.*

| # | where | consequence |
|---|---|---|
| 1 | train re-applied the final 6-way gate (pre-v6.0) | holdout gate never executed across 403 trials |
| 2 | train cheap pre-filter culled 8/8 pre-registrations | `us_equity` holdout adjudicated zero (fixed in v12.0) |
| 3 | **analytic floor gates the MC null** | **the only high-power gate in the system is unreachable** |

The cohort path was never dead. It was **never reachable.** Nothing has ever been measured through it,
because `enabled: false` and, had it been enabled, the floor would have returned `LOGGED` first.

---

## 6. What follows

Ordered by measured effect. These are findings, not a mandate — every gate change below is an operator
call under the CLAUDE.md rule.

**The headline changed during the investigation.** The per-candidate path is unfixable by threshold
tuning, but a gate with real power exists in-tree, unreachable behind a dead pre-filter. The single
highest-value action is no longer "get more data" — it is **make the MC null reachable and feed it a
better hypothesis bank.**

0. **Unblock the MC null (highest value, smallest diff).** Make `passes_analytic_floor` advisory —
   which is what it is documented to be in three places — or rebuild it from legs that are not the two
   0/170 seals. Without this, nothing else about the cohort path matters. Pairs with `enabled: true`.
   ⚠ Do **not** substitute the corrected contract here: measured null FPR 1.000 (§5b Finding 1).
1. **Hit rate is the lever, not depth.** `IR_cohort = δ·√K·(hit rate)` — power is **linear** in hit rate
   and only **√** in data, K, and years. Doubling hit rate ≈ quadrupling the panel. Spend effort on
   economically-motivated hypotheses over less-crowded data, not on more WQ101 permutations. This
   inverts the framing that has driven recent Crucible work.
2. **`us_equity` is the only substrate where the arithmetic closes — and only barely.** Measured breadth
   gives IC 0.020 → IR 1.457 against an MDE of 1.312. Every other substrate is closed for power reasons
   no threshold change can repair. A second `us_equity` round is *not* power-blocked, but it needs a new
   pre-registered bank (the 8 default seeds are spent) and the v12 train-cull fix.
3. **Reconcile the two economic legs, or drop one.** A conjunction whose members are calibrated 1,000
   years apart is not a gate, it is a single test with decorative company. Either raise `uplift_min` to
   what the significance leg actually implies at that depth, or state plainly that the machine tests
   significance alone.
4. **Dedup on statistical fingerprint, not formula string.** Recovers ~39% of the search and holdout
   budget for free, and stops LORD++ being charged multiple times for one hypothesis.
5. **Expand the terminal set beyond OHLCV, and finish the per-name `(T,N)` alt-data route.** The single
   PROMISING this project ever recorded came through a per-name cross-sectional path. The miner still
   cannot reach one on most panels.
6. **Fix the observability that made the record unreadable** — `rejection_class`, `implied_mde_at_test`,
   and `n_holdout_tested` should never be silently NULL. Two consecutive sessions lost time to an
   unreadable `promising=0`.

---

## Appendix — reproduction

Analysis scripts (scratchpad, not committed):
`lifetime.py` (ledger census) · `legs.py` (leg consistency) · `power.py` (derivation + expected
discoveries) · `plant.py`, `curve.py` (planted-alpha sweep through the production scorer) ·
`holdout.py` (holdout-gate aggregation).

Primary data: `results/crucible_orchestrator*/**/{orchestrator,trial_ledger}.db`,
`results/crucible_orchestrator_{fx,xa_h1}/leg_forensics.json`,
`results/crucible_prereg_forensics/us_equity_prereg_forensics.json`.
