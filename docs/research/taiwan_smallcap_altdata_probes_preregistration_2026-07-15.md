# PRE-REGISTRATION — Taiwan Small/Mid-Cap Alt-Data Cross-Sectional Probes

**Date:** 2026-07-15 · Session S553-cont · Branch `June2026`
**Author sets this record BEFORE any of the three alt-data channels is fetched or looked at.**
**Pathway:** parallel probe pathway with its own gates file — the frozen `signal_eval` funnel and
its verdicts are untouched (**CRU-1 MINOR**, lockbox/cohort ADR precedent).

> This is the §5 deliverable of `crucible_independent_audit_report_2026-07-14.md`:
> *"run low-N, pre-registered, mechanism-driven single-hypothesis probes on the full sample …
> implemented as a parallel pathway with its own gates file."* It is **not** a re-opening of the
> GP mass-miner (the sealed discovery instrument) and **not** a mining campaign — it is exactly
> **three** hypotheses, each with a stated mechanism and a pre-committed sign, fixed here in git
> before results exist. Multiplicity is `n_trials = 3` and nothing more.

---

## 0. Why this, why now (the two corrections it makes)

1. **Corrects the June mirage's WRONG SLICE.** The 2026-06-28 test (`taiwan_xsec_momentum_eval.py`)
   ran 12-1/6-1/3-1 price momentum on the **45-name 0050 large-cap** universe. First pass IC-IR
   **0.191 "PROMISING"** was a **size-confound mirage** — mandatory log-ADV size-neutralization
   collapsed it to **0.039** (survivorship was the minor culprit, 0.039→0.042; SIZE was the
   inflator — the "momentum" was the TSMC/semiconductor mega-cap trend). NO-GO. But that slice is
   the **opposite** of what the "less-efficient Taiwan" thesis predicts: the thesis lives in the
   **small/mid-cap** tier that big money and analyst coverage ignore, not in the most-liquid
   large-caps. So the thesis was neither cleanly supported nor killed — it was tested in the wrong
   place. This probe tests it **where it was actually predicted.**

2. **Makes the FinMind subscription earn its keep on DATA BREADTH, not better statistics.** Audit
   §5 honesty clause: *"the power problem is ultimately a **data problem** (breadth,
   frequency-with-capacity, niche datasets)."* §6.3: *"target different data, not better
   statistics … capacity-constrained niches per the small-operator reframe."* The three channels
   below are **niche, retail-behaviour datasets** on a **wider cross-section** (N≈200 vs 45) — the
   two axes (new mechanism × more names) the free daily price feed cannot supply.

**Falsification framing (read before trusting any GO).** The pre-registered expectation is that
NO-GO is a likely and fully informative outcome. Audit §5 MDE80 ≈ **0.7–0.9** on these substrates:
even a *real* 0.3–0.5 alt-data edge may be **undetectable** on ~11–13 years of daily data. A null
result here **closes** the "less-efficient Taiwan small-cap alt-data" thesis honestly; it does not
mean the mechanism is absent, only that it is not extractable at this power/cost.

---

## 1. Instruments, data, scope

- **Market universe (rank pool):** all TWSE **and** TPEx common stocks (`TaiwanStockInfo`, filtered
  to common equity — 4-digit ids, ETFs/warrants/preferred/DR excluded). Prices via FinMind
  `TaiwanStockPrice` (**RAW, unadjusted**, `adjusted:false`, LEAK-2). Dividends folded in causally
  (forward add-back, as `taiwan_panel_loader._apply_causal_total_return`) for the return LABEL only.
- **Delisting:** `TaiwanStockDelisting` — delisted names are included for their live window so a
  name drops out of the cross-section when it dies, not before (survivorship handled by the same
  PIT membership mask, **upper-bound caveat** below).
- **The three alt-data channels (the niche datasets):**
  | Channel | FinMind dataset | Native cadence | Public-availability lag (LEAK-2) |
  |---|---|---|---|
  | Month-revenue | `TaiwanStockMonthRevenue` | monthly | mandatory disclosure by the **10th of the following month** → signal becomes visible on **day 10 of month(revenue_month)+1**, never the revenue month-end |
  | Margin / short-sale | `TaiwanStockMarginPurchaseShortSale` | daily (post-close) | **T+1 trading day** (published after the close of day T) |
  | Shareholding distribution (集保 TDCC) | `TaiwanStockHoldingSharesPer` | weekly (per-Friday) | **as-of `date` + 6 calendar days** buffer (TDCC releases the following week) |
- **Sample window:** full available FinMind history per channel, intersected. Month-revenue and
  margin run to ~2005+; the 集保 distribution series is shorter (~2016+). Each probe uses its own
  channel's full history (the harness reports the effective `min_days`); **no window is chosen to
  flatter a result.**
- **Currency / cost:** TWD notional; the **0.30% securities-transaction SELL tax** is baked into the
  Taiwan cost trio (below). Prices RAW/unadjusted; the H1 causal-total-return add-back is applied to
  the return label only, never to a feature.

**Survivorship / PIT caveat (honest UPPER BOUND).** The free FinMind feed lists currently-trading
tickers; `TaiwanStockDelisting` completeness is not independently verified, and the cap-rank uses
best-available shares (below). Every IC reported is therefore an **upper bound**
(`panel.meta.survivorship_free = False`, printed by the harness). A survivorship-free PIT
reconstruction (TEJ-grade) is a **promotion gate**, not part of this probe.

---

## 2. Universe construction (LOCKED) — cap-rank 51–250 small/mid-cap band

At each **month-end** rebalance `t`:

1. Rank the common-stock pool by **market cap** = `close_t × shares_t`, where `shares_t` = total
   registered shares from the `TaiwanStockHoldingSharesPer` **"total" (合計)** tier, forward-filled
   from the last *available* weekly update (causal) and lagged per §1. (Shares from the 集保 total
   are used **only** for the cap-rank cut; they are independent of the log-ADV size proxy the
   harness neutralizes on, so membership and neutralization do not share a quantity.)
2. **Exclude the top-50** (the 0050 large-cap tier — the slice the June mirage already killed).
3. Take **ranks 51–250** → the small/mid-cap band (~200 names/month).
4. **Liquidity floor:** trailing 60-day TWD ADV ≥ `min_adv_twd` (gates file) — a tradeable
   cross-section.
5. **History floor:** ≥ 250 trading days of prior bars (drops IPO-noise names).
6. **Membership is a causal (T,N) mask**; a name is active only on days it satisfies 1–5 with data
   stamped ≤ `t`. Delisted names go inactive when their series ends.

`universe` id (locked, part of every spec hash): **`twse_smallcap_caprank_51_250`**.

---

## 3. The three probes (LOCKED — mechanism, construction, sign, hash)

All three: `family = "altdata"`, `horizons = (1,5,10,21,63)`, `primary_horizon = 21` (monthly
rebalance, **low turnover** — the only regime where the 0.30% sell tax does not dominate),
`neutralization = ("winsor","zscore","sector","size")`. **`"size"` is MANDATORY and non-negotiable
— it is the exact step that killed the June mirage** (`evaluate_signal` reads neutralization from
the spec, so size cannot be silently dropped). Signs are pre-committed; a signal whose realized IC
has the *opposite* sign is a FAIL, not a sign-flip opportunity.

### P1 — Month-revenue momentum · `tw_smallcap_mom_rev` · sign **+1** · hash `60680e61ff85`
- **Mechanism:** under thin analyst coverage, small/mid-cap monthly revenue announcements (mandatory
  by the 10th) are **underreacted to** → post-announcement drift. High YoY revenue growth predicts
  cross-sectional continuation.
- **Signal (causal):** `yoy = revenue[m] / revenue[m-12] - 1`, stamped as available on **day 10 of
  m+1**, forward-filled across the month, then aligned to trading days. Names with < 13 months of
  revenue history or a non-positive base are NaN.
- **Expected sign:** **+1** (long high YoY growth).

### P2 — Margin / short-sale crowding · `tw_smallcap_margin_crowd` · sign **−1** · hash `e0a4c719bfe0`
- **Mechanism:** rising **retail margin-financing balance** = leverage crowding into a name;
  crowded, tax-disadvantaged retail longs subsequently **underperform** (deleveraging / fragility).
  A contrarian crowding signal, not a momentum one.
- **Signal (causal):** `Δ21d of (MarginPurchaseTodayBalance / shares_t)` — the 21-day change in
  margin utilization (financing balance as a fraction of registered shares), stamped **T+1**. (Short
  balance is carried in the panel for a pre-registered robustness read only — NOT a second trial.)
- **Expected sign:** **−1** (long LOW / falling margin utilization; short the crowded).

### P3 — Big-holder shareholding concentration · `tw_smallcap_holder_conc` · sign **+1** · hash `1be26f02ee6a`
- **Mechanism:** the 集保 (TDCC) weekly distribution splits each name's register into holder-size
  tiers. A **rising share held by big holders (>400 board lots)** = informed accumulation by
  concentrated hands; **outperformance** follows. (The mirror, dispersion to many tiny retail
  holders, is distribution.)
- **Signal (causal):** `Δ4w of (percent of shares held in the >400-lot tiers)`, stamped as-of
  `date + 6d`, forward-filled to trading days. Tier boundary (400 lots) is fixed here, not tuned.
- **Expected sign:** **+1** (long rising big-holder concentration).

> **Spec-drift tripwire.** `tests/research/test_taiwan_smallcap_altdata.py` asserts each constructed
> `SignalSpec.content_hash()` equals the value above. Any post-hoc edit to a name/hypothesis/sign/
> horizon/neutralization/universe/cost_profile changes the hash and **fails the test** — the
> anti-p-hacking seal.

---

## 4. Metrics & Gates (FROZEN) — `configs/taiwan_smallcap_altdata.gates.yaml`

Own gates file, **CRU-1-registered** `gates_hash` = **`0ccf6dd584f0`** (pinned in
`tests/crucible/test_version.py::test_smallcap_altdata_probe_gates_hash_registered`; the frozen
funnel moats `519158fa1450` / `22a18172be1a` are untouched — CRU-1 MINOR). Scoring is the **existing cross-sectional IC funnel**
(`evaluate_batch`: T0 causality tripwire → T1 gross rank-IC → T2 capturability → T3 robustness →
T3.5 CPCV → deflation), the same instrument that killed the June mirage. Thresholds (never hardcoded
in code):

- **Gross (PROMISING floor):** IC-IR ≥ **0.05**, IC t-stat ≥ **3.0**, at `primary_horizon = 21`.
- **Deflation:** DSR ≥ **0.90**, BH-FDR q ≤ **0.10** across the `n_trials = 3` batch, HLZ t ≥ 3.0
  reported (not binding — `require_hlz: false`).
- **CPCV:** 6 groups, k=2, embargo 5d.
- **Robustness:** subperiod IC-IR ≥ 0.0; recent-OOS 2y reported.
- **Capturability (Taiwan cost trio):** `frictionless` / `us_equiv_ref` 0.0010 (contrast only) /
  **`standard` 0.0021** (e-broker commission + 0.30% sell tax ≈ 2× US) / `harsh` 0.0029; cost-wall
  caution at 0.30. **A signal that is gross-PROMISING but net-negative at `standard` is a NO-GO.**
- **Neutralization (read from the SPEC, echoed in gates for the record):** winsor [0.01,0.99] →
  zscore → **sector + size** demean.
- **Universe:** `min_names_per_day` ≥ **60**, `min_adv_twd` > 0 (a real tradeable cross-section).
- **Promotion:** `survivorship_free_required: true` + `tier2_audit_required: true` — **no probe
  result promotes to capital or reads a deploy verdict without a PIT rebuild + a Tier-2 deep
  lifecycle audit.** A GO here earns only "PROMISING, forward-incubate", never a position.

---

## 5. Predictions (pre-committed)

- **Base rate:** given MDE80 ≈ 0.7–0.9, the modal outcome is **0/3 clear the deflated funnel**.
- **Most likely to survive, if any:** P1 (month-revenue drift) — the best-documented Taiwan retail
  anomaly and the lowest-turnover of the three.
- **Kill conditions (any one → that probe is NO-GO):** wrong IC sign vs the pre-committed sign;
  gross IC-IR < 0.05 or t < 3.0; net Sharpe ≤ 0 at the `standard` cost; DSR < 0.90 or FDR q > 0.10;
  CPCV subperiod IC-IR < 0 in a majority of folds.
- **Stop rule (no expansion):** this is the whole campaign — 3 probes. If all 3 are NO-GO, the thesis
  is closed and the follow-up is **not** "try more channels" (that is the mining campaign §5
  forbids) but a data-quality/power post-mortem (would a survivorship-free PIT rebuild or a
  capacity-constrained niche change the verdict?). Adding a 4th channel requires a *new*
  pre-registration.

---

## 6. VERDICT (filled 2026-07-15, S553-cont-133, from `results/taiwan_smallcap_altdata/scorecard.json`)

> ### ⚠️ RESTATED 2026-07-31 — the numbers moved; **every verdict is unchanged**
>
> The digits first recorded here (2026-07-16 14:03) **do not reproduce**. Cause is **not code**: all 5
> commits touching `finrl_pro_ds/signals/` since the run — `92278fe4` (v4.0), `1b8df680` (v6.0),
> `406654ed` (v7.1), `34caacef` (v9.0), `4a231bee` (v10.0) — **plus the exact HEAD at record time
> (`4be2e6b3`)** were bisected: all 7 code states return a **bit-identical** frictionless Sharpe of
> `1.044701045608392`.
>
> Cause is **`data/taiwan_smallcap/pool.parquet`**, re-enumerated 2026-07-31 11:41 as a side effect of
> a fetcher run for an unrelated probe. Its `sector` column (FinMind `industry_category`) is a
> **mandatory neutralization control** — scores are residualized on sector dummies daily
> (`finrl_pro_ds/signals/features.py:164-168`) — so re-classified names move every downstream number.
> Eligible (non-singleton) sectors went **33 → 31**. Controlled check: collapsing sectors to one
> bucket gives frictionless 1.0512, the same order as the drift.
>
> **`spec_hash`, `liquid_days_ge25` and `n_names_pool` are all blind to the sector partition** — they
> matched exactly while the numbers moved, which is what made this look like a code regression. Only
> `gross.breadth` exposed it (its denominator counts sectors). The anti-p-hacking seal is intact: the
> three spec hashes and the gates hash are unchanged and were never touched after seeing results.
>
> **This is neither a stale record nor a repair.** `sector_id` is documented as a *current* snapshot
> applied retroactively (`features.py:41`, "v1 current GICS; PIT at GO-gate"), so the 07-16 and 07-31
> runs carry the same class of lookahead from different download dates — **today's number is not more
> correct than the original.** The 07-16 pool snapshot is unrecoverable (gitignored, no history, no
> backup), so the original digits can only be superseded, not reconstructed.
>
> **Remediation shipped with this restatement:** the sector map is pinned to `pool.frozen.parquet`
> (the fetcher only ever writes `pool.parquet`), and `panel_meta.sector_map` now stamps
> `sector_map_sha` — a digest over the `(ticker, sector)` pairs *for this panel's own names*, so an
> unrelated listing added to the pool does not trip it but any genuine change to this partition does.
> This run: `sector_map_sha=d835f37083ba`, 39 labels / 0 unknown, `frozen=True`.
>
> One further change is real and is an **improvement, not drift**: subperiod-1 now reports `—` instead
> of a spurious `1.536`. `crucible-v4.0` fixed the row-index subperiod split that gave subperiod 1 only
> 48 valid days against ~1323 for the others (documented as trap #1 in the S553-cont-133 audit).
>
> Both affected artifacts were regenerated against the frozen pool:
> `results/taiwan_smallcap_altdata/` and `results/taiwan_smallcap_altdata_lowturn/` (frictionless
> 0.7057 → 0.6848; `crucible_substrate_eligibility.py:98` consumes the latter). Campaigns recorded from
> 2026-07-31 12:00 onward were already on the new pool and reproduce exactly. Pre-restatement copies of
> both scorecards were archived before regeneration.
>
> Original values are preserved in the table below for the audit trail.

**Run record.** Full FinMind pull (Sponsor token, env-only, never committed): 2131-name TWSE+TPEx
common-stock pool → prices 2130 ids / month-revenue 2126 / margin 1995 / 集保 2106 / dividends 2029.
Panel: pool **N=612**, **T=5292** (2005-01-03..2026-07-15), **4017 liquid days**; membership
**2010-02-26..2026-07-15**, **median 200 names/month** (the 集保 register starts 2010-01-29 — *earlier*
than the ~2016 assumed in §1, so the band is longer than pre-registered, not shorter). Channel
coverage of active cells: mrev_yoy 99.3%, margin_util 97.2%, holder_conc 100.0%. Hygiene **PASS** on
all three (causal ✓, OHLC violations 0). All three spec content-hashes reproduced exactly
(`60680e61ff85` / `e0a4c719bfe0` / `1be26f02ee6a`) and the gates file (`0ccf6dd584f0`) was not edited
after seeing results — **the anti-p-hacking seal held.**

Values are **RESTATED (2026-07-31, frozen sector map)**, with the *as-first-recorded* 2026-07-16
figures in parentheses. Verdicts are identical in both.

| Probe | gross IC-IR @21 | IC t | net Sharpe @standard | DSR | FDR-q | verdict |
|---|---|---|---|---|---|---|
| P1 mom_rev | **+0.288** (+0.255) | **+18.23** (+16.09) | **+0.56** (+0.53) | 1.000 | 0.000 | **PROMISING (gross)** — survivorship-suspect; forward-incubate ONLY, never capital |
| P2 margin_crowd | −0.104 (−0.136) | −6.56 (−8.58) | −1.21 | 0.000 | 0.984 (0.997) | **NO-GO** — realized sign **INVERTED** vs the pre-committed −1 |
| P3 holder_conc | −0.098 (−0.117) | −6.13 (−7.31) | −0.82 | 0.000 | 0.984 (0.997) | **NO-GO** — realized sign **INVERTED** vs the pre-committed +1 |

**Synthesis — 1/3 PROMISING, 2/3 NO-GO; and the 1 is NOT an established edge.**

*What is PROVEN (measured, on this substrate):*
- **P2 and P3 are cleanly falsified.** Both realized the *opposite* of their pre-committed sign (§5
  kill condition #1), with negative net Sharpe at every cost model, DSR 0.000 and FDR-q 0.997. The
  contrarian margin-crowding mechanism and the informed-accumulation 集保 mechanism are both rejected
  in the small/mid band. Rising margin utilization mildly predicts *continuation*, not reversal.
- **P1 clears every pre-registered gate on the current-listing pool**: IC-IR 0.288 ≥ 0.05, t 18.23 ≥
  3.0, DSR 1.000 ≥ 0.90, FDR-q 0.000 ≤ 0.10, net Sharpe +0.56 at the 0.30%-sell-tax `standard` cost
  (still +0.37 at `harsh`), all **15/15 CPCV paths positive** (OOS mean 0.916, p05 0.542), IC rising
  monotonically with horizon (0.135→0.288→0.532 at 1→21→63d) and deciles monotone — the shape a slow
  fundamental drift *should* have. Turnover 6.8×/yr, max DD −6.1%.
  *(as first recorded 07-16: IC-IR 0.255, t 16.09, netSh +0.53/+0.34, CPCV 0.867/0.547,
  0.129→0.255→0.490, max DD −7.1%.)*
- **P1's IC DECAYS monotonically across the populated subperiods.** Restated subperiod IC-IR: P1
  `[—, 0.413, 0.300, 0.137]`, P3 `[—, −0.212, −0.171, 0.067]`, P2 `[—, −0.078, −0.131, −0.116]`.
  **The leading slot is now reported as `—` by the harness itself**: `crucible-v4.0` fixed the
  row-index split described in the caveat below, so the 48-valid-day window no longer produces a
  number to discard by hand. (As first recorded 07-16, before that fix, it emitted the spurious
  P1 `1.536` / P3 `1.459` / P2 `0.14`.) P1's real trajectory is **0.413 → 0.300 → 0.137** with
  recent-2y **+0.188**. The full-sample 0.288 is *not* inflated by the excluded window (1.2% of IC
  days) — it is a fair average of the three populated subperiods — but the **forward-looking
  expectation is ~0.14–0.19, not 0.288.**
- **⚠️ HARNESS CAVEAT found while auditing this result (latent, pre-existing, not introduced here).**
  `eval_harness.tier3_robustness` splits subperiods by **row index** (`np.linspace(0, panel.T, 5)`)
  over the panel's **full date range**, not over *valid/active* days. This panel spans 2005-01-03 but
  membership only begins 2010-02-26, so **subperiod 1 contains just 48 valid days** (2010-02-26..
  2010-05-05) versus **1323 each** for subperiods 2-4 (verified directly). P1's `1.536` and P3's
  `1.459` are therefore **small-sample noise on a ~96%-empty window — NOT evidence of a shared
  artifact, and NOT evidence of survivorship.** An earlier draft of this section read them as a
  cross-probe survivorship signature; **that reading was wrong and is retracted here.**
  *Verdict impact: NONE* — P1's binding `min_subperiod_ic_ir` (restated 0.137; 0.109 as first
  recorded) came from subperiod **4**, and P2/P3 fail on inverted sign + DSR 0.000 regardless. But the
  defect can in principle inflate `mean_subperiod_ic_ir` or spuriously trip the
  `min_subperiod_ic_ir >= 0.0` gate on any panel whose active window is shorter than its date range.
  **RESOLVED 2026-07-16 (S553-cont-134)** — fixed alongside wiring the `robustness.min_subperiod_ic_ir`
  gate, which had itself been unwired since v2.0; shipped as `crucible-v4.0` (`92278fe4`). The restated
  subperiod lists above show the repair: slot 1 is now `—` rather than a spurious spike.

- **⚠️ §1's "Delisting" clause was NEVER IMPLEMENTED — survivorship here is FULL, not partial.**
  §1 states *"`TaiwanStockDelisting` — delisted names are included for their live window so a name
  drops out of the cross-section when it dies, not before."* **The code does not do this.**
  `TaiwanStockDelisting` appears only in the fetcher's module docstring; it is never called, it is not
  in `_CHANNELS`, and `enumerate_common_stock_pool` queries **`TaiwanStockInfo` alone** (2131
  currently-listed ids). Verified directly. §1 is left **unedited** (editing a pre-registration after
  seeing results is exactly what the seal forbids) — the discrepancy is recorded here instead.
  *Verdict impact: none on the gates* (the harness correctly reports `survivorship_free=False` and the
  promotion gate binds), **but the survivorship exposure is WORSE than §1's prose implies**, which
  makes the PIT rebuild below more decisive, not less.

*What is SPECULATED (NOT established — no positive evidence from this run):*
- **Survivorship remains the pre-registered structural caveat** (§1/§4): the pool is enumerated from
  currently-listed tickers, so contamination is worst furthest back and decays toward the present, and
  "high revenue growth" is partly a proxy for *having survived*. This is a reason the 0.255 is an
  **UPPER BOUND** — but this run produced **no positive evidence** that the bias is what drives P1
  (the subperiod pattern I initially read that way does not survive the 48-day correction above).
- **The observed decay is not diagnostic on its own.** Genuine alpha decay (the tier being
  arbitraged), a regime effect, and survivorship all predict a decline toward the present, and are
  observationally similar on this data. **The pre-registered PIT rebuild is the experiment that
  discriminates them** — a promotion gate (§4), not a rescue analysis.

*Verdict actions (bound by §4/§5, no discretion taken):*
1. **P1 → "PROMISING, forward-incubate" only.** `survivorship_free_required: true` and
   `tier2_audit_required: true` both **BIND**: no capital, no paper sleeve, and no deploy-gating
   verdict may be read off this number. Every IC above is an **UPPER BOUND**.
2. **Decisive next test (not a new probe):** a survivorship-free PIT rebuild of the pool (augment with
   `TaiwanStockDelisting` ids, or TEJ-grade), then re-run *this frozen spec, unchanged*. If P1's IC
   survives at ~0.1-0.17 with the recent subperiod intact, it becomes a real (small) candidate for
   Tier-2. If it collapses toward the June mirage's fate, the small-cap alt-data thesis closes.
3. **Round 2 is NOT triggered.** Its trigger requires *structure without capture* (gross clears but
   cost/deflation kills) or a clean 0/3. P1 cleared gross **and** cost **and** deflation, so neither
   condition holds. Per §5's stop rule this campaign is complete at 3 probes; do not add channels.
4. **P1's cost wall is 0.479** (caution 0.30) — over half the frictionless Sharpe (1.008) is eaten by
   the 0.30% sell tax even at 21-day holding. Any future form must be *lower*-turnover, not richer.

---

## 7. Run protocol (operator, with token)

```bash
# 0. token in env only — NEVER committed/logged
export FINMIND_TOKEN=...            # Sponsor tier

# 1. fetch the three alt-data channels + the common-stock pool prices (rate-limited; --resume safe)
python scripts/data/fetch_taiwan_fundamentals_finmind.py \
    --datasets month_revenue,margin_short,shareholding \
    --pool auto --start 2005-01-01 --out data/taiwan_smallcap

# 2. build the monthly cap-rank 51-250 PIT membership
python scripts/research/taiwan_smallcap_universe.py \
    --data data/taiwan_smallcap --lo-rank 51 --hi-rank 250 --out data/taiwan_smallcap/universe

# 3. run the 3 pre-registered probes through the funnel (size-neutralization is in the specs)
python scripts/research/taiwan_smallcap_altdata_eval.py \
    --data data/taiwan_smallcap --gates configs/taiwan_smallcap_altdata.gates.yaml \
    --out results/taiwan_smallcap_altdata

# offline, no token — proves causality + spec-hash seal + lag-leak guards are green first
python -m pytest tests/research/test_taiwan_smallcap_altdata.py -q
```

Then fill §6, append a `randd_log.md` entry, and run `/audit` (+ `/math` if any formula changed).
The gates-file hash is already registered under **CRU-1** (audit §6.7) — `0ccf6dd584f0`, pinned in
`tests/crucible/test_version.py`; if you edit the gate after seeing results, that test goes red (by
design), and a *deliberate* change must bump the pinned reference.
