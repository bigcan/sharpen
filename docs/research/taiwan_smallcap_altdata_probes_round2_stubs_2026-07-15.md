# PRE-REGISTRATION STUBS (CONTINGENT) — Taiwan Small/Mid-Cap Alt-Data Probes, Round 2

**Date:** 2026-07-15 · Session S553-cont-133 · Branch `July2026`
**Status: NOT ACTIVE.** Design + frozen spec-hashes only. **No fetch, no code, no verdict.**
**Parent:** `taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md` (Round 1 — the active batch).

> These are the two next-best untouched-data probes, pre-designed so a Round 2 can be *slotted in*
> rather than improvised — but they are **held behind a trigger** on purpose. Audit §5 forbids the
> mass-mining sweep; the way you honour that while still keeping a pipeline of hypotheses is to
> **pre-register the design and the sign, then gate the run.** Writing the design down early is not
> the same as running it early.

---

## 0. The trigger — when (and only when) Round 2 runs

Round 2 is authorised **only if Round 1 (the 3 active probes) produces a specific, non-fishing
reason to keep looking**, namely ONE of:

1. **"Structure without capture"** — a Round-1 probe clears the *gross* IC floor (IC-IR ≥ 0.05,
   t ≥ 3.0) with the pre-committed sign but is killed only on **cost or deflation** (net Sharpe ≤ 0
   at the 0.30%-tax `standard` model, or DSR < 0.90). That is evidence a real mechanism exists but
   the current *form* doesn't capture it — a legitimate reason to test an adjacent mechanism.
2. **A clean, unanimous 0/3 NO-GO** *and* the operator judges the **value / smart-money-flow** axes
   (deliberately different from Round 1's revenue/leverage/register axes) worth one more angle
   before closing the small-cap thesis.

If Round 1 is a messy null with no gross signal anywhere, the correct move is **stop** (the §5
non-negotiable), not Round 2.

**Multiplicity honesty (binding).** Round 2 is its **own** `n_trials = 2` batch with its **own**
scorecard — it must **not** be run concurrently with Round 1 to quietly grow N, and it must **not**
be cherry-picked one-probe-at-a-time. If a Round-2 survivor is ever compared head-to-head with a
Round-1 survivor for promotion, the deflation must be re-computed over the **combined** trial count
(3 + 2 = 5), never per-batch. Pre-registering the design here does not pre-spend the multiplicity
budget; **running** it does.

---

## 1. The two candidate probes (designed + hash-frozen; sign pre-committed)

Both inherit Round 1's frame verbatim: `family = "altdata"`, `horizons = (1,5,10,21,63)`,
`primary_horizon = 21`, `neutralization = ("winsor","zscore","sector","size")` (**size MANDATORY**),
universe `twse_smallcap_caprank_51_250`, cost profile `taiwan_standard`, same gates file
(`configs/taiwan_smallcap_altdata.gates.yaml`, CRU-1 hash `0ccf6dd584f0`). The hashes below become
**binding** the moment the Round-2 eval is built (its `build_signals` will assert them, exactly as
Round 1 does — a spec-text drift then fails the test).

### R2-A — Value (book-to-market) · `tw_smallcap_value_bm` · sign **+1** · hash `d32074342ae6`
- **Dataset (untouched):** `TaiwanStockPER` — daily per-name P/E, **P/B**, dividend yield (already
  point-in-time as published each session; the P/B embeds the last-reported book value with its
  statutory report lag baked in by the exchange).
- **Mechanism:** the **value premium** — cheap stocks (high book-to-market) outperform, historically
  strongest in the *small/mid-cap* tier where mispricing persists. The most-documented cross-sectional
  equity anomaly there is; a clean, cheap first extension of the "different data" lever.
- **Signal (causal):** `bm = 1 / PBR`, stamped **T+1** (published after the close), aligned to trading
  days via the same as-of forward-fill as Round 1. Names with non-positive / missing PBR → NaN.
- **Pre-registered robustness reads (NOT extra trials):** earnings-yield `1/PER` and dividend yield,
  carried in the panel for a single robustness table only — they do **not** each earn a trial.
- **Expected sign:** **+1** (long high book-to-market).
- **Build cost:** LOW — one extra FinMind dataset, PIT-clean daily, trivial transform.

### R2-B — Smart-money branch concentration (分點) · `tw_smallcap_branch_conc` · sign **+1** · hash `6976763832da`
- **Dataset (untouched, niche):** broker-branch daily buy/sell per stock (`TaiwanStockTradingDailyReport`
  or the equivalent 分點 dataset — **confirm exact FinMind id + Sponsor tier + coverage before build**).
  This is the §6.3 "capacity-constrained niche / small-operator reframe" poster child — big money
  ignores it because it is messy and low-capacity, which is exactly why an edge could survive there.
- **Mechanism:** **concentrated** net accumulation by a few branches (主力 / informed hands) predicts
  continuation; **dispersed** net buying (broad retail) does not. Concentration, not raw net flow, is
  the signal.
- **Signal (causal):** `branch_net_conc` = trailing-`N`-day sum of
  `(net-buy shares of the top-K net-buying branches − net-buy shares of the top-K net-selling branches)`
  `/ shares_outstanding`, with **K and N FIXED here** (K = 15 branches, N = 21 trading days — not
  tuned), stamped **T+1** (branch report is EOD). Concentration is enforced by the top-K restriction.
- **Expected sign:** **+1** (long concentrated net accumulation).
- **Build cost + risk:** HIGH — the 分點 feed is large (every branch × stock × day), the wire format
  needs a live PF-XCHECK, and the signal is the noisiest of the five. Run this **only** if R2-A does not
  already answer the "different data" question, or if the niche credibility is worth the build.

---

## 2. What is inherited unchanged from Round 1

- **Universe:** cap-rank 51–250 monthly PIT membership (`twse_smallcap_caprank_51_250`).
- **Size-neutralization MANDATORY** (in each spec's `neutralization`; the step that killed the June
  large-cap mirage).
- **Gates / cost trio:** `configs/taiwan_smallcap_altdata.gates.yaml` (0.30% sell-tax `standard`
  0.0021; low-turnover 21d hold), CRU-1-registered `0ccf6dd584f0`. Round 2 reuses it — **no gate edit**.
- **Survivorship UPPER BOUND** (current-listing pool) and **the power wall** (audit §5 MDE80 ≈ 0.7–0.9):
  a Round-2 null is as informative as a Round-1 null, and just as likely.

## 3. Build-on-greenlight (mirrors Round 1 — deferred until the trigger fires)

Nothing is built now. On greenlight, the work is the Round-1 shape:
1. extend `scripts/data/fetch_taiwan_fundamentals_finmind.py` with the R2 dataset(s) + their causal
   `avail_date` lags (PER: T+1; 分點: T+1 EOD), fail-loud on schema drift;
2. add the two signal classes to a Round-2 eval (or extend the Round-1 eval) reading from
   `feature_slots`, with `build_signals` asserting the frozen hashes above;
3. offline tests: spec-hash seal, availability-lag leak guard, truncation-equivalence, evaluate_batch
   smoke — all green before any token is used;
4. run through the funnel as its **own** `n_trials = 2` batch; fill a Round-2 VERDICT; `/audit`.

## 4. Predictions (pre-committed)

- **Modal outcome:** 0/2 clear the funnel (same power wall).
- **More likely to survive, if either:** R2-A (value) — the strongest documented small-cap anomaly and
  the lowest-noise of the two.
- **Kill conditions (per probe):** wrong IC sign vs the pre-committed +1; gross IC-IR < 0.05 or t < 3.0;
  net Sharpe ≤ 0 at `standard`; DSR < 0.90 or FDR-q > 0.10 (over the combined trial count if promoted).
- **Stop rule:** Round 2 is the last pre-designed batch. A third round is **not** "more datasets" — it
  is a data-quality / survivorship-free-PIT-rebuild question, and requires a *new* pre-registration.
