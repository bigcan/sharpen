# PRE-REGISTRATION — Taiwan small/mid-cap PRICE-ONLY probes (R1/R2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE any signal was computed or any IC was seen.**
**Substrate:** `data/taiwan_smallcap/` cap-rank 51-250 band — the same locked universe and panel
builder as the 2026-07-15 campaign. **Zero data fetch:** both signals derive from `prices.parquet`,
already on disk. **Harness:** the locked `sharpen/signals/` scorecard, thresholds unchanged.

## 0. Why these, and why they are not a re-specification

The 2026-07-15 campaign probed three **alt-data** channels (month-revenue, margin, 集保 holdings) and
its stop rule bars its Round 2 (`value/book-to-market`, `分點` branch-concentration). Those channels
stay barred and untouched. R1/R2 are **price-only** and test mechanisms none of the prior probes
touched — neither is a different window or scaling of P1/P2/P3, which the stop rule exists to
prevent.

Taiwan's cash market is retail-dominated (retail has historically been a majority of TWSE turnover).
That single structural fact predicts two *distinct*, separately-documented effects, and the small/mid
band is where both should be strongest:

- uninformed retail order flow demanding immediacy ⇒ liquidity providers earn a premium ⇒
  short-horizon **cross-sectional reversal** (R1);
- retail lottery-preference plus limits to arbitrage ⇒ high-volatility "lottery" names are
  over-bought and subsequently underperform ⇒ the **idiosyncratic-volatility** effect (R2).

Neither Taiwan small/mid-cap reversal nor Taiwan IVOL appears in the NO-GO ledger. (What *is* closed
is Taiwan **large-cap momentum**, which failed as a size-confound — the opposite band and the
opposite sign convention.)

## 1. The two probes (LOCKED — signal, sign, mechanism fixed before any result)

Both: `family = "price"`, `horizons = (1,5,10,21,63)`, `primary_horizon = 21` (monthly rebalance —
the only turnover regime the 0.30% sell tax survives, per the P1 cost-wall finding),
`neutralization = ("winsor","zscore","sector","size")`. **`size` is MANDATORY** — it is the exact
step that killed the June large-cap mirage. Signs are pre-committed; a realized IC of the *opposite*
sign is a **FAIL**, not a sign-flip opportunity.

### R1 — Short-term reversal · `tw_smallcap_st_reversal` · sign **−1**
- **Signal (causal):** `r21[t] = close[t]/close[t-21] − 1`, using only bars ≤ t.
- **Expected sign: −1** (long past losers, short past winners).
- **Mechanism:** liquidity provision to uninformed retail flow.

### R2 — Idiosyncratic volatility · `tw_smallcap_ivol` · sign **−1**
- **Signal (causal):** trailing 63-day standard deviation of daily returns, bars ≤ t.
- **Expected sign: −1** (long low volatility).
- **Mechanism:** lottery preference + limits to arbitrage.
- **Stated proxy limitation, committed in advance:** this is *total* volatility, not a residual from
  a fitted factor model. The `sector` + `size` neutralization removes the dominant systematic
  components cross-sectionally, so it stands as an IVOL proxy — but if R2 passes, the honest
  follow-up is a residual-vol construction, and that limitation is recorded here **before** seeing
  the result rather than added as an excuse afterwards.

## 2. Pre-committed pass conditions

Unchanged from the locked gates: IC-IR ≥ 0.05, |IC t| ≥ 3.0, DSR ≥ 0.90, FDR-q ≤ 0.10, **realized
sign == committed sign**. The CRU-1-sealed `configs/taiwan_smallcap_altdata.gates.yaml`
(`0ccf6dd584f0`) is **not edited**; a sibling gates file carries only the `family`/name changes.

## 3. Multiplicity — declared CUMULATIVELY, before results

| campaign | probes | run? |
|---|---|---|
| 2026-07-15 alt-data (P1/P2/P3) | 3 | yes |
| 2026-07-31 institutional flow (Q1/Q2) | 2 | pre-registered, blocked on FinMind quota |
| 2026-07-31 price-only (R1/R2) | 2 | this file |
| **declared total** | **7** | |

**The two unrun institutional probes are still counted.** A pre-registered hypothesis enters the
ledger whether or not it is executed — that is precisely the file-drawer control a pre-registration
exists for, and dropping them because they happened to be blocked would understate the search.
Charging 7 rather than 2 is the `crucible-v9.0` (U5) `max(batch_pool, declared_hypotheses)` rule
applied honestly across campaigns.

## 4. Stop rule for THIS campaign

Complete at 2 probes. **No R3.** If both fail, the retail-microstructure thesis on this band is
recorded as falsified and is not re-probed with a different lookback, a different vol window, or a
conditioning filter. Re-specifying a dead mechanism until it passes is the exact N-grab this rule
prevents (and is what the 0/10 rescue-filter sweep on TX already demonstrated the cost of).

## 5. Promotion ceiling (binds regardless of result)

`survivorship_free_required: true` and `tier2_audit_required: true` both BIND. A pass earns
**"PROMISING, forward-incubate"** and nothing more — no capital, no paper sleeve, no deploy-gating
verdict. Measured context: P1's survivorship sensitivity was Δ IC-IR +0.001, so the bias is small on
this panel, but the flag stands.

**Cost reality check, committed in advance:** P1 at 21-day holding had a cost wall of 0.479 against a
frictionless Sharpe of 1.008 — the 0.30% sell tax takes about half. Any R1/R2 pass must be read
net, not gross; a high gross IC-IR that dies at `standard` cost is a **NO-GO**, not a "promising
signal with an execution problem."

## 6. Results — **BOTH PROBES NO-GO**

Run 2026-07-31, `results/taiwan_smallcap_price/scorecard.json`. Panel N=612, T=5292, 4017 liquid
days — identical to the P1 campaign. Coverage 99.9% / 100.0% of active cells.

| | R1 `st_reversal` | R2 `ivol` |
|---|---|---|
| spec hash | `19c28bf743d3` | `27d38ce84ff5` |
| IC-IR @21d | +0.066 | +0.086 |
| IC t | +4.20 | +5.43 |
| realized sign | −1 ✓ (as committed) | −1 ✓ (as committed) |
| decile spread | −0.0054 | −0.0120 |
| **decile monotonic** | **False** | **False** |
| **frictionless Sharpe** | **−0.182** | **−0.627** |
| net Sharpe @ standard | −0.835 | −0.837 |
| cost wall | 0.654 | 0.210 |
| turnover / yr | 11.90 | 4.39 |
| CPCV OOS mean / p05 | −0.295 / −0.791 | −0.554 / −1.091 |
| CPCV paths positive | 20% | 7% |
| recent-2y IC-IR | −0.069 | −0.151 |
| subperiod IC-IR | [−0.017, 0.32, −0.066] | [0.14, 0.009, 0.108] |
| harness verdict | LOGGED | "PROMISING" as run; **LOGGED** since `crucible-v11.0` (see §6.2) |

### 6.1 Why both fail: structure without capture

Both realized the **committed sign** and both have a nominally significant rank-IC (t 4.2 / 5.4).
Neither is tradeable, and the decisive number is not the cost model — it is that **frictionless
Sharpe is NEGATIVE for both** (−0.182, −0.627). The book loses money at **zero** cost.

The reconciliation is `decile_monotonic: False` on both. A positive rank-IC with a non-monotonic
decile profile means the correlation lives in cells that carry no weight in a long-short book — the
extremes, which the book actually trades, do not line up with the ranking. So the "signal" never
becomes a position that makes money, before costs are discussed at all.

This is exactly the failure mode the pre-registration's §5 cost-reality clause anticipated, one
level worse: not "gross edge killed by the 0.30% tax" but **no gross edge in the traded book to
begin with**.

Both also invert recently (recent-2y IC-IR −0.069 / −0.151, i.e. the *opposite* of the committed
sign in the live window) and both are CPCV-fragile (7-20% of 15 paths positive).

### 6.2 A harness finding: `PROMISING` does not mean capturable

R2 is labelled **PROMISING** while having a frictionless Sharpe of −0.627. That is not a bug in the
sense of a mistake — `scorecard.py:136` states it deliberately: *"capturability caveats (do not
change the gross-IC verdict — they flag, not gate)"*. The `promising` predicate reads DSR, IC-IR,
IC-t, FDR-q, optional HLZ, min-subperiod IC-IR and multiplicity provenance, and **never consults
`capturability`**. Verified: zero references to the capturability result inside the predicate.

The consequence matters for how the record reads: **"PROMISING" in this funnel means "statistically
detectable gross rank-IC", not "tradeable."** P1 happened to be genuinely capturable (frictionless
1.008, net 0.529), so the distinction never bit — but nothing in the verdict logic enforced that,
and R2 is a live demonstration that a money-losing book can carry the label. Any future citation of
a PROMISING result should quote the frictionless and net Sharpe alongside it.

#### 6.2.1 CLOSED — `crucible-v11.0` (2026-07-31)

The gap is fixed. `capturability.min_frictionless_sharpe` (ships **active at 0.0**, strict `>`) is
now a leg of the `promising` predicate, so a signal whose long-short book loses money at zero cost
cannot carry the label. `net@standard` stays a **caveat**, gating only under the opt-in
`capturability.require_positive_net_standard` — a cost model is venue-specific (Taiwan's 0.30%
sell-side tax is not Nasdaq's 10bps), whereas a negative frictionless book is unconditional. An
unmeasured capturability fails **closed**, per the DSR / min-subperiod precedent.

Verified by re-running both campaigns on the real panel, not by argument:

| | frictionless | verdict before | verdict after |
|---|---|---|---|
| `tw_smallcap_mom_rev` (P1) | +1.045 | PROMISING | **PROMISING** (unchanged) |
| `tw_smallcap_ivol` (R2) | −0.627 | PROMISING | **LOGGED** |
| `tw_smallcap_st_reversal` (R1) | −0.182 | LOGGED | LOGGED |

The change is monotone-stricter, so it can only demote; **R2 is the only recorded verdict that
moves**. Wiring a gate after results are known is legitimate in this direction — the candidate that
motivated the change is the one it demotes. No gates-YAML byte was edited: the three CRU-1-sealed
files and this campaign's own sibling file predate the key and inherit it by deep-merge, and a test
asserts their bytes still do not contain it, so `0ccf6dd584f0` and the funnel moats are untouched.
`to_markdown` also gained a `fricSh` column — the miss was partly a reporting failure, since the
table that carried R2's PROMISING showed `netSh@std` and `costWall` but never the frictionless
Sharpe that made it not a signal at all.

Note for anyone re-running P1: its frictionless Sharpe reads **1.0447** today against the **1.008**
recorded on 2026-07-16, on an apparently identical panel (same spec hashes, same 4017 liquid days).
That drift is unrelated to this gate — a control run on the pre-v11.0 source reproduces 1.0447
exactly.

**RESOLVED 2026-07-31 — and the guess in the first version of this paragraph was WRONG.** It named the
v7.1 per-name `(T,N)` alt-data bridge as the likeliest source. A bisect **falsified** that: all five
post-07-16 `sharpen/signals/` commits *plus the exact HEAD at record time* (`4be2e6b3`) — seven
code states — return a bit-identical `1.044701045608392`. The cause is **data**:
`data/taiwan_smallcap/pool.parquet` was re-enumerated 2026-07-31 11:41 by an unrelated probe's fetcher
run, and its `sector` column is a **mandatory neutralization control** (scores are residualized on
sector dummies daily, `sharpen/signals/features.py:164-168`), so re-classified names move every
downstream number — eligible sectors went 33 → 31. **All verdicts are unchanged in both readings.**
The sector map is now pinned to `pool.frozen.parquet` with a `sector_map_sha` stamp. Full restatement
and audit trail: `taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md` §6.

Worth carrying forward: `spec_hash`, `liquid_days_ge25` and `n_names_pool` are all **blind to the
sector partition** and matched exactly — which is precisely what made a data change look like a code
regression. A gitignored data file is an unpinned input to every "reproducible" verdict; hash what you
neutralize on, not just the spec.

### 6.3 Ledger entries (durable — do not re-test)

- **Taiwan small/mid-cap 21-day short-term reversal is FALSIFIED.** Right sign, no capture,
  inverts recently, regime-fragile. The liquidity-provision premium is not harvestable in this band
  at monthly rebalance.
- **Taiwan small/mid-cap volatility (IVOL proxy) is FALSIFIED.** Right sign, negative gross book
  Sharpe, 7% of CPCV paths positive, inverts recently.
- Per §4 the campaign is **complete at 2 probes**. No R3, and neither mechanism is to be re-probed
  with a different lookback, vol window, or conditioning filter. The retail-microstructure thesis
  on this band is recorded as tested and dead.
- The R2 residual-vol follow-up flagged in §1 is **not owed** — it was conditional on R2 passing.
