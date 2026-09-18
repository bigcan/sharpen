# PRE-REGISTRATION — Taiwan small/mid-cap SHORT-INTEREST probe (S1)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE the signal was computed or any IC was seen.**
**Substrate:** `data/taiwan_smallcap/`, cap-rank 51-250, same locked universe/panel as P1.
**Zero fetch** — `margin_short.parquet` and `shareholding.parquet` are already on disk.

## 0. Why one probe, and why this one

Four probes have now failed on this substrate today (P1@63d ruled out as a form, R1 reversal, R2
IVOL, plus the quota-blocked institutional pair unrun). With a base rate that low, adding a batch of
marginal mechanisms is negative-expected-value: it inflates multiplicity and dilutes any real
finding. So this is **a single probe on the highest-prior untested mechanism available**.

Relative short interest is among the most replicated cross-sectional return predictors in the
literature (Boehmer-Jones-Zhang and the large subsequent replication record): high relative short
interest predicts underperformance, and short sellers are the canonical informed cohort. It is
untested here, and the data has been sitting on disk unused — the 2026-07-15 campaign fetched
`short_balance` alongside `margin_balance` but explicitly reserved it: *"Short balance is carried in
the panel for a pre-registered robustness read only — NOT a second trial."* So it has never been a
signal. This makes it one, in a new campaign with its own cumulative multiplicity.

Taiwan-specific reason the prior is strong: margin short selling requires a borrow under strict
exchange mechanics and is costly to maintain, so a large short balance is an expensive position to
hold and therefore more likely to be informed than a cheap one.

## 1. The probe (LOCKED)

`family = "altdata"`, `horizons = (1,5,10,21,63)`, `primary_horizon = 21` (monthly rebalance — the
only turnover regime the 0.30% sell tax survives), `neutralization = ("winsor","zscore","sector",
"size")`. **`size` MANDATORY** — the step that killed the June large-cap mirage.

### S1 — Short-interest ratio · `tw_smallcap_short_interest` · sign **−1**
- **Signal (causal):** `short_util = short_balance / total_shares`, the short-interest ratio.
  `total_shares` is merged as-of **backward** so only a share count already public at the balance's
  own trading date is used; the resulting value is stamped `avail_date = date + 1 business day`
  (TWSE publishes balances after the close). Identical causal construction to P2's `margin_util`,
  with the short leg substituted for the long leg.
- **Expected sign: −1** (long LOW short interest, short HIGH short interest).

## 2. The built-in interpretation guard (committed in advance)

P2 tested the **long** analogue — Δ margin utilization, sign −1 — and the realized sign came out
**INVERTED**: rising retail long leverage predicts *continuation*, not reversal.

That result constrains how S1 must be read:

- S1 realizes **−1** (committed) ⇒ consistent with **informed short selling**: the long and short
  legs behave asymmetrically, which is what an information story predicts.
- S1 realizes **+1** (inverted) ⇒ short interest predicts continuation *just like* margin
  utilization did. Both legs moving the same way is a **common flow/momentum factor**, not informed
  positioning — and that is a **FAIL of S1**, to be recorded as such rather than re-labelled as "a
  momentum signal we found."

## 3. Pre-committed pass conditions

Unchanged locked thresholds: IC-IR ≥ 0.05, |IC t| ≥ 3.0, DSR ≥ 0.90, FDR-q ≤ 0.10, **realized sign
== committed sign**. The CRU-1-sealed `taiwan_smallcap_altdata.gates.yaml` (`0ccf6dd584f0`) is **not
edited**; a sibling gates file is used.

**Capturability is a first-class pass condition here, not a caveat.** Today's R1/R2 run established
that this funnel's `PROMISING` verdict never consults `capturability` (`scorecard.py:136` — "they
flag, not gate"), so a signal can carry the label while losing money at zero cost. For S1 the bar is
stated explicitly and in advance: **a non-positive frictionless Sharpe is a NO-GO regardless of
IC-IR, DSR or FDR-q**, and net Sharpe at `standard` cost must be positive to be worth anything.

## 4. Multiplicity — declared CUMULATIVELY

3 (alt-data P1/P2/P3, run) + 2 (institutional Q1/Q2, pre-registered, quota-blocked, still counted as
file-drawer control) + 2 (price R1/R2, run) + 1 (this) = **8 declared hypotheses on this substrate**.

## 5. Stop rule

Complete at 1 probe. **No S2.** If S1 fails, informed-short-positioning on this band is recorded as
falsified and is not re-probed with a different normalization (days-to-cover, float-adjusted,
Δ-instead-of-level) — re-specifying a dead mechanism is the N-grab this rule prevents.

## 6. Promotion ceiling

`survivorship_free_required` and `tier2_audit_required` both BIND. A pass earns **forward-incubate
only** — no capital, no paper sleeve, no deploy-gating verdict.

## 7. Results — **NO-GO**

Run 2026-07-31, `results/taiwan_smallcap_short/scorecard.json`, spec `81c0e92ef004`.
Panel N=612, T=5292; `short_util` coverage **97.2%** of active cells. Harness verdict: **LOGGED**.

### IC ladder (sign_used = −1, so a POSITIVE IC-IR means the committed sign held)

| horizon | 1d | 5d | 10d | **21d** | 63d |
|---|---|---|---|---|---|
| IC-IR | +0.048 | +0.106 | +0.104 | **+0.079** | **−0.109** |
| IC t | +3.02 | +6.70 | +6.61 | +5.00 | −6.84 |
| decile spread | +0.0002 | −0.0001 | −0.0010 | −0.0037 | −0.0208 |
| decile monotonic | True | True | False | False | False |

### Capturability (the pre-committed bar, §3)

| cost model | net Sharpe |
|---|---|
| frictionless | **+0.086** |
| us_equiv_ref (10bp) | −0.092 |
| **standard (Taiwan 0.21%)** | **−0.288** |
| harsh (0.29%) | −0.429 |

turnover 5.47/yr · cost wall 0.374 · max DD −0.191 at standard

### Why it fails

**§3 pre-committed that net Sharpe at `standard` cost must be positive to be worth anything. It is
−0.288.** The frictionless Sharpe is +0.086 — technically positive, but so close to zero that the
0.30% sell tax alone takes it four times over. This is a NO-GO on the bar set *before* the run, not
on a bar chosen after seeing it.

Three further independent problems, any one of which would be disqualifying:

1. **The 21d bootstrap CI straddles zero**: `[−0.0027, +0.0184]`, `p_le_0 = 0.0716`. The IC t-stat of
   +5.00 is inflated by overlapping forward returns; the block-bootstrap CI, which accounts for that
   autocorrelation, cannot separate the IC from zero at 5%. **Read the CI, not the t.**
2. **The sign flips at 63d** (IC-IR −0.109, t −6.84). A mechanism that predicts underperformance at
   a month and outperformance at a quarter is not one mechanism.
3. **It has inverted in the live window**: recent-2y IC-IR **−0.184**, i.e. the opposite of the
   committed sign. CPCV is fragile too (mean −0.046, p05 −0.488, only 40% of 15 paths positive).

Note also the incoherence between a positive rank-IC and a **negative decile spread** at 5-21d: the
correlation is not in the extremes the long-short book actually trades. Same "structure without
capture" pathology as today's R1/R2.

### §2's interpretation guard, applied

S1 realized the **committed −1** at the primary horizon, so this is *not* the "both legs are one
common flow factor" outcome §2 warned about. It is a cleaner and more boring failure: the sign is
right, the effect is real-ish gross, and it does not survive contact with the Taiwan sell tax or the
recent regime.

### Design lesson (worth carrying)

`deflation undefined (batch too small for DSR)` — declaring 8 cumulative hypotheses registered
correctly (`multiplicity_source: preregistered`) but **DSR still could not be computed, because the
batch held a single signal and DSR needs a trial Sharpe distribution**. So "one probe minimises
multiplicity" bought an *undefined* deflation rather than a lenient one. It does not change this
verdict — the cost and stability failures are decisive on their own — but a future single-probe run
should either be batched with its own pre-registered controls or use `sr_star` against the declared
count instead of relying on DSR.

### Ledger entry (durable — do not re-test)

**Taiwan small/mid-cap short interest (margin-short balance / shares) is FALSIFIED.** Right sign at 21d, no
capture after the sell tax, sign flips at 63d, inverted over the recent 2 years. Per §5 the campaign
is **complete at 1 probe** — no S2, and no re-specification as days-to-cover, float-adjusted, or
Δ-instead-of-level.
