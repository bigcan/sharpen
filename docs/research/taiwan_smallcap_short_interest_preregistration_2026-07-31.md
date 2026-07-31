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

Taiwan-specific reason the prior is strong: 融券 (margin short) requires a borrow under strict
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

## 7. Results

*(Empty at commit time on purpose — the sign commitment is verifiable from git history.)*
