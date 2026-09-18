# PRE-REGISTRATION — Taiwan small/mid-cap institutional-flow probes (Q1/Q2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE any signal was computed or any IC was seen.**
**Substrate:** `data/taiwan_smallcap/` cap-rank 51-250 band, the same locked universe as the
2026-07-15 campaign. **Harness:** `scripts/research/taiwan_smallcap_altdata_eval.py`, unchanged.

## 0. Why this is a new campaign, not the barred Round 2

The 2026-07-15 pre-registration's §5 stop rule closed **that** campaign at 3 probes and barred its
Round 2 (`value/book-to-market` + `broker-branch` concentration). Those two channels are **not** probed
here and remain barred. This campaign uses a channel that did not exist on disk on 2026-07-15 —
per-name daily **institutional net flow** (TWSE T86 via FinMind
`TaiwanStockInstitutionalInvestorsBuySell`), fetched by a new `institutional` channel added to
`scripts/data/fetch_taiwan_fundamentals_finmind.py`.

*(The `data/raw/taiwan_t86_institutional/t86_institutional.json` already in the repo is NOT this
data: it holds 8 unique stock_ids — ETFs like 0050/0056, median 6 names/day — and has **zero**
overlap with the 2,131-name pool. It is Crucible connector plumbing, not a cross-sectional panel.)*

**The stop rule's purpose is to prevent an N-grab, and that purpose is honoured by multiplicity
accounting, not by never probing again** — the Crucible brief's stated goal is *continuous*
discovery. So §4 below declares the **cumulative** hypothesis count across both campaigns, not just
this batch's two.

## 1. Mechanism

Taiwan's cash equity market is retail-dominated (retail has historically been a majority of TWSE
turnover) and the small/mid band carries thin analyst coverage. Foreign institutional investors are
the best-documented informed cohort in that market. If their net buying carries information the
market under-reacts to, it should predict **cross-sectional continuation** in exactly the band where
coverage is thinnest — the same "less-efficient slice" logic that produced P1.

## 2. The two probes (LOCKED — signal, sign, and purpose fixed before any result)

Both: `family = "altdata"`, `horizons = (1,5,10,21,63)`, `primary_horizon = 21`,
`neutralization = ("winsor","zscore","sector","size")`. **`size` is MANDATORY** — it is the step
that killed the June large-cap mirage. Signs are pre-committed; a realized IC of the *opposite* sign
is a **FAIL**, not a sign-flip opportunity.

### Q1 — Foreign institutional net flow · `tw_smallcap_foreign_flow` · sign **+1**
- **Signal (causal):** `foreign_flow_21 = Σ_21d(foreign_net) / Σ_21d(volume)` — 21-day net foreign
  share flow as a fraction of 21-day traded share volume. Dimensionless and scale-free, so it is not
  a size proxy in disguise. Stamped at `avail_date = date + 1 business day` (TWSE publishes T86
  **after** the close, so the first tradeable session is the next one — LEAK-2).
- **Expected sign: +1** (long names with high net foreign buying).

### Q2 — Investment-trust net flow · `tw_smallcap_trust_flow` · sign **+1**
- **Signal (causal):** identical construction on `trust_net`.
- **Expected sign: +1** (same sign on purpose — see below).
- **Q2 IS A MECHANISM TEST, NOT A SECOND SHOT AT A HIT.** Domestic investment trusts are
  institutional but far more prone to window-dressing and momentum-chasing, and the literature on
  their informativeness is much weaker than for foreign investors. So:
  - Q1 **≫** Q2 ⇒ consistent with *informed* foreign flow (the claimed mechanism);
  - Q1 **≈** Q2 ⇒ the signal is more likely generic **flow / price-pressure**, not information —
    which **falsifies the stated mechanism even if the IC is positive**, and the result must be
    reported that way rather than re-labelled.

This contrast is the reason there are exactly two probes and not five.

## 3. Pre-committed pass conditions

Unchanged from the locked gates (`configs/taiwan_smallcap_altdata.gates.yaml`, CRU-1 `0ccf6dd584f0`,
**not edited**): IC-IR ≥ 0.05, |IC t| ≥ 3.0, DSR ≥ 0.90, FDR-q ≤ 0.10, realized sign == committed
sign. A new sibling gates file is used only if the primary horizon differs; thresholds are identical.

## 4. Multiplicity — declared CUMULATIVELY, before results

Prior campaign 3 probes (`tw_smallcap_mom_rev`, `_margin_crowd`, `_holder_conc`) + this campaign 2
= **5 declared hypotheses on this substrate**, and that is the count the deflation is charged with,
not 2. Rationale: the substrate and universe are identical, so charging only the new batch would let
a campaign-splitting habit launder multiplicity — exactly the defect `crucible-v9.0` (U5) shipped
`max(batch_pool, declared_hypotheses)` to close.

## 5. Stop rule for THIS campaign

Complete at 2 probes. **No Q3.** If Q1 fails, the informed-flow thesis on this band is recorded as
falsified and not re-probed with a different flow window or scaling — re-specifying the same
mechanism until it passes is the N-grab this rule exists to prevent.

## 6. Promotion ceiling (binds regardless of result)

`survivorship_free_required: true` and `tier2_audit_required: true` both BIND. A pass earns
**"PROMISING, forward-incubate"** and nothing more — no capital, no paper sleeve, no deploy-gating
verdict. The panel is the same current-listing-derived pool as P1 (whose measured survivorship
sensitivity was Δ IC-IR +0.001, so this is a small effect here, but the flag stands).

## 7. Results — **BOTH FAIL ON SIGN**; the informed-flow hypothesis is falsified

Run 2026-07-31, `results/taiwan_smallcap_institutional/scorecard.json`. Panel N=612, T=5292.
`institutional.parquet`: 1,613,781 rows, **595 of 612 band members (97.2%)**, 2014-01-02→2026-07-31.
Slot coverage 74.8% of active cells (the panel starts 2005, the T86 feed 2014 — expected, not a
defect). Multiplicity charged at **8**, the honest cumulative count on this substrate, rather than
the 5 this document named before R1/R2/S1 were run.

| @21d primary | Q1 `foreign_flow` | Q2 `trust_flow` |
|---|---|---|
| committed sign | +1 | +1 |
| **realized IC-IR** | **−0.120** | **−0.069** |
| IC t | **−6.57** | **−3.77** |
| frictionless Sharpe | −0.003 | +0.157 |
| net @ standard | −0.685 | −0.581 |
| cost wall | 0.682 | 0.738 |
| turnover/yr | 7.35 | 7.80 |
| recent-2y IC-IR | +0.311 | −0.442 |
| DSR / FDR-q | 0.0000 / 0.973 | 0.0000 / 0.973 |

IC ladders (both negative at every horizon out to 21d):
Q1 −0.028 / −0.094 / −0.126 / **−0.120** / −0.005 · Q2 −0.053 / −0.078 / −0.097 / **−0.069** / −0.011

### Verdict: FAIL on the pre-committed sign

§2 committed sign **+1** for both and stated that *"a signal whose realized IC has the opposite sign
is a FAIL, not a sign-flip opportunity."* Both realized **negative** ICs, Q1 at t = −6.57. **Both
FAIL.** The result is recorded as a falsification, not re-labelled as a reversal discovery.

Independently, neither is tradeable in either direction: Q1's frictionless Sharpe is −0.003 (its
book makes nothing before costs) and both are cost-blocked at standard cost (−0.685, −0.581) with
cost walls near 0.7.

### What is actually falsified

**The informed-foreign-flow hypothesis for the Taiwan small/mid band is falsified.** Foreign
institutional net buying — the most-cited Taiwan anomaly, and §1's whole mechanism — predicts
cross-sectional **underperformance** over the following month, not continuation.

§2's mechanism guard resolves cleanly, just not into either branch it anticipated. Both legs came out
**negative**, with the foreign leg roughly twice the magnitude of the trust leg (−0.120 vs −0.069).
Both moving the same way is the guard's "generic flow / price pressure" outcome rather than the
informed-trading one, and the magnitude ordering is consistent with it: foreign flows are the larger
ones, so they exert the most temporary price impact and show the most subsequent reversion. **That
falsifies the stated mechanism, exactly as §2 committed it would.**

### Ledger (durable — do not re-test)

- **Taiwan small/mid-cap FOREIGN institutional flow (21d, volume-normalised): FALSIFIED on sign.**
  Realized IC-IR −0.120 (t −6.57) against a committed +1. Do not re-propose informed-foreign-flow
  continuation in this band.
- **Taiwan small/mid-cap INVESTMENT-TRUST flow: FALSIFIED on sign** (−0.069, t −3.77).
- **Recorded as an observation for any FUTURE pre-registration, explicitly NOT claimed here:** the
  realized sign is negative and statistically strong at 5-21d. A price-pressure/reversal hypothesis
  on institutional flow would be a legitimate *new* pre-registration with its own committed sign —
  but it must be registered before it is measured, and it must clear the capturability bar that Q1
  fails outright (frictionless −0.003). Reading it off this run would be the sign-flip opportunism
  §2 exists to prevent.
- Per §5 the campaign is **complete at 2 probes. No Q3**, and no re-specification with a different
  flow window or normalisation.
