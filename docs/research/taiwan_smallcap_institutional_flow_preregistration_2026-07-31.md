# PRE-REGISTRATION — Taiwan small/mid-cap institutional-flow probes (Q1/Q2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE any signal was computed or any IC was seen.**
**Substrate:** `data/taiwan_smallcap/` cap-rank 51-250 band, the same locked universe as the
2026-07-15 campaign. **Harness:** `scripts/research/taiwan_smallcap_altdata_eval.py`, unchanged.

## 0. Why this is a new campaign, not the barred Round 2

The 2026-07-15 pre-registration's §5 stop rule closed **that** campaign at 3 probes and barred its
Round 2 (`value/book-to-market` + `分點` branch-concentration). Those two channels are **not** probed
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
- **Q2 IS A MECHANISM TEST, NOT A SECOND SHOT AT A HIT.** Domestic investment trusts (投信) are
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

## 7. Results

*(To be filled in after the run. This section was empty when the file was committed —
see the commit that introduced it.)*
