# PRE-REGISTRATION — Taiwan small/mid-cap PRICE-ONLY probes (R1/R2)

**Written:** 2026-07-31 (S553-cont-146), **BEFORE any signal was computed or any IC was seen.**
**Substrate:** `data/taiwan_smallcap/` cap-rank 51-250 band — the same locked universe and panel
builder as the 2026-07-15 campaign. **Zero data fetch:** both signals derive from `prices.parquet`,
already on disk. **Harness:** the locked `finrl_pro_ds/signals/` scorecard, thresholds unchanged.

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

## 6. Results

*(Empty at commit time on purpose — the sign commitment is verifiable from git history.)*
