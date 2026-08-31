# Value Falsification — Pre-Registered Spec (gates locked BEFORE data)

> **Created:** 2026-06-18 | **Session:** 553-cont-53 | **Status:** PRE-REGISTERED (committed before any value-signal data is computed — R0/R1 discipline)
> **Research artifact:** `.agent/artifacts/cross_asset_value_sleeve_research.md` (verdict CONDITIONAL GO)
> **Script (written AFTER this commit):** `scripts/research/value_falsification.py`
> **Outputs:** `results/value_falsification/{results.json, verdict.json, summary.md}`

## Question

Does a cheap, causal, net-of-cost **LINEAR VALUE** book (cross-asset long-horizon reversal, the AMP-2013 "value" leg) (a) survive realistic costs, and (b) **add to the validated TSMOM momentum core** via its negative correlation (a combined-portfolio Sharpe lift + tail improvement)? If value dies net of cost, or fails to lift the combo, it does NOT earn a slot — ship momentum-alone (+ rates-carry). If it survives AND is additive, it earns the second-factor build.

This mirrors `xsec_momentum_falsification.py` (the momentum Step-2, net Sharpe 0.60 GO) **verbatim** in construction, costs, and metrics, so the two books are directly comparable. Value's lead-candidate status rests on AMP's empirical **value–momentum correlation ≈ −0.5 to −0.6**: the binding gate is ADDITIVITY, and value is allowed a *lower* standalone bar than momentum/carry got because its job is diversification, not standalone strength.

## Universe (free, causal; price-only)

Same 18 ETF proxies as the momentum core (so the combined book is on the identical universe; the curation-inflation caveat applies equally to both sleeves, so the *relative* additivity test is robust to it):
- **Equity:** `SPY, QQQ, IWM, EFA, EEM`
- **Rates:** `TLT, IEF, LQD`
- **Commodity:** `GLD, SLV, DBC, USO, DBA`
- **FX:** `UUP, FXE, FXY, FXB, FXA`

Prices: reuse `results/xsec_momentum/prices_daily.parquet` (yfinance daily adjusted close, 2006-01-01 → 2026-06-01). The 5-year value lookback ⇒ first value signal ≈ 2011; the scored value sample is **2011 → 2026** (≈15y) — which deliberately **includes the post-2015 window where equity book-value value and naive carry both decayed** (the stress test).

## Value signal definition (price-only, causal-by-construction; LEAK-2 tripwire)

For each asset i at rebalance date t (using only data ≤ t):

```
value_raw(i, t) = log P_i(t − L_long)  −  log P_i(t − L_skip)
               = − [ cumulative log-return over the window (t−L_long , t−L_skip) ]
```
with **`L_long = 1260` trading days (~5y)** and **`L_skip = 252` trading days (~1y)**.

> **Erratum (same-commit, pre-data):** the first formula line originally read `log P(t−L_skip) − log P(t−L_long)`, which has a sign error (it equals **+**cumulative-return, the opposite of reversal). The unambiguous economic INTENT — stated in the prose immediately below and in the second formula line — is **value = −cumulative-return** (long the asset that FELL = cheap). Corrected to `log P(t−L_long) − log P(t−L_skip)` to match intent. No gate or economic content changed; only the transcription of the algebra.
- **Positive value_raw ⇒ the asset FELL over the 5y→1y window ⇒ "cheap" ⇒ candidate LONG.** Negative ⇒ rose ⇒ "expensive" ⇒ candidate SHORT (mean-reversion).
- The **1-year skip is load-bearing**: the momentum signal uses `[t−252, t−5]` (3/6/12m sign-ensemble), so `L_skip = 252` makes the value window **non-overlapping with the entire momentum window** ⇒ value and momentum are orthogonal by construction (the AMP −0.5 correlation is structural, not mechanical overlap).

Two books (mirroring momentum's TSMOM-pooled + XSMOM-within-class):
1. **XS-VALUE (primary, AMP-canonical):** at each t, rank assets **within each asset class** by `value_raw`; long the top third (cheapest), short the bottom third (most expensive); vol-scale each leg by `target_vol / realized_vol`. This is `xsmom_weights()` with `value_raw` replacing the 12-1m momentum rank. Within-class ranking (relative cheapness) avoids the secular-trend short-bleed of an absolute time-series sign.
2. **TS-VALUE (secondary):** per asset, `sign(value_raw − class_cross_sectional_mean(value_raw))`, vol-scaled (demeaned within class so it is relative, not an absolute mean-reversion-to-zero bet). Reported, not the primary gate input.

## Construction (verbatim from momentum harness `xsec_momentum_falsification.py`)

- Per-asset weight = signal × (target_vol/realized_vol), realized vol = 63-day rolling `.shift(1)` (causal), per-asset target 10% vol, `LEV_CAP = 2.0`.
- Rebalance: **monthly** (primary) + weekly (secondary), last trading day of period.
- **Execution lag T+1** (LEAK-2): weights computed at t from data ≤ t, applied to returns from t+1. Turnover and cost derive from the same weight series that generates P&L.
- Cost models: `frictionless 0.0`, `standard 2bps (0.0002)`, `harsh 10bps (0.0010)` one-way per unit turnover.
- Sharpe/PF/corr leverage-invariant on the raw net series; return/DD reported after a single constant scalar to 10% portfolio vol.

## Apples-to-apples baseline (MANDATORY)

The momentum-alone baseline for the additivity gate is **re-computed on the SAME post-2011 common window** as value (NOT the full-history 0.60). The combined book uses risk-parity (each sleeve scaled to 10% vol, then summed, then the combined book scaled to 10% vol) on the common window. All Gate-2 comparisons are window-matched.

## GATES (LOCKED — standard 2 bps, monthly, post-2011 common window unless stated)

**Gate 1 — standalone viability (deliberately lenient; "not broken"):**
- **1a** pooled XS-VALUE net Sharpe ≥ **0.20** (lower than momentum's 0.40 — value earns its slot via correlation, not standalone strength; below 0.20 it is too weak to help)
- **1b** ≥ **2** of {equity, rates, commodity, FX} classes net-positive Sharpe
- **1c** frictionless − net Sharpe gap ≤ **0.10** (value is slow/low-turnover; a large gap = a cost artifact)

**Gate 2 — ADDITIVITY to momentum (the BINDING gate):**
- **2a** corr(XS-VALUE pooled net daily, momentum pooled net daily, common window) < **+0.10** (must be genuinely non-positive; AMP expects ≈ −0.5. A positive correlation ⇒ value is redundant ⇒ NO-GO)
- **2b** combined risk-parity (value + momentum, each scaled to 10% vol) net Sharpe ≥ **momentum-alone-same-window + 0.05**
- **2c** combined max-DD@10%vol ≤ momentum-alone-same-window max-DD@10%vol (diversification must not worsen the tail)

**Gate 3 — persistence + tail honesty:**
- **3a** the COMBINED book net Sharpe positive in ≥ **3/4** subperiods (2011-13, 2014-16, 2017-20, 2021-26) **AND** combined > momentum-alone in the **post-2017 subperiods specifically** (the decay stress: value must still add value in the regime where carry/equity-value died)
- **3b** report XS-VALUE pooled skew + CVaR95 (informational; value is long-reversal, not short-vol — no hard skew gate)

**Leak tripwire (mandatory, hard):** causal (T+1) vs same-day-execution pooled XS-VALUE Sharpe gap < **0.10** (momentum's was 0.02); a planted forward-shift of the value signal must inflate rank-IC toward ~1.0 (negative control).

**Shuffle-null (mandatory, report):** XS-VALUE rank-IC vs the 95th percentile of a label-shuffled null; report whether value IC escapes its null (honest "is there signal at all" check, as in the PRISM evals).

## Decision

- **G1 ✓ AND G2 ✓ → GO.** Chain Architect: add `value_signal` to `sharpen/features/cross_asset_signals.py`; assemble TSMOM+value(+rates-carry) static risk-parity book in `allocator_factory.py`; then test an RL *sleeve-allocation* overlay under a hard beat-static gate (Protocol v2; Tier-2 audit pre-capital). Do NOT re-tune the frozen momentum core (voids the Fable chain) — value is purely additive.
- **G1 ✓ AND G2 ✗** (value survives but doesn't lift the combo) → value **NO-GO as a sleeve**; assemble the TSMOM + rates-carry frontier; report the honest 2-sleeve ceiling.
- **G1 ✗** (value decayed like equity book-value / naive carry) → value **NO-GO**; same fallback as above.

Either outcome is a valued deliverable. Falsify-before-optimize: **no env wiring / no RL** until the linear value sleeve clears G1+G2.

## Anti-p-hacking commitments

- Gates above are final; no post-hoc threshold moves. `L_long=1260`, `L_skip=252`, top/bottom third, monthly, 2bps are fixed here BEFORE data.
- Report ALL books (XS + TS, monthly + weekly, all cost models, all classes) in `results.json`; the verdict reads ONLY the pre-registered cells.
- The combined-book comparison is window-matched (post-2011); the curated-universe inflation caveat is acknowledged and applies equally to both sleeves.
