# Carry Falsification — Pre-Registered Spec (gates locked BEFORE data)

> **Created:** 2026-06-12 | **Session:** 553-cont-45 | **Status:** PRE-REGISTERED (this doc committed before any signal data is touched — R0/R1 discipline)
> **Research artifact:** `.agent/artifacts/carry_signal_research.md` (verdict CONDITIONAL GO)
> **Script (to be written AFTER this commit):** `scripts/research/carry_falsification.py`
> **Outputs:** `results/carry_falsification/{results.json, verdict.json, summary.md}`

## Question

Does a cheap, causal, net-of-cost **LINEAR carry** book (a) survive realistic costs across asset classes, and (b) **add to the validated TSMOM momentum core** (low correlation + a combined-portfolio Sharpe lift)? If carry dies net of cost, or is positive-but-redundant with momentum, it does NOT earn a slot — ship momentum-alone. If it survives AND is additive, it earns the second-factor build.

This mirrors `xsec_momentum_falsification.py` (the momentum Step-2, net Sharpe 0.60 GO) **verbatim** in construction, costs, and metrics, so the two books are directly comparable. Carry faces a **stricter** gate than momentum did: standalone survival is necessary but not sufficient — it must also be additive.

## Universe (free, causal; signal data resolved at fetch)

Scored 3-class book (the pre-registered gate operates on these):
- **Equity** (tradeable return = ETF, yfinance): `SPY, QQQ, IWM, EFA, EEM`. Carry signal = trailing-12M dividend yield (from yfinance dividends) minus the US short rate.
- **Rates** (ETF): `SHY, IEF, TLT, LQD`. Carry signal = term-structure carry+roll proxy = curve slope (10y − 3m) and own yield level (FRED Treasury yields), longer-duration favored when the curve is upward-sloping.
- **FX — G10** (ETF return = currency ETF vs USD): `FXE(EUR), FXY(JPY), FXB(GBP), FXA(AUD), FXF(CHF), FXC(CAD)`; `UUP`=USD leg. Carry signal = (foreign short rate − USD short rate), cross-sectional long-high / short-low + TS sign.

Exploratory (NOT part of the scored gate; run data-permitting, reported separately): **EM-FX carry** `USD_TRY/ZAR/MXN` (OANDA practice spot, reuse `r1_illiquidity_fetch.py`; EM policy rates from FRED if reachable). This is the R1 sleeve candidate ([[project_r1_illiquidity_probe_weak_go_s553]]); included as a secondary probe because EM rate data is fragile.

**Excluded from v1:** commodity carry (roll yield needs front+second futures; ETFs hide the signal → v1.1 data-upgrade) and crypto funding carry (already decayed 0.39%/yr — `funding_arb_carry_falsification.py`).

Window: 2006-01-01 → 2026-06-01 daily (match momentum). Each class scored over its **valid-data window**; any FRED series that ends early is used only over its real range (no forward-fill of stale rates into the scored period — that would distort carry).

**Data sources:** ETF prices/dividends via yfinance (reuse `results/xsec_momentum/prices_daily.parquet` cache where possible). Rates/short-rates via **FRED free CSV** (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`, no API key). The fetcher is **defensive**: per signal it tries a list of candidate FRED IDs, takes the first returning valid data, and prints a data-availability report (exact IDs + date ranges resolved at runtime, logged into `results.json` — a plumbing detail, NOT a gate input).

## Construction (verbatim from momentum harness)

- Per-asset weight = signal × (target_vol / realized_vol), realized vol = 63-day rolling, `.shift(1)` (causal), per-asset target 10% vol, `LEV_CAP = 2.0`.
- Rebalance: **monthly** (primary) + weekly (secondary), last trading day of period.
- **Execution lag T+1** (LEAK-2): weights computed at rebalance t from data ≤ t, applied to returns from t+1. Turnover and cost derive from the same weight series that generates P&L.
- Cost models: `frictionless 0.0`, `standard 2bps (0.0002)`, `harsh 10bps (0.0010)` — one-way per unit turnover.
- Sharpe/PF/corr leverage-invariant on the raw net series; return/DD reported after a single constant scalar to 10% portfolio vol.

## Carry signal definitions (causal-by-construction; look-ahead tripwire each)

- **FX carry** `c_fx[t,i] = r_foreign_i[t] − r_usd[t]` using the latest short rate stamped ≤ t (rates forward-filled to daily WITHIN their valid range only). Cross-sectional: rank G10 by `c_fx`, long top / short bottom, vol-scaled; also TS sign(`c_fx`). Higher rate = positive expected carry.
- **Rates carry** `c_rt`: per bond ETF, signal = sign/size of the curve slope (`DGS10 − DGS3MO`) blended with own-tenor yield level, ≤ t; longer duration (TLT) gets more weight when the curve is upward and positive. Vol-scaled.
- **Equity carry** `c_eq[t,i] = divyield_i[t] − r_usd[t]`, trailing-12M dividends / price, ≤ t; cross-sectional rank (high-yield long / low-yield short), vol-scaled.
- Each signal: a negative test that the signal at t uses no data > t (planted-shift tripwire must inflate IC to ~1.0; correct signal must show near-zero same-day-vs-causal gap).

## GATES (LOCKED — standard 2 bps, monthly, pooled unless stated)

**Gate 1 — standalone viability** (mirror momentum):
- **1a** pooled carry net Sharpe ≥ **0.40**
- **1b** ≥ **3** of {equity, rates, FX-G10} classes net-positive Sharpe
- **1c** frictionless − net Sharpe gap ≤ **0.15** (edge is not a cost artifact, inverted)

**Gate 2 — ADDITIVITY to momentum** (the carry-specific gate; the reason a 2nd factor exists):
- **2a** corr(carry pooled net daily series, momentum pooled net daily series) < **0.30**
- **2b** combined risk-parity (50/50, each scaled to 10% vol) momentum+carry net Sharpe ≥ **0.65** (= momentum-alone 0.60 + 0.05)
- **2c** combined max-DD@10%vol ≤ momentum-alone max-DD@10%vol (diversification must not worsen the tail)

**Gate 3 — tail honesty + persistence** (report-and-flag; 3b is a soft gate):
- **3a** report pooled carry skew + CVaR95; **FLAG** if skew < **−1.0** (short-vol tail → sizing caveat for the architect)
- **3b** carry pooled net Sharpe positive in ≥ **3/4** subperiods (2006-09, 2010-15, 2016-20, 2021-26) — naive-carry-decay check

**Leak tripwire** (mandatory, hard): causal (T+1) vs same-day-execution pooled carry Sharpe gap < **0.10** (momentum's was 0.02).

## Decision rule (locked)

| Outcome | Condition | Action |
|---|---|---|
| **GO** | Gate 1 PASS **and** Gate 2 PASS (leak tripwire clean) | Carry earns the 2nd-factor slot → chain Architect: `carry_signal` in `cross_asset_signals.py`, FRED branch in `cross_asset_loader.py`, feed `carry_ary` (env slot already wired, ADR-6), ship a frozen linear **momentum+carry** core as the new RL beat-linear baseline. |
| **CARRY_REDUNDANT** | Gate 1 PASS **and** Gate 2 FAIL | Carry is positive but redundant/dilutive with momentum (the XSMOM lesson) → ship momentum-alone, document. |
| **NO_GO** | Gate 1 FAIL | Carry dies net of cost in this universe → document, reassess (commodity-included v1.1, or valuation-adjusted carry overlay). |

Honest expectation if GO: combined net Sharpe ~**0.65–0.8**, low corr→SPY, with a flagged short-vol tail. Do NOT re-tune the frozen momentum core (voids the Fable chain).

## Anti-p-hack commitments
- Gates above are final; no threshold edits after seeing results.
- Carry signal definitions are final (the canonical KMPV per-class definitions); no signal search.
- The exact FRED series IDs are a data-plumbing detail resolved at fetch (logged), NOT a tuned input.
- Verdict is programmatic from `verdict.json`; the run reports whatever the gates compute.
