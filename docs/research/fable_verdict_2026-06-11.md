# Fable Verdict — 2026-06-11

> Independent take-over review. Mandate: determine honestly whether this project
> has a real path to consistent net-of-cost out-of-sample profit — or call its
> time of death. Method: trust nothing produced here; re-establish ground truth
> from raw bytes through a clean-room oracle built and unit-tested this session.
>
> Companion (required interim checkpoint): `docs/research/fable_data_trust_2026-06-11.md`.
> Code: `scripts/research/fable/` · Evidence: `results/fable_verdict/`.

---

## VERDICT: **GO** — with a narrowed, honest mandate

There is a real, reproducible, implementable path to consistent net profit in
this project. It is **not** the path the first ~500 sessions pursued. The
original thesis — single-asset intraday directional RL, fast prop-firm passes,
PF 2–3 — is dead, and I re-verified its death on independently re-established
ground truth rather than taking the project's word for it. What survives my
clean-room re-derivation is the pivot the operator already chose: **a portfolio
of modest, economically-grounded, uncorrelated sleeves, linear-first, with RL
demoted to an optional overlay that must prove itself against a frozen linear
baseline.** The honest expectation is mid-single-digit annual return per 10%-vol
sleeve — real and compounding, not spectacular.

---

## 1. What I verified from scratch (and how)

**Clean-room oracle.** `fable_oracle.py` — weight backtester + position
repricer + metrics, zero shared code with the project, 14 unit tests including
a bidirectional look-ahead tripwire that pins execution timing to exactly one
bar. Everything below went through it.

**Raw data.** BTC 1-min Bybit: internally pristine (zero gaps in 1.22M minutes)
and matches Binance + OKX at identical UTC instants to ~5bps, including the
violent Feb-2026 ±14% days. ETF dailies: three pulls agree to 1e-7; extreme
days are real history. One landmine documented: the DATA-CLEAN "repair"
corrupts daily high/low wicks (5% threshold built for 1-min bars) — closes are
untouched; never use repaired daily H/L. Full detail in the data-trust report.

**The "kill" is real (canary re-priced).** I reverse-engineered the canary
trajectory conventions empirically (bar-END UTC+8 stamps; `position` =
leverage; multiplicative P&L — fit corr 1.00000 on 2,745 no-trade bars) and
re-priced all 20 seed-folds on validated prices with my own fee model:
recorded median PF 0.9056 vs mine 0.9043; 20/20 below 1.0 in both; worst DD
−30.5% in both; frictionless median 0.976 (best 1.07). **The policies lose on
real market prices — before fees they at best break even.** The falsification
that ended the V7 program was not an eval artifact. Combined with the X2 leak
mechanics (leaky PF 2.4–2.8 → de-leaked 0.80–1.02, collapse ordered by
coarse-octave size), the directional-RL era is closed on evidence, not vibes.

**The "survivor" is real (TSMOM re-derived).** Independent implementation of
the frozen linear rule on validated closes through my oracle reproduces the
legacy result to the third decimal: net Sharpe @2bps **0.601** (legacy 0.60),
PF 1.116 (=), per-class .37/.39/.44/.27 (=), subperiods .57/.76/.23/.82 (=),
corr→SPY 0.08 (=). It survives stresses the legacy never ran: +1-day execution
lag → 0.49; borrow+financing drag → 0.48; everything-harsh (10bps + borrow
100bps + financing + lag2) → **0.30**. All 12 parameter cells positive.
Bootstrap P(Sharpe≤0) = 0.001, t ≈ 2.7 over 20.4 years. Same-bar-execution
leak gap: 0.022 — no timing leak. At a 10%-vol target the book needs ~1.5×
gross leverage — implementable in ETFs, better in futures.

**Decisive experiment — pre-registered, run this session.** The last cheap
overfit channel was curated-universe selection. Same frozen rule on **32
liquid ETFs the project never touched** (9 sectors, 10 countries, 7
bonds/credit, 4 commodities, 2 REIT; mechanical inclusion rule):

| Gate (set before running) | Result | |
|---|---|---|
| G1: pooled net Sharpe @2bps ≥ 0.30 | **0.389** | PASS |
| G2: ≥60% instruments net-positive | **28/32 (87.5%)** | PASS |
| G3: frictionless−net gap ≤ 0.10 | **0.012** | PASS |

Reading it honestly: untouched 0.39 < curated 0.60, so curation flattered the
headline by ~0.2 Sharpe, and the untouched book's 2016-20 window is negative
(−0.09). The premium is structural (it generalizes), but the **honest planning
band is net Sharpe ~0.4–0.6**, with multi-year droughts a known property
(consistent with a century of trend-following literature).

**Second sleeve (medium trust).** Options-VRP short-straddle: their Tier-2
adversarial audit's recompute scripts re-ran clean in front of me today
(honest fully-costed Sharpe 1.06 / PF 1.18; band 0.6–1.1; VRP premium
positive-dominant Σ+8.86 vs Σ−1.79). I did **not** clean-room re-derive it
this session; it stays conditionally credible behind its NOW/NEXT buckets
plus one added gate below.

---

## 2. The decision, in operational terms

### KEEP / DEPLOY (the GO)

1. **Ship the frozen linear TSMOM core to paper immediately.** Curated-18
   universe, multi-lookback sign momentum, 10%-vol target (≈1.5× gross), monthly
   rebalance, T+1 execution. Do not wait for the RL allocator. Pre-registered
   deploy gate: ≥3 months paper with per-trade slippage reconciliation
   |realized−sim| < 5bps and equity tracking within the cost model; then small
   real capital. Expected reality at 10% vol: ~4–7%/yr, maxDD ~−20–25%,
   corr→SPY ≈ 0.08. On a $100k sleeve that is ~$4–7k/yr — say this number out
   loud and size effort accordingly.
2. **Options-VRP sleeve** proceeds behind its NOW bucket + NEXT bucket
   (margin model, real-chain validation) **plus one Fable gate**: a clean-room
   re-derivation of its daily P&L through an independent repricer (the same
   treatment the canary got) before any real capital. Paper sleeve OK after NOW.
3. **RL allocator** (`MultiAssetAllocatorEnv`, in build) continues **only** as
   an overlay behind the locked gate: nested walk-forward, must beat the frozen
   linear core by ≥ +0.05 net Sharpe pooled OOS with no fold worse than −0.10;
   else ship the linear core and stop. RL never discovers direction from raw
   prices again.
4. **The audit stack** (Tier-2 deep lifecycle audit, tripwire tests, LEAK-2,
   staged protocol v2, the falsify-before-optimize gate ordering) — this is the
   project's single most valuable asset. It caught the X2 leak; my independent
   re-verification confirms everything it certified post-pivot.

### KILL (do not revive without new information)

- **Single-asset intraday directional RL on price-derived features** — V7/SG-1
  multiscale, on BTC re-verified dead by me (frictionless ~breakeven, net PF
  0.90, −30% DDs). Any revival requires a genuinely NEW information axis that
  first clears a cheap pre-registered linear falsification net of costs.
- **Prop-firm fast-pass as the program yardstick.** A ~0.5-Sharpe monthly
  portfolio cannot pass a "10% in 30 days, 5% daily DD" challenge except by
  luck; the strategies that *looked* like they could were leak artifacts. Drop
  it, or fund it later from a separately validated high-frequency edge that
  does not exist today.
- **Every quantitative conclusion from the pre-de-leak era** (PF 2–3 wins, the
  old HPO selections, drift baselines on leaky obs) — already voided by the
  project; my re-derivations independently confirm the void was correct.

### WHY NOT "KILL EVERYTHING"

Because the one thing a kill verdict requires — *no demonstrable, reproducible,
implementable edge* — is contradicted by evidence I generated myself from raw
bytes: a 20-year, cost-stressed, leak-checked, parameter-robust, cross-universe
premium with a working data+eval+audit stack around it and a credible second
uncorrelated sleeve in the pipeline. Killing now would discard a verified asset
at exactly the moment the project finally has trustworthy ground truth.

### WHY NOT "PIVOT"

The pivot already happened (2026-06-02/03, operator-approved) and its first
gate is the thing my clean-room reproduction validated hardest. Demanding a new
paradigm would be churn, not rigor.

---

## 3. Honest framing the operator must accept with this GO

- The deployable expectation is **net Sharpe ~0.4–0.6 per sleeve**, droughts
  included (2016-20 was ~0 even in the curated book). "Consistently profitable"
  here means *positive expectancy compounding across years*, not positive every
  month. If that economic reality does not justify the operator's continued
  time at realistic capital, the rational move is to run the linear sleeve
  passively and stop active development — that is a portfolio-sizing decision,
  not a research failure.
- The 500-session sunk cost bought three real assets: a falsified hypothesis
  class (knowledge), a hardened audit/protocol stack (infrastructure), and a
  validated modest edge (the sleeve). Nothing more. Do not let the sunk cost
  inflate the next target.
- Free-data constraint holds: ETFs now; futures migration (better shorting,
  embedded financing, more instruments) is the natural upgrade and its costs
  are *lower*, not higher, than the stress numbers above.

## 4. Next session queue (in order)

1. Wire the linear core into the live paper harness (reuse the OANDA/IB stack),
   with the slippage-reconciliation logging defined above.
2. Fable clean-room repricer for the options-VRP daily P&L (canary treatment).
3. RL allocator Stage-3 nested WF vs frozen linear core — read behind a Tier-2
   audit, accept the gate result either way.
4. Quarterly: re-run `fable_untouched_universe.py` and the tripwire suite as a
   standing health check (one command, ~2 minutes, no GPU).

---

*Verification chain for every number in this document:*
`scripts/research/fable/fable_oracle.py` (+`test_fable_oracle.py`, 14 green) →
`fable_data_audit_etf.py` / `fable_data_audit_btc.py` →
`fable_tsmom_repro.py` → `fable_canary_reprice.py` →
`fable_untouched_universe.py` → JSON/CSV in `results/fable_verdict/`.
