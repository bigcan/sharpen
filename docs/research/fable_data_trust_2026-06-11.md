# Fable Data-Trust Report — 2026-06-11

> **Mandate:** independent verdict on FinRL-Pro_DS. This report is the required
> interim checkpoint: how much can the existing data and results be trusted,
> re-established from raw bytes through a clean-room oracle built and unit-tested
> this session. **Nothing here relies on project pipeline code.**
>
> Verification artifacts: `scripts/research/fable/` (oracle + tests + audits),
> `results/fable_verdict/` (JSON/CSV evidence). Verdict doc:
> `docs/research/fable_verdict_2026-06-11.md`.

## Method

1. **Clean-room oracle** (`fable_oracle.py`): weight-based backtester +
   unit-position repricer + metrics, written from scratch, sharing zero code with
   the project. 14 unit tests pass, including a **bidirectional look-ahead
   tripwire** (same-bar signal must print money at `exec_lag=0` and die at
   `exec_lag=1`; future-peek signal must print money at `exec_lag=1` — proving
   the lag is exactly one bar).
2. **Raw-data audits** from bytes (`fable_data_audit_{etf,btc}.py`), with
   external cross-checks against independent venues/pulls.
3. **Re-derivation of one "survivor" and one "kill"** through the oracle, never
   through project code.

## 1. Raw data verdicts

### BTC 1-minute Bybit (`data/btc_usdt_1min_bybit.parquet`) — **TRUSTED**

- 1,225,440 rows, 2024-01-01 → 2026-04-30, **zero gaps** (every dt = 60s), zero
  duplicate stamps, zero OHLC-invariant violations, zero nonpositive prices,
  zero zero-volume bars; open-vs-prev-close max 3.0bps (venue-continuous).
- **External cross-venue check passes**: at 2025-11-30 16:30 UTC the local bar
  (91,704/91,722/91,590/91,672) sits ~5bps under Binance spot and OKX spot at
  the same epoch — normal perp/spot basis. The dramatic 2026-02-05 (−14.0%) and
  2026-02-06 (+12.2%) days are **real**, confirmed on Binance daily closes to
  within ~6bps. 2024-era extremes match known history (ETF sell-the-news
  Jan-12, Mar-19/20 pullback, Aug-08 carry-crash rebound).
- File timestamps are **UTC**, bar-START labeled.

### ETF daily (results/xsec_momentum/*.parquet, yfinance auto-adjust) — **CLOSES TRUSTED; repaired HIGH/LOW CORRUPTED**

- `prices_daily.parquet` (the GO-verdict input): 5,133×18, monotonic, no
  duplicate dates, no weekend rows, no interior NaNs (leading NaNs = inceptions).
- **Three pulls agree**: cached close-only (2026-06-03 pull), cached OHLCV raw
  (2026-06-05 pull), and a fresh pull today — daily returns identical to ~1e-7;
  zero ticker-days deviating >10bps between pulls.
- Extreme days are real history: EEM +22.8% / SPY +14.5% on 2008-10-13, USO ±25%
  March 2020, SLV −28.5% on 2026-01-30 (confirmed in today's fresh pull).
- ⚠️ **The "DATA-CLEAN outlier repair" corrupts daily wicks**: `clean_ohlcv`'s
  5% threshold was designed for 1-min bars; on daily bars it clamped 37 highs +
  50 lows of **real volatile days** (up to 21.7% of a low). Open/close/volume
  untouched (verified byte-level). → Any future use of daily high/low from
  `ohlcv_daily.parquet` (ATR, slippage models, intrabar stops) is **invalid**;
  close-based work is unaffected.
- Residual risks (acceptable, documented): single upstream source (Yahoo) for
  2025-26 dates; backward dividend adjustment makes prices vintage-dependent
  (returns are stable); curated-universe selection bias is a *results* risk, not
  a data risk — addressed in the verdict doc via an untouched-universe test.

## 2. Legacy-result re-derivations (the checkpoint's core question)

### "Survivor" — linear TSMOM GO (results/xsec_momentum, 2026-06-03) — **REPRODUCES EXACTLY**

Independent vectorized re-implementation of the spec (sign-momentum 3/6/12m,
skip-5, vol-scaled 10%/asset cap 2, month-end rebalance, T+1 execution) priced
through my oracle on the validated closes:

| Metric | Legacy claim | Fable clean-room |
|---|---|---|
| Pooled net Sharpe @2bps | 0.60 | **0.601** |
| Pooled PF @2bps | 1.116 | **1.116** |
| Frictionless Sharpe | 0.62 | 0.619 |
| Harsh 10bps Sharpe | 0.53 | 0.532 |
| Per-class Sharpe (eq/rates/cmd/fx) | .37/.39/.44/.27 | .374/.391/.436/.273 |
| Subperiods 06-09/10-15/16-20/21-26 | .57/.76/.23/.82 | .57/.76/.23/.82 |
| corr→SPY | 0.08 | 0.08 |

Added stresses the legacy test never ran (all survive, attenuated):
extra-day execution lag → 0.49; short-borrow 50bps → 0.58; borrow 100bps +
financing 50bps → 0.48; **everything-harsh (10bps + borrow + financing + lag2)
→ 0.30**. Parameter sweep: all 12 lookback×vol-window cells positive
(0.46–0.66). Block-bootstrap 95% CI [0.20, 1.02], P(Sharpe≤0)=0.001, t≈2.7
over 20.4yr. Same-bar-execution leak A/B gap: 0.022 Sharpe (no timing leak).
Honesty note: the raw book runs ~11× gross leverage at ~75% vol; scaled to a
10%-vol target it needs ~1.5× gross — implementable; Sharpe is scale-invariant.

### "Kill" — gmgp1-btc clean-canary FAIL (the V7 falsification) — **REPRODUCES EXACTLY**

Empirically discovered trajectory conventions (offset scan over ±40 bars):
stamps are **bar-END in UTC+8** (+31×15min vs UTC bar-start), `position` is
**leverage (notional/equity)**, P&L compounds multiplicatively — fit
correlation **1.00000** on 2,745 no-trade bars. Re-priced all 4 folds × 5 seeds
(+ ens_mean) on the validated Bybit closes with the costcorr fee layer (taker
5.5bps + slip 5bps):

| | Recorded | Fable re-priced |
|---|---|---|
| Median solo PF (net) | 0.9056 | **0.9043** |
| Solo PFs < 1.0 | 20/20 | **20/20** |
| Worst trailing DD | −30.5% | −30.5% |
| Median frictionless PF | — | 0.976 (13/20 < 1.0; best 1.07) |

Path tracking error: median max-deviation ≈ $426 on $100k over a month of
15-min bars (residual = intra-bar fill approximation). **The canary kill is
real market P&L, not an eval artifact: the trained policies lose money on
cross-venue-validated prices, before fees at best break even, with −25..−30%
drawdowns at ≤1.74× leverage.** The verdict that ended the V7 directional
program stands on independently re-established ground truth.

## 3. Trust map (what to rely on, what to void, what to rebuild)

| Asset | Trust | Basis |
|---|---|---|
| BTC 1m Bybit parquet | **HIGH** | internal pristine + 3-venue external match |
| ETF daily closes (all 3 caches) | **HIGH** | 3-pull agreement + real-history extremes |
| ETF daily high/low in `ohlcv_daily.parquet` | **VOID** | repair clamps real wicks — do not use |
| TSMOM falsification result | **HIGH** | exact clean-room reproduction + added stress |
| Canary FAIL verdict (+ trajectory format) | **HIGH** | corr-1.0 re-pricing; conventions now documented |
| Post-pivot eval/accounting layer | **HIGH (for these artifacts)** | recorded equity = real P&L on real prices |
| All pre-de-leak V7/SG-1 results (PF 2–3 era) | **VOID** | X2 leak + HPO selection contamination (project's own finding; consistent with everything above) |
| Options-VRP audit numbers (PF 1.06 band 0.6–1.1) | **MEDIUM** | adversarially audited by Tier-2 + recompute scripts, but NOT yet re-derived clean-room — gate before capital |
| Gold/EURUSD/other datasets | **UNAUDITED** | not touched this session; audit before any reuse |

**Bottom line:** the data layer and the *post-pivot* results layer are sound.
The catastrophe of the first ~500 sessions was the leak + leaky-HPO selection,
not corrupted data and not a broken eval. Both the project's lead positive claim
and its lead negative claim survive independent re-derivation — which means
conclusions drawn from here can be trusted *if and only if* they go through the
same clean-room discipline.
