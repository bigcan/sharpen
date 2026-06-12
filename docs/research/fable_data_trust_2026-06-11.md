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

---

## Addendum (same day) — breadth pass over the remaining strategy verdicts

`fable_reverify_all.py` extended the re-pricing to the two remaining decisive
walk-forward verdicts with artifacts on disk.

### sg1-btc de-leaked cost-corrected X1 WF — **KILL REPRODUCES**

Certified at `wallclock +31 bars` (93 min at 3-min bars), **100.0% of 3,068
single-bar no-trade rows machine-exact** (median residual 5e-17). All 8 folds ×
3 seeds re-priced: recorded median solo PF 0.9792 vs mine 0.9817; frictionless
0.9923; **0/24 reach the 1.1 deploy bar in either accounting**. Second
directional kill independently confirmed. (The +31-bar stamp shift recurs in
every dataset — it is a trajectory-writer offset in *bars*, not a timezone;
the canary's "UTC+8" reading was the 15-min special case of the same shift.)

### gmgp1-gold de-leaked X2 WF — accounting HONEST, **dataset CORRUPT**

Three findings, in order of discovery:

1. **Config↔data provenance gap.** The WF config names
   `data/cme/gc_2025_lob1_1min_stitched.parquet`, but no alignment against that
   file (or any 1-min sibling) explains the trajectories — it is missing ~22-35%
   of trading minutes *including event minutes* (e.g. 13:30 UTC data releases).
   The env's actual mark-to-market series is
   `data/cme/gold_2025_2026q1_15min_stitched.parquet`, certified at `seq +31`
   with **99.7% machine-exact rows** (p95 residual 1.3e-16).
2. **The recorded numbers are honest — in fact conservative — on that dataset.**
   My reconstruction (fee-only) gives slightly *higher* PFs than recorded
   (recorded net includes spread/slippage I don't model). The
   "real-but-decaying" shape replicates: folds 0-1 profitable, folds 2-3
   breakeven, in both accountings.
3. **But the dataset itself contains confirmed corrupt segments.** Friday
   2025-08-15 21:00 prints a flat-OHLC bar at **3521.4 (+5.0% in one bar)**
   followed by a weekend and reversion to ~3360 — Yahoo GC=F shows no such move
   (3336 that day); a second flat-OHLC stale print sits Sunday 12:00. The
   recorded P&L of the BEST fold accrues **$5,053 of its $24,062 (~21%) across
   exactly that window**. A second suspect: stitched +4.4% on 2025-09-30 vs
   Yahoo +0.5%. October's violent moves DO verify externally (real). →
   **Gold's "real edge in favorable regimes" claim is void pending clean data**;
   the decay-to-breakeven conclusion and the non-deployment verdict stand.
4. **Cleaner blind spot (root cause).** `clean_ohlcv` validates *intra-bar*
   consistency only; a flat-OHLC stale print (O=H=L=C) passes all its checks,
   and there is no inter-bar continuity check to catch a +5% jump that exactly
   reverts. Any future data prep needs a jump-and-revert / stale-quote detector
   and an external cross-source check on extreme days.

### Trust-map updates

| Asset | Trust | Change |
|---|---|---|
| sg1-btc de-leak FAIL verdict | **HIGH** | re-priced, kill confirmed |
| `gold_2025_2026q1_15min_stitched.parquet` | **VOID for eval** | confirmed stale-print corruption inside profitable windows |
| gmgp1-gold "real-but-decaying" | **UNVERIFIED / likely inflated** | profits partly ride corrupt bars; do not cite as "directional RL almost works" |
| gmgp1-gold eval accounting | HIGH | recorded ≤ independent reconstruction (conservative) |
| sg1-eurusd Stage-3 PROMOTE | **VOID by contamination** | obtained on the leaky shared handler, never re-tested de-leaked — must not deploy |
