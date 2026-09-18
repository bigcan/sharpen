# Reproduce the headline results

The central claims of this repository re-run from a fresh clone in about a minute, on free public
data, with no API keys. Every figure below was checked this way against the published tree.

```bash
pip install -e ".[dev]"
```

Python 3.11+. The scripts download daily ETF prices from Yahoo Finance through `yfinance` on first
run and cache them under `results/`, which is gitignored. See [DATA.md](DATA.md) for the source's
terms.

---

## 1. The one survivor: cross-asset time-series momentum

```bash
python scripts/research/xsec_momentum_falsification.py
```

18 liquid ETFs across four asset classes, monthly rebalance, 2006–2026. Expected output (about 20
seconds, most of it the download):

```
GATE: pooled TSMOM monthly net Sharpe=0.601 (fric 0.619, gap 0.018), classes net-positive=4/4
DECISION: GO
```

Net Sharpe is at 2 bps per trade; the script also prints frictionless and 10 bps figures, and
cross-sectional momentum for comparison. Results go to `results/xsec_momentum/results.json`.

**0.601 is the flattering end of the range.** It is a total-return Sharpe (cash earns nothing, shorts
pay no borrow) on 18 ETFs chosen during development. Restated in excess of T-bills:

```bash
python scripts/research/tsmom_excess_return_check.py
```

```
pooled TSMOM net Sharpe, total return        0.601
  in excess of T-bills on net exposure        0.511
  ... and 50 bp/yr borrow on shorts           0.485
```

The same frozen rule on 32 ETFs never used in development scored **0.389**
([fable_verdict_2026-06-11.md](research/fable_verdict_2026-06-11.md)); that run's universe is not
cached by these scripts.

## 2. Why it still is not deployable: deflation

```bash
python scripts/research/audit_tailwind_book.py
```

Grades the book built on momentum plus a betting-against-beta sleeve against the pre-registered
deployment bar, using the prices cached by step 1. About 40 seconds. Expected:

```
R1 CLEARS (DSR>=0.95 AND PBO<=0.5): False
  DSR(N=24) = 0.896...  (min 0.95)   PBO = 0.00093...  (max 0.50)
VERDICT: BLOCK_multiplicity
```

The DSR is computed on the book with momentum cut to the untouched-universe **0.389**, not the
curated 0.601; on the curated book it would read 0.974 and pass. The project graded the honest
number. A probability of backtest overfitting near zero says the choice among the 18 *book types* in
the grid was not overfit; it does not measure universe curation, which is the larger risk here. A
deflated Sharpe below the bar says the edge is not large enough to be distinguished from the best of
24 trials with the required confidence. The two sleeves are combined with risk-parity weights from
their full-sample volatilities, a mild look-ahead in the combination step. Write-up:
[tailwind_v1_R1_dsr_pbo_2026-07-01.md](research/tailwind_v1_R1_dsr_pbo_2026-07-01.md).

## 3. The value test the epilogue is careful about

```bash
python scripts/research/value_falsification.py
```

Requires step 1's price cache. About 5 seconds. Expected:

```
  momentum-alone Sharpe        = 0.545   maxDD@10vol -22.54%
  value-alone Sharpe           = -0.364
  COMBINED Sharpe              = 0.126   maxDD@10vol -38.11%
DECISION: NO_GO_value_weak
```

This is a systematic cross-asset value *factor*, long/short and rebalanced. It is not a test of
fundamental value investing. Spec:
[value_falsification_spec_2026-06-18.md](research/value_falsification_spec_2026-06-18.md).

## 4. The power floor, no data needed

```bash
python scripts/research/planted_sweep.py --bar-only
python scripts/research/forward_power.py
```

The first is arithmetic and runs in about a second: the idealised minimum detectable ΔSharpe on 4
and 16 years of daily bars. The second measures the deployed validation contract on a synthetic
panel and takes about 13 minutes on a desktop CPU. See [METHODOLOGY.md §4](METHODOLOGY.md).

## 5. The validation funnel on synthetic data

```bash
python scripts/research/eval_signals.py --batch demo --panel "synthetic:1400,60" --out results/demo
```

Four demo signals, all `LOGGED`, in about 5 seconds. Column meanings are in
[guides/getting-started.md](guides/getting-started.md).

---

## If your numbers differ

- **Data revisions.** Yahoo's adjusted history can change when a fund pays or restates a
  distribution. Small differences in the third decimal are expected over time; the scripts pin the
  date range, not the vendor's adjustments.
- **A stale cache.** Delete `results/xsec_momentum/` and re-run step 1 to force a fresh download.
- **A real discrepancy.** Open a *Challenge a result* issue with your command, package versions and
  full output. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## What does not reproduce from this repository

Results that depended on paid or licensed data (CME futures, historical order books, some Taiwan
datasets) or on trained model checkpoints cannot be re-run from a clone. They are marked *not
shipped* in [NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md), and their scripts will ask for data you
have to obtain yourself.
