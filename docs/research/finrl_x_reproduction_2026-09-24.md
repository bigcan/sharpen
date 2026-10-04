# FinRL-X reproduction — 2026-09-24

**Subject:** AI4Finance-Foundation `FinRL-Trading`, rebranded FinRL-X (arXiv 2603.21330, submitted 2026-03-22),
cloned at `4409abe9` (master, 2026-09-18) to `C:/FinRL/FinRL-Trading`. Branch `finrl-x-eval`.
**Question:** do the README / paper headline numbers reproduce from FinRL-X's own code? This is the step before any
funnel test; nothing here scores the strategies against our nulls.

## Verdict

- **Adaptive Rotation — the zero-cost return reproduces; the stated costs are not in the code, and the edge over QQQ
  survives only cheap execution.** On dividend-adjusted prices FinRL-X's own code gives 4.91x (CAGR 22.7%) against the
  claimed 4.80x (22.3%). Risk comes out somewhat worse than claimed (vol 22.1% vs 20.3%, weekly max DD −23.6% vs
  −21.5%), so their Sharpe is 1.03, not 1.10. The strategy trades **62.5x its capital a year** (buys + sells; ~31x
  one-way). At the paper's own 10 bps per side it ends at 3.02x (CAGR 15.3%, Sharpe 0.69 by their definition, 0.77
  arithmetic): **below QQQ buy-and-hold** (3.95x, 0.84 / 0.91). The break-even against QQQ is **4.5 bps per side**
  on CAGR and **5.3 bps** on Sharpe. Our own cost priors bracket it: the US-equity funnel prices liquid large caps at
  2 bps, where the strategy still edges QQQ (CAGR 21.2% vs 19.3%, Sharpe 1.01 vs 0.91), while `sharpen/signals/costs.py`
  defaults single names to 10 bps. Costs make the in-sample edge thin, not absent; the hindsight universe (finding 6)
  is the decisive open test.
- **Rolling Strategy — not reproducible from the public repo** (finding 8).
- **Following the README as shipped does not give their table:** `deploy.sh` downloads price-only data, so the
  benchmarks come out at 3.75x / 2.44x instead of 4.02x / 2.80x, and the strategy picks change (5.08x at zero cost).
- **The one clean window is positive.** From 2026-02-06 (the day after the config was frozen) to 2026-09-18 the
  unchanged config made 1.21x after 10 bps (arithmetic Sharpe 2.68, weekly max DD −3.4%) vs QQQ 1.19x (1.50) and SPY
  1.11x (1.27). That is one look at 29 weekly returns (t ≈ 2.1 on its own Sharpe, untested against QQQ), and its
  return has not yet been factor-attributed.

## Claims under test

README and paper, backtest 2018-01-07 → 2025-10-24, "10 bps per side" costs, FMP data:

| | Cumulative | Ann. return | Ann. vol | Sharpe | Max DD | Calmar | Win rate |
|---|---|---|---|---|---|---|---|
| Rolling Strategy (NASDAQ-100 top 25% + DRL) | 5.98x | 25.85% | 27.85% | 0.93 | −38.95% | 0.66 | 54.36% |
| Adaptive Rotation | 4.80x | 22.32% | 20.30% | 1.10 | −21.46% | 1.04 | 54.77% |
| QQQ | 4.02x | 19.56% | 24.20% | 0.81 | −35.12% | 0.56 | 56.25% |
| SPY | 2.80x | 14.14% | 19.61% | 0.72 | −33.72% | 0.42 | 55.28% |

Paper trading 2025-10-26 → 2026-03-12 (ensemble of both, daily rebalance): +19.76%, Sharpe 1.96, max DD −12.22%.

## Method

- **Environment:** Python 3.11.9 venv inside the clone. `requirements.txt` does not install as published: it pins
  `finnhub>=2.4.19`, which is not a PyPI distribution (the package is `finnhub-python`; nothing imports it).
  PyPI downloads ran at ~40 kB/s, so only the Adaptive Rotation's dependencies were installed first (the ones
  `deploy.sh` checks for, plus pydantic and matplotlib): pandas 3.0.6, numpy 2.4.6, yfinance 1.7.0,
  pandas-market-calendars 5.4.0. Requirements are unpinned `>=`, so these are what their README yields today.
- **Code path:** FinRL-X's `deploy.sh`, unmodified, driven by `scripts/research/finrl_x_run_rotation.sh`, which
  points `deploy.sh`'s bare `python3`/`pip3` at the venv and sets `PYTHONUTF8=1` (their docs assume macOS, where UTF-8
  is the default; Windows would read the YAML as cp950). Config `AdaptiveRotationConf_v1.2.1.yaml` (the `deploy.sh`
  default), weekly `W-FRI`, daily fast-track on.
- **Two price bases:** (a) exactly as `deploy.sh` ships — Yahoo with `auto_adjust=False`, i.e. price-only closes;
  (b) `deploy.sh`'s own downloader, extracted at run time with only `auto_adjust=True`, i.e. dividend-adjusted closes.
  ⚠ `deploy.sh --skip-download` is ignored when the data directory does not exist: it then refills it with its own
  price-only download, silently swapping the basis. The script creates the directory first.
- Re-running the paper window through the script from a fresh download matched the original run to ~1e-8.
- **Independent re-derivation:** `scripts/research/finrl_x_rotation_repro.py` recomputes FinRL-X's equity curve from
  its exported weights and the same price files, asserts parity with the table FinRL-X printed (to half a printed
  unit; a missing or incomplete table fails), then re-prices the same weights across a per-side cost grid (turnover
  measured against drifted weights) and finds the break-even cost against QQQ. It also reports two numbers FinRL-X does
  not: the arithmetic Sharpe (rf = 0, as theirs) and the max drawdown on daily marks. Nine tests in
  `tests/research/test_finrl_x_rotation_repro.py` pin each property to a hand-computed value; each of eight deliberate
  bugs (target-to-target turnover, cash-free drift, no cost, weekly-only marks, a loose or fail-open parity check, a
  flipped break-even boundary, no all-cash guard) fails exactly the tests predicted for it.
- Every run archives FinRL-X's stdout, weights, summary, chart, the exact price files and a provenance file under
  `results/finrl_x/repro/<variant>/` (gitignored).

## Results

Adaptive Rotation v1.2.1, FinRL-X code unchanged. Every 0 bps row and benchmark passes parity with the table FinRL-X
printed; the cost rows are our re-pricing of the same weights.
Sharpe (FX) is FinRL-X's CAGR / vol; Sharpe (ar.) is mean weekly return × 52 / vol. Max DD is weekly-marked, with the
daily-marked value in brackets.

| Window / series | Cumulative | CAGR | Vol | Sharpe (FX / ar.) | Max DD | Turnover/yr (two-sided) |
|---|---|---|---|---|---|---|
| **2018-01-07 → 2025-10-24, dividend-adjusted** | | | | | | |
| Claimed (README) | 4.80x | 22.32% | 20.30% | 1.10 / – | −21.46% | – |
| Reproduced, 0 bps | 4.91x | 22.69% | 22.06% | 1.03 / 1.07 | −23.55% (−26.11%) | 62.5x |
| Reproduced, 10 bps/side | 3.02x | 15.25% | 22.00% | 0.69 / 0.77 | −24.75% (−26.49%) | 62.5x |
| QQQ (claimed 4.02x, 0.81) | 3.95x | 19.30% | 22.85% | 0.84 / 0.91 | −35.06% | – |
| SPY (claimed 2.80x, 0.72) | 2.75x | 13.90% | 19.35% | 0.72 / 0.79 | −31.83% | – |
| **Same window, price-only (`deploy.sh` as shipped)** | | | | | | |
| Reproduced, 0 bps | 5.08x | 23.24% | 22.10% | 1.05 / 1.08 | −23.26% (−25.83%) | 62.4x |
| Reproduced, 10 bps/side | 3.13x | 15.78% | 22.04% | 0.72 / 0.79 | −24.51% (−26.22%) | 62.4x |
| QQQ / SPY | 3.75x / 2.44x | 18.52% / 12.13% | | 0.81 / 0.62 (FX) | | |
| **Paper-trading window 2025-10-26 → 2026-03-12** (Rotation alone; their live ensemble made 1.20x) | | | | | | |
| Reproduced, 10 bps/side | 1.12x | | | 1.80 / 1.62 | −6.65% (−9.12%) | 74.2x |
| QQQ / SPY | 0.95x / 0.99x | | | | | |
| **Post-freeze 2026-02-06 → 2026-09-18** (config committed 2026-02-05, unchanged since) | | | | | | |
| Reproduced, 0 bps | 1.24x | | 13.33% | 3.19 / 3.00 | −3.26% (−4.64%) | 40.6x |
| Reproduced, 10 bps/side | 1.21x | | 13.24% | 2.78 / 2.68 | −3.42% (−4.79%) | 40.6x |
| QQQ / SPY | 1.19x / 1.11x | | 21.97% / 15.83% | 1.46 / 1.19 (FX) | −7.60% / −7.93% | |

**Cost sensitivity** (per side; CAGR / arithmetic Sharpe) and the break-even per-side cost at which the strategy falls
to QQQ:

| Window | 0 bps | 2 bps | 3 bps | 5 bps | 10 bps | QQQ | Break-even (CAGR / Sharpe) |
|---|---|---|---|---|---|---|---|
| Full, dividend-adjusted | 22.69% / 1.07 | 21.17% / 1.01 | 20.41% / 0.98 | 18.91% / 0.92 | 15.25% / 0.77 | 19.30% / 0.91 | 4.5 / 5.3 bps |
| Full, price-only | 23.24% / 1.08 | 21.71% / 1.03 | 20.95% / 1.00 | 19.45% / 0.94 | 15.78% / 0.79 | 18.52% / 0.88 | 6.3 / 7.0 bps |
| Post-freeze | 42.47% / 3.00 | 41.32% / 2.93 | 40.74% / 2.90 | 39.61% / 2.84 | 36.80% / 2.68 | 32.10% / 1.50 | 18.6 / 45.1 bps |

- Turnover is two-sided (Σ|Δw| over buys and sells), so cost = turnover × bps per side; 62.5x × 10 bps ≈ 6.3%/yr.
- The cost rows keep FinRL-X's same-close execution (finding 7), so they are still optimistic on timing.
- Turnover is real, not an artefact of drift: target-to-target Σ|Δw| averages 1.24 per weekly rebalance (about 2.8 of
  ~5 names replaced), and only 4% of weeks leave the book unchanged.
- The remaining ~2% benchmark gap on adjusted data (QQQ 3.95x vs 4.02x, SPY 2.75x vs 2.80x) is consistent with
  Yahoo-vs-FMP dividend adjustment or a slightly different start date.
- The price-only variant was run twice (identical tables). Run environment: pandas 3.0.6, numpy 2.4.6, yfinance 1.7.0;
  price files archived with each run.

## Findings in the code

1. **No transaction costs.** `_generate_performance_report` compounds Σ wᵢ·(p₁/p₀ − 1) between rebalance rows with no
   cost term (`src/strategies/run_adaptive_rotation_strategy.py:443-456`). The paper states 10 bps per side.
2. **The daily risk layer never reaches the P&L.** Stop-loss and fast risk-off events are logged and only update the
   position tracker; neither appends a row to the weights (`run_adaptive_rotation_strategy.py:268-311`). The equity
   curve holds each Friday's weights for the full week. Stops still affect later weeks through the re-entry
   cooldown, so the curve is neither "with stops" nor "without stops".
3. **Sharpe is CAGR / volatility** with no risk-free rate (`run_adaptive_rotation_strategy.py:476`). Every published
   Sharpe in the README matches this definition (e.g. 22.32% / 20.30% = 1.10).
4. **Weekly marks.** Volatility, drawdown and win rate come from weekly points (`:474-481`); the published figure
   plots daily values, so the figure and the table come from different code, and the figure's code is not in
   the repo. The report function itself was added on 2026-03-25, one day after the figures were committed.
5. **Price basis mismatch.** `deploy.sh:308` downloads price-only closes, but the published benchmark columns are
   total return: on the same Yahoo data QQQ gives 3.75–3.81x and SPY 2.44–2.48x price-only, against the claimed
   4.02x and 2.80x. A user who follows the README cannot reproduce the table.
6. **Hindsight universe.** The growth group is AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA
   (`AdaptiveRotationConf_v1.2.1.yaml:43-52`), i.e. 2026's Magnificent 7, committed on 2026-02-05 and backtested
   from 2018. It includes NVDA and TSLA, two of the period's biggest winners.
7. **Same-close execution.** Weights decided from Friday's close earn returns from that same close.
8. **The Rolling Strategy cannot be reproduced from the public repo.** Its DRL allocator
   (`src/strategies/fundamental_portfolio_drl.py:367-413`) reads two files that are not shipped
   (`./data_processor/sp500_tickers_daily_price_20250712.csv`, `./result/stock_selected.csv`) and needs `finrl`,
   `stable_baselines3`, `gym`/`gymnasium` and `pypfopt`, none of which are in `requirements.txt`. Its NASDAQ-100
   universe comes from FMP's *current* constituent list (`src/data/data_fetcher.py`, `fetch_nasdaq100_tickers`), with
   no membership history, so a 2018 selection would draw from 2026's index. The example notebook
   (`examples/FinRL_Full_selection.ipynb`) has the rolling call commented out and its backtest cell errors.
9. **Paper trading is a live record**, not reproducible from code. Their own chart attributes the gain to one
   "Precious Metals & Mining Stocks Rally" (Dec 2025 – late Jan 2026).

FinRL-X's code runs and its own numbers reproduce. This is
not a substitute for the Tier-2 lifecycle audit.

## Next (funnel, Adaptive Rotation only)

**Update 2026-09-25:** item 2 ran pre-registered and returned HINDSIGHT — 0 of 35 growth groups drawn from 2017's 20
largest tech/consumer names beat QQQ, so the Rotation is NO-GO and the items below are moot. See
`finrl_x_hindsight_universe_2026-09-25.md`.

1. Score at our own cost priors (2 bps for liquid large caps per the US-equity funnel; 10 / 25 bps standard / harsh
   single-name defaults in `sharpen/signals/costs.py`) and always report the break-even; at 62x turnover it sits at
   ~5 bps per side, so cost decides the sign against QQQ.
2. **Hindsight-universe null:** replace the growth group with a point-in-time list (largest tech / communication names
   as of each rebalance, or as of 2017-12-31) and change nothing else.
3. **Factor attribution:** regress weekly returns on the sleeves it rotates through (QQQ, SPY, GLD, TLT) and feed the
   funnel the factor-hedged series. The post-freeze window needs this before it means anything.
4. One-day execution lag (decide on Friday's close, trade the next close).
5. Keep extending the post-freeze window forward with no re-tuning; it is the only data this config has not seen.
6. Rolling Strategy: only with the two unshipped inputs and a NASDAQ-100 membership history; otherwise drop it.
