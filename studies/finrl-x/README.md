# FinRL-X audit

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investment), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan). Historical simulation and paper trading only; not investment advice.
> Video: https://youtu.be/hWqLBWK0aIQ

Does the "next generation of FinRL" build strategies you could trade? We ran AI4Finance's FinRL-X (the `FinRL-Trading` repository, arXiv 2603.21330) at commit `4409abe9`, with its own code, unmodified, and checked its README claims.

**Verdict:** FinRL-X's code runs and its own numbers reproduce: the Adaptive Rotation makes 4.91x from 2018 to 2025 (claimed 4.80x) at zero trading cost. But the backtest charges no cost, although the paper states 10 bps per side; at 10 bps it ends at 3.02x, below QQQ (3.95x), and its edge over QQQ survives only below about 4.5 bps per side. Its growth list is 2026's Magnificent 7, chosen in 2026 and backtested from 2018. Given lists an investor could have picked at the end of 2017, **0 of 35 beat QQQ**, and the 7 largest stocks of the day trail it by 7.22 points a year. The edge was the list. The Rolling Strategy (claimed 5.98x) cannot be reproduced from the public repository, and the +19.8% paper-trading record is a live account that cannot be re-run. Use Case 1 (allocation methods with a trend-timing overlay) shows a chart but no results table, and its timing code is not in the repository, so it was not tested.

## What was run

Every number here comes from FinRL-X's own code, unmodified, at commit `4409abe9`, run on dividend-adjusted prices from FinRL-X's own downloader (Yahoo Finance, free, no keys). Each run was first checked against the table FinRL-X itself prints, to the last printed digit: parity passed on all 37 runs. The cost-free, per-list and per-cost figures below are re-pricings of those runs, with trading costs applied per side.

Reproduction check (three lists plus QQQ):

| Growth list | Growth, no costs | Per year: no costs / 2 bps / 10 bps | vs QQQ at 2 bps |
|---|---|---|---|
| Magnificent 7 (as published) | 4.91x | 22.69% / 21.17% / 15.25% | +1.87 pts |
| 7 largest on 2017-12-29 | 2.66x | 13.42% / 12.08% / 6.87% | −7.22 pts |
| Best of 35 random lists | 3.95x | 19.31% / 17.89% / 12.37% | −1.41 pts |
| QQQ, buy and hold | 3.95x | 19.30% | |

Magnificent 7: trades 62.5x its capital a year (buys and sells counted); stops beating QQQ at 4.5 bps per side on return, 5.3 bps on Sharpe.

All 37 lists and the pre-registered verdict: 0 of 35 random lists beat QQQ at 2 bps; median −9.18 pts a year; verdict **HINDSIGHT**. The per-list results are in [`rerun-37-lists.txt`](rerun-37-lists.txt) (one line per list: parity check against FinRL-X's printed table, growth multiple, return a year at 0, 2 and 10 bps, difference from QQQ, and Sharpe at 2 bps).

Historical simulation, 2018-01-07 → 2025-10-24. All 37 lists reproduced FinRL-X's published results exactly when run on 2026-10-02/03 (cloud, 2 cores, about 3.2 hours). Yahoo occasionally revises its adjusted prices, so a rerun on later data can differ in the last digits. This study does not ship a one-command rerun; the data file above is the record.

## How the 2017 lists were chosen

- **Pool:** the 20 largest S&P 500 members on 2017-12-29 in the two sectors the Magnificent 7 belonged to then (Information Technology and Consumer Discretionary), sized with share counts from SEC filings made before that date: AAPL GOOGL MSFT AMZN META V HD INTC ORCL CMCSA CSCO DIS MA IBM MCD NVDA AVGO NKE TXN ACN.
- NVIDIA is in the pool (#16, $117B), which is generous to the strategy. Tesla is not: it joined the S&P 500 in December 2020.
- Limits: Electronic Arts and Twenty-First Century Fox could not be sized, and 145 end-2017 members that have since left the index cannot be priced from current data; any of them would have needed more than $100.5B (#20, ACN) to enter the pool.
- **Lists:** the Magnificent 7 as published; the 7 largest (AAPL GOOGL MSFT AMZN META V HD); 35 random 7-stock lists from the pool (seed 20260925, draw-spec sha256 `ebe2c118…`).
- **Rules written before the first run** (2026-09-25, 00:45 UTC, fingerprinted): HINDSIGHT if the Magnificent 7 beats 90% of the random lists and the median random list does not beat QQQ; ROBUST if the median random list and the 7 largest both beat QQQ; INCONCLUSIVE otherwise. Main measure: return above QQQ at 2 bps per side.

## Results (all 37 lists)

| Growth list | Per year, no costs | Per year, 2 bps | vs QQQ, 2 bps | Sharpe, 2 bps (vs QQQ) | vs QQQ, 10 bps |
|---|---|---|---|---|---|
| Magnificent 7 (published) | 22.69% | 21.17% | **+1.87 pts** | 1.01 (+0.10) | −4.05 pts |
| 7 largest on 2017-12-29 | 13.42% | 12.08% | −7.22 pts | 0.79 (−0.12) | −12.43 pts |
| 35 random lists: median | 11.24% | — | **−9.18 pts** | (−0.23) | −13.71 pts |
| 35 random lists: best | 19.31% | 17.89% | −1.41 pts | 0.91 (+0.00) | −6.93 pts |
| 35 random lists: worst | 8.01% | 6.83% | −12.47 pts | 0.51 (−0.40) | −17.07 pts |
| QQQ / SPY, buy and hold | 19.30% / 13.90% | | | 0.91 / 0.79 | |

**Why the list decides it.** The Rotation holds at most two stocks per group from a fixed list. QQQ is weighted by company size, so it owns more of new winners such as NVIDIA and Tesla as they grow; a list fixed in 2017 cannot.

**Trading costs** (Magnificent 7, per side):

| Cost | 0 bps | 2 bps | 3 bps | 5 bps | 10 bps | QQQ |
|---|---|---|---|---|---|---|
| Per-year return | 22.69% | 21.17% | 20.41% | 18.91% | 15.25% | 19.30% |
| Sharpe | 1.07 | 1.01 | 0.98 | 0.92 | 0.77 | 0.91 |

## What FinRL-X gets right

- **A sound design.** Every strategy ends in one target weight per stock per date, so modules swap cleanly. We scored FinRL-X's output without touching its code.
- **The code runs and is deterministic.** Re-runs are identical; the zero-cost numbers reproduce within about 2% on dividend-adjusted prices.
- **It is open source.** This audit was possible only because the code is public.
- **The fixes are straightforward:** charge costs in the P&L, apply the stop-losses (they are logged, not applied to returns), ship the Rolling Strategy's missing input files, and choose stock lists by a rule applied at each date.

## Follow-up: a strategy built on FinRL-X (keel-v1)

After the audit I asked the same AI (Claude, Opus 5.5) for a harder job: build a strategy on FinRL-X as the platform and test it once, on data nobody had touched. The rules, the windows, the pass/fail gates and the 84 variants it was allowed to try were committed before the one look, at 1973-01-02 to 2005-12-30.

| Measure (after costs, borrow fees, next-day execution) | keel-v1 | SPY |
|---|---|---|
| Sharpe | 0.44 | 0.33 |
| Return a year | 10.3% | 10.8% |
| Volatility | 8.8% | 16.1% |
| Max drawdown | -21.0% | -47.5% |
| 1973-74 / 2000-02 | -10% / -18% | -38% / -48% |

**Verdict: borderline NO-GO.** Alpha +1.15% a year with t 1.996 against the pre-registered bar of 2.0; deflated alpha 0.63 against 0.95; 6 of 8 gates passed; no second look. It beat SPY in 12 of 33 calendar years (strongest in crashes, lagging in booms). The trend sleeve's Sharpe of 0.60 as researched falls to 0.40 when traded the next day, financed at the T-bill rate and averaged over rebalance days. A forward lockbox is open, first check after 63 trading days (about late December 2026).

Full report: [keel-v1-final-report.md](keel-v1-final-report.md). Its file paths, such as `docs/research/finrlx_strategy_*` and `scripts/research/finrlx_strategy/`, are relative to the root of the Sharpen repository.

## Sources in this repo

- [`docs/research/finrl_x_final_report_2026-09-25.md`](../../docs/research/finrl_x_final_report_2026-09-25.md) (verdict)
- [`docs/research/finrl_x_reproduction_2026-09-24.md`](../../docs/research/finrl_x_reproduction_2026-09-24.md) (step 1: reproduction, code reading, cost grid)
- [`docs/research/finrl_x_hindsight_universe_2026-09-25.md`](../../docs/research/finrl_x_hindsight_universe_2026-09-25.md) (step 2: pre-registration and results)
- Scripts: [`finrl_x_rotation_repro.py`](../../scripts/research/finrl_x_rotation_repro.py), [`finrl_x_hindsight_runs.py`](../../scripts/research/finrl_x_hindsight_runs.py)

Not rerunnable from a fresh clone: [`scripts/research/finrl_x_pit_universe.py`](../../scripts/research/finrl_x_pit_universe.py), which built the 2017 pool, needs S&P 500 membership-history files that are not in the repo and an SEC user agent. The pool it produced is listed above, with every company's size and source in the step 2 doc.

FinRL-X: [github.com/AI4Finance-Foundation/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading) · paper: [arXiv 2603.21330](https://arxiv.org/abs/2603.21330)

Video: KISI Ep03, https://youtu.be/hWqLBWK0aIQ. Part of the KISI channel: see the source line at the top.
