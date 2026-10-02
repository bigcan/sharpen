# FinRL-X audit

Does the "next generation of FinRL" build strategies you could trade? We ran AI4Finance's FinRL-X (the `FinRL-Trading` repository, arXiv 2603.21330) at commit `4409abe9`, with its own code, unmodified, and checked its README claims.

**Verdict:** FinRL-X's code runs and its own numbers reproduce: the Adaptive Rotation makes 4.91x from 2018 to 2025 (claimed 4.80x) at zero trading cost. But the backtest charges no cost, although the paper states 10 bps per side; at 10 bps it ends at 3.02x, below QQQ (3.95x), and its edge over QQQ survives only below about 4.5 bps per side. Its growth list is 2026's Magnificent 7, chosen in 2026 and backtested from 2018. Given lists an investor could have picked at the end of 2017, **0 of 35 beat QQQ**, and the 7 largest stocks of the day trail it by 7.22 points a year. The edge was the list. The Rolling Strategy (claimed 5.98x) cannot be reproduced from the public repository, and the +19.8% paper-trading record is a live account that cannot be re-run.

## Rerun it (about 10 minutes on 3 or more cores, CPU only, free Yahoo data, no keys)

```bash
git clone https://github.com/bigcan/sharpen && cd sharpen
pip install -e ".[dev]"
python scripts/research/finrl_x_rerun.py
```

[`scripts/research/finrl_x_rerun.py`](../../scripts/research/finrl_x_rerun.py) clones FinRL-X at `4409abe9` into its own folder and virtual environment, downloads dividend-adjusted prices with FinRL-X's own downloader, runs FinRL-X's own Adaptive Rotation backtest with three growth lists, and re-prices each run with trading costs. Each run must first match the table FinRL-X itself prints, to the last printed digit. It needs `git` and an internet connection; one backtest takes about 10 minutes on one core, so the three lists take about 30 minutes on a single core. Expected output:

| Growth list | Growth, no costs | Per year: no costs / 2 bps / 10 bps | vs QQQ at 2 bps |
|---|---|---|---|
| Magnificent 7 (as published) | 4.91x | 22.69% / 21.17% / 15.25% | +1.87 pts |
| 7 largest on 2017-12-29 | 2.66x | 13.42% / 12.08% / 6.87% | −7.22 pts |
| Best of 35 random lists (`rand_02`) | 3.95x | 19.31% / 17.89% / 12.37% | −1.41 pts |
| QQQ, buy and hold | 3.95x | 19.30% | |

Magnificent 7: trades 62.5x its capital a year (buys and sells counted); stops beating QQQ at 4.48 bps per side on return, 5.34 bps on Sharpe.

All 37 lists and the pre-registered verdict (about 6 CPU-hours; `--jobs` sets how many lists run in parallel, and 12 at a time took 24 minutes on a 16-core desktop):

```bash
python scripts/research/finrl_x_rerun.py --full --jobs 8
```

Expected: 0 of 35 random lists beat QQQ at 2 bps; median −9.18 pts a year; best `rand_02` −1.41, worst `rand_24` −12.47; verdict **HINDSIGHT**.

Historical simulation, 2018-01-07 → 2025-10-24. Yahoo occasionally revises its adjusted prices; these numbers reproduced exactly on 2026-10-02.

## How the 2017 lists were chosen

- **Pool:** the 20 largest S&P 500 members on 2017-12-29 in the two sectors the Magnificent 7 belonged to then (Information Technology and Consumer Discretionary), sized with share counts from SEC filings made before that date: AAPL GOOGL MSFT AMZN META V HD INTC ORCL CMCSA CSCO DIS MA IBM MCD NVDA AVGO NKE TXN ACN.
- NVIDIA is in the pool (#16, $117B), which is generous to the strategy. Tesla is not: it joined the S&P 500 in December 2020.
- **Lists:** the Magnificent 7 as published; the 7 largest (AAPL GOOGL MSFT AMZN META V HD); 35 random 7-stock lists from the pool (seed 20260925, draw-spec sha256 `ebe2c118…`, which the rerun checks).
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

Full figure table with sources: [figures.md](figures.md).

## What FinRL-X gets right

- **A sound design.** Every strategy ends in one target weight per stock per date, so modules swap cleanly. We scored FinRL-X's output without touching its code.
- **The code runs and is deterministic.** Re-runs are identical; the zero-cost numbers reproduce within about 2% on dividend-adjusted prices.
- **It is open source.** This audit was possible only because the code is public.
- **The fixes are straightforward:** charge costs in the P&L, apply the stop-losses (they are logged, not applied to returns), ship the Rolling Strategy's missing input files, and choose stock lists by a rule applied at each date.

## Sources in this repo

- [`docs/research/finrl_x_final_report_2026-09-25.md`](../../docs/research/finrl_x_final_report_2026-09-25.md) (verdict)
- [`docs/research/finrl_x_reproduction_2026-09-24.md`](../../docs/research/finrl_x_reproduction_2026-09-24.md) (step 1: reproduction, code reading, cost grid)
- [`docs/research/finrl_x_hindsight_universe_2026-09-25.md`](../../docs/research/finrl_x_hindsight_universe_2026-09-25.md) (step 2: pre-registration and results)
- [`NEGATIVE_RESULTS.md` §4](../../NEGATIVE_RESULTS.md#4-equity-cross-section-and-factors) (the closed row)
- Scripts: [`finrl_x_rerun.py`](../../scripts/research/finrl_x_rerun.py), [`finrl_x_rotation_repro.py`](../../scripts/research/finrl_x_rotation_repro.py), [`finrl_x_hindsight_runs.py`](../../scripts/research/finrl_x_hindsight_runs.py)

Not rerunnable from a fresh clone: [`scripts/research/finrl_x_pit_universe.py`](../../scripts/research/finrl_x_pit_universe.py), which built the 2017 pool, needs S&P 500 membership-history files that are not in the repo and an SEC user agent. The pool it produced is listed above, with every company's size and source in the step 2 doc.

FinRL-X: [github.com/AI4Finance-Foundation/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading) · paper: [arXiv 2603.21330](https://arxiv.org/abs/2603.21330)

Video: KISI Ep03, "FinRL-X Claims 5.98x. Could You Have Picked Those Stocks in 2018?" (working title).
