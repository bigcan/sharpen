# FinRL-X — Final Report

## Does the "next generation of FinRL" build strategies that work?

**Date:** 2026-09-25 · **Branch:** `finrl-x-eval` · **PR:** #18 into `Sept2026` (CI passed, awaiting merge)

**Status:** Closed. No tradeable strategy: the Adaptive Rotation's edge comes from choosing its stocks with hindsight,
and the Rolling Strategy cannot be reproduced.

**Subject:** AI4Finance Foundation's `FinRL-Trading` repository, rebranded **FinRL-X** (paper: arXiv 2603.21330,
submitted 2026-03-22; Apache-2.0 license; 3,747 GitHub stars as of 2026-09-24), evaluated at commit `4409abe9`
(2026-09-18).

**Sources:**
- reproduction: `docs/research/finrl_x_reproduction_2026-09-24.md`;
- hindsight test, pre-registration and results: `docs/research/finrl_x_hindsight_universe_2026-09-25.md`;
- artifacts: `results/finrl_x/` (every run's code version, prices, weights and logs).

## Summary

- **The question.** FinRL-X presents itself as the successor to the FinRL framework. Its README shows two strategies
  beating QQQ from 2018 to 2025 and a paper-trading account up 19.8% in five months. Do its strategies work, and would
  they pass Sharpen (github.com/bigcan/sharpen), this project's falsification-first evaluation framework?
- **The answer: no tradeable strategy.** Nothing reached the Sharpen funnel; both strategies failed the checks before it.
  - **Adaptive Rotation** (claimed 4.80x vs QQQ's 4.02x). FinRL-X's own code reproduces it (4.91x), but only because
    the backtest charges **no trading costs**, although the paper says it charges 10 bps per side. The strategy trades
    about 62 times its capital a year. At the paper's own cost it ends below QQQ; it stops beating QQQ above about
    5 bps per side.
  - **Its edge is hindsight.** The "Growth Tech" list is 2026's Magnificent 7, hard-coded in February 2026 and
    backtested from 2018. Given lists an investor could have chosen at the start of 2018, **0 of 35 beat QQQ**, and the
    7 largest stocks of the day trail QQQ by 7.2 points a year.
  - **Rolling Strategy** (claimed 5.98x). It cannot be reproduced from the public repository.
  - **Paper trading** (+19.8%). A live account record that cannot be re-run. FinRL-X's own chart credits one
    precious-metals rally.
- **What worked.**
  - FinRL-X's code runs, and its zero-cost numbers reproduce within about 2%.
  - No look-ahead leak was found in the Rotation's signal code.
  - Our own checks: an independent re-calculation that matches FinRL-X's printed tables to the last digit, a test plan
    frozen before any run, and a 2017 stock list built only from data public at the time.
- **Cost:** two calendar days (2026-09-24 → 09-25), no paid data (SEC EDGAR and Yahoo Finance are free), and 41
  recorded backtests (plus validation re-runs) on a local 16-core machine.

## 1. The setup

**What FinRL-X is.** AI4Finance's rebuild of FinRL as a full trading platform: data, strategies, backtesting and live
execution through Alpaca.
- Every strategy ends in one output, a **target portfolio weight** for each stock on each date. Stock selection,
  allocation, timing and a risk overlay each transform those weights, so any module can be swapped. The same weights
  drive the backtest and the live broker.
- The paper and README present three use cases:
  1. **Allocation methods:** equal weight, mean-variance, minimum variance, reinforcement-learning (DRL) allocators and
     a trend-timing overlay.
  2. **Rolling Stock Selection + DRL:** each quarter, pick the top 25% of NASDAQ-100 stocks by a machine-learning
     score on fundamentals, then size them with a DRL agent.
  3. **Adaptive Multi-Asset Rotation:** three groups (Growth Tech, Real Assets, Defensive). Each week it picks at most
     two groups by performance against QQQ and at most two stocks per group by momentum, with a market-regime filter
     and stop-losses.

**The claims under test** (README and paper: 2018-01-07 → 2025-10-24, "10 bps per side" costs):

| | Growth | Per year | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|---|
| Rolling Strategy | 5.98x | 25.85% | 27.85% | 0.93 | −38.95% |
| Adaptive Rotation | 4.80x | 22.32% | 20.30% | 1.10 | −21.46% |
| QQQ | 4.02x | 19.56% | 24.20% | 0.81 | −35.12% |
| SPY | 2.80x | 14.14% | 19.61% | 0.72 | −33.72% |

Paper trading, 2025-10-26 → 2026-03-12 (both strategies combined): +19.76%, Sharpe 1.96, max drawdown −12.22%.

**How it was tested.**
- **Kept apart.** FinRL-X was cloned at a pinned commit and run in its own Python environment, never modified. This
  project's side lives on its own git branch.
- **Reproduce first, then test.** Step 1 runs FinRL-X's code as published and asks whether it gives the published
  numbers. Step 2 stress-tests the one strategy that reproduced, with the test plan fixed in advance.
- **Trust the arithmetic.** A separate re-implementation of FinRL-X's performance maths had to match the table
  FinRL-X itself prints, to the last printed digit, before any cost or new metric was layered on top.

## 2. Step 1 — Can the published numbers be reproduced?

**Installing it.** The documented `pip install -r requirements.txt` fails on a clean machine. It asks for a package
named `finnhub`, but the real one is `finnhub-python`, and nothing imports it anyway. The packages the DRL code needs
(`finrl`, `stable_baselines3`, `gym`, `pypfopt`) are not in the requirements at all.

**Adaptive Rotation, 2018-01-07 → 2025-10-24,** run through FinRL-X's own `deploy.sh`:

| | Growth | Per year | Volatility | Sharpe (theirs) | Max drawdown |
|---|---|---|---|---|---|
| Claimed | 4.80x | 22.32% | 20.30% | 1.10 | −21.46% |
| Reproduced, no costs | 4.91x | 22.69% | 22.06% | 1.03 | −23.55% |
| Reproduced, 10 bps per side | 3.02x | 15.25% | 22.00% | 0.69 | −24.75% |
| QQQ, same data | 3.95x | 19.30% | 22.85% | 0.84 | −35.06% |

- The zero-cost return reproduces within about 2%. Risk comes out somewhat worse than claimed.
- **Dividends matter, and the README cannot get them.** FinRL-X's download script fetches prices without dividends,
  while its published benchmark figures include them. On price-only data QQQ comes to 3.75x, not the claimed 4.02x.
  The table above uses dividend-adjusted prices, the basis the published figures imply. A reader who follows the
  README exactly gets different numbers.

**What the backtest code actually does** (file and line references are to FinRL-X at `4409abe9`):
1. **It charges no trading costs.** The equity curve compounds the weights' returns with no cost term
   (`run_adaptive_rotation_strategy.py`, lines 443–456). The paper states 10 bps per side.
2. **The stop-losses are logged, not applied.** Daily stop-loss and "fast risk-off" exits print messages and update a
   tracker, but never change the weights used for returns (lines 268–311). The backtest holds each Friday's portfolio
   for the whole week, so the advertised daily de-risking is not in the published numbers.
3. **Its Sharpe ratio is non-standard.** It divides the compound annual return by volatility (line 476). Every Sharpe
   in the README matches this formula, e.g. 22.32% / 20.30% = 1.10.
4. **The table and the chart come from different code.** The table's function marks the portfolio weekly
   (lines 474–481), while the published chart shows daily values. That function was added on 2026-03-25, one day after
   the figures were committed. The chart's code is not in the repository.
5. **Trades fill at the close that made the decision.** Weights computed from Friday's close earn returns from that
   same close. That is common in backtests and mildly optimistic.
6. **The stock list was chosen with hindsight.** See Step 2.

**Rolling Strategy — cannot be reproduced.**
- Its DRL allocator reads two files that are not in the repository: `./data_processor/sp500_tickers_daily_price_20250712.csv`
  and `./result/stock_selected.csv`.
- Its packages are missing from the requirements (above).
- Its NASDAQ-100 universe is downloaded as **today's** member list, with no history. Run over 2018–2025, it would pick
  stocks from the 2026 index.
- The example notebook has the rolling-selection call commented out, and its backtest cell ends in an error.

**Paper trading — cannot be re-run.** The +19.76% is a live account record, not code output. FinRL-X's own chart
labels the gain a "Precious Metals & Mining Stocks Rally" (December 2025 to late January 2026), with flat or falling
stretches before and after. A backtest of the Rotation alone over the same window made 1.12x after 10 bps (QQQ 0.95x,
SPY 0.99x).

## 3. How much do trading costs matter?

The Rotation replaces about 2.8 of its roughly 5 holdings every week, and only 4% of weeks leave the portfolio
unchanged. That is **62.5x its capital a year** counting buys and sells (about 31x one-way), so every basis point of
cost per trade costs about 0.6% a year.

| Cost per side | 0 bps | 2 bps | 3 bps | 5 bps | 10 bps | QQQ |
|---|---|---|---|---|---|---|
| Per-year return | 22.69% | 21.17% | 20.41% | 18.91% | 15.25% | 19.30% |
| Sharpe | 1.07 | 1.01 | 0.98 | 0.92 | 0.77 | 0.91 |

*Sharpe here is the standard version (mean return over volatility), so it differs slightly from FinRL-X's.*

- The strategy stops beating QQQ at **4.5 bps per side** on return and **5.3 bps** on Sharpe.
- The paper's 10 bps is its own assumption. This project's assumptions run from about 2 bps for liquid US large caps
  to 10 bps as its standard single-stock default, so the break-even sits between them. At 2 bps the strategy still
  edges QQQ over the backtest.
- So costs make the published edge thin, not absent. The decisive question is the stock list.

## 4. Step 2 — The hindsight test

**The problem.** The Rotation's growth list is AAPL, MSFT, NVDA, META, AMZN, GOOGL and TSLA, 2026's Magnificent 7. It
was committed on 2026-02-05 and backtested from 2018. In January 2018 nobody knew NVIDIA and Tesla would be among the
decade's biggest winners; Tesla was not even in the S&P 500 until December 2020. Whatever the intent, a list chosen
with knowledge of 2018–2025 cannot be used to judge a strategy over 2018–2025.

**The test.** Replace only the growth list with lists an investor could have chosen on 2017-12-29, and change nothing
else.
- **The pool:** the 20 largest S&P 500 members on 2017-12-29 in the two sectors the Magnificent 7 belonged to at the
  time (Information Technology and Consumer Discretionary). Market caps use share counts from SEC filings made before
  that date, times that day's closing price.
  - Result: AAPL GOOGL MSFT AMZN META V HD INTC ORCL CMCSA CSCO DIS MA IBM MCD NVDA AVGO NKE TXN ACN.
  - NVIDIA is in the pool (#16, $117B), which is generous to the strategy. Tesla is not: it was not an S&P 500 member.
- **Three arms, 37 backtests:**
  1. the Magnificent 7 as published;
  2. the 7 largest in the pool: AAPL GOOGL MSFT AMZN META V HD;
  3. 35 random 7-stock lists drawn from the pool (fixed random seed).
- **Rules fixed in advance.** The plan, the pool, the random lists and the pass/fail rules were written down and
  fingerprinted (SHA-256) on 2026-09-25 at 00:45 UTC, before the first backtest. The main measure is return above QQQ
  at 2 bps per side:
  - **HINDSIGHT** if the Magnificent 7 beats 90% of the random lists and the median random list does not beat QQQ;
  - **ROBUST** if the median random list and the 7-largest list both beat QQQ;
  - **INCONCLUSIVE** otherwise, or if the Sharpe comparison disagrees.
- **Execution.** All 37 runs used FinRL-X's own backtest code, 12 at a time (about 40 minutes). An automatic check
  confirmed that nothing but the growth list (and the folder each run writes to) changed.

**Building a 2017 list honestly is harder than it looks.** Each of these would have moved a top-20 name:
- Alphabet has two listed share classes and was first counted twice.
- Comcast's latest total share count on file was from 2010, before a stock split. It first came out at $83B instead of
  about $191B, which dropped it from the list until a staleness rule fixed it.
- Visa has three share classes plus convertible preferred stock. Only its own "as-converted" total (2,350 million
  shares, from its 2017 annual report) gives the right size, $268B.
- Disney and Broadcom re-registered with the SEC after 2017, so their 2017 filings sit under old IDs.
- The SEC's bulk data service silently left out Disney's latest share count.
- Sector labels have changed. Google and Facebook were "Information Technology" in 2017, and Visa and Mastercard moved
  to Financials in 2023.
- The final numbers closely match published end-2017 market caps (Apple $869B, Microsoft $660B, NVIDIA $117B).

**Results** (2018-01-07 → 2025-10-24, dividend-adjusted prices):

| Growth list | Per year, no costs | Per year, 2 bps | vs QQQ, 2 bps | Sharpe, 2 bps (vs QQQ) | vs QQQ, 10 bps |
|---|---|---|---|---|---|
| Magnificent 7 (published) | 22.69% | 21.17% | **+1.87 pts** | 1.01 (+0.10) | −4.05 pts |
| 7 largest on 2017-12-29 | 13.42% | 12.08% | −7.22 pts | 0.79 (−0.12) | −12.43 pts |
| 35 random lists: median | 11.24% | — | **−9.18 pts** | (−0.23) | −13.71 pts |
| 35 random lists: best | 19.31% | 17.89% | −1.41 pts | 0.91 (+0.00) | −6.93 pts |
| 35 random lists: worst | 8.01% | 6.83% | −12.47 pts | 0.51 (−0.40) | −17.07 pts |
| QQQ / SPY (buy and hold) | 19.30% / 13.90% | | | 0.91 / 0.79 | |

**Verdict: HINDSIGHT — the Adaptive Rotation is a no-go.** The return rule, the Sharpe rule and the paper's own
10 bps all agree.
- **0 of 35** random lists beat QQQ at 2 bps, and the Magnificent 7 beats all 35.
- Even with no costs, the median random list makes 11.2% a year against QQQ's 19.3%, and it trails SPY (13.9%) too.
  The 7 largest roughly match SPY.
- The best random list (ACN AMZN AVGO GOOGL MCD NVDA ORCL) holds two of the decade's biggest winners, NVIDIA and
  Broadcom, and only ties QQQ at zero cost (19.31% vs 19.30%).
- **Why.** The strategy rotates weekly among a fixed list and holds at most two stocks per group, so the list decides
  the result. QQQ is weighted by company size, so it automatically owns more of new winners like NVIDIA and Tesla as
  they grow. A list fixed in 2017 cannot.
- **The months after the freeze do not change this.** From 2026-02-06, the day after the list was committed, to
  2026-09-18, the Magnificent 7 version made 1.21x after 10 bps against QQQ's 1.19x. That is 29 weeks of the same
  hindsight list run forward, and it cannot separate the rotation logic from the list.
- **Checks.** All 37 runs match FinRL-X's own printed table; QQQ is identical across runs (one shared price snapshot);
  the Magnificent 7 re-run reproduces Step 1 exactly (4.9100x).

## 5. Why this is not a tradeable strategy

1. **The edge is the list, not the logic.** With any list knowable in advance, the Rotation loses to buying QQQ, by
   about 8 points a year at the median even before costs.
2. **Costs.** Even with the hindsight list, the edge over QQQ survives only below about 5 bps per side, and that still
   assumes filling at the decision close.
3. **The risk layer is untested.** The stop-losses are not in the backtested returns, so nobody knows what they do.
4. **The live record is one rally.** Five months, with the gain concentrated in one precious-metals run.
5. **The Rolling Strategy cannot be checked at all.**

## 6. What FinRL-X gets right

- **The architecture is sound.** The weight-centric contract is a clean seam; this evaluation scored FinRL-X's output
  without touching its code.
- **The code runs and is deterministic.** Re-runs are identical, and the zero-cost numbers reproduce within about 2% on
  total-return prices.
- **No look-ahead leak** was found in the Rotation's signal code on a first pass: it reads only data up to each
  decision date, and weekly bars are built only from past days. This is not a full audit.
- **It is open source.** Everything in this report could be checked because the code is public.
- **The fixes are straightforward:** charge costs in the P&L, apply the stops, ship the missing Rolling Strategy
  inputs, and choose stock lists by a rule applied at each date rather than by hand.

## 7. Assessment by the AI researcher

*This section is Claude's own assessment. Claude (Opus 5.5) designed, ran and audited this evaluation.*

**My prior was low before any run.** Raw Sharpe against QQQ is how market exposure usually looks. In this project's
record, long-only large-cap strategies have repeatedly turned out to be beta rather than skill.

**What surprised me was the size of the gap.** I expected the hindsight list to explain part of the edge. It explains
all of it and more: fed lists chosen in advance, the rotation logic itself loses about 8 points a year to QQQ before
costs. That is a result about the strategy's design, not just its backtest.

**Where I went wrong.**
- **A conclusion stated too strongly.** I first reported that costs "erase the edge over QQQ". That is true at the
  paper's 10 bps, but the formula audit's break-even check showed the edge survives below about 5 bps. I corrected the
  report before the hindsight test ran.
- **A unit left unstated.** I quoted "62x turnover" without saying it counts buys and sells together.
- **Harness slips, each caught before results were read:**
  - I edited a running script, which broke one run's archiving;
  - one "dividend-adjusted" run silently used price-only data, which identical numbers gave away;
  - my first 2017 sector mapping would have excluded Alphabet and Meta;
  - the first 2017 ranking had Comcast 57% too small.

**How confident I am.**
- **High, for the Rotation as designed.** The test was pre-registered, the margins are wide (0 of 35), the code is
  FinRL-X's own, and every run matches its printed tables.
- **Limits.** There is one window, 2018–2025, which is also the strategy's design period. The real-assets list (oil
  majors, miners, gold and silver) was left as published and may hide a second hindsight effect. That could only
  weaken the strategy further.
- **Scope.** None of this says FinRL-X is useless as infrastructure. It says its published performance is not evidence
  of an edge.

## 8. Lessons worth keeping

1. **Run the code before believing the chart.** Here the published chart came from code that is not in the repository.
2. **Check that the cost line exists,** not just that the paper mentions costs.
3. **A risk layer that is only logged is not a risk layer.** Check that it reaches the P&L.
4. **Test hand-picked stock lists against random ones.** Draw many lists from a pool known at the start. If the
   published list beats them all and the median loses, the edge is the list. It is the counterpart of shuffling trade
   timing to test a timing rule.
5. **Compare with the right benchmark after costs,** and report the break-even cost.
6. **Match their numbers before extending them.** A re-calculation that reproduces the printed table to the last digit
   is what makes every later number trustworthy.
7. **Point-in-time data is harder than it looks:** share classes, stale filings, changed company IDs and renamed
   sectors each moved a name in or out of the 2017 list.
8. **Fix the rules before running.** A plan fingerprinted before the first backtest means the verdict could not be
   tuned to the results.
9. **Check the price basis.** Dividends move QQQ from 3.75x to 3.95x over this window, and long-dated Treasuries (TLT)
   from 0.73x to 0.91x.

## 9. Status, costs and next steps

- **Status.** Closed; no FinRL-X strategy goes to the funnel. PR #18 (both steps) is open into `Sept2026` with CI
  passing, awaiting the operator's merge.
- **Spent.** Two calendar days (2026-09-24 → 09-25). No paid data: SEC EDGAR and Yahoo Finance. Local compute: 41
  recorded FinRL-X backtests (4 in Step 1, 37 in Step 2), plus validation re-runs.
- **What would reopen it:** evidence of an edge under a stock-list rule applied without hindsight, or the missing
  Rolling Strategy inputs from the authors.
- **Better uses of the next effort:** this project's validated path, cross-asset time-series momentum (TSMOM /
  TAILWIND).

## Appendix A — Glossary

- **Basis point (bp):** 0.01%. "10 bps per side" means each buy and each sell costs 0.1% of the amount traded.
- **Growth (x):** final value divided by starting value; 4.80x means $1 became $4.80.
- **Per-year return (CAGR):** the constant yearly growth rate that produces the same final value.
- **Sharpe ratio:** average return divided by volatility, per year. The standard version uses the average return;
  FinRL-X uses the compound annual return instead, which gives somewhat different numbers.
- **Max drawdown:** the largest fall from a peak to a later low.
- **Turnover:** how much of the portfolio is traded. "62.5x a year, counting buys and sells" is about 31x one-way.
- **Break-even cost:** the cost per trade at which the strategy stops beating its benchmark.
- **QQQ / SPY:** exchange-traded funds tracking the NASDAQ-100 and the S&P 500.
- **Total return vs price-only:** total return includes dividends; price-only ignores them.
- **Magnificent 7:** Apple, Microsoft, NVIDIA, Meta, Amazon, Alphabet and Tesla, the largest US tech-led stocks of the
  mid-2020s.
- **Point-in-time:** using only what was known on each date, e.g. index membership and share counts as they were then.
- **Hindsight (look-ahead) bias:** letting knowledge from after a date shape a decision made on that date.
- **Survivorship bias:** leaving out companies that later disappeared, which flatters results.
- **Pre-registration:** fixing the hypotheses, data, rules and pass bars in writing before seeing results.
- **Parity check:** confirming an independent calculation reproduces the original's printed numbers.
- **Random-universe test:** comparing a hand-picked stock list with many random lists drawn from a pool known at the
  start.
- **GICS:** the Global Industry Classification Standard, the sector labels used by index providers.
- **SEC EDGAR / XBRL:** the US regulator's public filings database, and the machine-readable format of the financial
  data in those filings.
- **Share classes / as-converted:** some companies have several kinds of shares; the "as-converted" count restates them
  all as one class.
- **Planted-bug (mutation) test:** deliberately breaking the code to confirm a test catches it.

## Appendix B — Commits (branch `finrl-x-eval`)

| Commit | Date | Step |
|---|---|---|
| `a7d5b5a3` | 2026-09-25 | Step 1: reproduction scripts, re-calculation with parity check, cost grid, tests, report |
| `6e8f411c` | 2026-09-25 | Step 2: point-in-time 2017 universe, pre-registered hindsight test, results |
| PR #18 | 2026-09-25 | Both steps into `Sept2026` (CI passed; awaiting merge) |

Tests: 20 automated checks. 17 deliberately planted bugs were each caught by the test predicted to catch it.
