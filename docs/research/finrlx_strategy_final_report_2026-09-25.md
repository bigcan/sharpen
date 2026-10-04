# FinRL-X Strategy Mission — Final Report

## Can a strategy built on FinRL-X beat the S&P 500 on data it has never seen?

**Date:** 2026-09-25 · **Branch:** `finrl-x-strategy` (cut from `finrl-x-eval`; not pushed)

**Status:** Borderline NO-GO. The strategy beat SPY on risk-adjusted return and halved its drawdowns in both test
windows, but on the untouched window its alpha missed the pre-registered significance bar (t 1.996 against 2.0; deflated
alpha 0.63 against 0.95). Nothing is tradeable. A forward lockbox is open.

**Subject:** `keel-v1`, a FinRL-X weight strategy: half SPY, half a cross-asset trend sleeve on 18 ETFs, built from this
project's one validated edge (time-series momentum), in an account a retail investor can actually hold (no margin loan).

**Sources:**
- pre-registration (frozen before the look): `docs/research/finrlx_strategy_prereg_2026-09-25.md`;
- forward lockbox registration: `docs/research/finrlx_strategy_forward_lockbox_2026-09-25.md`;
- result files: `docs/research/finrlx_strategy_artifacts/` (the one look, design grid, fidelity, parity, trial ledger);
- code: `scripts/research/finrlx_strategy/`, tests `tests/research/test_finrlx_strategy_*.py`;
- gates: `configs/finrlx_strategy.gates.yaml`, strategy and costs: `configs/finrlx_strategy.yaml`.

## Summary

- **The question.** The FinRL-X evaluation (2026-09-25) found its published strategies dead: the Rotation's edge was a
  hindsight stock list. The mission: build a new strategy on FinRL-X as the platform that passes Sharpen's tests and beats
  SPY after costs on data it has never seen. A clean NO-GO counts as a result.
- **The answer: a borderline NO-GO.**
  - On 1973–2005, a window this project had never touched, the book earned **Sharpe 0.44 against SPY's 0.33**, about the
    same return (10.3% vs 10.8% a year) at half the volatility, with a **maximum drawdown of 21% against 48%**. It lost
    10% in the 1973–74 bear market (SPY −38%) and 18% in the 2000–02 bust (SPY −48%).
  - Its **alpha was +1.15% a year with t = 1.996**. The pre-registered bar was 2.0. Deflated for the 84 variants tried,
    the alpha's probability of being real came out at 0.63 against a 0.95 bar. Six of the eight gates passed.
  - The same book on 2008–2026 (the design window) looked alike: Sharpe 0.67 vs 0.62, alpha t 1.28.
- **What was learned that matters beyond this strategy.**
  - The project's "trend-following Sharpe 0.60" is a research number. Traded one day after the decision and financed at
    the T-bill rate, it averages **0.40**. The published month-end rebalance day happened to be a lucky one.
  - Choosing the "best" of 24 variants on the 18-year design window was **worse than random**: the design ranking was
    anti-correlated with the clean-window ranking (−0.50). All 24 variants beat SPY's Sharpe on the clean window.
  - The long-only variant, which FinRL-X can trade as it stands, did best on the clean window. That was noticed after the
    look, so it proves nothing; it is now a forward-only hypothesis.
- **What worked.** FinRL-X's own backtester reproduces our engine to the last digit. A seal kept 1973–2005 unreadable
  until the rules were committed. Free 1970s data rebuilt the 18 ETFs closely enough that the strategy's returns on
  proxies match its returns on the real ETFs at 0.997 correlation (2008–2024).
- **Cost:** one calendar day, no paid data, local CPU only. The GPUs stayed idle on purpose (see §1).

## 1. The setup

**Starting point (read, not re-derived).** The only validated edge on record was cross-asset time-series momentum (TSMOM,
"trend"): net Sharpe about 0.6, nearly uncorrelated with SPY. Long-only large-cap stock picking had repeatedly turned out
to be beta; liquid large-cap cross-sectional equity was closed.

**Reinforcement learning: checked and set aside.** A planted-signal test (2026-09-23/24) had shown the project's SAC agent
cannot learn a weak edge. Its follow-up, with standardized inputs, came in this morning: **0 of 5 agents learned an
information coefficient of 0.05 with the right sign.** The pre-registered reading was "structural". So no strategy here
bets on RL, and the three idle RTX 4090s were not needed for daily ETF books.

**FinRL-X as the platform, honestly.** FinRL-X's contract is a table of target weights per date, and that is what this
strategy produces. Its machinery is narrower than its paper suggests:
- its backtester forces every weight row to sum to 1 (long-only, fully invested, no leverage), trades at the close that
  made the decision, and pays nothing on cash;
- its Alpaca executor clamps short weights to zero and renormalises leverage away.

So the strategy's long-only special case runs through FinRL-X's own backtester, and our engine adds what a real account
has: next-day execution, shorts with borrow fees, no interest on short proceeds, cash at the T-bill rate. **On the same
weights over 4,527 days, FinRL-X's engine and ours agree exactly** (0.0 difference with no costs, 0.0007% at 2 bps).

**Power, computed before building.** The alpha test's t-stat grows as information ratio × √years. At a realistic
information ratio of 0.35–0.6, one year of new data gives t ≈ 0.5; 30 years give 2–3. Only a multi-decade window this
project had never used could decide the question. Every day from 2006 to 2026 had already been used. The project had
never run its trend book before 2006, so the plan became: design on 2008–2026, then test once on 1973–2005 rebuilt from
free long-history data.

## 2. Step 1 — The design window (2008–2026, ETF data)

**What the project's trend edge is worth when you can actually trade it.**

| Trend book, 2007–2026 | Sharpe |
|---|---|
| As researched: trades at the decision close, total return | 0.55 |
| Financed at the T-bill rate (excess return) | 0.47 |
| Also traded one day after the decision | 0.35 |
| Averaged over all 21 possible rebalance days | **0.40** (range 0.32–0.49) |

The month-end rebalance date sat near the top of that range by luck. The strategy therefore splits its capital into four
tranches rebalanced a week apart, which earns the average instead of a lucky or unlucky day.

**The book.** Half SPY, held. Half a trend sleeve over 18 ETFs (US, international and emerging equity; Treasuries and
corporates; gold, silver, oil, broad commodities, agriculture; five currencies), using the project's frozen trend rule.
The sleeve is scaled to a small risk target, then cut if needed so the account never borrows: longs at most 100% of the
account, shorts at most 50%.

**24 design variants** (3 ways to size the sleeve × 2 risk levels × static or volatility-managed SPY × with or without
shorts), all logged. The pre-declared rule picked the one with the highest alpha t-stat: covariance sizing, 3% sleeve
risk, static SPY, shorts allowed.

| Design window | Selected book | SPY |
|---|---|---|
| Sharpe (excess of T-bills) | 0.673 | 0.618 |
| Alpha vs SPY | +1.1%/yr, t 1.28 | |
| Max drawdown | −23% | −51% |
| Break-even cost | 5× the assumed costs | |

On its own design window the book would already fail the alpha test. The one chance was that trend was stronger before
2009, as published research reports.

**Sharpen's signal funnel.** The trend signal passes Tier 0, the truncation test that recomputes it on truncated data. The
funnel's other tiers rank the 18 ETFs against each other each day; there the trend signal scores about zero (IC-IR
−0.03, against +0.02 for a noise control). That is expected: the funnel subtracts each day's average across assets, and
time-series trend makes its money from exactly that average (being net long or net short). So the funnel's ranking tiers
do not apply to this mechanism, and Tier 0 was the only one pre-registered as a gate.

## 3. Step 2 — Rebuilding 1973–2005 from free data

None of the 18 ETFs existed in 1973. Four parallel builders reconstructed each one from free long-history sources, each
proxy chosen by a rule fixed before its fit was measured, and fit measured on 2006–2026 only.

| Asset | Proxy (source) | Daily from |
|---|---|---|
| SPY | S&P 500 price + dividends (Yahoo, Shiller) | 1928 |
| QQQ, IWM | Nasdaq Composite (Yahoo); small-cap portfolio (Ken French) | 1971; 1926 |
| EFA, EEM | Ken French developed ex-US; Vanguard EM index fund | 1990; 1994 |
| TLT, IEF, LQD | Bond returns rebuilt from Federal Reserve yields | 1962; 1962; 1983 |
| GLD, SLV | London gold and silver fixes | 1968 |
| USO, DBC | Rolled oil and energy futures (EIA) | 1985 |
| FXE FXY FXB FXA, UUP | Fed exchange rates + foreign interest rates; rebuilt dollar index | 1971 |
| DBA | none: no free daily agriculture data | never |

Twelve assets are live on 1 January 1973; the rest join as their data starts.

**A seal.** Strategy code could not read any date before 2006 until the pre-registration was committed and its
fingerprint matched. Until then the builders saw only gap and outlier counts for that period.

**The check that matters.** The strategy run on the proxies and on the real ETFs over 2008–2024 produced returns
correlated **0.997 month to month** (the trend sleeve alone: 0.98), with the proxy version slightly worse (Sharpe 0.64
vs 0.66). The test would measure the tradeable strategy, not the quirks of the proxies.

## 4. Step 3 — The one look (1973–2005)

The rules, windows, gates, trial count (84) and a power estimate ("roughly 0.2–0.35 chance of passing if the design-window
edge is the true edge") were committed first (`674df095`). The evaluation ran once, 4 minutes 45 seconds, recorded with a
fingerprint of every file it depended on.

**Results vs SPY** (1973-01-02 → 2005-12-30, after 2 bps per trade on liquid ETFs and 5 bps on thin ones, borrow fees,
ETF expense ratios, next-day execution):

| | Book | SPY |
|---|---|---|
| Sharpe (excess of T-bills) | **0.442** | 0.325 |
| Return per year | 10.3% | 10.8% |
| Volatility | 8.8% | 16.1% |
| Max drawdown | **−21.0%** | −47.5% |
| Sortino | 0.61 | 0.46 |
| Alpha vs SPY | **+1.15%/yr, t 1.996** | |
| Beta to SPY | 0.54 | 1 |
| Average holdings | 93% long, 14% short | 100% long |
| Turnover / trading cost / borrow | 3.8× a year / 0.13% / 0.11% | |
| Break-even cost | 9× the assumed costs (≈18 bps liquid) | |

**Every pre-registered test:**

| Test | Result | Bar | |
|---|---|---|---|
| Sharpe above SPY's | 0.442 vs 0.325 | higher | pass |
| Alpha t-stat (monthly, Newey–West) | 1.996 | ≥ 2.0 | **fail** |
| Deflated alpha (84 trials) | 0.63 | ≥ 0.95 | **fail** |
| Deflated Sharpe (84 trials) | 0.976 | ≥ 0.95 | pass |
| Probabilistic Sharpe | 0.994 | ≥ 0.95 | pass |
| Backtest-overfitting probability (24 variants) | 0.004 | ≤ 0.5 | pass |
| Cost gap (no-cost minus net Sharpe) | 0.015 | ≤ 0.15 | pass |
| Alpha positive by quarter of the window | 4 of 4 | ≥ 3 | pass |
| Signal truncation test (Tier 0) | pass | pass | pass |
| H2: covariance sizing beats simple sizing | +0.01 Sharpe | +0.10 | not tested (H1 failed); would fail |
| H3: volatility-managed SPY beats held SPY | −0.01 Sharpe | +0.10 | not tested; would fail |

**Verdict: H1 FAILS. No second look.**

**How it behaved** (reported, not gating):

| Episode | Book | SPY |
|---|---|---|
| 1973–74 bear market | −10% | −38% |
| 1987 crash (Aug–Dec) | −12% | −22% |
| 1990 (Jul–Oct) | −5% | −14% |
| 2000–02 bust | −18% | −48% |
| 1982–2000 bull market | +1,038% | +2,400% |

It beat SPY in only 12 of 33 calendar years: this is an insurance-like book, strongest in crashes, lagging in booms.
The trend signal itself worked better here than on the design window: it called next month's direction right 56% of the
time (time-series rank correlation 0.075, against 0.032 on 2008–2026).

**Things noticed after the look (they cannot change the verdict):**
- **The deflated-alpha result depends on which trials count.** With all 24 variants (as pre-registered) it is 0.63–0.82
  depending on the trial count; with only the 12 long/short variants the book was chosen from, it is 0.92–0.95. The
  24 variants include structurally different long-only books whose spread raises the bar.
- **All 24 variants beat SPY's Sharpe** on 1973–2005, and 10 of 24 had alpha t ≥ 2, mostly the long-only ones
  (median alpha t 2.56 long-only vs 1.46 long/short). In that era short positions cost about 0.9% a year extra, because
  the proceeds of a short sale earn a retail investor nothing while T-bills paid 6.3% on average.
- **Leverage would probably not rescue it at a retail broker.** Levered to SPY's volatility the book would have earned
  about 1.9% a year more than SPY *before* financing. Retail margin loans typically cost several points over T-bills
  (Alpaca's current rate was not checked here); at about 2.3 points on the borrowed 83%, the gap is gone. With
  futures-style financing near the T-bill rate it would survive; FinRL-X has no futures executor.

## 5. Why this is not a tradeable strategy

1. **It failed its pre-registered test.** Narrowly, but the bar was set before the look, and moving it afterwards would
   make the whole exercise worthless.
2. **The edge is small.** An alpha of about 1% a year with an information ratio of about 0.35 needs decades to prove, and
   the only multi-decade clean window is now spent for this family.
3. **It lags in bull markets.** A book that trails SPY for 18 years (1982–2000) is hard to hold whatever its Sharpe.
4. **Execution needs work either way.** The long/short version needs shorting, which FinRL-X's Alpaca executor removes;
   a levered version needs financing close to the T-bill rate, which retail margin loans typically don't offer.

## 6. What FinRL-X contributed

- **The weight contract is a good seam.** The strategy is a table of weights, scored by FinRL-X's own engine and ours
  with identical results.
- **Its limits are real.** Long-only, fully invested, no execution lag, no costs on the rotation path, shorts clamped in
  live execution. A strategy that needs shorts or leverage has to be executed outside FinRL-X's Alpaca code today.
- **The long-only variant is FinRL-X-native.** If the forward lockbox favours it, it could run on FinRL-X as is.

## 7. Assessment by the AI researcher

*This section is Claude's own assessment. Claude (Opus 5.5) designed, ran and audited this mission.*

**My prior was that this was the best honest shot and still a long one.** Every other route to beating SPY on the record
was closed; trend had one validated, uncorrelated premium. I expected a small edge that would be hard to prove, and
wrote down a 0.2–0.35 pass probability before the look. The result sits right at that line.

**What surprised me.**
- How much of the project's headline trend Sharpe was execution convention: 0.60 on paper, 0.40 tradeable.
- That choosing among variants on 18 years of data was worse than random: the ranking reversed between windows.
- That the clean window favoured the long-only book FinRL-X can already trade.

**Where I went wrong.**
- **I selected the best of 24 variants on the design window.** With an information ratio around 0.3, 18 years cannot
  tell such variants apart; the selection mostly picked noise, and the clean window ranked my choice 11th. Pre-registering
  the simplest variant would have been better, and I should have seen it from my own power calculation.
- **I let the deflation trial set include structurally different books.** Counting all 24 variants was defensible and I
  fixed it in advance, but I did not think through how the long-only books' spread would raise the bar for a long/short
  book. That choice, more than the trial count, decided the deflated-alpha leg.
- **Harness slips, each caught before the look:** the package ran without the branch's code on the path until I fixed
  PYTHONPATH; an error in our own code that the truncation test caught; a funding step 20× too slow; a crash when a data series
  ended (now: an asset with no price keeps its position); expense ratios that would have been charged twice (caught by
  the commodity data builder).

**How confident I am.**
- **High that the verdict is right as a verdict:** the rules were fixed first, the look was taken once, and the numbers
  are consistent across two disjoint windows.
- **Moderate that the edge exists:** the point estimates are positive everywhere I looked (24/24 variants above SPY's
  Sharpe; alpha positive in every quarter of 1973–2005), but that is not what was pre-registered, and trend's existence
  in that era was already known from published research.
- **Scope:** this closes "beat SPY with free-data trend in a no-margin retail account" for now. It does not close trend
  as a diversifier, or trend with cheap leverage.

## 8. Lessons worth keeping

1. **Price the execution before believing the edge.** Next-day trading, cash financing and rebalance-day luck took a
   Sharpe of 0.60 to 0.40.
2. **Don't select among variants you can't tell apart.** Compute whether the design window can rank them; if not,
   pre-register the simplest.
3. **Decide what counts as a trial before counting.** The trial set's spread drives the deflated Sharpe as much as N.
4. **Old data can still be clean.** Rebuilding 1973–2005 from free sources gave 33 untouched years; check that proxies
   reproduce the strategy, not just each asset (0.997 here).
5. **Seal the clean window in code,** not by promise: unreadable until the rules are committed.
6. **Run the dry run.** A full rehearsal on the design window caught a crash that would have otherwise happened during
   the one look.
7. **A risk layer must reach the P&L.** Every cost, borrow fee and funding limit here is inside the returns.
8. **Report money, not just significance.** A book with SPY's return at half its drawdown is useful even when its alpha
   is not significant; it is just not proof of skill.

## 9. Status, costs and next steps

- **Status.** H1 NO-GO (borderline). Nothing tradeable, so no Tier-2 audit was run (it is required only before calling
  something tradeable) and no paper or live trading. Forward lockbox open for `keel-v1` and `keel-lo`
  (`python -m research.finrlx_strategy.lockbox`); first house check after 63 trading days (around late December 2026).
- **Spent.** One calendar day. No paid data (Ken French, Federal Reserve, Shiller, LBMA, EIA, OECD, BIS, Yahoo). Local
  CPU. 137 engine, bridge and data tests, including planted-bug checks; 84 logged variants plus 24 on the clean window.
- **Options that need a decision:**
  1. **Survivorship-free small-cap data** (paid; the record's stated condition for reopening stock selection, where being
     small is an advantage). Needs a budget approval.
  2. **Cheap leverage** through a futures account rather than Alpaca. The clean-window arithmetic says levered trend+SPY
     could beat SPY's return, but only on new data, so this would start as a forward test.
  3. **Let the lockbox run.** Free, slow, and able to falsify within a year rather than certify.

## Appendix A — Glossary

- **Alpha / beta:** beta is how much the book moves with SPY; alpha is its average return beyond that, per year.
- **Alpha t-stat:** alpha divided by its uncertainty. About 2 means roughly a 1-in-40 chance of seeing it with no edge.
- **Newey–West:** a way to compute that uncertainty that allows for returns being correlated over time.
- **Sharpe ratio (excess):** average return above the T-bill rate divided by volatility, per year.
- **Deflated Sharpe (DSR):** the probability that a Sharpe ratio is real after allowing for how many variants were tried.
- **Probabilistic Sharpe (PSR):** the probability that the true Sharpe is above zero, given the sample's shape and length.
- **PBO (probability of backtest overfitting):** how often the best variant in one half of the data ranks below the median
  in the other half.
- **Information ratio (IR):** alpha divided by the volatility of the book's returns not explained by SPY.
- **Time-series momentum (trend):** holding an asset long if its own past 3–12-month return is positive, short if negative.
- **Tranches:** splitting the book into four parts rebalanced on different days to avoid rebalance-day luck.
- **No margin loan / fully funded:** the account never borrows cash; shorts are backed by their own sale proceeds.
- **Borrow fee / short rebate:** the cost of borrowing shares to sell short; the interest (none, for retail) on the proceeds.
- **Break-even cost:** the trading cost at which the book's Sharpe falls to SPY's.
- **Proxy:** a series standing in for an ETF before it existed (e.g. Federal Reserve yields for a Treasury ETF).
- **Seal:** code that keeps the clean window unreadable until the pre-registration is committed.
- **Pre-registration:** fixing hypotheses, data, rules and pass bars in writing before seeing results.
- **Lockbox (forward incubation):** tracking a frozen strategy only on days after it was registered.
- **Parity check:** confirming two independent calculations give the same numbers.
- **Planted-bug test:** breaking the code on purpose to confirm a test catches it.

## Appendix B — Commits (branch `finrl-x-strategy`)

| Commit | Step |
|---|---|
| `13339096` | Seal, fully funded book engine, P&L, metrics, trial ledger, design grid defined before running |
| `eea4504f` | FinRL-X parity bridge, signal funnel, one-look runner, equity and rates proxies |
| `3d3eeae3` | Line-ending-proof hashing of the pre-registration |
| `674df095` | **Pre-registration frozen** (FX and commodity proxies, proxy-book fidelity, closed-market rule) |
| `2e9e39f7` | The one look (H1 FAIL), forward lockbox registration, result artifacts |
