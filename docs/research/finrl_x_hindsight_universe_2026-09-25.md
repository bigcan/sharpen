# FinRL-X Adaptive Rotation — hindsight-universe test (2026-09-25)

Follows `finrl_x_reproduction_2026-09-24.md` (finding 6). FinRL-X's growth group is AAPL, MSFT, NVDA, META, AMZN,
GOOGL, TSLA — 2026's Magnificent 7, committed on 2026-02-05 and backtested from 2018. The reproduced in-sample edge over
QQQ is thin (break-even ~4.5 bps per side). This test asks whether any of it survives a growth group an investor could
have chosen at the start of 2018.

## Pre-registration (frozen before any run; sha256 in `results/finrl_x/hindsight/prereg.sha256`)

**Point-in-time pool.** The 20 largest S&P 500 members on 2017-12-29 by market cap, among 2017-era GICS Information
Technology and Consumer Discretionary — the two sectors the Magnificent 7 sat in then (Alphabet and Facebook were IT
until 2018; Amazon and Tesla are Consumer Discretionary). Built by `scripts/research/finrl_x_pit_universe.py`:

- Market cap = shares × the close as printed on 2017-12-29 (Yahoo split-adjusted close × every later split factor,
  which also undoes spin-off adjustments such as IBM/Kyndryl).
- Shares = the latest SEC XBRL count **dated and filed on or before 2017-12-29**, dated within 200 days: cover page,
  else balance sheet, else quarterly diluted weighted shares. All 20 pool members were sized this way, filed
  2017-09-27 → 2017-12-21. Exceptions, all documented in the script: Disney and Broadcom read under their 2017 SEC
  registrants; Visa uses its own as-converted class A total (2,350M, 10-K 2017-11-17); one ticker per company (GOOGL).
- Result: AAPL GOOGL MSFT AMZN META V HD INTC ORCL CMCSA CSCO DIS MA IBM MCD NVDA AVGO NKE TXN ACN
  (#20 ACN $100.5B, #21 QCOM $94.4B). NVDA is in (#16, $117B); TSLA (~$50B) is not.
- Known gaps: Electronic Arts and Twenty-First Century Fox could not be sized; 145 end-2017 members that have since
  left the index cannot be priced from current data. Any of them would need > $100.5B to enter the pool.

**Arms** — every arm runs FinRL-X's own runner with the arguments `deploy.sh` uses (config v1.2.1, `W-FRI`, daily
fast-track on), 2018-01-07 → 2025-10-24, one dividend-adjusted Yahoo snapshot shared by all arms. The only config change
is `asset_groups.group_a_growth_tech.symbols` (plus per-arm output paths); a structural diff asserts it.

1. `mag7` — the published group, re-run on the shared snapshot (reference).
2. `pit_top7` — the pool's 7 largest: AAPL GOOGL MSFT AMZN META V HD.
3. `rand_00` … `rand_34` — 35 distinct random 7-name groups from the pool, seed 20260925; draw spec sha256
   `ebe2c118130ad1372b0a122dfe9828eb77877dc776180794552bf77215859d8f` (groups listed in `pool.json`).

**Metrics** — scored by `finrl_x_rotation_repro.py` (parity-checked against FinRL-X's printed table): ΔCAGR and
Δ arithmetic Sharpe versus QQQ at **2 bps per side (primary**, our US large-cap cost prior) and 10 bps (the paper's),
and the break-even cost versus QQQ.

**Decision rules** (primary metric: ΔCAGR vs QQQ at 2 bps):

- **HINDSIGHT** — `mag7` lies above the 90th percentile of the 35 random arms **and** the random-arm median is ≤ 0.
  The in-sample edge is attributed to choosing the universe with hindsight; the Rotation is recorded NO-GO.
- **ROBUST** — the random-arm median is > 0 **and** `pit_top7` is > 0. The rotation logic adds value on universes
  chosen ex ante; proceed to factor attribution (QQQ / SPY / GLD / TLT) and a one-day execution lag.
- **INCONCLUSIVE** — anything else, or the CAGR and Sharpe readings disagree. Report the distribution; no further
  investment without new evidence.

**Not tested here:** the real-assets group (XOM, CVX, COP, FCX, BHP, GLD, SLV) is a second possible hindsight axis
and is left as published. The 2018–2025 window is the strategy's design period for every arm.

## Results (run 2026-09-25, after the freeze)

**Verdict: HINDSIGHT — the Adaptive Rotation is NO-GO.** Both pre-registered readings agree, and so does the paper's
own 10 bps. No growth group an investor could have picked from 2017's 20 largest tech/consumer names beats QQQ; the
published Magnificent 7 beats every one of them.

| Arm | CAGR, 0 bps | CAGR, 2 bps | vs QQQ, 2 bps | Arith. Sharpe, 2 bps (vs QQQ) | vs QQQ, 10 bps |
|---|---|---|---|---|---|
| `mag7` (published) | 22.69% | 21.17% | **+1.87 pts** | 1.01 (+0.10) | −4.05 pts |
| `pit_top7` (AAPL GOOGL MSFT AMZN META V HD) | 13.42% | 12.08% | −7.22 pts | 0.79 (−0.12) | −12.43 pts |
| 35 random groups — median | 11.24% | — | **−9.18 pts** | (−0.23) | −13.71 pts |
| 35 random groups — best (`rand_02`) | 19.31% | 17.89% | −1.41 pts | 0.91 (+0.00) | −6.93 pts |
| 35 random groups — worst (`rand_24`) | 8.01% | 6.83% | −12.47 pts | 0.51 (−0.40) | −17.07 pts |
| QQQ / SPY (buy and hold) | 19.30% / 13.90% | | | 0.91 / 0.79 | |

- Primary rule (ΔCAGR at 2 bps): `mag7` +1.87 pts vs random p90 −4.24 and median −9.18 → HINDSIGHT. **0 of 35** random
  groups beat QQQ; `mag7` is above all 35. Sharpe reading: `mag7` +0.10 vs p90 −0.05 and median −0.23 → HINDSIGHT
  (2 of 35 random groups edge QQQ's Sharpe by ≤ 0.007).
- Even before any cost, the median ex-ante group trails QQQ by 8 pts a year and trails SPY too; the point-in-time
  top 7 roughly matches SPY. The best random group (ACN AMZN AVGO GOOGL MCD NVDA ORCL) holds two of the decade's
  biggest winners and only ties QQQ.
- Why: the strategy rotates weekly among a *fixed* list and holds at most two names per group, so its outcome is set
  by which names are on the list. QQQ's cap-weighting absorbs new winners automatically (NVDA, TSLA); a 2017 list
  cannot. The published list simply contains the 2018–2025 winners.
- Checks: all 37 arms pass parity with FinRL-X's printed table; QQQ is identical across arms (one snapshot); the `mag7`
  re-run reproduces the 2026-09-24 reproduction exactly (4.9100x, 22.6923%). Artifacts: `results/finrl_x/hindsight/`
  (`verdict.json`, per-arm configs, logs, weights, scores).
- The post-freeze window (1.21x vs QQQ 1.19x, 29 weeks) does not change this: it runs the same hindsight-chosen list
  forward for seven months and cannot separate the rotation logic from that list.

**Correction to the pre-registration text** (left unedited above, since it is frozen): TSLA sat outside the pool
because it joined the S&P 500 only on 2020-12-21, not merely because of its ~$50B size. The pool is unchanged.

**Scope of the NO-GO:** the Rotation as designed, with any growth list chosen without hindsight. The real-assets group
was left as published, and all arms share the 2018–2025 design window. Together with the reproduction (no costs in
the P&L, stops not applied, Rolling Strategy not reproducible), nothing in FinRL-X's published results survives to
the funnel; the workstream closes unless new evidence arrives.
