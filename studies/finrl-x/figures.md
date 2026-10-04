# Ep03 Figures

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investment), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan). Historical simulation and paper trading only; not investment advice.

> Figures for KISI Ep03; study overview in [README.md](README.md). **Reviewed and locked by Keng 2026-10-03.** Sources: the three `docs/research/finrl_x_*` docs on public Sharpen `main` @ 51a7a69, and a cloud rerun of FinRL-X @ `4409abe9` (Linux, Python 3.13, CPU, Yahoo via FinRL-X's own downloader with `auto_adjust=True`). Column "Rerun" = reproduced by that cloud rerun (✅) or record only (—). **All 37 lists rerun 2026-10-02/03: every figure from the report reproduced exactly, parity PASS on all 37.** Locks once Keng approves; re-checked against `studies/finrl-x/` when the repo work lands.

## Rerun record (cloud, 2026-10-02/03)
All 37 lists were run through FinRL-X's own code at `4409abe9`; parity against FinRL-X's printed table passed on every run. One FinRL-X backtest = **9 min 20 s** on one core; download 13 s; scoring 4 s. Per-list results: [rerun-37-lists.txt](rerun-37-lists.txt). This study does not ship a one-command rerun.

## The claims (FinRL-X README and paper; 2018-01-07 → 2025-10-24, "10 bps per side")
| # | Figure | Value | Source | Rerun |
|---|---|---|---|---|
| C1 | Rolling Strategy | **5.98x**, 25.85%/yr, vol 27.85%, Sharpe 0.93, max DD −38.95% | report §1 | — (not reproducible) |
| C2 | Adaptive Rotation | **4.80x**, 22.32%/yr, vol 20.30%, Sharpe 1.10, max DD −21.46% | report §1 | — |
| C3 | QQQ / SPY (their table) | 4.02x / 2.80x | report §1 | — |
| C4 | Paper trading, 2025-10-26 → 2026-03-12 | **+19.76%**, Sharpe 1.96, max DD −12.22% | report §1 | — (live record) |
| C5 | Their chart's label for the gain | "Precious Metals & Mining Stocks Rally" (Dec 2025 – late Jan 2026) | report §2 | — |

## Step 1 · reproduction (Adaptive Rotation, dividend-adjusted, FinRL-X's own code)
| # | Figure | Value | Source | Rerun |
|---|---|---|---|---|
| R1 | Reproduced, no costs | **4.91x**, 22.69%/yr, vol 22.06%, Sharpe (theirs) 1.03, max DD −23.55% | report §2 | ✅ |
| R2 | Reproduced, 10 bps per side | **3.02x**, 15.25%/yr, Sharpe (theirs) 0.69 | report §2 | ✅ |
| R3 | QQQ, same data | **3.95x**, 19.30%/yr, Sharpe (theirs) 0.84, max DD −35.06%; SPY 2.75x, 13.90% | report §2 | ✅ |
| R4 | Turnover | **62.5x** capital a year, buys and sells counted (~31x one-way) | report §3 | ✅ |
| R5 | Cost grid (per year / standard Sharpe) | 0 bps 22.69% / 1.07 · 2 bps 21.17% / 1.01 · 3 bps 20.41% / 0.98 · 5 bps 18.91% / 0.92 · 10 bps 15.25% / 0.77 · QQQ 19.30% / 0.91 | report §3 | ✅ |
| R6 | Break-even vs QQQ | **4.5 bps** per side on return (4.48), 5.3 on Sharpe (5.34) | report §3 | ✅ |
| F1 | Price-only QQQ (README's own download) | 3.75x vs claimed 4.02x | report §2, §8 | — |
| F2 | Each bp of cost per side costs | ≈ 0.6% a year | report §3 | derived (62.5 × 0.01%) |
| F3 | Holdings replaced per week | ~2.8 of ~5; only 4% of weeks unchanged | report §3 | — |
| F4 | Install | `pip install -r requirements.txt` fails (`finnhub` vs `finnhub-python`); DRL packages not listed | report §2 | ✅ (seen in cloud setup) |
| F5 | Code reading @ 4409abe9 | no cost term (run_adaptive_rotation_strategy.py l. 443–456) · stop-losses logged, not applied (l. 268–311) · Sharpe = CAGR ÷ vol (l. 476) · table weekly, chart daily; chart code not in repo · fills at the decision close | report §2 | — |
| F6 | Rolling Strategy inputs missing | `sp500_tickers_daily_price_20250712.csv`, `result/stock_selected.csv`; NASDAQ-100 list is today's members | report §2 | — |
| F7 | Rotation alone over the paper-trading window, after 10 bps | 1.12x (QQQ 0.95x, SPY 0.99x) | report §2 | — |
| F8 | Use Case 1 (allocation methods + KAMA timing) | README shows a chart only, no results table; no KAMA code and no code for that chart in the repo; not tested | README @ 4409abe9; repo search 2026-10-03 | — |

## Step 2 · the hindsight test
| # | Figure | Value | Source | Rerun |
|---|---|---|---|---|
| H1 | Growth list | AAPL MSFT NVDA META AMZN GOOGL TSLA, committed **2026-02-05**, backtested from 2018 | report §4 | — |
| H2 | Tesla joined the S&P 500 | December 2020 | report §4 | — |
| H3 | Pool | 20 largest S&P 500 members on 2017-12-29 in 2017-era IT + Consumer Discretionary (list in README); NVIDIA #16, **$117B** | report §4 | — (needs private membership files) |
| H4 | Lists tested | Mag 7 · 7 largest · **35** random (seed 20260925, sha256 `ebe2c118…`) = **37** backtests | report §4 | ✅ (draws regenerated) |
| H5 | Rules frozen | 2026-09-25 00:45 UTC, SHA-256, before the first run | report §4 | — |
| H6 | 7 largest | 13.42% no costs · 12.08% at 2 bps · **−7.22 pts** vs QQQ · Sharpe 0.79 · −12.43 pts at 10 bps | report §4 | ✅ (13.42 / 12.08 / 6.87) |
| H7 | Random lists beating QQQ at 2 bps | **0 of 35**; Mag 7 beats all 35 | report §4 | ✅ |
| H8 | Random median | 11.24% no costs · **−9.18 pts** at 2 bps · −13.71 at 10 bps | report §4 | ✅ |
| H9 | Best random (ACN AMZN AVGO GOOGL MCD NVDA ORCL) | 3.95x · 19.31% no costs (ties QQQ 19.30%) · −1.41 pts at 2 bps · −6.93 at 10 bps | report §4 | ✅ |
| H10 | Worst random | 8.01% · −12.47 pts at 2 bps | report §4 | ✅ |
| H11 | Mag 7 at 2 bps | 21.17% · **+1.87 pts** · Sharpe 1.01 (+0.10) · −4.05 pts at 10 bps | report §4 | ✅ |
| H12 | Verdict | **HINDSIGHT** (return rule, Sharpe rule and 10 bps agree) | report §4 | ✅ |
| H13 | After the freeze, 2026-02-06 → 2026-09-18, after 10 bps | Mag 7 1.21x vs QQQ 1.19x (29 weeks) | report §4 | — |
| H14 | 2017 list traps | Alphabet counted twice · Comcast $83B vs ~$191B (2010 share count) · Visa as-converted 2,350M shares → $268B · Disney, Broadcom under old SEC IDs · SEC bulk data missed Disney · sector labels changed | report §4 | — |
| H15 | End-2017 caps match | Apple $869B, Microsoft $660B, NVIDIA $117B | report §4 | — |

## Full rerun (cloud, 2026-10-02/03)
Harness: FinRL-X @ 4409abe9, one shared dividend-adjusted Yahoo snapshot (39 symbols), configs differ only in the growth list and output paths, 2 runs in parallel, ~10 min per pair. Per-list results: `rerun-37-lists.txt` (this folder).
- 35 random lists: **0 of 35** beat QQQ at 2 bps; median no-cost 11.24%; median vs QQQ −9.18 pts (2 bps), −13.71 pts (10 bps), Sharpe −0.23; best −1.41 (rand_02), worst −12.47 (rand_24).
- Pre-registered rules: return HINDSIGHT · Sharpe HINDSIGHT · 10 bps HINDSIGHT → **HINDSIGHT**.
- Note: two random lists (rand_20, rand_02) edge QQQ's Sharpe at 2 bps by ≤ 0.007 (hindsight doc: "2 of 35"); their returns still trail. The Mag 7 still beats the 90th percentile on Sharpe (+0.10 vs p90 −0.05).

## The study itself
| # | Figure | Value | Source |
|---|---|---|---|
| S1 | Time | 2 calendar days (2026-09-24 → 09-25) | report §9 |
| S2 | Data | free: SEC EDGAR + Yahoo Finance | report §9 |
| S3 | Backtests | 41 recorded (4 step 1, 37 step 2), plus validation re-runs | report §9 |
| S4 | Checks | 20 automated tests; 17 planted bugs each caught | report App. B |

## Follow-up: keel-v1 (record only, not rerun here)
Source: [keel-v1-final-report.md](keel-v1-final-report.md). Window 1973-01-02 to 2005-12-30. Design: half SPY, half 18-ETF cross-asset trend sleeve; rules, gates and 84 variants committed before the one look (commit 674df095). Versus SPY after costs, borrow fees and next-day execution: Sharpe 0.44 vs 0.33; return 10.3% vs 10.8% a year; volatility 8.8% vs 16.1%; max drawdown -21.0% vs -47.5%; 1973-74 -10% vs -38%; 2000-02 -18% vs -48%. Alpha +1.15% a year, t 1.996 vs bar 2.0; deflated alpha 0.63 vs 0.95; 6 of 8 gates; borderline NO-GO. Beat SPY in 12 of 33 calendar years. Trend Sharpe 0.60 as researched, 0.40 traded realistically. Forward lockbox open, first check after 63 trading days (about late December 2026).
