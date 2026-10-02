# Ep03 Figures

> Figures for KISI Ep03; study overview in [README.md](README.md). Checked against Sharpen branch `study/finrl-x` @ **0b79363b** (2026-10-02) and FinRL-X @ `4409abe9`. Rerun figures (R#) come from the commands below; record figures (F#) are quoted from the research docs and are not rerun.

## G1 · the one-line rerun ✅ (2026-10-02)
```
git clone https://github.com/bigcan/sharpen && cd sharpen
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/research/finrl_x_rerun.py
```
Fresh clone, fresh venv, Windows 11, Python 3.11, 16-core desktop, CPU only, no keys or private data. **8.4 min wall** (three backtests in parallel; includes cloning FinRL-X, building its environment and the Yahoo download). Output:

```
FinRL-X Adaptive Rotation @ 4409abe9, 2018-01-07 -> 2025-10-24, dividend-adjusted Yahoo closes, costs per side

Growth list                    Growth   Per year 0 / 2 / 10 bps    vs QQQ, 2 bps   Parity
Magnificent 7 (as published)    4.91x   22.69% / 21.17% / 15.25%     +1.87 pts   PASS
7 largest on 2017-12-29         2.66x   13.42% / 12.08% / 6.87%      -7.22 pts   PASS
Best random list (rand_02)      3.95x   19.31% / 17.89% / 12.37%     -1.41 pts   PASS
SPY, buy and hold               2.75x   13.90%
QQQ, buy and hold               3.95x   19.30%

Magnificent 7: turnover 62.5x a year (buys and sells); break-even vs QQQ 4.48 bps per side on return, 5.34 on Sharpe
Magnificent 7 cost grid (per year / Sharpe): 0 bps 22.69% / 1.07  2 bps 21.17% / 1.01  3 bps 20.41% / 0.98  5 bps 18.91% / 0.92  10 bps 15.25% / 0.77  QQQ 19.30% / 0.91
```
Matches the report and the 2026-10-02 cloud dry-run (Linux, Python 3.13, one core: 9 min 20 s per backtest) on every figure.

## G2 · all 37 lists ✅ (2026-10-02)
```
python scripts/research/finrl_x_rerun.py --full --jobs 12
```
Same fresh clone, straight after G1 (its three lists and the price snapshot are reused). **23.6 min wall** for the 34 remaining backtests, 12 at a time. Parity PASS on all 37. Summary lines:

```
Random lists beating QQQ at 2 bps: 0 of 35
Random lists vs QQQ at 2 bps, points a year: median -9.18, best rand_02 -1.41, worst rand_24 -12.47
Magnificent 7 beats 35 of 35 random lists
Verdict (rules frozen before the first run): HINDSIGHT (return rule HINDSIGHT, Sharpe rule HINDSIGHT, at 10 bps HINDSIGHT)
```

## Rerun figures
| # | Figure | Value | Source |
|---|---|---|---|
| R1 | Magnificent 7, no costs | **4.91x**, 22.69% a year | G1 |
| R2 | Magnificent 7 at 2 / 10 bps per side | 21.17% / **15.25%** a year | G1 |
| R3 | QQQ, same data | **3.95x**, 19.30% a year; SPY 2.75x, 13.90% | G1 |
| R4 | Magnificent 7 vs QQQ at 2 bps | **+1.87 pts** a year | G1 |
| R5 | Turnover | **62.5x** capital a year, buys and sells counted (about 31x one-way) | G1 |
| R6 | Cost grid (per year / standard Sharpe) | 0 bps 22.69% / 1.07 · 2 bps 21.17% / 1.01 · 3 bps 20.41% / 0.98 · 5 bps 18.91% / 0.92 · 10 bps 15.25% / 0.77 · QQQ 19.30% / 0.91 | G1 |
| R7 | Break-even vs QQQ | **4.48 bps** per side on return, 5.34 on Sharpe (voice: "about four and a half") | G1 |
| R8 | 7 largest on 2017-12-29 (AAPL GOOGL MSFT AMZN META V HD) | 2.66x · 13.42% no costs · 12.08% at 2 bps · **−7.22 pts** vs QQQ · 6.87% at 10 bps | G1 |
| R9 | Best random list (`rand_02`: ACN AMZN AVGO GOOGL MCD NVDA ORCL) | 3.95x · 19.31% no costs (ties QQQ's 19.30%) · 17.89% at 2 bps · **−1.41 pts** | G1 |
| R10 | Parity with FinRL-X's own printed table | PASS on every run | G1, G2 |
| R11 | Lists tested | Magnificent 7 · 7 largest · **35** random = **37** backtests; seed 20260925, draw-spec sha256 `ebe2c118…` regenerated and checked | G1 (the script asserts it) |
| R12 | Random lists beating QQQ at 2 bps | **0 of 35**; the Magnificent 7 beats all 35 | G2 |
| R13 | Random median | 11.24% a year no costs (`rand_05`) · **−9.18 pts** vs QQQ at 2 bps | G2 |
| R14 | Worst random list (`rand_24`: AAPL CSCO DIS GOOGL META NKE TXN) | 1.82x · 8.01% no costs · 6.83% at 2 bps · **−12.47 pts** | G2 |
| R15 | Verdict | **HINDSIGHT** (return rule, Sharpe rule and 10 bps agree) | G2 |

## Record figures (from the research docs; not rerun)
| # | Figure | Value | Source |
|---|---|---|---|
| F1 | Claimed, Rolling Strategy | **5.98x**, 25.85% a year, vol 27.85%, Sharpe 0.93, max drawdown −38.95% | final report §1 |
| F2 | Claimed, Adaptive Rotation | **4.80x**, 22.32% a year, vol 20.30%, Sharpe 1.10, max drawdown −21.46% | final report §1 |
| F3 | Claimed, QQQ / SPY | 4.02x / 2.80x | final report §1 |
| F4 | Paper trading, 2025-10-26 → 2026-03-12 | **+19.76%**, Sharpe 1.96, max drawdown −12.22%; a live record, cannot be re-run | final report §1, §2 |
| F5 | FinRL-X's own chart label for that gain | "Precious Metals & Mining Stocks Rally" (December 2025 to late January 2026) | final report §2 |
| F6 | Reproduced at 10 bps, growth | **3.02x** (Sharpe, FinRL-X's formula, 0.69) | final report §2 |
| F7 | Price-only QQQ (the README's own download) | 3.75x vs the claimed 4.02x | final report §2, §8 |
| F8 | Each bp of cost per side costs | about 0.6% a year (62.5 × 0.01%) | final report §3 |
| F9 | Holdings replaced per week | about 2.8 of about 5; only 4% of weeks unchanged | final report §3 |
| F10 | Install | `pip install -r requirements.txt` fails (`finnhub` vs `finnhub-python`); DRL packages not listed | final report §2 |
| F11 | Code reading @ `4409abe9` | no cost term (`run_adaptive_rotation_strategy.py` lines 443–456) · stop-losses logged, not applied (268–311) · Sharpe = compound return ÷ volatility (476) · table marked weekly, chart daily, chart code not in the repository · fills at the decision close | final report §2 |
| F12 | Rolling Strategy inputs missing | `sp500_tickers_daily_price_20250712.csv`, `result/stock_selected.csv`; its NASDAQ-100 list is today's members | final report §2 |
| F13 | Rotation alone over the paper-trading window, after 10 bps | 1.12x (QQQ 0.95x, SPY 0.99x) | final report §2 |
| F14 | Growth list | AAPL MSFT NVDA META AMZN GOOGL TSLA, committed **2026-02-05**, backtested from 2018 | final report §4 |
| F15 | Tesla joined the S&P 500 | December 2020 | final report §4 |
| F16 | Pool | 20 largest S&P 500 members on 2017-12-29 in 2017-era Information Technology + Consumer Discretionary; NVIDIA #16, **$117B**; Tesla (about $50B) not a member | final report §4; hindsight doc |
| F17 | Rules frozen | 2026-09-25 00:45 UTC, SHA-256, before the first run | final report §4 |
| F18 | After the freeze, 2026-02-06 → 2026-09-18, after 10 bps | Magnificent 7 1.21x vs QQQ 1.19x (29 weeks) | final report §4 |
| F19 | 2017 list traps | Alphabet counted twice · Comcast $83B vs about $191B (2010 share count) · Visa as-converted 2,350M shares → $268B · Disney, Broadcom under old SEC IDs · SEC bulk data missed Disney · sector labels changed | final report §4 |
| F20 | End-2017 caps match | Apple $869B, Microsoft $660B, NVIDIA $117B | final report §4 |
| F21 | Time | 2 calendar days (2026-09-24 → 09-25) | final report §9 |
| F22 | Data | free: SEC EDGAR + Yahoo Finance | final report §9 |
| F23 | Backtests | 41 recorded (4 in step 1, 37 in step 2), plus validation re-runs | final report §9 |
| F24 | Checks in the original study | 20 automated tests; 17 deliberately planted faults, each caught | final report, Appendix B |

## Wording rules (Keng, 2026-10-02)
"Hindsight", "chosen in 2026", "the edge was the list", "could you have picked these in 2018?". "FinRL-X's code runs and its own numbers reproduce" describes the reproduction. The stock list is described as a choice made with hindsight, not as a coding error. Fair to AI4Finance.

## Open
- **Headline number:** the 0-of-35 test is on the Adaptive Rotation (claimed 4.80x). 5.98x is the Rolling Strategy, which cannot be reproduced. The title and cold open must not imply the 0-of-35 test ran on the 5.98x strategy.
- **Not public yet:** this study is held on the private repository. The `git clone https://github.com/bigcan/sharpen` line works once it is published; recheck G1 against public `main` then and put that sha in the header.
