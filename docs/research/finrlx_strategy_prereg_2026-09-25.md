# Pre-registration — `keel-v1`: SPY core + implementable cross-asset trend, on FinRL-X, vs SPY

**Date:** 2026-09-25 · **Branch:** `finrl-x-strategy` · **Status:** frozen at commit; the sealed window opens only after
this file and `configs/finrlx_strategy.gates.yaml` are committed with this file's SHA-256 in the gates file.

## 1. Question

Does a FinRL-X strategy built from this project's only validated edge beat SPY after costs, on data it has never seen?
**Primary claim (H1):** on the sealed window, the book's net excess Sharpe is above SPY's AND its alpha vs SPY is
positive with a meaningful, deflated t-stat.

## 2. What the design window already says (2008-07-01 → 2026-06-30, ETF data, seen)

Design evidence only: this window chose the cell. Source: `results/finrlx_strategy/dev_grid.json`, `dev_report.json`.

| | Selected book | SPY |
|---|---|---|
| Net excess Sharpe | 0.673 | 0.618 |
| Alpha vs SPY, monthly NW | +1.09%/yr, t 1.28 (daily t 1.15) | — |
| Beta / residual IR | 0.50 / 0.36 | 1 / — |
| Volatility / max drawdown | 10.5% / −23.3% | 19.8% / −50.7% |
| Frictionless / harsh (x5) Sharpe | 0.686 / 0.618 | |
| Break-even cost (Sharpe = SPY) | 5.0x base costs (~10 bps liquid, 25 bps thin ETFs) | |
| Book DSR / PSR / MinTRL (N = 84) | 0.995 / 0.998 / 6.1 years | |
| Alpha-stream DSR (N = 84) | **0.63** (would FAIL the 0.95 gate) | |
| PBO over the 24 grid cells | 0.13 | |
| Alpha by quarter of the window | +1.21, +2.31, −0.69, +0.71 %/yr | |
| Paired bootstrap Sharpe difference | CI [−0.09, +0.20], P(≤0) = 0.25 | |
| Time-series IC (conviction vs next month) | pooled Spearman 0.032, sign hit rate 52.2% | |

On its design window the book would fail H1 on the alpha t (1.28 < 2) and the deflated-alpha leg. The Sharpen signal
funnel (cross-sectional) scores the trend conviction LOGGED with IC-IR −0.026 (noise control +0.019): as expected, since
the funnel z-scores each day's cross-section and removes the net long/short direction that time-series trend trades.
Tier 0 (truncation-equivalence leak test) passes for every candidate.

Also learned on the design window (reported so it is not rediscovered): the project's "TSMOM net Sharpe 0.60" is a
research basis (trades at the decision close, total not excess return). Financed at the T-bill rate and executed one bar
later it averages 0.40, and the month-end rebalance date was lucky: across the 21 possible rebalance days next-bar Sharpe
ranges 0.32–0.49. Tranching removes that luck.

## 3. The strategy (frozen)

- **FinRL-X contract.** Every date ends in target weights per ETF. The long-only special case runs through FinRL-X's own
  `BacktestEngine` with exact NAV parity to our engine (0.0 max difference at 0 bps, 7e-6 at 2 bps over 4,527 days;
  `tests/research/test_finrlx_strategy_bridge.py`). Our engine extends it with shorts, borrow and next-bar execution.
- **Universe:** SPY QQQ IWM EFA EEM · TLT IEF LQD · GLD SLV DBC USO DBA · UUP FXE FXY FXB FXA (fixed, chosen 2026-06).
- **Signal:** `sharpen.features.cross_asset_signals` unchanged: mean sign of 63/126/252-day returns, 5-day skip; raw
  weight = conviction × min(0.10 / 63-day vol, 2).
- **Book (cell `cov_sv03_static_sc0.5`):** 50% SPY core (static) + trend sleeve scaled to a 3% ex-ante vol with the full
  126-day covariance (lagged one bar), then scaled down (never up) so that longs ≤ 100% of NAV (no margin loan) and
  shorts ≤ 50%; four tranches decided at month-end + 0/5/10/15 trading days, book = mean of the latest four.
- **Execution and account:** decide at close d, trade at close d+1; holdings drift between trades; one-way cost 2 bps
  (liquid) / 5 bps (SLV DBC USO DBA UUP FXE FXY FXB FXA); borrow 50 / 100 bps a year; short proceeds earn nothing; idle
  cash earns the 3-month bill minus 15 bps; on proxies, each ETF's expense ratio is charged daily
  (`configs/finrlx_strategy.yaml`).

## 4. Hypotheses

- **H1 (primary):** the selected book vs SPY (total-return proxy, minus SPY's 9.45 bps fee).
- **H2:** the covariance allocator vs the correlation-blind linear allocator (`linear_sv03_static_sc0.5`), all else equal.
  Design window: +0.014 Sharpe, P(≤0) 0.36 — expected to fail the 0.10 bar; kept because it was declared.
- **H3:** a vol-managed core (`cov_sv03_vol_managed_sc0.5`: core = 0.5 × min(1, 0.16 / SPY 63-day vol)) vs the static
  core. Design window: −0.018 Sharpe — expected to fail.

## 5. Data for the sealed window

Daily total-return proxies, US trading calendar, built by `scripts/research/finrlx_strategy/data_*.py` (manifests with
SHA-256 in `data/raw/long_history/`). Each proxy was chosen by a rule declared before its fidelity was computed (the
rates rule was amended once after seeing the dev fit; recorded). Fidelity is measured on the design window only.
**An asset enters the book on its own once its proxy has the ~320 bars the signal needs (252-day lookback + 5-day
skip + 63-day vol; the 126-day covariance fits inside that); no manual inclusion.** Every recommended proxy is used; none is dropped for fidelity.

| ETF | Proxy | Daily from | Dev weekly corr | Main bias |
|---|---|---|---|---|
| SPY (benchmark + core) | ^GSPC + Shiller prior-month D/P | 1928 | 0.998 | smoothed dividends |
| QQQ | ^IXIC price | 1971-02 | 0.987 | no dividends; broad OTC index pre-1990 |
| IWM | Ken French size quintile 2 (VW) | 1926 | 0.992 | NYSE breakpoints |
| EFA | Ken French Developed ex US | 1990-07 | 0.958 | local closes, incl. Canada |
| EEM | Vanguard EM index fund (VEIEX) | 1994-05 | 0.974 | net of fund fees |
| TLT | H.15 20y/30y par bond, M=25, roll-down | 1962 | 0.986 | constant maturity |
| IEF | H.15 7y/10y par bond, M=8.5, roll-down | 1962 | 0.988 | constant maturity |
| LQD | 10y + Moody's (Aaa+Baa)/2 spread, M=10 | 1983 | 0.749 | no defaults; weak fidelity |
| GLD | LBMA gold PM fix | 1968 | 0.924 | London fix timing (10:00 NY) |
| SLV | LBMA silver fix | 1968 | 0.844 | fix timing (07:00 NY) |
| USO | rolled front-month WTI futures (EIA C1/C2) + T-bill | 1985 | 0.966 | one-day roll vs USO's four |
| DBC | DBC-weighted rolled energy futures + gold/silver | 1985 | 0.897 | no metals/grains; WTI-heavy before 1994 |
| DBA | **none** — no free daily agriculture series | — | — | absent from the sealed-window book |
| UUP | ICE DXY rebuilt from H.10 + rate differential + T-bill | 1971 | 0.959 | ignores futures basis |
| FXE / FXY / FXB / FXA | H.10 spot (DEM x 1.95583 before 1999) + foreign short rate | 1971 | 0.974–0.979 | interbank carry above the trusts' deposit rate |
| Cash (rf) | H.15 3-month bill, bond-equivalent | 1954 | — | — |

Live at the window start (1973-01-02): SPY QQQ IWM TLT IEF GLD SLV UUP FXE FXY FXB FXA (12). Enter later as their
proxies warm: LQD (~1984), USO and DBC (~1986), EFA (~1991), EEM (~1995). DBA never.

**Fees.** Proxy returns are charged each ETF's expense ratio minus any fee already inside the proxy: FX proxies are built
net of the trusts' fees (charged 0); EEM's proxy is a fund NAV already net of its own fee (residual 30 bps); commodity
proxies are loaded gross and charged the full fee (`configs/finrlx_strategy.yaml`).
**Closed markets.** If an asset has no price on an execution day, it is not traded that day; its holding is kept and the
tradeable longs shrink if needed to stay funded (`pnl.run`, tested).

**Proxy-BOOK fidelity (design window, the check that matters).** The selected strategy run on the proxy panel vs on the
real ETFs, same dates, LQD and DBA dropped from both (`results/finrlx_strategy/proxy_book_fidelity_dev.json`):

| 2008-07 → 2024-03 | Daily corr | Weekly | Monthly | Tracking error | Sharpe | Alpha t |
|---|---|---|---|---|---|---|
| Proxy book vs ETF book | 0.987 | 0.995 | 0.997 | 1.7%/yr | 0.637 vs 0.664 | 1.24 vs 1.43 |
| Trend sleeve alone | | | 0.982 | | | |

With LQD kept (2008-07 → 2016-09): monthly 0.997, sleeve 0.979. The proxies reproduce the tradeable strategy closely and
slightly conservatively. A full dry run of `one_look.py` on the design-window proxies (no seal, no record) reproduced the
design verdict: FAIL on alpha t (1.47) and deflated alpha (0.61); the other six legs pass.

## 6. Window, gates, deflation (machine-readable; read by `one_look.py`)

```yaml
window:
  start: "1973-01-02"      # after Bretton Woods; FX (from 1971-01), equity and rates proxies are warm by then
  end: "2005-12-31"
h1:
  cell: cov_sv03_static_sc0.5
h2:
  cell: cov_sv03_static_sc0.5
  baseline: linear_sv03_static_sc0.5
h3:
  cell: cov_sv03_vol_managed_sc0.5
  baseline: cov_sv03_static_sc0.5
deflation:
  n_trials: 84               # 60 distinct variants in results/finrlx_strategy/trial_ledger.jsonl + 24 prior TSMOM family
  prior_family_trials: 24    # configs/tailwind_v1.gates.yaml overfitting.dsr_n_trials
```

Gates: `configs/finrlx_strategy.gates.yaml` (committed with this file). H1 passes only if **all** hold on the sealed
window at base costs: Sharpe > SPY's; monthly NW alpha t ≥ 2.0; alpha-stream DSR ≥ 0.95 and book DSR ≥ 0.95 (trial
Sharpes = the 24 grid cells on the sealed window, N = 84); PSR ≥ 0.95; PBO over the 24 cells ≤ 0.5; frictionless minus
net Sharpe ≤ 0.15; alpha > 0 in ≥ 3 of 4 equal subperiods. H2 and H3 are tested only if H1 passes, each at 0.025:
Sharpe uplift ≥ 0.10 and paired-bootstrap P(uplift ≤ 0) ≤ 0.025. Reported, not gating: harsh-cost Sharpe, break-even
cost, the Sharpe-difference bootstrap CI, drawdown, exposure, time-series IC, per-subperiod numbers.

## 7. Power (stated before the look)

The alpha test's t is about IR × √years. At the design-window IR of 0.36 over 33 years, E[t] ≈ 2.1, so P(t ≥ 2) ≈ 0.5.
The deflated-alpha leg needs roughly IR ≥ SR* + 1.645/√33 ≈ 0.21 + 0.29 = 0.50 (SR* from the dev grid's IR dispersion),
so at IR 0.36 it fails more often than not. **If the design-window IR is the truth, H1 passes with probability of
roughly 0.2–0.35.** The literature reports stronger trend returns before 2009; at IR 0.6 the pass probability is above
0.8. A direct test that the Sharpe *difference* is positive is underpowered even here (t ≈ 1.8 at 30 years), which is
why it is reported and not gated. A forward lockbox alone could not decide this in any useful time (one year: t ≈ 0.4).

## 8. Look protocol

1. `one_look.py` refuses to run until the seal is lifted (this file committed, unmodified, SHA-256 in the gates file).
2. It fingerprints the code, config, gates, this file and the signal module, and appends to `results/finrlx_strategy/looks.jsonl`.
   A second run is allowed only with identical fingerprints (a replay); changed rules are refused.
3. Every grid cell's sealed-window result is appended to the trial ledger.
4. Nothing about the sealed window is inspected before the look except coverage and outlier counts (`seal.hygiene`).

## 9. What each outcome means

- **H1 PASS:** candidate, not tradeable. Next: the Tier-2 deep lifecycle audit (pre-authorised), a forward lockbox from
  the look date (house incubation: ≥ 63 bars, forward Sharpe ≥ 0.30), and an execution adapter for shorts (FinRL-X's
  `AlpacaManager` clamps negative weights to zero). No paper or live trading without the operator's approval.
- **H1 FAIL:** NO-GO for "beats SPY" at these gates, with the failing legs reported. No second look. A new idea needs a
  new pre-registered test on new data.

## 10. Known limits

- Proxies are not the ETFs: spot vs futures (commodities), missing or approximate carry (FX), constant-maturity bonds,
  a price-only Nasdaq, and today's ETF costs applied to 1973–2005.
- The trend family was documented in this era by published research (e.g. century-of-trend studies). A pass confirms that
  this project's implementation transfers to unseen data; it is not independent evidence that trend exists.
- The design window (2007–2026) is the proxies' calibration window, so proxy choice used post-2005 information about
  proxy-to-ETF fit (never about strategy performance).
