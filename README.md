# Sharpen

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/bigcan)

**S**emi-agentic **H**ypothesis-to-**A**lpha **R**esearch **P**latform with **E**mpirical **N**ull-testing.

Sharpen is a quantitative research platform and the research record of the program that built
it: nine months of systematically trying to find tradeable edges, and mostly failing to. 76
strategies and probes across eight families were specified, measured and closed. One weak
edge survived and still did not clear its own deployment gates. **Nothing here was ever traded
with real capital.**

It is published for two reasons. The validation machinery is reusable, and the record of what
it rejected, and why, saves you rebuilding the same things. And along the way the project caught
itself shipping bugs, two of them look-ahead leaks, that had manufactured or disguised the results
it was about to act on. Those are written up in **[docs/LEAKS_FOUND.md](docs/LEAKS_FOUND.md)**, the
best place to start.

> **Not investment advice. Not a trading product.** Every performance figure in this repository
> comes from a historical simulation or a paper run. Read [DISCLAIMER.md](DISCLAIMER.md).

---

## What the record shows

Each claim below points at evidence you can re-run; claim 4 needs free Taiwan data you fetch yourself.

1. **76 strategies and probes, across eight families, were tested and closed.** Directional RL,
   options and volatility premium, arbitrage and market making, equity cross-section, sleeves and
   overlays, retail technical rules, free-data probes and automated-mining substrates. Every one,
   with the number that closed it and the evidence that ships, is in
   **[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md)**.

2. **One weak survivor: cross-asset time-series momentum (TSMOM),** built from **free daily public
   OHLCV**. Its honest size is a range, and the top of it is the flattering end:

   | Measurement | Net Sharpe |
   |---|---|
   | 18 ETFs chosen during development, total return, 2 bps | 0.60 |
   | Same book, in excess of T-bills on its net exposure | 0.51 |
   | Same frozen rule on 32 ETFs never used in development | **0.39** |

   The deployment test was run on the honest end. The book built on it (TSMOM plus a
   betting-against-beta sleeve), with momentum cut to the untouched-universe 0.39, has a
   **deflated Sharpe of 0.896 against a 0.95 bar**, so it failed. (On the curated 0.60 it would
   pass at 0.974; the project did not take that number because the curation is exactly the
   overfitting channel the test exists to price.) Its probability of backtest overfitting is
   0.0009, but that grid compares book *types*, not universes, so it says little about curation.
   A real but sub-threshold edge, not a product. See
   [`tailwind_v1_R1_dsr_pbo_2026-07-01.md`](docs/research/tailwind_v1_R1_dsr_pbo_2026-07-01.md) and
   [`fable_verdict_2026-06-11.md`](docs/research/fable_verdict_2026-06-11.md).

3. **Three bugs had manufactured or disguised results.** Two were look-ahead leaks, one a seeding
   defect. A multi-timeframe look-ahead was the
   *entire* edge of one strategy (profit factor 1.69 → 1.02 once fixed); a signal gate read the bar
   it was about to trade; and `--seed` never reached the environments, so every earlier multi-seed
   comparison measured noise. [docs/LEAKS_FOUND.md](docs/LEAKS_FOUND.md).

4. **The automated alpha miner has never produced a discovery, and the record cannot say why.** A
   search of the same size, run on data containing no signal at all, reaches a *higher* gate
   statistic (`marginal_t` **3.68**) than anything the real mining record ever produced (**2.12**).
   That figure needs the Taiwan panel (free via FinMind, `scripts/data/fetch_taiwan_finmind.py`,
   not shipped) and the full search budget, about 10 minutes per replicate:
   `python scripts/research/null_grid_sim.py --substrate taiwan --mode null --pop-size 200
   --n-generations 40 --replicates 4 --base-sharpe -0.007`. The default synthetic run is a smoke
   test and reaches a lower ceiling (about 2.2). See
   [`f4_null_grid_reproduction_2026-09-03.md`](docs/research/f4_null_grid_reproduction_2026-09-03.md).

5. **Free daily data sets a hard floor on what any honest test here could detect.** An idealised
   single pre-registered test needs an annualised ΔSharpe of **1.42** on 4 years of daily bars, or
   **0.71** on 16 years, for 80% power. The deployed six-leg validation contract needs **3.54** and
   **1.86**: a near-constant ~2.5× surcharge that more data does not remove. Realistic
   single-signal edges are 0.3–0.5. Check the arithmetic yourself, no data needed:

   ```bash
   python scripts/research/planted_sweep.py --bar-only     # the idealised floor
   python scripts/research/forward_power.py                # the deployed contract, measured
   ```

   This is why the null results are **statements about the instrument, not about markets.**

### What this repository does **not** claim

- ❌ **"There is no alpha in public price data."** TSMOM, claim 2, is an alpha found in free public
  OHLCV. The nulls bound what *this apparatus, on this data, at this sample size* could detect.
- ❌ **That any strategy here is profitable, live-ready, or fit to trade.**

---

## Try it in 30 seconds

No data and no API keys. Python 3.11+.

```bash
pip install -e ".[dev]"
```

```bash
python scripts/research/eval_signals.py --batch demo --panel "synthetic:1400,60" --out results/demo
```

This scores four demo signals on a synthetic panel through the full validation funnel
(abridged columns):

```
| # | signal  | verdict | IC-IR  | DSR   | FDR-q | cpcvOOS | fricSh | netSh@std |
|---|---------|---------|--------|-------|-------|---------|--------|-----------|
| 1 | mom_60d | LOGGED  |  0.045 | 0.485 | 0.835 |    0.22 |   0.23 |     -0.59 |
| 2 | mom_20d | LOGGED  | -0.014 | 0.097 | 0.835 |   -0.15 |  -0.31 |     -1.66 |
| 3 | rev_5d  | LOGGED  | -0.043 | 0.018 | 0.835 |   -0.08 |  -0.27 |     -3.10 |
| 4 | vol_20d | LOGGED  | -0.054 | 0.016 | 0.835 |   -0.33 |  -0.22 |     -1.60 |
```

`LOGGED` means fully measured and did not clear the promotion bar. That is where almost every
candidate lands, which is what makes a `PROMISING` worth looking at. The scorecard also prints its
own caveats: the synthetic panel is not survivorship-free, so every figure is an **upper bound**.

[Getting started](docs/guides/getting-started.md) explains every column and walks through a full
Crucible discovery tick.

### Reproduce the headline numbers

The surviving momentum edge, its deflation failure and the value test all re-run from a fresh clone
in about a minute on free data:

```bash
python scripts/research/xsec_momentum_falsification.py   # net Sharpe 0.601, 4/4 classes
python scripts/research/tsmom_excess_return_check.py     # 0.511 in excess of T-bills
python scripts/research/audit_tailwind_book.py           # DSR 0.896 < 0.95, PBO 0.0009
python scripts/research/value_falsification.py           # value factor -0.364, NO-GO
```

Expected output and what to do if yours differs: [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## What is in the box

### Signal validation funnel — `sharpen/signals/`

A signal is a pre-registered, content-hashed spec. Each tier answers a different way a backtest
lies:

| Tier | Test |
|---|---|
| 0 | Causality (truncation tripwire), OHLC hygiene, coverage |
| 1 | Multi-horizon IC, decile spread, breadth, decay half-life |
| 2 | Capturability: net Sharpe per cost model, cost wall, turnover |
| 3 | Subperiod stability and recent out-of-sample |
| 3.5 | Combinatorial purged path resampling with embargo. Signals are fixed rules, nothing is refit, so this measures stability across paths rather than out-of-sample fit |
| 4 | Deflated Sharpe, effective N, FDR/BHY, Harvey-Liu-Zhu hurdle |
| 5 | Orthogonality to a factor book (residual Sharpe) |

Signal libraries: demo, WorldQuant 101, TradingView indicators, and a genetic DSL search
(`sharpen/signals/generation/`) with cohort-level Monte-Carlo null gating.

### Crucible — `sharpen/crucible/`

Automated hypothesis search with validation inside the loop. Free-data connectors (FRED, CFTC COT,
SEC EDGAR, GDELT, Stooq, TWSE, TAIFEX) feed an agent that proposes pre-registered hypotheses
**without access to any prior verdict**. Candidates are mined, deflated against everything the
search has tried, and forward-incubated on bars that postdate the hypothesis. It has produced no
survivor that cleared incubation (claim 4).

```bash
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4 --force
```

### RL training — Protocol v2

A staged protocol where each stage is one tracked run producing one decision artifact:
`data-prep` → `hpo` → `l1-multiseed` → `ensemble-confirm` → `wf` → `oos` → `paper-deploy`.
`scripts/validate_config.py` rejects fused pipelines. See [docs/protocol_v2.md](docs/protocol_v2.md).

```bash
python scripts/validate_config.py --config configs/<cfg>.yaml --stage hpo
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage hpo --agent sac
```

Before spending GPU time: single-asset directional RL did not survive a clean, de-leaked
re-baseline here ([audit](docs/research/gmgp1-btc_deep_lifecycle_audit_2026-06-03.md)). SAC, IQN,
BDQ and PPO are all implemented; none produced a strategy that cleared validation.
The shape that remained worth testing is a validated linear core with RL as an overlay that must
beat it out of sample.

### Execution

Six broker adapters (Bybit perpetuals, ccxt exchanges, DXtrade, Interactive Brokers futures, cTrader,
OANDA), a paper-portfolio executor with a parity harness, and a Docker stack with Prometheus,
Grafana, Telegram alerting, drift detection and a kill file. These ran against paper and demo
accounts only.

---

## Documentation

The hub is [docs/README.md](docs/README.md).

| Document | Covers |
|---|---|
| [NEGATIVE_RESULTS](NEGATIVE_RESULTS.md) | All 76 closed strategies and probes, with the number that closed each |
| [LEAKS_FOUND](docs/LEAKS_FOUND.md) | Three bugs that manufactured results, how each was caught, and the test guarding it |
| [Reproduce](docs/REPRODUCE.md) | Re-run the headline numbers from a fresh clone, with expected output |
| [Methodology](docs/METHODOLOGY.md) | How a result was judged real: pre-registration, leakage rules, deflation, power, controls, audits |
| [Validation archive](docs/research/README.md) | ~100 pre-registrations, audits and verdicts |
| [Getting started](docs/guides/getting-started.md) | Install and two verified first runs |
| [Data](docs/guides/data.md) · [sources and licensing](docs/DATA.md) | **Read before using real prices.** A fresh clone contains no market data |
| [Building a strategy](docs/guides/building-a-strategy.md) | Idea → build → validate, with the gate at each step |
| [Signal research](docs/guides/signal-research.md) | Writing a signal; the validation funnel |
| [Crucible](docs/guides/crucible.md) | The automated search |
| [RL pipeline](docs/guides/rl-pipeline.md) | Protocol v2 in practice |
| [Configuration](docs/guides/configuration.md) | Config and gate schemas |
| [Live trading](docs/guides/live-trading.md) | Brokers, Docker, observability, kill switch |
| [Architecture](docs/guides/architecture.md) · [Testing](docs/guides/testing.md) · [Troubleshooting](docs/guides/troubleshooting.md) | Package map, test suite, known failure modes |

---

## Invariants worth stealing

Each is guarded by a **negative** test (one that must fail if the defect is reintroduced), except
`DATA-CLEAN`, which is a working rule: the cleaner is tested, but nothing forces an experiment to
use it, and the TSMOM scripts read yfinance directly.

| ID | Rule |
|---|---|
| `LEAK-1` | Normalization statistics reset at every train/val/test boundary; never fit across splits |
| `LEAK-2` | No input at bar *t* may carry data stamped after *t*, including coarse-timeframe bars (map to the last **closed** one) and anything a gate or filter reads |
| `BUG-03` | Any hindsight-shaped reward term is zero in backtests |
| `DATA-CLEAN` | All OHLCV passes `scripts/clean_ohlcv.py` before an experiment |
| `CRU-1` | Crucible's gate definitions are hash-frozen; a new capability may not change a past verdict. Correctness fixes that do are MAJOR versions (`crucible-v14.0` moved one) |
| `CRU-2` | Crucible's hypothesis agent can read only dedup keys and killed families, never verdicts |

Gate thresholds are read from `configs/*.gates.yaml`. A few Crucible modules still carry in-code
fallback defaults for when a config block is missing; the frozen gates hash does not cover those.

The project was run semi-agentically: an AI coding agent did most of the implementation under the
rules in [CLAUDE.md](CLAUDE.md). That file ships as it was last used, minus private hostnames,
including the rules that were added after something went wrong. It reads as a live working file
because it was one: its "Active" list and prop-firm milestone describe plans at the time, none
of which reached capital. Session tags such as `S553` throughout the docs refer to the private
R&D log, which is not published.

---

## Project layout

```
sharpen/
├── signals/    Signal specs, alpha DSL, validation funnel
├── crucible/   Automated hypothesis search with validation in the loop
├── agents/     SAC, DSAC, PPO, DeepScalper
├── envs/       Trading environments and wrappers
├── data/       Loaders, feature engineering, splitter
├── paper/      Paper-portfolio executor and parity harness
├── crypto/  cfd/  futures/  live/   Broker adapters and live engine
└── ...         hpo, training, eval, monitoring, analytics, portfolio
configs/  scripts/  tests/  docs/  docker/live/
```

---

## Tests

```bash
python -m pytest
```

On the published tree at release, pytest's own summary read:

```
3261 passed, 25 skipped, 28 deselected, 1 xfailed, 22 warnings in 302.44s (0:05:02)
```

The 28 deselected tests are slow or integration tests excluded by the default `addopts`. They are
not part of that result, and at least one of them can hang rather than fail, so run them
individually with a timeout.

---

## Contributing, security, and a personal note

Challenges to a result are the most useful contribution: see [CONTRIBUTING.md](CONTRIBUTING.md).
Report security issues privately per [SECURITY.md](SECURITY.md). Participation follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

[EPILOGUE.md](EPILOGUE.md) is the author's personal conclusion. It is opinion, labelled as such, and
goes beyond what this repository shows.

---

## Support

If this work saved you from running a test you would otherwise have had to run yourself, you can
support its maintenance through [GitHub Sponsors](https://github.com/sponsors/bigcan). Sponsorship
pays for compute and upkeep of the open research; it buys no access to strategies, signals, or
advice, and nothing here is a claim about future returns.

---

## License

[Apache License 2.0](LICENSE). Copyright 2026 Keng Lee. Attribution for derived third-party code is
in [NOTICE](NOTICE). No market data is distributed; see [docs/DATA.md](docs/DATA.md).
