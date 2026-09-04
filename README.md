# Sharpen

**An end-to-end, all-in-one strategy development platform with built-in alpha mining.**

Mine alphas, build strategies, validate them against a backtest engine that won't lie to
you, size them into a portfolio, and run them on live brokers — one platform, one data
layer, one set of gates, from raw hypothesis to funded position.

```
ALPHA MINING  ──►  STRATEGY BUILD  ──►  BACKTEST VALIDATION  ──►  PORTFOLIO  ──►  PAPER  ──►  LIVE
free-data          signal · DSL ·        T0–T5 funnel ·             sleeves ·      fill      6 broker
connectors ·       RL policy ·           walk-forward ·             vol target ·   engine ·  adapters ·
agentic search     ensembles             CPCV · deflation           combiners      parity    Docker + alerts
```

**The validation engine is the reason to use it.** Most backtesters answer "did this make
money on this data?" Sharpen answers the question that actually predicts live performance:
**is this result distinguishable from luck, after costs, out of sample, given everything
else you tried?** That means deflated Sharpe, multiplicity correction across your whole
search, combinatorial purged cross-validation, an explicit cost wall, orthogonality to known
factors, walk-forward with stress replay, and forward incubation on unseen bars.

Nothing is bolted on. The alpha miner writes into the same funnel your hand-built signal
goes through; the portfolio you certify is wired to the executor that trades it; the
strategies that don't make it are recorded with their evidence in the
[validation archive](docs/research/README.md), so you don't rebuild them next quarter.

---

## What this repo does and does not claim

This is the platform's own track record, stated plainly. It is a tooling release, not a
strategy release, and **nothing here was ever deployed to capital.**

**Claimed, and reproducible from this repo:**

1. **~20 strategy families were specified, built, measured and falsified.** Each carries its
   decision doc in the [validation archive](docs/research/README.md).
2. **One weak survivor: cross-asset TSMOM.** Time-series momentum on 18 ETFs across 4 asset
   classes, from **free daily public OHLCV**, **net Sharpe ≈ 0.60 @ 2 bps**, low SPY
   correlation, PBO ≈ 0.0009 (i.e. not overfit). **It still failed this project's own
   deployment gates** — Deflated Sharpe 0.896 < 0.95 — and a Tier-2 audit returned BLOCK. A
   real but sub-threshold edge, not a product.
3. **The alpha miner has never produced a discovery, and the record cannot say why.** A search
   at the same budget, run on data with **no signal in it at all**, reaches a *higher* gate
   statistic (`marginal_t` **3.68**) than anything the real mining record ever produced
   (**2.12**).
4. **A power wall bounds what free daily data can settle.** An idealised single pre-registered
   test needs an annualised ΔSharpe of **1.42** (4 years of daily bars) or **0.71** (16 years)
   to detect an edge at 80% power — against realistic single-signal edges of **0.3–0.5**.
   Verify in one second, no data required:
   ```bash
   python scripts/research/planted_sweep.py --bar-only
   ```

**Explicitly NOT claimed:**

- ❌ **"There is no alpha in public OHLCV price data."** This repo does not support that and in
  fact contradicts it — TSMOM (claim 2) *is* an alpha found in free public OHLCV. The null
  results here bound what **this instrument, on this data, at this sample size** could have
  detected. That is a statement about the search apparatus, not about markets.
- ❌ **That any strategy here is profitable, live-ready, or fit to trade.** The one survivor is
  gated; everything else was falsified.

The reproductions and their tripwires live in `scripts/research/` and `tests/research/`; the
reasoning is in `docs/research/`.

---

## Try it in 30 seconds — no data, no API keys

```bash
pip install -e ".[dev]"
```

```bash
python scripts/research/eval_signals.py --batch demo --panel "synthetic:1400,60" --out results/demo
```

```
| # | signal   | family    | verdict | IC-IR  | DSR   | FDR-q | cpcvOOS | fricSh | netSh@std |
|---|----------|-----------|---------|--------|-------|-------|---------|--------|-----------|
| 1 | mom_60d  | technical | LOGGED  |  0.045 | 0.474 | 0.835 |    0.22 |   0.23 |     -0.59 |
| 2 | mom_20d  | technical | LOGGED  | -0.014 | 0.013 | 0.835 |   -0.15 |  -0.31 |     -1.66 |
```

You just built four strategies and validated all of them. `LOGGED` means "fully measured,
did not clear the promotion bar" — most candidates land here, which is what makes a
`PROMISING` worth acting on.

**[→ Getting started](docs/guides/getting-started.md)** walks through this plus a full
Crucible discovery tick, and explains what every column means.
**[→ Building a strategy](docs/guides/building-a-strategy.md)** is the end-to-end workflow:
idea → build → validate → paper → live.

---

## Documentation

**[docs/README.md](docs/README.md) is the documentation hub.**

| Guide | Covers |
|---|---|
| [Getting started](docs/guides/getting-started.md) | Install and two verified first runs |
| [Building a strategy](docs/guides/building-a-strategy.md) | **The end-to-end workflow** — idea to live, with the validation gate at each step |
| [Data](docs/guides/data.md) | **Read before using real prices.** `/data/` is gitignored — a fresh clone has none |
| [Signal research](docs/guides/signal-research.md) | Writing a signal; the T0–T5 validation funnel |
| [Crucible](docs/guides/crucible.md) | Automated strategy search that validates as it goes |
| [RL pipeline](docs/guides/rl-pipeline.md) | Training Protocol v2 in practice |
| [Configuration](docs/guides/configuration.md) | Config and gate schemas |
| [Live trading](docs/guides/live-trading.md) | Brokers, Docker, observability, kill switch |
| [Architecture](docs/guides/architecture.md) | Package map and data flow |
| [Testing](docs/guides/testing.md) | Suite, negative tests, contributing |
| [Troubleshooting](docs/guides/troubleshooting.md) | Observed failure modes |
| [Validation archive](docs/research/README.md) | ~100 preregistrations, audits and verdicts — what has already been tested |

---

## What is in the box

### Strategy construction + validation — `sharpen/signals/`

Build a signal as a pre-registered, content-hashed spec; get back a validated verdict with
the full evidence attached. Seven tiers, each answering a different way a backtest lies:

| Tier | Test |
|---|---|
| 0 | Causality (truncation tripwire) + OHLC hygiene + coverage |
| 1 | Multi-horizon IC, decile spread, breadth, decay half-life |
| 2 | Capturability — net Sharpe per cost model, cost wall, turnover |
| 3 | Subperiod stability + recent OOS |
| 3.5 | Combinatorial purged CV with embargo |
| 4 | Deflated Sharpe, effective N, FDR/BHY, HLZ hurdle |
| 5 | Orthogonality to a factor book — residual Sharpe |

Signal libraries: demo, WorldQuant 101, TradingView indicators, plus a genetic DSL search
(`sharpen/signals/generation/`) with cohort-level Monte-Carlo null gating.

### Crucible — `sharpen/crucible/` (`crucible-v13.1`)

Automated strategy generation with validation built into the loop. Free-data connectors
(FRED, CFTC COT, SEC EDGAR, GDELT, Stooq, TWSE, TAIFEX) feed an agent that proposes
pre-registered hypotheses **blind to all prior verdicts**, which are mined, deflated, and
forward-incubated in a lockbox on bars that postdate the hypothesis. It searches at a scale
no human can, and the multiplicity accounting charges every candidate it tries — so a
survivor is a survivor of the search, not of one lucky draw.

```bash
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4 --force
```

### RL training — Protocol v2

Six stages, each one WandB run producing one decision artifact: `data-prep` → `hpo` →
`l1-multiseed` → `ensemble-confirm` → `wf` → `oos` → `paper-deploy`. Fused pipelines are
rejected by `scripts/validate_config.py`.

```bash
python scripts/validate_config.py --config configs/<cfg>.yaml --stage hpo
```

```bash
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage hpo --agent sac
```

**Use SAC.** IQN, BDQ and PPO are implemented but have not cleared validation here, and
single-asset directional RL did not survive a clean de-leaked re-baseline — see the
[GMGP1-BTC audit](docs/research/gmgp1-btc_deep_lifecycle_audit_2026-06-03.md) before
investing GPU time in that shape. The productive pattern is a validated linear core with RL
as an overlay behind a beat-the-baseline gate.

### Live and paper trading

Six broker adapters — Bybit perps, ccxt exchanges, DXtrade, Interactive Brokers futures,
cTrader, OANDA — behind a Docker stack with Prometheus, Grafana, Telegram alerting, drift
detection and a kill file.

```bash
./scripts/manage_strategies.sh ps
```

---

## What the validation engine has found so far

Useful for calibrating expectations, and for not rebuilding work that is already done.

- **Validated and wired:** a cross-asset time-series momentum sleeve (~18 ETFs, 4 asset
  classes) at a net Sharpe near **0.60** with low SPY correlation. Running on the paper
  executor; held short of capital pending a deflated-Sharpe threshold.
- **Tested and closed:** roughly twenty strategy families, each with the preregistration
  that preceded it and the evaluation that closed it — options/VRP, several arbitrage
  mechanisms, funding-rate arb standalone, liquid large-cap cross-section, intraday retail
  patterns, market making. Full index in the
  [validation archive](docs/research/README.md).

That ratio is what a correctly calibrated validation engine produces. A backtester that
promotes most of what you feed it is not being generous — it is not measuring multiplicity.

---

## Project layout

```
sharpen/
├── signals/    Signal construction + alpha DSL + T0–T5 validation funnel
├── crucible/   Automated strategy search with validation in the loop
├── crypto/     Crypto envs, execution, live engine
├── agents/     SAC, DSAC, PPO, DeepScalper
├── data/       Loaders, feature engineering, splitter
├── envs/       V7 ContinuousSwing, wrappers, legacy V5/V6
├── cfd/        cTrader + OANDA execution
├── futures/    Interactive Brokers execution
├── paper/      Paper portfolio executor + parity harness
├── live/       Agent loading, ensembles, challenge state machine
└── ...         hpo, training, eval, monitoring, analytics, prop, portfolio
configs/  scripts/  tests/  docs/  docker/live/
```

**Boundary:** modify only `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`.

---

## Critical invariants

| ID | Rule |
|---|---|
| `LEAK-1` | Reset EMA-Z normalization at train/val/test boundaries. Never normalize across splits |
| `LEAK-2` | No feature at bar *t* may read data stamped `> t`. Each guard has a **negative** test |
| `BUG-01` | HPO objective is `profit_factor`; lock reward parameters during HPO |
| `BUG-03` | `hindsight_weight` must be `0.0` in backtests — it reads future prices |
| `BUG-04` | Dense reward on a switch bar uses the direction **before** the switch |
| `SHORT-ACCT` | Shorts must not accumulate `notional_debt` |
| `MARGIN-CFG` | BTC `margin_requirement: 0.05` (20×); `1.0` starves the agent |
| `DATA-CLEAN` | All OHLCV passes `scripts/clean_ohlcv.py` before experiments |
| `PF-XCHECK` | Cross-check PF via `mid_price` **and** `close`; >30% divergence halts |
| `CRU-1` | Crucible's `gates_hash` is frozen; a new capability must not change a past verdict |
| `CRU-2` | Crucible's agentic code reads only `ledger_agent_view` — never verdicts. The anti-oracle moat |

Full rule set and coding standards: [CLAUDE.md](CLAUDE.md).

---

## Requirements

Python 3.11+ · PyTorch 2.8+ · Gymnasium · Optuna · Weights & Biases · Parquet ·
Prometheus · Grafana · Docker · Ruff · Mypy · Pytest

A CUDA GPU (tested on RTX 4090 / 5090) is needed for RL training only. The signal and
Crucible research stacks are CPU-only.

```bash
python -m pytest
```

---

## Risk disclaimer

For research and educational purposes only. **Trading involves significant financial risk**
and reinforcement learning does not reduce it. Nothing here is investment advice, nothing
here is cleared for capital, and the authors are not responsible for trading losses.

## License

See [LICENSE](LICENSE).
