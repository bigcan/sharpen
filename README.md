# Sharpen

**A negative-results archive for systematic trading research.**

Sharpen is a staged research platform for RL and linear trading strategies (crypto spot/perp, CFD, futures, ETFs) — data prep → HPO → multiseed training → walk-forward + stress → recent-OOS + compliance → paper deploy. It was run in earnest for roughly a year.

**It did not produce a deployable strategy.** This repository is published for the negative results and the falsification machinery that produced them, not as a strategy release. Read the section below before drawing any conclusion from anything here.

---

## What this repo does and does not claim

**Claimed, and reproducible from this repo:**

1. **~20 strategy families were specified, built, measured and falsified.** Every NO-GO carries a decision doc. **Zero strategies were ever deployed to capital.**
2. **One weak survivor: cross-asset TSMOM.** Time-series momentum on 18 ETFs across 4 asset classes, from **free daily public OHLCV** (`yfinance`), **net Sharpe ≈ 0.60 @ 2 bps**, low SPY correlation, PBO ≈ 0.0009 (i.e. not overfit). **It still failed this project's own deployment gates** — Deflated Sharpe 0.896 < 0.95 — and a Tier-2 audit returned BLOCK. It is a real but sub-threshold edge, not a product.
3. **The automated alpha miner (Crucible) returned zero discoveries — and the record cannot tell you why.** An independent audit found the mining record is statistically indistinguishable from noise. Re-derived here: a search at the same budget, run on data with **no signal at all**, reaches a higher gate statistic (`marginal_t` **3.68**) than anything the real record ever produced (**2.12**).
4. **A power wall bounds what free daily data can settle.** An idealised single pre-registered test needs an annualised ΔSharpe of **1.42** (4 years of daily bars) or **0.71** (16 years) to detect an edge at 80% power. Realistic single-signal edges are **0.3–0.5**. Verify in one second, no data required:
   ```bash
   python scripts/research/planted_sweep.py --bar-only
   ```

**Explicitly NOT claimed:**

- ❌ **"There is no alpha in public OHLCV data."** This repo does not support that, and contradicts it: TSMOM (claim 2) *is* an alpha found in free public OHLCV. Our null results bound what **this instrument, on this data, at this sample size** could detect — nothing more.
- ❌ **That any strategy here is profitable, live-ready, or fit to trade.** Nothing was deployed to capital. The one survivor is gated.

The honest summary: **negative results about a search apparatus, not about markets.** The reproduction scripts and their tripwires are in `scripts/research/` and `tests/research/`; the reasoning is in `docs/research/`.

---

## Status

| Workstream | Env | Asset(s) | Timeframe | Status |
|------------|-----|----------|-----------|--------|
| **Cross-asset TSMOM** | linear (no RL) | 18 ETFs / 4 asset classes | daily | **Only validated edge.** Net Sharpe ≈ 0.60, low SPY correlation. **Capital BLOCKED** — DSR 0.896 < 0.95, Tier-2 verdict BLOCK. |
| **Crucible** (`crucible-v13.1`) | `sharpen/crucible/` + `signals/` | research, multi-asset | — | Agentic alpha-mining: free-data connectors (FRED/COT/EDGAR/GDELT/Stooq/TWSE/TAIFEX) → pre-registered hypotheses → deflated funnel → forward lockbox. **0 candidates have ever cleared the lockbox.** |
| **GMGP1 / SG-1** | V7 ContinuousSwing (SAC) | Gold / XAUUSD / BTC | 15 min / 3 min | **Falsified.** Single-asset directional RL did not survive a de-leaked re-baseline. Dormant. |

> **Single-asset directional RL is falsified in this repo.** SAC is the only agent that was ever competitive; IQN / BDQ / PPO code is present and none of it was profitable.

**Retired / shelved:** Sync-1H crypto, Funding-Arb (re-run only if funding > 8%/yr), DeepScalper (V5/V6), Market Making (V8), AlphaSeek HFT, PRISM ensemble (falsified, `prism.enabled: false`) — all archived with decision docs.

> Canonical internal state lives in `CLAUDE.md` and `.agent/memory/core.md`. Where this README and those disagree, those are authoritative.

---

## Stack

Python 3.11+ · PyTorch 2.8+ · Gymnasium · Optuna · Weights & Biases · Parquet · Prometheus · Grafana · Docker · Ruff · Mypy · Pytest

---

## Installation

Requires **Python 3.11+** and a CUDA-capable GPU (tested on RTX 4090 / 5090).

```bash
git clone https://github.com/Chiwin-Technology/sharpen.git
cd sharpen
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

---

## Quick Start

All training work follows the staged **Training Protocol v2** (`docs/protocol_v2.md`). Each stage is one WandB run producing one decision artifact — bare unstaged pipelines are rejected by `scripts/validate_config.py`.

### Run a pipeline stage

```bash
# Validate config for a stage (required before launch)
python scripts/validate_config.py --config configs/<cfg>.yaml --stage hpo

# Staged run (hpo | l1-multiseed | walk-forward | recent-oos | paper-deploy)
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage hpo
```

Reference configs per pipeline:

| Pipeline | Reference Config |
|----------|-----------------|
| GMGP1 (V7) | `configs/gmgp1_sac_gc_15min.yaml` |
| Sync-1H | `configs/synapse_crypto_1h_v2.yaml` |
| Funding-Arb | `configs/funding_arb_sac_10assets_hpo.yaml` |
| Live Trading | `configs/live_gmgp1_btc_bybit.yaml` |

### Deploy to remote GPU

```bash
python scripts/deploy_bare_metal.py \
    --config configs/<cfg>.yaml \
    --instance <gpuhub-instance> \
    --gpu <id> --collect
```

### Monitor & collect

```bash
python scripts/monitor_fleet.py                 # fleet-wide GPU + run health
python scripts/monitor_run.py --run_id <ID>     # single run
python scripts/collect_run.py --run_id <ID>     # pull metrics + checkpoint + report
python scripts/auto_collect_checkpoints.py      # polls WandB, SFTPs from GPUHub
```

---

## Alpha-Mining Platform (Crucible)

`sharpen/crucible/` (`crucible-v13.1`) is a continuous agentic alpha-discovery system built on top of the `sharpen/signals/` DSL + deflated evaluation funnel: free-data connectors (FRED, CFTC COT, SEC EDGAR, GDELT, Stooq, TWSE, TAIFEX) feed an agent that proposes pre-registered hypotheses (blind to verdicts), which are mined, deflated, and forward-incubated in a lockbox before any human Tier-2 audit. Full architecture: `docs/claude_md_reference.md`.

**It has never produced a discovery.** That is the interesting part, and `docs/research/` documents why the record cannot distinguish "nothing was there" from "this instrument could not have seen it". Two reproductions ship with the repo:

```bash
# Does the gate reject a planted PERFECT-foresight oracle? Where does its bar actually sit?
python scripts/research/planted_sweep.py --bar-only          # the power arithmetic, instantly
python scripts/research/planted_sweep.py                     # synthetic ladder + both controls

# What does the same search reach on data with NO signal in it at all?
python scripts/research/null_grid_sim.py --mode both
```

Every reproduction carries a **positive and a negative control** and refuses to print a verdict unless both behave — a harness that cannot also produce a correct null cannot be trusted to report power. That guard caught three real bugs during the write-up; see `docs/research/f4_null_grid_reproduction_2026-09-03.md`.

```bash
# Continuous nightly-tick discovery loop
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4
python scripts/research/crucible_orchestrator.py --mode real --start 2008-01-01 --nights 4

# Manual single-cycle loop / governance handoff / reproduce a past run
python scripts/research/crucible_hypothesis_loop.py --mode synthetic
python scripts/research/crucible_governance.py --lockbox <path> --gov <path> --out <dir> --cards <dir> --workstream crucible --scope overlay --now-ts <ISO8601>
python scripts/research/crucible_reproduce.py results/crucible_orchestrator/<mode>/<tick_ts>
```

**Prediction-market research (Polymarket)** has moved to its own repo: [`Chiwin-Technology/polymarket-updown-research`](https://github.com/Chiwin-Technology/polymarket-updown-research) (spun off 2026-07-05 — it was always self-contained, zero `sharpen` imports).

---

## Project Layout

```
sharpen/
├── agents/        # SAC, IQN, BDQ, PPO, DSAC implementations
├── envs/          # V7 ContinuousSwing, legacy V5/V6
├── crypto/        # CryptoPerp, FundingArb envs + execution (Bybit, Binance)
├── futures/       # IB futures (GC, MGC) execution
├── cfd/           # cTrader CFD execution (XAUUSD)
├── data/          # Loaders, feature engineering, splitter, Parquet handler
├── training/      # Trainers, HPO runners, accumulators
├── signals/       # Alpha-mining DSL + deflated (T0-T5) evaluation funnel
├── crucible/      # Crucible agentic alpha-mining platform (crucible-v13.1)
└── analytics/     # Pyfolio, WandB evaluator, gate evaluation
configs/  scripts/  tests/  docs/  docker/live/
```

**Boundary:** Only modify `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

---

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries |
| LEAK-2 | **Temporal causality.** No feature at bar `t` may use data stamped `> t` — multi-scale coarse-bar maps must read the last **CLOSED** coarse bar. Each guarded by a *negative* test that fails if look-ahead returns. This is the invariant that cost the most here: a single coarse-bar leak survived hundreds of green diff-scoped audits and, once fixed, erased a strategy's entire apparent edge. |
| BUG-01 | HPO objective = `profit_factor`; lock reward params during HPO |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting |
| BUG-04 | Dense reward on switch bars must use direction BEFORE switch |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt` |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20×); `1.0` = starvation |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`; >30% divergence = halt |

See `CLAUDE.md` for the full rule set and coding standards (Tensor Core alignment, `non_blocking=True`, no per-sample PER loops, etc.).

---

## Docker Live Trading & Monitoring

The live stack runs SAC agents in Docker containers with full observability on a self-hosted machine reached over a private network. **It is dormant** — no strategy is currently trading, on paper or otherwise. Retained because the observability wiring is the reusable part.

```bash
# Wrapper (preferred)
./scripts/manage_strategies.sh build <target>
./scripts/manage_strategies.sh up    <target>
./scripts/manage_strategies.sh logs  <target>
./scripts/manage_strategies.sh ps
```

| Service | Port | Purpose |
|---------|------|---------|
| IB Gateway | 4002 (paper) | Headless IB Gateway (IBC + Xvfb) for GC/MGC futures |
| Prometheus | 9090 | 15 s scrape of per-strategy metrics (9101–9107) |
| Grafana | 3000 | "FinRL Trading Overview" dashboard, alerting |
| Watchdog | — | Docker health event listener → Telegram alerts |
| Portainer | 9443 | Container management web UI |

Per-strategy metrics include portfolio value, position, drawdown, daily P&L, broker connection state, and trading-aware health JSON (not just process existence). Telegram alerts suppressed for TradFi strategies during market closure (Fri 21Z → Sun 22Z UTC).

---

## Agent Memory System

The repo includes a tiered persistent memory system that gives the AI agent continuity across chat sessions.

```
.agent/memory/core.md    # Project ground truth — status, decisions, runs (git-tracked)
randd_log.md             # Append-only R&D write buffer (rotates at 150 KB)
randd_archive/YYYY-MM.md # Rotated archives
```

Search tier: an `agent-memory` MCP server (self-hosted LanceDB + a local embedding model) indexes both the session summaries and the flat memory files; **the flat files remain authoritative**. End-of-session `/sync` writes memory → staleness check → git commit, and is gated on a size check so the boot payload cannot silently outgrow its budget.

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Risk Disclaimer

This software is for educational and research purposes only, and is published as a **record of what did not work**.

**Nothing in this repository is a trading recommendation, and nothing here was ever deployed to capital.** Every strategy in it was either falsified or blocked by its own gates — including the one surviving edge, which failed its deflated-Sharpe threshold. Backtested and paper results do not imply live performance; the repo's own history is largely a catalogue of results that looked real until a leak, a cost model, or a multiplicity correction removed them.

**Trading involves significant financial risk.** The authors accept no responsibility for losses. If you reuse anything here, reuse the falsification machinery, not the strategies.
