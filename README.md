# Sharpen

**RL Quant Trading Development Platform**

An institutional-grade research and deployment platform for reinforcement learning trading strategies across multiple asset classes (crypto spot/perp, CFD, futures) and timeframes. The platform takes a strategy from idea to live paper/real trading via a standardized staged pipeline: data prep → HPO → multiseed training → walk-forward + stress → recent-OOS + compliance → paper deploy.

**Ultimate goal:** a diversified portfolio of live-deployed RL strategies — uncorrelated across asset classes and timeframes — each generating sustained risk-adjusted alpha net of fees.
**Short-term milestone:** pass prop-firm challenges (FTMO, Velotrade, The5ers, HyroTrader, FundingPips, MFFU) as proof-of-capital.

> **Canonical project state** lives in `CLAUDE.md` (Project Brief) and `.agent/memory/core.md`. This README is a dev-facing overview — when in doubt, those two are authoritative.

---

## Active Workstreams

| Workstream | Env | Asset(s) | Timeframe | Status |
|------------|-----|----------|-----------|--------|
| **Cross-asset TSMOM** | linear (no RL) | ~18 ETFs / 4 asset classes | daily | **Sole live edge.** Linear time-series momentum, net Sharpe ~0.60, low SPY correlation. Gated at paper (DSR 0.918 < 0.95). |
| **Crucible** (`crucible-v2.8`) | `sharpen/crucible/` + `signals/` | research, multi-asset | — | **Active R&D thrust.** Continuous agentic alpha-mining: free-data connectors (FRED/COT/EDGAR/GDELT/Stooq/TWSE/TAIFEX) → pre-registered hypotheses → deflated funnel → forward lockbox. P0–P5 shipped; 0 survivors cleared lockbox yet. |
| **GMGP1** | V7 ContinuousSwing (SAC) | Gold / XAUUSD / BTC | 15 min | FTMO + Velotrade contender; paper trading live |
| **SG-1** | V7 ContinuousSwing (SAC) | XAUUSD / BTC | 3 min | Intraday diversity strategy; Arm B ablation in progress |

> **Active agent: SAC only.** IQN / BDQ / PPO code is present but none are profitable yet. Other algorithms are researched per gated plans.

**Retired / shelved:** Sync-1H crypto (pilot failure — retired), Funding-Arb (SHELVED, re-run only if funding > 8%/yr), DeepScalper (V5/V6), Market Making (V8, retired S442), AlphaSeek HFT (terminated NO-GO), PRISM ensemble (falsified, `prism.enabled: false`) — all archived with decision docs.

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

`sharpen/crucible/` (`crucible-v2.8`) is a continuous agentic alpha-discovery system built on top of the `sharpen/signals/` DSL + deflated evaluation funnel: free-data connectors (FRED, CFTC COT, SEC EDGAR, GDELT, Stooq, TWSE, TAIFEX) feed an agent that proposes pre-registered hypotheses (blind to verdicts), which are mined, deflated, and forward-incubated in a lockbox before any human Tier-2 audit. Full architecture: `docs/claude_md_reference.md`.

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
├── crucible/      # Crucible agentic alpha-mining platform (crucible-v2.8)
└── analytics/     # Pyfolio, WandB evaluator, gate evaluation
configs/  scripts/  tests/  docs/  docker/live/
```

**Boundary:** Only modify `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

---

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries |
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

The live stack runs SAC agents in Docker containers with full observability on a remote desktop (`<TAILSCALE_HOST>` via Tailscale).

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

Cloud tier: agent-memory MCP (LanceDB on GCS) as a search index; flat files remain authoritative. End-of-session `/sync` writes to memory → `Skill-Evolve` staleness check → git commit.

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Risk Disclaimer

This software is for educational and research purposes only. **Reinforcement learning trading involves significant financial risk.** The authors are not responsible for trading losses.
