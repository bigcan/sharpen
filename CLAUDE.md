# CLAUDE.md

## Project Brief

**Goal:** Profitable RL quant trading across asset classes.
Active agent: **SAC only** (IQN/BDQ/PPO all falsified). Workstreams: GMGP1 SAC Gold 15m, Sync-1H crypto, Funding-Arb, Market Making LOB.
Detailed state: `.agent/memory/core.md` (loaded at boot).

## Stack

Python 3.11+ · PyTorch 2.8+ · Gymnasium · Optuna · WandB · Parquet · Prometheus · Grafana · Docker · Ruff · Mypy · Pytest

## Commands

```bash
# Setup
pip install -e .[dev]

# Pipeline: HPO -> Train -> Backtest
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml
# Flags: --agent sac --trials N --steps N --backtest_only --checkpoint PATH

# Crypto pipelines
python scripts/crypto_hpo_runner.py --config <cfg> [--warm_start --max_windows N]
python scripts/funding_arb_hpo_runner.py --config <cfg>

# Deploy to remote GPU
python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> [--no_kill] --collect

# Quality
ruff check finrl_pro_ds && mypy finrl_pro_ds --ignore-missing-imports && pytest

# Monitoring (scripts)
python scripts/monitor_fleet.py              # Fleet-wide status (SSH + WandB)
python scripts/monitor_run.py --run_id <ID>  # Single run
python scripts/collect_run.py --run_id <ID>  # or --batch

# Docker Live Trading (docker/live/)
# Manage via: ./scripts/manage_strategies.sh {build|up|ps|logs} <target>
docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile all up -d                        # Start everything
# Profiles: ib, crypto, ctrader, monitoring, prism, all
# Monitoring stack: Prometheus (:9090), Grafana (:3000), Watchdog (Telegram alerts)
docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile monitoring up -d                 # Start monitoring only
docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.prism.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile prism up -d                      # Start PRISM stack only
docker compose -f docker/live/docker-compose.yaml \
               --profile monitoring build                 # Rebuild after config changes

# Checkpoint collection (auto-secure to local, synced to GCS)
python scripts/auto_collect_checkpoints.py                    # WandB-based, last 24h
python scripts/auto_collect_checkpoints.py --hours 48         # Longer lookback
python scripts/auto_collect_checkpoints.py --run_id <ID>      # Specific run
python scripts/auto_collect_checkpoints.py --all_instances    # Scan all GPUHub instances
# Flags: --dry_run (preview), --include_all (include HPO trial timestamp dirs)

# WandB: entity=bigcan-chiwin-technology, project=FinRL-Pro-DS
# Helpers: .agents/skills/wandb-primary/scripts/wandb_helpers.py (see WandB skill)
# ALWAYS pass metric_keys= explicitly -- FinRL metrics, NOT ML defaults:
#   HPO: "_debug/eval_profit_factor", "_research/sharpe_minute"
#   Backtest: "Profit_Factor_Daily", "Sharpe_Ratio", "Sortino_Ratio", "Total_Return", "Max_Drawdown"
```

## Project Map

```
finrl_pro_ds/
  agents/
    sac/sac_agent.py              # SAC -- continuous position control (ACTIVE)
    sac/networks.py               # SAC actor/critic networks
    deepscalper/                  # Legacy (IQN/BDQ falsified) -- do not extend
    ppo_scalper/                  # Legacy (PPO falsified) -- do not extend
  envs/
    continuous_swing_env.py       # V7 -- Box(-1,1) continuous MDP (GMGP1 SAC)
    market_making_env.py          # V8 -- Box(-1,1,3) MM with fill simulation
    swing_scalper_env.py          # V6 -- Discrete(2) legacy (no active runs)
    deep_scalper_env.py           # V5 -- Discrete(6) legacy (no active runs)
    augmented_wrapper.py          # Obs augmentation wrapper
  crypto/
    envs/crypto_perp_env.py      # Sync-1H: multi-asset perp futures
    envs/funding_arb_env.py      # Delta-neutral funding arb
    envs/multi_exchange_arb_env.py # Cross-exchange arb
    data/                         # crypto_loader, crypto_collector, crypto_array_builder
    features/                     # crypto_features, funding_arb_features
    features/prism_features.py    # PRISM feature integration (Chronos-2 + GAHMM)
    execution/                    # exchange_perp_broker, bybit_perp_broker, arbitrator
    live/                         # live_engine, live_obs_builder, bar_clock, metrics
    live/prism_overlay.py         # PRISM L2 position sizing overlay (regime-based)
    mlops/                        # crypto_risk_manager
  futures/
    execution/                    # ib_futures_broker, contract_manager
    live/                         # cme_bar_clock, cme_calendar
    data/                         # ib_data_loader
  cfd/
    execution/                    # ctrader_broker
    live/                         # cfd_bar_clock
    data/                         # ctrader_data_loader
  data/
    multiscale_handler.py         # Multi-scale OHLCV handler (SAC / GMGP1)
    lob_data_handler.py           # LOB microstructure handler (MM)
    mm_data_handler.py            # MM-specific data handler
    fill_model.py                 # L1/L2 fill simulation + adverse selection
    feature_engineering.py        # Feature Factory -- Micro (LOB) + Macro (OHLCV)
    parquet_handler.py            # Shared-memory data streaming
  training/sac_trainer.py         # SAC training loop (+ legacy deepscalper/ppo trainers)
  analytics/                      # pyfolio_analyzer, wandb_evaluator
scripts/        # Pipeline, deployment, monitoring, checkpoint collection, oracles, ETL
configs/        # YAML experiment configs
tests/          # pytest suite
.agent/skills/  # Agent skills (audit, memory, deploy, monitor, etc.)
docker/live/    # Docker Compose live trading orchestration
  docker-compose.yaml           # Multi-strategy orchestrator (6 strategies + infra)
  docker-compose.desktop.yaml   # Desktop overlay (port bindings for Win11 dev)
  docker-compose.prism.yaml     # PRISM overlay (prism-db, prism-api, prism-refit-worker)
  Dockerfile.live-engine        # Shared image for all strategies (PyTorch CPU + prometheus_client)
  Dockerfile.prism-db           # PostgreSQL 15 with baked schema for PRISM
  Dockerfile.watchdog           # Health watchdog (docker-py + Telegram alerts)
  Dockerfile.prometheus         # Prometheus with baked-in scrape config
  Dockerfile.grafana            # Grafana with baked-in provisioning + dashboards
  healthcheck.sh                # Trading-aware healthcheck (JSON state, not just pgrep)
  prism/schema.sql              # PRISM DB schema (6 tables: HMM models, predictions, market data)
  prism_sdk/                    # PRISM Python SDK (PRISMClient, types, vendored in Docker)
  prometheus/prometheus.yml     # Scrape config for strategy metrics endpoints
  grafana/                      # Provisioning (datasources, dashboards) + dashboard JSON
  .env.example                  # Template for credentials and config
```

**Boundary:** Only modify `finrl_pro_ds/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

## Env Contracts

### ContinuousSwingEnv (V7) -- `continuous_swing_env.py`
| Property | Spec |
|----------|------|
| Action | `Box(-1, 1, shape=(1,))` -- target position fraction. -1=full short, 0=flat, +1=full long. |
| Obs | `Dict{ scale_0: (W, F), scale_1: (W, F), ..., private: (5,) }` -- one key per OHLCV scale. |
| Private | `[current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]` -- 5 dims |
| Reward | DSR (default). Deadband 0.25. Fee curriculum via `fee_schedule`. |
| Done | Truncated at `episode_length`. Terminated on `max_drawdown_pct` breach. |

### MarketMakingEnv (V8) -- `market_making_env.py`
| Property | Spec |
|----------|------|
| Action | `Box(-1, 1, shape=(3,))` -- (spread_offset, inventory_skew, quote_intensity). |
| Obs | `Dict{ scale_0: (W,F), ..., lob: (W,n_lob), private: (12,) }` -- optional LOB encoder. |
| Private | 12 dims: inventory, fills, quote state, adverse selection metrics. |
| Reward | DSR on spread_capture + MtM - inventory_penalty - fees. Fill model: L1/L2. |
| Done | Truncation + drawdown + inventory hard stop. |

### CryptoPerpEnv -- `crypto/envs/crypto_perp_env.py`
| Property | Spec |
|----------|------|
| Action | `Box(-1, 1, shape=(n_assets,))` -- signed position weights. `long_only` mode available. |
| Obs | Flat 1D (~962 dims for 20 assets): margin + tech + positions + unrealized + funding + cost + ENB. |
| Reward | Sortino (default). Funding at UTC 00/08/16. Circuit breaker on portfolio value. |
| Done | Truncation at episode end. Termination on circuit breaker or liquidation. |

### FundingArbEnv -- `crypto/envs/funding_arb_env.py`
| Property | Spec |
|----------|------|
| Action | `Box(-1, 1, shape=(n_assets,))` -- arb weight (>0 standard, <0 reverse). |
| Obs | Flat 1D (~326 dims for 20 assets): portfolio + tech + weights + basis + funding + cost + delta + margin. |
| Reward | PV-return + delta penalty + turnover penalty. Funding at UTC 00/08/16. Deadband 0.01. |
| Done | Truncation. Circuit breaker on portfolio value. |

### Legacy Envs (no active runs)
- **V6 SwingScalperEnv** (`swing_scalper_env.py`): `Discrete(2)`, binary swing, private: 4 dims, cooldown_bars.
- **V5 DeepScalperEnv** (`deep_scalper_env.py`): `Discrete(6)` flat. Do NOT revert to MultiDiscrete.

**All envs return raw numpy dicts, NOT Gymnasium wrappers -- preserve this path.**

## Config Schema

Configs vary by pipeline. Do NOT invent keys -- read a reference config first.

| Pipeline | Reference Config | Top-Level Sections |
|----------|-----------------|---------------------|
| GMGP1 (V7) | `configs/gmgp1_sac_gc_15min.yaml` | data / features / env / network / agents.sac / training / hpo / wandb |
| Sync-1H | `configs/synapse_crypto_1h_v2.yaml` | strategy / universe / environment / agents / arbitrator / walk_forward / risk / execution |
| Funding Arb | `configs/funding_arb_sac_10assets_hpo.yaml` | strategy / universe / environment / agents / walk_forward |
| Market Making | `configs/mm_sac_btc_lob_10s.yaml` | data / features / env (fill_model, LOB) / network (lob_encoder, action_dim=3) / agents.sac / training / hpo / wandb |
| Live Trading | `configs/live_gmgp1_btc_bybit.yaml` | exchange / agent / agents.sac / network / features / bar_clock / trading / risk / wandb / safety / prism (+ contract for IB/cTrader) |

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries. Never normalize across splits. |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params during HPO. |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting (uses future prices). |
| BUG-04 | Dense reward on switch bars must use direction BEFORE switch. Save `direction_for_reward` before action processing. |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |

## Coding Standards

- All `.to(device)` calls **must** use `non_blocking=True`
- Replay buffers **must** support batch push -- no per-sample Python loops in hot paths
- New configs: `torch_compile: true`, `update_interval: 8` (tau auto-scales)
- Do NOT rewrite vectorized PER with per-element iteration
- Env returns raw numpy, not dicts -- preserve this path
- All `Linear` layer hidden dims must be **multiples of 8** (Tensor Core alignment)
- Use `logging` or `MLOpsLogger` -- never raw `print()` in production code

## Anti-Patterns (NEVER DO)

- **Never import from** `FinRLPodracer/` or `Podracer/`
- **Never normalize** across train/val/test splits (LEAK-1)
- **Never set** `hindsight_weight > 0` in backtest configs (BUG-03)
- **Never guess** config keys -- read a reference YAML in `configs/` first
- **Never use** `mid_price` without validating high/low against open/close (data corruption risk)
- **Never deploy** without running `monitor_fleet.py` first (check VRAM, active processes)
- **Never skip** Math skill verification on formula/equation changes in env/agent/reward code
- **Never claim** a file, function, class, config key, or CLI flag exists (or doesn't) without first verifying via Grep/Glob/Read. "I believe X exists" is not acceptable -- look it up.

## Agent Skills -- Auto-Dispatch

Skills are split: **user-scope** (`~/.claude/skills/`) for reusable methodology, **project-scope** (`.agent/skills/`) for project-specific logic. Read the relevant `SKILL.md` before executing. **Trigger proactively** -- don't wait for the user to ask.

| Skill | Trigger | Scope | Spec |
|-------|---------|-------|------|
| **Audit** | **Auto** after ANY code change to `finrl_pro_ds/`, `scripts/`, `configs/`. Skip `.md`-only. **Also auto after plan/feature/task implementation.** | User + Project | `~/.claude/skills/audit/SKILL.md` + `.agent/skills/audit/SKILL.md` (project addendum) |
| **Deploy** | User requests GPU launch, instance management, or run deployment. | Project | `.agent/skills/deploy/SKILL.md` |
| **Memory** | **Auto** at session start (boot) and end (`/sync`). Update `core.md` proactively on findings. | Project | `.agent/skills/memory/SKILL.md` |
| **Monitor** | Status checks, "how are runs", before deploying new runs, anomaly triage. | Project | `.agent/skills/monitor/SKILL.md` |
| **Optimization** | SPS regression, low GPU util, new hardware, perf tuning. **Auto before each deployment.** | Project | `.agent/skills/optimization/SKILL.md` |
| **Math** | Manual ("check math") + **auto after ANY formula/equation/numerical logic change.** | Project | `.agent/skills/math/SKILL.md` |
| **Dashboard** | **Auto** after `/monitor`, during `/sync`, on experiment state changes. | Project | `.agent/skills/dashboard/SKILL.md` |
| **WandB** | **Auto** for HPO analysis, run diagnostics, config diffing. Always override `metric_keys`. | User + Project | `~/.claude/skills/wandb/SKILL.md` + `.agents/skills/wandb-primary/SKILL.md` (project addendum) |
| **Researcher** | "Should we try X?", algorithm eval, lit review, root cause analysis. | User + Project | `~/.claude/skills/researcher/SKILL.md` + `.agent/skills/researcher/REFERENCE.md` |
| **Architect** | New module design, pipeline refactor, API/interface changes. **Auto** after Researcher GO. | User + Project | `~/.claude/skills/architect/SKILL.md` + `.agent/skills/architect/REFERENCE.md` |
| **Docker** | Docker, containers, compose, build, start/stop strategies, IBGateway, VNC, Portainer. **Auto** before live-trading container launch. | Project | `.agent/skills/docker/SKILL.md` |
| **Live-Trading** | Start/stop paper/live trading, launch strategy, `--mainnet`, graduation, kill file, risk config, cTrader OAuth. | Project | `.agent/skills/live-trading/SKILL.md` |
| **Live-Monitor** | Live P&L, positions, drawdown, "how are my strategies", container health, Grafana, Telegram alerts. **Auto** after live-trading launch. Periodic via `/loop`. | Project | `.agent/skills/live-monitor/SKILL.md` |

**Chaining rules:**
- Code change / implementation complete -> **Audit** (mandatory). +**Math** if formulas. +**Optimization** if perf.
- Deploy request -> **Monitor** -> **Optimization** (SPS check) -> **Deploy** -> **Monitor** -> **Dashboard**
- `/monitor` -> **Monitor** -> **Dashboard**. Experiment state change -> **Dashboard**.
- `/monitor` (with live trading active) -> **Monitor** (training) + **Live-Monitor** (trading) -> **Dashboard**
- Session start -> **Memory** boot. `/sync` -> **Memory** -> **Dashboard** -> git commit.
- Experiment result / HPO complete -> **WandB** -> **Memory** -> **Dashboard** -> git commit.
- Run stall/crash -> **Monitor** -> **WandB** (`diagnose_run`).
- Research question -> **Researcher** -> if GO -> **Architect** -> implement -> **Audit**.
- Root cause / new module -> **Researcher** <-> **Architect** -> implement -> **Audit**.
- Live trading launch -> **Live-Trading** pre-flight -> **Docker** (if container) -> **Live-Monitor** (verify startup) -> **Dashboard**
- "How are my strategies" (live) -> **Live-Monitor** -> **Dashboard**
- Paper graduation -> **Live-Monitor** (verify paper metrics) -> **Live-Trading** (switch to `--mainnet`)
- Container issue / alert triage -> **Live-Monitor** -> **Docker** (restart if needed)

**Disambiguation (Monitor vs Live-Monitor):**

| User Says | Route To |
|-----------|----------|
| "how are my runs" / SPS / Q-value / loss | **Monitor** (GPU training) |
| "how are my strategies" / P&L / positions / drawdown | **Live-Monitor** (live trading) |
| "check containers" / Docker status | **Docker** |
| "deploy to GPU" | **Deploy** |
| "start trading" / "go live" | **Live-Trading** + **Docker** |

## Memory Protocol (2-Tier + Cloud Search Index)

```
Tier 1: .agent/memory/core.md   -- Project status (~100 lines, deterministic boot context)
Tier 2: randd_log.md             -- R&D write buffer (~20 entries, auto-rotated at 150 KB)
        randd_archive/YYYY-MM.md -- Monthly archives (cold backup, grep-searchable)
Cloud:  agent-memory MCP         -- LanceDB on GCS, search index over ALL R&D entries
```

**Boot:** `core.md` loaded via system prompt hook. `memory_search` for semantic retrieval → grep `randd_log.md` for recent exact matches → grep `randd_archive/` as fallback.
**Commit:** Append `randd_log.md` → `memory_store` new entry to LanceDB → auto-rotate if >150 KB → update `core.md` → git commit.
**Auto-rotate:** `python scripts/rotate_randd_log.py --keep-months 1 --max-entries 20` (triggered during `/sync` when >150 KB). `--max-entries` acts as both cap and floor at month boundaries.
**Bulk re-index:** `python scripts/bulk_index_memory.py --force` after archive rotation or to rebuild the search index.
**Cloud:** Requires `GOOGLE_SERVICE_ACCOUNT` env var in `.mcp.json` pointing to `~/.openclaw/gcs-service-account.json`. Flat files remain authoritative — LanceDB is a search acceleration layer.

## Docker Monitoring Architecture

Live trading containers run on a **remote desktop** (<TAILSCALE_HOST>), accessed via Docker context `finrl-desktop` (`ssh://user@<TAILSCALE_HOST>`). Use `docker --context finrl-desktop` or `./scripts/manage_strategies.sh` (auto-sets context) for all monitoring commands. **Never use bare `docker ps`/`docker exec`** — that targets local Docker Desktop which has no trading containers.

Live trading containers export health + metrics for observability.

### Health Check (Layer 0)
`LiveTradingEngine` writes `/tmp/health_status.json` every bar with: timestamp, position, PV, drawdown, broker_connected, consecutive_errors, should_stop. `healthcheck.sh` reads this JSON and checks staleness (`MAX_STALE_SECONDS`, default 600s), broker connection, error count. Falls back to `pgrep` during bootstrap.

### Prometheus Metrics (Layer 1)
`finrl_pro_ds/crypto/live/metrics.py` — `TradingMetrics` class runs `prometheus_client` HTTP server on a **daemon thread** (completely decoupled from the async trading loop). `Gauge.set()` is thread-safe. Each strategy gets a unique port via `METRICS_PORT` env var (9101-9106). Disabled by default (`METRICS_PORT=0`).

**Exported metrics:** `finrl_position`, `finrl_portfolio_value`, `finrl_drawdown_pct`, `finrl_daily_loss_pct`, `finrl_broker_connected`, `finrl_bar_count`, `finrl_total_trades`, `finrl_total_fees`, `finrl_consecutive_errors`, `finrl_last_bar_timestamp`, `finrl_funding_rate`, `prism_position_multiplier`, `prism_composite_code`, `prism_api_latency_seconds`, `prism_api_errors_total`, `prism_fallback_active`.

### Grafana Dashboard (Layer 2)
Auto-provisioned via baked Dockerfile (`Dockerfile.grafana`). Dashboard: "FinRL Trading Overview" — 10 panels (PV, drawdown, position, daily P&L, broker status, bars, trades, errors, funding rate, fees). Alerting via Telegram contact point.

### Watchdog Container (Layer 3)
`scripts/watchdog_docker.py` — Docker events listener + 5-min periodic sweep + 30-min WandB check. Sends Telegram alerts on unhealthy/died/restart events. Read-only (never sends commands to trading containers). Requires `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`.

### Config Keys (monitoring section in live YAML)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `monitoring.metrics_port` | int | 0 | Prometheus HTTP port (0=disabled, also reads `METRICS_PORT` env var) |
| `monitoring.health_file` | str | `/tmp/health_status.json` | Health status JSON path |

### Docker Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `METRICS_PORT` | 0 | Prometheus metrics port per strategy |
| `MAX_STALE_SECONDS` | 600 | Healthcheck staleness threshold |
| `TELEGRAM_BOT_TOKEN` | (empty) | Watchdog Telegram bot token |
| `TELEGRAM_CHAT_ID` | (empty) | Watchdog Telegram chat ID |
| `GRAFANA_ADMIN_PASSWORD` | finrl | Grafana admin password |

### Port Assignments

| Container | Metrics Port | Service Port |
|-----------|-------------|-------------|
| gmgp1-gold | 9101 | — (shares ibgateway network) |
| sg1-gold | 9102 | — (shares ibgateway network) |
| gmgp1-btc | 9103 | — |
| funding-arb | 9104 | — |
| sync-1h | 9105 | — |
| gmgp1-xauusd | 9106 | — |
| Prometheus | — | 9090 |
| Grafana | — | 3000 |
| Portainer | — | 9443 |

## PRISM Integration (Probabilistic Regime-Informed System for Markets)

PRISM provides L2 position sizing via regime detection. It scales agent positions based on volatility regimes without requiring model retraining.

### Architecture

**Dual GAHMM** (Gaussian-Autoregressive HMM):
- Price Regime: 3 states (BEARISH, NEUTRAL, BULLISH)
- Vol Regime: 3 states (LOW_VOL, NORMAL_VOL, HIGH_VOL)
- Composite Code: 9-state grid (price × 3 + vol). Code 2 (BEARISH + HIGH_VOL) = crisis flatten.

**Chronos-2** forecasting: p10/p30/p50/p70/p90 quantile log-returns (available but not yet used in L2).

### L2 Overlay — Position Sizing

Injected at Step 5b in `LiveTradingEngine` (after SAC inference, before deadband):

```
target_position *= vol_regime_multiplier
```

| Vol Regime | Default Multiplier | Effect |
|------------|-------------------|--------|
| LOW_VOL | 1.3 | Confidence boost |
| NORMAL_VOL | 1.0 | No change |
| HIGH_VOL | 0.3 | De-risk |
| Crisis (code 2) | 0.0 | Flatten position |

Fallback on API error: multiplier = 1.0 (no overlay). 60-second cache TTL.

### Config Keys (`prism:` section in live YAMLs)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `prism.enabled` | bool | false | Activation flag |
| `prism.base_url` | str | `http://prism-api:8001` | PRISM API endpoint (Docker DNS) |
| `prism.api_key` | str | `""` | API key (empty = no auth, internal network) |
| `prism.ticker` | str | `"BTC-USD"` | Asset for regime detection |
| `prism.timeframe` | str | `"daily"` | HMM training window |
| `prism.timeout` | int | 5 | API call timeout (seconds) |
| `prism.cache_ttl` | int | 60 | Regime cache duration (seconds) |
| `prism.crisis_flatten` | bool | true | Enable code-2 position flattening |
| `prism.multipliers.LOW_VOL` | float | 1.3 | Low-vol regime multiplier |
| `prism.multipliers.NORMAL_VOL` | float | 1.0 | Normal-vol regime multiplier |
| `prism.multipliers.HIGH_VOL` | float | 0.3 | High-vol regime multiplier |

### Docker Stack (Profile: `prism`)

3 services in `docker-compose.prism.yaml`, all on `finrl-net`:

| Container | Image | Resources | Healthcheck |
|-----------|-------|-----------|-------------|
| prism-db | PostgreSQL 15 (baked schema) | 512 MB, 1 CPU | `pg_isready` 5s interval |
| prism-api | FastAPI (Chronos-2 + HMMs) | 4 GB, 2 CPU | HTTP `/health` 30s interval, 180s startup |
| prism-refit-worker | HMM refit scheduler | 2 GB, 1 CPU | Auto-restart on refit completion |

Refit schedule: price HMM every 24h, vol HMM every 12h.

### PRISM Docker Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `PRISM_DB_PASSWORD` | (required) | PostgreSQL password |
| `PRISM_API_KEY` | (empty) | API auth key |
| `PRISM_HMM_TICKERS` | `"BTC-USD,GC=F"` | Tickers for HMM fitting |

### PRISM Port Assignments

| Container | Service Port |
|-----------|-------------|
| prism-db | 5432 (internal) |
| prism-api | 8001 |

### PRISM Feature Columns (13 features, for future L1 integration)

`chronos_p10`, `chronos_p30`, `chronos_p50`, `chronos_p70`, `chronos_p90`, `chronos_spread`, `gahmm_price_bear`, `gahmm_price_neutral`, `gahmm_price_bull`, `gahmm_vol_low`, `gahmm_vol_normal`, `gahmm_vol_high`, `gahmm_composite_code`.

GAHMM probability features are passthrough (skip z-score normalization).

### Status

- **Deployed**: All 3 containers healthy (S299). HMMs fitted for BTC-USD + GC=F.
- **Disabled by default**: `prism.enabled: false` in all 7 live configs. Awaiting backtest validation.
- **Next**: Backtest with PRISM features (L1) → enable `prism.enabled: true` per strategy.

## Gotchas (Last verified: 2026-04-01)

- HPO uses NopPruner, no early-kill, 500K steps/trial
- RTX 5090 + CUDA 13.0: run `scripts/patch_torch_compile.py` on fresh deployments
- Taker fills same-bar; maker pends to next bar
- Gold data was corrupted (Session 106) -- always validate via `scripts/clean_ohlcv.py`
- Concurrent GPU runs: check VRAM (not run count) -- 2+ runs can share 1 GPU
- Legacy envs: V6 `Discrete(2)` private=4, V5 `Discrete(6)` -- do NOT modify action spaces
- Docker Desktop Windows: file bind mounts fail silently -- use baked Dockerfiles (COPY at build) instead of volume mounts for config files
- Prometheus/Grafana configs: edit source files in `docker/live/` then rebuild (`docker compose --profile monitoring build`)
- IB strategies share `ibgateway` network namespace -- Prometheus scrapes them via `ibgateway:<port>`, not by container name
- PRISM overlay is disabled by default (`prism.enabled: false`) -- must enable per-strategy after backtest validation
- PRISM API uses Docker DNS (`prism-api:8001`) -- only reachable inside `finrl-net`, not from host
- PRISM compose overlay (`docker-compose.prism.yaml`) must be included with `-f` flag alongside main compose
