# CLAUDE.md Reference (offloaded from root CLAUDE.md for token economy)

Full content moved here 2026-04-13. Root CLAUDE.md retains only load-bearing rules.
Consult this file when: designing new envs, configuring PRISM, wiring Docker monitoring,
or when invariants need deeper context than the table row provides.

## Full Project Map

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
scripts/
configs/
tests/
docker/live/
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

## Full Env Contracts

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

### Legacy Envs
- **V6 SwingScalperEnv** (`swing_scalper_env.py`): `Discrete(2)`, binary swing, private: 4 dims, cooldown_bars.
- **V5 DeepScalperEnv** (`deep_scalper_env.py`): `Discrete(6)` flat. Do NOT revert to MultiDiscrete.

**All envs return raw numpy dicts, NOT Gymnasium wrappers.**

## Full Agent Skills Dispatch Tables

### User-scope (generic, `~/.claude/skills/`)

| Skill | Trigger | Spec |
|-------|---------|------|
| **Audit** | Auto after ANY code change to `finrl_pro_ds/`, `scripts/`, `configs/`. Skip `.md`-only. Also auto after plan/feature/task implementation. | `~/.claude/skills/audit/SKILL.md` + `FINRL.md` |
| **Memory** | Auto at session start (boot) and end (`/sync`). Update `core.md` proactively on findings. | `~/.claude/skills/memory/SKILL.md` |
| **Optimization** | SPS regression, low GPU util, new hardware, perf tuning. Auto before each deployment. | `~/.claude/skills/optimization/SKILL.md` |
| **Math** | Manual ("check math") + auto after ANY formula/equation/numerical logic change. | `~/.claude/skills/math/SKILL.md` |
| **WandB** | Auto for HPO analysis, run diagnostics, config diffing. Always override `metric_keys`. | `~/.claude/skills/wandb/SKILL.md` + `FINRL.md` |
| **Researcher** | "Should we try X?", algorithm eval, lit review, root cause analysis. | `~/.claude/skills/researcher/SKILL.md` + `FINRL.md` |
| **Architect** | New module design, pipeline refactor, API/interface changes. Auto after Researcher GO. | `~/.claude/skills/architect/SKILL.md` + `FINRL.md` |
| **Skill-Evolve** | "audit skills", "skill health", "improve skills". Auto during `/sync` staleness check. Auto after new skill creation. | `~/.claude/skills/skill-evolve/SKILL.md` |
| **Randy** | R&D assistant. Nanobot status, R&D scheduling, auto-update. Auto memory sync during `/sync`. | `~/.claude/skills/randy/SKILL.md` + `FINRL.md` |

### Project-scope (FinRL infra/ops, `.claude/skills/`)

| Skill | Trigger | Spec |
|-------|---------|------|
| **Deploy** | User requests GPU launch, instance management, or run deployment. | `.claude/skills/deploy/SKILL.md` |
| **Monitor** | Status checks, "how are runs", before deploying new runs, anomaly triage. | `.claude/skills/monitor/SKILL.md` |
| **Dashboard** | Auto after `/monitor`, on experiment state changes. | `.claude/skills/dashboard/SKILL.md` |
| **Docker** | Docker, containers, compose, build, start/stop strategies, IBGateway, VNC, Portainer. Auto before live-trading container launch. | `.claude/skills/docker/SKILL.md` |
| **Live-Trading** | Start/stop paper/live trading, launch strategy, `--mainnet`, graduation, kill file, risk config, cTrader OAuth. Also: stack health, container crashes, strategy not trading, post-launch verification. | `.claude/skills/live-trading/SKILL.md` |
| **Live-Monitor** | Live P&L, positions, drawdown, "how are my strategies", container health, Grafana, Telegram alerts. Auto after live-trading launch. Periodic via `/loop`. | `.claude/skills/live-monitor/SKILL.md` |

### Full chaining rules
- Code change / implementation complete -> Audit (mandatory). +Math if formulas. +Optimization if perf.
- Deploy request -> Monitor -> Optimization (SPS check) -> Deploy -> Monitor -> Dashboard
- `/monitor` -> Monitor -> Dashboard. Experiment state change -> Dashboard.
- `/monitor` (with live trading active) -> Monitor (training) + Live-Monitor (trading) -> Dashboard
- Session start -> Memory boot. `/sync` -> Memory -> git commit.
- Experiment result / HPO complete -> WandB -> Memory -> Dashboard -> git commit.
- Run stall/crash -> Monitor -> WandB (`diagnose_run`).
- Research question -> Researcher -> if GO -> Architect -> implement -> Audit.
- Root cause / new module -> Researcher <-> Architect -> implement -> Audit.
- Live trading launch -> Live-Trading pre-flight -> Docker (if container) -> Live-Trading post-launch verification -> Live-Monitor (ongoing) -> Dashboard
- "How are my strategies" / "check the stack" -> Live-Trading periodic health check -> Live-Monitor (detailed P&L) -> Dashboard
- Container crash / alert triage -> Live-Trading stack diagnostics -> fix -> restart -> post-launch verification
- Paper graduation -> Live-Monitor (verify paper metrics) -> Live-Trading (switch to `--mainnet`)
- "audit skills" / "skill health" -> Skill-Evolve (full) -> Memory (log findings).
- `/sync` -> Memory -> Randy (sync_memory.py, if gateway running) -> Skill-Evolve (staleness check only, lightweight) -> git commit.
- New skill created -> Skill-Evolve (onboarding structural check).
- "nanobot status" / "assistant status" / `/randy` -> Randy (check_status.py).

### Disambiguation (Monitor vs Live-Monitor)

| User Says | Route To |
|-----------|----------|
| "how are my runs" / SPS / Q-value / loss | Monitor (GPU training) |
| "how are my strategies" / P&L / positions / drawdown | Live-Monitor (live trading) |
| "check the stack" / container crash / "not trading" | Live-Trading (stack diagnostics) |
| "check containers" / Docker status | Docker |
| "deploy to GPU" | Deploy |
| "start trading" / "go live" | Live-Trading + Docker |
| "audit skills" / "skill health" | Skill-Evolve (ecosystem) |
| "nanobot status" / "assistant" / `/randy` | Randy (R&D assistant) |

## Memory Protocol (detail)

```
Tier 1: .agent/memory/core.md   -- Project status (~100 lines, deterministic boot context)
Tier 2: randd_log.md             -- R&D write buffer (~20 entries, auto-rotated at 150 KB)
        randd_archive/YYYY-MM.md -- Monthly archives (cold backup, grep-searchable)
Cloud:  agent-memory MCP         -- LanceDB on GCS, search index over ALL R&D entries
```

**Boot:** `core.md` loaded via system prompt hook. `memory_search` for semantic retrieval -> grep `randd_log.md` for recent exact matches -> grep `randd_archive/` as fallback.
**Commit:** Append `randd_log.md` -> `memory_store` new entry to LanceDB -> auto-rotate if >150 KB -> update `core.md` -> git commit.
**Auto-rotate:** `python scripts/rotate_randd_log.py --keep-months 1 --max-entries 20` (triggered during `/sync` when >150 KB). `--max-entries` acts as both cap and floor at month boundaries.
**Bulk re-index:** `python scripts/bulk_index_memory.py --force` after archive rotation or to rebuild the search index.
**Cloud:** Requires `GOOGLE_SERVICE_ACCOUNT` env var in `.mcp.json` pointing to `~/.openclaw/gcs-service-account.json`. Flat files remain authoritative -- LanceDB is a search acceleration layer.

## Docker Monitoring Architecture (full)

Live trading containers run on a **remote desktop** (<TAILSCALE_HOST>), accessed via Docker context `finrl-desktop` (`ssh://user@<TAILSCALE_HOST>`). Use `docker --context finrl-desktop` or `./scripts/manage_strategies.sh` (auto-sets context) for all monitoring commands. **Never use bare `docker ps`/`docker exec`** -- that targets local Docker Desktop which has no trading containers.

### Health Check (Layer 0)
`LiveTradingEngine` writes `/tmp/health_status.json` every bar with: timestamp, position, PV, drawdown, broker_connected, consecutive_errors, should_stop. `healthcheck.sh` reads this JSON and checks staleness (`MAX_STALE_SECONDS`, default 600s), broker connection, error count. Falls back to `pgrep` during bootstrap.

### Prometheus Metrics (Layer 1)
`finrl_pro_ds/crypto/live/metrics.py` -- `TradingMetrics` class runs `prometheus_client` HTTP server on a **daemon thread**. `Gauge.set()` is thread-safe. Each strategy gets a unique port via `METRICS_PORT` env var (9101-9106). Disabled by default (`METRICS_PORT=0`).

**Exported metrics:** `finrl_position`, `finrl_portfolio_value`, `finrl_drawdown_pct`, `finrl_daily_loss_pct`, `finrl_broker_connected`, `finrl_bar_count`, `finrl_total_trades`, `finrl_total_fees`, `finrl_consecutive_errors`, `finrl_last_bar_timestamp`, `finrl_funding_rate`, `prism_position_multiplier`, `prism_composite_code`, `prism_api_latency_seconds`, `prism_api_errors_total`, `prism_fallback_active`.

### Grafana Dashboards (Layer 2)
Auto-provisioned via baked Dockerfile (`Dockerfile.grafana`). Two tiers:
- **Fleet Overview** (`dashboards/trading_overview.json`) -- all strategies overlaid on shared panels. 12 panels: PV, drawdown, position, daily P&L, broker status, bars, trades, errors, funding rate, fees, position divergence, PV divergence.
- **Per-strategy dashboards** (`dashboards/strategies/<name>.json`) -- one dashboard per active strategy, same 12 panels with `{strategy="<name>"}` pre-filtered. Generated from the fleet template + `docker/live/grafana/strategies.yml` registry via `python scripts/generate_grafana_dashboards.py`. Edit template / registry then rerun + rebuild -- do NOT hand-edit the generated files. `foldersFromFilesStructure: true` places them in a "strategies" folder in the Grafana UI.

Alerting via Telegram contact point.

### Watchdog Container (Layer 3)
`scripts/watchdog_docker.py` -- Docker events listener + 5-min periodic sweep + 30-min WandB check. Sends Telegram alerts on unhealthy/died/restart events. Read-only (never sends commands to trading containers). Requires `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`.

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
| gmgp1-gold | 9101 | -- (shares ibgateway network) |
| sg1-gold | 9102 | -- (shares ibgateway network) |
| gmgp1-btc | 9103 | -- |
| funding-arb | 9104 | -- |
| sync-1h | 9105 | -- |
| gmgp1-xauusd | 9106 | -- |
| velotrade-btc | 9107 | -- |
| Prometheus | -- | 9090 |
| Grafana | -- | 3000 |
| Portainer | -- | 9443 |

## PRISM Integration (FALSIFIED, kept for reference)

**Status 2026-04-13:** Both L1 and L2 overlays failed backtest gates. Workstream closed.
See `MEMORY.md` -> `decision_prism_l1_rejected.md`. Do not revive without new evidence.
Containers remain deployed; `prism.enabled: false` in all live configs.

### Architecture

**Dual GAHMM** (Gaussian-Autoregressive HMM):
- Price Regime: 3 states (BEARISH, NEUTRAL, BULLISH)
- Vol Regime: 3 states (LOW_VOL, NORMAL_VOL, HIGH_VOL)
- Composite Code: 9-state grid. Code 2 (BEARISH + HIGH_VOL) = crisis flatten.

**Chronos-2** forecasting: p10/p30/p50/p70/p90 quantile log-returns.

### L2 Overlay

Injected at Step 5b in `LiveTradingEngine`: `target_position *= vol_regime_multiplier`.

| Vol Regime | Default Multiplier |
|------------|-------------------|
| LOW_VOL | 1.3 |
| NORMAL_VOL | 1.0 |
| HIGH_VOL | 0.3 |
| Crisis (code 2) | 0.0 |

Fallback on API error: multiplier = 1.0. 60-second cache TTL.

### Config Keys (`prism:` section in live YAMLs)

| Key | Type | Default |
|-----|------|---------|
| `prism.enabled` | bool | false |
| `prism.base_url` | str | `http://prism-api:8001` |
| `prism.api_key` | str | `""` |
| `prism.ticker` | str | `"BTC-USD"` |
| `prism.timeframe` | str | `"daily"` |
| `prism.timeout` | int | 5 |
| `prism.cache_ttl` | int | 60 |
| `prism.crisis_flatten` | bool | true |
| `prism.multipliers.LOW_VOL` | float | 1.3 |
| `prism.multipliers.NORMAL_VOL` | float | 1.0 |
| `prism.multipliers.HIGH_VOL` | float | 0.3 |

### Docker Stack (Profile: `prism`)

3 services in `docker-compose.prism.yaml` on `finrl-net`:

| Container | Image | Resources |
|-----------|-------|-----------|
| prism-db | PostgreSQL 15 | 512 MB, 1 CPU |
| prism-api | FastAPI (Chronos-2 + HMMs) | 4 GB, 2 CPU |
| prism-refit-worker | HMM refit scheduler | 2 GB, 1 CPU |

Refit: price HMM every 24h, vol HMM every 12h.

### PRISM Docker Env Vars

| Var | Default |
|-----|---------|
| `PRISM_DB_PASSWORD` | (required) |
| `PRISM_API_KEY` | (empty) |
| `PRISM_HMM_TICKERS` | `"BTC-USD,GC=F"` |

### PRISM Ports

| Container | Port |
|-----------|------|
| prism-db | 5432 (internal) |
| prism-api | 8001 |

### PRISM Feature Columns (13 features)

`chronos_p10`, `chronos_p30`, `chronos_p50`, `chronos_p70`, `chronos_p90`, `chronos_spread`, `gahmm_price_bear`, `gahmm_price_neutral`, `gahmm_price_bull`, `gahmm_vol_low`, `gahmm_vol_normal`, `gahmm_vol_high`, `gahmm_composite_code`.

GAHMM probability features are passthrough (skip z-score normalization).

## Extended Commands

```bash
# Docker Live Trading (verbose forms — prefer ./scripts/manage_strategies.sh)
docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile all up -d
# Profiles: ib, crypto, ctrader, monitoring, prism, all

docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile monitoring up -d

docker compose -f docker/live/docker-compose.yaml \
               -f docker/live/docker-compose.prism.yaml \
               -f docker/live/docker-compose.desktop.yaml \
               --profile prism up -d

docker compose -f docker/live/docker-compose.yaml \
               --profile monitoring build

# Checkpoint collection
python scripts/auto_collect_checkpoints.py                    # WandB-based, last 24h
python scripts/auto_collect_checkpoints.py --hours 48
python scripts/auto_collect_checkpoints.py --run_id <ID>
python scripts/auto_collect_checkpoints.py --all_instances
# Flags: --dry_run, --include_all
```
