# CLAUDE.md Reference (offloaded from root CLAUDE.md for token economy)

Full content moved here 2026-04-13. Root CLAUDE.md retains only load-bearing rules.
Consult this file when: designing new envs, configuring PRISM, wiring Docker monitoring,
working on Crucible alpha-mining, or when invariants need deeper context than the table row provides.

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
  signals/                         # Alpha-mining DSL + deflated (T0-T5) evaluation funnel -- dependency of crucible/
    generation/cohort.py           # Weak-signal cohort construction (P2.7)
    generation/cohort_eval.py      # Cohort evaluator: SR* pre-filter + selection-aware MC null
    generation/cohort_mc.py        # Selection-aware Monte Carlo null (stationary bootstrap)
    generation/evolve.py           # C3 genetic search (DSL alpha generation)
    generation/dsl_signal.py       # DSL eval context + overlay path
    generation/config.py           # Generation config schema
    library/_alpha_dsl.py          # WQ101-style alpha DSL primitives
    features.py                    # Panel (OHLCV + feature_slots for macro/positioning, added P1a)
    spec.py                        # SignalSpec, content_hash
  crucible/                        # Crucible agentic alpha-mining platform (crucible-v2.8) -- see dedicated section below
    agentic/                       # Proposer/Author/DataScout -- agent proposes, statistics dispose (CR-1)
    data/                          # Free-data connectors: FRED, CFTC COT, EDGAR, GDELT, Stooq, TWSE, TAIFEX
    governance/                    # Survivor -> Tier-2 handoff pipeline (never runs the audit itself)
    lockbox/                       # Forward-incubation on post-hypothesis data (CR-8, the anti-oracle keystone)
    orchestrator/                  # Continuous nightly-tick loop, per-substrate online-FDR budget
    catalog.py                     # Central data catalog (SQLite) -- what the agent can hypothesize over
    ledger.py                      # Split trial ledger -- agent-blind verdicts, agent-visible dedup view
    manifest.py                    # RunManifest -- version + gates_hash + data_snapshot_hash + rng_seeds
    reproduce.py                   # `crucible reproduce <run_id>` -- re-executes, asserts bit-identical
    version.py                     # Current: crucible-v2.8
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

## Crucible Alpha-Mining Architecture (full)

**Purpose:** `finrl_pro_ds/crucible/` (current: `crucible-v2.8`, see `version.py`) is a continuous agentic alpha-discovery system built on top of the existing `finrl_pro_ds/signals/` DSL + T0-T5 deflated evaluation funnel (not a replacement). Falsification-first pipeline: ACQUIRE (free data connectors) -> HYPOTHESIZE (agent proposes pre-registered specs, blind to verdicts) -> MINE (DSL/genetic search) -> DEFLATE (T0-T5 gates) -> COMBINE + forward-incubate in a **lockbox** on data that postdates the hypothesis timestamp, before any human-initiated Tier-2 audit. Value proposition is the *filter*, not idea supply: rigorous statistical gatekeeping (pre-registration, split-ledger anti-oracle, per-substrate online-FDR) across capacity-constrained free data domains (FRED/ALFRED macro, CFTC COT positioning, SEC EDGAR fundamentals, GDELT sentiment, Stooq global market, TWSE/TAIFEX Taiwan). P0-P5 roadmap shipped (see below); zero PROMISING survivors have cleared the lockbox as of `crucible-v2.8`.

### Subpackages

| Path | Responsibility |
|------|----------------|
| `agentic/` | Proposer/Author/DataScout seam. Shipped default `LibrarySeedProposer` (deterministic, offline, curated WQ101 + macro-overlay seed bank). `HypothesisAuthor` reads ONLY `ledger_agent_view` (dedup keys + killed-family list -- never verdicts/DSR/holdout) + data catalog; emits pre-registered `SignalSpec`s hashed per `proposal_ts`. `DiscoveryCard` records verdict + narrative, read-only after scoring. |
| `data/` | Free-data connectors, uniform `DataConnector` protocol (`connector.py`) carrying `release_timestamp` (the PIT join-leak tripwire). Concrete: `fred.py`, `cftc_cot.py`, `edgar.py`, `gdelt.py`, `stooq.py`, `twse_institutional.py`, `taifex_positioning.py` (poll-only endpoint, local accumulation store). `quality_gate.py` enforces as-of-join reconstruction (fails on look-ahead). `panel_bridge.py`/`altdata_bridge.py` wire connectors into the live `Panel`. |
| `governance/` | Survivor -> notification -> Tier-2-handoff pipeline, run *around* the funnel. `driver.py` scans lockbox for CLEARED entries, builds a `Tier2Handoff` packet (verdict + forward evidence + exact `deep_strategy_audit.js` command), notifies operator (`notify.py`), idempotency guard (`store.py`, SQLite). **Never runs the audit itself** -- human-initiated per root `CLAUDE.md`'s Anti-Patterns. |
| `lockbox/` | Forward-incubation (CR-8, the keystone anti-oracle mechanism). `lockbox.py`: pre-registered criterion pinned at enrollment, mutable accrual state, status ∈ {INCUBATING, CLEARED, REJECTED}, verdict rendered EXACTLY ONCE. `incubation.py` computes forward evidence on bars strictly after `proposal_ts` via `finrl_pro_ds/paper/`. |
| `orchestrator/` | Continuous nightly-tick loop. `orchestrator.py`: `substrate_dirty` gate (mine only on new data/fresh specs, conserves FDR wealth), runs the hypothesis loop within budget. `budget.py` hard per-tick LLM-token/candidate caps. `burst.py` routes expensive HPO to GPUHub. `fdr.py` per-substrate online-FDR (LORD++) wealth. `substrate.py` multi-substrate coordination + versioned tick log. |
| `catalog.py` | Central SQLite data catalog -- what the agent can hypothesize over today (source, series, date_range, freshness, snapshot_hash). Pinned per run for reproducibility. |
| `ledger.py` | Split trial ledger (anti-oracle moat, CR-1): full `trial_ledger` (verdict + DSR + OOS delta, agent-blind) + SQL VIEW `ledger_agent_view` (dedup keys + killed-family list only, agent-visible). |
| `manifest.py` | `RunManifest` pins `crucible_version` + `gates_hash` + `data_snapshot_hash` + `rng_seeds`. Reproducible only if all four match. |
| `reproduce.py` | `crucible reproduce <run_id>` -- re-executes from manifest + recipe, asserts verdicts bit-identical. |

`finrl_pro_ds/signals/` is a **dependency**, not a sibling: Crucible reuses `spec.py`, `generation/grammar.py`, `generation/evolve.py` (C3 genetic search), `generation/dsl_signal.py`, `generation/fitness.py`, `base_sleeves.py`, `eval_harness.py` (T0-T5), `features.py` (`Panel.feature_slots` extension for non-OHLCV series, added P1a). `generation/cohort_eval.py` / `cohort.py` / `cohort_mc.py` (P2.7) add an opt-in weak-signal COHORT evaluator downstream of the main funnel.

### Running it

```bash
# Continuous orchestrator (P3) -- main entry point for unattended discovery
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4      # noise panel, reproducible, no data cost
python scripts/research/crucible_orchestrator.py --mode real --start 2008-01-01 --nights 4   # real data, ticks only when substrate_dirty

# Which substrate a real run mines comes from `generation.panel` in the --config gates file, NOT a
# CLI flag. The wired panels are {cross_asset, taiwan, taiwan_smallcap, us_equity, intraday,
# intraday_fx} (signals/generation/config.py::_WIRED_SUBSTRATES); each pins the exact base book the
# runner builds, so a config naming a book the runner does not build fails fast instead of silently
# scoring against nothing.
python scripts/research/crucible_orchestrator.py --mode real --nights 1 --force \
    --config configs/taiwan_smallcap_signal_eval.gates.yaml   # cap-rank 51-250 TW small/mid-cap
# NOTE this substrate stamps implied MDE ΔSR 1.55 against the guard's 0.50 ceiling, so the power
# guard REFUSES it; mining it anyway needs an explicit --force-underpowered (an operator decision —
# it spends LORD++ FDR wealth at low detection probability). Measure before deciding:
python scripts/research/crucible_real_alpha_breadth.py --panel taiwan_smallcap --hold 21

# Manual single-cycle loop (P2, no orchestrator)
python scripts/research/crucible_hypothesis_loop.py --mode {synthetic|real} [--start YYYY-MM-DD]

# Hand off a CLEARED lockbox survivor to a human Tier-2 audit (P5) -- never runs the audit itself
python scripts/research/crucible_governance.py --lockbox <path> --gov <path> --out <dir> --cards <dir> --workstream crucible --scope overlay --now-ts <ISO8601>

# Verify a past run reproduces bit-identical from its manifest
python scripts/research/crucible_reproduce.py results/crucible_orchestrator/<mode>/<tick_ts>

# Campaign record -- READ BEFORE MINING. Rebuilds the tick-level facts from every store (results/,
# other worktrees, scratch), dedupes rehearsal copies, and classifies each tick TESTED vs
# SCREENED_ONLY/SCREENED_UNKNOWN so a `promising=0` is never misread as a result.
python scripts/research/crucible_mining_log.py --root C:/FinRL/FinRL-Pro_DS --root C:/tmp
# -> docs/research/crucible_mining_log_facts.md (generated) + crucible_mining_log.md (curated:
#    per-campaign question / binding constraint / what would reopen it, plus the substrate board)

# Support scripts
python scripts/research/generate_alphas.py                  # base sleeve generation
python scripts/research/measure_altdata_pool_diversity.py    # data-breadth audit
python scripts/backup_crucible_ledger.py                     # ledger snapshot
```

### Gates (never hardcode -- same rule as training gates)

| File | Gates |
|------|-------|
| `configs/crucible_cohort.gates.yaml` | Opt-in (`enabled: false` by default). Admission: `min_cohort_size`, `max_cohort_size`, `max_pairwise_corr`. Selection-aware MC null: `alpha_cohort`, `mc_n_replicates`, `mc_block_length`. Separate file by design (ADR-1) -- keeps the main funnel `gates_hash` frozen. |
| `configs/crucible_lockbox.gates.yaml` | Forward incubation: `incubation.min_forward_bars` (~63, one quarter), `incubation.min_forward_sharpe` (~0.30 floor). Separate file by design (ADR-2), same reason. |

### Maturity (P0-P5 roadmap, all shipped)

`crucible-v2.0` froze the funnel `gates_hash` (`519158fa1450`) -- every later bump is MINOR (new connectors/capability) and must NOT change existing verdicts. P1a (v2.1, data representation/overlay path) -> P1b (FRED/COT + quality gate) -> P2 (agentic loop, manual) -> P3 (v2.4, continuous orchestrator) -> P4 (v2.5, lockbox) -> P5 (v2.6, GDELT/EDGAR/Stooq breadth + governance + `reproduce`) -> Phase 4 (v2.7, weak-signal cohort) -> P2.8 (Taiwan TWSE/TAIFEX breadth, current).

Design specs: `docs/research/crucible_agentic_discovery_spec.md` (canonical, §0-11), `crucible_weak_signal_ensemble_spec.md`, `crucible_diverse_proposer_spec.md`, `crucible_mc_null_spec.md`. Architecture notes: `.agent/artifacts/crucible_spec_fable_review.md`, `crucible_cohort_integration_architecture.md`, `crucible_taiwan_breadth_and_scheduling_architecture.md`.

## Prediction-Market Research (Polymarket) -- moved out

The Polymarket 5-min up/down maker-diagnostic + forward paper-test (`pm-updown-mm-v0`) that used to live at `scripts/data/*polymarket*` / `scripts/research/polymarket_updown_mm_diagnostic.py` / `configs/polymarket_updown_mm_paper.gates.yaml` was spun off 2026-07-05 into its own repo: [`Chiwin-Technology/polymarket-updown-research`](https://github.com/Chiwin-Technology/polymarket-updown-research). It was always self-contained (zero `finrl_pro_ds` imports, no fleet/Docker/skill integration), so nothing else in this repo depended on it. The Windows Scheduled Task that pulls the remote forward-collector's output now points at the new repo's checkout.

Kalshi credentials remain in this repo (`.env` `KALSHI_API_KEY_ID`, gitignored `kalshi_private_key.pem`) but there is no active Kalshi code here or in the new repo -- a prior SecureFinAI-contest client was deleted post-contest 2026-05-02.

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
- `/sync` -> Memory -> Skill-Evolve (staleness check only, lightweight) -> git commit.
- New skill created -> Skill-Evolve (onboarding structural check).

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

## Memory Protocol (detail)

```
Tier 1: .agent/memory/core.md   -- Project status (~100 lines, deterministic boot context)
Tier 2: randd_log.md             -- R&D write buffer (~20 entries, auto-rotated at 150 KB)
        randd_archive/YYYY-MM.md -- Monthly archives (cold backup, grep-searchable)
Search: agent-memory MCP         -- self-hosted LanceDB native-local on this workstation, embeddings via native Ollama nomic-embed-text (768-dim, cosine)
```

**Boot:** `core.md` loaded via system prompt hook. `memory_search` for semantic retrieval -> grep `randd_log.md` for recent exact matches -> grep `randd_archive/` as fallback.
**Commit:** Append `randd_log.md` -> `memory_store` new entry to LanceDB -> auto-rotate if >150 KB -> update `core.md` -> git commit.
**Auto-rotate:** `python scripts/rotate_randd_log.py --keep-months 1 --max-entries 20` (triggered during `/sync` when >150 KB). `--max-entries` acts as both cap and floor at month boundaries.
**Bulk re-index:** NOT part of the live path. `scripts/bulk_index_memory.py` still hardcodes the retired GCS table (`gs://openclaw-memory-lance/v1`) and Gemini `embedding-001` — a different table *and* a different vector space than the live index — and its `~/.openclaw/gcs-service-account.json` is absent, so it cannot run. It needs porting to the local Ollama+LanceDB stack before its next use. New rows are added per-call via `memory_store` from Claude Code (post-S517); no scheduled re-index path runs today.
**MCP transport:** `.mcp.json` invokes the server natively: `C:/nvm4w/nodejs/node.exe ~/mcp-servers/agent-memory/src/index.js` with env `OLLAMA_URL=http://localhost:11434`, `EMBED_MODEL=nomic-embed-text`, `LANCEDB_URI=~/.openclaw/lancedb/v1`. **No Docker** (decoupled from the offline `finrl-desktop` remote on 2026-06-06; prior `docker --context finrl-desktop exec` config saved at `.mcp.json.remote-docker-bak`). Requires native Ollama (winget `Ollama.Ollama`, auto-starts via `Ollama.lnk` in the user Startup folder, serves `:11434`) + the `nomic-embed-text` model. Server source out-of-repo at `~\mcp-servers\agent-memory\`. The local LanceDB index was **seeded 2026-06-06 by cloning the remote `agent-memory` volume** (`docker --context finrl-desktop cp agent-memory:/data/lancedb/v1` → `~/.openclaw/lancedb/v1`, 1574 rows). There is no flat-file reindex tool (`bulk_index_memory.py` is GCS+Gemini-only and unported), so re-clone from the remote volume to rebuild; flat files (`core.md`, `randd_log.md`) remain authoritative. Older GCS+Gemini rollback: `.mcp.json.gcs-rollback`. Migration record: `randd_log.md` Session 517.

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
