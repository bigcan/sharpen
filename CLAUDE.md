# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FinRL-Pro_DS** — Build a profitable RL trading system that consistently outperforms the market across asset classes. Currently implementing DeepScalper-style agents (IQN, BDQ, PPO) on crypto (BTC/USDT) and commodity futures (Gold), with a swing MDP architecture targeting PF > 1.3 OOS.

**Current Research Stage:** Stage 3 — Swing MDP Pivot (Binary {Long, Short} at 3-min, IQN agent).

**WandB Project:** `FinRL-Pro-DS` / entity `bigcan-chiwin-technology`

## Commands

### Setup

```bash
python -m venv .venv && source .venv/bin/activate  # Unix
python -m venv .venv && .\.venv\Scripts\Activate.ps1  # Windows
pip install -e .[dev]
```

### Run Pipeline

```bash
# Development (50K steps, 4 envs, fast iteration)
python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml

# PPO variant
python scripts/run_full_pipeline.py --config configs/tier2_ppo_dev.yaml --agent ppo

# Tier 1 scaled (5M steps, HPO)
python scripts/run_full_pipeline.py --config configs/tier1_scaled.yaml

# Backtest only (requires trained checkpoint)
python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml \
  --backtest_only --checkpoint checkpoints/<RunName>/checkpoint_final.pth

# Deploy to remote GPU (GPUHub/RunPod)
python scripts/deploy_bare_metal.py --config configs/deepscalper_rtx5090_production.yaml
```

Key CLI flags: `--agent {bdq|ppo}`, `--trials N`, `--steps N`, `--run_name NAME`, `--backtest_only`, `--checkpoint PATH`.

### Lint, Type Check, Test

```bash
ruff check finrl_pro_ds          # Lint
mypy finrl_pro_ds --ignore-missing-imports  # Type check
pytest                            # Full suite (40 tests)
pytest tests/deepscalper/ -v     # Module-specific
pytest -k "test_env" -v          # Pattern match
```

### Monitoring & Data Collection

```bash
python scripts/monitor_run.py --run_id <ID>
python scripts/collect_run.py --run_id <ID>   # Collect single run (WandB + SFTP + report)
python scripts/collect_run.py --batch          # Auto-collect all finished runs
```

### Memory Protocol — 3-Tier System + Vector Search

The agent has a persistent file-based memory system with semantic search. Full spec: `.agent/skills/memory/SKILL.md`.

```
Tier 1: .agent/memory/core.md        (git-tracked) — Project facts, status, decisions (~100 lines)
Tier 2: randd_log.md                  (git-tracked) — Durable R&D history (canonical experiment record)
Tier 3: .agent/memory/logs/YYYY-MM-DD.md (gitignored) — Ephemeral daily scratchpad
L2 Vector: ~/.agent-memory/lancedb/   (local) — Semantic search over all indexed content
```

**Semantic Memory (agent-memory MCP)**:
- When investigating a topic with prior history, **search vector DB first** before grepping files
- Tool reference: see "When to use which memory" table in MCP Servers section below

**Boot sequence** (every session, silent):
1. Read `core.md` — check `Last Updated` freshness (>3 days = stale warning)
2. Read last 2 entries from `randd_log.md` (scan for `## 20` date headers)
3. Read today's + yesterday's + T-2 daily logs (skip if missing)
4. Fallback: if ALL daily logs missing, R&D log is sole recent context
5. Validate artifact links in `core.md` — remove broken references silently

**Active maintenance** (during session):
- **Tier 3**: Log key decisions, bugs, deployments to today's daily log
- **Tier 1**: Update `core.md` immediately on significant findings (proactive, don't wait for user)
- **Tier 2**: Sync daily log → `randd_log.md` at session end or on `/commit`
- R&D log is reverse-chronological, append-only, one entry per day
- Full workflow: `.agent/workflows/memory-boot.md`

**Commit checklist** (on "update memory" / "commit" / session end):
1. **Tier 3 FIRST**: Append session work to `.agent/memory/logs/YYYY-MM-DD.md` — this is the source of truth for what happened today
2. **Tier 1**: Update `core.md` status, incidents, next steps
3. **Tier 2**: Sync daily log → `randd_log.md` (structured entry)
4. **Reindex**: If R&D log was modified, run `index_randd_log` to update vector DB
5. **git commit**: Stage and commit

> **MANDATORY**: Never skip the daily log. It is the primary record. Tier 2 is derived FROM it.

## Architecture

### Data Flow

```
btc_lob_2025.parquet
  → ParquetDataHandler (feature_engineering.py)
      SymLog → EMA-Z(span=120, shift=1) → tanh[-1,1]
      30 micro features (LOB-derived: OFI, spread, depth)
      15 macro features (OHLCV-derived: returns, volatility, RSI, session)
  → DeepScalperEnv (Gymnasium)
      obs = Dict{micro:(W,30), macro:(15,), private:(W,5)}
      action = Discrete(6): TakerBuy|MakerBuy|Hold|Cancel|MakerSell|TakerSell
      reward = NAV delta (bps) ± optional DSR blend
  → BDQ / PPO Agent
      MicroEncoder: LSTM(30→256) + LayerNorm
      MacroEncoder: MLP(15→256)
      Fusion MLP(512→256) → Dueling heads V(s) + A(s,a)
      Auxiliary head: volatility prediction (multi-task regularizer)
  → FlatReplayBuffer (2M entries, pre-allocated numpy)
  → Training Loop (AMP + Torch Compile + PER)
  → WandB logging + Checkpoints
  → Backtesting (val split, then test split)
```

### Three-Phase Pipeline (`scripts/run_full_pipeline.py`)

1. **HPO (Optuna / Hyperband):** Objective = `profit_factor` (not Sharpe — prevents gaming). Minimum 100 trades enforced. Reward params are locked; only optimizer hyperparameters are tuned.
2. **Training:** Single WandB run captures full pipeline. Cosine LR annealing, periodic + final checkpointing.
3. **Backtesting:** Deterministic (ε=0). Runs val split then test split. Normalization statistics are cut off at split boundaries (FIX LEAK-1).

### Key Modules

| Path | Responsibility |
|------|---------------|
| `finrl_pro_ds/envs/deep_scalper_env.py` | Gymnasium trading env — LOB execution, NAV reward, margin accounting |
| `finrl_pro_ds/agents/deepscalper/bdq_agent.py` | BDQ agent — ε-greedy, Bellman loss + auxiliary volatility loss |
| `finrl_pro_ds/agents/deepscalper/networks.py` | MicroEncoder (LSTM), MacroEncoder (MLP), DeepScalperNetwork (dueling) |
| `finrl_pro_ds/agents/ppo_scalper/ppo_agent.py` | PPO agent — GAE, clip ratio, entropy regularization |
| `finrl_pro_ds/data/feature_engineering.py` | 30 micro + 15 macro features, EMA-Z normalization |
| `finrl_pro_ds/data/parquet_handler.py` | Zero-copy Parquet streaming with optional shared memory |
| `finrl_pro_ds/training/deepscalper_trainer.py` | BDQ training loop, HPO dispatching, checkpointing |
| `scripts/run_full_pipeline.py` | Unified entry point: HPO → Train → Backtest |
| `scripts/deploy_bare_metal.py` | Remote GPU deployment (GPUHub/RunPod) |

### Private State (5-dim, as of Tier 2)

`[position, balance, remaining_time, order_direction, order_distance_bps]`

Config key: `private_input_size: 5`. This changed from 3 in Tier 1 — configs must match exactly.

### Action Space (Tier 2 Flattened)

`Discrete(6)` — single-head policy. Prior versions used `MultiDiscrete([5,9])` (price × quantity branches). Do not revert to MultiDiscrete.

## Post-Implementation Audit (MANDATORY)

After completing ANY code change to `finrl_pro_ds/**`, `scripts/**`, or `configs/**`, you MUST run the audit protocol before marking the task complete or moving to the next task:

1. Read `.agent/skills/audit/SKILL.md`
2. Execute all applicable phases (lint, tests, invariants, config, logic, performance)
3. Output the Audit Report with a PASS / PASS WITH NOTES / BLOCK verdict
4. If BLOCK: fix the finding before proceeding. Do not skip.
5. Update the daily log with the audit result

Skip only for: documentation-only changes (`.md`), memory system updates, pure formatting.

## Extension Boundary

Only modify code under:
- `finrl_pro_ds/**` — core package
- `tests/**` — test suite
- `configs/**` — YAML configs
- `scripts/**` — pipeline scripts
- `docs/**`, `README.md`, `AGENTS.md`

Do not modify upstream code (`FinRLPodracer/**`, `Podracer/**`).

## Critical Design Invariants

These are non-negotiable correctness constraints. Violating any of these causes silent data leakage or reward hacking:

| ID | Rule |
|----|------|
| **LEAK-1** | Normalization cutoff dates must reset rolling EMA-Z statistics at train/val/test boundaries. Never normalize across splits. |
| **BUG-03** | Hindsight reward (`hindsight_weight > 0`) uses future prices — it **must be disabled** (`hindsight_weight=0.0`) during backtesting. |
| **BUG-01** | HPO objective is `profit_factor`, not Sharpe. Reward structure params (gamma, sharpe_weight, hindsight_*) must be locked in HPO trials. |
| **T1.5v2** | Maker fills use candle high/low prices, not just best bid/offer snapshots. Taker fills are unconditional (taker crosses spread by definition). |
| **SHORT-ACCT** | Long and short leverage accounting are calculated with split formulas. Shorts must NOT accumulate `notional_debt` — buyback obligation is captured by `|pos|*mid` in equity (FIX V3-01). |
| **MARGIN-CFG** | `margin_requirement` must be `0.05` (20x leverage) for BTC futures. `1.0` (spot) causes margin starvation — agent can only hold ~1 BTC before all orders are rejected. |
| **TAKER-IMM** | Taker orders fill immediately in the same `step()` call. Maker orders pend for next-bar fill. `_try_fill_pending()` is called twice: once for previous maker fills, once after taker action creation. |
| **DATA-CLEAN** | All OHLCV source data must pass `scripts/clean_ohlcv.py` validation before any experiment (training, oracle, backtest). Checks: high ≥ max(O,C), low ≤ min(O,C), no NaN/zero/negative, no decimal-shift outliers (>5% ratio). Run on raw 1-min sources first, then re-derive downstream timeframes. Backups (`.parquet.bak`) are mandatory before modifying. |
| **PF-XCHECK** | Any PF simulation must be cross-checked against both `mid_price = (high+low)/2` AND `close` price. If the two PF values diverge by more than 30%, the data is suspect — halt and investigate before trusting results. This prevents the "oracle gate blind spot" where corrupted high/low inflates mid_price-based PF while close-based PF reveals the truth. |

## Known Open Issues (as of 2026-02-20)

From zero-trust adversarial audits (V2 + V3):

| ID | Severity | File | Status | Issue |
|----|----------|------|--------|-------|
| FIND-NEW-01 | **CRITICAL** | `bdq_agent.py` | **FIXED** | `step_count` incremented in both `train_step()` AND `decay_epsilon()` → epsilon decays 2x too fast |
| FIND-V3-01 | **CRITICAL** | `deep_scalper_env.py` | **FIXED** | Short-side leveraged equity inflated NAV by `notional*(1-margin_req)` per entry (~1900bps at 0.05 margin). `notional_debt` was incorrectly accumulated for shorts and added to equity. |
| FIND-NEW-04 | Medium | `wandb_evaluator.py` | **FIXED** | Sortino formula uses `std(negative_returns)` instead of correct `sqrt(mean(min(r,0)^2))` |
| FIND-NEW-09 | Medium | `wandb_evaluator.py` | **FIXED** | Annualization factor uses 252 (daily equity) — should be 525600 (minute-level crypto) |
| FIND-NEW-03 | Medium | `deepscalper_trainer.py` | **FALSE POSITIVE** | Gradient accumulator starts at 0 during `learning_starts` warmup — this is correct behavior (accumulator resets naturally when training begins) |
| FIND-NEW-12 | Low | `run_full_pipeline.py` | Open | PPO `RunningMeanStd` normalizer state not persisted in checkpoint (low impact: reward normalizer lives on env, resets on reset()) |

## Config Schema Reference

Key fields that are commonly misconfigured:

```yaml
network:
  micro_config:
    private_input_size: 5   # Must be 5 (Tier 2). NOT 3.

env:
  margin_requirement: 0.05  # Must be 0.05 (20x leverage). NOT 1.0 (spot — causes margin starvation).
  reward:
    hindsight_weight: 0.0   # Must be 0 during backtest
  action:
    # Tier 2: use discrete_dims: 6 (NOT MultiDiscrete)

agents:
  bdq:
    use_per: true           # Prioritized Experience Replay
```

## GPU/CPU Performance Optimizations

Six optimizations are built into the training pipeline. Four are **always-on** (no config needed), two are config-gated. When creating new scripts, configs, or training loops, preserve these:

| Optimization | Location | Config | Notes |
|-------------|----------|--------|-------|
| **Vectorized PER** | `per_buffer.py` — `get_batch()` / `batch_update()` | Always-on | Numpy vectorized sampling/update, no Python loops. Do NOT rewrite with per-element iteration. |
| **Raw numpy data path** | `parquet_handler.py`, `deep_scalper_env.py` | Always-on | Env returns raw numpy arrays, not dicts. Avoids per-step dict construction + `.get()` overhead. |
| **Batch replay push** | `flat_replay_buffer.py` — `push_batch()` | Always-on | Single call pushes all parallel env transitions with wraparound. Do NOT use per-env `push()` loops. |
| **non_blocking H2D** | All `.to(device)` calls in agents/trainers | Always-on | `non_blocking=True` overlaps CPU→GPU transfers with compute. Always include on new `.to(device)` calls. |
| **torch.compile** | `bdq_agent.py`, `ppo_agent.py` | `torch_compile: true` | JIT-fuses ops on CUDA. Disabled on some GPU/driver combos (RTX 5090 + CUDA 13.0 segfaults). New configs should default to `true`. |
| **High UTD ratio** | `deepscalper_trainer.py` | `update_interval: 8` | Multiple gradient updates per env step. GPU is idle 94% at UTD=2; UTD=8 fills that gap. Tau auto-scales to maintain constant target net tracking rate. Safe up to UTD=16. |

**Rules for new code:**
- New `.to(device)` calls **must** include `non_blocking=True`
- New replay buffers **must** support batch push (no per-sample loops)
- New configs **must** include `torch_compile: true` under `agents.bdq` or `agents.ppo`
- New configs **should** use `update_interval: 8` for GPU utilization (tau auto-scales, no manual adjustment needed)
- Do NOT introduce Python-level loops over batch dimensions in hot paths (sampling, updates, H2D transfers)

## Verification Tiers (from AGENTS.md)

Before deploying production configs:
1. **Tier 1 (Smoke):** 100 steps — verify no crashes
2. **Tier 2 (Pilot):** 100K steps — verify logic stability. **Required for prod approval.**
3. **Tier 3 (Production):** Full scale — verify convergence

## Agent Skills

Specialized skills are installed in `.agent/skills/`:

- **Memory Manager** — Persistent project context. Run `/memory-boot` at session start to load `core.md`.
- **Deployment Manager** — Robust remote GPU deployment with config validation.
- **Audit** — Comprehensive post-implementation audit. Run automatically after every code change: lint, tests, invariant check, config validation, logic spot-check, performance regression, memory update.
- **Optimization** — GPU training optimization based on NVIDIA guidelines. 10-phase protocol: hardware profiling, Tensor Core alignment audit, AMP/BF16 config, torch.compile tuning, memory overlap, UTD ratio, agent-specific settings, SPS diagnostics. Run before deploying new configs or diagnosing performance regressions.

## MCP Servers (`.mcp.json`)

| Server | Purpose | Tools |
|--------|---------|-------|
| `memory` | **Cloud** long-term memory — shared across agents (OpenClaw, Claude Code, Agent Zero). Gemini embeddings, GCS-backed. | `memory_store`, `memory_search`, `memory_forget`, `memory_count` |
| `agent-memory` | **Local** project-scoped semantic search — LanceDB + all-MiniLM-L6-v2, zero cloud. | `search_memory`, `index_randd_log`, `index_document`, `index_status`, `clear_index` |

### When to use which memory

| Use case | Tool |
|----------|------|
| Store a user preference, decision, or fact that applies **across projects/agents** | `memory_store` (cloud) |
| Search for past user preferences, cross-project context, or agent-shared knowledge | `memory_search` (cloud) |
| Search for **this project's** R&D history, experiment results, bug fixes, architecture decisions | `search_memory` (local) |
| Reindex after R&D log or document updates | `index_randd_log` / `index_document` (local) |
| Forget/correct a stored preference or fact | `memory_forget` (cloud) |

**Rules**:
- When the user says "remember this" or states a preference, store it in cloud memory (`memory_store`).
- When investigating project history, search local memory first (`search_memory`), then cloud if needed.
- **Proactive cloud saves**: After any significant experiment result, architecture decision, or infrastructure change, proactively call `memory_store`. Don't wait for `/sync` or session end.
- **Do NOT save to cloud**: Raw R&D log entries (too verbose), intermediate debug notes, daily log contents. Cloud memories should be concise facts, not full entries.

### Bidirectional Memory Sync (on `/sync`)

**Cloud → Local**: Pull cloud memories via `memory_search`, index significant ones into local LanceDB so `search_memory` returns unified results.

**Local → Cloud**: Push any key findings not already saved proactively during the session.

## R&D Log

All experimental findings are recorded in `randd_log.md` (reverse-chronological). New entries go at the top.
