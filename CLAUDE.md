# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeepScalper (FinRL-Pro_DS)** — Institutional-grade high-frequency crypto scalping using Deep Reinforcement Learning. Implements a Branching Dueling Q-Network (BDQ) agent trained on Limit Order Book (LOB) microstructure data, following [Sun et al. (2022)](docs/2201.09058v3.pdf). PPO is also supported.

**Current Research Stage:** Stage 2 — MDP Physics Fix (Tier 2 architectural alignment complete, Tier 3 pending).

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

### Memory Protocol — 3-Tier System

The agent has a persistent file-based memory system. Full spec: `.agent/skills/memory/SKILL.md`.

```
Tier 1: .agent/memory/core.md        (git-tracked) — Project facts, status, decisions (~100 lines)
Tier 2: randd_log.md                  (git-tracked) — Durable R&D history (canonical experiment record)
Tier 3: .agent/memory/logs/YYYY-MM-DD.md (gitignored) — Ephemeral daily scratchpad
```

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
4. **git commit**: Stage and commit

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

## Verification Tiers (from AGENTS.md)

Before deploying production configs:
1. **Tier 1 (Smoke):** 100 steps — verify no crashes
2. **Tier 2 (Pilot):** 100K steps — verify logic stability. **Required for prod approval.**
3. **Tier 3 (Production):** Full scale — verify convergence

## Agent Skills

Specialized skills are installed in `.agent/skills/`:

- **Memory Manager** — Persistent project context. Run `/memory-boot` at session start to load `core.md`.
- **Deployment Manager** — Robust remote GPU deployment with config validation.
- **Backtest Auditor** — Pre-flight checks before running backtests.
- **Log Analyzer** — Diagnoses crash causes, OOMs, and regressions from logs.
- **Research Logger** — Logs experimental findings to `randd_log.md` (reverse-chronological, structured).
- **Work Auditor** — Generates adversarial audit prompts for logic consistency checks.

## R&D Log

All experimental findings are recorded in `randd_log.md` (reverse-chronological). New entries go at the top. Use the Research Logger skill for structured entries.
