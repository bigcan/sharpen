# CLAUDE.md

## Project Brief

**Goal:** Help Keng build a robust, consistent, profitable RL quant trading system across asset classes.

RL agents (IQN, BDQ, SAC) on BTC/USDT and Gold futures. Swing MDP: binary {Long, Short}, target PF > 1.3 OOS. WandB: `FinRL-Pro-DS` / `bigcan-chiwin-technology`.

## Stack

Python 3.11+ · PyTorch 2.8+ · Gymnasium · Optuna · WandB · Parquet · Ruff · Mypy · Pytest

## Commands

```bash
# Setup
pip install -e .[dev]

# Pipeline: HPO → Train → Backtest
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml
# Flags: --agent {bdq|ppo} --trials N --steps N --backtest_only --checkpoint PATH

# Deploy to remote GPU
python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> --collect

# Quality
ruff check finrl_pro_ds && mypy finrl_pro_ds --ignore-missing-imports && pytest

# Monitoring
python scripts/monitor_run.py --run_id <ID>
python scripts/collect_run.py --run_id <ID>  # or --batch
```

## Project Map

```
finrl_pro_ds/
  agents/{deepscalper,ppo_scalper,sac}/  # RL agents (BDQ, PPO, SAC)
  envs/                                   # Gymnasium trading environments
  training/                               # Training loops + HPO dispatch
  data/                                   # Feature engineering, data handlers
scripts/        # Pipeline entry points, deployment, monitoring, oracles
configs/        # YAML experiment configs (one per run)
tests/          # pytest suite
.agent/skills/  # Agent skills: audit, memory, deploy, optimization
```

**Boundary:** Only modify `finrl_pro_ds/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries. Never normalize across splits. |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params (gamma, sharpe_weight, hindsight_*) during HPO. |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting (uses future prices). |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |

## Coding Standards

- All `.to(device)` calls **must** use `non_blocking=True`
- Replay buffers **must** support batch push — no per-sample Python loops in hot paths
- New configs: `torch_compile: true`, `update_interval: 8` (tau auto-scales)
- Do NOT rewrite vectorized PER with per-element iteration
- Env returns raw numpy, not dicts — preserve this path

## Agent Skills — Auto-Dispatch

Six skills in `.agent/skills/`. Read the relevant `SKILL.md` before executing. **Trigger proactively** — don't wait for the user to ask.

| Skill | Trigger | Spec |
|-------|---------|------|
| **Audit** | **Auto** after ANY code change to `finrl_pro_ds/`, `scripts/`, `configs/`. Skip `.md`-only. | `.agent/skills/audit/SKILL.md` |
| **Deploy** | User requests GPU launch, instance management, or run deployment. | `.agent/skills/deploy/SKILL.md` |
| **Memory** | **Auto** at session start (boot) and end (`/sync`). Update `core.md` proactively on findings. | `.agent/skills/memory/SKILL.md` |
| **Memory Search** | Investigating past experiments, bugs, decisions. Use tag-based grep before raw file reads. | `.agent/skills/memory-search/SKILL.md` |
| **Monitor** | Status checks, "how are runs", before deploying new runs, anomaly triage. `python scripts/monitor_fleet.py` | `.agent/skills/monitor/SKILL.md` |
| **Optimization** | SPS regression, low GPU util, new hardware, new training loop, perf tuning. Profile first (Phase 1). | `.agent/skills/optimization/SKILL.md` |

**Chaining rules:**
- Code change → **Audit** (mandatory) → if perf-relevant → **Optimization**
- Deploy request → **Monitor** (check fleet) → **Deploy** → **Monitor** (verify)
- Session start → **Memory** boot → **Memory Search** if investigating prior work
- Experiment result → **Memory** update `core.md` → `memory_store` if significant

## Memory Protocol

```
Tier 1: .agent/memory/core.md          — Project status (~100 lines, update proactively)
Tier 2: randd_log.md                    — R&D history (reverse-chron, append-only)
Tier 3: .agent/memory/logs/YYYY-MM-DD.md — Daily scratchpad (gitignored)
Vector: search_memory MCP tool          — Semantic search over indexed content
```

**Boot:** `core.md` → last 2 R&D entries → recent daily logs.
**Commit:** Daily log → `core.md` → `randd_log.md` → reindex → git commit.
**Cloud** (`memory_store`/`memory_search`): cross-project facts only. No raw logs.

## Gotchas

- `private_input_size: 5` (not 3 — changed in Tier 2)
- Action space `Discrete(6)` flat — do NOT revert to MultiDiscrete
- HPO uses NopPruner, no early-kill, 500K steps/trial
- `n_step: 3` required for IQN stability — all IQN configs must include it
- RTX 5090 + CUDA 13.0 may segfault with torch.compile
- Taker fills same-bar; maker pends to next bar
