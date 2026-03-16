# CLAUDE.md

## Project Brief

**Goal:** Help Keng build a robust, consistent, profitable RL quant trading system across asset classes.

## Current Research State

- **Stage:** 3 — Swing MDP (RL). Phase R.2 (Rehabilitation re-runs) + GMGP1 SAC active.
- **Active MDP (Design A):** Binary direction-switching `{Long, Short}`, Discrete(2), always in-market, dense per-bar reward, 3-bar cooldown.
- **Active MDP (Design B / GMGP1):** Continuous position control `Box(-1,1)`, SAC, multi-scale OHLCV, DSR reward, deadband, fee curriculum.
- **Active Agents:** IQN (distributional, NoisyNets), BDQ (branching dueling), SAC (continuous). PPO definitively falsified (0/15).
- **Target:** PF 1.1–1.3 OOS. Oracle ceilings: BTC 3.53 (6.5bps), Gold 10.34 (2bps) at 3-min close.
- **Assets:** BTC/USDT (Binance, 5bps taker) and Gold Futures (CME, 2bps taker).

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
  agents/
    deepscalper/bdq_agent.py      # BDQ — branching dueling Q-network
    deepscalper/iqn_agent.py      # IQN — implicit quantile network (NoisyNets)
    deepscalper/iqn_network.py    # IQN network architecture
    deepscalper/networks.py       # Shared encoder / Q-head networks
    deepscalper/per_buffer.py     # Prioritized experience replay
    deepscalper/nstep_buffer.py   # N-step return buffer
    deepscalper/flat_replay_buffer.py  # Simple uniform replay
    deepscalper/noisy_linear.py   # NoisyNet linear layer
    ppo_scalper/ppo_agent.py      # PPO (FALSIFIED — 0/15)
    sac/sac_agent.py              # SAC — continuous position control
    sac/networks.py               # SAC actor/critic networks
  envs/
    swing_scalper_env.py          # V6 — Discrete(2) binary swing MDP (Design A)
    continuous_swing_env.py       # V7 — Box(-1,1) continuous MDP (Design B / GMGP1)
    deep_scalper_env.py           # V5 — Discrete(6) flat (Stage 2, legacy)
    augmented_wrapper.py          # Obs augmentation wrapper
  training/                       # Training loops + HPO dispatch
  data/
    feature_engineering.py        # "Feature Factory" — Micro (LOB) + Macro (OHLCV) features
    parquet_handler.py            # Shared-memory data streaming (LOB-based envs)
    multiscale_handler.py         # Multi-scale OHLCV handler (SAC / GMGP1)
scripts/        # Pipeline entry points, deployment, monitoring, oracles
configs/        # YAML experiment configs (one per run)
tests/          # pytest suite
.agent/skills/  # Agent skills: audit, memory, deploy, optimization, monitor
```

**Boundary:** Only modify `finrl_pro_ds/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

## Env Contracts

### SwingScalperEnv (V6) — `swing_scalper_env.py`
| Property | Spec |
|----------|------|
| Action | `Discrete(2)` — 0=Long, 1=Short. Always in-market. |
| Obs | `Dict{ micro: (W, micro_dim), macro: (N_macro,), private: (W, 4) }` |
| Private | `[direction, bars_since_switch, unrealized_pnl_bps, normalized_atr]` — 4 dims |
| Reward | Dense per-bar directional PnL (bps) minus RT fee on switch. `reward_mode: "switch_centric"` alternative via config. |
| Done | Truncated at `episode_length` bars or data exhaustion. Terminated on `max_drawdown_pct` breach. |
| Cooldown | `cooldown_bars` (default 3) — switch requests during cooldown are silently ignored. |

### ContinuousSwingEnv (V7) — `continuous_swing_env.py`
| Property | Spec |
|----------|------|
| Action | `Box(-1, 1, shape=(1,))` — target position fraction. -1=full short, 0=flat, +1=full long. |
| Obs | `Dict{ scale_0: (W, F), scale_1: (W, F), ..., private: (5,) }` — one key per OHLCV scale. |
| Private | `[current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]` — 5 dims |
| Reward | Differential Sharpe Ratio (DSR). `reward_mode: "dsr"` (default). |
| Done | Same as V6 (truncation + drawdown). |
| Deadband | `deadband_threshold: 0.25` — position changes < threshold are ignored (prevents churn). |

### DeepScalperEnv (V5) — `deep_scalper_env.py` (Legacy / Stage 2)
| Property | Spec |
|----------|------|
| Action | `Discrete(6)` flat (BDQ) — do NOT revert to MultiDiscrete. |
| Obs | `Dict{ micro: (W, micro_dim), macro: (N_macro,), private: (W, private_dim) }` |
| Done | Same pattern. |

**All envs return raw numpy dicts, NOT Gymnasium wrappers — preserve this path.**

## Config Schema (Required Keys)

Every YAML config must include these top-level sections and keys. Do NOT invent keys.

```yaml
data:
  file_path: str          # Path to .parquet data file
  ticker: str             # Symbol name (used in logging)
  train_start_date: str   # ISO datetime
  train_end_date: str
  val_start_date: str
  val_end_date: str
  test_start_date: str
  test_end_date: str
  norm_cutoff_date: str   # EMA-Z normalization boundary (= val_start_date)

env:
  symbol: str
  initial_balance: float
  taker_fee: float
  num_envs: int           # Vectorized env count
  window_size: int
  max_drawdown_pct: float
  mdp_version: str        # "v5" | "v6" | "v7"
  episode_length: int
  random_start: bool
  action:
    discrete_dims: int    # V5/V6 only (2 for swing, 6 for legacy)
    cooldown_bars: int    # V6 only

network:
  micro_config:
    input_size: int
    private_input_size: int   # 4 for V6, 5 for V7
    hidden_size: int          # Must be multiple of 8 (Tensor Core alignment)

training:
  total_timesteps: int
  torch_compile: bool     # false for IQN (NoisyLinear incompatible)
  use_amp: bool

hpo:
  enabled: bool
  n_trials: int
  steps_per_trial: int

wandb:
  project: str
  entity: str
  tags: list              # Always include lowercase experiment ID (e.g. "k5")
```

**When unsure about a key, check an existing config in `configs/` — do NOT guess.**

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries. Never normalize across splits. |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params (gamma, sharpe_weight, hindsight_*) during HPO. |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting (uses future prices). |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| BUG-04 | Dense reward on switch bars must use direction BEFORE switch. Save `direction_for_reward` before action processing. |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |

## Coding Standards

- All `.to(device)` calls **must** use `non_blocking=True`
- Replay buffers **must** support batch push — no per-sample Python loops in hot paths
- New configs: `torch_compile: true`, `update_interval: 8` (tau auto-scales)
- Do NOT rewrite vectorized PER with per-element iteration
- Env returns raw numpy, not dicts — preserve this path
- All `Linear` layer hidden dims must be **multiples of 8** (Tensor Core alignment)
- Use `logging` or `MLOpsLogger` — never raw `print()` in production code

## Anti-Patterns (NEVER DO)

- **Never import from** `FinRLPodracer/` or `Podracer/`
- **Never use** `MultiDiscrete` action space — V5 uses `Discrete(6)` flat
- **Never normalize** across train/val/test splits (LEAK-1)
- **Never set** `hindsight_weight > 0` in backtest configs (BUG-03)
- **Never use** `torch_compile: true` with IQN/NoisyLinear — triggers recompilation storms
- **Never guess** config keys — read an existing YAML in `configs/` first
- **Never use** `mid_price` without validating high/low against open/close (data corruption risk)
- **Never skip** `n_step: 3` for IQN configs — essential for stability (Session 97 finding)
- **Never set** `private_input_size: 3` — it's **4** for V6 (changed in Tier 2), **5** for V7
- **Never revert** action space to `MultiDiscrete` or add Hold/Cancel/Maker actions to V6

## Agent Skills — Auto-Dispatch

Skills in `.agent/skills/`. Read the relevant `SKILL.md` before executing. **Trigger proactively** — don't wait for the user to ask.

| Skill | Trigger | Spec |
|-------|---------|------|
| **Audit** | **Auto** after ANY code change to `finrl_pro_ds/`, `scripts/`, `configs/`. Skip `.md`-only. | `.agent/skills/audit/SKILL.md` |
| **Deploy** | User requests GPU launch, instance management, or run deployment. | `.agent/skills/deploy/SKILL.md` |
| **Memory** | **Auto** at session start (boot) and end (`/sync`). Update `core.md` proactively on findings. Use `memory_search` MCP for semantic retrieval, grep on `randd_log.md` for exact tag matching. | `.agent/skills/memory/SKILL.md` |
| **Monitor** | Status checks, "how are runs", before deploying new runs, anomaly triage. `python scripts/monitor_fleet.py` | `.agent/skills/monitor/SKILL.md` |
| **Optimization** | SPS regression, low GPU util, new hardware, new training loop, perf tuning. Profile first (Phase 1). | `.agent/skills/optimization/SKILL.md` |
| **Math** | Manual ("check math", "verify formulas") + auto after changes to env/agent/feature code that touch formulas. | `.agent/skills/math/SKILL.md` |

**Chaining rules:**
- Code change → **Audit** (mandatory) → if perf-relevant → **Optimization** → if math-relevant → **Math**
- Deploy request → **Monitor** (check fleet) → **Deploy** → **Monitor** (verify)
- Session start → **Memory** boot (core.md loaded automatically) → `memory_search` MCP or grep `randd_log.md` for prior context
- Experiment result → **Memory** update `core.md` → append `randd_log.md` → git commit

## Memory Protocol (2-Tier + Cloud)

```
Tier 1: .agent/memory/core.md  — Project status (~100 lines, deterministic boot context)
Tier 2: randd_log.md            — R&D history (append-only, search via grep)
Cloud:  agent-memory MCP        — GCS LanceDB (326+ rows), semantic vector search via memory_search/memory_store
```

**Boot:** `core.md` (always loaded via system prompt hook). `memory_search` MCP or grep `randd_log.md` for prior context.
**Commit:** Append to `randd_log.md` → update `core.md` → `memory_store` key findings → git commit.
**Cloud:** agent-memory MCP server (LanceDB on `gs://openclaw-memory-lance/v1`, Gemini embeddings). Requires `GOOGLE_SERVICE_ACCOUNT` env var in `.mcp.json` pointing to `~/.openclaw/gcs-service-account.json`.
**Deprecated:** Daily logs (`.agent/memory/logs/`), `randd_archive.md`, snapshot rotation. No longer maintained.

## Gotchas (Last verified: 2026-03-13)

- `private_input_size: 4` for V6 SwingScalperEnv, `5` for V7 ContinuousSwingEnv (NOT 3)
- Action space: V6 `Discrete(2)`, V5 `Discrete(6)` flat — do NOT revert to MultiDiscrete
- HPO uses NopPruner, no early-kill, 500K steps/trial
- `n_step: 3` required for IQN stability — **all** IQN configs must include it
- RTX 5090 + CUDA 13.0 may segfault with `torch.compile`
- Taker fills same-bar; maker pends to next bar
- GMGP1 SAC: `torch_compile: true` OK (no NoisyLinear). IQN: `torch_compile: false` always.
- Gold data was corrupted (Session 106) — always validate via `scripts/clean_ohlcv.py` before use
