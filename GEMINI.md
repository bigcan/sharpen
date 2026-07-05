# DeepScalper (FinRL-Pro_DS) — Gemini Instructional Context

This document provides foundational mandates and technical specifications for Gemini CLI when working on the **DeepScalper** project.

> **Last updated: 2026-06-30.** Canonical project state lives in `CLAUDE.md` (Project Brief) and `.agent/memory/core.md` — treat those as authoritative over this guide.

## 0. Prime Directive
- **Primary Role**: Your prime directive is to **review, audit, and advise Claude Code's work**, and to help Claude accomplish its missions on this project.
- **Code Edits**: You must **make NO code edits** unless you are given clear permissions.

## 1. Project Brief & Status
- **Goal**: Profitable RL quant trading across asset classes (BTC, Gold, Crypto Perps, Funding Arb).
- **Active Agent**: **SAC only** (Implicit Quantile Network (IQN), Branching Dueling Q-Network (BDQ), and PPO are present but none profitable yet).
- **Live Direction**: Sole live edge = **cross-asset TSMOM** (linear time-series momentum across ~18 ETFs / 4 asset classes, net Sharpe ~0.60, low SPY correlation), currently gated at paper (DSR 0.918 < 0.95). Active R&D thrust = **Crucible** (`finrl_pro_ds/crucible/`, `crucible-v2.8`) — continuous agentic alpha-mining funnel (free-data connectors → pre-registered hypotheses → deflated eval → forward lockbox), built on the `finrl_pro_ds/signals/` DSL/eval funnel.
- **Other Workstreams**: GMGP1 SAC Gold 15m (FTMO/Velotrade contender, paper). **Retired/shelved** (do NOT present as active): Sync-1H crypto (pilot failure — retired), Funding-Arb (SHELVED, re-run only if funding > 8%/yr), MM-SAC / V8 (retired S442), PRISM (falsified), AlphaSeek (terminated NO-GO). The Polymarket prediction-market research thread has moved to its own repo, `Chiwin-Technology/polymarket-updown-research` — do not present it as present here.
- **Reference State**: `.agent/memory/core.md` (Read at boot for active decisions). R&D log: `randd_log.md`.

## 2. Technical Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.8+ (BF16 AMP, `torch.compile`).
- **RL Framework**: Custom SAC implementation (Continuous position control).
- **Quality**: Ruff (Linting), Mypy (Types), Pytest.
- **Ops**: Docker, Prometheus, Grafana, WandB, Optuna.
- **Data**: Parquet (Shared-memory streaming).
- **WandB**: `entity=bigcan-chiwin-technology`, `project=FinRL-Pro-DS`. Always pass `metric_keys=` explicitly (e.g. HPO: `_debug/eval_profit_factor`, Backtest: `Profit_Factor_Daily`).

## 3. Core Mandates & Anti-Patterns

### Project Boundary
Only modify `finrl_pro_ds/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.

### Critical Invariants
| ID | Rule |
|----|------|
| **LEAK-1** | Reset EMA-Z normalization at train/val/test splits. Never normalize across boundaries. |
| **BUG-01** | HPO objective = `profit_factor`. Lock reward params during HPO. |
| **BUG-03** | `hindsight_weight` must be `0.0` during backtesting (prevents look-ahead). |
| **BUG-04** | Dense rewards on switch bars must use direction BEFORE switch. Save `direction_for_reward` before action processing. |
| **SHORT-ACCT**| Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| **MARGIN-CFG** | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| **DATA-CLEAN** | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| **PF-XCHECK** | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |

### Anti-Patterns (NEVER DO)
- **Never import** from `FinRLPodracer/` or `Podracer/`.
- **Never normalize** across train/val/test splits (LEAK-1).
- **Never set** `hindsight_weight > 0` in backtest configs (BUG-03).
- **Never guess** config keys — read a reference YAML in `configs/` first.
- **Never claim** a file or symbol exists without verifying via `grep_search` or `glob`.
- **Never use** `mid_price` without validating high/low against open/close.
- **Never deploy** without running `monitor_fleet.py` first (VRAM, active processes).
- **Never skip** Math skill verification on formula/equation changes.
- **Never launch a fused HPO+train+eval pipeline.** All training follows Protocol v2.
- **Never hardcode gate thresholds in code or scripts.** All numeric gates live in `configs/<workstream>.gates.yaml`.

## 4. Coding Standards
- **Tensor Core Alignment**: All `Linear` layer hidden dims must be **multiples of 8** (e.g., 128, 256).
- **Efficient Tensors**: All `.to(device)` calls **must** use `non_blocking=True`.
- **Performance**: Use `update_interval: 8` for production (Tensor Core utilization) and `torch_compile: true`.
- **Vectorization**: No per-sample Python loops in hot paths (Replay Buffer, Reward logic). Replay buffers must support batch push.
- **Logging**: Use `logging` or `MLOpsLogger`. Never use raw `print()`.

## 5. Project Map (Active vs. Legacy)
- **`finrl_pro_ds/agents/sac/`**: **ACTIVE** (SAC Agent & Networks).
- **`finrl_pro_ds/agents/deepscalper/`**: **LEGACY** (Do not extend).
- **`finrl_pro_ds/envs/`**:
    - `continuous_swing_env.py` (V7): **ACTIVE** (Continuous positions).
    - `market_making_env.py` (V8): **RETIRED** (S442 - MM-SAC workstream closed).
    - `deep_scalper_env.py` (V5) / `swing_scalper_env.py` (V6): **LEGACY** (Discrete actions, do not modify action spaces).
- **`finrl_pro_ds/crypto/`**: Live Engine **ACTIVE**; Sync-1H **RETIRED** (pilot failure) and Funding-Arb **SHELVED** (re-run only if funding > 8%/yr).
- **`finrl_pro_ds/signals/`**: **ACTIVE** (alpha-mining DSL + deflated 6-tier evaluation funnel — dependency layer under Crucible).
- **`finrl_pro_ds/crucible/`**: **ACTIVE** (`crucible-v2.8` — continuous agentic alpha-mining: data connectors, agentic proposer/author, orchestrator, lockbox, governance. See `docs/claude_md_reference.md`).
- **`finrl_pro_ds/data/multiscale_handler.py`**: **ACTIVE** (SAC/GMGP1 scaling).

## 6. Env Contracts (ACTIVE)
*All envs return raw numpy dicts, NOT Gymnasium wrappers — preserve this path.*

### ContinuousSwingEnv (V7)
- **Action**: `Box(-1, 1, (1,))` -- target position fraction.
- **Obs**: `Dict{ scale_0: (W, F), ..., private: (5,) }`.
- **Private**: `[current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]`.
- **Reward**: DSR reward, deadband 0.25.

### CryptoPerpEnv / FundingArbEnv
- **Action**: `Box(-1, 1, (n_assets,))`.
- **CryptoPerp**: Flat obs ~962 dims (20 assets). Sortino.
- **FundingArb**: Flat ~326 dims. PV-return + delta penalty + turnover penalty.

## 7. Config Schema & References
Configs vary by pipeline. Do NOT invent keys — read a reference config first.

| Pipeline | Reference Config | Top-Level Sections |
|----------|-----------------|---------------------|
| GMGP1 (V7) | `configs/gmgp1_sac_gc_15min.yaml` | data, features, env, network, agents.sac, training, hpo, wandb |
| Sync-1H | `configs/synapse_crypto_1h_v2.yaml` | strategy, universe, environment, agents, arbitrator, risk, execution |
| Funding Arb | `configs/funding_arb_sac_10assets_hpo.yaml` | |
| Live Trading | `configs/live_gmgp1_btc_bybit.yaml` | exchange, agent, network, features, bar_clock, trading, risk, prism |

## 8. Training Protocol v2 (Mandatory)
All training work uses the staged protocol in `docs/protocol_v2.md`. 
Six stages: data-prep → hpo → l1-multiseed → walk-forward (+stress) → recent-oos (+compliance) → paper-deploy. 
One stage = one WandB run = one decision artifact.

**Before launching any training run:**
1. Run `python scripts/validate_config.py --config <cfg> --stage <stage>` — exits non-zero on protocol violations.
2. Confirm upstream manifest `status == "PASS"` for any `--upstream-run` references.
3. Off-policy resume (SAC, IQN) requires upstream `outputs.replay_buffer` — cold-buffer resume rejected unless `--allow-cold-replay` is set.

*Bare `run_full_pipeline.py` without `--stage` is a v2 violation.*

## 9. Execution & Monitoring

### Common Commands
```bash
# Pipeline: HPO -> Train -> Backtest (Must specify stage in V2)
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage <stage>

# Crypto
python scripts/crypto_hpo_runner.py --config <cfg> [--warm_start --max_windows N]
python scripts/funding_arb_hpo_runner.py --config <cfg>

# Deployment to Remote GPU
python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> --collect

# Monitoring
python scripts/monitor_fleet.py              # Check all instances
python scripts/monitor_run.py --run_id <ID>
python scripts/collect_run.py --run_id <ID>
python scripts/auto_collect_checkpoints.py   # Secure checkpoints to local/GCS

# Quality Checks
ruff check finrl_pro_ds && mypy finrl_pro_ds --ignore-missing-imports && pytest
```

### Docker & Live Trading
Live trading containers run on remote desktop (`<TAILSCALE_HOST>`) via Docker context `finrl-desktop`.
- **Manage**: Always use `./scripts/manage_strategies.sh {build|up|ps|logs} <target>` or `docker --context finrl-desktop`.
- **Observability**: Prometheus (9090), Grafana (3000), Watchdog Telegram, per-strategy metrics (9101-9107).

## 10. PRISM Architecture
- **PRISM**: Probabilistic Regime-Informed System for Markets. 
- **Status**: **Falsified (S413+)**. `prism.enabled: false` in all configs. Containers still deployed.

## 11. Gotchas & Verification
- **HPO**: Uses `NopPruner`, no early-kill, 500K steps/trial. Objective is `profit_factor`.
- **Hardware**: RTX 5090 + CUDA 13.0. Run `scripts/patch_torch_compile.py` on fresh deployments.
- **Fills**: Taker (same-bar), Maker (next-bar).
- **VRAM**: Check VRAM (not run count) for concurrent GPU runs — 2+ runs can share 1 GPU.
- **Windows Docker**: File bind mounts fail silently — use baked `COPY` in Dockerfiles.
- **Verification**: Never claim a file or flag exists without checking. Grep first.
- **Legacy Envs**: Do NOT modify action spaces.

## 12. Memory Protocol
Gemini MUST use the 2-Tier memory system:
1. **Tier 1 (`.agent/memory/core.md`)**: High-level status and active artifacts. Read first.
2. **Tier 2 (`randd_log.md`)**: Detailed R&D log. Update after each significant finding.
3. **Sync**: Perform `/memory-boot` to load context and ensure `core.md` is updated before ending a session.
