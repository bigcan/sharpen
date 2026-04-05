# DeepScalper (FinRL-Pro_DS) — Gemini Instructional Context

This document provides foundational mandates and technical specifications for Gemini CLI when working on the **DeepScalper** project.

## 1. Project Brief & Status
- **Goal**: Profitable RL quant trading across asset classes (BTC, Gold, Crypto Perps, Funding Arb).
- **Active Agent**: **SAC only** (Implicit Quantile Network (IQN), Branching Dueling Q-Network (BDQ), and PPO are falsified/legacy).
- **Primary Workstreams**: GMGP1 SAC Gold 15m, Sync-1H crypto, Funding-Arb, Market Making LOB.
- **Reference State**: `.agent/memory/core.md` (Read at boot for active decisions).

## 2. Technical Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.8+ (BF16 AMP, `torch.compile`).
- **RL Framework**: Custom SAC implementation (Continuous position control).
- **Quality**: Ruff (Linting), Mypy (Types), Pytest.
- **Ops**: Docker, Prometheus, Grafana, WandB, Optuna.
- **Data**: Parquet (Shared-memory streaming).

## 3. Core Mandates & Anti-Patterns

### Critical Invariants
| ID | Rule |
|----|------|
| **LEAK-1** | Reset EMA-Z normalization at train/val/test splits. Never normalize across boundaries. |
| **BUG-01** | HPO objective = `profit_factor`. Lock reward params during HPO. |
| **BUG-03** | `hindsight_weight` must be `0.0` during backtesting (prevents look-ahead). |
| **BUG-04** | Dense rewards on switch bars must use direction BEFORE switch. |
| **SHORT-ACCT**| Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| **MARGIN-CFG** | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| **DATA-CLEAN** | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. |

### Anti-Patterns (NEVER DO)
- **Never import** from `FinRLPodracer/` or `Podracer/`.
- **Never guess** config keys — read a reference YAML in `configs/` first.
- **Never claim** a file or symbol exists without verifying via `grep_search` or `glob`.
- **Never use** `mid_price` without validating high/low against open/close.
- **Never set** `hindsight_weight > 0` in backtest configs.
- **Never skip** Math skill verification on formula/equation changes.

## 4. Coding Standards
- **Tensor Core Alignment**: All `Linear` layer hidden dims must be **multiples of 8** (e.g., 128, 256).
- **Efficient Tensors**: All `.to(device)` calls **must** use `non_blocking=True`.
- **Performance**: Use `update_interval: 8` for production (Tensor Core utilization).
- **Vectorization**: No per-sample Python loops in hot paths (Replay Buffer, Reward logic).
- **Logging**: Use `logging` or `MLOpsLogger`. Never use raw `print()`.

## 5. Project Map (Active vs. Legacy)
- **`finrl_pro_ds/agents/sac/`**: **ACTIVE** (SAC Agent & Networks).
- **`finrl_pro_ds/agents/deepscalper/`**: **LEGACY** (Do not extend).
- **`finrl_pro_ds/envs/`**:
    - `continuous_swing_env.py` (V7): **ACTIVE** (Continuous positions).
    - `market_making_env.py` (V8): **ACTIVE** (MM with LOB).
    - `deep_scalper_env.py` (V5): **LEGACY** (Discrete actions).
- **`finrl_pro_ds/crypto/`**: **ACTIVE** (Sync-1H, Funding Arb, Live Engine).
- **`finrl_pro_ds/data/multiscale_handler.py`**: **ACTIVE** (SAC/GMGP1 scaling).

## 6. Env Contracts (ACTIVE)

### ContinuousSwingEnv (V7)
- **Action**: `Box(-1, 1, (1,))` -- target position fraction.
- **Obs**: `Dict{ scale_0: (W, F), ..., private: (5,) }`.
- **Private**: `[current_position, unrealized_pnl_norm, time_sin, time_cos, atr_ratio]`.

### MarketMakingEnv (V8)
- **Action**: `Box(-1, 1, (3,))` -- (spread_offset, inventory_skew, quote_intensity).
- **Obs**: `Dict{ ..., lob: (W, n_lob), private: (12,) }`.
- **Reward**: DSR on spread_capture + MtM - inventory_penalty - fees.

### CryptoPerpEnv / FundingArbEnv
- **Action**: `Box(-1, 1, (n_assets,))`.
- **Reward**: Sortino (Perp) / PV-return + delta penalty (Funding).

## 7. Config Schema & References

| Pipeline | Reference Config | Top-Level Sections |
|----------|-----------------|---------------------|
| GMGP1 (V7) | `configs/gmgp1_sac_gc_15min.yaml` | data, features, env, network, agents.sac, training, hpo, wandb |
| Sync-1H | `configs/synapse_crypto_1h_v2.yaml` | strategy, universe, environment, agents, arbitrator, risk, execution |
| Live Trading | `configs/live_gmgp1_btc_bybit.yaml` | exchange, agent, network, features, bar_clock, trading, risk, prism |

## 8. Execution & Monitoring

### Common Commands
```bash
# Pipeline: HPO -> Train -> Backtest
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml

# Deployment to Remote GPU
python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> --collect

# Monitoring
python scripts/monitor_fleet.py              # Check all instances
python scripts/auto_collect_checkpoints.py   # Secure checkpoints to local/GCS

# Quality Checks
ruff check finrl_pro_ds && mypy finrl_pro_ds --ignore-missing-imports && pytest
```

### Docker & Live Trading
Live trading runs on remote desktop (`<TAILSCALE_HOST>`) using Docker contexts.
- **Manage**: `./scripts/manage_strategies.sh {build|up|ps|logs} <target>`
- **Stack**: Prometheus (9090), Grafana (3000), PRISM API (8001).

## 9. PRISM & Monitoring Architecture
- **PRISM**: Probabilistic Regime-Informed System for Markets. Provides L2 position sizing via GAHMM (regime detection). Multipliers: LOW_VOL (1.3), NORMAL_VOL (1.0), HIGH_VOL (0.3).
- **Monitoring**: `TradingMetrics` (Prometheus) runs on a daemon thread, decoupled from the trading loop.

## 10. Gotchas & Verification
- **HPO**: Uses `NopPruner`, 500K steps/trial. Objective is `profit_factor`.
- **Hardware**: RTX 5090 + CUDA 13.0. Run `scripts/patch_torch_compile.py` on new setups.
- **Fills**: Taker (same-bar), Maker (next-bar).
- **VRAM**: Check VRAM (not run count) for concurrent GPU runs.
- **Windows Docker**: Volume mounts fail; use baked `COPY` in Dockerfiles.
- **Verification**: Never claim a file or flag exists without checking. Grep first.

## 11. Memory Protocol
Gemini MUST use the 2-Tier memory system:
1. **Tier 1 (`.agent/memory/core.md`)**: High-level status and active artifacts. Read first.
2. **Tier 2 (`randd_log.md`)**: Detailed R&D log. Update after each significant finding.
3. **Sync**: Perform `/memory-boot` to load context and ensure `core.md` is updated before ending a session.
