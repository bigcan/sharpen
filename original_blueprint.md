# Implementation Plan: DeepScalper Replication

## Goal Description
Replicate the **DeepScalper** framework (Deep Reinforcement Learning for Intraday Trading) as described in *Sun et al. (2022)*, integrated into the `FinRL-Pro_DS` ecosystem.

**Key Architecture Components:**
1.  **Multi-Modal Embedding**: Combining Micro-level (LOB) and Macro-level (OHLCV/Technicals) data.
2.  **Encoder-Decoder/Fusion**: Using LSTM for Micro sequence learning and MLP for Macro context.
3.  **Intraday Environment**: Simulation of order book dynamics and execution costs.

## Plan vs. Paper: Fidelity & Upgrades
| Feature | DeepScalper Paper (Sun et al.) | FinRL-Pro Synapse Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Micro(LOB)+Macro(Tech) Fusion | Micro(LOB)+Macro(Tech) Fusion | 🟢 **Aligned** |
| **Action Space** | Discrete Branching (Dir/Price/Vol) | Discrete Branching (Dir/Price/Vol) | 🟢 **Aligned** |
| **Primary Algorithm** | Dueling DQN | Branching Dueling DQN | 🟢 **Aligned** |
| **Agent Strategy** | Single Agent | **Synapse Dynamic Ensemble** (DQN+PPO+A2C) | 🚀 **Upgraded** |
| **Data Storage** | Flat Files (Implied) | **TimescaleDB** (High-Freq/Scalable) | 🚀 **Upgraded** |
| **Evaluation** | Custom Backtester | **VectorBT** (Institutional Grade) | 🚀 **Upgraded** |
| **Monitoring** | Static Plots | **WandB** (Real-time Tracking) | 🚀 **Upgraded** |
| **Framework** | Plain PyTorch/Gym | **Gymnasium** + FinRL-Pro Ecosystem | 🚀 **Upgraded** |
| **Asset Universe** | N/A | **Bitcoin Perpetual Futures** (BTC-USDT) | � **Defined** |

## User Review Required
> [!IMPORTANT]
> **Asset Scope**: We are restricting training and testing **exclusively to Bitcoin Perpetual Futures**.
> We will NOT support the Chinese assets (CSI 300, etc.) from the original paper.
> - **Symbol**: `BTCUSDT` (Perpetual Contract)
> - **Exchange**: Binance Futures (Simulated via Kaggle Data/Live API)

> [!IMPORTANT]
> **Ensemble Strategy**: We are adopting a **Synapse Dynamic Ensemble** approach.
> The final decision will be an aggregation of **Branching DQN** (Micro-structure expert), **PPO** (Stability anchor), and **A2C** (Trend follower), weighted dynamically by a **Softmax Gating Network** based on market context.

> [!IMPORTANT]
> **Data Availability**: DeepScalper relies heavily on **Limit Order Book (LOB)** data (e.g., Level 2 data). Standard OHLCV data is insufficient for the "Micro" component.
> *Action*: We will implement a **Synthetic LOB Generator** for development, but real application requires high-frequency data sources.

> [!NOTE]
> **Library Choice**: We will build the network using **PyTorch** and integrate it as a custom Policy into **Stable Baselines3** (standard FinRL backend) or **CleanRL**, wrapped in the `FinRL-Pro` Agent interface.

> [!IMPORTANT]
> **Tooling Mandate**:
> - **Evaluation**: MUST use `vectorbt` for all performance metrics and reporting.
> - **Monitoring**: MUST use `wandb` (Weights & Biases) for experiment tracking.
> - **Deployment**: Pipeline will deploy to `gpuhub` for training and testing.
> - **Data Source (Training)**: MUST use **Kaggle** (via CLI) to fetch High-Frequency LOB data (Option 1: `siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data`).
> - **Data Source (Live)**: Use **Binance** via `python-binance` for real-time streams.
> - **Security**: API Credentials (Binance & Kaggle) must be loaded from `.env` (git-ignored).

## Non-Goals
-   **Baseline Models**: We will **skip** implementation of separate, non-fusion baselines. The ensemble *is* the model.
-   **MLflow**: We are deprecating MLflow in favor of WandB for this module.

## Proposed Changes

### 0. Dependencies & Infrastructure
**File:** `pyproject.toml`
-   Add `vectorbt`, `wandb`, `python-binance`, `python-dotenv`, `kaggle` to dependencies.
**File:** `scripts/deploy_gpuhub.py`
-   Create standardized deployment script for GPUHub execution.
**File:** `.env`
-   Store `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY` safely.

### 0.5. Database Integration
**File:** `finrl_pro_ds/data/db.py`
-   **Schema Update:** Add `lob_snapshots` and `lob_data` tables to support high-frequency LOB data (Time, Ticker, Level, BidPrice, BidVol, AskPrice, AskVol).
**File:** `scripts/reset_db_schema.py`
-   Create script to drop/re-create LOB tables.
**File:** `finrl_pro_ds/data/handler.py`
-   Implement `DBMarketDataHandler` to fetch/stream data from TimescaleDB into the Gym Env.

### 0.6. Clean Slate & Scaffolding
-   **Operations**: Pipe hygiene script (`scripts/clean.py`).
-   **Structure**:
    -   `finrl_pro_ds/agents/deepscalper/`: Core model logic.
    -   `configs/deepscalper.yaml`: Centralized configuration.
    -   `tests/deepscalper/`: dedicated test suite.

### 1. Data Engineering & Preprocessing
**File:** `finrl_pro_ds/data/feature_engineering.py`
-   **Micro-Features (LOB)**:
    -   **Normalization**: Convert absolute prices to *relative percentage changes* (Log Returns) or *spread-relative* levels.
    -   **Order Flow Imbalance (OFI)**: Calculate *Aggressor Trade* Delta (Active Buying - Active Selling) vs. Passive Depth.
    -   **Crypto-Specifics**:
        -   **Funding Rates**: Include current funding rate and countdown (critical for perp holding costs).
        -   **Open Interest (OI)**: Track OI changes to detect liquidation cascades.
    -   **Stationarity**: Ensure all inputs are stationary (z-score normalization using rolling window).
-   **Macro-Features (Technicals)**:
    -   **Library**: Use `pandas-ta` to generate standard indicators.
    -   **Indicators**: RSI (14), MACD, Bollinger Bands, ATR (Volatility), OBV (Volume).
    -   **Horizon**: Multiple timeframes (1min, 5min, 15min) to capture broader trends.
-   **Multi-Modal Alignment**:
    -   **Frequency Mismatch**: Micro data is tick/snapshot level (~100ms), Macro is bar level (1min).
    -   **Strategy**: *Forward Fill* Macro features. Every Micro snapshot sees the most recent completed Macro bar's features.

### 2. New Environment Module
**File:** `finrl_pro_ds/envs/deep_scalper_env.py`
-   **Class:** `DeepScalperEnv(gymnasium.Env)`
-   **State Space:** Dictionary space `{ 'micro': (T, Levels, 4), 'macro': (Features,), 'private': (Pos, cash) }`.
-   **Action Space:** Discrete (0: Hold, 1: Buy, 2: Sell) or Continuous (if adapted).
-   **Reward:** PnL - **Maker/Taker Fees** - Funding Cost - Volatility Penalty.
    -   *Crucial Update*: Differentiate Maker (0.02%) vs Taker (0.04%) fee application.
-   **Execution Logic (Hardened)**:
    -   **Level Crossing**: For limit orders, only fill if `Ask Px < Limit Px` (Conservative). No "Touch Fills".
    -   **Latency Simulation**: Implement `t+1` tick execution delay to prevent lookahead bias.
    -   **Order Manager**: Logic to handle "Modify" (Cancel-Replace) vs "New" orders efficiently.
-   **Safety**: Circuit breakers for Max Drawdown and Position Size limits.
-   **Integration:** Must accept a `DatabaseClient` to stream LOB data.

### 3. Neural Network Architecture
**File:** `finrl_pro_ds/agents/deepscalper/networks.py`
#### [NEW] Component: Micro-Encoder
-   **Input:** LOB Snapshot Sequence `(Batch, Time, Levels * 4)`.
-   **Layer:** LSTM or GRU.
-   **Output:** Micro-Feature Vector `h_micro`.

#### [NEW] Component: Macro-Encoder
-   **Input:** Technical Indicators `(Batch, Features)`.
-   **Layer:** MLP (Dense Layers).
-   **Output:** Macro-Feature Vector `h_macro`.

#### [NEW] Component: Fusion Policy & Action Branching
-   **Input:** `h_micro`, `h_macro`.
-   **Fusion:** Concatenation -> Dense Layers -> Shared Representation `h_fusion`.
-   **Action Branching Heads:** The paper uses 3 independent decision branches:
    1.  **Direction Head**: 3 outputs (Buy, Sell, Hold).
    2.  **Price Head**: 5 outputs (Limit Price offsets).
    3.  **Quantity Head**: 5 outputs (Volume proportions, e.g., 10%, 25%, 50%, 75%, 100%).
-   **Output:** Separate Q-values for each branch (Total Actions = 3 + 5 + 5 = 13 outputs vs 3x5x5 = 75 in flat space).

### 4. Agent Ensemble (The "Scalper Squad")
**File:** `finrl_pro_ds/agents/deepscalper/ensemble.py`
-   **Class:** `DeepScalperEnsemble`
-   **Strategy:** **Synapse Dynamic Weighting** (Meta-Controller).
-   **Sub-Agents:**
    1.  **Branching Dueling DQN**: (The Expert) Limit Order specialist.
    2.  **PPO (Multi-Discrete)**: (The Anchor) Stability specialist.
    3.  **A2C (Multi-Discrete)**: (The Trend Follower) Volatility specialist.
-   **Aggregation Mechanism: Synapse Gating Network**
    -   **Input:** `Macro Context` (Technical Indicators).
    -   **Normalization (CRITICAL FIX)**:
        -   **DQN**: Apply `Softmax(Q_values / Temp)` to convert Q-values to pseudo-probabilities.
        -   **PPO/A2C**: Use Policy Probabilities directly.
    -   **Meta-Learner:** A small MLP that outputs a **Softmax** vector `[w_dqn, w_ppo, w_a2c]`.
    -   **Logic:** `Final_Probs = w_dqn * Probs_DQN + w_ppo * Probs_PPO + w_a2c * Probs_A2C`.

### 5. Training Pipeline
**File:** `scripts/train_ensemble.py`
-   **Strategy: Two-Phase Training (Stability)**
    -   **Phase 1 (Specialists)**: Train DQN, PPO, A2C independently on the environment. Freeze weights.
    -   **Phase 2 (Meta-Controller)**: Train ONLY the Gating Network (MLP) to select/weight the best specialist for the current state.
    -   **Evaluate:** VectorBT backtest of the Ensemble on unseen test data.

## Verification Plan

### 1. Automated Unit Tests
-   **Shape/PASS Test:** Verify the network accepts the Dict observation and outputs correct action shapes.
-   **Ensemble Voting Test:** Feed conflicting inputs to mock agents and verify the voting logic (Soft Majority) works as expected.
-   **Gradient Check:** Ensure gradients flow to both LSTM (Micro) and MLP (Macro) weights.

### 2. Pipeline Integration Tests (End-to-End)
-   **Data Ingestion Test**: Run `scripts/ingest_kaggle_lob.py` on a small sample, query TimescaleDB to verify correct schema and data integrity.
-   **Environment Streaming Test**: Initialize `DeepScalperEnv` connected to DB, step through 100 ticks, verify `t+1` latency logic and reward calculations.
-   **Training Smoke Test**: Run `scripts/train_ensemble.py` for 100 steps (1 epoch) with mock data. Verify:
    -   Weights update.
    -   WandB logs are created.
    -   Checkpoints are saved.
-   **Evaluation Pipeline Test**: Load a saved checkpoint, run `vectorbt` backtest, and generate an HTML report. Verify no `NaN` metrics.

### 3. Manual Verification
-   **Review Learning Curves:** Check for convergence (increasing Reward, decreasing Loss).
-   **Inspect Latency:** Ensure LOB processing is efficient enough for training (>100 steps/sec).

## Definition of Done (Post-Coding)
> [!IMPORTANT]
> The implementation of any phase is NOT complete until the following are executed:

1.  **Documentation Update**:
    -   Update repository root `README.md` to list the new `DeepScalper` module.
    -   Create/Update `finrl_pro_ds/agents/deepscalper/README.md` explaining the package structure.
2.  **Task Tracking**:
    -   Update `task.md` to mark completed items as `[x]`.
3.  **Hygiene**:
    -   Run `scripts/clean.py` to remove `__pycache__`, temporary logs, and artifacts.
    -   Ensure no sensitive API keys were accidentally committed (check `.env` usage).

---

## Implementation Updates Log

> [!NOTE]
> This section tracks significant updates to the DeepScalper pipeline implementation.

### 2026-02-01 11:05 | Ensemble Gating Weight Logging
**Commit:** `31aaf0c`
| File | Change |
|------|--------|
| `ensemble.py` | `predict()` now returns `(action, weights_dict)` |
| `backtest_deepscalper.py` | Adds `weight_dqn`/`weight_ppo`/`weight_a2c` columns to results |
| `wandb_evaluator.py` | New "Gating Weights" stacked area chart |

**Self-Audit Fixes:** Removed duplicate `self.device`, added defensive array checks, fixed PPO/A2C init args.

---

### 2026-02-01 10:00 | System Stabilization & Architecture Verification
**Status:** ✅ Mission Critical Success

| Fix | File | Description |
|-----|------|-------------|
| "Doom Loop" Resolution | `deepscalper_trainer.py` | Rewrote `evaluate()` with robust episode counting (handles `VectorEnv` auto-resets) |
| "Zombie Process" Killer | `deploy_bare_metal.py` | Aggressive `pkill -f` for all DeepScalper scripts before deployment |

**Architecture Verified:**
- Two-Phase Training (Specialists → Gating) correctly orchestrated in `run_full_pipeline.py`
- Weight freezing logic confirmed in `deepscalper_trainer.py`
- HPO defaults to `phase="full"` (Joint Optimization)

---

### 2026-02-01 02:00 | RTX 5090 Precision Optimization
**Status:** ✅ Implemented

| Setting | Implementation |
|---------|----------------|
| TF32 Precision | `torch.set_float32_matmul_precision('high')` |
| FP16 Enforcement | Explicit `autocast(..., dtype=torch.float16)` in update loops |
| Batch Scaling | Increased to `16,384` for GDDR7 bandwidth |

---

### 2026-01-30 13:00 | WandB Logging Audit Fixes
**Status:** ✅ All Critical Bugs Resolved

| Bug | File | Resolution |
|-----|------|------------|
| Duplicate PPO Loss | `deepscalper_trainer.py` | Removed duplicate `total_loss` accumulation |
| A2C NaN Return | `deepscalper_trainer.py` | Changed to `return None` on non-finite loss |
| Redundant Gating Logic | `deepscalper_trainer.py` | Simplified metric append |

---

### 2026-01-30 04:00 | Checkpoint Versioning Fix
**Status:** ✅ Fixed
- `DeepScalperTrainer` now creates unique checkpoint directories based on run name.
- Prevents data overwrites between runs.

---

### 2026-01-30 02:30 | V9.5 Replay Ratio Stabilization
**Status:** ✅ Fixed
- **Root Cause:** `dqn_update_interval: 0.5` with `num_envs: 24` caused 48 updates/step (Replay Ratio ~128).
- **Fix:** Increased `dqn_update_interval` to `8.0` for stable learning.
