# Implementation Plan: DeepScalper Replication

## Goal Description
Replicate the **DeepScalper** framework (Deep Reinforcement Learning for Intraday Trading) as described in *Sun et al. (2022)*, integrated into the `FinRL-Pro_DS` ecosystem.

**Key Architecture Components:**
1.  **Multi-Modal Embedding**: Combining Micro-level (LOB) and Macro-level (OHLCV/Technicals) data.
2.  **Encoder-Decoder/Fusion**: Using LSTM for Micro sequence learning and MLP for Macro context.
3.  **Single BDQ Agent**: A standard Branching Dueling Q-Network processing the fused state.
4.  **High-Performance Data Pipeline**: Utilizing **Shared Memory (SHM)** and `fastparquet` for zero-copy access to massive LOB datasets.

## Plan vs. Paper: Fidelity & Upgrades
| Feature | DeepScalper Paper (Sun et al.) | FinRL-Pro Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Micro(LOB)+Macro(Tech) Fusion | Micro(LOB)+Macro(Tech) Fusion | 🟢 **Aligned** |
| **Action Space** | Discrete Branching (Dir/Price/Vol) | Discrete Branching (Dir/Price/Vol) | 🟢 **Aligned** |
| **Primary Algorithm** | Dueling DQN | Branching Dueling DQN | 🟢 **Aligned** |
| **Agent Strategy** | Single Agent | **Single BDQ Agent** | 🟢 **Aligned** |
| **Data Storage** | Flat Files (Implied) | **TimescaleDB** (High-Freq/Scalable) | 🚀 **Upgraded** |
| **Evaluation** | Custom Backtester | **Pyfolio** (Institutional Grade) | 🚀 **Upgraded** |
| **Monitoring** | Static Plots | **WandB** (Real-time Tracking) | 🚀 **Upgraded** |
| **Framework** | Plain PyTorch/Gym | **Gymnasium** + FinRL-Pro Ecosystem | 🚀 **Upgraded** |
| **Asset Universe** | N/A | **Bitcoin Perpetual Futures** (BTC-USDT) |  **Defined** |

## User Review Required
> [!IMPORTANT]
> **Asset Scope**: We are restricting training and testing **exclusively to Bitcoin Perpetual Futures**.
> We will NOT support the Chinese assets (CSI 300, etc.) from the original paper.
> - **Symbol**: `BTCUSDT` (Perpetual Contract)
> - **Exchange**: Binance Futures (Simulated via Kaggle Data/Live API)

> [!IMPORTANT]
> **Data Availability**: DeepScalper relies heavily on **Limit Order Book (LOB)** data (e.g., Level 2 data). Standard OHLCV data is insufficient for the "Micro" component.
> *Action*: We will implement a **Synthetic LOB Generator** for development, but real application requires high-frequency data sources.

> [!NOTE]
> **Library Choice**: We will build the network using **PyTorch** and integrate it as a custom Policy into **Stable Baselines3** (standard FinRL backend) or **CleanRL**, wrapped in the `FinRL-Pro` Agent interface.

> [!IMPORTANT]
> **Tooling Mandate**:
> - **Evaluation**: MUST use `pyfolio` (via `empyrical`) for all performance metrics and reporting (VectorBT removed).
> - **Monitoring**: MUST use `wandb` (Weights & Biases) for experiment tracking.
> - **Deployment**: Pipeline will deploy to `gpuhub` via `deploy_bare_metal.py`.
> - **Hardware**: Optimized for **NVIDIA RTX 5090 (Blackwell)** (TF32, AMP, 24 Envs).
> - **Data Source (Training)**: MUST use **Kaggle** (via CLI) to fetch High-Frequency LOB data.
> - **Data Source (Live)**: Use **Binance** via `python-binance` for real-time streams.
> - **Security**: API Credentials (Binance & Kaggle) must be loaded from `.env` (git-ignored).

## Non-Goals
-   **Ensemble Methods**: We strictly adhere to the **Single BDQ** architecture. Multi-agent ensembles are out of scope for V1.
-   **Legacy Formats**: We do not support CSV/Pandas for core training; Parquet/Numpy is mandatory for performance.

---

## DeepScalper Technical Specification
*Based on recent paper analysis.*

### 1. Training Architecture & Objectives

DeepScalper does not rely on a standard DQN loss alone. It employs a multi-objective loss function and a specialized training-only reward structure.

#### **A. The Network Architecture (BDQ)**
*   **Input Processing:** The agent uses an encoder-decoder architecture.
    *   **Micro-Level Encoder:** Processes Limit Order Book (LOB) data and the trader's private state (position, cash) using **LSTM** layers to capture immediate supply/demand pressure.
    *   **Macro-Level Encoder:** Processes OHLCV (Open, High, Low, Close, Volume) data and technical indicators using an **MLP** (Multilayer Perceptron) to capture broader trends.
    *   **Fusion:** These embeddings are concatenated to form the state embedding $e_t$.
*   **Action Branching:** The BDQ splits the Q-value estimation into two independent branches to avoid the combinatorial explosion of actions:
    *   **Price Branch:** Estimating $Q$ values for discrete price levels.
    *   **Quantity Branch:** Estimating $Q$ values for discrete quantity proportions.
*   **Dueling Mechanism:** Both branches share a common state-value stream $V(s)$ but maintain separate advantage streams $Adv(s, a)$.

#### **B. The Loss Function (Hybrid Learning)**
DeepScalper optimizes a composite loss function that combines the standard Reinforcement Learning (RL) loss with a risk-aware auxiliary task.
*   **Primary RL Loss ($L_q$):** This is the standard Mean Squared Error (MSE) between the predicted Q-values and the TD targets (using a target network and Prioritized Experience Replay).
*   **Auxiliary Loss ($L_{vol}$):** The model simultaneously predicts **future volatility** (standard deviation of returns). This is a supervised regression task using the market embedding $e_t$.
*   **Total Loss:**
    $$L = L_q + \rho \times L_{vol}$$
    Where $\rho$ is a weight parameter balancing profit seeking (RL) and risk awareness (volatility prediction).

#### **C. Reward Shaping: The Hindsight Bonus**
A critical innovation in DeepScalper's training is the **Hindsight Bonus**, designed to prevent the agent from becoming "short-sighted" (capturing tiny fluctuations while missing the day's major trend).
*   **Formula:**
    $$r_{hind} = r_t + w \times (p_{t+h} - p_t) \times \text{position}_t$$
    *   $r_t$: Immediate PnL minus transaction costs.
    *   $w$: Weight of the hindsight bonus.
    *   $h$: The look-ahead horizon (time steps into the future).
*   **Training vs. Testing:** This bonus is **only applied during training**. During testing/backtesting, the agent is evaluated solely on real realized PnL ($r_t$).

---

### 2. Hyperparameter Optimization (HPO) Setup

DeepScalper utilizes **Grid Search** rather than advanced Bayesian or Random search methods to tune its hyperparameters.

#### **A. The Search Space**
The authors explored the following specific grids for their hyperparameters:

| Hyperparameter | Description | Grid Search Values | Optimal Findings (approx.) |
| :--- | :--- | :--- | :--- |
| **$h$** | **Hindsight Horizon** | `` (minutes) | Performance peaked at **180**, then decreased. |
| **$w$** | **Hindsight Weight** | `[1e-3, 5e-3, 1e-2, 5e-2, 1e-1]` | **0.1** ($1e^{-1}$) achieved highest profit. |
| **$\rho$** | **Aux. Task Weight** | `[0.5, 1.0]` | Results were robust, but **1.0** is a decent start. |
| **Hidden Units** | Network Size (MLP/GRU) | `` | Not specified, dependent on asset complexity. |
| **$\alpha$** | **Learning Rate** | Range `(1e-5, 1e-3)` | Tuned per asset. |

#### **B. Training Execution Details**
*   **Hardware:** The training was performed on a **Tesla V100 GPU**.
*   **Duration:** The model was trained for **5 epochs** on each financial asset.
*   **Optimizer:** The **Adam** optimizer was used for updating weights.
*   **Data Augmentation:** To improve data efficiency and prevent overfitting, the trader's **private state** (account balance/position) was augmented repeatedly during training.
*   **Robustness:** Experiments were run with **5 different random seeds** to ensure that the reported performance was not due to lucky initialization.

### Summary of Key "Tricks"
1.  **Prioritized Experience Replay:** Used to sample important transitions more frequently.
2.  **Target Networks:** Updated recursively to stabilize Q-learning.
116.  **Risk-Awareness:** The agent doesn't just maximize profit; the auxiliary task forces the shared embedding layer to encode market volatility risk.
117. 
118. ---
119. 
120. ## Remote Deployment Standard
121. 
122. ### 1. Canonical Naming
123. All runs must follow the strict canonical format enforced by `finrl_pro_ds.utils.naming`:
124. `DeepScalper_V{version}_{Platform}_{YYYYMMDD}_{HHMM}`
125. -   **No Suffixes**: Metadata (e.g., "Pilot", "HPO") must be stored in WandB **Tags**, not the run name.
126. -   **Platform**: `GPUHub` or `Local`.
127. 
128. ### 2. Deployment Workflow
129. -   **Script**: `scripts/deploy_bare_metal.py`
130. -   **Orchestration**:
131.     1.  **Pack**: Creates a filtered zip (excluding `data/`, `wandb/`, etc.).
132.     2.  **Upload**: SCPs zip and data (with Gold Cache verification) to `/workspace/DeepScalper`.
133.     3.  **Setup**: Installs dependencies (PyTorch 2.5+, CUDA 12.x) in a fresh Miniconda env.
134.     4.  **Launch**: Executes `run_full_pipeline.py` or specific scripts with `nohup`.


---

## Implementation Roadmap (Revised)

### 0. Dependencies & Infrastructure
**File:** `pyproject.toml`
-   Add `vectorbt`, `wandb`, `python-binance`, `python-dotenv`, `kaggle`.
**File:** `scripts/deploy_gpuhub.py`
-   Standardized deployment script.

### 1. Data Engineering & Preprocessing
**File:** `finrl_pro_ds/data/feature_engineering.py`
-   **Micro-Features (LOB)**: LSTM-ready sequences.
-   **Macro-Features (Technicals)**: MLP-ready vector.
-   **Auxiliary Target Generation**: Pre-calculate "Future Volatility" for the $L_{vol}$ loss.

### 2. Neural Network Architecture (Single BDQ)
**File:** `finrl_pro_ds/agents/deepscalper/networks.py`
-   Implement `DeepScalperNetwork` class:
    -   `MicroEncoder` (LSTM)
    -   `MacroEncoder` (MLP)
    -   `FusionLayer` -> `FeatureExtractor`
    -   `ValueHead`
    -   `ActionHeads` (Price, Quantity, Direction?? Paper specifies Price/Quantity, implementation usually needs Direction too or encodes it).
    -   **Auxiliary Head**: Predicts volatility from fusion layer.

### 3. Training Logic (Custom Trainer)
**File:** `finrl_pro_ds/training/trainer.py`
-   **Custom Loss Calculation**:
    -   Calculate $L_q$ (TD Error).
    -   Calculate $L_{vol}$ (MSE between Aux Head and Actual Volatility).
    -   Combine: $loss = L_q + \rho L_{vol}$.
-   **Reward Shaping Integration**:
    -   Augment rewards in the Replay Buffer with $r_{hind}$.

### 4. Verification Plan
-   **Unit Test**: Verify Hindsight Bonus calculation matches formula.
-   **Unit Test**: Verify Aux Loss gradient flow.
-   **Integration**: Train Single Agent on Smoke Test.
-   **Validation**: Check if Aux Loss decreases alongside RL Loss.

---

## Implementation Updates Log

### 2026-02-02 11:45 | Blueprint Alignment to Paper
**Status:** ✅ Completed
-   Replaced experimental "Synapse Ensemble" with **Single BDQ Agent** as per *Sun et al.*.
-   Added **Hindsight Bonus** and **Auxiliary Loss** specifications.
-   Defined **HPO Grid** based on paper findings.

### 2026-02-01 11:05 | Ensemble Gating Weight Logging (Legacy)
*Note: This feature is relevant only if we revert to ensemble, keeping for history.*
**Commit:** `31aaf0c`

### 2026-02-01 02:00 | RTX 5090 Precision Optimization
**Status:** ✅ Implemented
-   TF32 Precision & FP16 Enforcement.
