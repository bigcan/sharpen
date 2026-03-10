# DeepScalper (FinRL-Pro_DS) — Gemini Instructional Context

This document provides essential context and instructions for AI agents working on the **DeepScalper** project, an institutional-grade high-frequency trading (HFT) reinforcement learning pipeline.

## 1. Project Overview
- **Purpose**: Profitable HFT on BTC/USDT and Gold Futures (CME) using Deep Reinforcement Learning.
- **Core Technology Stack**:
    - **Language**: Python 3.10+
    - **Deep Learning**: PyTorch (2.8+), Torch Compile, AMP (Mixed Precision).
    - **RL Frameworks**: Custom implementations of BDQ (Branching Dueling Q-Network), IQN (Implicit Quantile Network), and SAC (Soft Actor-Critic).
    - **Environment**: Gymnasium-based `DeepScalperEnv`.
    - **Data Handling**: Shared-memory `ParquetDataHandler` for high-throughput LOB/OHLCV data streaming.
    - **Experiment Tracking**: WandB (Weights & Biases).
    - **Hyperparameter Optimization (HPO)**: Optuna.

## 2. Architecture & Components
The project follows a modular design centered around `finrl_pro_ds/`:

- **`envs/`**: Core trading environments.
    - `DeepScalperEnv`: Primary HFT environment for BTC/Gold with LOB data.
    - `SwingScalperEnv`: Binary direction-switching MDP (Phase K).
    - `ContinuousSwingEnv`: Continuous position control for SAC (GMGP1).
- **`agents/`**: RL agent implementations.
    - `deepscalper/bdq_agent.py`: Multi-branch action space (Direction, Price, Volume).
    - `iqn_agent.py`: Distributional RL with NoisyNets (Stage 3 primary).
    - `ppo_agent.py`: On-policy baseline.
- **`data/`**: Feature engineering and data loaders.
    - `feature_engineering.py`: "Feature Factory" for Micro (LOB) and Macro (OHLCV) features.
    - `parquet_handler.py`: High-performance data streaming.
- **`configs/`**: Hydra/YAML configuration files governing all experiments.
- **`scripts/`**: Entry points for the pipeline (`run_full_pipeline.py`) and deployment.

## 3. Building and Running

### Setup
```bash
# Initialize environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Execution
- **Run Pipeline (Local/Dev)**:
  ```bash
  python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml
  ```
- **Deploy to Remote GPU (Production)**:
  ```bash
  python scripts/deploy_bare_metal.py --script scripts/run_full_pipeline.py --config configs/deepscalper_rtx5090_production.yaml --run_name DS_Prod_V1
  ```
- **Backtest Single Run**:
  ```bash
  python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml --mode backtest --checkpoint <path_to_ckpt>
  ```

### Testing
```bash
# Run pytest suite
python -m pytest tests/ -v
```

## 4. Development Conventions

### Coding Standards
- **Strict Typing**: Use Python type hints for all functions.
- **Docstrings**: Follow Google-style docstrings.
- **No `print()`**: Use `logging` or `MLOpsLogger`.

### RL & Optimization Guidelines
- **Tensor Core Alignment**: Ensure all `Linear` layer dimensions (hidden dims) are **multiples of 8** (e.g., 64, 128, 256, 512).
- **torch.compile**:
    - **Enable** for standard architectures (BDQ, PPO) using `mode="default"`.
    - **Disable** for agents using `NoisyLinear` (IQN) as it triggers frequent recompilation.
- **Mixed Precision**: Use `use_amp: true` and prefer **BF16** on RTX 4090/5090.
- **UTD Ratio**: Use `update_interval: 8.0` for production training to maximize GPU utilization.

### Data Integrity
- **Point-In-Time (PIT)**: All features must be engineered to avoid look-ahead bias.
- **Normalization**: Use `norm_cutoff_date` to prevent training data stats from leaking into val/test.

## 5. Agent Memory System
The project uses a persistent memory system in `.agent//`.
- **`core.md`**: The primary source of truth for project status, preferences, and active decisions. **Always read this first.**
- **`logs/`**: Daily logs of R&D activities.
- **Usage**: Use `/memory-boot` to load context and `//save` to persist session progress.

## 6. Key Files & Locations
- **Main Entry**: `scripts/run_full_pipeline.py`
- **Agent Code**: `finrl_pro_ds/agents/`
- **Env Code**: `finrl_pro_ds/envs/`
- **Feature Factory**: `finrl_pro_ds/data/feature_engineering.py`
- **Active Research Plan**: `.agent/artifacts/stage3_research_plan.md`
- **R&D Log**: `randd_log.md`
