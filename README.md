# DeepScalper

**High-Frequency Crypto Scalping with Deep Reinforcement Learning (Bitcoin Futures)**

DeepScalper is an end-to-end institutional-grade reinforcement learning pipeline designed for sub-second intraday trading. It leverages an ensemble of **PPO, A2C, and DQN** agents to trade on Limit Order Book (LOB) data with micro-structure awareness.

## 🚀 Key Features
- **5-Phase MLOps Pipeline**: Unified schema for Tuner $\to$ Trainer $\to$ Auditor $\to$ Deployer.
- **Mach 3 Optimization**: Optimized for RTX 5090 (32GB VRAM), utilizing Shared Memory, AMP, and Torch Compile for max throughput.
- **Hybrid Data Engine**: Syncs 1-minute OHLCV bars with 250ms LOB snapshots for realistic latency simulation.
- **Institutional Governance**: Automated "Smoke Tests", Financial Audits (Sharpe/Sortino), and Risk Guardrails.

---

## 🛠️ Installation

Requires **Python 3.10+** and a CUDA-capable GPU.

```bash
# 1. Clone & Setup
git clone https://github.com/bigcan/FinRL-Pro-DS.git
cd FinRL-Pro-DS
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # or source .venv/bin/activate

# 2. Install Core Dependencies
pip install -r requirements.txt
pip install -e .

# 3. (Optional) Install Flash Attention / Torch Compile support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚡ Quick Start

### 1. Unified Configuration
All parameters are controlled via `configs/deepscalper_unified.yaml`.
- **Mode**: User `production` for real runs, `smoke_test` for CI checks.
- **Hardware**: Toggle `use_amp`, `torch_compile` based on your GPU.

### 2. Automated Pipeline (Recommended)
Run the end-to-end MLOps pipeline (HPO $\to$ Train $\to$ Backtest $\to$ Report).
```bash
python scripts/run_full_pipeline.py --config configs/deepscalper_unified.yaml --trials 20
```

### 3. Deployment (Remote Ops)
Deploy the **full pipeline** to a remote node (e.g., GPUHub).
```bash
# Correctly launches the pipeline script, NOT just the trainer
python scripts/deploy_bare_metal.py --script scripts/run_full_pipeline.py --config configs/deepscalper_unified.yaml --run_name DS_Production_V1
```

### 4. Manual Component Execution (Advanced)
If you need to run specific stages manually:
**Train**:
```bash
python scripts/train_deepscalper.py --config configs/deepscalper_unified.yaml
```
**Tune**:
```bash
python scripts/tune_deepscalper.py --trials 50
```
**Audit**:
```bash
python scripts/audit_model.py --checkpoint checkpoints/best_model.pth
```

---

## 📂 Project Structure

```text
FinRL-Pro-DS/
??? configs/                # Unified YAML configurations
??? data/                   # Dataset storage (Parquet)
??? finrl_pro_ds/           # Core Package
?   ??? agents/             # RL Algorithms (PPO, DQN, A2C)
?   ??? envs/               # DeepScalperEnv (LOB-aware)
?   ??? mlops/              # Watchdog, Logger, Risk Guards
?   ??? training/           # Trainer, Accumulators
??? scripts/                # Entry points (train, tune, deploy, audit)
??? tests/                  # Pytest suite
??? checkpoints/            # Model artifacts
??? reports/                # Audit reports
```

## ⚠️ Risk Disclaimer
This software is for educational and research purposes only. **Deep Reinforcement Learning involves significant financial risk.** The authors are not responsible for trading losses.

---
*Based on "DeepScalper: A Deep Reinforcement Learning Health Indicator for Cryptocurrency High-Frequency Trading" (Sun et al., 2022).*
