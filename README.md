# DeepScalper

**High-Frequency Crypto Scalping with Deep Reinforcement Learning (Bitcoin Futures)**

DeepScalper is an end-to-end institutional-grade reinforcement learning pipeline designed for sub-second intraday trading. It leverages a **Single Branching Dueling Q-Network (BDQ)** agent to trade on Limit Order Book (LOB) data with micro-structure awareness.

## 🚀 Key Features
- **Unified MLOps Pipeline**: Orchestrates HPO $\to$ Single-Phase Training $\to$ Backtesting.
- **WandB Standardization**: Automatic canonical naming (`DeepScalper_V1_GPUHub_YYYYMMDD_HHMM`) for all runs to ensure auditability.
- **Mach 3 Optimization**: Optimized for RTX 5090 (32GB VRAM), utilizing Shared Memory, AMP, and Torch Compile for max throughput.
- **Remote Ops & Monitoring**: Integrated deployment engine with real-time remote GPU/Process monitoring scripts.
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

## 🛡️ Three-Tier QA Protocol

To ensure reliability on expensive GPU resources, we strictly follow a three-tier execution strategy:

### 1. Smoke Test (`deepscalper_smoke_test.yaml`)
- **Purpose:** Logic verification & Crash detection.
- **Scope:** 10 steps, Synthetic Data, CPU/Local.
- **Goal:** Confirm code runs without syntax errors or immediate crashes.

### 2. Pilot Run (`deepscalper_pilot_test.yaml`)
- **Purpose:** Integration testing & "Trap" detection.
- **Scope:** 1 Month Data, 5,000 Steps/Trial, Remote GPU.
- **Goal:** Verify pipeline connectors (WandB, Shared Memory, HPO) and catch data-specific bugs (e.g., Short Dataset constraints, Memory Leaks) before committing to full training.
- **Success Criteria:** HPO completes 1 cycle, Evaluation runs without OOM/BrokenPipe.

### 3. Production Run (`deepscalper_production.yaml`)
- **Purpose:** Model Convergence & Maximizing ROI.
- **Scope:** Full Year Data, 10M Steps, 24 Envs, Torch Compile ON.
- **Goal:** Produce the final profitable agent.

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

### 5. Monitoring & Ops (Remote)
Once deployed, use these utilities to track progress without full SSH sessions:
```bash
# Check overall status (PID, Log tail)
python scripts/check_remote_status.py

# Live GPU monitoring (nvidia-smi)
python scripts/check_remote_gpu.py

# List active checkpoints and WandB artifacts
python scripts/check_remote_checkpoints.py
python scripts/list_wandb.py
```

---

## 🛡️ Quality Assurance Protocols (Verification Tiers)

To ensure stability and prevent ambiguity between debugging and production verification, we define the following strict testing tiers:

### Tier 1: Smoke Test (Sanity Check)
- **Goal**: Verify connectivity, environment setup, SSH access, and file uploads.
- **Config**: `--steps 100`, `--trials 1`.
- **Duration**: ~1 minute.
- **Outcome**: Confirms *connectivity*, NOT logic. **NEVER** cite as "Success" for pipeline logic.

### Tier 2: Pilot Run (Logic Verification)
- **Goal**: Verify end-to-end pipeline logic, checkpointing, and memory stability.
- **Config**: `deepscalper_pilot_test.yaml` (100k steps).
- **Duration**: ~10-15 minutes.
- **Outcome**: REQUIRED pass before full production run.

### Tier 3: Production Run
- **Goal**: Model convergence and maximum performance.
- **Config**: `deepscalper_unified.yaml` (Full steps, full data).
- **Outcome**: Final Model Artifacts.

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
