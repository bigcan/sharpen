# DeepScalper

**High-Frequency Crypto Scalping with Deep Reinforcement Learning (Bitcoin Futures)**

DeepScalper is an institutional-grade reinforcement learning pipeline for sub-second intraday trading. It implements a **Branching Dueling Q-Network (BDQ)** agent trained on Limit Order Book (LOB) micro-structure data, following [Sun et al. (2022)](docs/2201.09058v3.pdf).

---

## Key Features

- **End-to-End Pipeline**: HPO → Training → Backtesting in a single script
- **Multi-Modal Observations**: Fuses LOB micro-features (5-level OFI, spread, returns) with macro technical indicators (SMAs, z-scores)
- **Branching Dueling DQN**: Three action branches (direction, price, volume) with value-advantage decomposition
- **Auxiliary Volatility Prediction**: Side-task head improves representation learning (Section 4.4)
- **Shared Memory Data Streaming**: Zero-copy `ParquetDataHandler` for multi-env training throughput
- **WandB Integration**: Canonical run naming, metric logging, and institutional-grade reporting
- **RTX 5090 Optimized**: AMP, Torch Compile, and tuned batch sizes for 32GB VRAM

---

## Installation

Requires **Python 3.10+** and a CUDA-capable GPU.

```bash
# Clone & Setup
git clone https://github.com/bigcan/FinRL-Pro-DS.git
cd FinRL-Pro-DS
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # or: source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### Run the Full Pipeline (Recommended)

```bash
# Development (fast iteration, 50k steps)
python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml

# Production (RTX 5090, full training)
python scripts/run_full_pipeline.py --config configs/deepscalper_rtx5090_production.yaml
```

### Deploy to Remote GPU (GPUHub)

```bash
python scripts/deploy_bare_metal.py \
    --script scripts/run_full_pipeline.py \
    --config configs/deepscalper_rtx5090_production.yaml \
    --run_name DS_Production_V1
```

---

## Architecture

```
Observation Space                    Agent
─────────────────                    ─────
                                     ┌──────────────────┐
Micro (LOB × 5 levels)             │  MicroEncoder     │
  • Normalized prices (bps)    ────▶│  (LSTM/GRU)       │──┐
  • Log volumes (z-scored)          └──────────────────┘  │
  • OFI, Spread, Log-Returns                              │  ┌───────────┐
                                                          ├─▶│  Fusion   │
Private State                                             │  │  (MLP)    │
  • Position / Max Position    ─────────────────────────┘  └─────┬─────┘
  • Balance / Initial Balance                                     │
                                     ┌──────────────────┐        │
Macro (OHLCV)                       │  MacroEncoder     │        ▼
  • z_open, z_high, z_low     ────▶│  (MLP)            │──┐  ┌────────────────┐
  • z_close, z_adj_close            └──────────────────┘  │  │ Dueling Heads  │
  • SMA zd_{5..30}                                        ┘  │ • V(s)         │
                                                              │ • A_dir(s,a)   │
                                                              │ • A_price(s,a) │
                                                              │ • A_vol(s,a)   │
                                                              │ • σ_pred (aux) │
                                                              └────────────────┘
```

---

## Project Structure

```
FinRL-Pro-DS/
├── configs/
│   ├── deepscalper_dev.yaml              # Fast local development (50k steps)
│   └── deepscalper_rtx5090_production.yaml  # Full production training
├── finrl_pro_ds/                         # Core Package
│   ├── agents/deepscalper/
│   │   ├── bdq_agent.py                  # BDQ agent (replay buffer, ε-greedy)
│   │   └── networks.py                   # MicroEncoder, MacroEncoder, DeepScalperNetwork
│   ├── analytics/
│   │   ├── pyfolio_analyzer.py           # Performance metrics & tear sheets
│   │   └── wandb_evaluator.py            # WandB metric logging & analysis
│   ├── data/
│   │   ├── feature_engineering.py        # Micro (LOB) & Macro (OHLCV) features
│   │   ├── parquet_handler.py            # Shared-memory Parquet data streaming
│   │   ├── splitter.py                   # Rolling window train/val/test splits
│   │   └── binance_loader/              # Binance data download & processing
│   ├── envs/
│   │   └── deep_scalper_env.py           # Gymnasium trading environment
│   ├── training/
│   │   ├── deepscalper_trainer.py        # Training loop, HPO, checkpointing
│   │   └── accumulators.py              # Metric accumulators
│   └── utils/
│       └── naming.py                     # WandB run name standardization
├── scripts/
│   ├── run_full_pipeline.py              # Main entry: HPO → Train → Backtest
│   ├── deploy_bare_metal.py              # Remote GPU deployment
│   ├── ralph_autonomous.py               # Autonomous workflow agent
│   ├── fetch_wandb_run.py                # WandB run data fetcher
│   ├── notion_sync.py                    # Notion integration
│   └── data/                            # Data acquisition scripts
├── tests/                               # Pytest test suite
├── docs/2201.09058v3.pdf                # Original paper
└── requirements.txt
```

---

## Configuration

Two configuration modes:

| Config | Use Case | Steps | Envs | Purpose |
|--------|----------|-------|------|---------|
| `deepscalper_dev.yaml` | Local dev | 50k | 4 | Fast iteration & debugging |
| `deepscalper_rtx5090_production.yaml` | Production | 10M+ | 24 | Full convergence on RTX 5090 |

Both share the same fee structure (`maker_fee: 0.0002`, `taker_fee: 0.0004`) to ensure consistency.

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Expected: 19 passed, 1 known failure (test_reward_logic_risk_penalty)
```

---

## Risk Disclaimer

This software is for educational and research purposes only. **Deep Reinforcement Learning involves significant financial risk.** The authors are not responsible for trading losses.

---

*Based on "DeepScalper: A Risk-Aware Reinforcement Learning Framework for High-Frequency Trading" (Sun et al., 2022).*
