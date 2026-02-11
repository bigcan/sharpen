# DeepScalper

**High-Frequency Crypto Scalping with Deep Reinforcement Learning (Bitcoin Futures)**

DeepScalper is an institutional-grade reinforcement learning pipeline for sub-second intraday trading. It implements a **Branching Dueling Q-Network (BDQ)** agent trained on Limit Order Book (LOB) micro-structure data, following [Sun et al. (2022)](docs/2201.09058v3.pdf).

---

## Key Features

- **End-to-End Pipeline**: HPO → Training → Backtesting in a single script
- **Multi-Modal Observations**: Fuses LOB micro-features (5-level OFI, spread, returns) with macro technical indicators (SMAs, z-scores)
- **Branching Dueling DQN**: Three action branches (direction, price, volume) with value-advantage decomposition
- **Auxiliary Volatility Prediction**: Side-task head improves representation learning (Section 4.4)
- **Optional Risk-Aware Reward**: Differential Sharpe Ratio (Moody & Saffell 2001) — blendable, opt-in risk signal
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

Both share the same fee structure (`maker_fee: 0.0002`, `taker_fee: 0.0005`) to ensure consistency.

---

## Reward Function

The reward follows [Sun et al. (2022)](docs/2201.09058v3.pdf) Section 3.2 + 4.2:

```
r_t = (mid_{t+1} - mid_t) × pos_t − fees + w × pos_t × (mid_{t+h} - mid_t)
```

### Optional: Differential Sharpe Ratio (DSR)

An opt-in risk-aware reward component (Moody & Saffell 2001) that provides a dense, per-step signal
approximating the marginal contribution to the Sharpe ratio. When enabled, the final reward becomes:

```
reward = (1 − sharpe_weight) × paper_reward + sharpe_weight × DSR_t
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sharpe_weight` | `0.0` | Disabled by default. Set `0.1–0.5` to enable risk-aware training. |
| `sharpe_horizon` | `100` | EMA lookback window. Lower = faster adaptation, noisier signal. |

Configure in `env.reward` section of your YAML config. See `randd_log.md` for full design notes.

---

## Agent Memory System

This project includes a **persistent long-term memory system** that gives the AI agent continuity across chat sessions. It stores project context, daily session logs, and milestone snapshots in plain Markdown files.

### How It Works

```
.agent/memory/
├── core.md           ← Project facts, preferences, active decisions (git-tracked)
├── logs/
│   └── YYYY-MM-DD.md ← Daily session logs with timestamped entries (gitignored)
└── snapshots/
    └── YYYY-MM-DD-topic.md ← Session summaries (gitignored)
```

- **`core.md`** is the agent's ground truth — project context, your preferences, and active architectural decisions.
- **Daily logs** record key decisions, bug fixes, and deployments as timestamped entries.
- **Snapshots** capture end-of-session summaries with what was accomplished and next steps.

### User Commands

| Command | What It Does |
|---|---|
| `/memory-boot` | Loads core memory + recent logs at session start. Run this first in any new chat. |
| `//save` | Generates a session snapshot and appends it to today's log. Use at end of session. |

> **Tip**: The agent's skill description includes a directive to self-load memory at session start, but saying `/memory-boot` guarantees it.

### Editing Core Memory

To update project facts or preferences, just tell the agent (e.g., "update my preferences: I prefer verbose logging"). It will modify `core.md` under the relevant section.

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Expected: 40 passed (16 env + 24 regression)
```

---

## Risk Disclaimer

This software is for educational and research purposes only. **Deep Reinforcement Learning involves significant financial risk.** The authors are not responsible for trading losses.

---

*Based on "DeepScalper: A Risk-Aware Reinforcement Learning Framework for High-Frequency Trading" (Sun et al., 2022).*
