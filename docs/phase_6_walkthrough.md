# DeepScalper Phase 6: Real Data & Ensemble Training

## Status: ✅ COMPLETE

**Objective:** Transition from mock environments to a real training pipeline using LOB data from Parquet files, implementing the full "Brain" (Ensemble) and "Body" (Environment) architecture.

---

## 1. Core Implementation

### Data Pipeline (`ParquetDataHandler`)
- **Source:** `finrl_pro_ds/data/parquet_handler.py`
- **Function:** Streams high-frequency LOB data (wideth-format bid/ask) from Parquet files.
- **Integration:** Wired into `DeepScalperEnv` via the `make_env` factory in `scripts/train_deepscalper.py`.
- **Safety:** Implements non-leaky cursor logic (`step()`) and reset functionality.

### Trainer Architecture (`DeepScalperTrainer`)
The trainer now orchestrates the complex ensemble logic:

- **PPO Agent (On-Policy):**
  - **Loss:** Standard PPO CLIP loss.
  - **Advantage:** Generalized Advantage Estimation (GAE) with correct next-state bootstrapping (`val_next` computed at collection).
  - **Update:** Updates every `update_interval` steps, clears buffers on episode end.

- **A2C Agent (On-Policy):**
  - **Loss:** Advantage Actor-Critic (Policy Loss + Value Loss).
  - **Sync:** Updates synchronously with PPO.

- **DQN Agent (Off-Policy):**
  - **Memory:** `ReplayBuffer` stores transitions `(s, a, r, s', d)`.
  - **Training:** Trains individually using its own replay buffer (off-policy efficiency).
  - **Exploration:** Uses ε-greedy on the ensemble actions + internal exploration.

- **Synapse Gating Network:**
  - **Mechanism:** REINFORCE-based update.
  - **Logic:** `Loss = - (weights * reward).mean()`. Encourages weighting configurations that yield positive rewards.

## 2. Red Team Audit & Remediation (Jan 21, 2026)

A rigorous "Red Team" audit identified critical financial validity issues in the Phase 6 implementation.

### 🔴 Critical Findings
1.  **Sparse Reward:** Previous reward function only triggered on Realized PnL (closing positions), creating a credit assignment gap.
2.  **Incorrect Fee Logic:** All orders were charged `taker_fee` (0.04%), forcing negative EV for scalping strategies that rely on passive execution.
3.  **No Safety Stops:** The agent could lose infinite funds during exploration.

### ✅ Remediation Implemented
- **Dense Reward Signal:** Switched to **Unrealized PnL (Mark-to-Market)**. The agent now receives `Delta(Portfolio_Value)` at every step.
- **Maker/Taker Distinction:**
  - If `Limit_Buy >= Best_Ask` at submission: Charged **Taker Fee** (0.04%).
  - If `Limit_Buy < Best_Ask` at submission: Charged **Maker Fee** (0.02%).
- **Safe-Fail Mechanism:** Implemented a **20% Max Drawdown Stop**. Episode terminates immediately if `Portfolio Value < 0.8 * Initial Balance`.
- **Data Handler Patch:** Fixed `ParquetDataHandler` to robustly detect and align pre-computed macro features.

## 3. Verification Results (Smoke Test)

**Command:**
```bash
python scripts/train_deepscalper.py --config configs/smoke_test.yaml
```

**Outcome:**
- **Status:** PASS
- **Safety Check:** The smoke test successfully triggered the **Max Drawdown Stop** (20% loss) and terminated the episode gracefully, proving the safety guardrails are active.
- **Stability:** Data loading and training loop ran without crashes or tensor shape errors.

## 4. How to Run

1.  **Prepare Data:** Ensure `btc_lob_2025.parquet` is available.
2.  **Configure:** Update `configs/deepscalper_v1.yaml`:
    ```yaml
    data:
      file_path: "c:/data/btc_lob_2025.parquet"
      ticker: "BTCUSDT"
    training:
      total_timesteps: 100000
    ```
3.  **Execute:**
    ```powershell
    python scripts/train_deepscalper.py --config configs/deepscalper_v1.yaml
    ```

## 5. Next Steps
- [ ] **Hyperparameter Tuning:** Adjust `gamma`, `lam`, and learning rates in config.
- [ ] **Full Training Run:** Execute on full dataset and monitor WandB.
- [ ] **Metric Analysis:** Watch `loss/ppo`, `loss/gating`, and `cumulative_reward`.
