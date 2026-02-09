# DeepScalper Pipeline — Data Science Expert Audit

> **Objective:** You are a quantitative data science expert specializing in Deep Reinforcement Learning for financial markets. Your task is to **adversarially audit** the configuration, hyperparameters, and design choices of the DeepScalper pipeline to identify anything that could silently compromise the agent's learning ability, profitability, or generalization.
>
> **Target:** BTCUSDT Perpetual Futures, minute-level LOB data (Binance).
>
> **Paper Reference:** Sun et al. (2022) — "DeepScalper: A Risk-Aware RL Framework for Intraday Trading"
>
> **Produce a finding for each item below** as: `PASS` (correct), `FLAG` (suspicious, explain why), or `FAIL` (will harm performance, propose fix).

---

## 1. Reward Function Design

The reward is computed per step in `deep_scalper_env.py`:

```
r_t = [ (mid_{t+1} - mid_t) × pos_{t} − fees + w × pos_t × (mid_{t+h} - mid_t) ] × scaling
```

**Production values:**
| Parameter | Value | HPO Range |
|---|---|---|
| `reward_scaling` | 1.0 | [0.01, 1.0] log |
| `hindsight_weight` (w) | 0.1 | [0.001, 0.2] log |
| `hindsight_horizon` (h) | 180 steps (= 3 hours) | [30, 60, 90, 120, 150, 180] |
| `volatility_horizon` | 100 steps (aux target only) | Fixed |
| `sharpe_weight` (DSR) | 0.0 (disabled) | N/A |

**Audit items:**
- [ ] Is `reward_scaling=1.0` appropriate for BTC prices (mid_price ~$20K–$45K)? The raw PnL per step is `price_delta × position`. For BTC at $20K, a 1-tick move ($0.10) × 1 BTC position = $0.10 reward. Is this scale appropriate for a neural network with Huber loss?
- [ ] **Hindsight horizon = 180 minutes (3 hours):** The paper finds 180 optimal for stocks. Is this appropriate for BTC perpetuals which exhibit faster mean-reversion cycles and 24/7 trading?
- [ ] **Hindsight bonus applied only when `abs(prev_position) > 1e-12`:** This means the agent gets ZERO hindsight signal when flat. Should the hindsight bonus also consider the *action taken*, not just pre-existing position?
- [ ] **Fees subtracted from reward AND balance:** The fee is deducted from `self.balance` during execution (L377, L432) AND also added as `reward_fee = -self.step_transaction_costs` (L501-502). Verify this is not double-counting. Is the reward correctly reflecting PnL that already includes fees via balance change?
- [ ] **Reward magnitude vs. gradient scale:** With `batch_size=1024` and Huber loss, is the reward signal strong enough to produce meaningful gradients against the auxiliary volatility loss (weighted by `auxiliary_weight=1.0`)?

---

## 2. Exploration & Epsilon Decay

```python
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.999908  # Applied once per step batch (every num_envs=12 steps)
```

**Epsilon reaches `epsilon_end` at:** `ln(0.01) / ln(0.999908) ≈ 50,000 decay calls`. With `num_envs=12`, that's `50,000 × 12 = 600,000 environment steps` (of 1M total).

**Audit items:**
- [ ] Agent explores for 60% of training, then exploits for 40%. Is this a good ratio? The paper doesn't specify. With only 1M steps and 5K `learning_starts`, the agent has ~400K exploitation steps. Is that enough to converge?
- [ ] **Epsilon is shared across all 12 parallel envs.** Standard practice for DQN variants is shared epsilon. But should uncorrelated exploration noise (e.g., per-env ε offsets) be considered for better coverage of the state space?
- [ ] In HPO mode, epsilon is recalculated to reach 0.01 by end of trial (50K steps). With `num_envs` capped at 4 for HPO, that's ~12,500 decay calls. Is this too aggressive? Agents barely have time to learn before being forced to exploit.

---

## 3. Replay Buffer & Prioritized Experience Replay (PER)

```yaml
buffer_size: 2,000,000
batch_size: 1024
use_per: false  # PER is implemented but disabled in production config
```

**PER parameters (when enabled):**
| Parameter | Value |
|---|---|
| `per_alpha` | 0.6 |
| `per_beta_start` | 0.4 |  
| `per_beta_frames` | 100,000 |

**Audit items:**
- [ ] **PER is disabled in production.** The paper (Section 4.3) explicitly uses PER. Why is it disabled? Is this a deliberate choice or an oversight? What is the expected performance delta?
- [ ] **Buffer size = 2M transitions.** Each transition stores full dict observations (micro: 50×27, private: 50×2, macro: 11). Estimated memory: `2M × ~150KB ≈ 300GB`. This will NOT fit in RAM. The agent warns at >8GB but doesn't prevent it. **Is this a silent OOM crash waiting to happen?**
- [ ] **Replay ratio:** `update_interval=2.0` means 2 gradient updates per environment step batch. With `batch_size=1024` and `num_envs=12`, the replay ratio is `2 × 1024 / 12 ≈ 170`. Is this replay ratio sensible? Standard DQN uses ratio ~1-4. High ratio can cause overfitting to buffer contents.
- [ ] **Beta annealing horizon = 100K frames.** With 1M total steps, beta reaches 1.0 at 10% of training. The rest of training uses full IS correction. Is 100K too fast?

---

## 4. Network Architecture

```yaml
micro_config:
  input_size: 27          # LOB features per timestep  
  private_input_size: 2   # [normalized_position, normalized_balance]
  hidden_size: 256        # LSTM hidden
  rnn_type: "LSTM"

macro_config:
  input_size: 11          # Technical indicators
  hidden_sizes: [256, 128]  # MLP layers
```

**Action space:** `MultiDiscrete([3, 5, 5])` → Branching DQN with 3 heads (direction, price, volume).

**Audit items:**
- [ ] **LSTM hidden=256 for 27 features.** The MicroEncoder uses a single-layer LSTM with 256 hidden units to encode LOB snapshots. Is this capacity sufficient or excessive for 27 input features over a 50-step window?
- [ ] **MacroEncoder MLP [256, 128] for 11 features.** This seems potentially over-parameterized for 11 inputs. Could this memorize noise in the macro features?
- [ ] **No dropout, no layer normalization.** The paper doesn't mention regularization. Given the limited training data (~8 days), is the network at risk of overfitting?
- [ ] **Private state as separate LSTM input.** The private state (position + balance) is concatenated with the LSTM hidden state after encoding. This means the network cannot condition its temporal attention on current position. Is this optimal?
- [ ] **Auxiliary volatility prediction head.** The `aux_head` predicts future volatility with `auxiliary_weight=1.0`. This is equal weight to the 3-branch Q-loss combined. Should the volatility loss dominate the gradient signal? What is the typical magnitude of `loss_aux` vs. `loss_dir + loss_price + loss_vol`?

---

## 5. Training Loop & Optimization

```yaml
total_timesteps: 1,000,000
learning_starts: 5,000
update_interval: 2.0        # 2 gradient updates per step batch
target_update_freq: 7,500   # Hard target sync
batch_size: 1024
learning_rate: 0.0001       # Fixed (not in HPO)
gamma: 0.995
torch_compile: true
use_amp: true               # FP16 mixed precision
gradient_clip: 1.0          # Hard-coded in bdq_agent.py
```

**Audit items:**
- [ ] **gamma = 0.995.** For minute-level data, the effective horizon is `1/(1-γ) = 200 steps ≈ 3.3 hours`. The hindsight horizon is also 180 minutes. These are roughly aligned. But is a 3-hour discount horizon appropriate for a scalper? Scalpers typically operate on much shorter horizons (minutes to tens of minutes). Should gamma be lower (e.g., 0.99 = 100 steps)?
- [ ] **learning_rate is NOT in HPO search space.** The config fixes it at 1e-4. The HPO searches [5e-5, 5e-4]. But the production default is 1e-4 which is then *overridden* by HPO. If HPO is disabled, is 1e-4 a safe default?
- [ ] **`target_update_freq=7500` (hard sync).** This is measured in *gradient update steps*, not environment steps. With `update_interval=2.0`, that's `7500/2 = 3750 environment step batches × 12 envs = 45,000 env steps` between target syncs. Only ~22 target syncs occur in 1M steps. Is this too infrequent? Could soft (Polyak) updates provide better stability?
- [ ] **`learning_starts=5000` env steps.** With `num_envs=12`, the buffer has `5000 × 12 = 60,000` transitions before first update. But `batch_size=1024`, so the buffer is well-populated. However, those 60K transitions are from a random policy (ε=1.0). Is learning from purely random initial data effective for LOB trading?
- [ ] **`torch.compile` + AMP (FP16).** AMP uses float16 for forward pass. For Q-values that may be very small (reward ≈ $0.10 at BTC price levels), is float16 mantissa (10 bits, ~3 decimal digits) sufficient to distinguish between similar Q-values across 5 price levels?
- [ ] **No learning rate scheduler.** The learning rate is fixed at 1e-4 throughout training. Should it decay (e.g., cosine annealing) to improve convergence in later stages?

---

## 6. Data & Feature Engineering

**Training data:** `btc_lob_jan2023.parquet` — January 10–18, 2023 (8 days of minute-level data).
**Validation:** January 19 (1 day). **Test:** January 20 (1 day).

**Micro features (27 dims):**
- 20: LOB L1–L5 (bid_px, bid_vol, ask_px, ask_vol) × 5 levels, normalized
- 5: Order Flow Imbalance per level
- 1: Spread (basis points)
- 1: Log return

**Macro features (11 dims):**
- 5: Z-scored OHLCV (z_open, z_high, z_low, z_close, z_volume)
- 6: Z-scored moving average deviation ratios (zd_5, zd_10, zd_15, zd_20, zd_25, zd_30)

**Audit items:**
- [ ] **8 days of training data.** This is extremely limited. At minute resolution, that's ~11,520 rows. With `window_size=50`, effective samples ≈ 11,470. With `total_timesteps=1M` and `num_envs=12`, the agent replays this data `1M / 11,470 ≈ 87 times`. Is this severe overfitting to a single market regime?
- [ ] **No walk-forward validation in production config.** A `RollingWindowSplitter` exists in the codebase but is not wired into the production pipeline. All training uses a single fixed split (Jan 10–18 train, Jan 19 val, Jan 20 test). This means zero protection against non-stationarity.
- [ ] **Z-score normalization of macro features.** Z-scores are computed over the entire dataset (not rolling). This leaks future information into the features if computed before the train/test split. Verify: are z-scores computed per-split or globally?
- [ ] **Missing features vs. paper.** The paper mentions RSI, MACD, and Bollinger Bands. The implementation uses only z-scored OHLCV and SMA deviation ratios. Are the paper's recommended indicators covered by these alternatives?
- [ ] **LOB price normalization.** Bid/Ask prices are normalized (column prefix `n_`). What normalization method? If min-max: the range depends on the full dataset → lookahead bias. If z-score: BTC price is trending, z-score will be biased.
- [ ] **1 day of test data (Jan 20).** A single day backtest is statistically meaningless. The Sharpe ratio from ~1,440 minute-level returns has enormous estimation error. Is the pipeline claiming statistical significance from this?

---

## 7. HPO Design

```yaml
n_trials: 50
steps_per_trial: 50,000  # 5% of full training
sampler: TPESampler(seed=42)
pruner: HyperbandPruner(min_resource=15000, max_resource=50000, reduction_factor=3)
```

**Search space (10 parameters):**
| Parameter | Type | Range |
|---|---|---|
| hindsight_horizon | Categorical | [30, 60, 90, 120, 150, 180] |
| hindsight_weight | Float (log) | [1e-3, 0.2] |
| reward_scaling | Float (log) | [1e-2, 1.0] |
| auxiliary_weight | Categorical | [0.5, 1.0] |
| learning_rate | Float (log) | [5e-5, 5e-4] |
| target_update_freq | Categorical | [5000, 7500, 10000, 15000] |
| batch_size | Categorical | [256, 512, 1024] |
| gamma | Categorical | [0.99, 0.995] |
| epsilon_end | Float | [0.01, 0.10] |

**HPO objective:** Raw (non-annualized) mean/std ratio of portfolio returns over 5,000 evaluation steps.

**Audit items:**
- [ ] **50 trials with 10 parameters.** TPE needs ~10-20 trials per parameter to converge. With 10 params, 50 trials may be insufficient. The search space has `6 × continuous × continuous × 2 × continuous × 4 × 3 × 2 × continuous ≈ 288 discrete combinations` × infinite continuous axes. Is 50 trials enough?
- [ ] **HPO trained on 50K steps, production trained on 1M steps (20× more).** Parameters optimized for 50K may not transfer. Especially: `target_update_freq` (designed for 50K horizon), `epsilon_end` (exploration budget), and `batch_size` (buffer fill dynamics). Are these safe to extrapolate?
- [ ] **HPO evaluation uses only 5,000 steps (≈83 minutes of market data).** This is extremely noisy. Two identical agents will produce wildly different Sharpes over 83 minutes. Can Optuna's TPE reliably discriminate signal from noise?
- [ ] **Pruning at min_resource=15K steps.** The agent has only done `(15K - 5K) / 2 ≈ 5000 gradient updates` at this point (with update_interval=2). Is this enough to judge a trial? LSTM needs warm-up time.
- [ ] **Parameters NOT searched:** `buffer_size`, `num_envs`, `window_size`, `per_alpha/beta`, network hidden sizes. Are any of these more impactful than what's being searched?
- [ ] **Seed=42 for TPE.** This makes HPO deterministic but means the results are a single sample of the HPO landscape. Is there value in running multiple HPO seeds?

---

## 8. Environment Mechanics

```yaml
initial_balance: 100,000 USDT
margin_requirement: 1.0  # Spot-equivalent (no leverage)
maker_fee: 0.0002 (2 bps)
taker_fee: 0.0004 (4 bps)
max_position: 1.0 BTC
max_drawdown_pct: 0.30  # 30% stop-loss
base_slippage_bps: 1.0
slippage_impact_factor: 0.5
```

**Action space:** `MultiDiscrete([3, 5, 5])`
- Direction: [Hold, Buy, Sell]
- Price offset ticks: [-1, 0, 1, 2, 3] from best bid/ask
- Volume proportion: [0.1, 0.25, 0.5, 0.75, 1.0] × `max_position`

**Audit items:**
- [ ] **max_position = 1 BTC ≈ $20K–45K (2023).** With $100K balance and `margin_requirement=1.0` (spot), a full 1 BTC position = 20–45% capital allocation. Is this realistic for a scalper? Should the agent use leverage (e.g., 2–5× for futures) to match real exchange conditions?
- [ ] **Pending order system.** Actions are placed as pending orders at time T, filled at time T+1 against new market data. This simulates limit-order latency. But the agent NEVER sees whether its order was filled until the next step. Should fill/reject feedback be part of the observation?
- [ ] **Volume bins are ABSOLUTE proportions of max_position, not position-relative.** Buying 1.0 × `max_position` = 1 BTC is the same regardless of current position. The agent must learn position management purely through the reward signal. Is this a design limitation?
- [ ] **30% max drawdown stop.** At $100K initial, the agent terminates at $70K. With 1 BTC position and BTC volatility of ~3% daily, the agent could hit 30% drawdown in ~10 days of adverse moves. Is this too loose for a scalper?
- [ ] **Slippage model: `base + (size/liquidity) × factor`.** The `slippage_impact_factor=0.5` means a 0.5 bp additional cost per unit of liquidity consumed. Is this calibrated to real Binance BTCUSDT LOB depth? At what trade sizes does slippage become material?
- [ ] **Fee asymmetry (maker=2bps, taker=4bps).** The price offset determines maker/taker status: offset -1 (crossing spread) = taker, offset ≥0 = maker. But in the action space, taker is only 1 out of 5 price choices. Does this bias the agent toward passive orders that may never fill?

---

## 9. Training Data Economics

**Quick math for the agent to be profitable:**

```
Round-trip cost = maker_fee + taker_fee = 2 + 4 = 6 bps
Plus slippage = ~2 bps round trip (estimated)
Total cost per round trip ≈ 8 bps = 0.08%

At $20K BTC, 1 BTC position:
  - Round-trip cost: $20K × 0.0008 = $16
  - Per-minute BTC volatility (σ): ~0.03% ≈ $6
  - Signal required to be profitable: >2.7× per-minute volatility
```

**Audit items:**
- [ ] **Cost-to-signal ratio.** The agent needs to predict price moves >8 bps to break even on a round trip. BTC minute-level returns have σ ≈ 3 bps. The agent needs a prediction accuracy of >2.7σ per trade to be profitable. Is an LSTM on 8 days of data capable of this?
- [ ] **Trade frequency.** The agent can trade every minute. With 75 action combinations (3×5×5), how many result in "Hold"? Only direction=0 (any price, any volume) = 25/75 = 33% of actions are Hold. During random exploration (ε=1.0), the agent trades 67% of steps. At 6 bps per trade, random exploration costs ~$16 × 0.67 × 11,470 ≈ $123K (more than entire balance). **Does the agent go bankrupt during exploration?**
- [ ] **Position "churn" cost.** Even deterministic actions may flip positions frequently. Without a position change penalty, the agent may learn to rapidly oscillate between long and short, generating fees without directional conviction. Is this observed in training logs?

---

## 10. Generalization & Overfitting Risks

**Audit items:**
- [ ] **Single regime training.** January 2023 BTC traded between ~$16.5K–$23K (strong uptrend after FTX crash bottom). The agent learns patterns from this specific regime. What happens in ranging, crashing, or volatile regime-change markets?
- [ ] **No data augmentation beyond private state.** The `private_state_augment_prob` injects random positions at episode start. But the market data sequence is identical every episode. The agent sees the exact same LOB tape ~87 times. Standard time-series augmentation (jitter, scaling, window warping) is absent.
- [ ] **Validation loss not used for early stopping.** The trainer runs for exactly `total_timesteps` with no validation-based stopping criterion. Overfitting can occur silently.
- [ ] **Backtest on adjacent day (Jan 20).** The test set is the very next day after training. Market microstructure (spread patterns, volume profiles) on adjacent days is highly autocorrelated. A true out-of-sample test would use data from a different month or market regime.

---

## Deliverable

For each audited item, provide:
1. **Verdict:** `PASS`, `FLAG`, or `FAIL`
2. **Evidence:** Quantitative reasoning (back-of-envelope calculations encouraged)
3. **Risk Level:** Low / Medium / High / Critical
4. **Recommendation:** If FLAG or FAIL, propose a specific change with expected impact

**Priority ranking:** After auditing all items, rank the top 5 findings by expected impact on agent profitability, from highest to lowest.
