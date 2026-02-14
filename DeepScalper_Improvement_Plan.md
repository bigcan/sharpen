# DeepScalper Improvement Plan — Post-V1 Production Run

> **Context**: Analysis of run `1l6wi43d` (Feb 13 2026). OOS Test Sharpe +0.76, Return -7.6%, PF 1.04, MaxDD -47.9%.
> **Target**: PF > 1.3, Sharpe > 1.5, MaxDD < 25%.

---

## Executive Summary

The V1 production run proves the pipeline works end-to-end and the agent has learned *something* beyond random — PF=1.04 is above break-even. However, the agent is far from consistently profitable. Analysis of the codebase reveals **3 categories** of issues in descending order of expected impact:

1. **Metrics & Analytics Bugs** — Zero Sortino/Calmar/Omega means we're flying partially blind
2. **Exploration & Learning Efficiency** — Uniform random exploration over 45 actions wastes >>50% of exploration budget on nonsensical combinations; Q-values at ~24 suggest overestimation
3. **Data Scarcity & Overfitting** — 194K rows × 5 epochs with augmentation *disabled* means the agent memorizes rather than generalizes

The plan below is organized into 3 tiers by implementation complexity.

---

## Sprint Progress

| Sprint | Items | Goal | Status |
|--------|-------|------|--------|
| **S1** | 1.2 + 1.4 + 1.5 + 1.3 | Reduce overfitting, clean exploration | ✅ DONE |
| **S2** | 2.1 + 2.2 (Boltzmann + Polyak) | Better exploration + stable Q-values | ❌ REGRESSED (Boltzmann reverted, Polyak retained) |
| **S2.5** | Target Q-clip ±5000 | Fix Q-value divergence from S2 | ✅ DONE |
| **S3** | S1 + Polyak + Q-clip | Verify stabilization | ✅ DONE (`6ezrc832`) — ⚠️ Overfit persists |
| **S3→** | 2.5 (Drawdown penalty) | Risk control | ❌ FAIL (Strategy) — Code verified, policy failed |
| **S4** | 3.1 (Walk-forward) | Honest OOS evaluation | ⏳ READY (Validation Needed First) |

### Key Decisions Made
- **No leverage**: `margin_requirement: 1.0` (spot only). BTC vol ~54% is too high for margin.
- **Boltzmann abandoned**: Caused Q-divergence. Epsilon-greedy restored.
- **Polyak kept**: τ=0.005 soft target updates beneficial.
- **Target Q-clip**: ±5000 active as safety net.

### Run Log

| Date | Sprint | Run ID | Result |
|------|--------|--------|--------|
| 2026-02-13 | S1 | `ox89l7q3` | Test Sharpe +4.62 / Val -37.3 (Overfit) |
| 2026-02-13 | S2 | `2f4d1i79` | ❌ Val -9.65, Test -18.75, Q-max 1072 |
| 2026-02-14 | S3 | `6ezrc832` | ⚠️ Q-max 145↓, Val -4.90, Test +4.07, Val frozen |
| 2026-02-14 | S3→ | `5satim2n` | ❌ Test Sharpe -24.03, MaxDD -100% (Drawdown Trap) |

---

## Tier 1: Quick Wins (< 1 day each)

### 1.1 Fix Broken Risk Metrics (Sortino, Calmar, Omega) — ⚠️ PARTIAL

> **Status**: SPS logging fixed. Sortino/Calmar/Omega still returning 0 in Sprint 3 report.

| | |
|---|---|
| **Problem** | Sortino=0, Calmar=0, Omega=0 — all exactly zero. These are critical risk-adjusted metrics for evaluating whether the agent manages downside. The analytics pipeline likely has a division-by-zero or empty-array bug. |
| **Proposed Change** | Locate the backtest analytics function (likely in the training scripts or a separate metrics module). Fix: (a) guard `downside_deviation` against zero with a small epsilon, (b) ensure `max_drawdown` denominator is non-zero for Calmar, (c) compute Omega correctly as `sum(positive excess returns) / sum(negative excess returns)` with a threshold. |
| **Expected Impact** | No performance improvement, but gives us **3 additional signals** for evaluating future runs. Without these, we can't distinguish a lucky agent from a robust one. |
| **Risk** | None — analytics-only change, no training code touched. |
| **Verification** | Run backtest on existing checkpoint. Sortino, Calmar, Omega should be non-zero. Cross-validate against a manual calculation on the PnL timeseries. |

---

### 1.2 Enable Data Augmentation (`AugmentedDataWrapper`) — ✅ DONE (S1)

| | |
|---|---|
| **Problem** | The agent trains on 194K rows × 5 epochs = effectively the *same* 970K transitions. With `augmented_wrapper.py` disabled, every epoch sees identical observations. This is the single biggest contributor to overfitting. |
| **Proposed Change** | In the production config, add: |

```yaml
env:
  augmentation:
    enabled: true
    volatility_scale_range: [0.7, 1.5]    # Conservative to start
    spread_scale_range: [0.9, 1.2]
    volume_noise_std: 0.1
    macro_noise_std: 0.05
    ofi_scale_range: [0.9, 1.1]
```

| | |
|---|---|
| **Expected Impact** | Each episode sees a slightly different market regime. Reduces overfitting to specific price patterns. Expected: validation-test gap narrows, Sharpe variance across runs decreases. **PF +0.05-0.15.** |
| **Risk** | Too-aggressive perturbation (especially volatility) could destabilize learning. Start conservative (ranges above are tighter than defaults). The wrapper already exists and handles the `private` state correctly (untouched). |
| **Verification** | A/B run: one with augmentation, one without. Compare val/test Sharpe gap. The augmented run should have smaller gap. |

> **Result**: Enabled with `private_state_augment_prob: 1.0` per paper alignment.

---

### 1.3 Q-Value Clipping / Rescaling — ✅ DONE (S1 + S2.5)

| | |
|---|---|
| **Problem** | Q-values at ~24 for bps-normalized rewards are extremely high. With rewards in the O(1) bps range and γ=0.99, steady-state Q should be ~O(100) (= 1 bps / (1-0.99)). But 24 is still suspiciously large for step-level rewards that are often near zero, suggesting cumulative overestimation despite Double DQN. |
| **Proposed Change** | Add **reward clipping** in the environment's `step()` to cap at ±50 bps (prevents outlier transitions from inflating Q). Additionally, log Q-value statistics (std, max, min) to WandB for monitoring. |

```python
# In deep_scalper_env.py step(), after reward computation:
reward = float(np.clip(reward, -50.0, 50.0))  # ±50 bps cap
```

| | |
|---|---|
| **Expected Impact** | Stabilizes Q-value estimates. Prevents rare large rewards (e.g., large price move × max position) from corrupting priorities in PER. **Indirect improvement in policy stability.** |
| **Risk** | Clipping too aggressively can suppress legitimate large-move signals. ±50 bps is generous (~0.5% per step with 5× leverage). Monitor Q-value distributions to calibrate. |
| **Verification** | Check WandB: Q-value mean should stabilize below ~15. Loss should decrease more smoothly. |

> **Result**: Reward clipping ±50 bps (S1). Target Q-clip ±5000 added (S2.5) after Q-max hit 1072 in S2. S3 Q-max reduced to 145.

---

### 1.4 Reduce / Remove `hold_bonus_bps` — ✅ DONE (S1)

| | |
|---|---|
| **Problem** | `hold_bonus_bps=0.1` rewards staying flat. With 38% win rate and PF barely above 1.0, the agent may already be over-cautious. The hold bonus incentivizes *inaction*, which in a fee-heavy environment might be rational but prevents the agent from discovering profitable patterns. |
| **Proposed Change** | Set `hold_bonus_bps: 0.0` in the next run. The paper uses no such bonus. |
| **Expected Impact** | More active trading, potentially higher win rate as the agent tries more entries. Risk of more fees, but this is already pressure-tested via the reward function. |
| **Risk** | Agent may churn if it starts trading randomly. Mitigated by the fee penalty in the reward. |
| **Verification** | Compare trade count, win rate, PF between runs with/without hold bonus. |

> **Result**: Set to 0.0 in S1. Trade count healthy at 744 in S3.

---

### 1.5 Tune Epsilon Schedule — ✅ DONE (S1)

| | |
|---|---|
| **Problem** | Final ε=0.04 means 4% of actions are still random across all 45 action combinations. With `exploration_fraction=0.2`, ε decays over the first 20% of training (200K of 1M steps per epoch). By step 200K, the random exploration probability is nearly exhausted, but the buffer is only ~10% full. |
| **Proposed Change** | (a) Extend `exploration_fraction` to **0.5** so ε decays over 500K steps, keeping exploration alive while the buffer fills. (b) Reduce `epsilon_end` to **0.005** (0.5%) for deployment/backtest — currently 1% still introduces noise in evaluation. |
| **Expected Impact** | Better exploration coverage during early learning. Cleaner greedy policy at convergence. **Sharpe +0.1-0.3 from less noise at evaluation time.** |
| **Risk** | Longer exploration delays convergence slightly. Offset by more diverse experience in the buffer. |
| **Verification** | Check WandB epsilon curve, and correlate with val Sharpe at different stages. |

> **Result**: `exploration_fraction: 0.5`, `epsilon_end: 0.005` applied in S1.

---

## Tier 2: Architectural Improvements (1-3 days each)

### 2.1 Structured Exploration (Replace Uniform Random with Boltzmann / ε-z) — ❌ TRIED & REVERTED

| | |
|---|---|
| **Problem** | Current exploration selects uniformly random from `5 × 9 = 45` action combos. But most combos are nonsensical (e.g., aggressive price offset × max sell when long). Uniform exploration wastes ~80% of random actions on clearly suboptimal moves, filling the buffer with garbage transitions. |
| **Proposed Change** | **Option A — Boltzmann (Softmax) Exploration**: Replace uniform random with temperature-scaled softmax of Q-values. The agent already has `get_probs()` implemented! Instead of `random_mask → randint`, use: |

```python
# In bdq_agent.py predict():
if random_mask.any():
    temp = max(self.epsilon * 10.0, 0.1)  # Temperature decays with epsilon
    p_price, p_qty = self.get_probs(micro, private_in, macro, temp=temp)
    r_price = torch.multinomial(p_price, 1).squeeze(1)
    r_qty = torch.multinomial(p_qty, 1).squeeze(1)
    random_actions = torch.stack([r_price, r_qty], dim=1)
```

**Option B — NoisyNet**: Replace ε-greedy entirely with parametric noise in the network's linear layers (Fortunato et al., 2018). This requires modifying the `DeepScalperNetwork` to use `NoisyLinear` in the advantage heads.

| | |
|---|---|
| **Expected Impact** | Boltzmann: exploration concentrates on near-optimal actions, dramatically improving buffer quality. **Expected PF +0.1-0.3, Sharpe +0.3-0.5.** This is likely the single highest-leverage architectural change. |
| **Risk** | Boltzmann can collapse to greedy too early if temperature decays too fast. Use a floor temperature of 0.1. NoisyNet requires more extensive changes but has been proven in DQN variants. |
| **Verification** | Log action distribution entropy to WandB. Boltzmann should show gradually decreasing entropy (vs. constant high entropy for uniform random). PF should improve noticeably. |

> **Result**: ❌ Implemented in S2, caused Q-value divergence (Q-max >1000). Reverted to epsilon-greedy in S3. Boltzmann code removed. May revisit with NoisyNet approach later.

---

### 2.2 Soft Target Updates (Polyak Averaging) — ✅ DONE (S2, retained)

| | |
|---|---|
| **Problem** | Hard target update every 7,500 agent steps creates periodic discontinuities in the Q-target landscape. With `batch_size=1024` and `update_interval=0.25`, the agent does ~250 gradient steps between target syncs — a jarring change each time. This contributes to Q-value overestimation and training instability. |
| **Proposed Change** | Replace hard target copy with Polyak averaging (τ=0.005): |

```python
# In train_step(), replace the periodic hard update block:
with torch.no_grad():
    for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
```

| | |
|---|---|
| **Expected Impact** | Smoother Q-target evolution, reduced overestimation. **Expected: Q-values stabilize to O(5-10) range, loss decreases more monotonically.** |
| **Risk** | τ too large → target tracks policy too closely (becomes like online Q-learning). τ too small → slow learning. τ=0.005 is standard for DQN variants. |
| **Verification** | Compare Q-value trajectories on WandB between hard update vs. Polyak runs. Polyak should show smoother, lower Q-values. |

> **Result**: ✅ Implemented in S2, retained through S3. τ=0.005 working well.

---

### 2.3 Reward Normalization (Running Mean/Std) — ✅ DONE (S3)

| | |
|---|---|
| **Problem** | The reward is in bps but its distribution shifts dramatically across different market regimes. A trending period has large PnL rewards; a ranging period has near-zero rewards. This non-stationarity makes the TD-target scale unpredictable, destabilizing learning. |
| **Proposed Change** | Add a running reward normalizer in the trainer: |

```python
class RunningRewardNormalizer:
    def __init__(self, clip=10.0, decay=0.999):
        self.mean = 0.0
        self.var = 1.0
        self.clip = clip
        self.decay = decay
    
    def normalize(self, reward):
        self.mean = self.decay * self.mean + (1 - self.decay) * reward
        self.var = self.decay * self.var + (1 - self.decay) * (reward - self.mean)**2
        std = max(np.sqrt(self.var), 1e-8)
        return np.clip((reward - self.mean) / std, -self.clip, self.clip)
```

| | |
|---|---|
| **Expected Impact** | Stabilizes Q-value scale across regimes. Agent sees O(1) rewards regardless of absolute market conditions. **Expected: more consistent learning across walk-forward windows.** |
| **Risk** | Normalization removes absolute magnitude information. The agent can't distinguish "high-vol profitable" from "low-vol profitable". Mitigated by keeping the macro encoder's volatility features informative. |
| **Verification** | Log raw vs. normalized reward distributions to WandB. Normalized should have mean ~0, std ~1 throughout training. |

> **Result**: ✅ Implemented in `deepscalper_trainer.py`. Running mean/var with clip=10.0. Active in production.

---

### 2.4 Redesign Auxiliary Volatility Head — ⏳ NOT STARTED

| | |
|---|---|
| **Problem** | `loss_aux_ratio = 9e-5`. The volatility prediction head contributes essentially nothing to the total loss. With `auxiliary_weight=0.1`, this means the vol prediction loss itself is ~6e-4 (total loss ~6.17). The vol head isn't learning, or the target isn't informative. |
| **Proposed Change** | Two-part fix: (a) Scale the `volatility_target` to match Q-value magnitude (currently `v * 100`, may need further normalization). (b) **Alternative**: Replace vol prediction with a **return distribution prediction** (Bellemare et al., 2017 — distributional RL lite). Predict next-step return quintiles from `fusion_dim`. This gives richer gradient signal. |
| **Expected Impact** | A meaningful aux loss acts as a regularizer on the shared `fusion_dim` representation, preventing overfitting to the Q-branches alone. **Expected: better feature extraction, PF +0.05-0.10.** |
| **Risk** | If the auxiliary task is too hard or unrelated, it degrades Q-learning. Return quintile prediction has the advantage of being directly related to the trading objective. |
| **Verification** | Check `loss_aux` on WandB — should be O(0.1-1.0) after rescaling (vs. 9e-5 currently). Monitor whether Q-branch losses also improve. |

---

### 2.5 Position-Aware Reward Shaping — ✅ DONE (S3→)

| | |
|---|---|
| **Problem** | MaxDD=-47.9% with 5× leverage. The agent holds large positions through adverse moves. The reward function penalizes fees but not drawdown risk. The DSR component with `sharpe_weight=0.3` helps, but it's an EMA over 100 steps — too slow to react to sudden adverse moves. |
| **Proposed Change** | Add a drawdown-penalty term to the reward: |

```python
# In step(), after PnL calculation:
drawdown = 1.0 - (current_portfolio_value / peak_portfolio_value)
if drawdown > 0.10:  # >10% drawdown
    reward -= drawdown * 5.0  # 5 bps penalty per % drawdown beyond 10%
```

Track `peak_portfolio_value` (rolling max) as episode state.

| | |
|---|---|
| **Expected Impact** | Agent learns to cut losses early, reducing tail risk. **Expected: MaxDD improves from ~48% to <30%.** Sharpe may decrease slightly as the agent becomes more conservative, but risk-adjusted returns should improve. |
| **Risk** | Too-aggressive penalty makes the agent liquidate at every small dip. The 10% threshold and 5.0 scale need tuning. |
| **Verification** | Compare MaxDD, Calmar ratio, and PnL distribution tails between runs. A survival analysis (episodes reaching >20% drawdown) should show improvement. |

> **Note**: With leverage removed, the 10% threshold and 5.0 scale may need recalibration for spot-only returns.

---

### 2.6 Reward Function Experiment: Let HPO Decide — ⏳ PLANNED

| | |
|---|---|
| **Problem** | The current reward blends 3 objectives (PnL + DSR + Drawdown Penalty) with 5+ hyperparameters. The `sharpe_weight=0.3` was chosen arbitrarily. We don't know the optimal PnL-vs-DSR ratio. |
| **Proposed Experiment** | Widen `sharpe_weight` HPO range to `[0.0, 1.0]` and let the optimizer find the best blend empirically: |

| Param | HPO Range |
|-------|-----------|
| `sharpe_weight` | `[0.0, 1.0]` |
| `dsr_scale` | `[10, 1000]` (log) |
| `drawdown_penalty_factor` | `[50, 500]` |
| `hindsight_weight` | `[0.001, 0.2]` (log) |

| | |
|---|---|
| **Interpretation** | If HPO picks `sharpe_weight > 0.8` → DSR dominates, simplify reward. If `< 0.2` → DSR isn't helping, remove it. |
| **Control** | Current S3.5 run (`sharpe_weight=0.3`, `dsr_scale=100`). |
| **Key Metrics** | Total PnL, Sharpe, MaxDD, Trade Count, Win Rate. |
| **Verification** | Check HPO convergence on `sharpe_weight` across top-5 trials. Consistent clustering = strong signal. |

> **Implementation**: One config change (`sharpe_weight` HPO range). `dsr_scale` already implemented. No code changes needed.

---

## Tier 3: Strategic Changes (3-7 days each)

### 3.1 Walk-Forward Evaluation with Multiple Windows — ⏳ PLANNED (S4)

| | |
|---|---|
| **Problem** | The V1 run uses a single fixed split (5mo train / 2wk val / 2wk test). Test Sharpe=+0.76 but val Sharpe=-0.32 — a huge gap that could be pure luck. A single OOS window tells us almost nothing about generalization. |
| **Proposed Change** | Use the existing `RollingWindowSplitter` to train/evaluate across 3-5 rolling windows: |

```
Window 1: Train Jan-Apr, Val May 1-14, Test May 15-31
Window 2: Train Feb-May, Val Jun 1-14, Test Jun 15-30
Window 3: Train Mar-Jun, Val Jul 1-14, Test Jul 15-31 (need Jul data)
```

Aggregate metrics across all windows. The **mean** OOS Sharpe across windows is the true signal.

| | |
|---|---|
| **Expected Impact** | Eliminates single-window luck. If the agent performs consistently across 3+ windows, we have genuine evidence of an edge. If not, it reveals the strategy is regime-dependent. **This is the most important strategic change for honest evaluation.** |
| **Risk** | Requires more data (Jul+ data) for additional windows, or shorter train windows. Runtime scales linearly with window count (~6h × 3 = 18h). |
| **Verification** | Report mean ± std of all metrics across windows. A consistent strategy should have std(Sharpe) < 0.5 across windows. |

---

### 3.2 Expand Dataset (More Months of LOB Data) — ⏳ NOT STARTED

| | |
|---|---|
| **Problem** | 194K rows = 6 months of minute data. For a DRL agent with ~300K parameters, this is a very small training set. The agent effectively sees each unique market microstructure pattern ~5 times (5 epochs). In classical ML, this would be severe overfitting territory. |
| **Proposed Change** | Acquire 6+ additional months of CoinAPI LOB data (Jul 2025 – Dec 2025 at minimum). Process through the existing `feature_engineering.py` pipeline. This doubles the training data AND provides additional OOS months for walk-forward. |
| **Expected Impact** | More training data is almost always beneficial for DRL. The agent sees more diverse market regimes (trending, ranging, volatile, calm). **Expected: reduced overfitting, more stable OOS Sharpe.** |
| **Risk** | Processing cost (CoinAPI charges per data point). Storage (~500MB additional parquet). Training time scales ~linearly with data size. |
| **Verification** | Compare learning curves (buffer-fill rate, val Sharpe convergence) between 6-month and 12-month datasets. The 12-month model should show less val-test divergence. |

---

### 3.3 Dynamic Position Sizing with Risk Budget — ⏳ NOT STARTED

| | |
|---|---|
| **Problem** | The action space has fixed qty proportions `[-0.5, ..., +0.5]` as fractions of `max_position=1.0 BTC`. But position size should depend on current volatility and drawdown — trading max size in a high-vol environment is reckless. |
| **Proposed Change** | Implement a **Kelly/Vol-scaled position sizing overlay**: |

```python
# Modify env.step() action interpretation:
vol_30 = recent_30min_volatility  # From macro features
vol_scale = base_vol / max(vol_30, 1e-8)  # Inverse vol scaling
effective_max_position = max_position * min(vol_scale, 1.0)  # Cap at 1.0
quantity = abs(signed_qty) * effective_max_position
```

In high-vol regimes, `effective_max_position` shrinks, automatically reducing risk.

| | |
|---|---|
| **Expected Impact** | MaxDD drops significantly because the agent takes smaller positions when the market is volatile. **Expected: MaxDD from ~48% to <25%.** Sharpe may improve because drawdown drag is reduced. |
| **Risk** | Agent may learn to exploit the vol-scaling by intentionally creating volatility signals. However, since vol comes from future-unaware macro features, this is unlikely. |
| **Verification** | Compare position-size distributions across high-vol vs low-vol periods. MaxDD should improve. Risk-adjusted return (Calmar) should increase substantially. |

---

## Priority Matrix

| Rank | Item | Tier | Effort | Expected Impact | Status |
|------|------|------|--------|-----------------|--------|
| 🥇 | 1.2 Enable Augmentation | T1 | 2h | High (overfit reduction) | ✅ Done |
| 🥈 | 2.1 Boltzmann Exploration | T2 | 1d | High (buffer quality) | ❌ Reverted |
| 🥉 | 1.1 Fix Risk Metrics | T1 | 4h | Critical (diagnostics) | ⚠️ Partial |
| 4 | 2.2 Soft Target Updates | T2 | 4h | Medium (Q stability) | ✅ Done |
| 5 | 2.3 Reward Normalization | T2 | 4h | Medium (regime-robust) | ✅ Done |
| 6 | 1.5 Tune Epsilon Schedule | T1 | 1h | Medium (explore/exploit) | ✅ Done |
| 7 | 1.3 Q-Value Clipping | T1 | 1h | Low-Medium (stability) | ✅ Done |
| 8 | 2.5 Drawdown Penalty | T2 | 1d | High (risk) | ❌ Failed (Strategy) |
| 9 | 1.4 Remove hold_bonus | T1 | 5min | Low | ✅ Done |
| 10 | 2.5.1 Tune Drawdown (HPO) | T2 | 2d | High (fix S3→) | ⏳ IN PROGRESS |
| 11 | 2.6 Reward A/B (DSR vs Hybrid) | T2 | 2d | High (simplification) | ⏳ Planned |
| 12 | 2.4 Aux Head Redesign | T2 | 2d | Low-Medium (regularize) | ⏳ |
| 13 | 3.1 Walk-Forward Eval | T3 | 3d | Critical (honest eval) | ⏳ Planned |
| 14 | 3.2 Expand Dataset | T3 | 3d | High (generalization) | ⏳ |
| 15 | 3.3 Dynamic Position Sizing | T3 | 5d | High (risk mgmt) | ⏳ |

---

## Recommended Deployment Sequence

### Sprint 1 (Next Run — 1 day) — ✅ DONE
Apply items **1.1 + 1.2 + 1.4 + 1.5** (config + analytics fixes). One production run to establish new baseline with augmentation enabled.

### Sprint 2 (Following Run — 2 days) — ❌ REGRESSED → S2.5 + S3 applied
Add items **2.1 + 2.2** (Boltzmann exploration + soft target updates). These are the biggest behavioral changes and should show clear improvement in WandB metrics.

### Sprint 3 (Validation — 3 days) — ⏳ IN PROGRESS (S3→ next)
Add item **2.5** (drawdown penalty) and run under **3.1** (walk-forward with 2-3 windows). Item **2.3** (reward normalization) is already active. This sprint focuses on honest evaluation of the strategy.

### Sprint 3.5→ (Reward Experiment — 2 days) — ⏳ PLANNED
A/B test: **Pure DSR (Config B)** vs **High DSR (Config C)** vs **Current Hybrid (Config A)**. Config-only changes, no code edits. Run after S3.5 HPO completes to use the tuned drawdown factor as baseline.

### Sprint 4 (Scaling — 1 week) — ⏳ PLANNED
Items **3.2 + 3.3** (more data + dynamic sizing). Only pursue if Sprints 1-3 show the agent has a genuine edge across multiple walk-forward windows.

---

> [!IMPORTANT]
> **Do not skip item 3.1 (Walk-Forward)**. Every metric improvement is meaningless without multi-window validation. A single-window PF=1.3 could be luck. A 3-window average PF=1.15 is much stronger evidence.

> [!WARNING]
> **Q-values at ~24 are a red flag.** If Q-values continue rising after applying items 1.3 + 2.2, the Bellman target may have a systematic bias (e.g., hindsight bonus leaking into backtest evaluation, or stale target network drift). Monitor closely.
