# DeepScalper Pipeline — Round 3 Deterministic Extermination Audit

## CONTEXT FOR THE AUDITOR

You are performing **Round 3** of an adversarial audit on a Branching Dueling DQN (BDQ) trading pipeline. Two prior audit rounds found and fixed bugs. **Bugs keep emerging.** Your job is to be the FINAL auditor. You must trace every numeric path, every branch, every edge case, and every data flow **with concrete numbers**. No hand-waving. No "this looks correct." Every claim must have a worked example.

### What was already fixed (DO NOT TRUST — RE-VERIFY):
1. Sell-side `avg_price` update rewritten to 4-branch structure
2. `.copy()` restored in `_get_observation()`
3. `_feature_data` assignment added to all 3 branches in parquet_handler
4. Stale fallback defaults removed from backtest agent creation
5. HPO parameter routing made explicit with fail-loud ValueError
6. Float comparisons (`== 0`, `> 0`) replaced with tolerance (`1e-12`)
7. Dead `macro_features` variable removed from parquet_handler
8. Backtest agent creation harmonized with trainer constructor

**ASSUME ALL PRIOR FIXES ARE WRONG. Re-derive correctness from the code.**

---

## AUDIT METHODOLOGY

For EVERY check below, you must:
1. **Read the exact code** (cite file, line numbers)
2. **Trace with concrete numbers** (pick specific values, compute step by step)
3. **State the invariant** being tested
4. **Verdict**: ✅ PROVEN CORRECT / 🔴 BUG / 🟡 FRAGILE

---

## MODULE 1: ENVIRONMENT — `finrl_pro_ds/envs/deep_scalper_env.py`

Read the FULL file (all ~650 lines). Then perform every check below.

### 1.1 Constructor Invariants
- [ ] `__init__` reads config. Trace EVERY `config.get()` / `config[key]` call against `configs/deepscalper_rtx5090_production.yaml`. Are there mismatches between config key names used in code vs YAML?
- [ ] `max_position`: code reads from `config.get("action", {}).get("max_position", ...)` AND `config.get("max_position", ...)`. Verify precedence order. What if BOTH exist?
- [ ] `maker_fee` / `taker_fee` vs `transaction_fee`: Verify the fee selection logic. What if config has BOTH `maker_fee` AND `transaction_fee`? Which wins?
- [ ] `reward_config`: Is it `config.get("reward", {})` or `config.get("env", {}).get("reward", {})`? The YAML nests reward under `env:`. Does the env constructor receive the full config or just the `env:` sub-dict? Trace from `make_env()` in `run_full_pipeline.py` to see what's passed.
- [ ] `vol_proportions = [0.1, 0.25, 0.5, 0.75, 1.0]` — are these 5 bins? Does `action_space = MultiDiscrete([3, 5, 5])` match? The 5 price bins and 5 volume bins must be consistent with the arrays that index into them (lines ~380-395).

### 1.2 Order Execution — Buy Side (lines ~280-310)
Trace with THESE EXACT numbers. Write out every variable at every line.

**Test Vector B1**: `position=0, avg_price=0, balance=100000, fill_price=50000, exec_qty=0.5, fee=0`
- Expected: `position=0.5, avg_price=50000, balance=75000`

**Test Vector B2**: `position=0.5, avg_price=50000, balance=75000, fill_price=52000, exec_qty=0.3, fee=0`
- Expected: `position=0.8, avg_price=50750, balance=59400`

**Test Vector B3**: `position=-1.0, avg_price=48000, balance=148000, fill_price=47000, exec_qty=0.5, fee=0`
- Expected: `position=-0.5, avg_price=48000 (unchanged), balance=124500`

**Test Vector B4**: `position=-0.5, avg_price=48000, balance=124500, fill_price=47000, exec_qty=0.5, fee=0`
- Expected: `position=0, avg_price=0, balance=101000`

**Test Vector B5**: `position=-0.5, avg_price=48000, balance=124500, fill_price=47000, exec_qty=1.0, fee=0`
- Expected: `position=0.5, avg_price=47000, balance=77500`

**Test Vector B6**: `position=0, avg_price=0, balance=100, fill_price=50000, exec_qty=0.5, fee=0`
- Expected: cost=25000 > balance=100 → ORDER REJECTED, nothing changes

For EACH: trace through the actual code line by line. If the code produces a different result, that's a BUG.

### 1.3 Order Execution — Sell Side (lines ~320-370)
Same drill with concrete numbers.

**Test Vector S1**: `position=2.0, avg_price=50000, balance=0, fill_price=51000, exec_qty=1.0, fee=0`
- Expected: `position=1.0, avg_price=50000 (unchanged), balance=51000`

**Test Vector S2**: `position=1.0, avg_price=50000, balance=51000, fill_price=51000, exec_qty=1.0, fee=0`
- Expected: `position=0, avg_price=0, balance=102000`

**Test Vector S3**: `position=0, avg_price=0, balance=102000, fill_price=51000, exec_qty=1.0, fee=0`
- Expected: `position=-1.0, avg_price=51000, balance=153000`

**Test Vector S4**: `position=-1.0, avg_price=51000, balance=153000, fill_price=52000, exec_qty=0.5, fee=0`
- Expected: `position=-1.5, avg_price=51333.33, balance=179000`
- `avg_price = (1.0*51000 + 0.5*52000) / 1.5 = 77000/1.5 = 51333.33`

**Test Vector S5**: `position=0.3, avg_price=50000, fill_price=51000, exec_qty=1.0`
- Expected: `position=-0.7, avg_price=51000 (fill), flip from long to short`

**Test Vector S6 — FLOAT EDGE**: `position=0.1, avg_price=50000, fill_price=51000, exec_qty=0.1`
- After: position should be snapped to 0.0, avg_price=0
- Actually trace: `position = 0.1 - 0.1`. Is `0.1 - 0.1 == 0.0` in IEEE 754? YES (exact). But what about `position = 1.0 - 0.7 - 0.3`? That's `2.2204e-16`. Does the tolerance catch it?

### 1.4 Order Execution — Fee Calculation
- [ ] Buy: `fee = fill_price * exec_qty * taker_fee` or `maker_fee`? Which is used? Is it always taker for market orders?
- [ ] Sell: `fee = fill_price * exec_qty * taker_fee`? Verify.
- [ ] Slippage: How is `fill_price` computed from `best_bid/best_ask`? Is slippage applied additively or multiplicatively? Is slippage applied in the CORRECT direction (buy = higher, sell = lower)?
- [ ] **CRITICAL**: After fee deduction, does `balance` ever go negative? The buy-side has a `cost <= balance` guard but the fee is INCLUDED in cost. Verify: `cost = fill_price * exec_qty + fee` or `cost = fill_price * exec_qty` with fee deducted separately?

### 1.5 Position Limit Enforcement
- [ ] Buy side: `max(0, self.max_position - self.position)` — if `position = 4.5, max_position = 5.0`, can buy `0.5`. If `position = 5.0`, can buy `0`. If `position = -2.0`, can buy `7.0`. Is 7.0 correct? It allows going from -2 to +5. Is that intended?
- [ ] Sell side: `max(0, self.position - (-self.max_position))` = `max(0, self.position + self.max_position)`. If `position = -4.5, max_position = 5.0`, can sell `0.5`. If `position = 2.0`, can sell `7.0`. Cross-check symmetry with buy.
- [ ] What if `exec_qty` after proportion scaling is less than `lot_size`? Is there a minimum order size check?

### 1.6 Reward Calculation (lines ~400-460)
- [ ] `pnl = current_portfolio_value - prev_portfolio_value` — is `prev_portfolio_value` updated BEFORE or AFTER the order executes?
- [ ] Risk penalty: `risk_penalty_weight * abs(pnl)` when `pnl < 0`. Does it double-penalize? (pnl is already negative, then you subtract MORE)
- [ ] Transaction cost penalty: `cost_penalty_weight * abs(exec_qty) * mid_price` — is this applied even when no trade happens (action = HOLD)? It shouldn't be.
- [ ] Hindsight bonus: `hindsight_weight * position * (future_price - current_mid)` — when position is short, position < 0, future_price < current_mid → bonus is positive. CORRECT?
- [ ] Scaling: `reward *= scaling` — what's the default scaling? If it's 0, all rewards are zero. Check config.
- [ ] **CRITICAL**: `prev_portfolio_value` assignment. Is it set BEFORE `_get_portfolio_value()` is called for reward, or after? If set too early, the reward is always 0.

### 1.7 State Update Order (lines ~230-480)
Trace the EXACT sequence in `step()`:
```
1. _update_state(step_data)  — updates micro_window, best_bid/ask, macro
2. Process pending_order     — executes order, changes position/balance
3. Update private_window     — writes new position/balance to window
4. Process new action        — creates pending_order for NEXT step
5. Compute reward            — uses portfolio value
6. _get_observation()        — returns obs
```
- [ ] Is `prev_portfolio_value` set between steps 2 and 5? Or is it set at the END of the previous step? Trace EXACTLY where `self.prev_portfolio_value = ...` appears.
- [ ] When pending_order is None (first step or HOLD), does the code skip steps 2-3 cleanly?
- [ ] Can `step_data` be None mid-episode? When `handler.step()` returns None, is `terminated` set correctly?

### 1.8 Observation Space
- [ ] `micro_window` shape: `(window_size, num_micro_features)`. What is `num_micro_features`? Trace from `NUM_MICRO_FEATURES` constant. Is it 27? Count the actual columns from feature_engineering.
- [ ] `current_macro` shape: `(NUM_MACRO_FEATURES,)` = `(11,)`. Count the actual MACRO_COLS list.
- [ ] `private_window` shape: `(window_size, 2)`. The 2 features are position_normalized and balance_normalized. How are they normalized? Trace the normalization formula. Can it produce NaN or Inf?
- [ ] Position normalization: `position / max_position`. If `max_position = 0`? (Default is 1.0, so unlikely, but check.)
- [ ] Balance normalization: `balance / initial_balance`. If `initial_balance = 0`? Check constructor default.

### 1.9 Reset
- [ ] Does `reset()` zero out `position, balance, avg_price, prev_portfolio_value`?
- [ ] Does it reset `pending_order` to None?
- [ ] Does it call `handler.reset()`?
- [ ] Does it return `_get_observation()` (with `.copy()`)?
- [ ] Are `micro_window` and `private_window` re-initialized to zeros?

---

## MODULE 2: DATA — `finrl_pro_ds/data/parquet_handler.py`

Read lines 1-263 (load_data is the critical path).

### 2.1 Data Loading
- [ ] `pd.read_parquet(file_path)` — any error handling if file doesn't exist?
- [ ] Date filtering: `start_date` and `end_date` — are they inclusive or exclusive? Is the comparison `>=` and `<=`, or `>` and `<`?
- [ ] After filtering, if `df` is empty (no data in date range), what happens?

### 2.2 Feature Engineering Integration
- [ ] `process_micro(df)` is called. Does it modify `df` in-place? If yes, does `self._feature_data = df` capture those changes?
- [ ] After `process_micro`, which columns exist in `df`? List them. Do they match `NUM_MICRO_FEATURES = 27`?
- [ ] The 27 micro features: 5 levels × (bid_price, bid_vol, ask_price, ask_vol) = 20, plus OFI (5), plus spread (1), plus log_ret (1) = 27. Verify this count against the actual code.

### 2.3 Macro Feature Branches (lines ~100-155)
- [ ] Branch 1 condition: `all(col in df.columns for col in env_macro_cols)`. What is `env_macro_cols`? Is it the same as `MACRO_COLS` from deep_scalper_env.py?
- [ ] Branch 2 condition: OHLCV columns exist. What processing happens? Is the result stored in `df` or a separate variable?
- [ ] Branch 3 (else): Warning printed. `_feature_data = df` still set? Verify.
- [ ] After all branches: `self._feature_cols = self._feature_data.columns.tolist()` — does this include timestamp? Should it? (Timestamp is not a numeric feature.)

### 2.4 Numpy Conversion (lines ~155-200)
- [ ] `self._data_array = self._feature_data[self._feature_cols].values` — is this float64? Is it contiguous? Does it contain NaN?
- [ ] Macro array: `self._macro_data` — shape and dtype. Is it aligned with `_data_array` by index?
- [ ] Volatility array: `self._vol_data` — pre-computed rolling volatility. Shape? Any NaN at boundaries?

### 2.5 Step/Reset Pointer Logic
- [ ] `reset()`: sets `self._ptr = window_size`. Why `window_size`? (Answer: first `window_size` rows are needed for the initial window.)
- [ ] `step()`: returns `self._data_array[self._ptr]` and increments `_ptr`. What if `_ptr >= len(_data_array)`? Returns None? Check.
- [ ] `get_lookahead_price(horizon)`: `self._ptr + horizon - 1`. Off-by-one? If `_ptr` was just incremented in `step()`, does this point to the right row?
- [ ] `get_lookahead_volatility(horizon)`: Same off-by-one check.

---

## MODULE 3: FEATURE ENGINEERING — `finrl_pro_ds/data/feature_engineering.py`

### 3.1 Normalization
- [ ] `_add_normalized_features()`: Which columns are normalized? What method? Z-score? Min-max? Clip?
- [ ] `log_ret` calculation: `np.log(mid / mid.shift(1))`. First row is NaN. Is it filled? With what? (Should be 0.)
- [ ] Division-by-zero: `mid_price = (bid_price_1 + ask_price_1) / 2`. If both are 0? Is there a guard?
- [ ] OFI calculation: Sum of bid_vol_i - ask_vol_i for i in 1..5? Or something else? Trace the exact formula.
- [ ] Are normalized features clipped to prevent extreme values? What range?

### 3.2 Macro Features (process_macro)
- [ ] Z-score normalization: `(x - mean) / std`. Rolling or global? If rolling, what window? If std=0, does it produce Inf/NaN?
- [ ] `zd_k` features: Z-scored return over k periods. Formula: `(close - close.shift(k)) / close.shift(k)` then z-score? Verify.

---

## MODULE 4: AGENT — `finrl_pro_ds/agents/deepscalper/bdq_agent.py`

### 4.1 Constructor
- [ ] `action_dims` parameter: default is `(3,5,5)`. Assertion checks `action_dims == network_config.get('action_space_dims', (3,5,5))`. If network was built with `(3,7,7)` but agent is called with default → assertion fires. GOOD. But what if NEITHER specifies it and both use default `(3,5,5)`? Silent match — correct behavior.
- [ ] `use_amp`: On CPU, does `torch.amp.GradScaler("cuda")` work? Or does it error? Check for CPU guard.
- [ ] `epsilon_start`, `epsilon_end`, `epsilon_decay`: Is epsilon updated per step or per episode? Trace `update_epsilon()`.

### 4.2 Action Selection
- [ ] `select_action(state, deterministic=False)`: 
  - Random action: `[random.randint(0, d-1) for d in self.action_dims]`. Correct range? randint is inclusive on both ends. So `randint(0, 2)` for 3 bins gives 0,1,2. Correct.
  - Greedy action: `q_values[i].argmax().item()` for each branch. Shape of `q_values`? It should be `(num_branches,)` where each element is a tensor of shape `(num_actions_in_branch,)`.
- [ ] `deterministic=True` skips epsilon check. Always greedy. Used in backtest. Correct.

### 4.3 Training Step
- [ ] `train_step()` returns None if `len(memory) < batch_size`. Correct guard.
- [ ] Double DQN: `next_actions = online_net(ns).argmax()`, `target_q = target_net(ns).gather(next_actions)`. Verify this is Double DQN (online selects, target evaluates), NOT vanilla DQN (target selects AND evaluates).
- [ ] Loss: Is it Huber loss or MSE? Check.
- [ ] Auxiliary loss: Volatility prediction. `aux_pred` from network vs `aux_target` from buffer. MSE loss? Weighted by `auxiliary_weight`?
- [ ] Gradient clipping: Is it applied? What value?
- [ ] Target network update: `target_update_freq` — soft update or hard copy? Trace the code.
- [ ] AMP (mixed precision): `autocast` context and `scaler.step()`. If `scaler` is CPU-based, does `scaler.step(optimizer)` work? Or does it skip?

### 4.4 Replay Buffer
- [ ] `push()`: Stores `(state, action, reward, next_state, done, aux_target)`. `state` is a dict. Is the dict itself stored, or a deep copy?
- [ ] `sample()`: Returns `random.sample(self.buffer, batch_size)`. This returns REFERENCES to stored tuples. If the buffer overwrites the slot later (ring buffer), does the sampled reference still point to valid data?
- [ ] **CRITICAL**: Is the buffer a ring buffer or a list that grows? Check: `self.buffer = []`, `self.position = 0`. If `len(buffer) < capacity`, it appends. If `len >= capacity`, it overwrites `buffer[position]`. The `random.sample` returns references to list elements. When `buffer[position]` is overwritten, does the old reference survive? YES — Python lists store references, not values. Overwriting `buffer[i] = new_tuple` makes `buffer[i]` point to `new_tuple`, but any previous reference to the old tuple still holds. SAFE.
- [ ] `stack_dict_keys()`: Converts list of state dicts to batched numpy arrays. `np.array([s[key] for s in states])`. If states have different shapes (e.g., one env had different window_size), this would fail with a ragged array. Is this possible? (No — all envs share the same config. But verify.)

### 4.5 Save/Load
- [ ] `save(path)`: Saves `online_net.state_dict()`, `optimizer.state_dict()`, `epsilon`. Does it save the target net? If not, after load, is `target_net` synced from `online_net`?
- [ ] `load(path)`: Loads into `online_net`, sets `epsilon`. Does it call `self.target_net.load_state_dict(self.online_net.state_dict())`? If not, target net is STALE after load—affects training if resumed.

---

## MODULE 5: NETWORK — `finrl_pro_ds/agents/deepscalper/networks.py`

### 5.1 Architecture Dimensions
Trace with production config values:
- `micro_input_size = 27`, `hidden_size = 256`, `num_layers = 1`, `rnn_type = "LSTM"`
- `macro_input_size = 11`, `macro_hidden_sizes = (128, 128)`
- `private_input_size = 2`
- `action_space_dims = (3, 5, 5)`

- [ ] MicroEncoder output shape: `(batch, hidden_size)` = `(batch, 256)`. The LSTM processes `(batch, window, 27)` and returns last hidden state. Verify `h[-1]` indexing.
- [ ] MacroEncoder output shape: `(batch, 128)` (last hidden layer).
- [ ] PrivateEncoder: How is it handled? Is it part of MicroEncoder? Check `private_input_size` usage. Is the LSTM input `27 + 2 = 29` or are they separate streams?
- [ ] Fusion: How are micro, macro, private embeddings combined? Concatenation? Addition? What's the total dimension going into the Q-head?
- [ ] Dueling architecture: Value head outputs scalar, Advantage heads output `(num_actions,)` per branch. `Q = V + A - mean(A)`. Verify this formula in code.
- [ ] Number of action branches: `len(action_space_dims) = 3`. Each branch has its own advantage head. Verify the `nn.ModuleList` length.

### 5.2 Forward Pass
- [ ] Input dict: `{"micro": (B, W, 27), "macro": (B, 11), "private": (B, W, 2)}`. Does the network handle all three? Or does it only use micro and macro?
- [ ] Output: List of Q-value tensors, one per branch. Shapes: `[(B, 3), (B, 5), (B, 5)]`. Verify.
- [ ] Auxiliary output: `aux_pred` scalar for volatility prediction. Shape: `(B, 1)`. Verify.

---

## MODULE 6: TRAINER — `finrl_pro_ds/training/deepscalper_trainer.py`

### 6.1 VectorEnv Setup
- [ ] `AsyncVectorEnv` vs `SyncVectorEnv`: Which is used? Is `num_envs` read from config?
- [ ] `shared_memory`: Config option? If True, does it use SharedMemory for observations? Does this affect `.copy()` necessity?

### 6.2 Training Loop (lines ~62-165)
- [ ] `obs, info = env.reset()` — `obs` is dict of batched arrays: `{micro: (num_envs, W, 27), ...}`
- [ ] `obs = {k: v[i] ...}` — WRONG. Re-read the actual code. Is it `s = {k: v[i] ...}` inside a loop over envs?
- [ ] `agent.select_action(s)` — receives single-env state. Returns list of 3 ints.
- [ ] `env.step(actions)` — `actions` is array of shape `(num_envs, 3)`. Is it constructed correctly from per-env action lists?
- [ ] After step: `for i in range(num_envs): agent.memory.push(s[i], a[i], r[i], ns[i], d[i], aux[i])`. Is `aux` computed? Where does the auxiliary target (volatility) come from? Is it from `info`?
- [ ] `agent.train_step()` — called every step? Or every N steps? Check frequency.
- [ ] `agent.update_epsilon()` — called every step? Check.
- [ ] Target network update: Is `agent.maybe_update_target()` called? Or is it inside `train_step()`?

### 6.3 Logging
- [ ] WandB log: Are the keys (`train/reward_mean`, `train/loss_total`, etc.) consistent between what's logged and what's read in HPO/backtest?
- [ ] Episode tracking: When an env finishes (done=True), is the episode reward computed correctly? Is it sum or mean of step rewards?

---

## MODULE 7: PIPELINE — `scripts/run_full_pipeline.py`

### 7.1 Config Flow
Trace the config from YAML load to each consumer:
```
YAML → base_config (dict) → copy.deepcopy → final_config
                                   ↓
                          merge_configs(final_config, best_params)
                                   ↓
                      ┌────────────┼────────────┐
                      ↓            ↓            ↓
                 make_env()   Trainer()    run_backtest()
```
- [ ] `make_env()` receives `config` and passes `config["env"]` to DeepScalperEnv? Or the full config? TRACE THIS.
- [ ] If `make_env()` passes `config["env"]`, then the env constructor's `config.get("reward")` reads from `config["env"]["reward"]`. But the YAML has `env: reward:`. So `config["env"]["reward"]` is correct. Verify.
- [ ] If `make_env()` passes the FULL config, then `config.get("reward")` would fail (it's nested under `env`). Which is it?

### 7.2 HPO Objective (lines ~254-350)
- [ ] `trial.suggest_*` calls: List ALL of them with their ranges.
- [ ] Each suggested param is set into `config["env"]["reward"][key]` or `config["agents"]["bdq"][key]`. Verify every assignment.
- [ ] The trial runs training. What metric is returned? `study.optimize(objective, ...)` — is the objective maximized or minimized? What direction is set? Does the returned metric match the direction?
- [ ] Pruning: Is Optuna pruning enabled? `trial.report(value, step)` — is `value` the same metric used for optimization?

### 7.3 Post-HPO Routing (lines ~355-390)
- [ ] `best = study.best_trial` — `best.params` contains ALL `trial.suggest_*` params.
- [ ] Routing sets: `reward_params`, `agent_params`. Are they exhaustive?
- [ ] `reward_key_map`: Maps `"cost_penalty" → "transaction_cost_penalty"`. Does the env constructor read `transaction_cost_penalty` or `cost_penalty` from its reward config? TRACE THE EXACT LINE.
- [ ] After routing, `merge_configs(final_config, best_params_dict)`. Does this deep-merge correctly? Test: if `final_config["env"]["reward"]` has keys A, B, C and `best_params_dict["env"]["reward"]` has keys B, D, does the result have A, B(new), C, D?

### 7.4 Backtest Agent Creation (lines ~490-520)
- [ ] Does it pass `action_dims` from config? (Was fixed in Round 2 — VERIFY the fix is present.)
- [ ] Does it pass `use_amp` from config?
- [ ] Does it call `agent.load(checkpoint_path)`? What does `load()` do to epsilon?
- [ ] After load, is `deterministic=True` used for all action selection in backtest?
- [ ] Backtest env: Is it a single env (not VectorEnv)? Is it created with the SAME config as training?

### 7.5 Backtest Metrics (lines ~520-600)
- [ ] Sharpe ratio: Is it computed on step-level returns or episode-level? What's the annualization factor? For HFT tick data, this matters enormously.
- [ ] Max drawdown: Computed from portfolio value series? Peak-to-trough?
- [ ] Are metrics logged to WandB AND returned?

---

## MODULE 8: CROSS-CUTTING NUMERIC INVARIANTS

### 8.1 Portfolio Accounting Invariant
At ALL times, this must hold:
```
portfolio_value = balance + position * mid_price
```
- [ ] After a BUY of qty at price p with fee f:
  - `balance -= (p * qty + f)`
  - `position += qty`
  - New PV = `(balance - p*qty - f) + (position + qty) * mid`
  - If `mid ≈ p`: PV change ≈ `-f` (loss from fee only). CORRECT?

- [ ] After a SELL of qty at price p with fee f:
  - `balance += (p * qty - f)`
  - `position -= qty`
  - New PV = `(balance + p*qty - f) + (position - qty) * mid`
  - If `mid ≈ p`: PV change ≈ `-f` (loss from fee only). CORRECT?

- [ ] **CRITICAL**: For SHORT positions, PV = balance + position * mid = balance - |position| * mid. As mid increases, PV DECREASES (loss). As mid decreases, PV INCREASES (profit). This is correct short behavior. VERIFY the code computes this, not `balance - position * mid`.

### 8.2 NaN/Inf Propagation
- [ ] Feature engineering: Any division by zero? (mid_price=0, std=0, volume=0)
- [ ] Normalization: Division by `initial_balance` (could be 0?), `max_position` (could be 0?)
- [ ] Reward: Division anywhere? `scaling` could be 0?
- [ ] Network: Any `log()` or `sqrt()` that could produce NaN?
- [ ] Loss: If Q-values are very large, could Huber loss overflow?

### 8.3 Dtype Consistency
- [ ] Are observation arrays always float32? The network expects float32. If the env returns float64, does the conversion happen?
- [ ] `observation_space` dtype: Is it `np.float32`? Does `_get_observation()` return arrays matching this dtype?
- [ ] Reward: Is it a Python float? Or numpy scalar? Does VectorEnv handle both?

### 8.4 Seed Reproducibility
- [ ] `env.reset(seed=...)` — is the seed propagated to numpy, random, and torch?
- [ ] `random.sample()` in replay buffer — is it seeded?
- [ ] `np.random` in feature engineering — is it seeded?

---

## MODULE 9: CONFIGURATION CONSISTENCY

Open `configs/deepscalper_rtx5090_production.yaml` and verify:

- [ ] `network.micro_config.input_size` matches `NUM_MICRO_FEATURES` in env
- [ ] `network.macro_config.input_size` matches `NUM_MACRO_FEATURES` in env
- [ ] `network.micro_config.private_input_size` matches the private feature count (2)
- [ ] `network.action_space_dims` matches `env.action.direction_bins`, `price_bins`, `volume_bins`
- [ ] `agents.bdq.buffer_size` is large enough for `training.total_timesteps / num_envs` (or at least > batch_size)
- [ ] `env.reward.scaling` is not 0
- [ ] `env.initial_balance` is not 0
- [ ] `env.action.max_position` is not 0

---

## MODULE 10: TEST SUITE GAPS

After all auditing, check `tests/` for coverage:

- [ ] Is there a test for EVERY order execution scenario (B1-B6, S1-S6)?
- [ ] Is there a test for reward calculation with known inputs?
- [ ] Is there a test for portfolio accounting invariant (PV = balance + pos * mid)?
- [ ] Is there a test for feature engineering producing correct shapes and no NaN?
- [ ] Is there a test for the full step() → train_step() → action loop?
- [ ] Is there a test for save/load roundtrip (weights + epsilon)?
- [ ] Is there a test for the config merge function?

If any test is missing, FLAG it with the exact test code that should be added.

---

## OUTPUT FORMAT

### For each check:
```
### [Module].[Section] — [Check Name]
**File**: path, lines X-Y
**Invariant**: [what must be true]
**Trace**: [step-by-step computation with concrete numbers]
**Verdict**: ✅/🔴/🟡
**Evidence**: [code citation]
```

### Bug Report Template:
```
### 🔴 BUG: [Title]
**File**: path:line
**Severity**: Critical/Medium/Low
**Trigger**: [exact scenario]
**Expected**: [correct behavior]
**Actual**: [buggy behavior]
**Root Cause**: [why]
**Fix**:
\```python
# Before (buggy):
...
# After (fixed):
...
\```
```

### Final Summary:
```
Checks performed: X
Bugs found: Y (Z critical, W medium, V low)
Fragile spots: N
Missing tests: M
Overall health: [SHIP IT / NEEDS FIXES / DO NOT DEPLOY]
```

---

## RULES
1. You MUST read every file listed. No skipping.
2. You MUST trace with the exact test vectors provided. No substituting "similar" numbers.
3. If a check passes, explain WHY with line citations.
4. If you find yourself writing "this looks correct" without a worked example, STOP and add one.
5. Pay attention to the BOUNDARIES: first step, last step, empty data, zero values, max values.
6. Every `config.get(key, default)` is suspicious. The default might not match production.
7. Every `if/elif/else` must have all branches traced.
8. Floating point: `0.1 + 0.2 != 0.3`. Always check.
9. Off-by-one: `range(n)` is `[0, n-1]`. `array[ptr]` after `ptr += 1` skips the first or last element?
10. DO NOT STOP until you have checked every single box above.
