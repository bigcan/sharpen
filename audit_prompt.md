# DeepScalper Pipeline Audit Prompt
**Zero Tolerance for Logic Errors & Bugs**

## Mission
Perform a **deep, intensive, thorough audit** of all DeepScalper pipeline updates implemented after run `hlmj8s0t`. Your goal is to identify any logic errors, bugs, edge cases, or deviations from best practices. **Zero tolerance for defects.**

## Scope of Audit
The following updates must be audited:

### 1. "Doom Loop" Fix (`deepscalper_trainer.py`)
**Location:** `DeepScalperTrainer.evaluate()` method
**Claimed Fix:** Replaced `np.all(dones)` infinite loop with robust per-environment episode counting.

**Audit Checklist:**
- [ ] Verify episode counting logic handles `VectorEnv` auto-resets correctly
- [ ] Confirm no off-by-one errors in episode termination condition
- [ ] Check edge case: What happens if `num_eval_episodes = 0`?
- [ ] Check edge case: What if an environment never terminates (infinite episode)?
- [ ] Verify metrics (rewards, steps) are correctly aggregated per episode
- [ ] Confirm no memory leaks from accumulated observations/actions

### 2. "Zombie Process" Fix (`deploy_bare_metal.py`)
**Location:** Process cleanup section (~L182-200)
**Claimed Fix:** Aggressive `pkill -f` for all DeepScalper scripts before deployment.

**Audit Checklist:**
- [ ] Verify the kill command syntax is correct for Linux
- [ ] Check if `|| true` properly suppresses errors when no process exists
- [ ] Confirm all relevant scripts are in the kill list (train, tune, backtest, wandb-service)
- [ ] Edge case: Does this accidentally kill the deployer script itself?
- [ ] Edge case: What if multiple deployments run simultaneously?

### 3. WandB Logging Fixes (`deepscalper_trainer.py`)
**Locations:** `update_ppo()`, `update_a2c()`, `update_gating()`

**Audit Checklist:**
- [ ] **PPO**: Confirm no duplicate `total_loss` accumulation (was bug, claimed fixed)
- [ ] **A2C**: Verify `return None` on NaN loss is handled correctly by callers
- [ ] **Gating**: Check that metric aggregation is simplified and correct
- [ ] Edge case: What if all losses in a batch are NaN?
- [ ] Edge case: What if accumulators are empty at log time (`np.mean([])` = NaN)?
- [ ] Verify `info.get("volatility_target")` has proper defaults

### 4. Hardware Optimization (TF32/FP16)
**Location:** `DeepScalperTrainer.__init__()` and update methods

**Audit Checklist:**
- [ ] Verify `torch.set_float32_matmul_precision('high')` is called at correct scope
- [ ] Confirm `autocast(..., dtype=torch.float16)` is explicitly set (not default BF16)
- [ ] Check all three update loops (PPO, A2C, Gating) use consistent precision
- [ ] Edge case: What happens on non-Blackwell GPUs that don't support TF32?

### 5. Ensemble Gating Weight Logging
**Locations:** `ensemble.py`, `backtest_deepscalper.py`, `wandb_evaluator.py`

**Audit Checklist:**
- [ ] Verify `predict()` return signature change doesn't break existing callers
- [ ] Confirm weight columns are correctly added to DataFrame (no length mismatch)
- [ ] Check stacked area chart in WandB has correct data format
- [ ] Edge case: What if gating weights sum to > 1.0 or < 1.0?
- [ ] Edge case: What if weights contain NaN values?

### 6. Architecture Verification
**Locations:** `run_full_pipeline.py`, `deepscalper_trainer.py`, `tune_deepscalper.py`

**Audit Checklist:**
- [ ] Verify Phase 1 (Specialists) correctly freezes gating network weights
- [ ] Verify Phase 2 (Gating) correctly freezes specialist network weights
- [ ] Confirm checkpoint path is correctly passed between phases
- [ ] Check HPO `phase="full"` correctly trains all components jointly
- [ ] Edge case: What if Phase 1 checkpoint doesn't exist when Phase 2 starts?

## Files to Audit
```
finrl_pro_ds/training/deepscalper_trainer.py
finrl_pro_ds/agents/deepscalper/ensemble.py
finrl_pro_ds/agents/deepscalper/dqn_agent.py
finrl_pro_ds/analytics/wandb_evaluator.py
scripts/deploy_bare_metal.py
scripts/run_full_pipeline.py
scripts/backtest_deepscalper.py
scripts/tune_deepscalper.py
configs/deepscalper_unified.yaml
```

## Deliverables
1. **Bug Report**: List each bug found with:
   - Location (file:line)
   - Severity (Critical/Medium/Low)
   - Root cause
   - Proposed fix

2. **Code Smell Report**: Any suboptimal patterns that could cause future issues

3. **Fix Implementation**: If bugs are found, implement and test fixes

4. **Verification**: Run smoke tests to confirm fixes don't introduce regressions

## Audit Standards
- **Zero tolerance** for logic errors
- **Zero tolerance** for unhandled edge cases
- **Zero tolerance** for type mismatches
- All `np.mean([])` or division-by-zero must have guards
- All optional dict keys must use `.get()` with defaults
- All loops must have termination guarantees
