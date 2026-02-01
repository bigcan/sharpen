# Deep Diagnostic Request: DeepScalper "Doom Loop" Analysis

## Context
We just terminated a DeepScalper training run (`hlmj8s0t`) on the remote Blackwell server. The run exhibited "Zombie" behavior:
- **Process Status**: High CPU usage (>1700%) indicating active computation.
- **WandB Status**: API reported ~4.9M steps completed and "Running".
- **Real-time Status**: `check_step_growth.py` confirmed **0 steps advanced** over a 30-second window.
- **Logs**: The logs were flooded with `Hit Max Drawdown Stop (20%). Terminating Episode.` warnings.
- **Process Age**: The PID `395139` appeared to have an uptime of 29 days, while the WandB run claimed to be created 42 hours ago (massive discrepancy needing investigation).

## The Issue
The agent appears to be stuck in an infinite or extremely slow **Evaluation/Backtesting Loop**. It hits the "Max Drawdown" stop immediately, terminates the episode, and likely restarts the evaluation episode instantly without returning control to the training loop. This results in high CPU usage (running the environment steps) but zero training progress (frames aren't counting towards `total_timesteps` during eval, or the counter is stuck).

## Your Mission
Perform a deep code audit and diagnostic of the Training & Evaluation loop to fix this "Doom Loop".

### 1. Audit the Evaluation Logic
Inspect `finrl_pro_ds/training/deepscalper_trainer.py` (specifically `evaluate` method) and `finrl_pro_ds/analytics/wandb_evaluator.py`.
- **Look for**: A loop that retries failed evaluation episodes endlessly.
- **Look for**: Handling of `Hit Max Drawdown Stop` during evaluation. Does it count as a completed episode?
- **Verify**: Are `total_timesteps` updated during evaluation? (Likely not, which explains why WandB paused).

### 2. Investigate the "29 Day" PID Mystery
How could the process be 29 days old if the run was created 2 days ago?
- **Hypothesis**: Is it a zombie process from a previous run that `wandb` re-attached to?
- **Hypothesis**: Did the `deploy_bare_metal.py` script fail to clean up old processes?

### 3. Fix the "Max Drawdown" Trap
If the agent is bad (early training) and hits max drawdown instantly:
- It should NOT retry endlessly.
- It should log a poor reward (-1 or similar) and return to training to improve.
- **Action**: Implement a hard limit on evaluation episode retries or accept the failure as a valid data point.

## Relevant Files
- `finrl_pro_ds/training/deepscalper_trainer.py`
- `finrl_pro_ds/analytics/wandb_evaluator.py`
- `configs/deepscalper_unified.yaml`
- `scripts/deploy_bare_metal.py`

## Artifacts to Reference
- `scripts/check_step_growth.py` (The script that proved the stall)
- `scripts/compact_check.py` (Process diagnostics)

## Goal
Produce a fix for `deepscalper_trainer.py` that prevents the agent from getting stuck in a Max Drawdown loop, ensuring it always returns to training even if evaluation fails.
