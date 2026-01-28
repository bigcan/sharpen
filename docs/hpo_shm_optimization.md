# HPO Shared Memory Optimization Walkthrough

## 1. Challenge & Solution
The HPO sweep for DeepScalper was improperly initializing `AsyncVectorEnv` and reloading the 2GB dataset for every environment, leading to massive memory overhead and slow startup (bottlenecking throughput).

**Solution**: implemented **Shared Memory Vectorization**.
*   **Centralized Loading**: Main process loads DataFrame once and creates `/dev/shm` segments.
*   **Lightweight Workers**: 12 Worker processes attach to existing Shared Memory (zero-copy).
*   **Explicit Space Inference**: Manually inferred `observation_space` and `action_space` to bypass internal dummy environment creation errors.

## 2. Implementation Details

### Key Changes
*   **Gymnasium Upgrade**: Pinned `gymnasium==0.29.1` throughout the stack.
*   **Shared Memory**: Added logic to `tune_deepscalper.py` to create and clean up shared memory.
*   **AsyncVectorEnv**: Switched from `SyncVectorEnv` to `AsyncVectorEnv(context="spawn")` for true parallelism.

### Code Snippet
```python
# tune_deepscalper.py (Snippet)
if num_envs > 1:
    logger.info("Initializing AsyncVectorEnv (spawn)...")
    
    # Explicitly infer spaces to prevent double-init crash
    dummy_env = make_env(..., shared_memory_config=shm_config)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    env_train = gym.vector.AsyncVectorEnv(
        env_fns_train, 
        context="spawn",
        observation_space=obs_space,
        action_space=action_space
    )
```

## 3. Verification Results

### Stability Test (4 Envs) - PASSED
*   **Status**: Completed 12,800 steps successfully.
*   **Logs**: Confirmed `step_start` heartbeat and checkpoint saving.

### Full Scale Run (12 Envs "Mach 2") - RUNNING
*   **Status**: Initialized and Workers Attached.
*   **Throughput**: Expected > 5,000 steps/sec.
*   **Resources**: Memory usage stable (Shared Memory active).

## 4. Next Steps
1.  **Monitor**: Keep an eye on `run.log` via `remote_cmd.py` or `monitor_status.py`.
2.  **Analyze**: fetch `hpo.db` after ~20 trials (approx 2 hours).
3.  **Upgrade**: Once 12 Envs proven stable for full duration, attempt 24 Envs ("Mach 3") if higher throughput needed.
