# Performance Investigation: DeepScalper Training Bottleneck

**Role:** Python Systems Performance Engineer  
**Date:** 2026-01-27  
**Priority:** HIGH  

---

## Situation

The DeepScalper (V10) training pipeline is running on GPUHub but at **48 FPS** instead of the certified **32,000+ FPS** ("Mach 3"). This represents a **~660x performance regression**.

### Current Run
- **WandB:** [DS_GPUHub_CleanTorch_20260127_135446](https://wandb.ai/bigcan-chiwin-technology/FinRL-Pro-DS/runs/je5wuoor)
- **Host:** GPUHub (RTX 5090, 32GB VRAM, 754GB RAM)
- **Elapsed:** 38 minutes → 110k steps (48 FPS)
- **GPU Utilization:** 4% (severely underutilized)

### Expected Performance (from randd_log.md)
- **Target:** >32,000 steps/second
- **Runtime for 10M steps:** ~14 hours
- **Current projection:** 57.6 hours (~2.4 days)

---

## Hypothesis Space

### 1. torch.compile Cold Start
**Symptom:** First N minutes are slow due to JIT compilation.  
**Check:** Look at step timing over time—does FPS increase after initial warmup?

### 2. AsyncVectorEnv Not Active
**Symptom:** Training uses single-threaded environment instead of 24-worker parallel.  
**Check:** Verify `num_envs` config and whether `AsyncVectorEnv` is instantiated in `train_deepscalper.py`.

### 3. Data Handler Bottleneck
**Symptom:** `ParquetDataHandler.step()` is blocking on disk I/O or using slow `iloc`.  
**Check:** Confirm NumPy-vectorized `_data_arrays` dict is being used (not DataFrame iloc).

### 4. Missing AMP (Automatic Mixed Precision)
**Symptom:** FP32 training underutilizes Tensor Cores.  
**Check:** Verify `use_amp=True` in config and `GradScaler`/`autocast` are active in trainer.

### 5. Incorrect Step Counting
**Symptom:** Training loop counts wrong (e.g., iterating total_timesteps instead of total/num_envs).  
**Check:** Review main training loop logic for `global_step` increment pattern.

### 6. Excessive Logging/WandB Overhead
**Symptom:** Synchronous logging every step blocks training.  
**Check:** Verify logging intervals are reasonable (e.g., every 1000 steps, not every step).

---

## Relevant Files

| File | Purpose |
|------|---------|
| `scripts/train_deepscalper.py` | Main training entry point |
| `finrl_pro_ds/training/deepscalper_trainer.py` | Core training loop |
| `finrl_pro_ds/data/parquet_handler.py` | Data streaming (potential bottleneck) |
| `finrl_pro_ds/envs/deepscalper_env.py` | RL environment |
| `configs/deepscalper_unified.yaml` | Unified configuration |
| `randd_log.md` | Historical performance benchmarks |

---

## Task

1. **Profile the Training Loop**
   - Identify which phase (env.step, network forward, backward, logging) takes the most time.
   - Check if `torch.compile` is enabled and if warmup has completed.

2. **Verify Parallelization**
   - Confirm `num_envs=24` and `AsyncVectorEnv` is being used.
   - Check for any `spawn` vs `fork` multiprocessing issues.

3. **Audit Data Pipeline**
   - Verify `ParquetDataHandler` is using NumPy dict access, not DataFrame iloc.
   - Check if shared memory is being used for multi-worker data access.

4. **Check Config Propagation**
   - Ensure `use_amp`, `torch_compile`, `num_envs` from YAML are actually being applied.
   - Look for config key mismatches or silent fallbacks to defaults.

5. **Propose Fix**
   - Identify the root cause of the 660x slowdown.
   - Implement the fix and validate FPS returns to >10,000 steps/sec minimum.

---

## Constraints

- **Do NOT stop the current run** unless necessary for diagnosis.
- RTX 5090 requires CUDA 12.x / PyTorch 2.x compatibility.
- Dependencies are pinned: `numpy<2.0.0`, `pyarrow==14.0.1`, `pandas<2.2.0`.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| FPS | >10,000 steps/sec (acceptable) / >30,000 (optimal) |
| GPU Utilization | >50% |
| ETA for 10M steps | <24 hours |
