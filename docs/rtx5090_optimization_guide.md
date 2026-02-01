# RTX 5090 Optimization Guide for DeepScalper

**Date:** Feb 1, 2026
**Hardware:** NVIDIA RTX 5090 (Blackwell Architecture)
**Context:** FinRL-Pro_DS

## Executive Summary
This document outlines the critical software and configuration optimizations required to unlock the performance of the RTX 5090 for Financial Reinforcement Learning (FinRL). Based on the report *"Precision Optimization for FinRL on NVIDIA RTX 5090: Architectural Analysis and Numerical Strategy"*, we have shifted away from the industry standard BF16 (used in LLMs) to a **Twinned TF32/FP16** strategy to prevent mathematical instability in policy gradients.

## 1. Precision Strategy: The "Mantissa First" Rule

### The Trap: BFloat16 (BF16)
*   **Architecture:** 1 Sign, 8 Exponent, **7 Mantissa**.
*   **Result:** The 7-bit mantissa provides insufficient resolution for the small probability differences in financial PPO ratios ($r_t(\theta)$).
*   **Symptom:** "Training-Inference Mismatch" where rounding errors trigger false clipping ($\epsilon=0.2$) or entropy collapse.
*   **Verdict:** **PROHIBITED** for DeepScalper.

### The Solution: TensorFloat-32 (TF32) + FP16
*   **TensorFloat-32 (TF32):**
    *   **Architecture:** 10-bit Mantissa (matches FP16) + 8-bit Exponent (matches FP32).
    *   **Enforcement:** `torch.set_float32_matmul_precision('high')`.
    *   **Benefit:** Uses 5th Gen Tensor Cores for matrix multiplication while maintaining the precision floor required for financial signals.
*   **Float16 (FP16) Mixed Precision:**
    *   **Architecture:** 10-bit Mantissa.
    *   **Enforcement:** `autocast(dtype=torch.float16)`.
    *   **Benefit:** 24x reduction in sequence-level log-prob mismatch compared to BF16. Requires `GradScaler` to handle dynamic range.

## 2. Implementation

### Code Changes (`DeepScalperTrainer`)
```python
# Initialization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')  # Enable TF32

# Training Loop
with torch.amp.autocast('cuda', dtype=torch.float16):  # Enforce FP16 (Not BF16)
    logits, values = network(states)
    loss = ...
scaler.scale(loss).backward()  # Must use GradScaler
```

### Configuration Changes (`deepscalper_unified.yaml`)
To leverage the 32GB GDDR7 memory (1,792 GB/s bandwidth):
*   **Batch Size:** Increased to **16,384** (from 2048/4096).
*   **Update Interval:** Matched to 16,384.

## 3. Performance Expectations
*   **Throughput:** Linear scaling expected vs RTX 4090.
*   **Stability:** Entropy collapse issues should be resolved by the higher mantissa precision of FP16/TF32.
*   **Throughput Target:** >100,000 SPS (estimated).
