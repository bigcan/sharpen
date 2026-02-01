# DeepScalper Hyperparameter Reference (RTX 5090 Optimization)

This document provides a comprehensive consolidated hyperparameter reference table showing PPO, A2C, and DQN algorithms side-by-side, specifically tuned for the RTX 5090 architecture (Blackwell).

![Hyperparameter Table](/brain/406783f0-d02d-42a7-8511-cf2beb318618/uploaded_media_1769944446197.png)

## Consolidated Configuration Table

| Parameter | PPO Default | PPO RTX5090 | A2C Default | A2C RTX5090 | DQN Default | DQN RTX5090 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **learning_rate** | 3e-4 | **3e-4 → 5e-5 (decay)** | 7e-4 | **7e-4 → 1e-4 (decay)** | 1e-4 | **1e-4 (fixed)** |
| **n_steps / batch_size** | 2048 | **3000–4000** | 5 | **15–25** | 32 | **128–256** |
| **n_epochs / n_envs** | 10 | **12–15** | N/A | **8–16 envs** | N/A | N/A |
| **gamma** | 0.99 | 0.99 (keep) | 0.99 | 0.99 (keep) | 0.99 | 0.99–0.995 |
| **gae_lambda / entropy** | 0.95 | 0.95 (keep) | 1.0 | 0.95–0.98 | N/A | N/A |
| **clip_range / ent_coef** | 0.2 | 0.2 (or 0.18–0.22) | 0.0 | **0.01–0.05** | N/A | N/A |
| **vf_coef / buffer_size** | N/A | N/A | 0.5 | 0.5–1.0 | 1M | **1–2M** |
| **max_grad_norm** | 0.5 | 0.5 (keep) | 0.5 | 0.5 (keep) | 10 | 10 (keep) |
| **target_update_interval** | N/A | N/A | N/A | N/A | 10,000 | **5,000–10,000** |
| **learning_starts** | N/A | N/A | N/A | N/A | 100 | **2,000–5,000** |
| **exploration_fraction** | N/A | N/A | N/A | N/A | 0.1 | 0.15–0.2 |
| **Mixed Precision** | Off | **Enable FP16** | Off | **Enable FP16** | Off | **Enable FP16** |
| **Ensemble Seeds** | N/A | **0, 42, 84 (distinct)** | N/A | **0, 42, 84 (distinct)** | N/A | **0, 42, 84 (distinct)** |

## Key Parameters Explained

### PPO Configuration
*   **learning_rate**: `3e-4 → 5e-5` (linear decay) — Critical for preventing late-stage oscillation.
*   **n_steps**: `3000–4000` — Larger batches exploit RTX 5090's GDDR7 bandwidth.
*   **batch_size**: `256–512` — Compute-bound training with Tensor Cores.
*   **n_epochs**: `12–15` — More optimization iterations per rollout.
*   **Mixed Precision**: Enable FP16 for 2–3x speedup.

### A2C Configuration
*   **learning_rate**: `7e-4 → 1e-4` (decay) — High sensitivity; critical parameter.
*   **n_steps**: `15–25` — Most important parameter for A2C stability.
*   **n_envs**: `8–16` parallel — A2C requires CPU vectorization (not GPU-friendly).
*   **ent_coef**: `0.01–0.05` — **CRITICAL**: Enable exploration (default 0.0 disables it).
*   **Status**: ❌ **NOT RECOMMENDED** for RTX 5090 due to GPU synchronization bottleneck.

### DQN Configuration
*   **learning_rate**: `1e-4` (fixed) — **DO NOT schedule**; breaks Q-learning stability.
*   **batch_size**: `128–256` — Leverage GDDR7 bandwidth for smooth gradient estimates.
*   **buffer_size**: `1–2M` — RTX 5090's 32GB supports 2M transitions.
*   **learning_starts**: `2,000–5,000` — Let exploration fill buffer before training.
*   **target_update_interval**: `5,000–10,000` — Balance stability vs. adaptation speed.
*   **Status**: ✅ **ALTERNATIVE** choice; excellent sample efficiency and ensemble stability.
