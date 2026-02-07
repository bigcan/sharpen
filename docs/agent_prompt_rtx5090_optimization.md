# RTX 5090 Optimization Guide: A2C, SAC & PPO Agents

> **Purpose:** This document is a **standalone prompt** for an AI coding agent tasked with implementing or optimizing A2C, SAC, or PPO reinforcement learning agents on NVIDIA RTX 5090 (Blackwell) hardware. It contains all necessary context — no external references required.
>
> **Hardware Target:** RTX 5090 — 32GB GDDR7, 1.8 TB/s bandwidth, 5th Gen Tensor Cores, Blackwell architecture.
>
> **Project Context:** FinRL-Pro_DS — High-frequency cryptocurrency trading (BTC LOB data, ~1Hz resolution). Single-agent BDQ is already optimized. This guide covers the three remaining agent architectures.

---

## 1. Universal Precision Rules (ALL Agents)

These rules apply to **every** agent regardless of algorithm. They are non-negotiable on RTX 5090.

### 1.1 FP16 Mixed Precision — MANDATORY

```python
# ✅ CORRECT — Explicit FP16
with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
    # Forward pass + loss computation here
    ...

# ❌ WRONG — Defaults to BF16 on Blackwell
with torch.amp.autocast(device_type="cuda", enabled=True):
    ...
```

**Why:** On Blackwell/Ada GPUs, `torch.amp.autocast` defaults to `torch.bfloat16`, NOT `torch.float16`. BF16 has a 7-bit mantissa (vs FP16's 10-bit), which causes:
- **Policy entropy collapse** in PPO/A2C (action probabilities lose precision)
- **Q-value drift** in SAC (soft Q-functions accumulate rounding error)
- **Training-inference mismatch** (inference runs FP32, training ran BF16)

### 1.2 GradScaler — REQUIRED with FP16

```python
self.scaler = torch.amp.GradScaler()

# Training step:
with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
    loss = compute_loss(...)

self.optimizer.zero_grad()
self.scaler.scale(loss).backward()
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
self.scaler.step(self.optimizer)
self.scaler.update()
```

**Why:** FP16 has a narrow dynamic range (6e-8 to 65504). Without scaling, small gradients underflow to zero. The `GradScaler` dynamically scales the loss to keep gradients in FP16's representable range.

**Note:** If you see `scaler.get_scale()` decreasing to very small values (<1.0), it means FP16 is unstable for your loss magnitudes. This is a signal, not an error — the scaler is working correctly.

### 1.3 TF32 MatMul — MANDATORY

```python
# Call ONCE at program startup, before any tensor operations
torch.set_float32_matmul_precision('high')
```

**Why:** Without this, all FP32 matrix multiplications use full IEEE FP32 (23-bit mantissa). TF32 uses 10-bit mantissa — identical precision to FP16 but with FP32's exponent range. On 5th Gen Tensor Cores, TF32 is **~8x faster** than IEEE FP32 with no measurable accuracy loss for RL.

### 1.4 FP8 — DO NOT USE

FP8 (`torch.float8_e4m3fn`, `Float8Linear`) is designed for LLM inference quantization. It has a **3-bit mantissa** — completely inadequate for:
- Policy gradient estimation (PPO/A2C)
- Soft Q-function approximation (SAC)
- Any RL where reward signals are small relative to noise

**Rule:** Do not use FP8 anywhere in the training pipeline. Not in networks, not in buffers, not in normalization.

### 1.5 torch.compile — USE WITH CAUTION

```python
# ✅ Safe: Compile the policy network
self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")

# ⚠️ Risky: Don't compile the entire training step
# torch.compile struggles with dynamic control flow (epsilon-greedy, masking, etc.)
```

**Guidance:**
- Compile **forward-pass-only** networks (actor, critic, value)
- Don't compile functions with branching, dynamic shapes, or Python-side logic
- First training step will be slow (compilation). Subsequent steps benefit.
- Use `mode="reduce-overhead"` for RL (small batches, frequent calls)

---

## 2. PPO (Proximal Policy Optimization)

### 2.1 Algorithm Profile

| Property | Value |
|---|---|
| **Family** | On-Policy, Policy Gradient |
| **Data usage** | Collect rollout → train → **discard everything** |
| **Key insight** | Clipped surrogate objective prevents catastrophic policy updates |
| **Bottleneck on 5090** | Memory bandwidth (large rollout buffers) |

### 2.2 Batch Size — THE Critical Setting

```yaml
ppo:
  # RTX 5090 Optimized
  n_steps: 2048              # Steps per env before update
  num_envs: 12               # Parallel environments (GPUHub limit)
  batch_size: 16384           # Mini-batch for gradient computation
  n_epochs: 10               # Passes over the rollout buffer
```

**Math:**
```
Rollout buffer size = n_steps × num_envs = 2048 × 12 = 24,576 transitions
Mini-batches per epoch = rollout_size / batch_size = 24,576 / 16,384 ≈ 1.5 → 2
Total gradient updates = n_epochs × mini-batches = 10 × 2 = 20 updates per rollout
```

**Why 16K:**
- PPO gets **one chance** to learn from each rollout before discarding it
- Small batches (256) produce noisy gradients → wasted rollout data
- The RTX 5090's 1.8 TB/s bandwidth can feed 16K samples to Tensor Cores without stalling
- Beyond 32K, gradient variance stops decreasing (diminishing returns)

**Learning Rate Scaling:**
```python
# Linear scaling rule: if you double batch_size, double LR
base_lr = 3e-4       # For batch_size = 256
scaled_lr = 3e-4 * (batch_size / 256)  # = 1.92e-2 for batch_size=16384

# In practice, cap at ~3e-3 and use warmup
lr = min(scaled_lr, 3e-3)
```

### 2.3 PPO-Specific Precision Considerations

```python
class PPOAgent:
    def compute_loss(self, rollout_batch):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            # Actor forward pass
            action_logits = self.actor(states)
            dist = Categorical(logits=action_logits)  # or Normal for continuous
            
            # ⚠️ CRITICAL: Log-prob computation in FP16 can lose precision
            # for very unlikely actions. This is acceptable — the clipping
            # objective (clip_range=0.2) prevents these from dominating anyway.
            new_log_probs = dist.log_prob(actions)
            
            # Ratio and clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss (can use FP16 safely — values are typically in [-100, 100] range)
            value_pred = self.critic(states)
            value_loss = F.mse_loss(value_pred, returns)
            
            # Entropy bonus (FP16 safe — entropy is a scalar summary)
            entropy = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return loss
```

**PPO Gotcha — GAE Computation:**
```python
# ⚠️ DO NOT put GAE (Generalized Advantage Estimation) inside autocast
# GAE is a sequential scan with cumulative products — FP16 accumulation
# causes catastrophic drift over long horizons (2048 steps)

# ✅ CORRECT: Compute GAE in FP32, then cast to FP16 for training
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # This runs in FP32 (no autocast)
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages  # FP32 tensor, cast happens inside autocast block
```

### 2.4 PPO RTX 5090 Config Template

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 16384
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null        # Don't clip value loss (let it learn freely)
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  
  # RTX 5090 Hardware
  num_envs: 12
  use_amp: true              # FP16 autocast
  torch_compile: true        # Compile actor + critic networks
  
  # Buffer: NO replay buffer (on-policy)
  # All data discarded after n_epochs passes
```

---

## 3. A2C (Advantage Actor-Critic)

### 3.1 Algorithm Profile

| Property | Value |
|---|---|
| **Family** | On-Policy, Policy Gradient (synchronous) |
| **Data usage** | Collect short rollout → single gradient update → discard |
| **Key insight** | Simpler than PPO (no clipping), faster wall-clock, higher variance |
| **Bottleneck on 5090** | GPU underutilization (small batches, frequent updates) |

### 3.2 Batch Size — Smaller Than PPO

```yaml
a2c:
  n_steps: 5                  # Very short rollouts (A2C specialty)
  num_envs: 12
  # "batch_size" = n_steps × num_envs = 60 transitions per update
  # A2C does NOT mini-batch — the entire rollout IS the batch
```

**Why A2C uses tiny batches:**
- A2C's advantage is **speed** — update after every 5 steps, not every 2048
- The synchronous multi-env setup provides variance reduction (12 envs = 12 independent gradient samples)
- Making batches larger (n_steps=2048) would just turn A2C into a worse PPO (no clipping protection)

**RTX 5090 Impact:**
- A2C will **underutilize** the GPU — 60 samples per update doesn't saturate 1.8 TB/s bandwidth
- Mitigation: Increase `num_envs` to 32+ (if GPUHub allows) or increase `n_steps` to 20-50
- Alternative: Use A2C as a **fast signal generator** and funnel its outputs into an ensemble

### 3.3 A2C-Specific Precision Considerations

```python
class A2CAgent:
    def compute_loss(self, rollout):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            values = self.critic(states)
            action_dist = self.actor(states)
            log_probs = action_dist.log_prob(actions)
            
            # ⚠️ A2C SPECIFIC: No ratio clipping means gradient magnitudes
            # can be large. GradScaler + grad clipping are ESSENTIAL.
            advantages = returns - values.detach()
            
            # Normalize advantages (critical for FP16 stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = action_dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return loss
```

**A2C Gotcha — Advantage Normalization:**
Without PPO's clipping, A2C is sensitive to advantage scale. In FP16, un-normalized advantages (range: -1000 to +1000) can cause `log_prob * advantage` to overflow FP16's max (65504). **Always normalize advantages.**

### 3.4 A2C RTX 5090 Config Template

```yaml
a2c:
  learning_rate: 0.0007       # Higher than PPO (smaller effective batch)
  n_steps: 20                 # Slightly longer than default for GPU utilization
  gamma: 0.99
  gae_lambda: 1.0             # A2C traditionally uses full returns (lambda=1)
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  
  # RTX 5090 Hardware  
  num_envs: 12
  use_amp: true
  torch_compile: true
  
  # Normalization: CRITICAL for A2C stability
  normalize_advantages: true
  
  # No replay buffer, no epochs, no mini-batching
```

---

## 4. SAC (Soft Actor-Critic)

### 4.1 Algorithm Profile

| Property | Value |
|---|---|
| **Family** | Off-Policy, Maximum Entropy |
| **Data usage** | Store in replay buffer, sample mini-batches repeatedly |
| **Key insight** | Maximizes reward AND entropy simultaneously → robust exploration |
| **Bottleneck on 5090** | Replay buffer memory (large buffers eat VRAM/RAM) |

### 4.2 Batch Size — Similar to BDQ

```yaml
sac:
  batch_size: 256              # Classic SAC default
  buffer_size: 1000000         # 1M transitions
  learning_starts: 10000       # Collect 10K transitions before training
  gradient_steps: 1            # 1 gradient update per env step
```

**Why 256 (not 16K like PPO):**
- SAC samples from a 1M-entry replay buffer — samples are already decorrelated
- Each gradient step uses a **fresh random sample** from the buffer
- Increasing to 16K means fewer gradient steps for the same compute budget → slower convergence
- The RTX 5090 bandwidth advantage is better spent on **more frequent updates** (gradient_steps=2-4) than larger batches

**RTX 5090 Tuning:**
```yaml
# Option A: Larger batch (moderate)
batch_size: 512
gradient_steps: 1

# Option B: More updates per step (preferred for 5090)
batch_size: 256
gradient_steps: 4     # 4 gradient updates per env step → 4x GPU utilization
```

Option B is preferred because SAC's off-policy nature means more updates = more learning per environment interaction.

### 4.3 SAC-Specific Precision Considerations

SAC is the **most precision-sensitive** algorithm because it computes:
1. **Two Q-networks** (double Q-clipping)
2. **Log-probabilities** from a squashed Gaussian (tanh transform)
3. **Automatic temperature tuning** (alpha)

```python
class SACAgent:
    def compute_critic_loss(self, batch):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            # Current Q-values
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            
            # Target computation (NO autocast for target — use FP32)
            # ⚠️ SAC CRITICAL: Target Q uses minimum of two Q-networks
            # Plus entropy bonus. FP16 accumulation here causes Q-value drift.
        
        # ✅ CORRECT: Compute target in FP32
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Loss in FP16 is fine (it's just MSE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        return critic_loss
    
    def compute_actor_loss(self, batch):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            actions, log_probs = self.actor.sample(states)
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            q_min = torch.min(q1, q2)
            
            # ⚠️ SAC GOTCHA: log_prob from squashed Gaussian involves:
            # log_prob = gaussian_log_prob - log(1 - tanh(x)^2 + 1e-6)
            # The tanh correction term can be VERY small near saturation
            # FP16 may round 1e-6 to 0, causing log(0) = -inf
            # Solution: Use 1e-4 as epsilon, or compute correction in FP32
            
            actor_loss = (self.alpha * log_probs - q_min).mean()
        
        return actor_loss
```

**SAC Gotcha — Squashed Gaussian Log-Prob:**
```python
# ❌ DANGEROUS in FP16
log_prob -= torch.log(1 - torch.tanh(raw_action)**2 + 1e-6)
# When tanh(x) ≈ 1.0, the correction is ~1e-6, which FP16 rounds to 0

# ✅ SAFE: Use larger epsilon or compute in FP32
log_prob -= torch.log1p(-torch.tanh(raw_action).pow(2) + 1e-4)
# Or force this specific line to FP32:
with torch.amp.autocast(device_type="cuda", enabled=False):
    correction = torch.log1p(-torch.tanh(raw_action.float()).pow(2) + 1e-6)
log_prob = log_prob - correction.half()
```

**SAC Gotcha — Alpha (Temperature) Tuning:**
```python
# Alpha is typically ~0.2 and updated via gradient descent
# FP16 gradient for alpha can be extremely small (1e-5 to 1e-7)
# GradScaler handles this, but verify alpha doesn't collapse to 0

# ✅ SAFE: Keep alpha optimizer separate, potentially in FP32
self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
# Don't put alpha update inside autocast — it's a single scalar
```

### 4.4 SAC RTX 5090 Config Template

```yaml
sac:
  learning_rate: 0.0003        # Same LR for actor and critic (SAC convention)
  alpha_lr: 0.0003             # Temperature learning rate
  batch_size: 256
  buffer_size: 1000000         # 1M
  gamma: 0.99
  tau: 0.005                   # Soft target update (Polyak averaging)
  gradient_steps: 4            # RTX 5090: more updates per step
  learning_starts: 10000
  target_entropy: "auto"       # = -dim(action_space)
  
  # RTX 5090 Hardware
  num_envs: 1                  # SAC traditionally uses 1 env (off-policy)
  use_amp: true
  torch_compile: true
  
  # Replay buffer: 1M transitions
  # At 256 batch × 4 gradient_steps, each transition reused ~1000 times over training
```

---

## 5. Comparative Reference Card

### 5.1 Settings At A Glance

| Setting | PPO | A2C | SAC | BDQ (Current) |
|---|---|---|---|---|
| **Batch size** | **16,384** | 60-240 (n_steps×envs) | 256 | 1,024 |
| **Buffer** | None (rollout only) | None | 1M replay | 2M replay |
| **Update freq** | Every rollout (~24K steps) | Every 5-20 steps | Every 1-2 steps | Every 2 steps |
| **Gradient steps** | 10 epochs × ~2 mini-batches | 1 (single pass) | 4 per env step | 1 per update |
| **Learning rate** | 3e-4 (scale with batch) | 7e-4 | 3e-4 | 1e-4 |
| **FP16 autocast** | ✅ (exclude GAE) | ✅ (normalize advantages) | ✅ (exclude target Q, careful with log_prob) | ✅ |
| **GradScaler** | ✅ | ✅ | ✅ (separate alpha) | ✅ |
| **TF32** | ✅ | ✅ | ✅ | ✅ |
| **num_envs** | 12 | 12-32 | 1 | 12 |
| **torch.compile** | ✅ actor + critic | ✅ actor + critic | ✅ actor + critics | ✅ policy_net |

### 5.2 Precision Danger Zones (Per Agent)

| Agent | FP16 Safe Zone | FP16 Danger Zone (Keep in FP32) |
|---|---|---|
| **PPO** | Forward pass, loss, value estimation | GAE computation (sequential scan) |
| **A2C** | Forward pass, loss | Advantage computation if not normalized |
| **SAC** | Forward pass, critic loss, actor loss | Target Q computation, squashed Gaussian log-prob correction, alpha update |
| **BDQ** | Forward pass, Q-value gathering, loss | (None identified — simpler architecture) |

### 5.3 GPU Utilization Expectations

| Agent | RTX 5090 GPU Util (Expected) | Bottleneck | Mitigation |
|---|---|---|---|
| **PPO** | 60-80% | Large batch fills Tensor Cores well | Already optimal at 16K |
| **A2C** | 10-25% | Tiny batches underutilize | Increase num_envs or n_steps |
| **SAC** | 30-50% | Moderate batch, 4x gradient steps helps | gradient_steps=4, compile networks |
| **BDQ** | 40-60% | 1024 batch, every-2-step updates | Already optimal |

---

## 6. Implementation Checklist

When implementing or auditing any agent on RTX 5090, verify:

- [ ] `torch.set_float32_matmul_precision('high')` called at program startup
- [ ] `torch.amp.autocast(dtype=torch.float16)` has explicit `dtype` parameter
- [ ] `torch.amp.GradScaler()` created and used for all backward passes
- [ ] `scaler.unscale_()` called before `clip_grad_norm_()` 
- [ ] No FP8 usage anywhere (`float8`, `Float8Linear`)
- [ ] GAE/advantage computed in FP32 (PPO/A2C)
- [ ] Target Q computed in FP32 with `torch.no_grad()` (SAC/BDQ)
- [ ] Squashed Gaussian log-prob uses `1e-4` epsilon or FP32 fallback (SAC)
- [ ] Alpha/temperature update NOT inside autocast (SAC)
- [ ] Batch size appropriate for algorithm family (see §5.1)
- [ ] `torch.compile(mode="reduce-overhead")` on networks only, not training loops
- [ ] Advantage normalization enabled (A2C — mandatory; PPO — recommended)

---

## 7. Anti-Patterns (DO NOT DO)

```python
# ❌ Anti-Pattern 1: Global autocast (covers everything including unsafe ops)
with torch.amp.autocast(...):
    gae = compute_gae(rewards, values)  # Sequential scan drifts in FP16
    target_q = compute_target(...)      # Accumulation error
    loss = compute_loss(gae, target_q)

# ❌ Anti-Pattern 2: Sharing batch_size across algorithms
config["batch_size"] = 16384  # PPO optimal, catastrophic for SAC

# ❌ Anti-Pattern 3: Using BF16 "because it doesn't need GradScaler"
# BF16's 7-bit mantissa causes policy collapse in RL. Always use FP16.

# ❌ Anti-Pattern 4: Compiling the training loop
compiled_train_step = torch.compile(agent.train_step)
# This will fail or be extremely slow due to dynamic control flow

# ❌ Anti-Pattern 5: Same learning rate for on-policy and off-policy
# PPO at LR=1e-4 is too slow. SAC at LR=3e-3 is too fast.
```

---

*Document Version: 1.0 | Created: 2026-02-07 | Target: RTX 5090 Blackwell | Project: FinRL-Pro_DS*
