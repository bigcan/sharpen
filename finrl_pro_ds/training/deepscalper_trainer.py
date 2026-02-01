import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import time
from collections import deque, defaultdict
import random
import wandb # Added for WandB logging
import filelock # Audit Fix 1: Prevention of Race Conditions
from torch.distributions import Categorical

from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.mlops.logger import MLOpsLogger
from finrl_pro_ds.mlops.watchdog import TrainingWatchdog
from finrl_pro_ds.training.accumulators import GradientAccumulator
from torch.cuda.amp import GradScaler, autocast

class DeepScalperTrainer:
    """
    Dedicated Trainer for DeepScalper Ensemble.
    Handles the complex interaction between:
    - Environment (Micro/Macro obs)
    - 3 Sub-Agents (DQN, PPO, A2C)
    - Synapse Gating Network
    - Replay Buffers
    """
    def __init__(
        self,
        env,
        ensemble_agent: DeepScalperEnsemble,
        config: Dict[str, Any],
        logger: Optional[MLOpsLogger] = None,
        device: str = "cpu",
        run_name: str = "default_run"
    ):
        self.env = env
        self.ensemble = ensemble_agent
        self.config = config
        self.logger = logger or MLOpsLogger()
        self.device = torch.device(device)
        self.run_name = run_name

        # OPTIMIZATION: Enforce TensorFloat-32 (TF32) for RTX 5090 (Blackwell)
        # 10-bit mantissa (same as FP16) + 8-bit exponent (same as FP32)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            print("PRECISION: Enforced TensorFloat-32 (TF32) 'high' precision.")
                # Training Hyperparameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99) # Default global gamma, but agents might have their own
        self.total_timesteps = config.get("total_timesteps", 100000)
        # self.learning_rate is depcreated for agent-specific configs, but kept as fallback
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.checkpoint_interval = config.get("checkpoint_interval", 10000)
        self.dqn_update_interval = config.get("dqn_update_interval", 4)  # Train DQN every N env steps
        
        # Ensure checkpoint dir exists
        self.checkpoint_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Parse Agent Configs (if available, else fallback to global LR)
        agents_config = config.get("agents", {})
        
        ppo_config = agents_config.get("ppo", {})
        a2c_config = agents_config.get("a2c", {})
        gating_config = agents_config.get("gating", {})
        
        ppo_lr = ppo_config.get("learning_rate", self.learning_rate)
        a2c_lr = a2c_config.get("learning_rate", self.learning_rate)
        gating_lr = gating_config.get("learning_rate", self.learning_rate)
        
        self.ppo_entropy_coef = ppo_config.get("entropy_coef", 0.01)
        self.a2c_entropy_coef = a2c_config.get("entropy_coef", 0.01)
        
        # Gating Optimizer
        self.gating_optimizer = optim.Adam(
            self.ensemble.gating.parameters(), 
            lr=gating_lr
        )
        
        self.ppo_optimizer = optim.Adam(self.ensemble.ppo.network.parameters(), lr=ppo_lr)
        self.a2c_optimizer = optim.Adam(self.ensemble.a2c.network.parameters(), lr=a2c_lr)
        
        # AMP Scalers (Independent)
        self.use_amp = config.get("use_amp", False) and torch.cuda.is_available()
        self.scaler_ppo = GradScaler(enabled=self.use_amp)
        self.scaler_a2c = GradScaler(enabled=self.use_amp)
        self.scaler_gating = GradScaler(enabled=self.use_amp)
        
        # Accumulator (For tracking batch targets mainly, or if we switched to pseudo-updates)
        self.accumulator = GradientAccumulator(self.batch_size, num_envs=config.get("env", {}).get("num_envs", 1))

        self.ppo_buffer = [] 
        self.a2c_buffer = []
        self.gating_buffer = []
        
        self.global_step = 0
        self.dqn_updates_accumulator = 0.0

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = True
        module.train()

    def load_checkpoint(self, path: str, strict: bool = False, load_optimizers: bool = False):
        """
        Load agent states from checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: If True, raise error if any agent state fails to load
            load_optimizers: If True, also load optimizer states (for mid-phase resume)
        """
        print(f"Loading checkpoint from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Track loading success for strict mode
        load_errors = []
        
        # Load State Dictionaries
        try:
            self.ensemble.dqn.policy_net.load_state_dict(checkpoint["dqn"])
            print("Loaded DQN state.")
        except Exception as e:
            load_errors.append(f"DQN: {e}")
            print(f"WARNING: Failed to load DQN state: {e}")

        try:
            self.ensemble.ppo.network.load_state_dict(checkpoint["ppo"])
            print("Loaded PPO state.")
        except Exception as e:
            load_errors.append(f"PPO: {e}")
            print(f"WARNING: Failed to load PPO state: {e}")

        try:
            self.ensemble.a2c.network.load_state_dict(checkpoint["a2c"])
            print("Loaded A2C state.")
        except Exception as e:
            load_errors.append(f"A2C: {e}")
            print(f"WARNING: Failed to load A2C state: {e}")

        try:
            self.ensemble.gating.load_state_dict(checkpoint["gating"])
            print("Loaded Gating state.")
        except Exception as e:
            load_errors.append(f"Gating: {e}")
            print(f"WARNING: Failed to load Gating state: {e}")
        
        # Restore global step for proper resume
        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"Resumed from global_step: {self.global_step}")
        else:
            print("WARNING: Checkpoint does not contain global_step. Starting from 0.")
        
        # Optionally load optimizer states (for mid-phase resume)
        if load_optimizers and "optimizers" in checkpoint:
            print("Loading optimizer states...")
            try:
                if "gating" in checkpoint["optimizers"]:
                    self.gating_optimizer.load_state_dict(checkpoint["optimizers"]["gating"])
                if "ppo" in checkpoint["optimizers"]:
                    self.ppo_optimizer.load_state_dict(checkpoint["optimizers"]["ppo"])
                if "a2c" in checkpoint["optimizers"]:
                    self.a2c_optimizer.load_state_dict(checkpoint["optimizers"]["a2c"])
                print("Optimizer states loaded.")
            except Exception as e:
                print(f"WARNING: Failed to load optimizer states: {e}")
        elif load_optimizers:
            print("WARNING: Checkpoint does not contain optimizer states.")
        
        # Strict mode validation
        if strict and load_errors:
            raise RuntimeError(f"Checkpoint load failed in strict mode. Errors: {load_errors}")
            
        print("Checkpoint loaded successfully.")

    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        last_gae = 0
        
        # Ensure input lists are correct length
        # Appending 0 for terminal value handled in loop
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages.insert(0, last_gae)
            
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update_ppo(self, agent: DeepScalperPPO, optimizer: optim.Optimizer, buffer: List):
        if not buffer: return
        
        # Unpack Buffer
        micro_s, private_s, macro_s, actions, old_log_probs, rewards, values, next_values_list, dones = zip(*buffer)
        
        # Convert to Tensors (Full Batch)
        # We process GAE on the full batch first (time-sequential)
        micro_s = torch.stack(micro_s)
        private_s = torch.stack(private_s)
        macro_s = torch.stack(macro_s)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(old_log_probs).detach() # (B, 3)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device).detach()
        next_values = torch.tensor(next_values_list, dtype=torch.float32).to(self.device).detach()
        
        # Compute Advantages (GAE) on full sequence
        advantages = self.compute_gae(rewards, values_t, next_values, dones, self.gamma, 0.95)
        returns = advantages + values_t
        
        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Mini-Batch Updates
        # Default mini_batch_size to 64 or derived from config if available (self.batch_size is global)
        mini_batch_size = 64 
        dataset_size = len(rewards)
        indices = np.arange(dataset_size)
        
        total_loss = 0.0
        n_updates = 0

        for _ in range(4): # K_epochs
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                idx = indices[start:end]
                
                # Mini-batch Slices
                mb_micro = micro_s[idx]
                mb_private = private_s[idx]
                mb_macro = macro_s[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                with autocast(enabled=self.use_amp, dtype=torch.float16):
                    logits_dir, logits_price, logits_vol, current_values = agent.network(mb_micro, mb_private, mb_macro)
                    
                    # Calculate current log probs (re-eval)
                    dist_dir = Categorical(logits=logits_dir)
                    dist_price = Categorical(logits=logits_price)
                    dist_vol = Categorical(logits=logits_vol)
                    
                    curr_log_prob_dir = dist_dir.log_prob(mb_actions[:, 0])
                    curr_log_prob_price = dist_price.log_prob(mb_actions[:, 1])
                    curr_log_prob_vol = dist_vol.log_prob(mb_actions[:, 2])
                    
                    curr_log_probs = curr_log_prob_dir + curr_log_prob_price + curr_log_prob_vol
                    
                    # Ratios
                    old_log_probs_sum = mb_old_log_probs.sum(dim=1)
                    ratios = torch.exp(curr_log_probs - old_log_probs_sum)
                    
                    # Surrogate Losses
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * mb_advantages
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (mb_returns - current_values.squeeze()).pow(2).mean()
                    entropy = dist_dir.entropy() + dist_price.entropy() + dist_vol.entropy()
                    entropy_loss = -self.ppo_entropy_coef * entropy.mean()
                    
                    loss = policy_loss + value_loss + entropy_loss
                
                if not torch.isfinite(loss):
                    print(f"WARNING: PPO Loss is {loss.item()} (NaN/Inf). Skipping update.", flush=True)
                    optimizer.zero_grad() # Clear any bad grads
                    continue

                optimizer.zero_grad()
                self.scaler_ppo.scale(loss).backward()
                self.scaler_ppo.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
                self.scaler_ppo.step(optimizer)
                self.scaler_ppo.update()
                
                total_loss += loss.item()
                n_updates += 1
            
        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
        # For simplicity, we assume entropy from last batch is representative, or we could averge it. 
        # But we only have scope to return avg loss easily without big refactor. 
        # Actually let's return a dict with the last entropy for now as proxy.
        return {
            "loss": avg_loss,
            "entropy": entropy.mean().item()
        }

    def update_a2c(self, agent: DeepScalperA2C, optimizer: optim.Optimizer, buffer: List):
        if not buffer: return
        
        # A2C typically updates on the full batch collected (synchronous)
        # But we can also mini-batch if memory is constrained. 
        # DeepScalper A2C usually one update per roll-out.
        # We keep full batch for A2C to distinguish from PPO, but verify memory safety.
        # If buffer is huge, we might need to split, but A2C gradient is usually over the whole set.
        
        micro_s, private_s, macro_s, actions, _, rewards, values, next_values_list, dones = zip(*buffer)
        
        micro_s = torch.stack(micro_s)
        private_s = torch.stack(private_s)
        macro_s = torch.stack(macro_s)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        next_values = torch.tensor(next_values_list, dtype=torch.float32).to(self.device).detach()
        advantages = self.compute_gae(rewards, values_t, next_values, dones, self.gamma, 0.95)
        returns = advantages + values_t
        
        # Full Batch Update for A2C
        with autocast(enabled=self.use_amp, dtype=torch.float16):
            logits_dir, logits_price, logits_vol, current_values = agent.network(micro_s, private_s, macro_s)
            
            dist_dir = Categorical(logits=logits_dir)
            dist_price = Categorical(logits=logits_price)
            dist_vol = Categorical(logits=logits_vol)
            
            log_prob_dir = dist_dir.log_prob(actions[:, 0])
            log_prob_price = dist_price.log_prob(actions[:, 1])
            log_prob_vol = dist_vol.log_prob(actions[:, 2])
            log_probs = log_prob_dir + log_prob_price + log_prob_vol
            
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            
            entropy = dist_dir.entropy() + dist_price.entropy() + dist_vol.entropy()
            entropy_loss = -self.a2c_entropy_coef * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
        
        if not torch.isfinite(loss):
            print(f"WARNING: A2C Loss is {loss.item()} (NaN/Inf). Skipping update.", flush=True)
            return None

        optimizer.zero_grad()
        self.scaler_a2c.scale(loss).backward()
        self.scaler_a2c.unscale_(optimizer)
        nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
        
        self.scaler_a2c.step(optimizer)
        self.scaler_a2c.update()
        
        return {
            "loss": policy_loss.item() + value_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item()
        }

    def update_gating(self, buffer: List):
        """Update Gating Network using REINFORCE"""
        if not buffer: return
        
        macro_s, weights, rewards, dones = zip(*buffer)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.float32).to(self.device) # Unused in this simple reinforce logic
        
        # Normalize Rewards (Advantage Proxy)
        # Standardize for stability
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        macro_s = torch.stack(macro_s)
        old_weights = torch.stack(weights).detach() 
        
        with autocast(enabled=self.use_amp, dtype=torch.float16):
            # Forward Pass
            curr_weights = self.ensemble.gating(macro_s)
            
            # Loss: Minimize - (NewWeights * OldWeights * Adv)
            # This intuitively pushes NewWeights towards OldWeights where Adv was high.
            loss = - (curr_weights * old_weights * adv.unsqueeze(1)).sum(dim=1).mean()
        
        if not torch.isfinite(loss):
            print(f"WARNING: Gating Loss is {loss.item()} (NaN/Inf). Skipping update.", flush=True)
            return 0.0

        self.gating_optimizer.zero_grad()
        self.scaler_gating.scale(loss).backward()
        self.scaler_gating.step(self.gating_optimizer)
        self.scaler_gating.update()
        
        return {
            "loss": loss.item(),
            "weights": curr_weights.mean(dim=0).float().detach().cpu().numpy() # [w_dqn, w_ppo, w_a2c]
        }

        
    def train(self, phase: str = "full"):
        """
        Main Training Loop
        phase: 'full' | 'specialists' | 'gating'
        """
        self.logger.log_event("deepscalper.training.start", context={"phase": phase})
        print(f"Starting Training Phase: {phase.upper()}")

        # Phase-Specific Setup
        if phase == "specialists":
            print("PHASE: SPECIALISTS - Freezing Gating Network")
            self._freeze_module(self.ensemble.gating)
            self._unfreeze_module(self.ensemble.dqn.policy_net)
            self._unfreeze_module(self.ensemble.ppo.network)
            self._unfreeze_module(self.ensemble.a2c.network)
        elif phase == "gating":
            print("PHASE: GATING - Freezing Specialist Agents")
            self._unfreeze_module(self.ensemble.gating)
            self._freeze_module(self.ensemble.dqn.policy_net)
            self._freeze_module(self.ensemble.dqn.target_net)  # Also freeze target net
            self._freeze_module(self.ensemble.ppo.network)
            self._freeze_module(self.ensemble.a2c.network)
        else: # full
            print("PHASE: FULL - Training All Components")
            self._unfreeze_module(self.ensemble.gating)
            self._unfreeze_module(self.ensemble.dqn.policy_net)
            self._unfreeze_module(self.ensemble.ppo.network)
            self._unfreeze_module(self.ensemble.a2c.network)
        
        # Detect Vector Env
        is_vector_env = False
        import gymnasium as gym
        if isinstance(self.env, gym.vector.VectorEnv):
            is_vector_env = True
            num_envs = self.env.num_envs
            # Ensure env is reset
            obs, info = self.env.reset()
        else:
            num_envs = 1
            obs, info = self.env.reset()
            # If standard env, ensure batch dim is handled in _unpack_obs
        
        # GUARD: Prevent infinite loop if num_envs is 0
        if num_envs < 1:
            print(f"WARNING: num_envs detected as {num_envs}. Forcing to 1 to prevent infinite loop.", flush=True)
            num_envs = 1
            
        micro, private, macro = self._unpack_obs(obs) 
        # _unpack_obs handles adding batch dim if missing for single env.
        # For VecEnv, micro is (B, W, F), private is (B, W, F_p), macro is (B, F). Perfect.
        
        # Compile Model if requested
        print(f"DEBUG: self.config['torch_compile'] = {self.config.get('torch_compile', 'Not Set')}")
        if self.config.get("torch_compile", False) and hasattr(torch, "compile"):
            print("Compiling models with torch.compile...")
            try:
                # Only compile active components to save time/errors? 
                # Or just compile everything once. Compiling frozen models is fine.
                self.ensemble.dqn.policy_net = torch.compile(self.ensemble.dqn.policy_net)
                self.ensemble.ppo.network = torch.compile(self.ensemble.ppo.network)
                self.ensemble.a2c.network = torch.compile(self.ensemble.a2c.network)
                self.ensemble.gating = torch.compile(self.ensemble.gating)
                print("Models compiled successfully.")
            except Exception as e:
                print(f"WARNING: torch.compile failed: {e}. Proceeding without compilation.")
        
        # Tracking
        if is_vector_env:
            episode_rewards = np.zeros(num_envs, dtype=np.float32)
            episode_lengths = np.zeros(num_envs, dtype=np.int32)
        else:
            episode_rewards = 0.0
            episode_steps = 0 # Legacy Scalar
            
        episode_rewards_total = 0
        episode_count = 0
        
        # Metric Aggregators
        dqn_metrics_accum = defaultdict(list)
        ppo_metrics_accum = defaultdict(list)
        a2c_metrics_accum = defaultdict(list)
        gating_metrics_accum = defaultdict(list)
        vol_target_accum = []
        inventory_accum = []
        action_counts = {"dir": [0]*3, "price": [0]*5, "vol": [0]*5}
        
        # Init Watchdog (SPS Monitor)
        print("DEBUG: Initializing Watchdog...", flush=True)
        self.watchdog = TrainingWatchdog(
            lambda: self.global_step, 
            timeout_seconds=300,
            min_sps=100.0 if self.config.get("env", {}).get("num_envs", 1) > 8 else 0.0 # Only enforce SPS on high throughput
        )
        print("DEBUG: Starting Watchdog...", flush=True)
        self.watchdog.start()
        print("DEBUG: Watchdog started.", flush=True)
        
        print("DEBUG: Entering Training Loop...", flush=True)
        try:
            while self.global_step < self.total_timesteps:
                start_step = self.global_step
                step = self.global_step  # For backward compatibility with logging
                # print(f"DEBUG: Loop Start. Step: {step}, Global: {self.global_step}, NumEnvs: {num_envs}", flush=True)
                
                # 1. Select Action (Voting)
                if step % 1000 == 0:
                     self.logger.log_event("deepscalper.training.step_start", context={"step": step})
                
                # Action Selection (Flattened & Robust)
                with torch.no_grad():
                    # Ensemble Gating
                    weights = self.ensemble.gating(macro) # (B, 3)
                    
                    # Individual Probs
                    p_dqn_dir, p_dqn_price, p_dqn_vol = self.ensemble.dqn.get_probs(micro, private, macro)
                    logits_ppo_dir, logits_ppo_price, logits_ppo_vol, val_ppo = self.ensemble.ppo.network(micro, private, macro)
                    
                    p_ppo_dir = torch.softmax(logits_ppo_dir, dim=1)
                    p_ppo_price = torch.softmax(logits_ppo_price, dim=1)
                    p_ppo_vol = torch.softmax(logits_ppo_vol, dim=1)
                    
                    logits_a2c_dir, logits_a2c_price, logits_a2c_vol, val_a2c = self.ensemble.a2c.network(micro, private, macro)
                    p_a2c_dir = torch.softmax(logits_a2c_dir, dim=1)
                    p_a2c_price = torch.softmax(logits_a2c_price, dim=1)
                    p_a2c_vol = torch.softmax(logits_a2c_vol, dim=1)
                    
                    # Ensemble Aggregate
                    w_dqn = weights[:, 0].unsqueeze(1)
                    w_ppo = weights[:, 1].unsqueeze(1)
                    w_a2c = weights[:, 2].unsqueeze(1)
                    
                    final_dir = w_dqn * p_dqn_dir + w_ppo * p_ppo_dir + w_a2c * p_a2c_dir
                    final_price = w_dqn * p_dqn_price + w_ppo * p_ppo_price + w_a2c * p_a2c_price
                    final_vol = w_dqn * p_dqn_vol + w_ppo * p_ppo_vol + w_a2c * p_a2c_vol
                    
                    # Sample Action
                    dist_dir = Categorical(probs=final_dir)
                    dist_price = Categorical(probs=final_price)
                    dist_vol = Categorical(probs=final_vol)
                    
                    a_dir = dist_dir.sample()
                    a_price = dist_price.sample()
                    a_vol = dist_vol.sample()
                    
                    # Action Vector: (B, 3)
                    action_vector = torch.stack([a_dir, a_price, a_vol], dim=1).cpu().numpy()

                    # Track Actions
                    for b_idx in range(action_vector.shape[0]):
                        action_counts["dir"][action_vector[b_idx, 0]] += 1
                        action_counts["price"][action_vector[b_idx, 1]] += 1
                        action_counts["vol"][action_vector[b_idx, 2]] += 1
                    
                    # Log Probs for PPO
                    ens_log_dir = dist_dir.log_prob(a_dir)
                    ens_log_price = dist_price.log_prob(a_price)
                    ens_log_vol = dist_vol.log_prob(a_vol)
                    
                    ppo_log_prob = torch.stack([ens_log_dir, ens_log_price, ens_log_vol], dim=1)

                # 2. Step Environment
                if is_vector_env:
                    next_obs, reward, terminated, truncated, info = self.env.step(action_vector)
                else:
                    # Unwrap for single env (1, 3) -> (3,)
                    next_obs, reward, terminated, truncated, info = self.env.step(action_vector[0])
                
                next_micro, next_private, next_macro = self._unpack_obs(next_obs)
                
                # Handle Done
                if is_vector_env:
                    dones = terminated | truncated # Element-wise OR
                else:
                    dones = terminated or truncated
                
                # 3. Store Transitions & Track Rewards
                if is_vector_env:
                    # Calculate Next Values for PPO/A2C (Missing in original code)
                    with torch.no_grad():
                        _, _, _, val_next_ppo_batch = self.ensemble.ppo.network(next_micro, next_private, next_macro)
                        _, _, _, val_next_a2c_batch = self.ensemble.a2c.network(next_micro, next_private, next_macro)
                        
                        # Flatten if needed or ensure shape matches reward[i]
                        val_next_ppo = val_next_ppo_batch.squeeze()
                        val_next_a2c = val_next_a2c_batch.squeeze()
                        
                        # Handle scalar squeeze edge case (if num_envs=1 but somehow flagged as vector)
                        if val_next_ppo.ndim == 0:
                            val_next_ppo = val_next_ppo.unsqueeze(0)
                            val_next_a2c = val_next_a2c.unsqueeze(0)

                    # VECTOR ENV STORAGE LOOP
                    for i in range(num_envs):
                        episode_rewards[i] += reward[i]
                        episode_lengths[i] += 1
                        
                        done = dones[i]
                        
                        # Store transition
                        # Note: values, log_probs are tensors (B, ...), need to extract [i]
                        # Observations are (B, ...), extract [i]
                        
                        state_dict = {"micro": micro[i].cpu().numpy(), "private": private[i].cpu().numpy(), "macro": macro[i].cpu().numpy()}
                        next_state_dict = {"micro": next_micro[i].cpu().numpy(), "private": next_private[i].cpu().numpy(), "macro": next_macro[i].cpu().numpy()}
                        
                        # Only push to DQN if we are training DQN (Full or Specialists)
                        if phase in ["full", "specialists"]:
                            self.ensemble.dqn.memory.push(
                                state_dict, 
                                action_vector[i], 
                                reward[i], 
                                next_state_dict, 
                                done,
                                float(info.get("volatility_target", [0.0]*num_envs)[i]) if isinstance(info.get("volatility_target"), (list, np.ndarray)) else 0.0
                            )
                        
                        # Only append to PPO/A2C buffers if active
                        if phase in ["full", "specialists"]:
                            self.ppo_buffer.append((
                                micro[i], private[i], macro[i], action_vector[i], ppo_log_prob[i],
                                reward[i], val_ppo[i].item(), val_next_ppo[i].item(), done
                            ))
                            self.a2c_buffer.append((
                                micro[i], private[i], macro[i], action_vector[i], None,
                                reward[i], val_a2c[i].item(), val_next_a2c[i].item(), done
                            ))
                        
                        # Only append to Gating buffer if active
                        if phase in ["full", "gating"]:
                            self.gating_buffer.append((
                                macro[i], weights[i], reward[i], done
                            ))

                        # Accumulate Context Metrics
                        # Info might be list or dict depending on Wrapper, assuming VectorEnv wrapper handles it
                        vt_raw = info.get("volatility_target")
                        if isinstance(vt_raw, (list, np.ndarray)):
                             cur_vt = float(vt_raw[i])
                        else:
                             cur_vt = 0.0
                        
                        vol_target_accum.append(cur_vt)
                        # private is Tensor (B, W, 2), we want [i, -1, 0] (Position)
                        inventory_accum.append(abs(private[i, -1, 0].item()))

                        if done:
                            # Log metrics for THIS environment
                            metrics = {
                                "train/episode_reward": episode_rewards[i], 
                                "train/episode_length": episode_lengths[i], 
                                "train/global_step": self.global_step
                            }
                            wandb.log(metrics)
                            
                            episode_rewards_total += episode_rewards[i]
                            episode_count += 1
                            
                            episode_rewards[i] = 0.0
                            episode_lengths[i] = 0
                            
                else:
                    # SINGLE ENV STORAGE
                    episode_rewards += reward
                    episode_steps += 1
                    
                    done = dones
                    
                    with torch.no_grad():
                        _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_private, next_macro)
                        _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_private, next_macro)
                    
                    # Buffer Push
                    state_dict = {"micro": micro.squeeze(0).cpu().numpy(), "private": private.squeeze(0).cpu().numpy(), "macro": macro.squeeze(0).cpu().numpy()}
                    next_state_dict = {"micro": next_micro.squeeze(0).cpu().numpy(), "private": next_private.squeeze(0).cpu().numpy(), "macro": next_macro.squeeze(0).cpu().numpy()}
                    
                    if phase in ["full", "specialists"]:
                        self.ensemble.dqn.memory.push(
                            state_dict, 
                            action_vector[0], 
                            reward, 
                            next_state_dict, 
                            done,
                            float(info.get("volatility_target", 0.0))
                        )
                        
                        self.ppo_buffer.append((
                            micro.squeeze(0), private.squeeze(0), macro.squeeze(0), action_vector[0], ppo_log_prob.squeeze(0),
                            reward, val_ppo.item(), val_next_ppo.item(), done
                        ))
                        self.a2c_buffer.append((
                            micro.squeeze(0), private.squeeze(0), macro.squeeze(0), action_vector[0], None,
                            reward, val_a2c.item(), val_next_a2c.item(), done
                        ))
                        
                    if phase in ["full", "gating"]:
                        self.gating_buffer.append((
                            macro.squeeze(0), weights.squeeze(0), reward, done
                        ))
                    
                    # Accumulate Context Metrics (Single Env)
                    vol_target_accum.append(float(info.get("volatility_target", 0.0)))
                    # single env private is (1, W, 2), squeeze -> (W, 2)
                    inventory_accum.append(abs(private.squeeze(0)[-1, 0].item()))
                    
                    if done:
                        metrics = {"train/episode_reward": episode_rewards, "train/episode_length": episode_steps, "train/global_step": self.global_step}
                        wandb.log(metrics)
                        episode_rewards = 0
                        episode_steps = 0
                        episode_count += 1
                        episode_rewards_total += metrics["train/episode_reward"]
                        
                        obs, info = self.env.reset()
                        next_micro, next_private, next_macro = self._unpack_obs(obs)
                
                # Increment global step by number of envs
                self.global_step += num_envs

                # 4. Updates
                
                # A. Train DQN (Accumulate gradients to match update ratio)
                metrics_dqn = None
                if phase in ["full", "specialists"]:
                    self.dqn_updates_accumulator += num_envs / self.dqn_update_interval
                    
                    while self.dqn_updates_accumulator >= 1.0:
                        metrics_dqn = self.ensemble.dqn.train_step()
                        if metrics_dqn:
                             for k, v in metrics_dqn.items():
                                 dqn_metrics_accum[k].append(v)
                        self.dqn_updates_accumulator -= 1.0
                
                # B. Train PPO/A2C
                # Check if we crossed an update interval boundary
                update_interval = self.config.get("update_interval", 256)
                prev_interval_idx = start_step // update_interval
                curr_interval_idx = self.global_step // update_interval
                
                if curr_interval_idx > prev_interval_idx:
                    ppo_res = None
                    a2c_res = None
                    gating_res = None
                    
                    if phase in ["full", "specialists"]:
                        ppo_res = self.update_ppo(self.ensemble.ppo, self.ppo_optimizer, self.ppo_buffer)
                        a2c_res = self.update_a2c(self.ensemble.a2c, self.a2c_optimizer, self.a2c_buffer)
                    
                    if phase in ["full", "gating"]:
                        gating_res = self.update_gating(self.gating_buffer)
                    
                    if ppo_res:
                        for k, v in ppo_res.items(): ppo_metrics_accum[k].append(v)
                    if a2c_res:
                        for k, v in a2c_res.items(): a2c_metrics_accum[k].append(v)
                    if gating_res:
                        for k, v in gating_res.items(): 
                            gating_metrics_accum[k].append(v) # Handles both scalar and array
                    
                    # Clear buffers regardless, they are stale
                    self.ppo_buffer = [] 
                    self.a2c_buffer = []
                    self.gating_buffer = []
                
                # C. Log Batch Metrics
                if self.global_step % self.config.get("log_interval", 1000) < num_envs:
                    self.logger.log_event("deepscalper.training.batch", context={
                        "step": self.global_step,
                        "dqn_loss": metrics_dqn["loss_total"] if metrics_dqn else 0.0,
                        "reward_mean": episode_rewards_total / max(1, episode_count)
                    })
                
                # Periodic Evaluation / Logging
                log_interval = self.config.get("log_interval", 1000)
                if (self.global_step // log_interval) > (start_step // log_interval):
                     # Construct Log Dict
                     log_dict = {
                         "train/step_reward_mean": episode_rewards_total / max(1, episode_count) if episode_count > 0 else 0.0,
                         "train/global_step": self.global_step
                     }
                     # Aggregate DQN
                     for k, v in dqn_metrics_accum.items():
                         log_dict[f"train/dqn/{k}"] = np.mean(v)
                     dqn_metrics_accum.clear()
                     
                     # Aggregate PPO
                     for k, v in ppo_metrics_accum.items():
                         log_dict[f"train/ppo/{k}"] = np.mean(v)
                     ppo_metrics_accum.clear()
                     
                     # Aggregate A2C
                     for k, v in a2c_metrics_accum.items():
                         log_dict[f"train/a2c/{k}"] = np.mean(v)
                     a2c_metrics_accum.clear()

                     # Aggregate Gating
                     for k, v in gating_metrics_accum.items():
                         if k == "weights":
                             if len(v) > 0:
                                 avg_weights = np.mean(np.stack(v), axis=0)
                                 log_dict["train/gating/weight_dqn"] = avg_weights[0]
                                 log_dict["train/gating/weight_ppo"] = avg_weights[1]
                                 log_dict["train/gating/weight_a2c"] = avg_weights[2]
                         else:
                             log_dict[f"train/gating/{k}"] = np.mean(v)
                     gating_metrics_accum.clear()
                     
                     # Action Dist
                     total_acts = sum(action_counts["dir"])
                     if total_acts > 0:
                         for i in range(3): log_dict[f"train/action_dist/dir_{i}"] = action_counts["dir"][i] / total_acts
                         # Reset stats
                         action_counts = {"dir": [0]*3, "price": [0]*5, "vol": [0]*5}
                     
                     # Market Context
                     if vol_target_accum:
                         log_dict["train/market/volatility_target"] = np.mean(vol_target_accum)
                         vol_target_accum = []
                     if inventory_accum:
                         log_dict["train/market/inventory_abs_mean"] = np.mean(inventory_accum)
                         inventory_accum = []
                     
                     wandb.log(log_dict)

                # Save Checkpoint
                if (self.global_step // self.checkpoint_interval) > (start_step // self.checkpoint_interval):
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step}.pth")
                    self.save_checkpoint(ckpt_path)

                # Update Obs
                micro = next_micro
                private = next_private
                macro = next_macro
                obs = next_obs
                # print("DEBUG: Loop End. Obs updated.", flush=True)
        
        except Exception as e:
            print(f"FATAL EXCEPTION IN TRAINING LOOP: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e
        finally:
            print("DEBUG: Stopping Watchdog...", flush=True)
            if hasattr(self, 'watchdog'):
                self.watchdog.stop()

        # Save Final Checkpoint
        final_ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_final_{self.global_step}.pth")
        print(f"Saving final checkpoint to {final_ckpt_path}...", flush=True)
        self.save_checkpoint(final_ckpt_path)

        self.logger.log_event("deepscalper.training.complete")
        
        if episode_count > 0:
            avg_reward = episode_rewards_total / episode_count
        else:
            avg_reward = 0.0
            
        return avg_reward
        
    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def _unpack_obs(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dict observation to tensors on device.
        Expected keys: 'micro' (Window, Feat), 'macro' (Feat)
        """
        # Handle cases where env returns numpy arrays
        if isinstance(obs, dict):
            micro_np = obs.get("micro")
            private_np = obs.get("private")
            macro_np = obs.get("macro")
            
            if micro_np is None:
                raise ValueError(f"Observation missing 'micro' key. Keys found: {list(obs.keys())}")
            if private_np is None:
                raise ValueError(f"Observation missing 'private' key. Keys found: {list(obs.keys())}")
            
            # If batch dim missing, add it
            if len(micro_np.shape) == 2:
                micro_t = torch.tensor(micro_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                micro_t = torch.tensor(micro_np, dtype=torch.float32).to(self.device)
                
            if len(private_np.shape) == 2: # (Window, Features)
                 private_t = torch.tensor(private_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            elif len(private_np.shape) == 1: # (Features)
                 # If env gives 1D private? Network expects Window
                 # But we assume Env gives Window. If Env gives (F), we unsqueeze(0) for batch, but still 1D.
                 # Let's assume Env gives (W, F) as standard.
                 private_t = torch.tensor(private_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                 private_t = torch.tensor(private_np, dtype=torch.float32).to(self.device)
                
            # Handle optional macro if system design allows, though DeepScalper requires it
            if macro_np is None:
                 # If using a mock env that might return None or different structure?
                 # DeepScalper requires macro.
                 raise ValueError("Observation missing 'macro' key.")

            if len(macro_np.shape) == 1:
                macro_t = torch.tensor(macro_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                macro_t = torch.tensor(macro_np, dtype=torch.float32).to(self.device)
                
            return micro_t, private_t, macro_t
        else:
            raise ValueError(f"Expected dict observation, got {type(obs)}")

    def _get_clean_state_dict(self, model: nn.Module) -> Dict:
        """Helper to strip _orig_mod. prefix from torch.compile"""
        state_dict = model.state_dict()
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save_checkpoint(self, path: str):
        """Save all agent states and optimizer states for resume"""
        state = {
            "dqn": self._get_clean_state_dict(self.ensemble.dqn.policy_net),
            "ppo": self._get_clean_state_dict(self.ensemble.ppo.network),
            "a2c": self._get_clean_state_dict(self.ensemble.a2c.network),
            "gating": self._get_clean_state_dict(self.ensemble.gating),
            "config": self.config,
            "global_step": self.global_step,
            "optimizers": {
                "gating": self.gating_optimizer.state_dict(),
                "ppo": self.ppo_optimizer.state_dict(),
                "a2c": self.a2c_optimizer.state_dict()
            }
        }
        
        lock_path = os.path.join(self.checkpoint_dir, ".lock")
        try:
            with filelock.FileLock(lock_path, timeout=10):
                torch.save(state, path)
                self.logger.log_event("deepscalper.model.saved", context={"path": path})
        except filelock.Timeout:
            print(f"WARNING: Could not acquire lock for checkpoint {path}. Skipping save.", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint {path}: {e}", flush=True)

    def evaluate(self, eval_env, num_episodes=5) -> Dict[str, float]:
        """
        Evaluate the current ensemble on a separate environment (Validation Set).
        Returns metrics dict (Sharpe, Total Reward, etc.)
        """
        print(f"Starting Evaluation on {num_episodes} episodes...")
        total_rewards = []
        
        # Detect Vector Env
        is_vector = False
        import gymnasium as gym
        if isinstance(eval_env, gym.vector.VectorEnv):
             is_vector = True
             num_envs = eval_env.num_envs
        else:
             num_envs = 1
             
        # Prepare for evaluation loop
        # We need to handle VectorEnv (auto-reset) and Single Env (manual reset) differently
        
        if is_vector:
            # --- VECTOR ENV STRATEGY ---
            finished_episodes = [0] * num_envs
            current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
            
            # Reset once at start
            obs, info = eval_env.reset()
            micro, private, macro = self._unpack_obs(obs)
            
            # Run untill we collect enough episodes
            # Safety: Max steps to prevent infinite loop if something is really broken
            max_steps = 100000 
            step_count = 0
            
            # We want total_episodes >= num_episodes
            while sum(finished_episodes) < num_episodes:
                with torch.no_grad():
                    # Gating
                    weights = self.ensemble.gating(macro)
                    
                    # Agent Probs
                    p_dqn_dir, p_dqn_price, p_dqn_vol = self.ensemble.dqn.get_probs(micro, private, macro)
                    logits_ppo_dir, logits_ppo_price, logits_ppo_vol, _ = self.ensemble.ppo.network(micro, private, macro)
                    p_ppo_dir = torch.softmax(logits_ppo_dir, dim=1)
                    p_ppo_price = torch.softmax(logits_ppo_price, dim=1)
                    p_ppo_vol = torch.softmax(logits_ppo_vol, dim=1)
                    
                    logits_a2c_dir, logits_a2c_price, logits_a2c_vol, _ = self.ensemble.a2c.network(micro, private, macro)
                    p_a2c_dir = torch.softmax(logits_a2c_dir, dim=1)
                    p_a2c_price = torch.softmax(logits_a2c_price, dim=1)
                    p_a2c_vol = torch.softmax(logits_a2c_vol, dim=1)
                    
                    # Ensemble (Weighted Average)
                    w_dqn = weights[:, 0].unsqueeze(1)
                    w_ppo = weights[:, 1].unsqueeze(1)
                    w_a2c = weights[:, 2].unsqueeze(1)
                    
                    final_dir = w_dqn * p_dqn_dir + w_ppo * p_ppo_dir + w_a2c * p_a2c_dir
                    final_price = w_dqn * p_dqn_price + w_ppo * p_ppo_price + w_a2c * p_a2c_price
                    final_vol = w_dqn * p_dqn_vol + w_ppo * p_ppo_vol + w_a2c * p_a2c_vol
                    
                    # Sample
                    from torch.distributions import Categorical
                    a_dir = Categorical(probs=final_dir).sample()
                    a_price = Categorical(probs=final_price).sample()
                    a_vol = Categorical(probs=final_vol).sample()
                    
                    action_vector = torch.stack([a_dir, a_price, a_vol], dim=1).cpu().numpy()

                # Step
                next_obs, reward, terminated, truncated, info = eval_env.step(action_vector)
                dones = terminated | truncated
                
                # Accumulate Rewards
                current_ep_rewards += reward
                
                # Check completions
                if np.any(dones):
                    for i, d in enumerate(dones):
                        if d:
                            finished_episodes[i] += 1
                            if sum(finished_episodes) <= num_episodes: # Only record needed episodes? Or all?
                                # Let's record all valid completions
                                total_rewards.append(current_ep_rewards[i])
                            current_ep_rewards[i] = 0.0
                
                obs = next_obs
                micro, private, macro = self._unpack_obs(obs)
                step_count += 1
                
                if step_count > max_steps:
                    print(f"WARNING: Evaluation timed out after {max_steps} steps.")
                    break
        else:
            # --- SINGLE ENV STRATEGY (Legacy Logic) ---
            for _ in range(num_episodes):
                obs, info = eval_env.reset()
                micro, private, macro = self._unpack_obs(obs)
                
                done = False
                ep_reward = 0.0
                
                while not done:
                    with torch.no_grad():
                        # Gating
                        weights = self.ensemble.gating(macro)
                        
                        # Agent Probs
                        p_dqn_dir, p_dqn_price, p_dqn_vol = self.ensemble.dqn.get_probs(micro, private, macro)
                        logits_ppo_dir, logits_ppo_price, logits_ppo_vol, _ = self.ensemble.ppo.network(micro, private, macro)
                        p_ppo_dir = torch.softmax(logits_ppo_dir, dim=1)
                        p_ppo_price = torch.softmax(logits_ppo_price, dim=1)
                        p_ppo_vol = torch.softmax(logits_ppo_vol, dim=1)
                        
                        logits_a2c_dir, logits_a2c_price, logits_a2c_vol, _ = self.ensemble.a2c.network(micro, private, macro)
                        p_a2c_dir = torch.softmax(logits_a2c_dir, dim=1)
                        p_a2c_price = torch.softmax(logits_a2c_price, dim=1)
                        p_a2c_vol = torch.softmax(logits_a2c_vol, dim=1)
                        
                        # Ensemble (Weighted Average)
                        w_dqn = weights[:, 0].unsqueeze(1)
                        w_ppo = weights[:, 1].unsqueeze(1)
                        w_a2c = weights[:, 2].unsqueeze(1)
                        
                        final_dir = w_dqn * p_dqn_dir + w_ppo * p_ppo_dir + w_a2c * p_a2c_dir
                        final_price = w_dqn * p_dqn_price + w_ppo * p_ppo_price + w_a2c * p_a2c_price
                        final_vol = w_dqn * p_dqn_vol + w_ppo * p_ppo_vol + w_a2c * p_a2c_vol
                        
                        # Sample
                        from torch.distributions import Categorical
                        a_dir = Categorical(probs=final_dir).sample()
                        a_price = Categorical(probs=final_price).sample()
                        a_vol = Categorical(probs=final_vol).sample()
                        
                        # Single Action
                        action = np.array([a_dir.item(), a_price.item(), a_vol.item()])

                    next_obs, reward, terminated, truncated, info = eval_env.step(action)
                    micro, private, macro = self._unpack_obs(next_obs)
                    ep_reward += reward
                    
                    if terminated or truncated:
                        done = True
                
                total_rewards.append(ep_reward)
                
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        # Calculate Sharpe (Simplified: Mean / Std of episode rewards is NOT Sharpe)
        # Sharpe is Mean(Returns) / Std(Returns).
        # We need Daily Returns ideally.
        # But for HPO on episode level, Sharpe ~ Avg_Rew / Std_Rew (if rew is PnL)
        # Let's use Risk-Adjusted Return proxy: Mean / (Std + epsilon)
        sharpe_proxy = avg_reward / (std_reward + 1e-6)
        
        return {
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "sharpe": float(sharpe_proxy)
        }

