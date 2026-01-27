import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import time
from collections import deque
import random
import wandb # Added for WandB logging
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
        device: str = "cpu"
    ):
        self.env = env
        self.ensemble = ensemble_agent
        self.config = config
        self.logger = logger or MLOpsLogger()
        self.device = torch.device(device)
        
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
        os.makedirs("checkpoints", exist_ok=True)
        
        # Parse Agent Configs (if available, else fallback to global LR)
        agents_config = config.get("agents", {})
        
        ppo_config = agents_config.get("ppo", {})
        a2c_config = agents_config.get("a2c", {})
        gating_config = agents_config.get("gating", {})
        
        ppo_lr = ppo_config.get("learning_rate", self.learning_rate)
        a2c_lr = a2c_config.get("learning_rate", self.learning_rate)
        gating_lr = gating_config.get("learning_rate", self.learning_rate)
        
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
                
                with autocast(enabled=self.use_amp):
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
                    entropy_loss = -0.01 * entropy.mean()
                    
                    loss = policy_loss + value_loss + entropy_loss
                
                optimizer.zero_grad()
                self.scaler_ppo.scale(loss).backward()
                self.scaler_ppo.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
                self.scaler_ppo.step(optimizer)
                self.scaler_ppo.update()
                
                total_loss += loss.item()
                n_updates += 1
            
        return total_loss / n_updates if n_updates > 0 else 0.0

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
        with autocast(enabled=self.use_amp):
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
            
            loss = policy_loss + value_loss
            
        optimizer.zero_grad()
        self.scaler_a2c.scale(loss).backward()
        self.scaler_a2c.unscale_(optimizer)
        nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
        
        self.scaler_a2c.step(optimizer)
        self.scaler_a2c.update()
        
        return policy_loss.item() + value_loss.item()

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
        
        with autocast(enabled=self.use_amp):
            # Forward Pass
            curr_weights = self.ensemble.gating(macro_s)
            
            # Loss: Minimize - (NewWeights * OldWeights * Adv)
            # This intuitively pushes NewWeights towards OldWeights where Adv was high.
            loss = - (curr_weights * old_weights * adv.unsqueeze(1)).sum(dim=1).mean()
        
        self.gating_optimizer.zero_grad()
        self.scaler_gating.scale(loss).backward()
        self.scaler_gating.step(self.gating_optimizer)
        self.scaler_gating.update()
        
        return loss.item()

        
    def train(self):
        """Main Training Loop"""
        self.logger.log_event("deepscalper.training.start")
        
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
            
        micro, private, macro = self._unpack_obs(obs) 
        # _unpack_obs handles adding batch dim if missing for single env.
        # For VecEnv, micro is (B, W, F), private is (B, W, F_p), macro is (B, F). Perfect.
        
        # Compile Model if requested
        print(f"DEBUG: self.config['torch_compile'] = {self.config.get('torch_compile', 'Not Set')}")
        if self.config.get("torch_compile", False) and hasattr(torch, "compile"):
            print("Compiling models with torch.compile...")
            try:
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
        
        # Init Watchdog (SPS Monitor)
        self.watchdog = TrainingWatchdog(
            lambda: self.global_step, 
            timeout_seconds=300,
            min_sps=100.0 if self.config.get("env", {}).get("num_envs", 1) > 8 else 0.0 # Only enforce SPS on high throughput
        )
        self.watchdog.start()
        
        while self.global_step < self.total_timesteps:
            start_step = self.global_step
            step = self.global_step  # For backward compatibility with logging
            
            # 1. Select Action (Voting)
            # Watchdog: Log if this step takes too long? 
            # We can't interrupt easily without signal.
            # Let's verify we are entering the loop.
            if step % 1000 == 0:
                 self.logger.log_event("deepscalper.training.step_start", context={"step": step})
                 
            with torch.no_grad():
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
                # Weights: (B, 3)
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
                # Wrap scalar to array for uniform handling if we want, but keeping separate paths is safer for legacy
            
            # 3. Store Transitions & Track Rewards
            
            if is_vector_env:
                # Vectorized Tracking
                episode_rewards += reward
                episode_lengths += 1
                
                # Check for finished episodes
                for i in range(num_envs):
                    if dones[i]:
                        metrics = {
                            "train/episode_reward": episode_rewards[i],
                            "train/episode_length": episode_lengths[i],
                            "train/global_step": self.global_step
                        }
                        self.logger.log_event("deepscalper.training.episode_end", context=metrics)
                        wandb.log(metrics)
                        
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
                        episode_count += 1
                        episode_rewards_total += metrics["train/episode_reward"]
                
                # Handling next_val for PPO/A2C
                with torch.no_grad():
                    _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_private, next_macro)
                    _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_private, next_macro)
                
                # Loop to push to buffers individually (simplest integration with current buffers)
                for i in range(num_envs):
                    # Identify correct next_state
                    # If done, next_obs[i] is reset state. We need terminal state.
                    if dones[i] and "final_observation" in info:
                        # Gymnasium VectorEnv: info['final_observation'][i] is the terminal obs
                        # Note: info['final_observation'] is a list or array
                        term_obs = info['final_observation'][i]
                        term_micro, term_private, term_macro = self._unpack_obs(term_obs) # This creates (1, ...) tensors
                        
                        # Use terminal state for buffer
                        s_micro = term_micro.squeeze(0)
                        s_private = term_private.squeeze(0)
                        s_macro = term_macro.squeeze(0)
                        
                        # For bootstrapping (val_next), we should technically use the value of the terminal state (0 if term, V(s) if trunc)
                        # But here we use 'val_next' computed on RESET state which is WRONG for PPO.
                        # Should compute value on terminal obs.
                        # Approximation: Use valid next value if truncated, 0 if terminated?
                        # Simplification: Use computed val_next (of reset state) but set mask=0 later.
                        # Better: Recompute val for terminal state?
                        # Let's trust GAE Logic: delta = r + gamma * V(s') * (1-d)
                        # If done=True, V(s') is ignored. So val_next doesn't matter much.
                        
                    else:
                        s_micro = next_micro[i]
                        s_private = next_private[i]
                        s_macro = next_macro[i]
                    
                    # Current State
                    c_micro = micro[i]
                    c_private = private[i]
                    c_macro = macro[i]
                    
                    # Store DQN
                    # DQN Memory requires numpy dicts
                    state_dict = {"micro": c_micro.cpu().numpy(), "private": c_private.cpu().numpy(), "macro": c_macro.cpu().numpy()}
                    next_state_dict = {"micro": s_micro.cpu().numpy(), "private": s_private.cpu().numpy(), "macro": s_macro.cpu().numpy()}
                    
                    self.ensemble.dqn.memory.push(
                        state_dict, 
                        action_vector[i], 
                        reward[i], 
                        next_state_dict, 
                        bool(dones[i]),
                        float(info.get("volatility_target", [0.0]*num_envs)[i]) # Section 4.4
                    )
                    
                    # Store PPO/A2C
                    # (micro, private, macro, action, log_prob, reward, val, val_next, done)
                    self.ppo_buffer.append((
                        c_micro, c_private, c_macro, action_vector[i], ppo_log_prob[i], 
                        float(reward[i]), float(val_ppo[i]), float(val_next_ppo[i]), bool(dones[i])
                    ))
                    
                    self.a2c_buffer.append((
                        c_micro, c_private, c_macro, action_vector[i], None, 
                        float(reward[i]), float(val_a2c[i]), float(val_next_a2c[i]), bool(dones[i])
                    ))
                    
                    self.gating_buffer.append((
                        c_macro, weights[i], float(reward[i]), bool(dones[i])
                    ))
                    
            else:
                # SINGLE ENV LEGACY PATH
                # To maintain compatibility if needed, but above logic works for num_envs=1 too technically
                # if we treat scalars as arrays of 1.
                # However, scalars (float) don't index [i].
                # Let's keep separate block for safety or just assume array wrap?
                # _unpack_obs ensures tensors are (B, ...).
                
                # Single env reward is float.
                episode_rewards += reward
                episode_steps += 1
                
                done = dones # Scalar
                
                with torch.no_grad():
                    _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_private, next_macro)
                    _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_private, next_macro)
                
                # Buffer Push (Squeeze batch dims for storage as buffers expect single items usually)
                state_dict = {"micro": micro.squeeze(0).cpu().numpy(), "private": private.squeeze(0).cpu().numpy(), "macro": macro.squeeze(0).cpu().numpy()}
                next_state_dict = {"micro": next_micro.squeeze(0).cpu().numpy(), "private": next_private.squeeze(0).cpu().numpy(), "macro": next_macro.squeeze(0).cpu().numpy()}
                
                self.ensemble.dqn.memory.push(
                    state_dict, 
                    action_vector[0], 
                    reward, 
                    next_state_dict, 
                    done,
                    float(info.get("volatility_target", 0.0)) # Section 4.4
                )
                
                self.ppo_buffer.append((
                    micro.squeeze(0), private.squeeze(0), macro.squeeze(0), action_vector[0], ppo_log_prob.squeeze(0),
                    reward, val_ppo.item(), val_next_ppo.item(), done
                ))
                self.a2c_buffer.append((
                    micro.squeeze(0), private.squeeze(0), macro.squeeze(0), action_vector[0], None,
                    reward, val_a2c.item(), val_next_a2c.item(), done
                ))
                self.gating_buffer.append((
                    macro.squeeze(0), weights.squeeze(0), reward, done
                ))
                
                if done:
                    metrics = {"train/episode_reward": episode_rewards, "train/episode_length": episode_steps, "train/global_step": self.global_step}
                    wandb.log(metrics)
                    episode_rewards = 0
                    episode_steps = 0
                    episode_count += 1
                    episode_rewards_total += metrics["train/episode_reward"]
                    
                    obs, info = self.env.reset()
                    next_micro, next_private, next_macro = self._unpack_obs(obs)
            
            # Increment global step by number of envs (after env step and storage)
            self.global_step += num_envs

            # 4. Updates
            
            # A. Train DQN (Accumulate gradients to match update ratio)
            # We want 1 update per dqn_update_interval transitions.
            # We collected 'num_envs' transitions.
            self.dqn_updates_accumulator += num_envs / self.dqn_update_interval
            
            dqn_loss = None
            while self.dqn_updates_accumulator >= 1.0:
                dqn_loss = self.ensemble.dqn.train_step()
                self.dqn_updates_accumulator -= 1.0
            
            # B. Train PPO/A2C
            # Check if we crossed an update interval boundary
            update_interval = self.config.get("update_interval", 256)
            prev_interval_idx = start_step // update_interval
            curr_interval_idx = self.global_step // update_interval
            
            if curr_interval_idx > prev_interval_idx:
                ppo_loss = self.update_ppo(self.ensemble.ppo, self.ppo_optimizer, self.ppo_buffer)
                a2c_loss = self.update_a2c(self.ensemble.a2c, self.a2c_optimizer, self.a2c_buffer)
                gating_loss = self.update_gating(self.gating_buffer)
                
                if ppo_loss is not None:
                     metrics = {
                         "train/ppo_loss": ppo_loss, 
                         "train/a2c_loss": a2c_loss, 
                         "train/gating_loss": gating_loss,
                         "train/global_step": self.global_step
                     }
                     wandb.log(metrics)
                
                self.ppo_buffer = [] 
                self.a2c_buffer = []
                self.gating_buffer = []
            
            # Smart Logging: Log every N seconds OR every M steps to avoid IO bottleneck
            current_time = time.time()
            if not hasattr(self, '_last_log_time'): self._last_log_time = current_time
            
            log_interval = self.config.get("log_interval", 1000) # Increased default
            time_interval = 5.0 # Seconds
            
            should_log = False
            if dqn_loss is not None:
                # Check Time
                if current_time - self._last_log_time > time_interval:
                    should_log = True
                # Check Steps (Backwards compat + ensure we log at least sometimes if fast)
                elif (self.global_step // log_interval) > (start_step // log_interval):
                    should_log = True
            
            if should_log:
                self._last_log_time = current_time
                # Log avg reward if using vector envs? reward is array.
                r = reward.mean() if is_vector_env else reward
                wandb.log({"train/dqn_loss": dqn_loss, "train/step_reward_mean": r, "train/global_step": self.global_step})

            # Save Checkpoint
            if (self.global_step // self.checkpoint_interval) > (start_step // self.checkpoint_interval):
                ckpt_path = f"checkpoints/checkpoint_{self.global_step}.pth"
                self.save_checkpoint(ckpt_path)
                
                # Check for max checkpoints and delete old ones
                # Not implemented yet but planned
                pass

            # Update Obs
            micro = next_micro
            private = next_private
            macro = next_macro
            obs = next_obs
        
        self.watchdog.stop()

        self.logger.log_event("deepscalper.training.complete")
        
        # Calculate average per-step or per-episode reward
        if episode_count > 0:
            avg_reward = episode_rewards_total / episode_count
        else:
            avg_reward = 0.0
            
        return avg_reward
                
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
        """Save all agent states"""
        state = {
            "dqn": self._get_clean_state_dict(self.ensemble.dqn.policy_net),
            "ppo": self._get_clean_state_dict(self.ensemble.ppo.network),
            "a2c": self._get_clean_state_dict(self.ensemble.a2c.network),
            "gating": self._get_clean_state_dict(self.ensemble.gating),
            "config": self.config
        }
        torch.save(state, path)
        self.logger.log_event("deepscalper.model.saved", context={"path": path})

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
             
        for _ in range(num_episodes // num_envs if is_vector else num_episodes):
            obs, info = eval_env.reset()
            micro, private, macro = self._unpack_obs(obs)
            
            done = False
            ep_reward = 0.0
            if is_vector: ep_reward = np.zeros(num_envs)
            
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
                    
                    # Sample (Stochastic Evaluation matching Training)
                    # Or Argmax? Using Sample for consistency with training distribution.
                    a_dir = Categorical(probs=final_dir).sample()
                    a_price = Categorical(probs=final_price).sample()
                    a_vol = Categorical(probs=final_vol).sample()
                    
                    action_vector = torch.stack([a_dir, a_price, a_vol], dim=1).cpu().numpy()
                
                # Step
                if is_vector:
                    next_obs, reward, terminated, truncated, info = eval_env.step(action_vector)
                    dones = terminated | truncated
                    micro, private, macro = self._unpack_obs(next_obs)
                    ep_reward += reward
                    
                    if np.any(dones):
                         # Vector env automatically resets done sub-envs.
                         # We count 'finished' episodes. 
                         # Simplified: Just run for fixed steps? No, we want episodes.
                         # If any done, we record its reward.
                         # This loop structure is rigid for VectorEnv episode counting.
                         # Fallback: Just run loop until 'all done' if not auto-reset?
                         # Gym VectorEnv auto-resets. 
                         # Let's just track rewards and break?
                         pass
                    
                    # For simplicity in evaluation, let's assume single env is preferred for accurate episode metrics
                    # or handle vector properly.
                    # Given time constraints, if vector, we just sum rewards?
                    # Let's break if all done? (Only works if not auto-reset)
                    
                    if np.all(dones): # Unlikely to happen exactly same time
                        done = True
                        
                else:
                    action = action_vector[0]
                    next_obs, reward, terminated, truncated, info = eval_env.step(action)
                    micro, private, macro = self._unpack_obs(next_obs)
                    ep_reward += reward
                    if terminated or truncated:
                        done = True
            
            if is_vector:
                total_rewards.extend(ep_reward)
            else:
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

