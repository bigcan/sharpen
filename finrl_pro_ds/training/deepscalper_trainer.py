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
        
        # Buffers
        # DQN has its own buffer. PPO/A2C need rollout buffers.
        self.ppo_buffer = [] 
        self.a2c_buffer = []
        self.gating_buffer = []
        
        self.global_step = 0

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
        micro_s, macro_s, actions, old_log_probs, rewards, values, next_values_list, dones = zip(*buffer)
        
        # Convert to Tensors
        micro_s = torch.stack(micro_s)
        macro_s = torch.stack(macro_s)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(old_log_probs).detach() # (B, 3)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Advantages (GAE)
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device).detach()
        next_values = torch.tensor(next_values_list, dtype=torch.float32).to(self.device).detach()
        
        advantages = self.compute_gae(rewards, values_t, next_values, dones, self.gamma, 0.95)
        returns = advantages + values_t
        
        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Epochs
        for _ in range(4): # K_epochs
            # Forward Pass
            logits_dir, logits_price, logits_vol, current_values = agent.network(micro_s, macro_s)
            
            # Calculate current log probs of the wrapper actions
            # Actions: (B, 3) -> Dir, Price, Vol
            # We need log_prob for EACH branch
            dist_dir = Categorical(logits=logits_dir)
            dist_price = Categorical(logits=logits_price)
            dist_vol = Categorical(logits=logits_vol)
            
            curr_log_prob_dir = dist_dir.log_prob(actions[:, 0])
            curr_log_prob_price = dist_price.log_prob(actions[:, 1])
            curr_log_prob_vol = dist_vol.log_prob(actions[:, 2])
            
            # Sum or Average log probs across branches? Sum is joint prob.
            curr_log_probs = curr_log_prob_dir + curr_log_prob_price + curr_log_prob_vol
            
            # CRITICAL FIX for Audit 1.1: Use correct old_log_probs (from buffer)
            # The buffer stores log_prob of the ACTION taken, under the POLICY that took it (Ensemble).
            # But wait, 'old_log_probs' passed here comes from the buffer.
            # In train(), we capture 'ppo_log_prob' which was PPO's log prob of the action.
            # The Audit says: "PPO's importance sampling ratio becomes undefined because action was NOT sampled from pi_old".
            # Correct fix: Ideally PPO trains on its own data. Or we use V-Trace / Importance Sampling against the BEHAVIOR policy (Ensemble).
            # Simplified Fix (Approximation): 
            # Treat the Ensemble's choice as "the action". 
            # We want PPO to increase prob of this action if Advantage > 0.
            # Ratio = pi_new(a) / pi_old_ppo(a). 
            # If we use pi_old_ppo(a) as the denominator, it cancels out the fact that PPO *assigned* that prob at time t.
            # This is standard Off-Policy PPO (PPO-O).
            # The issue identified in Audit is that "a was NOT sampled from pi_old".
            # Actually, standard PPO requires: Ratio = pi_now(a) / pi_behavior(a).
            # Here pi_behavior = Ensemble.
            # So the denominator 'old_log_probs_sum' MUST BE the log_prob under the ENSEMBLE.
            # Let's assume the buffer contains ENSEMBLE log probs.
            # We need to change what we store in the buffer in train().
            
            old_log_probs_sum = old_log_probs.sum(dim=1)
            
            # Ratios
            ratios = torch.exp(curr_log_probs - old_log_probs_sum)
            
            # Surrogate Losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            entropy = dist_dir.entropy() + dist_price.entropy() + dist_vol.entropy()
            entropy_loss = -0.01 * entropy.mean()
            
            loss = policy_loss + value_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
            optimizer.step()
            
        return loss.item()

    def update_a2c(self, agent: DeepScalperA2C, optimizer: optim.Optimizer, buffer: List):
        if not buffer: return
        
        micro_s, macro_s, actions, _, rewards, values, next_values_list, dones = zip(*buffer)
        
        micro_s = torch.stack(micro_s)
        macro_s = torch.stack(macro_s)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        next_values = torch.tensor(next_values_list, dtype=torch.float32).to(self.device).detach()
        advantages = self.compute_gae(rewards, values_t, next_values, dones, self.gamma, 0.95)
        returns = advantages + values_t
        
        # Single Update Step
        logits_dir, logits_price, logits_vol, current_values = agent.network(micro_s, macro_s)
        
        dist_dir = Categorical(logits=logits_dir)
        dist_price = Categorical(logits=logits_price)
        dist_vol = Categorical(logits=logits_vol)
        
        log_prob_dir = dist_dir.log_prob(actions[:, 0])
        log_prob_price = dist_price.log_prob(actions[:, 1])
        log_prob_vol = dist_vol.log_prob(actions[:, 2])
        log_probs = log_prob_dir + log_prob_price + log_prob_vol
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
        
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        nn.utils.clip_grad_norm_(agent.network.parameters(), 0.5)
        optimizer.step()
        
        return policy_loss.item() + value_loss.item()

    def update_gating(self, buffer: List):
        """Update Gating Network using REINFORCE"""
        # We want to increase prob of weights that led to high rewards.
        # Inputs: Macro states, Gating Weights (actions), Rewards.
        # Since gating weights are continuous outputs of Softmax, we treat them as 'actions' 
        # but REINFORCE typically needs discrete choices or Gaussian.
        # Here: We differentiate the ENTIRE chain if we had full differentiability, but we don't.
        # Simplified: Treat the 'dominant' agent as the choice and REINFORCE that?
        # Better: PPO for the Gating Network?
        # Simplest feasible for now: Use the rewards to weight the gradients of the gating net output.
        # Or: Supervised Proxy -> Which agent *would have* performed best? (requires hindsight)
        
        # Let's use a simple REINFORCE-like update on the weight vectors.
        if not buffer: return
        
        macro_s, weights, rewards, dones = zip(*buffer)
        
        # Calculate Advantages (using rewards-baseline)
        # Simple Baseline: Mean reward of batch
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Normalize Rewards (Advantage Proxy)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        macro_s = torch.stack(macro_s)
        old_weights = torch.stack(weights).detach() # Weights used during sampling
        
        # Forward Pass
        curr_weights = self.ensemble.gating(macro_s)
        
        # Audit Fix 1.2: Correct Gradient Flow
        # We want to increase the probability of the weights used if advantage > 0.
        # But 'weights' is continuous.
        # Loss = - (current_weights * old_weights * advantage).sum(dim=1).mean()
        # This treats 'old_weights' as a direction vector we want to align with.
        
        loss = - (curr_weights * old_weights * adv.unsqueeze(1)).sum(dim=1).mean()
        
        self.gating_optimizer.zero_grad()
        loss.backward()
        self.gating_optimizer.step()
        
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
            
        micro, macro = self._unpack_obs(obs) 
        # _unpack_obs handles adding batch dim if missing for single env.
        # For VecEnv, micro is (B, W, F), macro is (B, F). Perfect.
        
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
        
        for step in range(self.total_timesteps):
            self.global_step = step
            
            # 1. Select Action (Voting)
            with torch.no_grad():
                weights = self.ensemble.gating(macro) # (B, 3)
                
                # Individual Probs
                p_dqn_dir, p_dqn_price, p_dqn_vol = self.ensemble.dqn.get_probs(micro, macro)
                logits_ppo_dir, logits_ppo_price, logits_ppo_vol, val_ppo = self.ensemble.ppo.network(micro, macro)
                p_ppo_dir = torch.softmax(logits_ppo_dir, dim=1)
                p_ppo_price = torch.softmax(logits_ppo_price, dim=1)
                p_ppo_vol = torch.softmax(logits_ppo_vol, dim=1)
                
                logits_a2c_dir, logits_a2c_price, logits_a2c_vol, val_a2c = self.ensemble.a2c.network(micro, macro)
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
            next_obs, reward, terminated, truncated, info = self.env.step(action_vector)
            
            next_micro, next_macro = self._unpack_obs(next_obs)
            
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
                
                # Handling next_val for PPO/A2C
                with torch.no_grad():
                    _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_macro)
                    _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_macro)
                
                # Loop to push to buffers individually (simplest integration with current buffers)
                for i in range(num_envs):
                    # Identify correct next_state
                    # If done, next_obs[i] is reset state. We need terminal state.
                    if dones[i] and "final_observation" in info:
                        # Gymnasium VectorEnv: info['final_observation'][i] is the terminal obs
                        # Note: info['final_observation'] is a list or array
                        term_obs = info['final_observation'][i]
                        term_micro, term_macro = self._unpack_obs(term_obs) # This creates (1, ...) tensors
                        
                        # Use terminal state for buffer
                        s_micro = term_micro.squeeze(0)
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
                        s_macro = next_macro[i]
                    
                    # Current State
                    c_micro = micro[i]
                    c_macro = macro[i]
                    
                    # Store DQN
                    # DQN Memory requires numpy dicts
                    state_dict = {"micro": c_micro.cpu().numpy(), "macro": c_macro.cpu().numpy()}
                    next_state_dict = {"micro": s_micro.cpu().numpy(), "macro": s_macro.cpu().numpy()}
                    
                    self.ensemble.dqn.memory.push(
                        state_dict, 
                        action_vector[i], 
                        reward[i], 
                        next_state_dict, 
                        bool(dones[i])
                    )
                    
                    # Store PPO/A2C
                    # (micro, macro, action, log_prob, reward, val, val_next, done)
                    self.ppo_buffer.append((
                        c_micro, c_macro, action_vector[i], ppo_log_prob[i], 
                        float(reward[i]), float(val_ppo[i]), float(val_next_ppo[i]), bool(dones[i])
                    ))
                    
                    self.a2c_buffer.append((
                        c_micro, c_macro, action_vector[i], None, 
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
                    _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_macro)
                    _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_macro)
                
                # Buffer Push (Squeeze batch dims for storage as buffers expect single items usually)
                state_dict = {"micro": micro.squeeze(0).cpu().numpy(), "macro": macro.squeeze(0).cpu().numpy()}
                next_state_dict = {"micro": next_micro.squeeze(0).cpu().numpy(), "macro": next_macro.squeeze(0).cpu().numpy()}
                
                self.ensemble.dqn.memory.push(state_dict, action_vector[0], reward, next_state_dict, done)
                
                self.ppo_buffer.append((
                    micro.squeeze(0), macro.squeeze(0), action_vector[0], ppo_log_prob.squeeze(0),
                    reward, val_ppo.item(), val_next_ppo.item(), done
                ))
                self.a2c_buffer.append((
                    micro.squeeze(0), macro.squeeze(0), action_vector[0], None,
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
                    
                    obs, info = self.env.reset()
                    next_micro, next_macro = self._unpack_obs(obs)
            
            # 4. Updates
            
            # A. Train DQN
            dqn_loss = self.ensemble.dqn.train_step()
            
            # B. Train PPO/A2C
            if (step + 1) % self.config.get("update_interval", 256) == 0:
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
            
            if step % self.config.get("log_interval", 100) == 0 and dqn_loss is not None:
                # Log avg reward if using vector envs? reward is array.
                r = reward.mean() if is_vector_env else reward
                wandb.log({"train/dqn_loss": dqn_loss, "train/step_reward_mean": r, "train/global_step": self.global_step})

            # Save Checkpoint
            if (step + 1) % self.checkpoint_interval == 0:
                ckpt_path = f"checkpoints/checkpoint_{step+1}.pth"
                self.save_checkpoint(ckpt_path)
                
                # Check for max checkpoints and delete old ones
                # Not implemented yet but planned
                pass

            # Update Obs
            micro = next_micro
            macro = next_macro
            obs = next_obs
        
        self.logger.log_event("deepscalper.training.complete")
                
    def _unpack_obs(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dict observation to tensors on device.
        Expected keys: 'micro' (Window, Feat), 'macro' (Feat)
        """
        # Handle cases where env returns numpy arrays
        if isinstance(obs, dict):
            micro_np = obs.get("micro")
            macro_np = obs.get("macro")
            
            if micro_np is None:
                raise ValueError(f"Observation missing 'micro' key. Keys found: {list(obs.keys())}")
            
            # If batch dim missing, add it
            if len(micro_np.shape) == 2:
                micro_t = torch.tensor(micro_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                micro_t = torch.tensor(micro_np, dtype=torch.float32).to(self.device)
                
            # Handle optional macro if system design allows, though DeepScalper requires it
            if macro_np is None:
                 # If using a mock env that might return None or different structure?
                 # DeepScalper requires macro.
                 raise ValueError("Observation missing 'macro' key.")

            if len(macro_np.shape) == 1:
                macro_t = torch.tensor(macro_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                macro_t = torch.tensor(macro_np, dtype=torch.float32).to(self.device)
                
            return micro_t, macro_t
        else:
            raise ValueError(f"Expected dict observation, got {type(obs)}")

    def save_checkpoint(self, path: str):
        """Save all agent states"""
        state = {
            "dqn": self.ensemble.dqn.policy_net.state_dict(),
            "ppo": self.ensemble.ppo.network.state_dict(),
            "a2c": self.ensemble.a2c.network.state_dict(),
            "gating": self.ensemble.gating.state_dict(),
            "config": self.config
        }
        torch.save(state, path)
        self.logger.log_event("deepscalper.model.saved", context={"path": path})

