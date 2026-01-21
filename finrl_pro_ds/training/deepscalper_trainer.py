import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import time
from collections import deque
import random
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
        self.gamma = config.get("gamma", 0.99)
        self.total_timesteps = config.get("total_timesteps", 100000)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.target_update_freq = config.get("target_update_freq", 1000)
        
        # Gating Optimizer
        self.gating_optimizer = optim.Adam(
            self.ensemble.gating.parameters(), 
            lr=self.learning_rate
        )
        
        # We assume agents have their own optimizers initialized internally
        # but we might need to access them for coordinated updates if not.
        # Check: dqn_agent.py initializes its own optimizer.
        # PPO/A2C likely need optimizers attached or passed in.
        # For this implementation, we will assume policy agents need optimizers created here
        # or we update `policy_agents.py` to include them. 
        # Looking at policy_agents.py, it's just a network wrapper currently. 
        # We need to add optimizers for PPO and A2C here.
        
        self.ppo_optimizer = optim.Adam(self.ensemble.ppo.network.parameters(), lr=self.learning_rate)
        self.a2c_optimizer = optim.Adam(self.ensemble.a2c.network.parameters(), lr=self.learning_rate)
        
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
        
        macro_s = torch.stack(macro_s)
        weights = torch.stack(weights).detach() # The weights we outputted
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Forward pass to get current gradients
        current_weights = self.ensemble.gating(macro_s) # (B, 3)
        
        # Simple REINFORCE-like: 
        # maximize Sum( weight_i * reward )
        # usage of 'weights' (old) vs 'current_weights' (new):
        # We want to encourage the network to output the weights that led to high reward.
        # Loss = - (current_weights * rewards.unsqueeze(1) * weights).mean()
        # Interpretation: If reward is high, increase prob of the weights we chose.
        # But 'weights' are continuous... 
        
        # Let's try: Loss = - (current_weights * rewards.unsqueeze(1)).sum(dim=1).mean()
        # This pushes ALL weights up if reward is positive. We need to push the DOMINANT ones?
        # The 'weights' variable contains the actual mix used.
        # If we just maximize Expected[Reward], and Reward depends on Weights...
        # Differentiable Reward? No.
        
        # Conservative approach: 
        # Approximate gradient: (R - Baseline) * grad(log P) is for discrete.
        # For continuous actions (weights), we can treat it as Beta distribution or Dirichlet?
        
        # Fallback to simple correlation: 
        # If Reward > 0, minimize MSE(current_weights, ideal_weights)? No ideal known.
        
        # Simplest: Maximize (current_weights DOT observed_weights) * Reward
        # If observed_weights were good (High Reward), make current_weights similar.
        # Loss = - ( (current_weights * weights).sum(dim=1) * rewards ).mean()
        
        surrogate = (current_weights * weights).sum(dim=1) * rewards
        loss = -surrogate.mean()
        
        self.gating_optimizer.zero_grad()
        loss.backward()
        self.gating_optimizer.step()
        
        return loss.item()

        
    def train(self):
        """Main Training Loop"""
        self.logger.log_event("deepscalper.training.start")
        
        obs, info = self.env.reset()
        micro, macro = self._unpack_obs(obs)
        
        episode_rewards = 0
        episode_steps = 0
        episode_count = 0
        
        for step in range(self.total_timesteps):
            self.global_step = step
            
            # 1. Select Action (Voting)
            # We need to capture INTERMEDIATE outputs for training PPO/A2C/Gating
            # ensemble.predict is too high level. We need to manually call components here.
            
            with torch.no_grad():
                # Gating Weights
                weights = self.ensemble.gating(macro) # (1, 3)
                
                # Individual Probs
                # DQN
                p_dqn_dir, p_dqn_price, p_dqn_vol = self.ensemble.dqn.get_probs(micro, macro)
                
                # PPO
                logits_ppo_dir, logits_ppo_price, logits_ppo_vol, val_ppo = self.ensemble.ppo.network(micro, macro)
                p_ppo_dir = torch.softmax(logits_ppo_dir, dim=1)
                p_ppo_price = torch.softmax(logits_ppo_price, dim=1)
                p_ppo_vol = torch.softmax(logits_ppo_vol, dim=1)
                
                # A2C
                logits_a2c_dir, logits_a2c_price, logits_a2c_vol, val_a2c = self.ensemble.a2c.network(micro, macro)
                p_a2c_dir = torch.softmax(logits_a2c_dir, dim=1)
                p_a2c_price = torch.softmax(logits_a2c_price, dim=1)
                p_a2c_vol = torch.softmax(logits_a2c_vol, dim=1)
                
                # Ensemble Aggregate
                w_dqn, w_ppo, w_a2c = weights[0]
                
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
                
                action_vector = np.array([a_dir.item(), a_price.item(), a_vol.item()])

                # Calculate PPO-specific Log Probs for THIS action (for "off-policy" PPO update)
                # Note: We use the action selected by ensemble.
                ppo_log_dir = Categorical(logits=logits_ppo_dir).log_prob(a_dir)
                ppo_log_price = Categorical(logits=logits_ppo_price).log_prob(a_price)
                ppo_log_vol = Categorical(logits=logits_ppo_vol).log_prob(a_vol)
                ppo_log_prob = torch.stack([ppo_log_dir, ppo_log_price, ppo_log_vol], dim=1)

            # 2. Step Environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_vector)
            
            next_micro, next_macro = self._unpack_obs(next_obs)
            done = terminated or truncated
            
            episode_rewards += reward
            episode_steps += 1
            
            # 3. Store Transitions
            
            # Store for DQN (Off-policy)
            # We must squeeze the batch dim (1, W, F) -> (W, F) because 
            # the replay buffer expects single observations, and stacking adds the batch dim back.
            state_dict = {"micro": micro.squeeze(0).cpu().numpy(), "macro": macro.squeeze(0).cpu().numpy()}
            next_state_dict = {"micro": next_micro.squeeze(0).cpu().numpy(), "macro": next_macro.squeeze(0).cpu().numpy()}
            
            self.ensemble.dqn.memory.push(state_dict, action_vector, reward, next_state_dict, done)
            
            # Store for PPO/A2C
            # Buffer: (micro, macro, action, log_prob, reward, val, val_next, done)
            # We assume val_next ~= val from next step (bootstrapping 1 step)
            
            # Optimization: We already computed next_micro, next_macro. 
            # We need V(s') for PPO/A2C. This requires a forward pass. 
            # It's expensive but necessary for correct GAE with one-step lookahead storage.
            
            with torch.no_grad():
                _, _, _, val_next_ppo = self.ensemble.ppo.network(next_micro, next_macro)
                _, _, _, val_next_a2c = self.ensemble.a2c.network(next_micro, next_macro)
            
            self.ppo_buffer.append((micro.squeeze(0), macro.squeeze(0), action_vector, ppo_log_prob.squeeze(0), reward, val_ppo.item(), val_next_ppo.item(), done))
            self.a2c_buffer.append((micro.squeeze(0), macro.squeeze(0), action_vector, None, reward, val_a2c.item(), val_next_a2c.item(), done))
            
            # Gating Buffer: (macro, weights, reward, done)
            # weights is (1, 3) tensor
            self.gating_buffer.append((macro.squeeze(0), weights.squeeze(0), reward, done))
            
            # 4. Update Steps
            
            # A. Train DQN
            dqn_loss = self.ensemble.dqn.train_step()
            
            # B. Train PPO/A2C (Batch/Epoch based)
            # Update every N steps or episode end? PPO usually fixed horizon.
            update_interval = 256 # Example horizon
            
            if (step + 1) % update_interval == 0:
                ppo_loss = self.update_ppo(self.ensemble.ppo, self.ppo_optimizer, self.ppo_buffer)
                a2c_loss = self.update_a2c(self.ensemble.a2c, self.a2c_optimizer, self.a2c_buffer)
                gating_loss = self.update_gating(self.gating_buffer)
                
                # Log losses
                if ppo_loss is not None:
                     self.logger.log_event("deepscalper.training.update", context={
                         "step": step, "ppo_loss": ppo_loss, "a2c_loss": a2c_loss, "gating_loss": gating_loss
                     })
                
                self.ppo_buffer = [] # Clear buffers
                self.a2c_buffer = []
                self.gating_buffer = []
                
            # C. Train Gating
            # self.update_gating(...)
            
            if step % 100 == 0 and dqn_loss is not None:
                self.logger.log_event("deepscalper.training.step", context={
                    "step": step, 
                    "dqn_loss": dqn_loss,
                    "reward": reward
                })
                
            # Handle Episode End
            if done:
                self.logger.log_event("deepscalper.training.episode_end", context={
                    "episode": episode_count,
                    "reward": episode_rewards,
                    "length": episode_steps
                })
                
                obs, info = self.env.reset()
                micro, macro = self._unpack_obs(obs)
                episode_rewards = 0
                episode_steps = 0
                episode_count += 1
                
                # Critical: Clear buffers on episode end to prevent cross-episode contamination
                # PPO/A2C are on-policy and typically don't span episodes blindly without correct handling.
                self.ppo_buffer = []
                self.a2c_buffer = []
                self.gating_buffer = []
            else:
                micro = next_micro
                macro = next_macro
                obs = next_obs
                
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

