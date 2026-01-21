import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
import time
from collections import deque
import random

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
        
        self.global_step = 0
        
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
            # We use the ensemble to predict
            # Note: During training, we might want to force exploration for DQN specific branches?
            # The ensemble.predict uses DQN's softmax probs which includes temperature.
            # Epsilon-greedy is handled inside DQN if called directly, but ensemble calls get_probs.
            # We rely on the softmax temperature and the inherent stochasticity of PPO/A2C for exploration.
            
            action_vector = self.ensemble.predict(micro, macro) # Returns np array [dir, price, vol]
            
            # 2. Step Environment
            # Action vector needs to be converted if Env expects something else, 
            # but DeepScalperEnv expects [dir, price, vol] usually.
            
            next_obs, reward, terminated, truncated, info = self.env.step(action_vector)
            
            next_micro, next_macro = self._unpack_obs(next_obs)
            done = terminated or truncated
            
            episode_rewards += reward
            episode_steps += 1
            
            # 3. Store Transitions
            
            # Store for DQN (Off-policy)
            # Reconstruct dict state for DQN buffer compatibility
            state_dict = {"micro": micro.cpu().numpy(), "macro": macro.cpu().numpy()}
            next_state_dict = {"micro": next_micro.cpu().numpy(), "macro": next_macro.cpu().numpy()}
            
            self.ensemble.dqn.memory.push(
                state_dict, 
                action_vector, 
                reward, 
                next_state_dict, 
                done
            )
            
            # Store for PPO/A2C (On-policy - simplified)
            # We need log_probs for PPO. The ensemble doesn't return them currently.
            # For this MVP phase, we will focus on the DQN update loop 
            # and getting the Gating network training hooked up.
            # PPO/A2C full implementation requires trajectory collection.
            
            # 4. Update Steps
            
            # A. Train DQN
            dqn_loss = self.ensemble.dqn.train_step()
            
            # B. Train Gating Network (Meta-Controller)
            # Objective: Maximize reward by adjusting weights?
            # Or use a separate meta-gradient approach.
            # Simple approach: Reinforce the weight selection based on reward?
            # For now, we'll placeholder this or use a simple supervised signal if available.
            
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

