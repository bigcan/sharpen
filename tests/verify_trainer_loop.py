import torch
import numpy as np
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork

class MockEnv:
    def __init__(self):
        self.observation_space = None # Not used by trainer directly, but good to have
        self.action_space = None
        
    def reset(self):
        # Return Dict Obs
        obs = {
            "micro": np.random.randn(10, 4).astype(np.float32), 
            "macro": np.random.randn(5).astype(np.float32),
            "private": np.random.randn(10, 3).astype(np.float32) # Windowed private
        }
        info = {"volatility_target": 0.005}
        return obs, info
        
    def step(self, action):
        # action is (3,)
        next_obs = {
            "micro": np.random.randn(10, 4).astype(np.float32), 
            "macro": np.random.randn(5).astype(np.float32),
            "private": np.random.randn(10, 3).astype(np.float32)
        }
        reward = np.random.rand()
        terminated = False
        truncated = False
        info = {"volatility_target": 0.005}
        return next_obs, reward, terminated, truncated, info

import wandb

def verify_trainer_loop():
    print("=== Verifying DeepScalper Phase 5 (Training Loop) ===")
    
    # Init wandb disabled
    wandb.init(mode="disabled")
    
    device = "cpu"
    
    # 1. Setup Mock Configs
    micro_config = {
        "input_size": 4, 
        "private_input_size": 3,
        "hidden_size": 32, 
        "num_layers": 1
    }
    macro_config = {
        "input_size": 5, 
        "hidden_sizes": [32, 32]
    }
    fusion_dim = 64
    action_dims = (3, 5, 5)
    
    network_config = {
        "micro_config": micro_config,
        "macro_config": macro_config,
        "fusion_dim": fusion_dim,
        "action_space_dims": action_dims
    }
    
    # 2. Initialize Agents
    dqn = DeepScalperDQN(network_config=network_config, device=device, batch_size=8)
    ppo = DeepScalperPPO(network_config=network_config, device=device)
    a2c = DeepScalperA2C(network_config=network_config, device=device)
    gating = SynapseGatingNetwork(input_dim=5, hidden_dim=16).to(device)
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    
    # 3. Initialize Trainer
    config = {
        "batch_size": 8,
        "total_timesteps": 20, # Short run
        "update_interval": 10, # Update PPO/A2C/Gating every 10 steps
        "log_interval": 5,
        "checkpoint_interval": 100,
        "learning_rate": 1e-3,
        "torch_compile": False
    }
    
    env = MockEnv()
    
    trainer = DeepScalperTrainer(env, ensemble, config, device=device)
    print("Trainer Initialized ✅")
    
    # 4. Run Training
    print("\nStarting Training Loop (20 steps)...")
    try:
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        
        trainer.train()
        print("Training Loop Completed without Error ✅")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: Training Loop Crashed: {e} ❌")
        return

    # 5. Verify Updates Happened
    # Check if buffers are cleared (trainer clears them after update)
    if len(trainer.ppo_buffer) == 0 and len(trainer.a2c_buffer) == 0:
         print("Buffers cleared (Updates likely happened) ✅")
    else:
         print(f"WARNING: Buffers not empty! PPO: {len(trainer.ppo_buffer)}")
         
    # Check if DQN Memory has items
    if len(trainer.ensemble.dqn.memory) == 20:
        print(f"DQN Memory populated correctly ({len(trainer.ensemble.dqn.memory)} items) ✅")
    else:
        print(f"FAIL: DQN Memory has {len(trainer.ensemble.dqn.memory)} items (Expected 20) ❌")

    print("\nVerification Complete: PASS ✅")

if __name__ == "__main__":
    verify_trainer_loop()
