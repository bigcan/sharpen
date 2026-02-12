
import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer

class MockEnv:
    def __init__(self, num_envs=12):
        self.num_envs = num_envs
        self.action_space = MagicMock()
        self.observation_space = MagicMock()
        
    def reset(self):
        # Return mock obs with shape (B, Window, Features)
        micro = np.zeros((self.num_envs, 50, 44), dtype=np.float32)
        macro = np.zeros((self.num_envs, 10), dtype=np.float32)
        private = np.zeros((self.num_envs, 50, 2), dtype=np.float32)
        return {"micro": micro, "macro": macro, "private": private}, {}
    
    def step(self, actions):
        obs, _ = self.reset()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, truncated, infos

class TestTrainerHPO(unittest.TestCase):
    def setUp(self):
        self.config = {
            "env": {
                "action": {
                    "price_bins": 5,
                    "signed_qty_proportions": [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
                }
            },
            "network": {
                "micro_config": {"input_size": 44, "private_input_size": 2, "hidden_size": 256, "rnn_type": "LSTM"},
                "macro_config": {"input_size": 10, "hidden_sizes": [256, 128]}
            },
            "agents": {
                "bdq": {
                    "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 1000, 
                    "batch_size": 32, "target_update_freq": 1000, 
                    "epsilon_decay": 0.99999, # Default slow decay
                    "learning_starts": 100,
                    "checkpoint_interval": 1000
                }
            },
            "training": {
                "total_timesteps": 100000, "log_interval": 100,
                "use_amp": False, "verbose_logging": False
            },
            "hpo": {
                "enabled": True,
                "steps_per_trial": 50000
            }
        }
        
    def test_hpo_epsilon_decay_override(self):
        num_envs = 10
        env = MockEnv(num_envs=num_envs)
        
        # Init Trainer in HPO Mode
        trainer = DeepScalperTrainer(env, self.config, hpo_mode=True, device="cpu")
        
        # Default Epsilon Decay from config is 0.99999
        # But we expect the trainer to override it in train()
        # Wait, the override happens IN train(), not __init__
        
        # We need to run train() briefly to trigger the override or check logic
        # Actually, let's just inspect the logic by running a partial train
        
        # Catch print output? Or just check attribute after calling train
        # We can't easily run train without running the loop.
        # But we can check if the logic works by simulating what train() does.
        
        # Let's mock agent.decay_epsilon to see what decay value it uses?
        # No, the trainer sets self.agent.epsilon_decay attribute.
        
        # Call train() for 1 step
        try:
            trainer.train(start_step=0, optuna_trial=None, pruning_callback=None)
        except Exception as e:
            # It might fail due to missing components in mock, but let's see
            pass
            
        # Check if epsilon_decay was modified
        # Expected calculation:
        steps = 50000
        n_calls = steps / num_envs # 5000
        epsilon_end = 0.01
        # decay = exp(ln(0.01) / 5000)
        expected_decay = np.exp(np.log(0.01) / n_calls) # ~ 0.999079
        
        print(f"\nTrainer Agent Epsilon Decay: {trainer.agent.epsilon_decay}")
        print(f"Expected Decay: {expected_decay}")
        
        self.assertAlmostEqual(trainer.agent.epsilon_decay, expected_decay, places=5)
        self.assertNotEqual(trainer.agent.epsilon_decay, 0.99999) # Should NOT be default

if __name__ == '__main__':
    unittest.main()
