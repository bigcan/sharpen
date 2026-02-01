
import pytest
import unittest
import torch
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock, patch

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C

class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
             "micro": gym.spaces.Box(low=-1, high=1, shape=(50, 20)),
             "macro": gym.spaces.Box(low=-1, high=1, shape=(11,)),
             "private": gym.spaces.Box(low=0, high=1, shape=(50, 2))
        })
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
    
    def reset(self, **kwargs):
        return {
            "micro": np.random.randn(50, 20).astype(np.float32),
            "macro": np.random.randn(11).astype(np.float32),
            "private": np.zeros((50, 2)).astype(np.float32)
        }, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}

class TestDeepScalperRobustness(unittest.TestCase):

    def setUp(self):
        # Setup Minimal Ensemble
        net_config = {
            "micro_config": {"input_size": 20, "hidden_size": 64, "private_input_size": 2},
            "macro_config": {"input_size": 11, "hidden_sizes": [64]}
        }
        self.device = "cpu"
        
        self.dqn = MagicMock(spec=DeepScalperDQN)
        self.ppo = MagicMock(spec=DeepScalperPPO)
        self.a2c = MagicMock(spec=DeepScalperA2C)
        self.gating = SynapseGatingNetwork(input_dim=11, hidden_dim=64)
        
        # Mock get_probs to return dummy probabilities
        # Shapes: (Batch, 3), (Batch, 5), (Batch, 5)
        dummy_probs = (
            torch.tensor([[0.33, 0.33, 0.34]]), 
            torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]), 
            torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        )
        self.dqn.get_probs.return_value = dummy_probs
        self.ppo.get_probs.return_value = dummy_probs
        self.a2c.get_probs.return_value = dummy_probs
        
        # Add network attribute for Trainer compilation/freezing checks
        self.dqn.policy_net = MagicMock(spec=torch.nn.Module)
        self.dqn.target_net = MagicMock(spec=torch.nn.Module)
        self.ppo.network = MagicMock(spec=torch.nn.Module)
        self.ppo.network.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        self.a2c.network = MagicMock(spec=torch.nn.Module)
        self.a2c.network.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        # self.gating is already defined above, but fine to redefine or rely on previous
        
        self.ensemble = DeepScalperEnsemble(self.dqn, self.ppo, self.a2c, self.gating, device=self.device)
        self.env = MockEnv()
        self.config = {"env": {"num_envs": 1}}
        self.trainer = DeepScalperTrainer(self.env, self.ensemble, self.config, device=self.device)

    def test_evaluate_zero_episodes(self):
        """Test that evaluate handles num_episodes=0 gracefully."""
        # This currently fails or warns in the implementation (returns NaN or crashes)
        metrics = self.trainer.evaluate(self.env, num_episodes=0)
        
        # Expectation: Should return a dict with valid types, likely empty metrics or 0.0
        # Specifically checking it doesn't crash
        self.assertIsInstance(metrics, dict)
        if 'sharpe' in metrics:
            self.assertFalse(np.isnan(metrics['sharpe']), "Sharpe should not be NaN")
            self.assertEqual(metrics['sharpe'], 0.0)

    def test_ensemble_nan_handling(self):
        """Test that ensemble predict handles NaN in macro features gracefully."""
        
        # Create NaN input
        micro = torch.randn(1, 50, 20)
        private = torch.randn(1, 50, 2)
        macro = torch.randn(1, 11)
        macro[0, 0] = float('nan')
        
        # Expectation: The current implementation might crash in Softmax or propagate NaNs
        # We want it to use default weights
        
        try:
            actions, weights = self.ensemble.predict(micro, private, macro)
            
            # Check weights are not NaN
            w_dqn = weights["w_dqn"][0]
            self.assertFalse(np.isnan(w_dqn), "DQN Weight should not be NaN")
            
            # Check if it defaulted (Logic to be implemented: 0.33 each)
            # self.assertAlmostEqual(w_dqn, 0.333, delta=0.01)
            
        except RuntimeError as e:
            self.fail(f"Ensemble crashed on NaN input: {e}")

if __name__ == '__main__':
    unittest.main()
