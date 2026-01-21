import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork

# Shared Config
MICRO_CONFIG = {"input_size": 20, "hidden_size": 32, "rnn_type": "LSTM"}
MACRO_CONFIG = {"input_size": 11, "hidden_sizes": (32,)}
NET_CONFIG = {"micro_config": MICRO_CONFIG, "macro_config": MACRO_CONFIG}
TRAIN_CONFIG = {
    "batch_size": 4, 
    "gamma": 0.99, 
    "total_timesteps": 10,
    "agents": {
        "dqn": {"learning_rate": 0.001, "gamma": 0.99},
        "ppo": {"learning_rate": 0.002, "gamma": 0.99},
        "a2c": {"learning_rate": 0.003, "gamma": 0.99},
        "gating": {"learning_rate": 0.004}
    }
}

class MockEnv:
    def __init__(self):
        self.observation_space = None
        self.action_space = None
        
    def reset(self):
        # Micro: (20, 20), Macro: (11,)
        # Note: Trainer expects dict obs
        return {
            "micro": np.random.randn(20, 20),
            "macro": np.random.randn(11)
        }, {}
    
    def step(self, action):
        return self.reset()[0], 1.0, False, False, {}

@pytest.fixture
def trainer_setup():
    dqn = DeepScalperDQN(NET_CONFIG)
    ppo = DeepScalperPPO(NET_CONFIG)
    a2c = DeepScalperA2C(NET_CONFIG)
    gating = SynapseGatingNetwork(input_dim=11)
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating)
    env = MockEnv()
    
    trainer = DeepScalperTrainer(
        env=env,
        ensemble_agent=ensemble,
        config=TRAIN_CONFIG
    )
    return trainer

def test_trainer_initialization(trainer_setup):
    trainer = trainer_setup
    assert trainer.ensemble is not None
    assert trainer.env is not None
    assert trainer.batch_size == 4

def test_unpack_obs(trainer_setup):
    trainer = trainer_setup
    mock_obs = {
        "micro": np.random.randn(20, 20).astype(np.float32),
        "macro": np.random.randn(11).astype(np.float32)
    }
    micro_t, macro_t = trainer._unpack_obs(mock_obs)
    
    assert isinstance(micro_t, torch.Tensor)
    assert isinstance(macro_t, torch.Tensor)
    # Check batch dimension added
    assert micro_t.shape == (1, 20, 20)
    assert macro_t.shape == (1, 11)

def test_trainer_run_step(trainer_setup):
    trainer = trainer_setup
    # Run a very short training loop
    trainer.total_timesteps = 5
    trainer.train()
    
    # Check if buffer has data
    assert len(trainer.ensemble.dqn.memory) == 5

def test_trainer_save_checkpoint(trainer_setup, tmp_path):
    trainer = trainer_setup
    path = tmp_path / "test_ckpt.pth"
    trainer.save_checkpoint(str(path))
    assert path.exists()
