import pytest
import torch
import numpy as np
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork

# Shared Config
MICRO_CONFIG = {"input_size": 20, "hidden_size": 32, "rnn_type": "LSTM"}
MACRO_CONFIG = {"input_size": 11, "hidden_sizes": (32,)}
NET_CONFIG = {"micro_config": MICRO_CONFIG, "macro_config": MACRO_CONFIG}

@pytest.fixture
def dummy_input():
    # Batch size 1
    micro = torch.randn(1, 10, 20) # (B, Window, Feat)
    macro = torch.randn(1, 11)     # (B, Feat)
    return micro, macro

def test_dqn_agent_initialization():
    agent = DeepScalperDQN(network_config=NET_CONFIG)
    assert agent.policy_net is not None
    assert agent.target_net is not None

def test_dqn_predict(dummy_input):
    micro, macro = dummy_input
    agent = DeepScalperDQN(network_config=NET_CONFIG)
    
    action = agent.predict(micro, macro)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    # Check bounds
    assert 0 <= action[0] < 3
    assert 0 <= action[1] < 5
    assert 0 <= action[2] < 5

def test_dqn_train_step_mock():
    agent = DeepScalperDQN(network_config=NET_CONFIG, batch_size=2)
    # Fill memory
    dummy_obs = {"micro": np.zeros((10, 20)), "macro": np.zeros(11)}
    for _ in range(5):
        agent.memory.push(
            state=dummy_obs,
            action=[0, 0, 0],
            reward=1.0,
            next_state=dummy_obs,
            done=False
        )
    
    loss = agent.train_step()
    assert loss is not None
    assert isinstance(loss, float)

def test_ppo_agent_predict(dummy_input):
    micro, macro = dummy_input
    agent = DeepScalperPPO(network_config=NET_CONFIG)
    action = agent.predict(micro, macro)
    assert action.shape == (3,)

def test_ensemble_voting(dummy_input):
    micro, macro = dummy_input
    
    dqn = DeepScalperDQN(network_config=NET_CONFIG)
    ppo = DeepScalperPPO(network_config=NET_CONFIG)
    a2c = DeepScalperA2C(network_config=NET_CONFIG)
    gating = SynapseGatingNetwork(input_dim=11)
    
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating)
    
    action = ensemble.predict(micro, macro)
    assert action.shape == (3,)

def test_dqn_get_probs(dummy_input):
    micro, macro = dummy_input
    agent = DeepScalperDQN(network_config=NET_CONFIG)
    pd, pp, pv = agent.get_probs(micro, macro, temp=1.0)
    
    # Check shapes
    assert pd.shape == (1, 3)
    assert pp.shape == (1, 5)
    assert pv.shape == (1, 5)
    
    # Check probability properties
    assert torch.allclose(pd.sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.all(pd >= 0) and torch.all(pd <= 1)

def test_a2c_agent_predict(dummy_input):
    micro, macro = dummy_input
    agent = DeepScalperA2C(network_config=NET_CONFIG)
    action = agent.predict(micro, macro)
    assert action.shape == (3,)

def test_gating_network_output(dummy_input):
    _, macro = dummy_input
    gating = SynapseGatingNetwork(input_dim=11)
    weights = gating(macro)
    assert weights.shape == (1, 3)
    # Assert softmax sum to 1
    assert torch.abs(weights.sum() - 1.0) < 1e-5
