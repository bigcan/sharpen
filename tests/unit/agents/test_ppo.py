"""Unit tests for PPO Agent."""

import numpy as np
import pytest
import torch

from finrl_pro_ds.agents.ppo import PPOAgent


@pytest.fixture
def ppo_agent():
    return PPOAgent(state_dim=10, action_dim=2, batch_size=4)


def test_ppo_initialization(ppo_agent):
    assert ppo_agent.state_dim == 10
    assert ppo_agent.action_dim == 2
    assert isinstance(ppo_agent.policy, torch.nn.Module)


def test_select_action(ppo_agent):
    state = np.random.random(10)
    action, log_prob = ppo_agent.select_action(state)
    
    assert action.shape == (2,)
    assert isinstance(log_prob, float)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_store_transition(ppo_agent):
    state = np.random.random(10)
    action = np.random.random(2)
    ppo_agent.store_transition(state, action, 1.0, False, -0.5)
    
    assert len(ppo_agent.buffer["states"]) == 1
    assert len(ppo_agent.buffer["actions"]) == 1


def test_update(ppo_agent):
    # Fill buffer with dummy data
    for _ in range(10):
        state = np.random.random(10)
        action = np.random.random(2)
        ppo_agent.store_transition(state, action, 1.0, False, -0.5)
        
    loss_info = ppo_agent.update()
    
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert "entropy" in loss_info
    
    # Buffer should not be cleared automatically by update in this implementation
    # (It's usually cleared by the runner)
    assert len(ppo_agent.buffer["states"]) == 10
