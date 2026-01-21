"""Unit tests for SAC Agent."""

import numpy as np
import pytest
import torch

from finrl_pro_ds.agents.sac import SACAgent


@pytest.fixture
def sac_agent():
    return SACAgent(state_dim=10, action_dim=2, batch_size=4)


def test_sac_initialization(sac_agent):
    assert sac_agent.state_dim == 10
    assert sac_agent.action_dim == 2
    assert isinstance(sac_agent.actor, torch.nn.Module)
    assert isinstance(sac_agent.critic, torch.nn.Module)


def test_select_action(sac_agent):
    state = np.random.random(10)
    action = sac_agent.select_action(state)
    
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_store_transition(sac_agent):
    state = np.random.random(10)
    action = np.random.random(2)
    next_state = np.random.random(10)
    sac_agent.store_transition(state, action, next_state, 1.0, False)
    
    assert sac_agent.replay_buffer.size == 1


def test_update(sac_agent):
    # Fill buffer with dummy data
    for _ in range(10):
        state = np.random.random(10)
        action = np.random.random(2)
        next_state = np.random.random(10)
        sac_agent.store_transition(state, action, next_state, 1.0, False)
        
    loss_info = sac_agent.update()
    
    assert "critic_loss" in loss_info
    assert "actor_loss" in loss_info
    assert "alpha_loss" in loss_info
