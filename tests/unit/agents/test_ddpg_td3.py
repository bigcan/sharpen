"""Unit tests for DDPG and TD3 Agents."""

import numpy as np
import pytest
import torch

from finrl_pro_ds.agents.ddpg import DDPGAgent
from finrl_pro_ds.agents.td3 import TD3Agent


@pytest.fixture
def ddpg_agent():
    return DDPGAgent(state_dim=10, action_dim=2, batch_size=4)


@pytest.fixture
def td3_agent():
    return TD3Agent(state_dim=10, action_dim=2, batch_size=4)


def test_ddpg_initialization(ddpg_agent):
    assert ddpg_agent.state_dim == 10
    assert ddpg_agent.action_dim == 2
    assert isinstance(ddpg_agent.actor, torch.nn.Module)
    assert isinstance(ddpg_agent.critic, torch.nn.Module)


def test_td3_initialization(td3_agent):
    assert td3_agent.state_dim == 10
    assert td3_agent.action_dim == 2
    assert isinstance(td3_agent.actor, torch.nn.Module)
    assert isinstance(td3_agent.critic, torch.nn.Module)


def test_ddpg_select_action(ddpg_agent):
    state = np.random.random(10)
    action = ddpg_agent.select_action(state)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_td3_select_action(td3_agent):
    state = np.random.random(10)
    action = td3_agent.select_action(state)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_ddpg_update(ddpg_agent):
    for _ in range(10):
        state = np.random.random(10)
        action = np.random.random(2)
        next_state = np.random.random(10)
        ddpg_agent.store_transition(state, action, next_state, 1.0, False)
        
    loss_info = ddpg_agent.update()
    assert "critic_loss" in loss_info
    assert "actor_loss" in loss_info


def test_td3_update(td3_agent):
    for _ in range(10):
        state = np.random.random(10)
        action = np.random.random(2)
        next_state = np.random.random(10)
        td3_agent.store_transition(state, action, next_state, 1.0, False)
        
    loss_info = td3_agent.update()
    assert "critic_loss" in loss_info
    # actor_loss might be 0 if skipped due to policy_freq, but key should exist
    assert "actor_loss" in loss_info
