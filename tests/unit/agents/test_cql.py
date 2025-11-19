"""Unit tests for CQL Agent."""

import numpy as np
import pytest
import torch

from finrl_pro.agents.cql import CQLAgent


@pytest.fixture
def cql_agent():
    return CQLAgent(state_dim=10, action_dim=2, batch_size=4)


def test_cql_initialization(cql_agent):
    assert cql_agent.state_dim == 10
    assert cql_agent.action_dim == 2
    assert cql_agent.cql_weight == 1.0
    assert isinstance(cql_agent.actor, torch.nn.Module)


def test_select_action(cql_agent):
    state = np.random.random(10)
    action = cql_agent.select_action(state)
    
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_update_cql_loss(cql_agent):
    # Fill buffer with dummy data
    for _ in range(10):
        state = np.random.random(10)
        action = np.random.random(2)
        next_state = np.random.random(10)
        cql_agent.store_transition(state, action, next_state, 1.0, False)
        
    loss_info = cql_agent.update()
    
    assert "critic_loss" in loss_info
    assert "cql_loss" in loss_info
    assert "actor_loss" in loss_info
    
    # CQL loss should be non-zero (or at least computed)
    # Note: It can be negative or positive depending on Q-values.
    assert isinstance(loss_info["cql_loss"], float)
