"""Tests for AlphaSeekAgent and AlphaSeekEnsemble."""

import os
import tempfile

import pytest
import torch

from sharpen.alphaseek.agent_wrapper import ACTION_DIM, STATE_DIM, AlphaSeekAgent
from sharpen.alphaseek.ensemble import AlphaSeekEnsemble, EnsembleStrategy
from sharpen.alphaseek.nets import QNetTwin, QNetTwinDuel

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory with saved weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a QNetTwinDuel (D3QN) model
        net = QNetTwinDuel(dims=[128, 128, 128], state_dim=STATE_DIM, action_dim=ACTION_DIM)
        torch.save(net, os.path.join(tmpdir, "act.pth"))
        yield tmpdir


@pytest.fixture
def temp_twin_checkpoint_dir():
    """Create a temporary checkpoint directory with QNetTwin weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = QNetTwin(dims=[128, 128, 128], state_dim=STATE_DIM, action_dim=ACTION_DIM)
        torch.save(net, os.path.join(tmpdir, "act.pth"))
        yield tmpdir


@pytest.fixture
def random_state():
    """Random state tensor matching AlphaSeek state contract."""
    return torch.randn(1, STATE_DIM)


# ─── AlphaSeekAgent Tests ────────────────────────────────────────────────────


class TestAlphaSeekAgent:
    def test_create_d3qn(self):
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        assert agent.agent_type == "D3QN"
        assert not agent.is_loaded

    def test_create_double_dqn(self):
        agent = AlphaSeekAgent(agent_type="DoubleDQN")
        assert agent.agent_type == "DoubleDQN"

    def test_create_twin_d3qn(self):
        agent = AlphaSeekAgent(agent_type="TwinD3QN")
        assert agent.agent_type == "TwinD3QN"

    def test_invalid_agent_type(self):
        with pytest.raises(ValueError, match="Unknown agent_type"):
            AlphaSeekAgent(agent_type="InvalidAgent")

    def test_load_checkpoint(self, temp_checkpoint_dir):
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        agent.load(temp_checkpoint_dir)
        assert agent.is_loaded

    def test_load_missing_checkpoint(self, tmp_path):
        agent = AlphaSeekAgent(agent_type="D3QN")
        with pytest.raises(FileNotFoundError):
            agent.load(str(tmp_path))

    def test_predict_without_load(self, random_state):
        agent = AlphaSeekAgent(agent_type="D3QN")
        with pytest.raises(RuntimeError, match="not loaded"):
            agent.predict(random_state)

    def test_q_values_shape(self, temp_checkpoint_dir, random_state):
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        agent.load(temp_checkpoint_dir)
        q_vals = agent.q_values(random_state)
        assert q_vals.shape == (1, ACTION_DIM)

    def test_predict_returns_valid_action(self, temp_checkpoint_dir, random_state):
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        agent.load(temp_checkpoint_dir)
        action = agent.predict(random_state)
        assert action in (0, 1, 2)

    def test_deterministic_prediction(self, temp_checkpoint_dir, random_state):
        """Same input should always give same output (explore_rate=0)."""
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        agent.load(temp_checkpoint_dir)
        action1 = agent.predict(random_state)
        action2 = agent.predict(random_state)
        assert action1 == action2

    def test_1d_state_input(self, temp_checkpoint_dir):
        """Agent should handle 1D state (auto unsqueeze)."""
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(128, 128, 128))
        agent.load(temp_checkpoint_dir)
        state_1d = torch.randn(STATE_DIM)
        q_vals = agent.q_values(state_1d)
        assert q_vals.shape == (1, ACTION_DIM)

    def test_repr(self, temp_checkpoint_dir):
        agent = AlphaSeekAgent(agent_type="D3QN")
        assert "unloaded" in repr(agent)
        agent.load(temp_checkpoint_dir)
        assert "loaded" in repr(agent)


# ─── AlphaSeekEnsemble Tests ─────────────────────────────────────────────────


def _make_loaded_agents(checkpoint_dirs: list[tuple[str, str]], net_dims=(128, 128, 128)):
    agents = []
    for agent_type, ckpt_dir in checkpoint_dirs:
        agent = AlphaSeekAgent(agent_type=agent_type, net_dims=net_dims)
        agent.load(ckpt_dir)
        agents.append(agent)
    return agents


class TestAlphaSeekEnsemble:
    @pytest.fixture
    def three_agent_dirs(self, tmp_path):
        """Create 3 checkpoint dirs with different agent types."""
        dirs = []
        for agent_type, net_cls in [
            ("D3QN", QNetTwinDuel),
            ("DoubleDQN", QNetTwin),
            ("TwinD3QN", QNetTwin),
        ]:
            d = tmp_path / agent_type
            d.mkdir()
            net = net_cls(dims=[128, 128, 128], state_dim=STATE_DIM, action_dim=ACTION_DIM)
            torch.save(net, str(d / "act.pth"))
            dirs.append((agent_type, str(d)))
        return dirs

    def test_create_ensemble(self, three_agent_dirs):
        agents = _make_loaded_agents(three_agent_dirs)
        ensemble = AlphaSeekEnsemble(agents)
        assert ensemble.n_agents == 3

    def test_empty_ensemble_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            AlphaSeekEnsemble([])

    def test_unloaded_agent_raises(self):
        agent = AlphaSeekAgent(agent_type="D3QN")
        with pytest.raises(RuntimeError, match="not loaded"):
            AlphaSeekEnsemble([agent])

    def test_predict_majority_vote(self, three_agent_dirs, random_state):
        agents = _make_loaded_agents(three_agent_dirs)
        ensemble = AlphaSeekEnsemble(agents, strategy=EnsembleStrategy.MAJORITY_VOTE)
        action_int, metadata = ensemble.predict(random_state)
        assert action_int in (-1, 0, 1)
        assert "q_values" in metadata
        assert "raw_actions" in metadata
        assert len(metadata["raw_actions"]) == 3

    def test_predict_q_average(self, three_agent_dirs, random_state):
        agents = _make_loaded_agents(three_agent_dirs)
        ensemble = AlphaSeekEnsemble(agents, strategy=EnsembleStrategy.Q_AVERAGE)
        action_int, metadata = ensemble.predict(random_state)
        assert action_int in (-1, 0, 1)

    def test_predict_weighted(self, three_agent_dirs, random_state):
        agents = _make_loaded_agents(three_agent_dirs)
        weights = [0.5, 0.3, 0.2]
        ensemble = AlphaSeekEnsemble(
            agents, strategy=EnsembleStrategy.WEIGHTED, agent_weights=weights,
        )
        action_int, metadata = ensemble.predict(random_state)
        assert action_int in (-1, 0, 1)

    def test_predict_confidence_gated(self, three_agent_dirs, random_state):
        agents = _make_loaded_agents(three_agent_dirs)
        ensemble = AlphaSeekEnsemble(
            agents,
            strategy=EnsembleStrategy.CONFIDENCE_GATED,
            confidence_threshold=0.001,
        )
        action_int, metadata = ensemble.predict(random_state)
        assert action_int in (-1, 0, 1)

    def test_wrong_weights_length(self, three_agent_dirs):
        agents = _make_loaded_agents(three_agent_dirs)
        with pytest.raises(ValueError, match="agent_weights length"):
            AlphaSeekEnsemble(agents, strategy=EnsembleStrategy.WEIGHTED, agent_weights=[0.5, 0.5])

    def test_metadata_keys(self, three_agent_dirs, random_state):
        agents = _make_loaded_agents(three_agent_dirs)
        ensemble = AlphaSeekEnsemble(agents)
        _, metadata = ensemble.predict(random_state)
        assert "q_values" in metadata
        assert "q_means" in metadata
        assert "raw_actions" in metadata
        assert "confidence" in metadata
        assert "agents_agree" in metadata
        assert "strategy" in metadata
