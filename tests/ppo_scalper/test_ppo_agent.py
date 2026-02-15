"""
Unit tests for PPO network, rollout buffer, and agent.

Tests cover:
1. PPOActorCritic shape validation
2. Action masking correctness
3. RolloutBuffer store, GAE, and minibatch iteration
4. PPOAgent predict/train interface

Run: python -m pytest tests/ppo_scalper/test_ppo_agent.py -v
"""
import numpy as np
import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

torch = pytest.importorskip("torch")


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def network_config():
    return {
        "micro_config": {
            "input_size": 27,
            "private_input_size": 3,
            "hidden_size": 64,    # Small for tests
            "rnn_type": "LSTM",
        },
        "macro_config": {
            "input_size": 11,
            "hidden_sizes": [64, 32],
        },
        "fusion_dim": 64,
        "action_space_dims": (5, 9),
    }


@pytest.fixture
def batch_data(network_config):
    """Create batch of observations."""
    B, W = 4, 50
    micro = torch.randn(B, W, 27)
    private = torch.randn(B, W, 3)
    macro = torch.randn(B, 11)
    return micro, private, macro


@pytest.fixture
def ppo_network(network_config):
    from finrl_pro_ds.agents.ppo_scalper.networks import PPOActorCritic
    return PPOActorCritic(**network_config)


# ============================================================================
# TEST: PPOActorCritic NETWORK
# ============================================================================
class TestPPOActorCritic:
    
    def test_forward_shapes(self, ppo_network, batch_data):
        """Test that forward produces correct output shapes."""
        micro, private, macro = batch_data
        B = micro.shape[0]
        
        actions, log_probs, values, entropy, hidden = ppo_network(
            micro, private, macro
        )
        
        assert actions.shape == (B, 2), f"Expected (B, 2), got {actions.shape}"
        assert log_probs.shape == (B,), f"Expected (B,), got {log_probs.shape}"
        assert values.shape == (B,), f"Expected (B,), got {values.shape}"
        assert entropy.shape == (B,), f"Expected (B,), got {entropy.shape}"
        assert isinstance(hidden, tuple) and len(hidden) == 2

    def test_deterministic_mode(self, ppo_network, batch_data):
        """Deterministic mode should produce consistent actions."""
        micro, private, macro = batch_data
        
        actions1, _, _, _, _ = ppo_network(micro, private, macro, deterministic=True)
        actions2, _, _, _, _ = ppo_network(micro, private, macro, deterministic=True)
        
        assert torch.equal(actions1, actions2)

    def test_action_masking(self, ppo_network, batch_data):
        """Masked actions should never be selected."""
        micro, private, macro = batch_data
        B = micro.shape[0]
        
        # Mask: only allow qty index 4 (center = hold)
        qty_mask = torch.zeros(B, 9)
        qty_mask[:, 4] = 1.0
        
        for _ in range(10):
            actions, _, _, _, _ = ppo_network(
                micro, private, macro, qty_mask=qty_mask
            )
            assert (actions[:, 1] == 4).all(), "Masked qty should always be 4"

    def test_evaluate_actions(self, ppo_network, batch_data):
        """evaluate_actions should return matching shapes."""
        micro, private, macro = batch_data
        B = micro.shape[0]
        
        actions = torch.randint(0, 5, (B,)).unsqueeze(1)
        qty_actions = torch.randint(0, 9, (B,)).unsqueeze(1)
        all_actions = torch.cat([actions, qty_actions], dim=1)
        
        log_probs, values, entropy = ppo_network.evaluate_actions(
            micro, private, macro, all_actions
        )
        
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == (B,)

    def test_gradient_flow(self, ppo_network, batch_data):
        """Gradients should flow to all parameters."""
        micro, private, macro = batch_data
        
        actions, log_probs, values, entropy, _ = ppo_network(
            micro, private, macro
        )
        
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        
        for name, param in ppo_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ============================================================================
# TEST: ROLLOUT BUFFER
# ============================================================================
class TestRolloutBuffer:
    
    def test_store_and_gae(self):
        """Test storing data and computing GAE."""
        from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer
        
        T, B = 8, 2
        buf = RolloutBuffer(
            rollout_steps=T, num_envs=B,
            micro_shape=(50, 27), private_shape=(50, 3),
            macro_shape=(11,), n_action_branches=2,
        )
        
        for t in range(T):
            obs = {
                "micro": np.random.randn(B, 50, 27).astype(np.float32),
                "private": np.random.randn(B, 50, 3).astype(np.float32),
                "macro": np.random.randn(B, 11).astype(np.float32),
            }
            buf.store(
                obs=obs,
                actions=np.random.randint(0, 5, (B, 2)),
                log_probs=np.random.randn(B).astype(np.float32),
                rewards=np.random.randn(B).astype(np.float32),
                values=np.random.randn(B).astype(np.float32),
                dones=np.zeros(B, dtype=np.float32),
            )
        
        assert buf.full
        
        # Compute GAE
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        buf.compute_gae(gamma=0.99, gae_lambda=0.95,
                        last_values=last_values, last_dones=last_dones)
        
        assert not np.isnan(buf.advantages).any()
        assert not np.isnan(buf.returns).any()

    def test_minibatch_iteration(self):
        """Test minibatch iterator yields correct shapes."""
        from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer
        
        T, B = 16, 2
        buf = RolloutBuffer(
            rollout_steps=T, num_envs=B,
            micro_shape=(50, 27), private_shape=(50, 3), macro_shape=(11,),
        )
        
        for t in range(T):
            obs = {
                "micro": np.zeros((B, 50, 27), dtype=np.float32),
                "private": np.zeros((B, 50, 3), dtype=np.float32),
                "macro": np.zeros((B, 11), dtype=np.float32),
            }
            buf.store(obs=obs, actions=np.zeros((B, 2), dtype=np.int64),
                      log_probs=np.zeros(B, dtype=np.float32),
                      rewards=np.zeros(B, dtype=np.float32),
                      values=np.zeros(B, dtype=np.float32),
                      dones=np.zeros(B, dtype=np.float32))
        
        buf.compute_gae(0.99, 0.95, np.zeros(B), np.zeros(B))
        
        batch_size = 8
        total_samples = 0
        for batch in buf.iterate_minibatches(batch_size):
            assert batch["micro"].shape[0] <= batch_size
            assert batch["micro"].shape[1:] == (50, 27)
            assert batch["actions"].shape[1] == 2
            total_samples += batch["micro"].shape[0]
        
        assert total_samples == T * B

    def test_reset(self):
        """Test buffer reset."""
        from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer
        
        buf = RolloutBuffer(4, 1, (50, 27), (50, 3), (11,))
        
        for t in range(4):
            obs = {
                "micro": np.zeros((1, 50, 27), dtype=np.float32),
                "private": np.zeros((1, 50, 3), dtype=np.float32),
                "macro": np.zeros((1, 11), dtype=np.float32),
            }
            buf.store(obs=obs, actions=np.zeros((1, 2), dtype=np.int64),
                      log_probs=np.zeros(1, dtype=np.float32),
                      rewards=np.zeros(1, dtype=np.float32),
                      values=np.zeros(1, dtype=np.float32),
                      dones=np.zeros(1, dtype=np.float32))
        
        assert buf.full
        buf.reset()
        assert not buf.full
        assert buf.pos == 0


# ============================================================================
# TEST: PPO AGENT
# ============================================================================
class TestPPOAgent:
    
    def test_predict_interface(self, network_config):
        """Test PPOAgent.predict returns correct types."""
        from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            network_config=network_config,
            lr=3e-4, device="cpu",
        )
        
        B, W = 2, 50
        micro = torch.randn(B, W, 27)
        private = torch.randn(B, W, 3)
        macro = torch.randn(B, 11)
        
        actions, log_probs, values = agent.predict(micro, private, macro)
        
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (B, 2)
        assert actions.dtype == np.int64
        assert isinstance(log_probs, np.ndarray)
        assert log_probs.shape == (B,)
        assert isinstance(values, np.ndarray)
        assert values.shape == (B,)

    def test_hidden_state_management(self, network_config):
        """Test LSTM hidden state reset and masking."""
        from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            network_config=network_config,
            lr=3e-4, device="cpu",
        )
        
        # Initial state should be None
        assert agent._hidden_state is None
        
        # Predict to create hidden state
        micro = torch.randn(1, 50, 27)
        private = torch.randn(1, 50, 3)
        macro = torch.randn(1, 11)
        agent.predict(micro, private, macro)
        assert agent._hidden_state is not None
        
        # Reset
        agent.reset_hidden_state()
        assert agent._hidden_state is None

    def test_save_load(self, network_config, tmp_path):
        """Test checkpoint save/load."""
        from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            network_config=network_config,
            lr=3e-4, device="cpu",
        )
        
        path = str(tmp_path / "test_ckpt.pth")
        agent.save(path)
        assert os.path.exists(path)
        
        # Load into new agent
        agent2 = PPOAgent(
            network_config=network_config,
            lr=3e-4, device="cpu",
        )
        agent2.load(path)
        
        # Check params match
        for (n1, p1), (n2, p2) in zip(
            agent.network.named_parameters(),
            agent2.network.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch at {n1}"
