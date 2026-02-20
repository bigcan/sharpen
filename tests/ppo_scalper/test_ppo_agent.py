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
            "input_size": 30,
            "private_input_size": 5,  # FIX FIND-V3-06: Match Tier 2 env output
            "hidden_size": 64,    # Small for tests
            "encoder_type": "mlp",
            "window_size": 15,
        },
        "macro_config": {
            "input_size": 15,
            "hidden_sizes": [64, 32],
        },
        "fusion_dim": 64,
        "action_space_dims": 6,  # Tier 2: Discrete(6)
    }


@pytest.fixture
def batch_data(network_config):
    """Create batch of observations."""
    B, W = 4, 15
    micro = torch.randn(B, W, 30)
    private = torch.randn(B, W, 5)
    macro = torch.randn(B, 15)
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
        
        actions, log_probs, values, entropy = ppo_network(
            micro, private, macro
        )
        
        assert actions.shape == (B,), f"Expected (B,), got {actions.shape}"
        assert log_probs.shape == (B,), f"Expected (B,), got {log_probs.shape}"
        assert values.shape == (B,), f"Expected (B,), got {values.shape}"
        assert entropy.shape == (B,), f"Expected (B,), got {entropy.shape}"

    def test_deterministic_mode(self, ppo_network, batch_data):
        """Deterministic mode should produce consistent actions."""
        micro, private, macro = batch_data
        ppo_network.eval()  # Disable dropout for deterministic test
        actions1, _, _, _ = ppo_network(micro, private, macro, deterministic=True)
        actions2, _, _, _ = ppo_network(micro, private, macro, deterministic=True)
        ppo_network.train()
        assert torch.equal(actions1, actions2)

    def test_action_masking(self, ppo_network, batch_data):
        """Masked actions should never be selected."""
        micro, private, macro = batch_data
        B = micro.shape[0]
        
        # Mask: only allow action index 3 (center = hold)
        qty_mask = torch.zeros(B, 6)
        qty_mask[:, 3] = 1.0
        
        for _ in range(10):
            actions, _, _, _ = ppo_network(
                micro, private, macro, qty_mask=qty_mask
            )
            assert (actions == 3).all(), "Masked action should always be 3"

    def test_evaluate_actions(self, ppo_network, batch_data):
        """evaluate_actions should return matching shapes."""
        micro, private, macro = batch_data
        B = micro.shape[0]
        
        all_actions = torch.randint(0, 6, (B,))
        
        log_probs, values, entropy = ppo_network.evaluate_actions(
            micro, private, macro, all_actions
        )
        
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == (B,)

    def test_gradient_flow(self, ppo_network, batch_data):
        """Gradients should flow to all parameters."""
        micro, private, macro = batch_data
        
        actions, log_probs, values, _ = ppo_network(
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
            micro_shape=(15, 30), private_shape=(15, 5),
            macro_shape=(15,), n_action_branches=1,
        )
        
        for t in range(T):
            obs = {
                "micro": np.random.randn(B, 15, 30).astype(np.float32),
                "private": np.random.randn(B, 15, 5).astype(np.float32),
                "macro": np.random.randn(B, 15).astype(np.float32),
            }
            buf.store(
                obs=obs,
                actions=np.random.randint(0, 6, (B, 1)),
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
            micro_shape=(15, 30), private_shape=(15, 5), macro_shape=(15,),
        )
        
        for t in range(T):
            obs = {
                "micro": np.zeros((B, 15, 30), dtype=np.float32),
                "private": np.zeros((B, 15, 5), dtype=np.float32),
                "macro": np.zeros((B, 15), dtype=np.float32),
            }
            buf.store(obs=obs, actions=np.zeros((B, 1), dtype=np.int64),
                      log_probs=np.zeros(B, dtype=np.float32),
                      rewards=np.zeros(B, dtype=np.float32),
                      values=np.zeros(B, dtype=np.float32),
                      dones=np.zeros(B, dtype=np.float32))
        
        buf.compute_gae(0.99, 0.95, np.zeros(B), np.zeros(B))
        
        batch_size = 8
        total_samples = 0
        for batch in buf.iterate_minibatches(batch_size):
            assert batch["micro"].shape[0] <= batch_size
            assert batch["micro"].shape[1:] == (15, 30)
            assert batch["actions"].shape[1] == 1
            total_samples += batch["micro"].shape[0]
        
        assert total_samples == T * B

    def test_reset(self):
        """Test buffer reset."""
        from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer
        
        buf = RolloutBuffer(4, 1, (15, 30), (15, 5), (15,))
        
        for t in range(4):
            obs = {
                "micro": np.zeros((1, 15, 30), dtype=np.float32),
                "private": np.zeros((1, 15, 5), dtype=np.float32),
                "macro": np.zeros((1, 15), dtype=np.float32),
            }
            buf.store(obs=obs, actions=np.zeros((1, 1), dtype=np.int64),
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
        
        B, W = 2, 15
        micro = torch.randn(B, W, 30)
        private = torch.randn(B, W, 5)
        macro = torch.randn(B, 15)
        
        actions, log_probs, values = agent.predict(micro, private, macro)
        
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (B,)
        assert actions.dtype == np.int64
        assert isinstance(log_probs, np.ndarray)
        assert log_probs.shape == (B,)
        assert isinstance(values, np.ndarray)
        assert values.shape == (B,)

    def test_stateless_agent(self, network_config):
        """V5: Agent should have no hidden state (stateless MLP)."""
        from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            network_config=network_config,
            lr=3e-4, device="cpu",
        )
        
        # No _hidden_state attribute
        assert not hasattr(agent, '_hidden_state') or agent._hidden_state is None
        
        # Predict should work without hidden state
        micro = torch.randn(1, 15, 30)
        private = torch.randn(1, 15, 5)
        macro = torch.randn(1, 15)
        actions, log_probs, values = agent.predict(micro, private, macro)
        assert actions.shape == (1,)
        
        # reset/mask should be no-ops (no error)
        agent.reset_hidden_state()
        agent.mask_hidden_state(np.zeros(1))

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

    def test_checkpoint_architecture_mismatch(self, network_config, tmp_path):
        """AUDIT FIX C4: V4→V5 checkpoint mismatch should not crash."""
        from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent

        # Save with LSTM encoder (simulates V4 checkpoint)
        lstm_config = dict(network_config)
        lstm_config["micro_config"] = dict(lstm_config["micro_config"])
        lstm_config["micro_config"]["encoder_type"] = "lstm"

        agent_v4 = PPOAgent(
            network_config=lstm_config,
            lr=3e-4, device="cpu",
        )
        path = str(tmp_path / "v4_ckpt.pth")
        agent_v4.save(path)

        # Load V4 checkpoint into V5 MLP agent — should warn, not crash
        agent_v5 = PPOAgent(
            network_config=network_config,  # MLP encoder
            lr=3e-4, device="cpu",
        )
        agent_v5.load(path)  # Should NOT raise RuntimeError

        # Agent should still work with fresh weights
        micro = torch.randn(1, 15, 30)
        private = torch.randn(1, 15, 5)
        macro = torch.randn(1, 15)
        actions, _, _ = agent_v5.predict(micro, private, macro)
        assert actions.shape == (1,)


# ============================================================================
# TEST: ENCODER TYPE REGRESSION (AUDIT FIX E1)
# ============================================================================
class TestEncoderTypeRegression:
    """Verify both MLP and LSTM encoder paths construct and run correctly."""

    @pytest.mark.parametrize("encoder_type", ["mlp", "lstm"])
    def test_encoder_forward(self, encoder_type):
        """Both encoder types should produce correct output shapes."""
        from finrl_pro_ds.agents.ppo_scalper.networks import PPOActorCritic

        config = {
            "micro_config": {
                "input_size": 30,
                "private_input_size": 5,  # FIX FIND-V3-06: Match Tier 2 env output
                "hidden_size": 64,
                "encoder_type": encoder_type,
                "window_size": 15,
            },
            "macro_config": {
                "input_size": 15,
                "hidden_sizes": [64, 32],
            },
            "fusion_dim": 64,
            "action_space_dims": 6,
        }

        net = PPOActorCritic(**config)

        B, W = 2, 15
        micro = torch.randn(B, W, 30)
        private = torch.randn(B, W, 5)
        macro = torch.randn(B, 15)

        actions, log_probs, values, entropy = net(micro, private, macro)
        assert actions.shape == (B,)
        assert log_probs.shape == (B,)
        assert values.shape == (B,)

    @pytest.mark.parametrize("encoder_type", ["mlp", "lstm"])
    def test_encoder_evaluate_actions(self, encoder_type):
        """Both encoder types should work with evaluate_actions."""
        from finrl_pro_ds.agents.ppo_scalper.networks import PPOActorCritic

        config = {
            "micro_config": {
                "input_size": 30,
                "private_input_size": 5,  # FIX FIND-V3-06: Match Tier 2 env output
                "hidden_size": 64,
                "encoder_type": encoder_type,
                "window_size": 15,
            },
            "macro_config": {
                "input_size": 15,
                "hidden_sizes": [64, 32],
            },
            "fusion_dim": 64,
            "action_space_dims": 6,  # Tier 2: Discrete(6)
        }

        net = PPOActorCritic(**config)

        B = 2
        micro = torch.randn(B, 15, 30)
        private = torch.randn(B, 15, 5)
        macro = torch.randn(B, 15)
        actions = torch.randint(0, 6, (B,))

        log_probs, values, entropy = net.evaluate_actions(
            micro, private, macro, actions
        )
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == (B,)

