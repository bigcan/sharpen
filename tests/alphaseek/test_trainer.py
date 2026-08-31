"""Tests for AlphaSeek Phase 6 training infrastructure.

Tests replay buffer, agent classes, and trainer using small configs.
Skips if LOB parquet data is not available.
"""

import os

import pytest
import torch

# Skip if LOB parquet not available
LOB_PARQUET = "data/lob_parquet/btcusdt_lob_1s.parquet"
HAS_LOB_DATA = os.path.exists(LOB_PARQUET)

skip_no_data = pytest.mark.skipif(
    not HAS_LOB_DATA,
    reason=f"LOB parquet not found: {LOB_PARQUET}",
)


# ===========================================================================
# Replay Buffer Tests (no data dependency)
# ===========================================================================

class TestAlphaSeekReplayBuffer:
    """Test GPU replay buffer round-trip and shapes."""

    def _make_buffer(self, max_size=100, num_seqs=4, state_dim=10, action_dim=1):
        from sharpen.alphaseek.replay_buffer import AlphaSeekReplayBuffer

        return AlphaSeekReplayBuffer(
            max_size=max_size,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=-1,  # CPU for tests
            num_seqs=num_seqs,
        )

    def test_update_and_sample(self):
        buf = self._make_buffer()
        horizon_len = 10
        num_seqs = 4

        states = torch.randn(horizon_len, num_seqs, 10)
        actions = torch.randint(0, 3, (horizon_len, num_seqs, 1)).float()
        rewards = torch.randn(horizon_len, num_seqs)
        undones = torch.ones(horizon_len, num_seqs)

        buf.update((states, actions, rewards, undones))
        assert len(buf) == horizon_len

        s, a, r, u, ns = buf.sample(batch_size=8)
        assert s.shape == (8, 10)
        assert a.shape == (8, 1)
        assert r.shape == (8,)
        assert u.shape == (8,)
        assert ns.shape == (8, 10)

    def test_wraparound(self):
        buf = self._make_buffer(max_size=20, num_seqs=2)

        for _ in range(3):
            states = torch.randn(10, 2, 10)
            actions = torch.zeros(10, 2, 1)
            rewards = torch.ones(10, 2)
            undones = torch.ones(10, 2)
            buf.update((states, actions, rewards, undones))

        # After 30 steps with max_size=20, buffer should be full
        assert buf.if_full
        assert len(buf) == 20

    def test_save_load(self, tmp_path):
        buf = self._make_buffer(max_size=50, num_seqs=2)

        states = torch.randn(10, 2, 10)
        actions = torch.zeros(10, 2, 1)
        rewards = torch.ones(10, 2)
        undones = torch.ones(10, 2)
        buf.update((states, actions, rewards, undones))

        save_dir = str(tmp_path / "buf")
        buf.save(save_dir)

        buf2 = self._make_buffer(max_size=50, num_seqs=2)
        buf2.load(save_dir)
        assert len(buf2) == len(buf)


# ===========================================================================
# Agent Config Tests (no data dependency)
# ===========================================================================

class TestAlphaSeekAgentConfig:
    def test_from_dict(self):
        from sharpen.alphaseek.agents import AlphaSeekAgentConfig

        cfg = AlphaSeekAgentConfig.from_dict({
            "learning_rate": 1e-5,
            "net_dims": "128,128,128",
            "gamma": 0.99,
        })
        assert cfg.learning_rate == 1e-5
        assert cfg.net_dims == (128, 128, 128)
        assert cfg.gamma == 0.99

    def test_from_dict_list_net_dims(self):
        from sharpen.alphaseek.agents import AlphaSeekAgentConfig

        cfg = AlphaSeekAgentConfig.from_dict({"net_dims": [256, 256]})
        assert cfg.net_dims == (256, 256)

    def test_defaults(self):
        from sharpen.alphaseek.agents import AlphaSeekAgentConfig

        cfg = AlphaSeekAgentConfig()
        assert cfg.state_dim == 10
        assert cfg.action_dim == 3
        assert cfg.if_off_policy is True


# ===========================================================================
# Agent Tests (no data dependency — synthetic env)
# ===========================================================================

class TestAlphaSeekAgents:
    """Test agent construction and forward pass with synthetic data."""

    def _make_agent(self, agent_cls_name="DoubleDQN", num_envs=4):
        from sharpen.alphaseek.agents import AGENT_MAP, AlphaSeekAgentConfig

        agent_class = AGENT_MAP[agent_cls_name]
        cfg = AlphaSeekAgentConfig(
            num_envs=num_envs,
            net_dims=(32, 32),  # tiny for speed
            batch_size=8,
            learning_rate=1e-3,
        )
        agent = agent_class(
            net_dims=cfg.net_dims,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            gpu_id=-1,
            args=cfg,
        )
        return agent, cfg

    @pytest.mark.parametrize("agent_name", ["D3QN", "DoubleDQN", "TwinD3QN"])
    def test_agent_construction(self, agent_name):
        agent, _ = self._make_agent(agent_name)
        assert agent.act is not None
        assert agent.cri_target is not None
        assert agent.state_dim == 10
        assert agent.action_dim == 3

    @pytest.mark.parametrize("agent_name", ["D3QN", "DoubleDQN", "TwinD3QN"])
    def test_get_action(self, agent_name):
        agent, _ = self._make_agent(agent_name, num_envs=4)
        state = torch.randn(4, 10)
        action = agent.act.get_action(state)
        assert action.shape == (4, 1)
        assert action.dtype == torch.int64 or action.dtype == torch.long

    @pytest.mark.parametrize("agent_name", ["D3QN", "DoubleDQN", "TwinD3QN"])
    def test_get_q1_q2(self, agent_name):
        agent, _ = self._make_agent(agent_name, num_envs=4)
        state = torch.randn(4, 10)
        q1, q2 = agent.act.get_q1_q2(state)
        assert q1.shape == (4, 3)
        assert q2.shape == (4, 3)

    def test_soft_update(self):
        agent, _ = self._make_agent("DoubleDQN")
        # Store original target params
        orig_params = [p.clone() for p in agent.cri_target.parameters()]

        # Modify current network
        with torch.no_grad():
            for p in agent.act.parameters():
                p.add_(torch.randn_like(p))

        agent._soft_update(agent.cri_target, agent.act, 0.5)

        # Target should have moved
        for orig, new in zip(orig_params, agent.cri_target.parameters()):
            assert not torch.allclose(orig, new.data)

    def test_save_load_agent(self, tmp_path):
        agent, _ = self._make_agent("D3QN")
        save_dir = str(tmp_path / "agent")
        agent.save_agent(save_dir)

        agent2, _ = self._make_agent("D3QN")
        agent2.load_agent(save_dir)

        # Verify parameters match
        for p1, p2 in zip(agent.act.parameters(), agent2.act.parameters()):
            assert torch.allclose(p1, p2)


# ===========================================================================
# Trainer Tests (requires LOB data)
# ===========================================================================

@skip_no_data
class TestAlphaSeekTrainer:
    """Integration tests with real LOB data. Small configs for speed."""

    def _make_trainer(self, agent_name="DoubleDQN", num_sims=4):
        from sharpen.alphaseek.agents import AGENT_MAP
        from sharpen.alphaseek.lob_trade_simulator import (
            EvalLOBTradeSimulator,
            LOBTradeSimulator,
        )
        from sharpen.alphaseek.trainer import AlphaSeekTrainer

        agent_class = AGENT_MAP[agent_name]
        train_sim = LOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET,
            num_sims=num_sims,
            step_gap=2,
            gpu_id=-1,
        )
        eval_sim = EvalLOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET,
            num_sims=num_sims,
            step_gap=2,
            gpu_id=-1,
        )
        config = {
            "net_dims": [32, 32],
            "batch_size": 8,
            "learning_rate": 1e-3,
            "reward_scale": 100,
            "repeat_times": 1,
        }
        trainer = AlphaSeekTrainer(
            agent_class=agent_class,
            train_sim=train_sim,
            eval_sim=eval_sim,
            config=config,
            gpu_id=-1,
        )
        return trainer

    def test_smoke_train(self):
        trainer = self._make_trainer(num_sims=4)
        metrics = trainer.train(break_step=100)
        assert "total_return" in metrics
        assert "sharpe" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "hold_rate" in metrics

    def test_evaluation_metrics(self):
        trainer = self._make_trainer(num_sims=4)
        trainer.train(break_step=50)
        metrics = trainer.evaluate()
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["n_steps"], int)
        assert metrics["n_steps"] > 0

    def test_checkpoint_roundtrip(self, tmp_path):
        trainer = self._make_trainer(num_sims=4)
        trainer.train(break_step=50)

        save_path = str(tmp_path / "ckpt")
        trainer.save_checkpoint(save_path)

        trainer2 = self._make_trainer(num_sims=4)
        trainer2.load_checkpoint(save_path)

        # Both agents should produce same Q-values
        state = torch.randn(4, 12)  # v3 state_dim
        q1_orig = trainer.agent.act(state)
        q1_loaded = trainer2.agent.act(state)
        assert torch.allclose(q1_orig, q1_loaded, atol=1e-5)


# ===========================================================================
# Segment Filter Tests (requires LOB data)
# ===========================================================================

@skip_no_data
class TestSegmentFilter:
    """Test LOBTradeSimulator segment_filter and get_segment_info."""

    def test_get_segment_info(self):
        from sharpen.alphaseek.lob_trade_simulator import LOBTradeSimulator

        info = LOBTradeSimulator.get_segment_info(LOB_PARQUET)
        assert "segment_id" in info.columns
        assert "n_rows" in info.columns
        assert "duration_hours" in info.columns
        assert len(info) > 0

    def test_segment_filter_reduces_data(self):
        from sharpen.alphaseek.lob_trade_simulator import LOBTradeSimulator

        info = LOBTradeSimulator.get_segment_info(LOB_PARQUET)
        all_segs = info["segment_id"].tolist()

        if len(all_segs) < 2:
            pytest.skip("Need at least 2 segments to test filtering")

        sim_full = LOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET, num_sims=4, gpu_id=-1,
        )
        sim_filtered = LOBTradeSimulator(
            lob_parquet_path=LOB_PARQUET, num_sims=4, gpu_id=-1,
            segment_filter=[all_segs[0]],
        )
        assert sim_filtered.full_seq_len < sim_full.full_seq_len
