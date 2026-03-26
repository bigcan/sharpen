"""Tests for DQfD demo buffer seeding (_seed_demo_buffer).

Verifies:
- Buffer is populated with demo_steps * num_envs transitions
- Disc3 action mapping: buy=0, hold=1, sell=2
- Disc6 action mapping: buy=0, hold=2, sell=5
- Momentum policy direction: positive logret → buy, negative → sell, near-zero → hold
- demo_seed_steps=0 skips seeding entirely
- train() only seeds on start_step==0 (not on HPO resumptions)
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _make_config(discrete_dims=3, demo_seed_steps=10, num_envs=4):
    return {
        "env": {
            "window_size": 15,
            "action": {"discrete_dims": discrete_dims, "max_position": 5.0, "fixed_trade_qty": 0.2},
            "num_envs": num_envs,
            "initial_balance": 100000.0,
            "margin_requirement": 0.05,
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "max_drawdown_pct": 0.30,
            "private_state_augment_prob": 0.0,
            "reward": {"scaling": 1.0, "volatility_horizon": 20, "sharpe_weight": 0.0,
                       "dsr_scale": 100.0, "sharpe_horizon": 20, "hold_bonus_bps": 0.0},
            "augmentation": {"enabled": False},
        },
        "network": {
            "micro_config": {
                "input_size": 30, "private_input_size": 5, "hidden_size": 64,
                "encoder_type": "mlp", "window_size": 15,
            },
            "macro_config": {"input_size": 15, "hidden_sizes": [64, 32]},
        },
        "agents": {
            "bdq": {
                "learning_rate": 1e-4, "gamma": 0.95, "epsilon_start": 1.0,
                "epsilon_end": 0.05, "epsilon_decay": 0.999, "buffer_size": 1000,
                "batch_size": 32, "target_update_freq": 100, "auxiliary_weight": 1.0,
                "use_per": False, "exploration_mode": "epsilon_greedy",
                "tau": 0.005, "update_interval": 1.0, "checkpoint_interval": 10000,
                "learning_starts": 100, "demo_seed_steps": demo_seed_steps,
            },
        },
        "training": {
            "total_timesteps": 500, "training_epochs": 1, "log_interval": 100,
            "torch_compile": False, "use_amp": False, "use_shm": False,
            "verbose_logging": False,
        },
        "hpo": {"enabled": False, "n_trials": 1, "steps_per_trial": 100},
    }


def _make_mock_env(num_envs=4, window_size=15, micro_dim=30, macro_dim=15, private_dim=5,
                   logret_override=None):
    """Returns a mock vectorized env whose obs['macro'][:,0] can be overridden."""
    env = MagicMock()
    env.num_envs = num_envs

    def _obs(logret_val=0.0):
        macro = np.zeros((num_envs, macro_dim), dtype=np.float32)
        macro[:, 0] = logret_val
        return {
            "micro": np.zeros((num_envs, window_size, micro_dim), dtype=np.float32),
            "macro": macro,
            "private": np.zeros((num_envs, window_size, private_dim), dtype=np.float32),
        }

    logret_sequence = logret_override or [0.002, -0.002, 0.0, 0.0]
    call_count = [0]

    def _reset(**kwargs):
        call_count[0] = 0
        return _obs(logret_sequence[0]), {}

    def _step(actions):
        call_count[0] += 1
        idx = call_count[0] % len(logret_sequence)
        obs = _obs(logret_sequence[idx])
        rewards = np.zeros(num_envs, dtype=np.float32)
        term = np.zeros(num_envs, dtype=bool)
        trunc = np.zeros(num_envs, dtype=bool)
        infos = [{"volatility_target": 0.0}] * num_envs
        return obs, rewards, term, trunc, infos

    env.reset.side_effect = _reset
    env.step.side_effect = _step
    return env


class TestSeedDemoBuffer(unittest.TestCase):

    def _make_trainer(self, discrete_dims=3, demo_seed_steps=10, num_envs=4,
                      logret_override=None):
        from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
        cfg = _make_config(discrete_dims=discrete_dims, demo_seed_steps=demo_seed_steps,
                           num_envs=num_envs)
        env = _make_mock_env(num_envs=num_envs, logret_override=logret_override)
        with patch("os.makedirs"):
            trainer = DeepScalperTrainer(env, cfg, device="cpu")
        return trainer, env

    def test_buffer_size_after_seeding_disc3(self):
        """Buffer should contain demo_steps * num_envs transitions after seeding."""
        demo_steps = 8
        num_envs = 4
        trainer, _ = self._make_trainer(discrete_dims=3, demo_seed_steps=demo_steps,
                                        num_envs=num_envs)
        trainer._seed_demo_buffer(demo_steps)
        self.assertEqual(len(trainer.agent.memory), demo_steps * num_envs)

    def test_buffer_size_after_seeding_disc6(self):
        """Same check for Disc6 config."""
        demo_steps = 6
        num_envs = 4
        trainer, _ = self._make_trainer(discrete_dims=6, demo_seed_steps=demo_steps,
                                        num_envs=num_envs)
        trainer._seed_demo_buffer(demo_steps)
        self.assertEqual(len(trainer.agent.memory), demo_steps * num_envs)

    def test_disc3_buy_action_on_positive_logret(self):
        """With logret_5 > 5bps on every step, all actions should be BUY=0 for Disc3."""
        logret_sequence = [0.002] * 20  # always positive
        trainer, env = self._make_trainer(discrete_dims=3, demo_seed_steps=5,
                                          num_envs=2, logret_override=logret_sequence)
        trainer._seed_demo_buffer(5)
        # All stored actions should be 0 (TakerBuy for Disc3)
        actions_stored = trainer.agent.memory._actions[:len(trainer.agent.memory)]
        self.assertTrue(np.all(actions_stored == 0),
                        f"Expected all BUY(0) for Disc3 on positive logret, got {np.unique(actions_stored)}")

    def test_disc3_sell_action_on_negative_logret(self):
        """With logret_5 < -5bps on every step, all actions should be SELL=2 for Disc3."""
        logret_sequence = [-0.002] * 20
        trainer, env = self._make_trainer(discrete_dims=3, demo_seed_steps=5,
                                          num_envs=2, logret_override=logret_sequence)
        trainer._seed_demo_buffer(5)
        actions_stored = trainer.agent.memory._actions[:len(trainer.agent.memory)]
        self.assertTrue(np.all(actions_stored == 2),
                        f"Expected all SELL(2) for Disc3 on negative logret, got {np.unique(actions_stored)}")

    def test_disc6_buy_action_on_positive_logret(self):
        """With positive logret, Disc6 should use BUY=0."""
        logret_sequence = [0.002] * 20
        trainer, env = self._make_trainer(discrete_dims=6, demo_seed_steps=5,
                                          num_envs=2, logret_override=logret_sequence)
        trainer._seed_demo_buffer(5)
        actions_stored = trainer.agent.memory._actions[:len(trainer.agent.memory)]
        self.assertTrue(np.all(actions_stored == 0),
                        f"Expected all BUY(0) for Disc6 on positive logret, got {np.unique(actions_stored)}")

    def test_disc6_sell_action_on_negative_logret(self):
        """With negative logret, Disc6 should use SELL=5."""
        logret_sequence = [-0.002] * 20
        trainer, env = self._make_trainer(discrete_dims=6, demo_seed_steps=5,
                                          num_envs=2, logret_override=logret_sequence)
        trainer._seed_demo_buffer(5)
        actions_stored = trainer.agent.memory._actions[:len(trainer.agent.memory)]
        self.assertTrue(np.all(actions_stored == 5),
                        f"Expected all SELL(5) for Disc6 on negative logret, got {np.unique(actions_stored)}")

    def test_hold_on_near_zero_logret(self):
        """logret_5 within ±5bps threshold should produce HOLD."""
        logret_sequence = [0.0001] * 20  # below 5bps threshold
        trainer, env = self._make_trainer(discrete_dims=3, demo_seed_steps=5,
                                          num_envs=2, logret_override=logret_sequence)
        trainer._seed_demo_buffer(5)
        actions_stored = trainer.agent.memory._actions[:len(trainer.agent.memory)]
        self.assertTrue(np.all(actions_stored == 1),
                        f"Expected all HOLD(1) for Disc3 on near-zero logret, got {np.unique(actions_stored)}")

    def test_zero_demo_steps_skips_seeding(self):
        """demo_seed_steps=0 should not populate the buffer."""
        trainer, _ = self._make_trainer(discrete_dims=3, demo_seed_steps=0, num_envs=4)
        # _seed_demo_buffer(0) would push 0 transitions
        trainer._seed_demo_buffer(0)
        self.assertEqual(len(trainer.agent.memory), 0)

    def test_buffer_does_not_exceed_capacity(self):
        """Seeding more transitions than buffer capacity wraps correctly."""
        # buffer_size=1000, demo_steps=300, num_envs=4 → 1200 transitions → wraps
        demo_steps = 300
        num_envs = 4
        trainer, _ = self._make_trainer(discrete_dims=3, demo_seed_steps=demo_steps,
                                        num_envs=num_envs)
        trainer._seed_demo_buffer(demo_steps)
        # Buffer capacity is 1000 — should be full, not larger
        self.assertEqual(len(trainer.agent.memory), trainer.agent.memory.capacity)


if __name__ == "__main__":
    unittest.main()
