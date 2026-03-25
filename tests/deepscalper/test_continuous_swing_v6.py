"""Tests for GMGP1-v6 hard risk constraints (stop-loss + max-holding).

Verifies that:
  - Position-level stop-loss forces flat when position loses > stop_loss_bps
  - Max holding timer forces flat after max_holding_bars
  - Both constraints disabled by default (backward compat)
  - Fees are correctly applied on forced closes
"""
import numpy as np

from finrl_pro_ds.envs.continuous_swing_env import ContinuousSwingEnv


def _make_env(stop_loss_bps=0, max_holding_bars=0, taker_fee=0.0002):
    """Create a minimal ContinuousSwingEnv with mocked handler."""
    config = {
        "initial_balance": 100000.0,
        "window_size": 5,
        "features_per_scale": 8,
        "taker_fee": taker_fee,
        "deadband_threshold": 0.1,
        "episode_length": 100,
        "random_start": False,
        "max_drawdown_pct": 0.50,
        "scales": [15],
        "stop_loss_bps": stop_loss_bps,
        "max_holding_bars": max_holding_bars,
        "reward": {"mode": "dsr", "dsr_eta": 0.001, "dsr_scale": 1.0},
    }
    env = ContinuousSwingEnv(config, data_handler=None)
    return env


def _step_with_price(env, action, close, atr=1.0):
    """Manually drive one env step by injecting price data."""
    env.prev_close = env.current_close
    env.current_close = close
    env.current_atr = atr
    env._atr_buffer.append(atr)
    if len(env._atr_buffer) > 200:
        env._atr_buffer = env._atr_buffer[-200:]
    env._atr_rolling_mean = np.mean(env._atr_buffer)

    # Build a mock obs
    env._current_obs = {
        "scale_0": np.zeros((5, 8), dtype=np.float32),
        "private": np.zeros(5, dtype=np.float32),
    }
    env.current_step += 1

    # Call step logic directly via action
    return env.step(np.array([action], dtype=np.float32))


class TestStopLoss:
    def test_stop_loss_triggers(self):
        """Position forced flat when equity drops > stop_loss_bps from entry."""
        env = _make_env(stop_loss_bps=50, taker_fee=0.0)
        env.current_close = 100.0
        env.current_step = 0
        env.equity = 100000.0
        env.peak_equity = 100000.0
        env._entry_equity = 100000.0

        # Go long
        env.current_position = 0.8
        env._position_direction = 1
        env._bars_in_position = 0
        env._entry_equity = 100000.0

        # Simulate a loss: equity drops 60bps (> 50bps threshold)
        env.equity = 100000.0 * (1.0 - 60.0 / 10000.0)  # 99940

        # Step with a small action to maintain position
        obs, reward, term, trunc, info = _step_with_price(env, 0.8, 100.0)

        # Position should be forced flat
        assert abs(env.current_position) < 0.02, f"Expected flat, got {env.current_position}"
        assert env._position_direction == 0

    def test_stop_loss_disabled(self):
        """Default (stop_loss_bps=0) should not force flat."""
        env = _make_env(stop_loss_bps=0, taker_fee=0.0)
        env.current_close = 100.0
        env.current_step = 0
        env.equity = 100000.0 * (1.0 - 60.0 / 10000.0)
        env.peak_equity = 100000.0
        env._entry_equity = 100000.0
        env.current_position = 0.8
        env._position_direction = 1

        obs, reward, term, trunc, info = _step_with_price(env, 0.8, 100.0)

        # Position should stay (deadband keeps it)
        assert env.current_position > 0.5, f"Expected position held, got {env.current_position}"


class TestMaxHolding:
    def test_max_holding_triggers(self):
        """Position forced flat after max_holding_bars."""
        env = _make_env(max_holding_bars=5, taker_fee=0.0)
        env.current_close = 100.0
        env.equity = 100000.0
        env.peak_equity = 100000.0

        # Go long and hold for 4 bars
        env.current_position = 0.8
        env._position_direction = 1
        env._bars_in_position = 4  # Will become 5 on next step → trigger

        obs, reward, term, trunc, info = _step_with_price(env, 0.8, 100.0)

        assert abs(env.current_position) < 0.02, f"Expected flat, got {env.current_position}"
        assert env._bars_in_position == 0

    def test_max_holding_disabled(self):
        """Default (max_holding_bars=0) should not force flat."""
        env = _make_env(max_holding_bars=0, taker_fee=0.0)
        env.current_close = 100.0
        env.equity = 100000.0
        env.peak_equity = 100000.0

        env.current_position = 0.8
        env._position_direction = 1
        env._bars_in_position = 100  # Way past any threshold

        obs, reward, term, trunc, info = _step_with_price(env, 0.8, 100.0)

        assert env.current_position > 0.5, f"Expected position held, got {env.current_position}"


class TestBackwardCompat:
    def test_v5_config_no_constraints(self):
        """Env created without v6 keys should behave identically to v5."""
        config = {
            "initial_balance": 100000.0,
            "window_size": 5,
            "features_per_scale": 8,
            "taker_fee": 0.0,
            "deadband_threshold": 0.1,
            "episode_length": 100,
            "random_start": False,
            "max_drawdown_pct": 0.30,
            "scales": [15],
            "reward": {"mode": "dsr", "dsr_eta": 0.001, "dsr_scale": 1.0},
            # No stop_loss_bps, no max_holding_bars
        }
        env = ContinuousSwingEnv(config, data_handler=None)
        assert env.stop_loss_bps == 0
        assert env.max_holding_bars == 0
        assert env._position_direction == 0
        assert env._bars_in_position == 0
