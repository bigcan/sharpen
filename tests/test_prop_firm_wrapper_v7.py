"""Tests for PropFirmWrapperV7 — Dict and Flat obs support."""
import gymnasium as gym
import numpy as np
import pytest

from sharpen.envs.prop_firm_wrapper import PropFirmWrapperV7


# ---------------------------------------------------------------------------
# Helpers: minimal mock envs
# ---------------------------------------------------------------------------

class MockDictEnv(gym.Env):
    """Minimal V7-like env with Dict obs, timestamps, and step_idx."""

    def __init__(self, n_scales: int = 3, private_dim: int = 5, window: int = 30, features: int = 8):
        super().__init__()
        self.n_scales = n_scales
        self.private_dim = private_dim
        self.window = window
        self.features = features

        spaces = {}
        for i in range(n_scales):
            spaces[f"scale_{i}"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window, features), dtype=np.float32,
            )
        spaces["private"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(private_dim,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.initial_capital = 100_000.0
        self._equity = self.initial_capital
        self._step = 0
        # Generate timestamps: 3-min bars starting 2025-06-01 00:00 UTC
        base = int(np.datetime64("2025-06-01T00:00", "s").astype("int64"))
        self.timestamps = np.array([base + i * 180 for i in range(10_000)], dtype=np.int64)
        self.step_idx = 0

    def _obs(self):
        obs = {}
        for i in range(self.n_scales):
            obs[f"scale_{i}"] = np.zeros((self.window, self.features), dtype=np.float32)
        obs["private"] = np.zeros(self.private_dim, dtype=np.float32)
        return obs

    def reset(self, **kwargs):
        self._equity = self.initial_capital
        self._step = 0
        self.step_idx = 1
        return self._obs(), {"portfolio_value": self._equity}

    def step(self, action):
        self._step += 1
        self.step_idx = min(self._step + 1, len(self.timestamps) - 1)
        return self._obs(), 0.0, False, False, {"portfolio_value": self._equity}

    def set_equity(self, val: float):
        self._equity = val

    def advance_day(self):
        """Jump step_idx forward by ~480 bars (1 day at 3-min)."""
        self.step_idx = min(self.step_idx + 480, len(self.timestamps) - 1)


class MockFlatEnv(gym.Env):
    """Minimal flat-obs env (mimics CryptoPerpEnv)."""

    def __init__(self, obs_dim: int = 50):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.initial_capital = 100_000.0
        self._equity = self.initial_capital
        self._step = 0
        base = int(np.datetime64("2025-06-01T00:00", "s").astype("int64"))
        self.timestamps = np.array([base + i * 180 for i in range(10_000)], dtype=np.int64)
        self.step_idx = 0

    def reset(self, **kwargs):
        self._equity = self.initial_capital
        self._step = 0
        self.step_idx = 1
        return np.zeros(self.observation_space.shape[0], dtype=np.float32), {"portfolio_value": self._equity}

    def step(self, action):
        self._step += 1
        self.step_idx = min(self._step + 1, len(self.timestamps) - 1)
        return np.zeros(self.observation_space.shape[0], dtype=np.float32), 0.0, False, False, {"portfolio_value": self._equity}

    def set_equity(self, val: float):
        self._equity = val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDictObsAugmentation:
    """Test PropFirmWrapperV7 with Dict observation space."""

    def test_obs_space_extended(self):
        env = MockDictEnv(private_dim=5)
        wrapped = PropFirmWrapperV7(env, augment_obs=True)
        assert wrapped.observation_space["private"].shape == (8,)
        # Scale spaces unchanged
        assert wrapped.observation_space["scale_0"].shape == (30, 8)

    def test_obs_augmented_on_reset(self):
        env = MockDictEnv(private_dim=5)
        wrapped = PropFirmWrapperV7(env, augment_obs=True)
        obs, info = wrapped.reset()
        assert obs["private"].shape == (8,)
        # Extra dims: dd_remaining=1.0 (no DD), daily_remaining=1.0, profit_progress=0.0
        np.testing.assert_allclose(obs["private"][-3:], [1.0, 1.0, 0.0], atol=1e-6)

    def test_obs_augmented_on_step(self):
        env = MockDictEnv(private_dim=5)
        wrapped = PropFirmWrapperV7(env, augment_obs=True)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(np.array([0.0]))
        assert obs["private"].shape == (8,)

    def test_no_augmentation_passthrough(self):
        env = MockDictEnv(private_dim=5)
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        assert wrapped.observation_space["private"].shape == (5,)
        obs, _ = wrapped.reset()
        assert obs["private"].shape == (5,)


class TestFlatObsAugmentation:
    """Test PropFirmWrapperV7 with flat Box observation space."""

    def test_obs_space_extended(self):
        env = MockFlatEnv(obs_dim=50)
        wrapped = PropFirmWrapperV7(env, augment_obs=True)
        assert wrapped.observation_space.shape == (53,)

    def test_obs_augmented_on_step(self):
        env = MockFlatEnv(obs_dim=50)
        wrapped = PropFirmWrapperV7(env, augment_obs=True)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(np.array([0.0]))
        assert obs.shape == (53,)

    def test_no_augmentation_passthrough(self):
        env = MockFlatEnv(obs_dim=50)
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        assert wrapped.observation_space.shape == (50,)


class TestEODTrailingDrawdown:
    """Test EOD trailing drawdown termination."""

    def test_terminates_on_dd_breach(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            augment_obs=False,
        )
        wrapped.reset()

        # Simulate 11% equity loss
        env.set_equity(89_000.0)
        _, _, terminated, _, info = wrapped.step(np.array([0.0]))
        assert terminated
        assert info.get("prop_firm_termination") == "eod_trailing_drawdown"

    def test_no_termination_within_limit(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            augment_obs=False,
        )
        wrapped.reset()

        # 5% loss — within limit
        env.set_equity(95_000.0)
        _, _, terminated, _, info = wrapped.step(np.array([0.0]))
        assert not terminated
        assert "prop_firm_termination" not in info


class TestDailyLossLimit:
    """Test daily loss limit termination."""

    def test_terminates_on_daily_loss(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.05,
            max_trailing_drawdown_pct=0.20,  # high so DD doesn't trigger first
            augment_obs=False,
        )
        wrapped.reset()

        # 6% intraday loss
        env.set_equity(94_000.0)
        _, _, terminated, _, info = wrapped.step(np.array([0.0]))
        assert terminated
        assert info.get("prop_firm_termination") == "daily_loss_limit"

    def test_disabled_when_zero(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.0,
            max_trailing_drawdown_pct=0.20,
            augment_obs=False,
        )
        wrapped.reset()

        env.set_equity(94_000.0)
        _, _, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert not terminated  # DD is only 6%, within 20%


class TestProfitTarget:
    """Test profit target early termination."""

    def test_terminates_on_profit_target(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            profit_target_pct=0.10,
            success_bonus=10.0,
            augment_obs=False,
        )
        wrapped.reset()

        # +11% return
        env.set_equity(111_000.0)
        _, reward, terminated, _, info = wrapped.step(np.array([0.0]))
        assert terminated
        assert info.get("prop_firm_termination") == "profit_target_reached"
        assert info.get("challenge_passed") is True
        assert reward == 10.0  # base reward 0.0 + success_bonus 10.0

    def test_no_termination_below_target(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            profit_target_pct=0.10,
            augment_obs=False,
        )
        wrapped.reset()

        env.set_equity(105_000.0)  # +5%
        _, _, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert not terminated


class TestRewardShaping:
    """Test drawdown proximity penalty."""

    def test_penalty_applied_near_dd_limit(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            drawdown_penalty_start=0.05,
            drawdown_penalty_scale=5.0,
            augment_obs=False,
        )
        wrapped.reset()

        # 7% drawdown — above 5% start, below 10% limit
        env.set_equity(93_000.0)
        _, reward, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert not terminated
        assert reward < 0.0  # should have penalty

    def test_no_penalty_below_start(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            drawdown_penalty_start=0.05,
            drawdown_penalty_scale=5.0,
            augment_obs=False,
        )
        wrapped.reset()

        # 3% drawdown — below 5% start
        env.set_equity(97_000.0)
        _, reward, _, _, _ = wrapped.step(np.array([0.0]))
        assert reward == 0.0  # no penalty


class TestDailyLossPenalty:
    """Test daily-loss proximity quadratic shaping (S469 ablation)."""

    def test_zero_when_disabled(self):
        # Default: daily_loss_penalty_scale=0.0 → no penalty regardless of daily loss
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.05,
            drawdown_penalty_scale=0.0,  # disable DD shaping to isolate
            # daily_loss_penalty_scale defaults to 0.0
            augment_obs=False,
        )
        wrapped.reset()

        # 4% daily loss — well past start but scale=0 → 0 reward
        env.set_equity(96_000.0)
        _, reward, _, _, _ = wrapped.step(np.array([0.0]))
        assert reward == 0.0

    def test_zero_below_start(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.05,
            drawdown_penalty_scale=0.0,
            daily_loss_penalty_start=0.03,
            daily_loss_penalty_scale=5.0,
            augment_obs=False,
        )
        wrapped.reset()

        # 2% daily loss — below 3% start
        env.set_equity(98_000.0)
        _, reward, _, _, _ = wrapped.step(np.array([0.0]))
        assert reward == 0.0

    def test_quadratic_in_band(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.05,
            drawdown_penalty_scale=0.0,
            daily_loss_penalty_start=0.03,
            daily_loss_penalty_scale=5.0,
            augment_obs=False,
        )
        wrapped.reset()

        # 4% daily loss — midpoint of [0.03, 0.05] → frac = 0.5 → penalty = -5.0 * 0.25
        env.set_equity(96_000.0)
        _, reward, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert not terminated  # 4% < 5% cap
        assert reward == pytest.approx(-1.25, abs=1e-6)

    def test_bounded_at_cap(self):
        # At the cap, penalty = -scale exactly (frac=1.0).
        # But the wrapper terminates on daily_loss > cap; so we check penalty
        # at *just below* cap using a non-terminating daily loss.
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_daily_loss_pct=0.10,  # wider cap so 5% daily loss doesn't terminate
            drawdown_penalty_scale=0.0,
            daily_loss_penalty_start=0.0,  # start at 0 → frac = daily_loss / cap
            daily_loss_penalty_scale=5.0,
            augment_obs=False,
        )
        wrapped.reset()

        # 5% daily loss on a 10% cap → frac = 0.5 → penalty = -5.0 * 0.25
        env.set_equity(95_000.0)
        _, reward, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert not terminated
        assert reward == pytest.approx(-1.25, abs=1e-6)


class TestInfoFields:
    """Test that info dict contains prop firm metadata."""

    def test_info_fields_present(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        wrapped.reset()

        env.set_equity(98_000.0)
        _, _, _, _, info = wrapped.step(np.array([0.0]))
        assert "eod_drawdown" in info
        assert "eod_peak_equity" in info
        assert "cumulative_return" in info
        assert "profit_progress" in info
        assert "drawdown_budget_remaining" in info
        assert info["cumulative_return"] == pytest.approx(-0.02, abs=1e-6)


class TestBaseEnvTermination:
    """Test that base env termination is passed through."""

    def test_base_terminated_passthrough(self):
        """If inner env terminates, wrapper should not override."""
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        wrapped.reset()

        # Monkey-patch step to return terminated=True
        original_step = env.step
        def terminated_step(action):
            obs, r, _, truncated, info = original_step(action)
            return obs, r, True, truncated, info
        env.step = terminated_step

        _, _, terminated, _, _ = wrapped.step(np.array([0.0]))
        assert terminated


class TestStaticPeak:
    """FTMO Phase 1 Challenge: peak must lock at initial_capital."""

    def test_static_peak_default_is_true(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        assert wrapped.static_peak is True

    def test_static_peak_locks_across_day_boundary(self):
        """Peak stays at initial_capital even after profit + day boundary."""
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            profit_target_pct=1.0,  # disable
            augment_obs=False,
            static_peak=True,
        )
        wrapped.reset()

        # Ramp to 110K on day 1
        env.set_equity(110_000.0)
        wrapped.step(np.array([0.0]))
        # Advance ~1 day (480 bars at 3-min) — mock.step() recomputes step_idx from _step
        env._step += 480
        env.set_equity(95_000.0)  # 5% below static $100K floor
        _, _, terminated, _, info = wrapped.step(np.array([0.0]))

        assert info["eod_peak_equity"] == 100_000.0, "static peak must not ratchet"
        assert not terminated, "5% DD from static $100K is within 10% limit"

    def test_rolling_peak_ratchets_when_opt_in(self):
        """static_peak=False restores legacy funded-account trailing behavior."""
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(
            env,
            max_trailing_drawdown_pct=0.10,
            profit_target_pct=1.0,
            augment_obs=False,
            static_peak=False,
        )
        wrapped.reset()

        # Day 1: accumulate profit
        env.set_equity(110_000.0)
        wrapped.step(np.array([0.0]))

        # Day boundary crossing step — wrapper ratchets peak to current equity (110K)
        env._step += 480
        env.set_equity(110_000.0)
        _, _, _, _, info = wrapped.step(np.array([0.0]))
        assert info["eod_peak_equity"] == pytest.approx(110_000.0), "peak should ratchet to 110K"

        # Now drop to 95K — 13.6% below rolling $110K peak, breach 10% DD
        env.set_equity(95_000.0)
        _, _, terminated, _, info = wrapped.step(np.array([0.0]))
        assert terminated, "rolling peak must enforce 10% DD from ratcheted $110K"


# ---------------------------------------------------------------------------
# Deprecation warnings (rev-2 adapter)
# ---------------------------------------------------------------------------

class TestDeprecationWarnings:
    """Ensure the rev-2 adapter surfaces the profit_progress deprecation."""

    def test_profit_progress_access_warns(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([0.0]))

        with pytest.warns(DeprecationWarning, match="profit_progress"):
            _ = info["profit_progress"]

    def test_profit_progress_get_warns(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([0.0]))

        with pytest.warns(DeprecationWarning, match="profit_progress"):
            _ = info.get("profit_progress")

    def test_non_profit_progress_keys_do_not_warn(self):
        env = MockDictEnv()
        wrapped = PropFirmWrapperV7(env, augment_obs=False)
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([0.0]))

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            _ = info["eod_drawdown"]
            _ = info["cumulative_return"]
            _ = info.get("portfolio_value")
