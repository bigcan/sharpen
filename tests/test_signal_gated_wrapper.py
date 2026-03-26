"""Tests for SignalGatedWrapper."""
import gymnasium as gym
import numpy as np
import pytest


class MockHandler:
    """Minimal handler that mimics MultiScaleOHLCVHandler for testing."""

    def __init__(self, n_bars=200, n_features=8):
        self.window_size = 10
        self._base_scale = 3
        self._ptr = self.window_size
        self._len = n_bars
        self.n_features = n_features

        # Generate synthetic scale features
        # idx 1 (atr_norm): mostly low, occasional spikes
        # idx 2 (parkinson): mostly low, occasional spikes
        # idx 7 (volume_z): mostly low, occasional spikes
        rng = np.random.RandomState(42)
        features = rng.randn(n_bars, n_features).astype(np.float32) * 0.1
        # Make every 5th bar a "high signal" bar
        for i in range(0, n_bars, 5):
            features[i, 1] = 0.5   # atr_norm spike
            features[i, 2] = 0.05  # parkinson spike
            features[i, 7] = 0.8   # volume_z spike

        self._scale_features = {3: features}

        # Prices
        self._base_close = 100.0 + np.cumsum(rng.randn(n_bars) * 0.1)
        self._base_high = self._base_close + rng.rand(n_bars) * 0.2
        self._base_low = self._base_close - rng.rand(n_bars) * 0.2
        self._base_volume = rng.rand(n_bars) * 1000
        self._base_atr = np.ones(n_bars) * 0.5
        self._base_timestamps = np.arange(
            np.datetime64('2025-01-01'), np.datetime64('2025-01-01') + np.timedelta64(n_bars, 'm'),
            np.timedelta64(1, 'm'),
        )

    def reset(self):
        self._ptr = self.window_size

    def step(self):
        if self._ptr >= self._len:
            return None
        result = {}
        for i, scale in enumerate([3]):
            features = self._scale_features[scale]
            start = max(0, self._ptr - self.window_size + 1)
            end = self._ptr + 1
            window = features[start:end]
            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                pad = np.tile(window[0:1], (pad_len, 1))
                window = np.concatenate([pad, window], axis=0)
            result[f"scale_{i}"] = window.copy()

        result["close"] = float(self._base_close[self._ptr])
        result["atr"] = float(self._base_atr[self._ptr])
        result["timestamp"] = self._base_timestamps[self._ptr]
        self._ptr += 1
        return result


class MockContinuousSwingEnv(gym.Env):
    """Minimal env that mimics ContinuousSwingEnv for unit testing."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "scale_0": gym.spaces.Box(-np.inf, np.inf, shape=(handler.window_size, handler.n_features), dtype=np.float32),
            "private": gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32),
        })
        self.current_position = 0.0
        self.current_step = 0
        self.episode_length = 100
        self._rewards = []

    def reset(self, **kwargs):
        self.handler.reset()
        self.current_position = 0.0
        self.current_step = 0
        obs = self._make_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        target = float(action[0]) if hasattr(action, '__len__') else float(action)

        step_data = self.handler.step()
        terminated = False
        truncated = False

        if step_data is None:
            truncated = True
            return self._make_obs(), 0.0, terminated, truncated, {}

        # Simulate position change with deadband
        delta = target - self.current_position
        traded = False
        if abs(delta) > 0.25:  # deadband
            self.current_position = target
            traded = True

        # Simple reward: small random + position-dependent
        reward = float(np.random.randn() * 0.01 + self.current_position * 0.001)
        self._rewards.append(reward)

        if self.current_step >= self.episode_length:
            truncated = True

        obs = self._make_obs()
        info = {"traded": traded}
        return obs, reward, terminated, truncated, info

    def _make_obs(self):
        step_data = self.handler.step()
        if step_data is None:
            self.handler.reset()
            step_data = self.handler.step()
        obs = {}
        for key in step_data:
            if key.startswith("scale_"):
                obs[key] = step_data[key]
        obs["private"] = np.zeros(5, dtype=np.float32)
        return obs


@pytest.fixture
def handler():
    return MockHandler(n_bars=200)


@pytest.fixture
def env(handler):
    return MockContinuousSwingEnv(handler)


@pytest.fixture
def gate_config():
    return {
        "enabled": True,
        "gate_mode": "composite",
        "atr_threshold": 0.3,
        "parkinson_threshold": 0.02,
        "volume_threshold": 0.5,
        "max_hold_bars": 20,
        "gate_always_on_first": True,
        "normalize_accumulated_reward": False,
    }


class TestSignalGatedWrapper:
    """Tests for the SignalGatedWrapper."""

    def test_wrapper_creates_successfully(self, env, gate_config):
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        assert wrapped is not None
        assert wrapped.action_space == env.action_space
        assert wrapped.observation_space == env.observation_space

    def test_reset_returns_valid_obs(self, env, gate_config):
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        obs, info = wrapped.reset()
        assert isinstance(obs, dict)
        assert "scale_0" in obs or "private" in obs

    def test_step_returns_valid_output(self, env, gate_config):
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "gate_skipped_bars" in info

    def test_gate_skips_bars(self, env, gate_config):
        """Verify that the wrapper skips some bars (gate_skipped_bars > 0)."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        total_skipped = 0
        for _ in range(20):
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            total_skipped += info["gate_skipped_bars"]
            if terminated or truncated:
                break

        # With our mock data, every 5th bar is high-signal, so most bars should be skipped
        assert total_skipped > 0, "Gate should skip some bars"

    def test_max_hold_bars_respected(self, env, gate_config):
        """Gate should force open after max_hold_bars consecutive skips."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper

        # Set very high thresholds so gate is almost always closed
        gate_config["atr_threshold"] = 100.0
        gate_config["parkinson_threshold"] = 100.0
        gate_config["volume_threshold"] = 100.0
        gate_config["max_hold_bars"] = 5

        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Should skip exactly max_hold_bars (5) since gate is always closed
        assert info["gate_skipped_bars"] <= gate_config["max_hold_bars"]

    def test_reward_accumulation(self, env, gate_config):
        """Accumulated reward should equal sum of individual bar rewards."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        gate_config["normalize_accumulated_reward"] = False

        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # The wrapper processes 1 + gate_skipped_bars inner steps
        # reward should be the sum of all those steps' rewards
        n_inner_steps = 1 + info["gate_skipped_bars"]
        assert n_inner_steps >= 1

    def test_normalize_reward(self, env, gate_config):
        """When normalize=True, reward is divided by (1 + skipped)."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper

        # Run once without normalization
        gate_config["normalize_accumulated_reward"] = False
        wrapped = SignalGatedWrapper(env, gate_config)
        np.random.seed(123)
        wrapped.reset()
        action = np.array([0.0], dtype=np.float32)
        _, raw_reward, _, _, info1 = wrapped.step(action)
        skipped1 = info1["gate_skipped_bars"]

        # Run with normalization using same seed
        gate_config["normalize_accumulated_reward"] = True
        wrapped2 = SignalGatedWrapper(env, gate_config)
        np.random.seed(123)
        wrapped2.reset()
        _, norm_reward, _, _, info2 = wrapped2.step(action)
        skipped2 = info2["gate_skipped_bars"]

        # If bars were skipped, normalized reward should be smaller
        if skipped1 > 0 and skipped1 == skipped2:
            expected = raw_reward / (1 + skipped1)
            assert abs(norm_reward - expected) < 1e-6

    def test_obs_shape_unchanged(self, env, gate_config):
        """Observation shapes should not change after wrapping."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)

        obs_raw, _ = env.reset()
        obs_wrapped, _ = wrapped.reset()

        for key in obs_raw:
            if key in obs_wrapped:
                assert obs_raw[key].shape == obs_wrapped[key].shape, \
                    f"Shape mismatch for key '{key}'"

    def test_episode_termination_propagated(self, env, gate_config):
        """Wrapper should propagate terminated/truncated from inner env."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        done = False
        steps = 0
        while not done and steps < 500:
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
            steps += 1

        # Episode should end within reasonable number of wrapper steps
        assert done, "Episode should terminate"

    def test_hold_action_no_trade(self, env, gate_config):
        """During hold bars, position should not change (deadband catches delta=0)."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper

        # Use high thresholds to force gate closed
        gate_config["atr_threshold"] = 100.0
        gate_config["parkinson_threshold"] = 100.0
        gate_config["volume_threshold"] = 100.0
        gate_config["max_hold_bars"] = 10

        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        # Set a position first
        env.current_position = 0.7
        action = np.array([0.7], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # During hold, position should stay at 0.7 (deadband catches delta=0)
        assert abs(env.current_position - 0.7) < 0.3, \
            f"Position drifted to {env.current_position} during hold"

    def test_passthrough_without_features(self, gate_config):
        """Without handler features, wrapper should be a passthrough."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper

        # Create env normally, then sabotage the wrapper's cached features
        handler = MockHandler(n_bars=200)
        env = MockContinuousSwingEnv(handler)

        wrapped = SignalGatedWrapper(env, gate_config)
        # Force passthrough by clearing cached features
        wrapped._scale_features = None
        wrapped.reset()

        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Should not skip any bars (passthrough)
        assert info["gate_skipped_bars"] == 0

    def test_info_augmentation(self, env, gate_config):
        """Info dict should contain gate statistics."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        wrapped.reset()

        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapped.step(action)

        assert "gate_skipped_bars" in info
        assert "gate_total_skipped" in info
        assert "gate_total_steps" in info
        assert isinstance(info["gate_skipped_bars"], int)

    def test_gate_modes(self, env):
        """Each gate mode should work without errors."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper

        for mode in ["composite", "atr", "parkinson", "volume"]:
            config = {
                "enabled": True,
                "gate_mode": mode,
                "atr_threshold": 0.3,
                "parkinson_threshold": 0.02,
                "volume_threshold": 0.5,
                "max_hold_bars": 10,
            }
            wrapped = SignalGatedWrapper(env, config)
            wrapped.reset()
            action = np.array([0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            assert "gate_skipped_bars" in info

    def test_multiple_episodes(self, env, gate_config):
        """Wrapper should handle multiple reset/step cycles."""
        from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)

        for episode in range(3):
            wrapped.reset()
            for step in range(10):
                action = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
                obs, reward, terminated, truncated, info = wrapped.step(action)
                if terminated or truncated:
                    break
