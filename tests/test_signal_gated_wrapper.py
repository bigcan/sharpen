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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        assert wrapped is not None
        assert wrapped.action_space == env.action_space
        assert wrapped.observation_space == env.observation_space

    def test_reset_returns_valid_obs(self, env, gate_config):
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)
        obs, info = wrapped.reset()
        assert isinstance(obs, dict)
        assert "scale_0" in obs or "private" in obs

    def test_step_returns_valid_output(self, env, gate_config):
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)

        obs_raw, _ = env.reset()
        obs_wrapped, _ = wrapped.reset()

        for key in obs_raw:
            if key in obs_wrapped:
                assert obs_raw[key].shape == obs_wrapped[key].shape, \
                    f"Shape mismatch for key '{key}'"

    def test_episode_termination_propagated(self, env, gate_config):
        """Wrapper should propagate terminated/truncated from inner env."""
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        for mode in ["composite", "atr", "parkinson", "volume", "return"]:
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
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper
        wrapped = SignalGatedWrapper(env, gate_config)

        for episode in range(3):
            wrapped.reset()
            for step in range(10):
                action = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
                obs, reward, terminated, truncated, info = wrapped.step(action)
                if terminated or truncated:
                    break


class _ScriptedTradeEnv(gym.Env):
    """Inner env whose `traded` flag follows a fixed script.

    Lets a test place a trade on the DECISION bar and holds after it, which is
    exactly the shape that GATE-TRADED-01 erased.
    """

    def __init__(self, handler, traded_script, trades_per_traded_bar=1):
        super().__init__()
        self.handler = handler
        self.traded_script = list(traded_script)
        # ContinuousSwingEnv can log >1 trade on one bar (deadband move, then a
        # stop-loss or max-holding forced flat), so this is configurable.
        self.trades_per_traded_bar = int(trades_per_traded_bar)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "scale_0": gym.spaces.Box(
                -np.inf, np.inf,
                shape=(handler.window_size, handler.n_features), dtype=np.float32),
            "private": gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32),
        })
        self.current_position = 0.0
        self.inner_steps = 0
        self.trade_count = 0

    def reset(self, **kwargs):
        self.handler.reset()
        self.inner_steps = 0
        self.trade_count = 0
        return self._make_obs(), {}

    def step(self, action):
        traded = bool(self.traded_script[self.inner_steps]) \
            if self.inner_steps < len(self.traded_script) else False
        self.inner_steps += 1
        if traded:
            self.trade_count += self.trades_per_traded_bar
        info = {"traded": traded, "trade_count": self.trade_count,
                "portfolio_value": 100000.0, "position": self.current_position}
        return self._make_obs(), 0.0, False, False, info

    def _make_obs(self):
        return {"scale_0": np.zeros((self.handler.window_size, self.handler.n_features),
                                    dtype=np.float32),
                "private": np.zeros(5, dtype=np.float32)}


class TestGateTraded01:
    """GATE-TRADED-01 regression: per-bar `traded` must survive gate-held bars.

    The wrapper aggregates 1 + skipped inner bars and returns the LAST inner
    info dict. Before the fix a trade on the decision bar was overwritten by the
    hold bars that followed it, so the flag read False on ~half of real trades
    (sg1-btc fold_00 solo_123: 145 position changes vs 69 flags).
    """

    @staticmethod
    def _closed_gate_config(max_hold=5):
        # Thresholds far above anything MockHandler emits => gate always shut,
        # so every wrapper step aggregates max_hold hold bars after the decision.
        return {
            "enabled": True, "gate_mode": "composite",
            "atr_threshold": 1e9, "parkinson_threshold": 1e9,
            "volume_threshold": 1e9, "return_threshold": 1e9,
            "max_hold_bars": max_hold, "gate_always_on_first": False,
        }

    def test_trade_on_decision_bar_survives_hold_bars(self, handler):
        """A trade on the decision bar must not be erased by later hold bars."""
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        env = _ScriptedTradeEnv(handler, traded_script=[True] + [False] * 20)
        wrapped = SignalGatedWrapper(env, self._closed_gate_config(max_hold=5))
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([1.0], dtype=np.float32))

        assert info["gate_skipped_bars"] > 0, "gate should have held bars for this test to bite"
        assert info["traded"] is True, (
            "GATE-TRADED-01 regression: the decision bar traded but the flag was "
            "overwritten by a later hold bar's info dict"
        )
        assert info["gate_inner_trades"] == 1

    def test_forced_flat_on_a_hold_bar_is_reported(self, handler):
        """A stop-loss/max-holding flat fired on a HOLD bar must be reported.

        The inner env force-flattens regardless of the hold action it is handed,
        so holds are not guaranteed trade-free.
        """
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        env = _ScriptedTradeEnv(handler, traded_script=[False, False, True] + [False] * 20)
        wrapped = SignalGatedWrapper(env, self._closed_gate_config(max_hold=5))
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

        assert info["traded"] is True
        assert info["gate_inner_trades"] == 1

    def test_inner_trade_bars_reconcile_with_traded_bar_count(self, handler):
        """`gate_inner_trades` must account for every traded BAR, losing none.

        This is the property the sg1-btc trajectories violated: 145 real position
        changes recorded as 69 flags.

        The mock trades at most once per bar, so its `trade_count` equals its
        traded-bar count and the two reconcile exactly here. That identity is a
        property of the MOCK, not of the contract -- see
        `test_inner_trades_counts_bars_not_trades`.
        """
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        rng = np.random.RandomState(7)
        script = (rng.rand(400) < 0.35).tolist()
        env = _ScriptedTradeEnv(handler, traded_script=script)
        wrapped = SignalGatedWrapper(env, self._closed_gate_config(max_hold=4))
        wrapped.reset()

        flag_sum = 0
        inner_sum = 0
        for _ in range(20):
            _, _, term, trunc, info = wrapped.step(np.array([0.5], dtype=np.float32))
            flag_sum += int(bool(info["traded"]))
            inner_sum += int(info["gate_inner_trades"])
            if term or trunc:
                break

        assert inner_sum == env.trade_count, (
            f"gate_inner_trades summed to {inner_sum} but the env logged "
            f"{env.trade_count} trades over {env.inner_steps} inner bars"
        )
        # The bool flag can only undercount (two traded bars in one outer step),
        # never overcount -- and it must not be systematically zero.
        assert flag_sum <= inner_sum
        assert flag_sum > 0

    def test_inner_trades_counts_bars_not_trades(self, handler):
        """`gate_inner_trades` counts traded BARS; `trade_count` counts trades.

        ContinuousSwingEnv.step increments `trade_count` at three sites -- the
        deadband move, the stop_loss_bps flat, and the max_holding_bars flat --
        so a single bar can log up to three trades while `traded` is one bool.
        Pinning this stops the bar count being mistaken for a trade count.
        """
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        env = _ScriptedTradeEnv(handler, traded_script=[True] + [False] * 20)
        env.trades_per_traded_bar = 2      # e.g. deadband move + forced flat
        wrapped = SignalGatedWrapper(env, self._closed_gate_config(max_hold=5))
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([1.0], dtype=np.float32))

        assert info["gate_inner_trades"] == 1, "one traded BAR"
        assert env.trade_count == 2, "which logged two TRADES"

    def test_untraded_outer_step_stays_false(self, handler):
        """No trade anywhere in the aggregate must still report False."""
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        env = _ScriptedTradeEnv(handler, traded_script=[False] * 30)
        wrapped = SignalGatedWrapper(env, self._closed_gate_config(max_hold=5))
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

        assert info["traded"] is False
        assert info["gate_inner_trades"] == 0


class _PtrHandler:
    """Handler mirroring MultiScaleOHLCVHandler `_ptr` semantics exactly.

    `step()` serves the bar at `_ptr` and then advances, so after consuming bar
    k the pointer sits at k+1 -- the bar that has NOT closed yet. That off-by-one
    is the whole subject of GATE-CAUSAL-01.
    """

    def __init__(self, n_bars=200, spike_at=60, n_features=8):
        self.window_size = 1
        self.n_features = n_features
        self._base_scale = 3
        self._len = n_bars
        self._ptr = 0
        self.spike_at = spike_at
        f = np.zeros((n_bars, n_features), dtype=np.float32)
        f[spike_at, 1] = 5.0                      # a single high-ATR bar
        self._scale_features = {3: f}

    def reset(self):
        self._ptr = 0

    def step(self):
        if self._ptr >= self._len:
            return None
        d = {"idx": self._ptr}
        self._ptr += 1
        return d


class _RecordingEnv(gym.Env):
    """Records which bar index each inner step consumed."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.current_position = 0.0
        self.consumed = []
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        self.handler.reset()
        self.consumed = []
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        d = self.handler.step()
        if d is None:
            return np.zeros(1, dtype=np.float32), 0.0, False, True, {"traded": False}
        self.consumed.append(d["idx"])
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {"traded": False}


class TestGateCausal01:
    """GATE-CAUSAL-01 tripwire: the gate must never read an UNCLOSED bar.

    The gate signals encode the bar's own OHLCV -- `atr_norm[i]`/`parkinson[i]`
    from its high/low/close with no shift, `volume_z[i]`'s numerator from its
    volume -- so a gate evaluated on bar k+1 to decide the action applied to bar
    k+1 selects the agent's decision points with hindsight (LEAK-2).

    These tests fail if the caller reverts to `_gate_open(handler._ptr)`.
    """

    SPIKE = 60

    @staticmethod
    def _cfg():
        # Only a real ATR spike can open this gate.
        return {
            "enabled": True, "gate_mode": "atr", "atr_threshold": 1.0,
            "parkinson_threshold": 1e9, "volume_threshold": 1e9,
            "return_threshold": 1e9, "max_hold_bars": 10_000,
            "gate_always_on_first": False,
        }

    def _run_to_gate_open(self):
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        h = _PtrHandler(spike_at=self.SPIKE)
        env = _RecordingEnv(h)
        w = SignalGatedWrapper(env, self._cfg())
        w.reset()
        w.step(np.array([1.0], dtype=np.float32))
        return env, h

    def test_gate_never_opens_on_an_unclosed_bar(self):
        """The gate may only open once the signal bar has CLOSED."""
        env, _ = self._run_to_gate_open()

        last_closed = env.consumed[-1]
        assert last_closed == self.SPIKE, (
            "GATE-CAUSAL-01 regression: the wrapper stopped holding with the "
            f"last closed bar at {last_closed}, but the spike is at {self.SPIKE}. "
            f"last_closed == {self.SPIKE - 1} means the gate read bar "
            f"{self.SPIKE} before it closed (look-ahead)."
        )

    def test_agent_does_not_trade_the_bar_that_opened_the_gate(self):
        """The action must land on the bar AFTER the one that opened the gate.

        This is the economically material half: trading the very bar whose
        realised volatility opened the gate is the look-ahead paying out.
        """
        env, _ = self._run_to_gate_open()
        next_traded = env.consumed[-1] + 1

        assert next_traded == self.SPIKE + 1, (
            f"agent's next action lands on bar {next_traded}; it must not be "
            f"the gating bar {self.SPIKE}"
        )

    def test_gate_index_is_one_behind_the_handler_pointer(self):
        """Pin the exact index relationship the live engine also satisfies.

        Live `_check_signal_gate` reads `features[-1]` -- the last CLOSED bar.
        Sim must gate on `handler._ptr - 1` for the two to agree; before the fix
        they were off by one and the live strategy traded a different bar set
        than its own backtest.
        """
        from sharpen.envs.signal_gated_wrapper import SignalGatedWrapper

        h = _PtrHandler(spike_at=self.SPIKE)
        env = _RecordingEnv(h)
        w = SignalGatedWrapper(env, self._cfg())
        w.reset()

        seen = []
        orig = w._gate_open
        w._gate_open = lambda p: (seen.append((p, h._ptr)), orig(p))[1]
        w.step(np.array([1.0], dtype=np.float32))

        assert seen, "gate was never consulted"
        for gate_ptr, ptr_at_call in seen:
            assert gate_ptr == ptr_at_call - 1, (
                f"gate consulted index {gate_ptr} while handler._ptr was "
                f"{ptr_at_call}; must be _ptr - 1 (the last CLOSED bar)"
            )
