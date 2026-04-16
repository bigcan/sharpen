"""Tests for PropFirmWrapper — prop firm challenge constraint wrapper.

Covers:
1. Observation space augmentation (+3 dims)
2. EOD trailing drawdown floor updates only at day boundary
3. Daily loss limit terminates correctly
4. Profit target triggers early termination with success bonus
5. Reward penalty increases quadratically near drawdown limit
6. Base env termination passes through unchanged
7. Challenge info dict contains expected keys
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from finrl_pro_ds.crypto.envs.prop_firm_wrapper import PropFirmWrapper


# ---------------------------------------------------------------------------
# Minimal base environment for testing
# ---------------------------------------------------------------------------

class _MockCryptoEnv(gym.Env):
    """Minimal mock env that returns controllable portfolio values.

    Simulates a simple env where portfolio_value can be set externally
    via the ``_pv_sequence`` list.  Each step() pops the next PV and
    includes it in the info dict, mimicking CryptoPerpEnv behavior.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim: int = 10,
        pv_sequence: list[float] | None = None,
        initial_capital: float = 100_000.0,
        n_bars: int = 200,
        bar_interval_hours: int = 1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        self.initial_capital = initial_capital
        self._pv_sequence = list(pv_sequence) if pv_sequence else [initial_capital] * n_bars
        # Pad PV sequence to n_bars so base env doesn't terminate early
        while len(self._pv_sequence) < n_bars:
            self._pv_sequence.append(self._pv_sequence[-1])
        self._bar_interval_s = bar_interval_hours * 3600

        # Timestamps: start 2024-01-01 00:00 UTC
        base_ts = 1704067200
        n = max(n_bars, len(self._pv_sequence))
        self.timestamps = np.array(
            [base_ts + i * self._bar_interval_s for i in range(n + 1)],
            dtype=np.int64,
        )
        self.step_idx = 0
        self._done = False

    def reset(self, **kwargs):
        self.step_idx = 0
        self._done = False
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_idx += 1
        pv_idx = min(self.step_idx - 1, len(self._pv_sequence) - 1)
        pv = self._pv_sequence[pv_idx]

        terminated = self.step_idx >= len(self._pv_sequence)
        truncated = False
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_value": pv,
            "step_return": 0.0,
        }
        return obs, 0.0, terminated, truncated, info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObsAugmentation:
    """Test observation space is extended by 3 dims."""

    def test_augmented_obs_shape(self):
        base_env = _MockCryptoEnv(obs_dim=10)
        env = PropFirmWrapper(base_env, augment_obs=True)
        obs, _ = env.reset()
        assert obs.shape == (13,), f"Expected (13,), got {obs.shape}"
        assert env.observation_space.shape == (13,)

    def test_no_augmentation(self):
        base_env = _MockCryptoEnv(obs_dim=10)
        env = PropFirmWrapper(base_env, augment_obs=False)
        obs, _ = env.reset()
        assert obs.shape == (10,)
        assert env.observation_space.shape == (10,)

    def test_augmented_dims_range(self):
        """Extra dims should be in [0, 1]."""
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=[100_000.0] * 10)
        env = PropFirmWrapper(base_env, augment_obs=True)
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))

        extra = obs[-3:]
        assert all(0.0 <= v <= 1.0 for v in extra), f"Extra dims out of range: {extra}"


class TestEODTrailingDrawdown:
    """Test EOD trailing drawdown floor updates only at day boundary."""

    def test_intraday_spike_does_not_raise_floor(self):
        """PV spike within a day should NOT raise the drawdown floor."""
        # 24 hourly bars in one day: PV rises to 110K then drops to 95K
        # EOD floor should stay at initial 100K (not ratchet to 110K)
        pv = [100_000.0] * 5 + [110_000.0] * 5 + [95_000.0] * 14
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.10,
            max_daily_loss_pct=0.0,
            profit_target_pct=1.0,  # Disable profit target
        )
        env.reset()

        terminated = False
        for _ in range(len(pv)):
            _, _, terminated, _, info = env.step(np.array([0.0]))
            if terminated:
                break

        # 95K is 5% below 100K (initial/EOD peak) — within 10% limit
        # Should NOT terminate
        assert not terminated, (
            f"Should not terminate: EOD peak should be 100K, not intraday peak. "
            f"Info: {info}"
        )

    def test_eod_floor_ratchets_at_day_boundary(self):
        """Floor should update when a new day starts — only with static_peak=False."""
        # Day 1 (24h bars): PV = 110K at end
        # Day 2 (24h bars): PV drops — floor should now be 110K
        day1 = [110_000.0] * 24
        day2 = [99_500.0] * 24  # 9.5% below 110K → within 10%
        pv = day1 + day2
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.10,
            max_daily_loss_pct=0.0,
            profit_target_pct=1.0,
            static_peak=False,  # funded-account trailing DD
        )
        env.reset()

        last_info = {}
        for _ in range(len(pv)):
            _, _, terminated, _, last_info = env.step(np.array([0.0]))
            if terminated:
                break

        # EOD peak should be 110K (set at day 1→2 boundary)
        assert last_info.get("eod_peak_equity", 0) >= 109_000.0

    def test_static_peak_locks_at_initial_capital(self):
        """FTMO Phase 1 Challenge rule: peak locked at initial_capital forever."""
        # Day 1: PV ramps to 110K (profit target hit, but target disabled here)
        # Day 2: PV = 95K — 5% below initial 100K (static floor), 13.6% below rolling 110K
        day1 = [110_000.0] * 24
        day2 = [95_000.0] * 24
        pv = day1 + day2
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.10,
            max_daily_loss_pct=0.0,
            profit_target_pct=1.0,  # Disable profit target
            static_peak=True,  # FTMO-compliant (default)
        )
        env.reset()

        last_info = {}
        terminated = False
        for _ in range(len(pv)):
            _, _, terminated, _, last_info = env.step(np.array([0.0]))
            if terminated:
                break

        # Static peak must remain at initial_capital ($100K), never ratchet to $110K
        assert last_info.get("eod_peak_equity") == 100_000.0, (
            f"static_peak=True must lock peak at initial_capital; got {last_info.get('eod_peak_equity')}"
        )
        # 95K is 5% below static 100K floor — within 10% limit → no termination
        assert not terminated, "Should not terminate: 5% DD from static $100K is within 10% limit"

    def test_terminates_on_drawdown_breach(self):
        """Should terminate when EOD drawdown exceeds limit."""
        # Day 1: PV = 100K, Day 2: PV drops to 89K (11% DD > 10% limit)
        day1 = [100_000.0] * 24
        day2 = [89_000.0] * 24
        pv = day1 + day2
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.10,
            max_daily_loss_pct=0.0,
            profit_target_pct=1.0,
        )
        env.reset()

        terminated = False
        info = {}
        for _ in range(len(pv)):
            _, _, terminated, _, info = env.step(np.array([0.0]))
            if terminated:
                break

        assert terminated
        assert info.get("prop_firm_termination") == "eod_trailing_drawdown"


class TestDailyLossLimit:
    """Test daily loss limit."""

    def test_daily_loss_terminates(self):
        """Should terminate when daily loss exceeds limit."""
        # Day 1: starts at 100K, drops to 95K (5% daily loss > 4% limit)
        pv = [95_000.0] * 24
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.50,  # High (won't trigger)
            max_daily_loss_pct=0.04,
            profit_target_pct=1.0,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.array([0.0]))
        assert terminated
        assert info.get("prop_firm_termination") == "daily_loss_limit"

    def test_daily_loss_resets_at_midnight(self):
        """Daily loss should reset when a new UTC day starts."""
        # Day 1: PV drops to 97K (3% loss — below 4% limit)
        # Day 2: PV = 97K (new day start) → no daily loss yet
        day1 = [97_000.0] * 24
        day2 = [97_000.0] * 24  # Flat on day 2
        pv = day1 + day2
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.50,
            max_daily_loss_pct=0.04,
            profit_target_pct=1.0,
        )
        env.reset()

        terminated = False
        for _ in range(len(pv)):
            _, _, terminated, _, info = env.step(np.array([0.0]))
            if terminated:
                break

        # Should not terminate — 3% daily loss within 4% limit, and day 2 is flat
        assert not terminated

    def test_disabled_daily_loss(self):
        """Daily loss should be disabled when set to 0.0."""
        pv = [80_000.0] * 10  # 20% drop, but daily loss disabled
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.50,
            max_daily_loss_pct=0.0,  # Disabled
            profit_target_pct=1.0,
        )
        env.reset()

        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            if terminated:
                break

        assert not terminated


class TestProfitTarget:
    """Test profit target early termination."""

    def test_profit_target_reached(self):
        """Should terminate with success when profit target is hit."""
        pv = [111_000.0] * 10  # 11% return > 10% target
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            profit_target_pct=0.10,
            max_trailing_drawdown_pct=0.50,
            max_daily_loss_pct=0.0,
            success_bonus=10.0,
        )
        env.reset()

        _, reward, terminated, _, info = env.step(np.array([0.0]))
        assert terminated
        assert info.get("challenge_passed") is True
        assert info.get("prop_firm_termination") == "profit_target_reached"
        assert reward >= 10.0  # Includes success bonus

    def test_below_target_continues(self):
        """Should not terminate when below profit target."""
        pv = [109_000.0] * 10  # 9% return < 10% target
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            profit_target_pct=0.10,
            max_trailing_drawdown_pct=0.50,
            max_daily_loss_pct=0.0,
        )
        env.reset()

        _, _, terminated, _, info = env.step(np.array([0.0]))
        assert not terminated
        assert info.get("profit_progress", 0) < 1.0


class TestRewardShaping:
    """Test drawdown proximity penalty."""

    def test_no_penalty_below_start(self):
        """No penalty when drawdown is below the start threshold."""
        pv = [99_000.0] * 24  # 1% DD (below 5% start)
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=pv, bar_interval_hours=1)
        env = PropFirmWrapper(
            base_env,
            max_trailing_drawdown_pct=0.10,
            drawdown_penalty_start=0.05,
            drawdown_penalty_scale=5.0,
            profit_target_pct=1.0,
            max_daily_loss_pct=0.0,
        )
        env.reset()

        _, reward, _, _, _ = env.step(np.array([0.0]))
        # Base reward is 0.0, penalty should be 0 (DD below start)
        assert reward == 0.0

    def test_penalty_increases_with_drawdown(self):
        """Penalty should increase as drawdown approaches limit."""
        # We need day boundary crossing to set the EOD peak, then
        # check penalty on subsequent steps
        # Day 1: 100K, Day 2: varying drawdowns
        day1 = [100_000.0] * 24
        # Day 2: progressively worse drawdowns
        base_env_6pct = _MockCryptoEnv(
            obs_dim=5,
            pv_sequence=day1 + [94_000.0] * 24,  # 6% DD
            bar_interval_hours=1,
        )
        env_6pct = PropFirmWrapper(
            base_env_6pct,
            max_trailing_drawdown_pct=0.10,
            drawdown_penalty_start=0.05,
            drawdown_penalty_scale=5.0,
            profit_target_pct=1.0,
            max_daily_loss_pct=0.0,
        )
        env_6pct.reset()

        base_env_8pct = _MockCryptoEnv(
            obs_dim=5,
            pv_sequence=day1 + [92_000.0] * 24,  # 8% DD
            bar_interval_hours=1,
        )
        env_8pct = PropFirmWrapper(
            base_env_8pct,
            max_trailing_drawdown_pct=0.10,
            drawdown_penalty_start=0.05,
            drawdown_penalty_scale=5.0,
            profit_target_pct=1.0,
            max_daily_loss_pct=0.0,
        )
        env_8pct.reset()

        # Step through day 1
        for _ in range(24):
            env_6pct.step(np.array([0.0]))
            env_8pct.step(np.array([0.0]))

        # First step of day 2
        _, reward_6pct, _, _, _ = env_6pct.step(np.array([0.0]))
        _, reward_8pct, _, _, _ = env_8pct.step(np.array([0.0]))

        # 8% DD should have a more negative penalty than 6% DD
        assert reward_8pct < reward_6pct, (
            f"8% DD penalty ({reward_8pct}) should be more negative than 6% ({reward_6pct})"
        )


class TestInfoDict:
    """Test challenge info keys in step output."""

    def test_info_keys_present(self):
        """Info dict should contain prop firm tracking keys."""
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=[100_000.0] * 10)
        env = PropFirmWrapper(base_env, profit_target_pct=0.10)
        env.reset()

        _, _, _, _, info = env.step(np.array([0.0]))

        expected_keys = [
            "eod_drawdown",
            "eod_peak_equity",
            "cumulative_return",
            "profit_progress",
            "drawdown_budget_remaining",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"


class TestBaseEnvPassthrough:
    """Test that base env termination passes through correctly."""

    def test_base_terminated_passes_through(self):
        """If base env terminates, wrapper should pass it through."""
        # Only 2 steps in sequence, n_bars=2 → base env terminates at step 2
        base_env = _MockCryptoEnv(obs_dim=5, pv_sequence=[100_000.0, 100_000.0], n_bars=2)
        env = PropFirmWrapper(base_env, profit_target_pct=1.0)
        env.reset()

        env.step(np.array([0.0]))
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated
