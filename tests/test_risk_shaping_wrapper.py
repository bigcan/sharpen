"""Tests for RiskShapingWrapper — phase-invariant DD + daily-loss shaping.

Covers:
- Obs passthrough when augment_obs="off"
- Obs augmentation (+2 dims) when augment_obs="2d"
- Legacy 3d_legacy mode is explicitly rejected (ADR-1)
- No profit-target termination (unlike PropFirmWrapperV7)
- DD breach still terminates
- Daily-loss breach still terminates
- info dict contract (no profit_progress, no challenge_passed)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from finrl_pro_ds.envs.risk_shaping_wrapper import RiskShapingWrapper


# ---------------------------------------------------------------------------
# Mock envs (mirroring tests/test_prop_firm_wrapper_v7.py shape)
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
        self.step_idx = min(self.step_idx + 480, len(self.timestamps) - 1)


class MockFlatEnv(gym.Env):
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
        return np.zeros(self.observation_space.shape[0], dtype=np.float32), {
            "portfolio_value": self._equity,
        }

    def step(self, action):
        self._step += 1
        self.step_idx = min(self._step + 1, len(self.timestamps) - 1)
        return (
            np.zeros(self.observation_space.shape[0], dtype=np.float32),
            0.0,
            False,
            False,
            {"portfolio_value": self._equity},
        )

    def set_equity(self, val: float):
        self._equity = val

    def advance_day(self):
        self.step_idx = min(self.step_idx + 480, len(self.timestamps) - 1)


# ---------------------------------------------------------------------------
# Obs passthrough / augmentation
# ---------------------------------------------------------------------------

def test_obs_passthrough_dict():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, augment_obs="off")
    assert wrapped.observation_space["private"].shape == (5,)

    obs, _ = wrapped.reset()
    assert obs["private"].shape == (5,)
    for i in range(3):
        assert obs[f"scale_{i}"].shape == (30, 8)


def test_obs_passthrough_flat():
    base = MockFlatEnv(obs_dim=50)
    wrapped = RiskShapingWrapper(base, augment_obs="off")
    assert wrapped.observation_space.shape == (50,)

    obs, _ = wrapped.reset()
    assert obs.shape == (50,)


def test_augment_2d_dict():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, augment_obs="2d", max_trailing_drawdown_pct=0.10)
    assert wrapped.observation_space["private"].shape == (7,)

    obs, _ = wrapped.reset()
    assert obs["private"].shape == (7,)
    # Last 2 dims: dd_remaining, daily_remaining. At reset, equity==initial so both = 1.0
    np.testing.assert_allclose(obs["private"][5:], [1.0, 1.0], atol=1e-6)


def test_augment_2d_flat():
    base = MockFlatEnv(obs_dim=50)
    wrapped = RiskShapingWrapper(base, augment_obs="2d", max_trailing_drawdown_pct=0.10)
    assert wrapped.observation_space.shape == (52,)

    obs, _ = wrapped.reset()
    assert obs.shape == (52,)
    np.testing.assert_allclose(obs[-2:], [1.0, 1.0], atol=1e-6)


def test_rejects_3d_legacy():
    """ADR-1: 3d_legacy is explicitly unsupported."""
    base = MockFlatEnv(obs_dim=50)
    with pytest.raises(ValueError, match="3d_legacy"):
        RiskShapingWrapper(base, augment_obs="3d_legacy")  # type: ignore[arg-type]


def test_rejects_bool_augment_obs():
    """Old-style `True` should also fail — migration forces explicit enum."""
    base = MockFlatEnv(obs_dim=50)
    with pytest.raises(ValueError):
        RiskShapingWrapper(base, augment_obs=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No profit-target termination
# ---------------------------------------------------------------------------

def test_no_profit_termination_even_at_plus_15pct():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, max_trailing_drawdown_pct=0.10)
    wrapped.reset()

    # Drive equity to +15% — must NOT terminate
    base.set_equity(115_000.0)
    obs, reward, terminated, truncated, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert not terminated, "RiskShapingWrapper must not terminate on profit (no target)"
    assert not truncated
    assert info.get("cumulative_return") == pytest.approx(0.15)
    assert "challenge_passed" not in info, "no challenge_passed key — that's V7 only"
    assert info.get("prop_firm_termination") != "profit_target_reached"


def test_no_profit_progress_key():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base)
    wrapped.reset()
    base.set_equity(105_000.0)
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))
    # Parent wrapper does NOT emit profit_progress (only the V7 adapter does)
    assert "profit_progress" not in info


# ---------------------------------------------------------------------------
# DD + daily-loss terminations
# ---------------------------------------------------------------------------

def test_dd_breach_terminates():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, max_trailing_drawdown_pct=0.10, static_peak=True)
    wrapped.reset()

    base.set_equity(89_000.0)  # -11% from initial, breaches 10%
    _, _, terminated, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert terminated
    assert info["prop_firm_termination"] == "eod_trailing_drawdown"
    assert info["eod_drawdown"] > 0.10


def test_daily_loss_breach_terminates():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(
        base, max_trailing_drawdown_pct=0.20,  # loose DD so daily-loss trips first
        max_daily_loss_pct=0.05,
    )
    wrapped.reset()

    base.set_equity(94_000.0)  # -6% from start of day, breaches 5% daily
    _, _, terminated, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert terminated
    assert info["prop_firm_termination"] == "daily_loss_limit"


def test_dd_penalty_applied_below_termination():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(
        base,
        max_trailing_drawdown_pct=0.10,
        drawdown_penalty_start=0.05,
        drawdown_penalty_scale=5.0,
    )
    wrapped.reset()

    base.set_equity(92_000.0)  # -8%, above 5% penalty start, below 10% termination
    _, reward, terminated, _, _ = wrapped.step(np.array([0.0], dtype=np.float32))

    assert not terminated
    assert reward < 0.0, "DD proximity penalty should produce negative reward"


def test_daily_loss_penalty_applied():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(
        base,
        max_trailing_drawdown_pct=0.20,
        max_daily_loss_pct=0.05,
        daily_loss_penalty_start=0.02,
        daily_loss_penalty_scale=3.0,
    )
    wrapped.reset()

    base.set_equity(96_500.0)  # -3.5% daily, above 2% penalty start, below 5% term
    _, reward, terminated, _, _ = wrapped.step(np.array([0.0], dtype=np.float32))

    assert not terminated
    assert reward < 0.0


# ---------------------------------------------------------------------------
# info dict contract
# ---------------------------------------------------------------------------

def test_info_keys_present():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, max_trailing_drawdown_pct=0.10)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    for k in ("eod_drawdown", "eod_peak_equity", "cumulative_return",
              "drawdown_budget_remaining", "portfolio_value"):
        assert k in info, f"expected key {k} in info"


def test_cumulative_return_correct():
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base)
    wrapped.reset()

    base.set_equity(107_500.0)  # +7.5%
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))
    assert info["cumulative_return"] == pytest.approx(0.075)


def test_base_env_termination_passes_through():
    """If base env sets terminated=True on its own, wrapper must honor it."""
    class TerminatingBase(MockFlatEnv):
        def step(self, action):
            obs, r, _, trunc, info = super().step(action)
            return obs, r, True, trunc, info  # force terminated

    wrapped = RiskShapingWrapper(TerminatingBase(obs_dim=10))
    wrapped.reset()
    _, _, terminated, _, _ = wrapped.step(np.array([0.0], dtype=np.float32))
    assert terminated


def test_static_peak_locks_at_initial():
    """static_peak=True: peak stays at initial_capital even after gains + day crossing."""
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, static_peak=True)
    wrapped.reset()

    base.set_equity(120_000.0)
    wrapped.step(np.array([0.0], dtype=np.float32))
    base._step += 480  # cross day boundary
    base.set_equity(119_000.0)
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert info["eod_peak_equity"] == pytest.approx(100_000.0)


def test_trailing_peak_ratchets_on_eod():
    """static_peak=False: peak ratchets up at day boundary."""
    base = MockDictEnv(private_dim=5)
    wrapped = RiskShapingWrapper(base, static_peak=False)
    wrapped.reset()

    base.set_equity(110_000.0)
    wrapped.step(np.array([0.0], dtype=np.float32))
    # Cross day boundary by bumping internal step count past 1 day of bars
    base._step += 480
    base.set_equity(110_000.0)
    _, _, _, _, info = wrapped.step(np.array([0.0], dtype=np.float32))

    # Peak should have ratcheted up from 100K to 110K
    assert info["eod_peak_equity"] == pytest.approx(110_000.0)
