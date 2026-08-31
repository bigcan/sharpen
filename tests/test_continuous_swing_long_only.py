"""V7 ContinuousSwingEnv long_only mode — NEGATIVE tests.

These are written to FAIL if shorting is ever reintroduced into the long-only path, which is
the only thing separating the gmgp1-spy variant B from variant A. Each clamp site in step()
gets its own test, because they are independent and a fix at one does not imply the others:

  1. the action clamp            (step: target_position = clip(raw, _action_low, 1.0))
  2. the ATR high-vol cap        (step: clip floor at 0.0 rather than -atr_cap_max_position)
  3. the position accumulator    (step: clip(pos, _position_floor, max_leverage))

A/B PARITY is asserted too: with a non-negative action sequence the two variants must produce
BYTE-IDENTICAL trajectories. That is what makes the head-to-head a test of the short
constraint rather than of two incidentally-different environments.
"""
import numpy as np
import pytest

from sharpen.envs.continuous_swing_env import ContinuousSwingEnv
from tests.test_continuous_swing_dual_equity import _FakeHandler


def _make_env(long_only, *, max_leverage=1.0, atr_cap_max_position=0.5,
              atr_cap_percentile=90, deadband=0.1):
    handler = _FakeHandler()
    config = {
        "initial_balance": 100000.0,
        "window_size": 10,
        "features_per_scale": 8,
        "scales": [3],
        "taker_fee": 0.0002,
        "slippage_base_bps": 0.0,
        "deadband_threshold": deadband,
        "max_leverage": max_leverage,
        "atr_cap_percentile": atr_cap_percentile,
        "atr_cap_max_position": atr_cap_max_position,
        "episode_length": 0,
        "random_start": False,
        "long_only": long_only,
        "reward": {"mode": "raw"},
    }
    return ContinuousSwingEnv(config=config, data_handler=handler)


def _roll(env, actions):
    env.reset()
    positions = []
    for a in actions:
        _obs, _r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        positions.append(info["position"])
        if term or trunc:
            break
    return positions


# Sweeps the full [-1, 1] range repeatedly and crosses the deadband in both directions.
_SWEEP = list(np.sin(np.linspace(0, 6 * np.pi, 100)).astype(np.float64))
# Strictly non-negative — the half of the action range the two variants must agree on.
_LONG_SIDE = [max(0.0, a) for a in _SWEEP]


def test_action_space_low_is_zero_when_long_only():
    assert _make_env(True).action_space.low[0] == pytest.approx(0.0)
    assert _make_env(False).action_space.low[0] == pytest.approx(-1.0)


def test_long_only_never_holds_a_short_position():
    """NEGATIVE: the whole point. A full two-sided sweep must never open a short."""
    pos = _roll(_make_env(True), _SWEEP)
    assert min(pos) >= 0.0, f"long_only opened a short: min position {min(pos)}"
    # Guard against the test passing vacuously by never trading at all.
    assert max(pos) > 0.1, "long_only never took a long position — test is vacuous"


def test_two_sided_control_does_short_on_the_same_sweep():
    """Control: the SAME sequence must short in the two-sided env, else the test above
    proves nothing about the clamp (it would just mean the actions never went short)."""
    pos = _roll(_make_env(False), _SWEEP)
    assert min(pos) < -0.1, "two-sided env never shorted — long_only test is not load-bearing"


def test_long_only_atr_cap_floors_at_flat_not_at_a_short():
    """NEGATIVE, clamp site 2. Force the ATR cap to bind on every bar (percentile 0 => every
    ATR exceeds the threshold). The cap must clamp toward flat, never into a short."""
    pos = _roll(_make_env(True, atr_cap_max_position=0.5, atr_cap_percentile=0), _SWEEP)
    assert min(pos) >= 0.0, f"ATR cap opened a short in long_only: {min(pos)}"
    ctrl = _roll(_make_env(False, atr_cap_max_position=0.5, atr_cap_percentile=0), _SWEEP)
    assert min(ctrl) < 0.0, "ATR-capped two-sided control never shorted — test not load-bearing"


@pytest.mark.parametrize("lev", [1.0, 2.0, 3.0])
def test_long_only_position_floor_holds_under_leverage(lev):
    """NEGATIVE, clamp site 3. _position_floor is derived from max_leverage; a regression that
    hardcoded -max_leverage would surface here and not at lev=1."""
    pos = _roll(_make_env(True, max_leverage=lev, deadband=0.05), _SWEEP)
    assert min(pos) >= 0.0, f"long_only breached the floor at leverage {lev}: {min(pos)}"
    assert max(pos) <= lev + 1e-9


def test_long_only_and_two_sided_are_identical_on_non_negative_actions():
    """A/B PARITY. Given actions that never ask to short, the constraint must be inert —
    identical trajectories. If this fails, the variants differ by more than the short ban and
    any performance gap between them is confounded."""
    a = _roll(_make_env(False), _LONG_SIDE)
    b = _roll(_make_env(True), _LONG_SIDE)
    assert a == pytest.approx(b, abs=0.0), "long_only perturbed the long-side trajectory"


def test_default_is_two_sided():
    """Absent config key => unchanged V7 behaviour (every existing config/checkpoint)."""
    handler = _FakeHandler()
    env = ContinuousSwingEnv(
        config={"window_size": 10, "features_per_scale": 8, "scales": [3],
                "episode_length": 0, "random_start": False, "reward": {"mode": "raw"}},
        data_handler=handler,
    )
    assert env.long_only is False
    assert env.action_space.low[0] == pytest.approx(-1.0)
