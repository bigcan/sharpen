"""TRAJ-TS-01 regression: trajectory timestamps must name the bar actually priced.

Rollout scripts stamped each trajectory row with
``base_timestamps[env.current_step - 1]``. ``current_step`` is an EPISODE counter
that starts at 0 on reset, while ``MultiScaleOHLCVHandler.reset`` puts the data
pointer at ``window_size`` — so every recorded timestamp was early by that
constant (measured at 32 bars on the gmgp1-btc canary folds). Metrics were
unaffected, but joining a trajectory to price data silently was not: the
risk-overlay lab had to recover the true alignment by FFT cross-correlation
before it could use those files at all.

``ContinuousSwingEnv.current_timestamp`` is now the single source of truth, and
these tests pin it to ground truth rather than to pointer arithmetic: the
timestamp must name the bar whose close the env actually priced.

Companion to ``tests/test_continuous_swing_causality.py`` (earn timing) and
``tests/data/test_multiscale_causality.py`` (coarse-bar X2 leak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler
from sharpen.envs.continuous_swing_env import ContinuousSwingEnv

_SCALES = [15, 60, 240]
_WINDOW = 30
_N_MIN = 12000
_SEED = 20260826


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    rng = np.random.RandomState(_SEED)
    incr = rng.normal(0.0, 5e-4, size=_N_MIN)
    close = 100.0 * np.exp(np.cumsum(incr))
    half = np.abs(rng.normal(0.0, 3e-4, size=_N_MIN)) * close + 1e-6
    open_ = close * np.exp(rng.normal(0.0, 2e-4, size=_N_MIN))
    p = tmp_path_factory.mktemp("traj_ts") / "synth.parquet"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=_N_MIN, freq="1min"),
        "open": open_,
        "high": np.maximum(close + half, np.maximum(open_, close)),
        "low": np.minimum(close - half, np.minimum(open_, close)),
        "close": close,
        "volume": np.abs(rng.normal(0.0, 1.0, size=_N_MIN)) * 100.0 + 10.0,
    }).to_parquet(p)
    return p


def _make_env(parquet) -> ContinuousSwingEnv:
    handler = MultiScaleOHLCVHandler(
        file_path=str(parquet), ticker="SYNTH",
        feature_config={"scales": _SCALES, "window_size": _WINDOW},
    )
    config = {
        "mdp_version": "v7", "initial_balance": 100000.0, "window_size": _WINDOW,
        "features_per_scale": 8, "scales": _SCALES,
        "taker_fee": 0.0, "slippage_base_bps": 0.0, "deadband_threshold": 0.0,
        "atr_cap_percentile": 100, "atr_cap_max_position": 1.0, "max_leverage": 1.0,
        "episode_length": 0, "random_start": False, "max_drawdown_pct": 1.0,
        "gap_detection": False, "reward": {"mode": "raw"},
    }
    return ContinuousSwingEnv(config=config, data_handler=handler)


class TestTrajectoryTimestamp:

    def test_timestamp_names_the_bar_actually_priced(self, parquet):
        """Ground truth: the stamped bar's close must be the close the env priced.

        Independent of pointer arithmetic -- if the index were off by any amount
        the closes would not match, because a random walk has effectively unique
        closes.
        """
        env = _make_env(parquet)
        env.reset()
        h = env.handler
        checked = 0
        for _ in range(200):
            _, _, term, trunc, _ = env.step(np.array([0.5], dtype=np.float32))
            if term or trunc:
                break
            ts = env.current_timestamp
            assert ts is not None
            idx = int(np.searchsorted(h._base_timestamps, ts))
            assert h._base_timestamps[idx] == ts
            assert h._base_close[idx] == pytest.approx(env.current_close), (
                "TRAJ-TS-01 regression: current_timestamp names a bar whose close "
                f"is {h._base_close[idx]}, but the env priced {env.current_close}"
            )
            checked += 1
        assert checked > 100, "test did not exercise enough steps"

    def test_episode_counter_form_is_wrong_and_stays_rejected(self, parquet):
        """The old `base_timestamps[current_step - 1]` form must NOT be correct.

        The handler resets its pointer to `window_size` and the env consumes one
        bar while priming its observation, so the counter form runs early by a
        fixed lead. The invariant worth pinning is that the lead is POSITIVE and
        CONSTANT -- constant is what made the bug survive review (trajectories
        looked internally consistent, just uniformly shifted), and positive is
        what makes the counter form wrong. The exact number is incidental to the
        reset convention and is deliberately not asserted.

        If this ever starts matching, the pointer convention has changed and the
        rollout scripts need re-verifying -- it is not a licence to go back to
        the counter.
        """
        env = _make_env(parquet)
        env.reset()
        h = env.handler

        offsets = []
        for _ in range(25):
            env.step(np.array([0.5], dtype=np.float32))
            correct = env.current_timestamp
            legacy = h._base_timestamps[env.current_step - 1]
            assert correct != legacy, (
                "the episode-counter form now agrees with the pointer form; the "
                "handler's reset convention has changed and the rollout scripts "
                "must be re-verified"
            )
            offsets.append(
                int(np.searchsorted(h._base_timestamps, correct)) - (env.current_step - 1)
            )

        assert len(set(offsets)) == 1, f"lead drifted across steps: {sorted(set(offsets))}"
        assert offsets[0] > 0, f"counter form should run EARLY, measured lead {offsets[0]}"

    def test_timestamps_advance_one_base_bar_per_step(self, parquet):
        """No duplicates, no gaps, no rewinds across a run."""
        env = _make_env(parquet)
        env.reset()
        seen = []
        for _ in range(120):
            _, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
            if term or trunc:
                break
            seen.append(env.current_timestamp)

        arr = pd.to_datetime(pd.Series(seen))
        deltas = arr.diff().dropna().unique()
        assert len(deltas) == 1, f"irregular timestamp spacing: {deltas}"
        assert deltas[0] == pd.Timedelta(minutes=min(_SCALES))

    def test_returns_none_rather_than_a_wrong_stamp(self, parquet):
        """With no handler the property yields None, never a fabricated index."""
        env = _make_env(parquet)
        env.reset()
        env.handler = None
        assert env.current_timestamp is None
