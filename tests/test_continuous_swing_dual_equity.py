"""N2 PF-XCHECK dual-equity tests.

Covers the env-side shadow accumulator (IC-N2) and the handler high/low
emission (IC-N1):

- SENS-5: close-marked trajectory is byte-identical with record_dual_equity
  on vs off (the shadow never perturbs dynamics).
- SENS-4: equity_mid is a write-only shadow (asserted via SENS-5 + info key).
- Lockstep: when (H+L)/2 == close, equity_mid == equity exactly.
- Gap mask: a gap-detected bar zeroes BOTH the close and the mid return.
- IC-N1: MultiScaleOHLCVHandler.step() emits high/low.
"""
import numpy as np
import pandas as pd
import pytest

from sharpen.envs.continuous_swing_env import ContinuousSwingEnv


class _FakeHandler:
    """Minimal MultiScaleOHLCVHandler stand-in emitting the N2 high/low keys."""

    def __init__(self, n_bars=120, n_features=8, zero_range=False, gap_bar=None):
        self.window_size = 10
        self._base_scale = 3
        self.scales = [3]
        self._ptr = self.window_size
        self._len = n_bars
        self.n_features = n_features
        rng = np.random.RandomState(7)
        self._features = (rng.randn(n_bars, n_features) * 0.1).astype(np.float32)
        self._base_close = 100.0 + np.cumsum(rng.randn(n_bars) * 0.2)
        if zero_range:
            self._base_high = self._base_close.copy()
            self._base_low = self._base_close.copy()
        else:
            # Asymmetric range so (H+L)/2 != close (an independent mid mark).
            up = 0.1 + rng.rand(n_bars) * 0.3
            down = 0.1 + rng.rand(n_bars) * 0.3
            self._base_high = self._base_close + up
            self._base_low = self._base_close - down
        if gap_bar is not None:
            # Inject a large (>> ATR) jump from gap_bar onward to trip gap detection.
            self._base_close[gap_bar:] += 50.0
            self._base_high[gap_bar:] += 50.0
            self._base_low[gap_bar:] += 50.0
        self._base_atr = np.ones(n_bars) * 0.5
        self._base_timestamps = np.arange(
            np.datetime64("2025-01-01T00:00"),
            np.datetime64("2025-01-01T00:00") + np.timedelta64(n_bars, "m"),
            np.timedelta64(1, "m"),
        )

    def reset(self):
        self._ptr = self.window_size

    def step(self):
        if self._ptr >= self._len:
            return None
        start = max(0, self._ptr - self.window_size + 1)
        window = self._features[start : self._ptr + 1]
        if len(window) < self.window_size:
            pad = np.tile(window[0:1], (self.window_size - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        result = {"scale_0": window.copy()}
        result["close"] = float(self._base_close[self._ptr])
        result["high"] = float(self._base_high[self._ptr])
        result["low"] = float(self._base_low[self._ptr])
        result["atr"] = float(self._base_atr[self._ptr])
        result["timestamp"] = self._base_timestamps[self._ptr]
        self._ptr += 1
        return result


def _make_env(record_dual_equity, *, zero_range=False, gap_bar=None, gap_detection=False):
    handler = _FakeHandler(zero_range=zero_range, gap_bar=gap_bar)
    config = {
        "initial_balance": 100000.0,
        "window_size": 10,
        "features_per_scale": 8,
        "scales": [3],
        "taker_fee": 0.0002,
        "slippage_base_bps": 0.0,
        "deadband_threshold": 0.1,
        "max_leverage": 1.0,
        "gap_detection": gap_detection,
        "gap_atr_mult": 3.0,
        "episode_length": 0,
        "random_start": False,
        "record_dual_equity": record_dual_equity,
        "reward": {"mode": "raw"},
    }
    return ContinuousSwingEnv(config=config, data_handler=handler)


def _roll(env, actions):
    """Drive a scripted action sequence; collect per-bar telemetry."""
    env.reset()
    pv, pv_mid, pos, rew, tc = [], [], [], [], []
    for a in actions:
        _obs, reward, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        pv.append(info["portfolio_value"])
        pv_mid.append(info["portfolio_value_mid"])
        pos.append(info["position"])
        rew.append(float(reward))
        tc.append(info["trade_count"])
        if term or trunc:
            break
    return {"pv": pv, "pv_mid": pv_mid, "pos": pos, "rew": rew, "tc": tc}


# Deterministic, deadband-crossing action sequence.
_ACTIONS = list(np.sin(np.linspace(0, 6 * np.pi, 100)).astype(np.float64))


def test_env_dual_equity_byte_identical_close_path():
    """SENS-5: enabling record_dual_equity must not change the close trajectory."""
    off = _roll(_make_env(False), _ACTIONS)
    on = _roll(_make_env(True), _ACTIONS)
    assert off["pv"] == pytest.approx(on["pv"], abs=0.0, rel=0.0)
    assert off["pos"] == pytest.approx(on["pos"], abs=0.0, rel=0.0)
    assert off["rew"] == pytest.approx(on["rew"], abs=0.0, rel=0.0)
    assert off["tc"] == on["tc"]


def test_env_dual_equity_info_key_present_only_when_recording():
    """SENS-4 surface: portfolio_value_mid is None when off, a float when on."""
    off = _roll(_make_env(False), _ACTIONS)
    on = _roll(_make_env(True), _ACTIONS)
    assert all(v is None for v in off["pv_mid"])
    assert all(isinstance(v, float) for v in on["pv_mid"])


def test_env_dual_equity_lockstep_when_mid_equals_close():
    """When (H+L)/2 == close every bar, equity_mid tracks equity exactly."""
    on = _roll(_make_env(True, zero_range=True), _ACTIONS)
    assert on["pv"] == pytest.approx(on["pv_mid"], rel=1e-12, abs=1e-9)


def test_env_dual_equity_diverges_with_real_range():
    """With a non-degenerate range, the mid curve generally differs from close."""
    on = _roll(_make_env(True), _ACTIONS)
    # At least one bar where the two marks disagree (sanity: the shadow is live).
    diffs = [abs(a - b) for a, b in zip(on["pv"], on["pv_mid"])]
    assert max(diffs) > 1e-6


def test_env_dual_equity_gap_masks_both_returns():
    """A gap-detected bar contributes ~0 (fee-only) to BOTH equity curves."""
    gap_bar = 60
    env = _make_env(True, gap_bar=gap_bar, gap_detection=True)
    env.reset()
    # Hold a full long position so a non-masked 50-pt jump would be huge.
    prev_pv = prev_pv_mid = env.initial_balance
    gap_step_pv_delta = gap_step_pv_mid_delta = None
    for _i in range(100):
        _o, _r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        pv, pv_mid = info["portfolio_value"], info["portfolio_value_mid"]
        # Detect the step whose bar index == gap_bar (handler ptr is 1-ahead).
        if env.handler._ptr - 1 == gap_bar:
            gap_step_pv_delta = abs(pv - prev_pv) / prev_pv
            gap_step_pv_mid_delta = abs(pv_mid - prev_pv_mid) / prev_pv_mid
        prev_pv, prev_pv_mid = pv, pv_mid
        if term or trunc:
            break
    assert gap_step_pv_delta is not None, "gap bar never reached"
    # Without masking, a 50/150 ≈ 33% jump would appear; masked → only fees (<1%).
    assert gap_step_pv_delta < 0.01
    assert gap_step_pv_mid_delta < 0.01


@pytest.fixture
def synthetic_parquet(tmp_path):
    n = 400
    ts = pd.date_range("2025-01-01", periods=n, freq="1min")
    rng = np.random.RandomState(3)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.1)
    high = close + np.abs(rng.randn(n)) * 0.2 + 0.05
    low = close - np.abs(rng.randn(n)) * 0.2 - 0.05
    open_ = close + rng.randn(n) * 0.05
    vol = np.abs(rng.randn(n)) * 1000 + 100
    df = pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )
    p = tmp_path / "synth.parquet"
    df.to_parquet(p)
    return str(p)


def test_handler_step_emits_high_low(synthetic_parquet):
    """IC-N1: handler.step() result carries high/low with low <= close <= high."""
    from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler

    h = MultiScaleOHLCVHandler(
        file_path=synthetic_parquet,
        ticker="TEST",
        feature_config={"scales": [3], "window_size": 10},
    )
    h.reset()
    r = h.step()
    assert r is not None
    assert "high" in r and "low" in r
    assert r["low"] <= r["close"] <= r["high"]
