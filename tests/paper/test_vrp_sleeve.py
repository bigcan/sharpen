"""VRPSleeve (return-stream sleeve) + calendar-resample tests (step 3).

The calendar resample (native 24/7 crypto → portfolio business-day) is the #1 bug
surface (MS-ADR-3): the VRP panel is epoch-MILLISECONDS, the portfolio calendar is
epoch-SECONDS, and weekend P&L must mark at the next business day. These tests pin the
telescoping-compound invariant, the weekend lumping, the ms→s conversion, and the
load-bearing forward look-ahead tripwire — all on a synthetic panel (no network).

Spec: ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md`` (MS-ADR-3/10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.crypto.data.options_array_builder import OptionsPanels
from sharpen.paper import SleeveContext, VRPSleeve, resample_returns_to_calendar


# --------------------------------------------------------------------------- #
# resample_returns_to_calendar — the MS-ADR-3 core
# --------------------------------------------------------------------------- #
def test_resample_telescoping_invariant():
    """Σ-compounded resampled return == native compound over the covered span (telescopes)."""
    dates = pd.date_range("2021-03-01", periods=60, freq="D", tz="UTC")
    native_ts = (dates.asi8 // 10**9).astype(np.int64)
    rng = np.random.default_rng(1)
    native_ret = rng.normal(0.001, 0.02, 60)
    target = (pd.bdate_range("2021-03-01", periods=40, tz="UTC").asi8 // 10**9).astype(np.int64)
    ts, ret = resample_returns_to_calendar(native_ts, native_ret, target)
    assert ts.size > 0
    n = int(np.searchsorted(native_ts, int(ts[-1]), side="right"))
    expected = float(np.prod(1.0 + native_ret[:n]) - 1.0)
    assert abs(float(np.prod(1.0 + ret) - 1.0) - expected) < 1e-12


def test_resample_weekend_lumps_into_monday():
    """Sat+Sun native P&L compounds into the Monday business mark (the next reconciliation)."""
    fri = pd.Timestamp("2021-03-05", tz="UTC")          # a Friday
    native = pd.date_range(fri, periods=4, freq="D", tz="UTC")   # Fri, Sat, Sun, Mon
    native_ts = (native.asi8 // 10**9).astype(np.int64)
    native_ret = np.array([0.01, 0.02, 0.03, 0.04])
    target = np.array([native_ts[0], native_ts[3]], dtype=np.int64)   # Fri, Mon
    ts, ret = resample_returns_to_calendar(native_ts, native_ret, target)
    np.testing.assert_array_equal(ts, target)
    # Fri = just Fri (0.01); Mon = Sat·Sun·Mon compounded.
    expected_mon = (1.02 * 1.03 * 1.04) - 1.0
    np.testing.assert_allclose(ret, [0.01, expected_mon], atol=1e-12)


def test_resample_drops_uncovered_target_bars():
    """Target bars outside native coverage are dropped (not zero-filled — MS-ADR-3)."""
    native_ts = (pd.date_range("2021-06-01", periods=30, freq="D", tz="UTC").asi8 // 10**9).astype(np.int64)
    native_ret = np.full(30, 0.001)
    # target spans well before AND after native coverage
    target = (pd.bdate_range("2021-01-01", periods=300, tz="UTC").asi8 // 10**9).astype(np.int64)
    ts, _ = resample_returns_to_calendar(native_ts, native_ret, target)
    assert ts.min() >= native_ts[0] and ts.max() <= native_ts[-1]


def test_resample_empty_when_no_overlap():
    native_ts = (pd.date_range("2021-06-01", periods=5, freq="D", tz="UTC").asi8 // 10**9).astype(np.int64)
    target = (pd.bdate_range("2025-01-01", periods=5, tz="UTC").asi8 // 10**9).astype(np.int64)
    ts, ret = resample_returns_to_calendar(native_ts, np.zeros(5), target)
    assert ts.size == 0 and ret.size == 0


# --------------------------------------------------------------------------- #
# VRPSleeve on a synthetic panel
# --------------------------------------------------------------------------- #
def _synth_panel(T: int = 220, seed: int = 3) -> OptionsPanels:
    """Minimal BTC OptionsPanels with epoch-MILLISECOND timestamps (build_panels convention)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=T, freq="D", tz="UTC")   # 24/7 crypto
    ts_ms = (dates.asi8 // 1_000_000).astype(np.int64)
    spot = (30000.0 * np.cumprod(1 + rng.normal(0.0003, 0.03, T)))[:, None]
    iv = np.clip(0.6 + rng.normal(0, 0.05, T), 0.2, 1.5)[:, None]
    funding = np.full((T, 1), 0.0001)
    return OptionsPanels(
        assets=["BTC"], timestamps=ts_ms, spot_ary=spot, iv_ary=iv,
        iv_rv_spread_ary=np.zeros((T, 1)), rv_ary={30: np.zeros((T, 1))}, funding_ary=funding)


def _cfg(instrument: str = "naked_straddle") -> dict:
    return {"sleeves": {"vrp": {"asset": "BTC",
                                "sim": {"premium_frac": 0.05, "roll_days": 21, "instrument": instrument}}}}


def _target_calendar(T: int = 150) -> np.ndarray:
    return (pd.bdate_range("2021-01-04", periods=T, tz="UTC").asi8 // 10**9).astype(np.int64)


def _ctx(panel=None, target=None) -> SleeveContext:
    return SleeveContext(data={"panel": panel or _synth_panel()},
                         union_timestamps=target if target is not None else _target_calendar())


def test_vrp_sleeve_oracle_stream():
    s = VRPSleeve(_cfg()).sim_oracle(_ctx())
    assert s.union_weights is None and not s.is_allocator
    assert s.n_steps > 0 and np.isfinite(s.step_returns).all()
    assert "crypto_vol" in s.class_pnl
    for k in ("cvar95_pct_daily", "cvar99_pct_daily", "sortino", "max_dd_pct", "skew"):
        assert k in s.risk_extra
    assert s.risk_extra["instrument"] == "naked_straddle"


def test_vrp_sleeve_ms_to_seconds_alignment():
    """Emitted stamps are epoch-SECONDS aligned to the business-day target (NOT raw ms)."""
    panel, target = _synth_panel(), _target_calendar()
    s = VRPSleeve(_cfg()).sim_oracle(_ctx(panel, target))
    assert set(s.timestamps.tolist()).issubset(set(target.tolist()))
    assert s.timestamps[0] >= int(panel.timestamps[0] // 1000)        # within native coverage (seconds)
    assert s.timestamps[-1] <= int(panel.timestamps[-1] // 1000)


def test_vrp_forward_safe_matches_oracle():
    """The sim is causal/online ⇒ the SAFE forward recompute reproduces the oracle exactly."""
    s, ctx = VRPSleeve(_cfg()), _ctx()
    o, f = s.sim_oracle(ctx), s.forward_recompute(ctx)
    np.testing.assert_array_equal(o.timestamps, f.timestamps)
    np.testing.assert_array_equal(o.step_returns, f.step_returns)


def test_vrp_forward_lookahead_caught():
    """A panel look-ahead (use tomorrow's IV today) moves the stream — the load-bearing
    tripwire (the executor's daily_return_te_bps catches this on the combined book)."""
    s, ctx = VRPSleeve(_cfg()), _ctx()
    o = s.sim_oracle(ctx)

    def iv_lookahead(spot, iv, funding):
        iv2 = iv.copy()
        iv2[:-1] = iv[1:]                      # tomorrow's IV leaked into today's pricing
        return spot, iv2, funding

    bug = s.forward_recompute(ctx, broken_assembler=iv_lookahead)
    assert np.abs(bug.step_returns - o.step_returns).max() > 1e-6


def test_vrp_instrument_knob_rejects_ungated():
    """MS-ADR-10 guardrail: only naked_straddle is gated; iron_fly/strangle are rejected."""
    with pytest.raises(NotImplementedError, match="iron_fly"):
        VRPSleeve(_cfg(instrument="iron_fly"))
    with pytest.raises(NotImplementedError, match="strangle"):
        VRPSleeve(_cfg(instrument="strangle"))


def test_vrp_standalone_native_calendar():
    """With no union calendar the sleeve emits on its own (seconds) calendar (executor
    normally supplies the business-day target)."""
    panel = _synth_panel(T=80)
    s = VRPSleeve(_cfg()).sim_oracle(SleeveContext(data={"panel": panel}, union_timestamps=None))
    assert s.n_steps == 80
    np.testing.assert_array_equal(s.timestamps, np.asarray(panel.timestamps, np.int64) // 1000)
