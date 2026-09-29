"""Tripwires for the linear-core decision lead (``execution.decision_lead_bars``, R-3 fix).

The env fills a step-``k`` decision at close ``k+1``. The linear-core arrays are causal at
``t-1``, so the legacy drive (row ``k`` at step ``k``) fills the month-end rebalance at the
NEXT month's first close. The lead reads row ``k+1`` at step ``k`` so the fill lands on the
month-end close. Pinned in both directions:

  * timing: the lead fills the month-end rebalance AT the month-end close, the legacy drive
    one close later, and an absent key is byte-identical to the legacy drive;
  * causality (LEAK-2): the decision inputs built by the REAL momentum and BAB builders are
    unchanged when bars after ``t`` are bumped or bar ``t+1`` is crashed; and the check has
    teeth — a conviction that reads its own bar, falsely declared lag-1, is caught by it;
  * fail-closed: an undeclared or 0 cutoff lag is refused; the unconfirmed final bar is
    never acted on;
  * alpha alignment: under the lead the two-sleeve executor's combined weights change only on
    steps whose FILL bar is a month-end (no second trade the close after).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.data.cross_asset_loader import build_allocator_arrays, build_defensive_arrays
from sharpen.envs.allocator_factory import (
    _apply_decision_lead,
    decision_lead_bars,
    linear_core_trajectory,
    monthly_rebal_conviction,
)
from sharpen.features import cross_asset_signals as cas
from sharpen.paper import TwoSleeveExecutor

ROOT = Path(__file__).resolve().parents[2]


def _cfg(lead: int | None) -> dict:
    cfg: dict = {"env": {"min_trade_pct": 0.0}}
    if lead is not None:
        cfg["execution"] = {"decision_lead_bars": lead}
    return cfg


def _month_end_idx(ts: np.ndarray) -> np.ndarray:
    months = pd.to_datetime(ts, unit="s").to_period("M")
    return pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()


def _synthetic(T: int = 90, n: int = 2, *, lag: int | None = 1) -> dict:
    """Constant vol 0.10 (= target, so weight == held conviction); raw conviction +1 until
    the second month-end's month, -1 from then on — one flip, at a known month-end."""
    ts = (pd.bdate_range("2021-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    me = _month_end_idx(ts)
    conv = np.ones((T, n))
    conv[me[0] + 1:] = -1.0                  # held flips at the SECOND month-end, me[1]
    arrays = {
        "price_ary": np.full((T, n), 100.0), "tech_ary": conv.astype(np.float32),
        "vol_ary": np.full((T, n), 0.10), "carry_ary": np.zeros((T, n)),
        "volume_ary": np.full((T, n), 1e12), "timestamps": ts, "conviction_ary": conv,
        "assets": [f"a{i}" for i in range(n)], "vol_cutoff_lag": 1,
    }
    if lag is not None:
        arrays["conviction_cutoff_lag"] = lag
    return arrays


# ------------------------------------------------------------------------- timing
def test_lead_fills_the_month_end_rebalance_at_the_month_end_close():
    arrays = _synthetic()
    d = int(_month_end_idx(arrays["timestamps"])[1])        # month-end where held flips
    w_legacy = linear_core_trajectory(arrays, _cfg(0))["weights"]
    w_lead = linear_core_trajectory(arrays, _cfg(1))["weights"]
    # weights[k] fills at close k+1: the row filled AT the month-end close d is k = d-1.
    assert np.all(w_lead[d - 1] < 0) and np.all(w_lead[d - 2] > 0)
    assert np.all(w_legacy[d - 1] > 0) and np.all(w_legacy[d] < 0)   # one close later


def test_absent_key_is_the_legacy_drive_and_bad_values_raise():
    arrays = _synthetic()
    a = linear_core_trajectory(arrays, _cfg(None))
    b = linear_core_trajectory(arrays, _cfg(0))
    assert np.array_equal(a["weights"], b["weights"])
    assert np.array_equal(a["step_returns"], b["step_returns"])
    assert decision_lead_bars({}) == 0
    # the integer 0 or 1 only: int() would coerce True / 1.0 / "1" into a lead (Tier-2 N4)
    for bad in (2, -1, True, 1.0, 1.5, "1", None):
        with pytest.raises(ValueError, match="integer 0 or 1"):
            decision_lead_bars({"execution": {"decision_lead_bars": bad}})


@pytest.mark.parametrize("name", ["tailwind_v1.yaml", "tailwind_v1_challenge.yaml"])
def test_tailwind_configs_pin_the_lead(name):
    """Tier-2 N4: the TAILWIND book decides a bar early. Deleting or changing the key would
    silently swap the book every executor-path number describes (lead 0: DSR 0.930, not 0.967)."""
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    assert cfg["execution"]["decision_lead_bars"] == 1
    assert decision_lead_bars(cfg) == 1


# ------------------------------------------------------------------------- fail-closed
@pytest.mark.parametrize("lag", [None, 0])
def test_lead_refuses_an_unproven_conviction_cutoff(lag):
    with pytest.raises(ValueError, match="conviction_cutoff_lag"):
        linear_core_trajectory(_synthetic(lag=lag), _cfg(1))


def test_lead_refuses_an_unproven_vol_cutoff():
    arrays = _synthetic()
    del arrays["vol_cutoff_lag"]
    with pytest.raises(ValueError, match="vol_cutoff_lag"):
        linear_core_trajectory(arrays, _cfg(1))


def test_rl_gate_baseline_refuses_the_lead():
    """The RL-beats-linear baseline must share the policy's timing; a lead baseline would
    read one bar more than the policy it is scored against."""
    from sharpen.envs.allocator_factory import evaluate_linear_core, evaluate_linear_core_levered
    arrays = _synthetic()
    for fn in (evaluate_linear_core, evaluate_linear_core_levered):
        with pytest.raises(ValueError, match="executor setting"):
            fn(arrays, _cfg(1))
        assert fn(arrays, _cfg(0))["n_steps"] == len(arrays["timestamps"]) - 1


def test_lead_reads_vol_row_k_plus_1():
    """Pin the documented contract: at decision k the vol row read is k+1 (returns <= k)."""
    arrays = _synthetic()
    arrays["vol_ary"] = np.arange(arrays["vol_ary"].size, dtype=np.float64).reshape(
        arrays["vol_ary"].shape) + 1.0
    conv_m = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    lead_arrays, _ = _apply_decision_lead(arrays, conv_m)
    assert np.array_equal(lead_arrays["vol_ary"][:-1], arrays["vol_ary"][1:])
    assert arrays["vol_ary"] is not lead_arrays["vol_ary"]          # input not mutated


def test_lead_never_acts_on_the_unconfirmed_final_bar():
    arrays = _synthetic()
    base = linear_core_trajectory(arrays, _cfg(1))["weights"]
    flipped = dict(arrays)
    conv = arrays["conviction_ary"].copy()
    conv[-1] *= -1.0                        # the final bar is always flagged a "month-end"
    flipped["conviction_ary"] = conv
    assert np.array_equal(base, linear_core_trajectory(flipped, _cfg(1))["weights"])


# ------------------------------------------------------------------------- causality
_ASSETS = ["EQ1", "EQ2", "EQ3", "RT1", "RT2", "RT3"]
_CLASS = {"EQ1": "equity", "EQ2": "equity", "EQ3": "equity",
          "RT1": "rates", "RT2": "rates", "RT3": "rates"}


def _panel(T: int = 700, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=T)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(3e-4, 0.01, (T, 6)), axis=0),
                         index=idx, columns=_ASSETS)
    volume = pd.DataFrame(1e6, index=idx, columns=_ASSETS)
    return close, volume


def _lead_inputs(close: pd.DataFrame, volume: pd.DataFrame, sleeve: str) -> tuple:
    start, end = close.index[0], close.index[-1]
    if sleeve == "momentum":
        arrays = build_allocator_arrays(cas.compute(close), close, volume, _ASSETS, start, end)
    else:
        arrays = build_defensive_arrays(close, volume, _ASSETS, _CLASS, start, end,
                                        beta_window=60, min_periods=30)
    conv_m = monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    lead_arrays, conv_lead = _apply_decision_lead(arrays, conv_m)
    return lead_arrays["vol_ary"], conv_lead


def _t_before_month_end(close: pd.DataFrame) -> int:
    """A decision bar t whose NEXT bar is a month-end: the lead reads that month-end's
    conviction row at step t, the strongest case for a same-bar read to show up."""
    ts = (close.index.asi8 // 10**9).astype(np.int64)
    return int(next(m for m in _month_end_idx(ts) if m > 450) - 1)


@pytest.mark.parametrize("sleeve", ["momentum", "defensive"])
def test_lead_decision_inputs_ignore_the_fill_bar_and_the_future(sleeve):
    close, volume = _panel()
    t = _t_before_month_end(close)
    vol0, conv0 = _lead_inputs(close, volume, sleeve)
    future = close.copy()
    future.iloc[t + 1:] *= 1.5                               # every bar after decision bar t
    crash = close.copy()
    crash.iloc[t + 1] *= 0.01                                # the fill bar of decision t
    for label, perturbed in (("future", future), ("fill-bar crash", crash)):
        vol1, conv1 = _lead_inputs(perturbed, volume, sleeve)
        assert np.allclose(conv0[:t + 1], conv1[:t + 1], atol=1e-12), label
        assert np.allclose(vol0[:t + 1], vol1[:t + 1], atol=1e-12), label


def test_the_causality_check_catches_a_same_bar_signal():
    """Teeth: a conviction reading its own bar, falsely declared lag-1, moves the decision
    at t when the fill bar t+1 is crashed — the check above would fail on it."""
    close, volume = _panel()
    t = _t_before_month_end(close)

    def lead_conv(c: pd.DataFrame) -> np.ndarray:
        same_bar = np.sign(c.pct_change().fillna(0.0)).to_numpy()   # reads bar t itself
        arrays = build_allocator_arrays(cas.compute(c), c, volume, _ASSETS,
                                        c.index[0], c.index[-1])
        arrays["conviction_ary"] = same_bar                  # declared lag stays 1: a lie
        conv_m = monthly_rebal_conviction(arrays["timestamps"], same_bar)
        return _apply_decision_lead(arrays, conv_m)[1]

    crash = close.copy()
    crash.iloc[t + 1] *= 0.01
    assert not np.allclose(lead_conv(close)[t], lead_conv(crash)[t])


# ------------------------------------------------------------------------- alpha alignment
def _two_sleeve_bundle(T: int = 400, n: int = 4, seed: int = 11) -> dict:
    """Constant vol and month-held convictions, so combined weights change only when a
    sleeve's conviction or the risk-parity alpha switches."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2019-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    price = 100 * np.cumprod(1 + rng.normal(3e-4, 0.01, (T, n)), axis=0)
    names = [f"A{i}" for i in range(n)]

    def sleeve(conv: np.ndarray) -> dict:
        return {"price_ary": price.copy(), "tech_ary": conv.astype(np.float32),
                "vol_ary": np.full((T, n), 0.10), "carry_ary": np.zeros((T, n)),
                "volume_ary": np.full((T, n), 1e12), "timestamps": ts, "conviction_ary": conv,
                "assets": names, "conviction_cutoff_lag": 1, "vol_cutoff_lag": 1}

    union = {"price_ary": price.copy(), "volume_ary": np.full((T, n), 1e12),
             "carry_ary": np.zeros((T, n)), "timestamps": ts, "assets": names}
    return {"momentum": sleeve(np.sign(rng.normal(0, 1, (T, n)))),
            "defensive": sleeve(np.tanh(rng.normal(0, 1, (T, n)))), "union": union}


@pytest.mark.parametrize("lead", [0, 1])
def test_combined_weights_change_only_on_the_rebalance_fill_step(lead):
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("execution", {})["decision_lead_bars"] = lead
    cfg["env"]["min_trade_pct"] = 0.0
    bundle = _two_sleeve_bundle()
    _, detail = TwoSleeveExecutor(cfg).sim_oracle(bundle)
    w = detail["combined_w"]
    changed = set((np.flatnonzero(np.abs(np.diff(w, axis=0)).sum(axis=1) > 1e-12) + 1).tolist())
    # Pre-existing, out of scope: `_monthly_held` flags a window's LAST row as a month-end
    # (the P2-01 analogue for alpha), so alpha may switch on the final step under either
    # setting. Only interior month-ends are the calendar under test here.
    changed.discard(w.shape[0] - 1)
    me = _month_end_idx(bundle["union"]["timestamps"])[:-1]    # drop the final (unconfirmed) bar
    rebalance_steps = {int(d) - 1 for d in me} if lead == 1 else {int(d) for d in me}
    assert changed, "nothing ever rebalanced — the test would be vacuous"
    assert changed <= rebalance_steps, sorted(changed - rebalance_steps)[:5]
