"""Tests for allocator_factory — the config→env-kwarg dispatch + frozen linear core.

The dispatch concern flagged in the Phase-4 resume note: the config ``env:`` block
keys do not all match the env constructor (``env.taker_fee`` → ``taker_fee_pct``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sharpen.envs import allocator_factory as factory

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "cross_asset_momentum.yaml").read_text(encoding="utf-8"))


def _arrays(T=300, N=3):
    rng = np.random.default_rng(3)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (T, N)), axis=0)
    tech_dim = 7
    return {
        "price_ary": price,
        "tech_ary": rng.normal(0, 1, (T, N * tech_dim)).astype(np.float32),
        "vol_ary": np.full((T, N), 0.10),
        "carry_ary": np.zeros((T, N)),
        "volume_ary": np.full((T, N), 1e9),
        "timestamps": (np.arange(T, dtype=np.int64) * 86400),
        "conviction_ary": np.sign(rng.normal(0, 1, (T, N))),
    }


# --------------------------------------------------------------------------- #
# config → kwarg mapping
# --------------------------------------------------------------------------- #
def test_taker_fee_renamed_to_taker_fee_pct():
    env = factory.make_allocator_env(_arrays(), CFG)
    assert env.taker_fee_pct == CFG["env"]["taker_fee"] == 0.0002


def test_all_mapped_keys():
    env = factory.make_allocator_env(_arrays(), CFG)
    e = CFG["env"]
    assert env.lev_cap == e["lev_cap"]
    assert env.max_gross_exposure == e["max_gross_exposure"]
    assert env.target_vol_asset == e["target_vol_asset"]
    assert env.reward_type == e["reward_type"]
    assert env.dsr_eta == e["dsr_eta"]
    assert env.turnover_penalty == e["turnover_penalty"]
    assert env.slippage_base_bps == e["slippage_base_bps"]
    assert env.slippage_impact_bps == e["slippage_impact_bps"]
    assert env.min_trade_pct == e["min_trade_pct"]
    assert env.circuit_breaker_threshold == e["circuit_breaker_threshold"]
    assert env.initial_capital == e["initial_capital"]
    assert env.allow_short is True


def test_overrides_take_precedence():
    env = factory.make_allocator_env(_arrays(), CFG, overrides={"turnover_penalty": 0.0077})
    assert env.turnover_penalty == 0.0077


def test_explicit_taker_fee_pct_wins_over_alias():
    cfg = {"env": {"taker_fee": 0.0002, "taker_fee_pct": 0.0009}}
    env = factory.make_allocator_env(_arrays(), cfg)
    assert env.taker_fee_pct == 0.0009


def test_eval_mode_forces_deterministic_start():
    cfg = {"env": {"random_start": True}}
    env = factory.make_allocator_env(_arrays(), cfg, eval_mode=True)
    assert env.random_start is False


def test_bool_coercion():
    cfg = {"env": {"allow_short": 0}}
    env = factory.make_allocator_env(_arrays(), cfg)
    assert env.allow_short is False


# --------------------------------------------------------------------------- #
# monthly rebalance conviction
# --------------------------------------------------------------------------- #
def test_monthly_rebal_changes_only_at_month_end():
    import pandas as pd
    T = 200
    ts = (pd.bdate_range("2020-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    conv = np.random.default_rng(1).normal(0, 1, (T, 4))
    out = factory.monthly_rebal_conviction(ts, conv)
    dates = pd.to_datetime(ts, unit="s")
    months = dates.to_period("M")
    last_idx = set(pd.Series(np.arange(T)).groupby(months.values).max().to_numpy().tolist())
    # The new monthly conviction takes effect AT the month-end index (decided at
    # month-end, governs the next move); it is held constant in between.
    changes = np.where(np.any(out[1:] != out[:-1], axis=1))[0] + 1  # indices where value changed
    for c in changes:
        assert c in last_idx, f"conviction changed at {c}, not a month-end"


# --------------------------------------------------------------------------- #
# frozen linear core eval
# --------------------------------------------------------------------------- #
def test_evaluate_linear_core_runs_full_episode():
    arrays = _arrays(T=400, N=3)
    m = factory.evaluate_linear_core(arrays, CFG)
    assert m["n_steps"] == arrays["price_ary"].shape[0] - 1
    for k in ("net_sharpe", "net_sortino", "total_return", "max_drawdown", "turnover_ann"):
        assert np.isfinite(m[k]), f"{k} not finite"


def test_evaluate_linear_core_requires_conviction_ary():
    arrays = _arrays()
    del arrays["conviction_ary"]
    with pytest.raises(KeyError):
        factory.evaluate_linear_core(arrays, CFG)


# --------------------------------------------------------------------------- #
# linear_core_weights — the paper executor's frozen-core weight source (ADR-7)
# --------------------------------------------------------------------------- #
def test_linear_core_weights_shape_and_finite():
    arrays = _arrays(T=400, N=3)
    w = factory.linear_core_weights(arrays, CFG)
    assert w.shape == (arrays["price_ary"].shape[0] - 1, 3)
    assert np.isfinite(w).all()
    cap = CFG["env"]["max_gross_exposure"]
    assert np.all(np.abs(w).sum(axis=1) <= cap + 1e-9), "gross exposure exceeds cap"


def test_linear_core_weights_requires_conviction_ary():
    arrays = _arrays()
    del arrays["conviction_ary"]
    with pytest.raises(KeyError):
        factory.linear_core_weights(arrays, CFG)


def test_linear_core_weights_match_independent_redrive():
    """Locks ADR-7 parity-by-construction: the weight trajectory equals an independent
    monthly-conviction drive of the same env (execution levers off), so the paper
    executor's targets are byte-identical to the evaluate_linear_core gate baseline."""
    arrays = _arrays(T=400, N=3)
    w = factory.linear_core_weights(arrays, CFG)
    conv_monthly = factory.monthly_rebal_conviction(arrays["timestamps"], arrays["conviction_ary"])
    env = factory.make_allocator_env(
        arrays, CFG,
        overrides={"no_trade_band": 0.0, "rebalance_interval": 1, "cost_penalty_scale": 0.0},
        eval_mode=True,
    )
    env.reset()
    manual = []
    done = False
    while not done:
        k = env.step_idx
        _, _, term, trunc, info = env.step(conv_monthly[k])
        manual.append(info["position"])
        done = term or trunc
    np.testing.assert_array_equal(w, np.asarray(manual, dtype=np.float64))


def test_linear_core_weights_rescale_daily_with_vol():
    """Chosen design (Option-2, daily vol-rescale): between month-ends the held
    conviction is RE-VOL-SCALED every bar, so the weight tracks target_vol/vol — it is
    NOT held constant (the monthly-hold alternative). Also pins the exact transform
    w[j] = clip(conv·clip(target_vol/vol[j], <=lev_cap), -lev_cap, lev_cap)."""
    import pandas as pd
    T = 70
    ts = (pd.bdate_range("2020-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    vol = (0.10 * (1.0 + 0.3 * (np.arange(T) % 2))).reshape(T, 1)   # alternates 0.10 / 0.13
    arrays = {
        "price_ary": np.full((T, 1), 100.0),
        "tech_ary": np.zeros((T, 7), dtype=np.float32),
        "vol_ary": vol,
        "carry_ary": np.zeros((T, 1)),
        "volume_ary": np.full((T, 1), 1e12),
        "timestamps": ts,
        "conviction_ary": np.ones((T, 1)),                          # constant long conviction
    }
    cfg = {"env": {"target_vol_asset": 0.10, "lev_cap": 2.0, "max_gross_exposure": 1e9,
                   "taker_fee": 0.0002, "allow_short": True}}
    w = factory.linear_core_weights(arrays, cfg)                    # (T-1, 1)
    j = 45                                                          # well past the first month-end
    assert w[j, 0] != w[j + 1, 0], "weight held constant between bars — daily vol-rescale missing"
    expected = min(0.10 / vol[j, 0], 2.0)                           # conviction = +1, gross non-binding
    np.testing.assert_allclose(w[j, 0], expected, rtol=1e-12)
