"""KEYSTONE: MultiAssetAllocatorEnv reproduces the validated linear core.

The cross-asset pivot earns an RL build only if the env's accounting + cost stack
faithfully reproduces the independently-validated linear TSMOM book (net Sharpe ~0.60
@2bps; ``results/xsec_momentum/``). This test drives the env as a ZERO-RL allocator —
fed the falsification's own daily weights — and asserts its net Sharpe matches the
linear core. If it does, any later RL gain over that core is REAL, not a sim artifact;
the RL-beats-linear gate is well-posed.

Mechanics (why this is a fair reproduction):
  - Identity vol-scaling: ``vol_ary = target_vol_asset`` ⇒ env scale = 1, so the env
    weight equals the conviction we feed (the vol-scaling math itself is covered by
    ``tests/envs/test_allocator_action.py``; here we isolate accounting + cost).
  - We feed ``conviction[k] = w_daily[k+1] / lev_cap``. The ``/lev_cap`` keeps conviction
    in [-1, 1] (Sharpe is leverage-invariant, INCLUDING the proportional turnover cost,
    so the constant 1/lev_cap factor cancels). The ``+1`` index aligns the env's
    internal T+1 execution lag (a position held during the move t→t+1 was set at the
    prior decision) with the falsification's ``w_eff = w_daily.shift(1)``.
  - 2bps taker fee, zero slippage, non-binding gross cap, no dust filter, no turnover
    penalty, circuit breaker off — matching the falsification's flat ``Σ|Δw|·2bps`` cost.

The residual gap (env fixed-entry-notional hold vs the falsification's constant-weight
daily rebalance) is second-order, absorbed by the ±0.05 tolerance (ADR / Test Plan).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from finrl_pro_ds.features import cross_asset_signals as cas

ROOT = Path(__file__).resolve().parents[2]
PRICE_CACHE = ROOT / "results" / "xsec_momentum" / "prices_daily.parquet"
LINEAR_CORE_NET_SHARPE = 0.601     # results/xsec_momentum/ verdict (Phase-1 parity target)
LEV_CAP = 2.0
COST = 0.0002                      # 2bps standard
ANN = 252

UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
CLASS_OF = {t: c for c, v in UNIVERSE.items() for t in v}


def _last_trading_of_month(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    key = index.to_period("M")
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(key).max().values)


def _sharpe(series: np.ndarray) -> float:
    s = series.std()
    return float(series.mean() / s * np.sqrt(ANN)) if s > 0 else 0.0


def _falsification_net_sharpe(weights_rebal: pd.DataFrame, rets: pd.DataFrame) -> float:
    """Reference linear-core net Sharpe (replicates xsec_momentum_falsification.backtest)."""
    daily = rets.index
    w_daily = weights_rebal.reindex(daily).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)
    gross = (w_eff * rets).sum(axis=1)
    dw = weights_rebal.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = weights_rebal.iloc[0].abs().sum()
    dw_daily = dw.reindex(daily).fillna(0.0)
    net = (gross - dw_daily * COST).dropna().to_numpy()
    return _sharpe(net)


def _env_net_sharpe(close: pd.DataFrame, w_daily: pd.DataFrame) -> tuple[float, int]:
    """Drive the env as a zero-RL allocator fed the falsification's daily weights."""
    T, n = close.shape
    arrays = dict(
        price_ary=close.to_numpy(dtype=np.float64),
        tech_ary=np.zeros((T, n), dtype=np.float32),
        vol_ary=np.full((T, n), 0.10),                  # identity vol-scaling
        carry_ary=np.zeros((T, n), dtype=np.float64),
        volume_ary=np.full((T, n), 1e15),               # huge ⇒ ~0 impact slippage
        timestamps=(np.arange(T, dtype=np.int64) * 86400),
    )
    env = MultiAssetAllocatorEnv(
        **arrays, target_vol_asset=0.10, lev_cap=LEV_CAP, max_gross_exposure=1e9,
        taker_fee_pct=COST, slippage_base_bps=0.0, slippage_impact_bps=0.0,
        min_trade_pct=0.0, turnover_penalty=0.0, reward_type="simple",
        circuit_breaker_threshold=0.0, initial_capital=100_000.0,
    )
    conviction = w_daily.to_numpy() / LEV_CAP           # (T, N) ∈ [-1, 1]
    env.reset()
    step_returns: list[float] = []
    for k in range(T - 1):
        idx = min(k + 1, T - 1)                          # T+1-lag alignment
        _, _, terminated, truncated, info = env.step(conviction[idx])
        step_returns.append(info["step_return"])
        if terminated or truncated:
            break
    return _sharpe(np.array(step_returns)), len(step_returns)


@pytest.mark.skipif(not PRICE_CACHE.exists(),
                    reason="cached falsification prices absent (run xsec_momentum_falsification.py)")
def test_env_reproduces_linear_core_net_sharpe():
    close = pd.read_parquet(PRICE_CACHE)
    close = close[[t for t in CLASS_OF if t in close.columns]].sort_index()
    rets = close.pct_change()

    long = cas.compute(close, asset_class=CLASS_OF)
    bw = long.pivot(index="date", columns="ticker", values="baseline_weight").reindex(columns=close.columns)

    warmup = max(cas.DEFAULT_LOOKBACKS) + cas.DEFAULT_SKIP + cas.DEFAULT_VOL_WINDOW
    rebal = _last_trading_of_month(close.index)
    rebal = rebal[rebal >= close.index[warmup]]
    weights_rebal = bw.loc[rebal].fillna(0.0)
    w_daily = weights_rebal.reindex(close.index).ffill().fillna(0.0)

    reference = _falsification_net_sharpe(weights_rebal, rets)
    env_sharpe, n_steps = _env_net_sharpe(close, w_daily)

    assert n_steps == len(close) - 1, "env terminated early — accounting/liquidation bug"
    # Parity vs the independently-recomputed linear book (the real claim).
    assert abs(env_sharpe - reference) < 0.05, (
        f"env net Sharpe {env_sharpe:.3f} drifted from the linear core "
        f"{reference:.3f} — the env accounting/cost stack does NOT reproduce the "
        f"validated book; any RL 'gain' would be a sim artifact."
    )
    # Anchor to the documented linear-core verdict (~0.60).
    assert abs(env_sharpe - LINEAR_CORE_NET_SHARPE) < 0.06, (
        f"env net Sharpe {env_sharpe:.3f} is far from the documented linear core "
        f"{LINEAR_CORE_NET_SHARPE}."
    )
