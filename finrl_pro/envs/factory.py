"""Environment factory helpers for FinRL Pro (dynamic env)."""

from __future__ import annotations

from finrl_pro.data.loader_pro import Assembly
from finrl_pro.envs.pro_stock_env import ProStockEnv


def make_pro_env(
    asm: Assembly,
    *,
    gamma: float = 0.999,
    turbulence_thresh: float = 30.0,
    min_stock_rate: float = 0.1,
    max_stock: float = 1e2,
    initial_capital: float = 1e6,
    buy_cost_pct: float = 1e-3,
    sell_cost_pct: float = 1e-3,
    reward_scaling: float = 2 ** -13,
) -> ProStockEnv:
    return ProStockEnv(
        price_ary=asm.price_ary,
        tech_ary=asm.tech_ary,
        turbulence_ary=asm.turbulence_ary,
        gamma=gamma,
        turbulence_thresh=turbulence_thresh,
        min_stock_rate=min_stock_rate,
        max_stock=max_stock,
        initial_capital=initial_capital,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
    )

