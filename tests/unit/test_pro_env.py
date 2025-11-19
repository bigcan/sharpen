import numpy as np
import pytest

from finrl_pro.envs.pro_stock_env import ProStockEnv


def test_pro_env_shapes_and_step():
    T, stock_dim, tech_dim = 20, 3, 5
    price = np.random.rand(T, stock_dim).astype(np.float32) * 100 + 50
    tech = np.random.rand(T, stock_dim * tech_dim).astype(np.float32)
    turb = np.abs(np.random.randn(T).astype(np.float32))

    env = ProStockEnv(price_ary=price, tech_ary=tech, turbulence_ary=turb)
    assert env.state_dim == 1 + 2 + 3 * stock_dim + stock_dim * tech_dim
    s = env.reset()
    assert s.shape == (env.state_dim,)
    a = np.zeros(env.action_dim, dtype=np.float32)
    ns, r, done, info = env.step(a)
    assert ns.shape == (env.state_dim,)
    assert isinstance(r, float)

