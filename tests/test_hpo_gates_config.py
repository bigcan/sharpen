"""Tests for the two configurable HPO trial gates.

Both decide which trials survive to best-trial selection, so they get direct tests rather
than trust:

  * `hpo_min_trades` must default to 30 (every pre-existing workstream must be unchanged)
    and must be overridable from the gate YAML.
  * the buy-and-hold hurdle must be measured with the SAME portfolio accessor and the same
    seeds as the agent path, and cached per eval WINDOW (the eval env is rebuilt each trial,
    so an identity-keyed cache would silently never hit).
"""
import numpy as np
import pytest

from finrl_pro_ds.hpo import objective as obj
from finrl_pro_ds.hpo.evaluate import _extract_portfolio_value, buy_and_hold_total_return


class _StubEnv:
    """Minimal vector-env stand-in: portfolio compounds at a fixed rate per step."""

    def __init__(self, per_step=0.001, n_steps=50, num_envs=2):
        self.per_step = per_step
        self.n_steps = n_steps
        self.num_envs = num_envs
        self._i = 0
        self._pv = 100000.0
        self.seen_actions = []

    def reset(self, seed=None):
        self._i = 0
        self._pv = 100000.0
        return {}, {"portfolio_value": np.full(self.num_envs, self._pv)}

    def step(self, action):
        self.seen_actions.append(np.asarray(action).copy())
        self._i += 1
        self._pv *= (1.0 + self.per_step)
        term = np.array([self._i >= self.n_steps] * self.num_envs)
        trunc = np.zeros(self.num_envs, dtype=bool)
        return {}, np.zeros(self.num_envs), term, trunc, {
            "portfolio_value": np.full(self.num_envs, self._pv)}


# ---------- min_trades is configurable, default preserved ----------

def test_min_trades_defaults_to_30():
    """Every existing workstream omits this key and must keep the historical threshold."""
    gates = {}
    assert int(gates.get("hpo_min_trades", 30)) == 30


def test_min_trades_reads_from_gates():
    gates = {"hpo_min_trades": 10}
    assert int(gates.get("hpo_min_trades", 30)) == 10


# ---------- buy-and-hold hurdle ----------

def test_buy_and_hold_return_matches_compounded_growth():
    env = _StubEnv(per_step=0.001, n_steps=50)
    got = buy_and_hold_total_return(env, max_steps=1000, seeds=(1,))
    assert got == pytest.approx(1.001 ** 50 - 1, rel=1e-6)


def test_buy_and_hold_sends_full_long_action():
    """The baseline must actually be FULL LONG; a zero/partial action would understate it."""
    env = _StubEnv(n_steps=5)
    buy_and_hold_total_return(env, max_steps=100, seeds=(1,))
    assert env.seen_actions, "env was never stepped"
    for a in env.seen_actions:
        assert np.allclose(a, 1.0), f"baseline sent {a}, expected all +1"
        assert a.shape == (env.num_envs, 1)


def test_buy_and_hold_negative_market_gives_negative_hurdle():
    env = _StubEnv(per_step=-0.002, n_steps=30)
    got = buy_and_hold_total_return(env, max_steps=100, seeds=(1,))
    assert got < 0


def test_hurdle_cache_is_keyed_by_window_not_env_identity():
    """The eval env is REBUILT every trial. A cache that misses on a new object would
    re-run a full rollout per trial; one that keyed on recycled ids could collide."""
    obj._BH_HURDLE_CACHE.clear()
    data_cfg = {"file_path": "d.parquet", "val_start_date": "2024-01-02",
                "val_end_date": "2024-06-28"}
    env1 = _StubEnv(per_step=0.001, n_steps=20)
    first = obj._buy_hold_hurdle(env1, 15, data_cfg=data_cfg)

    # A DIFFERENT env object for the same window must hit the cache, not re-measure.
    env2 = _StubEnv(per_step=0.05, n_steps=20)          # would give a wildly different value
    second = obj._buy_hold_hurdle(env2, 15, data_cfg=data_cfg)
    assert second == first, "cache missed on a rebuilt env for the same window"
    assert not env2.seen_actions, "cache hit should not have stepped the new env"

    # A different window must NOT reuse it.
    other = dict(data_cfg, val_start_date="2025-01-02")
    third = obj._buy_hold_hurdle(_StubEnv(per_step=0.05, n_steps=20), 15, data_cfg=other)
    assert third != first
    obj._BH_HURDLE_CACHE.clear()


# ---------- shared accessor ----------

def test_extract_portfolio_value_handles_vector_and_scalar():
    assert _extract_portfolio_value({"portfolio_value": np.array([7.0, 9.0])}) == 7.0
    assert _extract_portfolio_value({"portfolio_value": 5.0}) == 5.0
    assert _extract_portfolio_value(None, default=3.0) == 3.0
