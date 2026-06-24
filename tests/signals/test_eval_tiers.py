"""Unit tests: Tier-2 capturability, Tier-3 robustness, Tier-5 orthogonality."""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.signals import Gates, SignalSpec
from finrl_pro_ds.signals.eval_harness import (
    FactorBook,
    _daily_ls_returns,
    _neutralized_eff,
    tier2_capturability,
    tier3_robustness,
    tier5_orthogonality,
)
from finrl_pro_ds.signals.features import Panel


class Trail:
    def __init__(self, k: int = 1) -> None:
        self.k = k
        self.spec = SignalSpec(name=f"trail{k}", hypothesis="trailing return predicts",
                               family="technical", expected_sign=1, horizons=(1, 5),
                               neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.log(panel.close)
        out = np.full(c.shape, np.nan)
        out[self.k:] = c[self.k:] - c[:-self.k]
        return out


def _momentum_panel(t: int = 600, n: int = 50, rho: float = 0.45, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    eps = 0.01 * rng.standard_normal((t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = rho * ret[i - 1] + eps[i]
    close = np.exp(np.cumsum(ret, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.0005 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.001 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.001 * rng.standard_normal((t, n))))
    vol = rng.uniform(1e5, 1e7, size=(t, n))
    dates = (np.datetime64("2012-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"M{i:03d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 5, size=n),
                 {"survivorship_free": False, "source": "synthetic"})


def _gates() -> Gates:
    return Gates.from_dict({"universe": {"min_names_per_day": 4},
                            "coverage": {"min_days": 150}})


# ---- Tier 2 ------------------------------------------------------------------

def test_tier2_positive_net_sharpe_on_momentum() -> None:
    p = _momentum_panel()
    cap = tier2_capturability(Trail(1), p, _gates(), neutralization=(), expected_sign=1,
                              hold_horizon=1)
    assert cap.frictionless_sharpe > 0.0
    assert cap.by_cost["standard"].net_sharpe > 0.0          # momentum survives modest cost
    assert cap.cost_wall >= 0.0                              # costs only reduce Sharpe


def test_tier2_cost_ordering() -> None:
    p = _momentum_panel()
    cap = tier2_capturability(Trail(1), p, _gates(), neutralization=(), expected_sign=1,
                              hold_horizon=5)
    f = cap.by_cost["frictionless"].net_sharpe
    s = cap.by_cost["standard"].net_sharpe
    h = cap.by_cost["harsh"].net_sharpe
    assert f >= s >= h                                       # more cost -> lower Sharpe


# ---- Tier 3 ------------------------------------------------------------------

def test_tier3_subperiod_stability() -> None:
    p = _momentum_panel()
    rob = tier3_robustness(Trail(1), p, _gates(), neutralization=(), expected_sign=1,
                           horizon=1, n_subperiods=4)
    assert rob.n_subperiods == 4
    assert len(rob.subperiod_ic_ir) == 4
    assert rob.min_subperiod_ic_ir > 0.0                    # works in every subperiod
    assert rob.recent_ic_ir > 0.0


# ---- Tier 5 ------------------------------------------------------------------

def test_tier5_self_factor_is_fully_explained() -> None:
    p = _momentum_panel()
    eff = _neutralized_eff(Trail(1), p, (), 1, 1)
    sd, sr = _daily_ls_returns(eff, p)
    book_self = FactorBook(sd, ("SELF",), sr.reshape(-1, 1))
    o = tier5_orthogonality(Trail(1), p, book_self, neutralization=(), expected_sign=1,
                            horizon=1)
    assert o.max_abs_corr > 0.99 and o.r2_explained > 0.99   # signal == the factor


def test_tier5_random_factor_is_orthogonal() -> None:
    p = _momentum_panel()
    eff = _neutralized_eff(Trail(1), p, (), 1, 1)
    sd, _ = _daily_ls_returns(eff, p)
    rng = np.random.default_rng(0)
    book_rand = FactorBook(sd, ("RND1", "RND2"), rng.standard_normal((len(sd), 2)))
    o = tier5_orthogonality(Trail(1), p, book_rand, neutralization=(), expected_sign=1,
                            horizon=1)
    assert o.max_abs_corr < 0.3 and o.r2_explained < 0.2     # unrelated -> orthogonal
    assert o.n_days == len(sd)
