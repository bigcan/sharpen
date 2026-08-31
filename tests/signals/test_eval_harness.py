"""Unit tests: Tier-0 causality gate (with leaky negative control) and Tier-1 power."""
from __future__ import annotations

import numpy as np

from sharpen.signals import Gates, SignalSpec, evaluate_signal, make_synthetic_panel
from sharpen.signals.eval_harness import (
    assert_causal,
    tier0_hygiene,
    tier1_gross_power,
)
from sharpen.signals.features import Panel


# ---- candidate signals (implement the Signal protocol) ----------------------

class CausalTrailing:
    """k-day trailing log-return — uses only data <= t (causal)."""

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


class LeakyForward:
    """Forward 1-day return — peeks at close[t+1] (a deliberate LEAK / negative control)."""

    def __init__(self) -> None:
        self.spec = SignalSpec(name="leaky", hypothesis="peeks ahead", family="technical",
                               expected_sign=1, horizons=(1,), neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        c = panel.close
        out = np.full(c.shape, np.nan)
        out[:-1] = c[1:] / c[:-1] - 1.0
        return out


def _momentum_panel(t: int = 400, n: int = 40, rho: float = 0.4, seed: int = 0) -> Panel:
    """Panel with AR(1) return momentum so a trailing-return signal is genuinely predictive."""
    rng = np.random.default_rng(seed)
    eps = 0.01 * rng.standard_normal((t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = rho * ret[i - 1] + eps[i]
    close = np.exp(np.cumsum(ret, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.0005 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.001 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.001 * rng.standard_normal((t, n))))
    volume = rng.uniform(1e5, 1e7, size=(t, n))
    dates = (np.datetime64("2012-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"M{i:03d}" for i in range(n)), open_, high, low, close,
                 volume, np.ones((t, n), bool), close * volume,
                 rng.integers(0, 5, size=n), {"survivorship_free": False, "source": "synthetic"})


# ---- Tier 0 ------------------------------------------------------------------

def test_assert_causal_passes_causal_signal() -> None:
    p = make_synthetic_panel(T=300, N=20, seed=3)
    ok, msg = assert_causal(CausalTrailing(1), p)
    assert ok, msg


def test_assert_causal_catches_leak() -> None:
    p = make_synthetic_panel(T=300, N=20, seed=3)
    ok, msg = assert_causal(LeakyForward(), p)
    assert not ok
    assert "differs" in msg or "look-ahead" in msg


def test_tier0_hygiene_gate() -> None:
    p = make_synthetic_panel(T=300, N=20, seed=4)
    h = tier0_hygiene(CausalTrailing(1), p, min_days=200, min_names_per_day=4)
    assert h.passed and h.causal and h.ohlc_violations == 0
    hl = tier0_hygiene(LeakyForward(), p, min_days=200)
    assert not hl.passed and not hl.causal


# ---- Tier 1 ------------------------------------------------------------------

def test_tier1_detects_momentum_alpha() -> None:
    p = _momentum_panel(rho=0.4, seed=1)
    gp = tier1_gross_power(CausalTrailing(1), p, horizons=(1, 5), primary_horizon=1,
                           neutralization=(), expected_sign=1)
    h1 = gp.by_horizon[1]
    assert h1.ic_mean > 0.02
    assert h1.ic_tstat > 3.0
    assert h1.decile_spread > 0.0
    assert gp.breadth > 0.5


def test_tier1_zero_on_random_panel() -> None:
    p = make_synthetic_panel(T=400, N=40, seed=2)
    gp = tier1_gross_power(CausalTrailing(1), p, horizons=(1,), primary_horizon=1,
                           neutralization=(), expected_sign=1)
    assert abs(gp.by_horizon[1].ic_mean) < 0.1
    assert gp.by_horizon[1].ic_tstat < 3.0


def test_expected_sign_flips_direction() -> None:
    p = _momentum_panel(rho=0.4, seed=5)
    gp_pos = tier1_gross_power(CausalTrailing(1), p, horizons=(1,), primary_horizon=1,
                               neutralization=(), expected_sign=1)
    gp_neg = tier1_gross_power(CausalTrailing(1), p, horizons=(1,), primary_horizon=1,
                               neutralization=(), expected_sign=-1)
    a = gp_pos.by_horizon[1].ic_mean
    b = gp_neg.by_horizon[1].ic_mean
    assert a > 0 and b < 0
    assert np.isclose(a, -b, atol=1e-9)


def test_min_names_gate_threaded_into_ic() -> None:
    # audit F2: the per-day IC must honour gates.min_names_per_day, not a hardcoded 4.
    p = _momentum_panel(t=300, n=40)  # 40 active names every day
    loose = tier1_gross_power(CausalTrailing(1), p, (1,), primary_horizon=1,
                              neutralization=(), expected_sign=1, min_names=4)
    strict = tier1_gross_power(CausalTrailing(1), p, (1,), primary_horizon=1,
                               neutralization=(), expected_sign=1, min_names=50)
    assert loose.by_horizon[1].n_days > 0
    assert strict.by_horizon[1].n_days == 0   # 40 names < 50 -> no day qualifies


def test_two_sided_signal_is_rejected() -> None:
    # audit F1: expected_sign=0 would pick its sign from full-sample returns (look-ahead);
    # it must be rejected at the harness entry, not silently evaluated.
    p = make_synthetic_panel(T=300, N=20, seed=4)

    class TwoSided:
        spec = SignalSpec(name="twosided", hypothesis="ambiguous direction",
                          family="technical", expected_sign=0, neutralization=())

        def compute(self, panel: Panel) -> np.ndarray:
            return np.log(panel.close)

    gates = Gates.from_dict({"universe": {"min_names_per_day": 4},
                             "coverage": {"min_days": 150}})
    card = evaluate_signal(TwoSided(), p, gates)
    assert card.verdict == "GATE_FAIL"
    assert card.gross is None
    assert any("two-sided" in r or "expected_sign=0" in r for r in card.hygiene.reasons)
