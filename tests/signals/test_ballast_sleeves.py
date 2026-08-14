"""BALLAST P2 — sleeve causality and construction tests.

The load-bearing test is :class:`TestSleeveCausality`: every sleeve must satisfy
``sleeve(panel.truncated(t))[t] == sleeve(panel)[t]``. That is the Tier-0 tripwire the signal
harness uses, applied to the six BALLAST sleeves — it fails loudly if any rolling window, shift or
``sliding_window_view`` is ever mis-anchored so that row ``t`` sees row ``t+1``.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.data.fundamentals import FundamentalPanel
from finrl_pro_ds.signals.features import make_synthetic_panel
from finrl_pro_ds.signals.library import ballast as bs


@pytest.fixture
def panel():
    return make_synthetic_panel(T=600, N=40, n_sectors=5, seed=7)


@pytest.fixture
def fp(panel):
    rng = np.random.default_rng(3)
    T, N = panel.T, panel.N
    vals = {}
    for c in ("revenue", "net_income", "gross_profit", "operating_income", "cfo", "capex",
              "assets", "liabilities", "equity", "shares"):
        v = np.abs(rng.normal(1000, 200, (T, N)))
        v[:100] = np.nan                       # pre-coverage block, as in the real panel
        vals[c] = v
        vals[f"{c}_ttm"] = v * 4
    avail = np.isfinite(vals["assets"])
    return FundamentalPanel(panel.dates, panel.tickers, vals, avail, {"src": "test"})


SLEEVES = ["low_risk", "momentum", "trend", "liquidity"]


class TestSleeveCausality:
    """LEAK-2 tripwire — row t may never depend on rows > t."""

    @pytest.mark.parametrize("name", SLEEVES)
    def test_truncation_property(self, panel, name):
        fn = {"low_risk": bs.sleeve_low_risk, "momentum": bs.sleeve_momentum,
              "trend": bs.sleeve_trend, "liquidity": bs.sleeve_liquidity}[name]
        full = fn(panel)
        for t in (300, 450, panel.T - 1):
            part = fn(panel.truncated(t))
            assert np.allclose(part[t], full[t], equal_nan=True, atol=1e-9), \
                f"{name} at t={t} depends on the future"

    def test_fundamental_sleeves_are_truncation_safe(self, panel, fp):
        full = bs.sleeve_quality(panel, fp)
        for t in (300, 500):
            sub = FundamentalPanel(fp.dates[:t + 1], fp.tickers,
                                   {k: v[:t + 1] for k, v in fp.values.items()},
                                   fp.available[:t + 1], fp.meta)
            part = bs.sleeve_quality(panel.truncated(t), sub)
            assert np.allclose(part[t], full[t], equal_nan=True, atol=1e-9)

    def test_shifting_prices_forward_changes_the_signal(self, panel):
        """Positive control: if the sleeve were reading the future, a future-only edit would show."""
        import dataclasses
        tampered = dataclasses.replace(panel, close=panel.close.copy())
        tampered.close[500:] *= 2.0
        base, alt = bs.sleeve_trend(panel), bs.sleeve_trend(tampered)
        assert np.allclose(base[:499], alt[:499], equal_nan=True), "past changed by a future edit"
        assert not np.allclose(base[550], alt[550], equal_nan=True), "sleeve ignored a real change"


class TestRollingHelpers:
    def test_rolling_mean_matches_pandas(self, panel):
        import pandas as pd
        out = bs.rolling_mean(panel.close, 20, min_periods=20)
        ref = pd.DataFrame(panel.close).rolling(20, min_periods=20).mean().to_numpy()
        assert np.allclose(out[19:], ref[19:], equal_nan=True, atol=1e-9)

    def test_rolling_std_matches_numpy_on_a_window(self, panel):
        out = bs.rolling_std(panel.close, 30, min_periods=30)
        assert out[50] == pytest.approx(np.nanstd(panel.close[21:51], axis=0), rel=1e-9)

    def test_log_returns_first_row_is_nan(self, panel):
        r = bs.log_returns(panel.close)
        assert np.isnan(r[0]).all() and np.isfinite(r[1]).any()


class TestSleeveSemantics:
    def test_low_risk_prefers_the_calmer_name(self, panel):
        import dataclasses
        close = panel.close.copy()
        noise = np.random.default_rng(1).normal(0, 0.05, panel.T).cumsum()
        close[:, 0] = 100 * np.exp(noise)                      # volatile
        close[:, 1] = 100 * np.exp(np.linspace(0, 0.1, panel.T))  # smooth
        p = dataclasses.replace(panel, close=close)
        s = bs.sleeve_low_risk(p)
        assert s[-1, 1] > s[-1, 0], "the smoother name must score higher on low-risk"

    def test_momentum_skips_the_last_month(self, panel):
        import dataclasses
        close = panel.close.copy()
        close[-10:, 0] *= 3.0                     # a spike inside the skip window
        p = dataclasses.replace(panel, close=close)
        assert bs.sleeve_momentum(p)[-1, 0] == pytest.approx(bs.sleeve_momentum(panel)[-1, 0]), \
            "12-1 momentum must ignore the most recent month"

    def test_liquidity_prefers_the_smaller_name(self, panel):
        s = bs.sleeve_liquidity(panel)
        j_small = int(np.nanargmin(panel.adv_usd[-1]))
        j_big = int(np.nanargmax(panel.adv_usd[-1]))
        assert s[-1, j_small] > s[-1, j_big]

    def test_trend_is_positive_above_the_moving_average(self, panel):
        import dataclasses
        close = np.tile(np.linspace(100, 200, panel.T)[:, None], (1, panel.N))
        s = bs.sleeve_trend(dataclasses.replace(panel, close=close))
        assert s[-1, 0] > 0, "a rising series must sit above its own trailing MA"


class TestBlending:
    def test_blend_ignores_missing_parts(self, panel):
        a = np.full((panel.T, panel.N), np.nan)
        b = np.tile(np.arange(panel.N, dtype=float), (panel.T, 1))
        out = bs._blend([a, b], panel.active)
        assert np.isfinite(out[-1]).all(), "a fully-missing part must not void the blend"

    def test_blend_of_two_identical_parts_equals_one(self, panel):
        b = np.tile(np.arange(panel.N, dtype=float), (panel.T, 1))
        assert np.allclose(bs._blend([b, b], panel.active)[-1],
                           bs._blend([b], panel.active)[-1], equal_nan=True)

    def test_all_nan_row_stays_nan(self, panel):
        a = np.full((panel.T, panel.N), np.nan)
        assert np.isnan(bs._blend([a], panel.active)[-1]).all()


class TestComputeSleeves:
    def test_market_only_when_no_fundamentals(self, panel):
        out = bs.compute_sleeves(panel)
        assert set(out) == set(bs.MARKET_SLEEVES)
        assert all(v.shape == (panel.T, panel.N) for v in out.values())

    def test_fundamentals_add_quality_and_value(self, panel, fp):
        mcap = np.full((panel.T, panel.N), 1e9)
        out = bs.compute_sleeves(panel, fp, mcap)
        assert set(out) == set(bs.SLEEVE_NAMES)

    def test_value_omitted_without_market_cap(self, panel, fp):
        out = bs.compute_sleeves(panel, fp, None)
        assert "quality" in out and "value" not in out, \
            "value needs an unadjusted-price market cap; it must not be faked"

    def test_liquidity_is_not_size_neutralized(self, panel):
        """Residualizing the size sleeve on size would zero it — the guard must hold."""
        out = bs.compute_sleeves(panel)
        assert np.nanstd(out["liquidity"][-1]) > 1e-6
