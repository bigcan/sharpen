"""Exposure_Frac metric tests.

Exposure gates a real decision in the gmgp1-spx500 A/B (a long-only book that mostly sits
flat posts a flattering Sharpe while capturing nothing), so the metric gets the same
treatment as any other decision input:

  * a flat book and a fully-invested book must NOT report the same exposure
  * "no position column" must be NaN, never 0.0 — otherwise a missing column is
    indistinguishable from a book that never traded, which is the exact confusion the
    metric exists to prevent
  * long/short split must be signed correctly, so a long-only run reads short_frac == 0
"""
import numpy as np
import pandas as pd
import pytest

from sharpen.analytics.wandb_evaluator import WandbFinRLEvaluator


def _frame(positions=None, n=200, seed=0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    pv = 100000 * np.cumprod(1 + rng.randn(n) * 0.0005)
    df = pd.DataFrame({"account_value": pv}, index=idx)
    if positions is not None:
        df["position"] = positions
    return df


def _metrics(df):
    ev = WandbFinRLEvaluator.__new__(WandbFinRLEvaluator)      # bypass __init__ (needs wandb/config)
    return WandbFinRLEvaluator.calculate_metrics(ev, df, "test")


def test_missing_position_column_is_nan_not_zero():
    """NEGATIVE: absent data must not masquerade as a measured zero."""
    m = _metrics(_frame(positions=None))
    assert np.isnan(m["Exposure_Frac"]), "missing position column reported a number"


def test_always_flat_book_reports_zero_exposure():
    m = _metrics(_frame(positions=np.zeros(200)))
    assert m["Exposure_Frac"] == pytest.approx(0.0)
    assert m["Long_Exposure_Frac"] == pytest.approx(0.0)
    assert m["Short_Exposure_Frac"] == pytest.approx(0.0)


def test_always_invested_book_reports_full_exposure():
    m = _metrics(_frame(positions=np.ones(200)))
    assert m["Exposure_Frac"] == pytest.approx(1.0)
    assert m["Long_Exposure_Frac"] == pytest.approx(1.0)


def test_flat_and_invested_are_distinguishable():
    """The whole point: these two must not look the same on the exposure axis."""
    flat = _metrics(_frame(positions=np.zeros(200)))
    invested = _metrics(_frame(positions=np.ones(200)))
    assert flat["Exposure_Frac"] != invested["Exposure_Frac"]


def test_long_short_split_is_signed():
    pos = np.concatenate([np.ones(100), -np.ones(60), np.zeros(40)])
    m = _metrics(_frame(positions=pos))
    assert m["Exposure_Frac"] == pytest.approx(0.8)
    assert m["Long_Exposure_Frac"] == pytest.approx(0.5)
    assert m["Short_Exposure_Frac"] == pytest.approx(0.3)


def test_long_only_book_reports_no_short_exposure():
    """Mirrors variant B: shorts banned => short_frac must be exactly 0, not merely small."""
    pos = np.concatenate([np.ones(120), np.zeros(80)])
    m = _metrics(_frame(positions=pos))
    assert m["Short_Exposure_Frac"] == pytest.approx(0.0)
    assert m["Long_Exposure_Frac"] == pytest.approx(0.6)
