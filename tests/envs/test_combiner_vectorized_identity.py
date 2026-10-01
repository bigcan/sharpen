"""Bit-identity tripwires for the vectorized sleeve combiner (Crucible deep audit, 2026-09-30).

``_monthly_held`` / ``risk_parity_alphas`` / ``dynamic_sleeve_alphas`` were rewritten from per-row
Python loops to whole-array numpy (they dominated a Crucible mining tick's runtime). The live paper
executors (TAILWIND) share these functions, so the rewrite must not move a single bit. Each
reference below is a VERBATIM copy of the pre-vectorization loop; the tests assert exact equality
(``np.array_equal`` with NaN-position equality), never a tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.envs.allocator_factory import (
    _month_end_mask,
    _monthly_held,
    _trailing_ann_perf,
    _trailing_ann_vol,
    _trailing_mean_abs_corr,
    dynamic_sleeve_alphas,
    risk_parity_alphas,
)


# --------------------------------------------------------------------------- verbatim references
def _ref_monthly_held(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64)
    out = np.array(values, dtype=np.float64, copy=True)
    if len(ts) == 0:
        return out
    months = pd.to_datetime(ts, unit="s").to_period("M")
    last_of_month = pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()
    is_me = np.zeros(len(ts), dtype=bool)
    is_me[last_of_month] = True
    held = out[0].copy() if out.ndim > 1 else out[0]
    for k in range(len(ts)):
        if is_me[k]:
            held = out[k].copy() if out.ndim > 1 else out[k]
        out[k] = held
    return out


def _ref_risk_parity(sleeve_returns, timestamps, *, window=252, min_periods=63, monthly_meta=True,
                     target_portfolio_vol=None, vol_floor=1e-4):
    names = list(sleeve_returns)
    K = len(np.asarray(timestamps))
    N = len(names)
    sig = {s: _trailing_ann_vol(sleeve_returns[s], window=window, min_periods=min_periods)
           for s in names}
    if monthly_meta:
        sig = {s: _ref_monthly_held(sig[s], timestamps) for s in names}
    alphas = {s: np.full(K, 1.0 / N, dtype=np.float64) for s in names}
    for k in range(K):
        sk = np.array([sig[s][k] for s in names], dtype=np.float64)
        usable = np.isfinite(sk) & (sk > vol_floor)
        if not usable.all():
            continue
        inv = 1.0 / sk
        if target_portfolio_vol is None:
            a = inv / inv.sum()
        else:
            a = (1.0 / N) * float(target_portfolio_vol) * inv
        for j, s in enumerate(names):
            alphas[s][k] = a[j]
    return alphas


def _ref_dynamic(sleeve_returns, timestamps, *, window=252, min_periods=63, monthly_meta=True,
                 vol_floor=1e-4, tilt_strength=0.0, perf_window=126, perf_min_periods=63,
                 perf_metric="sharpe", tilt_clip=1.5, redundancy_strength=0.0):
    names = list(sleeve_returns)
    K = len(np.asarray(timestamps))
    N = len(names)
    sigma = {s: _trailing_ann_vol(sleeve_returns[s], window=window, min_periods=min_periods)
             for s in names}
    shat = {}
    for s in names:
        perf, denom = _trailing_ann_perf(sleeve_returns[s], window=perf_window,
                                         min_periods=perf_min_periods, metric=perf_metric)
        v = perf.copy()
        neutral = ~np.isfinite(v) | ~np.isfinite(denom) | (denom <= vol_floor)
        v[neutral] = 0.0
        shat[s] = v
    lam_r = float(redundancy_strength)
    if lam_r > 0.0:
        rho = _trailing_mean_abs_corr(sleeve_returns, window=perf_window,
                                      min_periods=perf_min_periods)
    else:
        rho = {s: np.zeros(K, dtype=np.float64) for s in names}
    if monthly_meta:
        sigma = {s: _ref_monthly_held(sigma[s], timestamps) for s in names}
        shat = {s: _ref_monthly_held(shat[s], timestamps) for s in names}
        if lam_r > 0.0:
            rho = {s: _ref_monthly_held(rho[s], timestamps) for s in names}
    alphas = {s: np.full(K, 1.0 / N, dtype=np.float64) for s in names}
    lam = float(tilt_strength)
    c = float(tilt_clip)
    for k in range(K):
        sk = np.array([sigma[s][k] for s in names], dtype=np.float64)
        usable = np.isfinite(sk) & (sk > vol_floor)
        if not usable.all():
            continue
        inv = 1.0 / sk
        shk = np.array([shat[s][k] for s in names], dtype=np.float64)
        tilt = np.exp(lam * np.clip(shk, -c, c))
        rhk = np.array([rho[s][k] for s in names], dtype=np.float64)
        rhk = np.where(np.isfinite(rhk), np.clip(rhk, 0.0, 1.0), 0.0)
        redund = np.exp(-lam_r * rhk)
        wk = inv * tilt * redund
        a = wk / wk.sum()
        for j, s in enumerate(names):
            alphas[s][k] = a[j]
    return alphas


# --------------------------------------------------------------------------- fixtures
def _daily_ts(k: int, start: str = "2011-01-03") -> np.ndarray:
    d = pd.bdate_range(start, periods=k)
    return (d.astype("int64") // 10**9).to_numpy().astype(np.float64)


def _sleeves(rng, k: int, n: int, *, nan_frac: float = 0.0, degenerate: bool = False) -> dict:
    out = {}
    for j in range(n):
        r = rng.standard_normal(k) * rng.uniform(0.002, 0.03)
        if nan_frac:
            r[rng.random(k) < nan_frac] = np.nan
        if degenerate and j == 0:
            r[: k // 3] = 0.0                    # a zero-vol stretch (σ ≤ floor → equal-weight rows)
        out[f"s{j}"] = r
    return out


def _same(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(np.array_equal(a[s], b[s], equal_nan=True) for s in a)


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("seed", range(6))
def test_month_end_mask_matches_pandas_groupby(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 900))
    ts = _daily_ts(k)
    if seed % 2:                                 # unsorted + duplicated stamps: groupby().max() semantics
        ts = rng.permutation(np.concatenate([ts, ts[: k // 5]]))
    months = pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s").to_period("M")
    ref = np.zeros(len(ts), dtype=bool)
    ref[pd.Series(np.arange(len(ts))).groupby(months.values).max().to_numpy()] = True
    assert np.array_equal(_month_end_mask(ts), ref)


@pytest.mark.parametrize("seed", range(6))
def test_monthly_held_bit_identical(seed):
    rng = np.random.default_rng(100 + seed)
    k = int(rng.integers(1, 700))
    ts = _daily_ts(k, start=str(rng.choice(["2007-06-15", "2019-12-31", "2023-02-27"])))
    v1 = rng.standard_normal(k)
    v1[rng.random(k) < 0.2] = np.nan
    v2 = rng.standard_normal((k, 3))
    assert np.array_equal(_monthly_held(v1, ts), _ref_monthly_held(v1, ts), equal_nan=True)
    assert np.array_equal(_monthly_held(v2, ts), _ref_monthly_held(v2, ts), equal_nan=True)


def test_monthly_held_empty():
    assert _monthly_held(np.array([]), np.array([])).shape == (0,)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("monthly_meta", [True, False])
def test_risk_parity_alphas_bit_identical(seed, monthly_meta):
    rng = np.random.default_rng(200 + seed)
    k, n = int(rng.integers(40, 900)), int(rng.integers(1, 9))
    rets = _sleeves(rng, k, n, nan_frac=0.05 * (seed % 3), degenerate=bool(seed % 2))
    ts = _daily_ts(k)
    min_periods = 5 + (seed * 11) % 75
    for tpv in (None, 0.10):
        got = risk_parity_alphas(rets, ts, monthly_meta=monthly_meta, target_portfolio_vol=tpv,
                                 min_periods=min_periods)
        ref = _ref_risk_parity(rets, ts, monthly_meta=monthly_meta, target_portfolio_vol=tpv,
                               min_periods=min_periods)
        assert _same(got, ref)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("monthly_meta", [True, False])
def test_dynamic_sleeve_alphas_bit_identical(seed, monthly_meta):
    rng = np.random.default_rng(300 + seed)
    k, n = int(rng.integers(40, 900)), int(rng.integers(1, 10))
    rets = _sleeves(rng, k, n, nan_frac=0.04 * (seed % 3), degenerate=bool(seed % 2))
    ts = _daily_ts(k)
    min_periods = int(rng.integers(5, 70))
    kw = dict(monthly_meta=monthly_meta, window=min_periods + int(rng.integers(0, 250)),
              min_periods=min_periods, tilt_strength=float(rng.choice([0.0, 1.0, 3.0])),
              perf_window=126, perf_min_periods=int(rng.integers(10, 126)),
              perf_metric=str(rng.choice(["sharpe", "sortino"])),
              redundancy_strength=float(rng.choice([0.0, 0.0, 2.0])))
    assert _same(dynamic_sleeve_alphas(rets, ts, **kw), _ref_dynamic(rets, ts, **kw))


def test_production_shape_bit_identical():
    """The exact call ``fitness._combined_book`` makes (FitnessConfig defaults), on a book with a
    candidate that is all-NaN for a stretch — the equal-weight fallback path the audit exercised."""
    rng = np.random.default_rng(7)
    k = 1200
    rets = _sleeves(rng, k, 4)
    cand = rng.standard_normal(k) * 0.01
    cand[300:520] = np.nan
    rets["_candidate"] = cand
    ts = _daily_ts(k)
    kw = dict(window=252, min_periods=63, monthly_meta=True, tilt_strength=0.0, perf_window=126,
              perf_min_periods=63, tilt_clip=1.5, redundancy_strength=0.0)
    assert _same(dynamic_sleeve_alphas(rets, ts, target_portfolio_vol=None, **kw),
                 _ref_dynamic(rets, ts, **kw))
