"""WorldQuant "101 Formulaic Alphas" operator vocabulary (Kakushadze 2015, arXiv:1601.00991).

All operators act on ``(T, N)`` float arrays (T days × N names). Two families:
  * **cross-sectional** (``rank``, ``scale``, ``indneutralize``) — per-day, across names (axis 1);
  * **time-series** (``delay``, ``delta``, ``ts_*``, ``correlation``, ``decay_linear`` …) —
    per-name, over a TRAILING window of ``d`` days (axis 0).

Every time-series operator uses only data ``<= t`` (trailing ``rolling``/backward ``shift``),
so any alpha built from these is causal by construction — and the harness Tier-0 truncation
tripwire re-verifies that per alpha (it cannot, however, catch a transcription error, so the
formulas in ``alphas101`` are transcribed against the paper directly).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _df(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(x, dtype=np.float64))


# ----------------------------------------------------------- cross-sectional ----

def rank(x: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional percentile rank in [0, 1] (NaN-safe; NaN stays NaN)."""
    return _df(x).rank(axis=1, pct=True).to_numpy()


def scale(x: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Per-day rescale so the row's sum of absolute values equals ``a``."""
    df = _df(x)
    norm = df.abs().sum(axis=1).replace(0.0, np.nan)
    return df.mul(a).div(norm, axis=0).to_numpy()


def indneutralize(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-day demean ``x`` within each group (sector) — the paper's ``IndClass`` neutralize."""
    out = np.asarray(x, dtype=np.float64).copy()
    g = np.asarray(groups)
    for gid in np.unique(g):
        cols = np.where(g == gid)[0]
        sub = out[:, cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN group slice → NaN
            m = np.nanmean(sub, axis=1, keepdims=True)
        out[:, cols] = sub - m
    return out


# -------------------------------------------------------------- element-wise ----

def signedpower(x: np.ndarray, a: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.abs(x) ** a


def s_min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(x, float), np.asarray(y, float))


def s_max(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, float), np.asarray(y, float))


def log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.where(x > 0, x, np.nan))


def sign(x: np.ndarray) -> np.ndarray:
    return np.sign(np.asarray(x, dtype=np.float64))


def where(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ternary ``cond ? a : b`` (the paper's ``(cond) ? a : b``), broadcasting scalars."""
    return np.where(cond, a, b).astype(np.float64)


# --------------------------------------------------------------- time-series ----

def delay(x: np.ndarray, d: int) -> np.ndarray:
    """Value ``d`` days ago (backward shift; first ``d`` rows NaN). Causal."""
    x = np.asarray(x, dtype=np.float64)
    if d <= 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def delta(x: np.ndarray, d: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - delay(x, d)


def ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    return _df(x).rolling(d, min_periods=d).sum().to_numpy()


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    return _df(x).rolling(d, min_periods=d).mean().to_numpy()


def stddev(x: np.ndarray, d: int) -> np.ndarray:
    return _df(x).rolling(d, min_periods=d).std().to_numpy()


def ts_min(x: np.ndarray, d: int) -> np.ndarray:
    return _df(x).rolling(d, min_periods=d).min().to_numpy()


def ts_max(x: np.ndarray, d: int) -> np.ndarray:
    return _df(x).rolling(d, min_periods=d).max().to_numpy()


def _windows(x: np.ndarray, d: int):
    """Trailing ``d``-length windows as a strided VIEW (T-d+1, N, d); None if d > T."""
    x = np.asarray(x, dtype=np.float64)
    return sliding_window_view(x, d, axis=0) if d <= x.shape[0] else None


def product(x: np.ndarray, d: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    w = _windows(x, d)
    if w is not None:
        out[d - 1:] = np.prod(w, axis=-1)
    return out


def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """Rolling rank of today's value within the trailing ``d``-day window, in [0, 1]."""
    return _df(x).rolling(d, min_periods=d).rank(pct=True).to_numpy()


def _ts_arg(x: np.ndarray, d: int, fn) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    w = _windows(x, d)
    if w is not None:
        res = fn(w, axis=-1).astype(np.float64)
        res[np.isnan(w).any(axis=-1)] = np.nan        # argmax over a NaN window is undefined
        out[d - 1:] = res
    return out


def ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    """Index (0 = oldest .. d-1 = today) of the max within the trailing ``d``-day window."""
    return _ts_arg(x, d, np.argmax)


def ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    return _ts_arg(x, d, np.argmin)


def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """Linearly-decaying weighted MA over ``d`` days (most-recent weight highest, weights sum 1)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    w = _windows(x, d)
    if w is not None:
        wts = np.arange(1, d + 1, dtype=np.float64)
        wts /= wts.sum()
        out[d - 1:] = w @ wts
    return out


def _roll_sum(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d, min_periods=d).sum()


def correlation(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Per-name rolling Pearson correlation of ``x`` and ``y`` over the trailing ``d`` days."""
    x_, y_ = _df(x), _df(y)
    sx, sy = _roll_sum(x_, d), _roll_sum(y_, d)
    sxx, syy, sxy = _roll_sum(x_ * x_, d), _roll_sum(y_ * y_, d), _roll_sum(x_ * y_, d)
    cov = sxy - sx * sy / d
    var = (sxx - sx * sx / d) * (syy - sy * sy / d)
    denom = var.pow(0.5).replace(0.0, np.nan)
    return (cov / denom).to_numpy()


def covariance(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """Per-name rolling covariance of ``x`` and ``y`` over the trailing ``d`` days."""
    x_, y_ = _df(x), _df(y)
    sx, sy, sxy = _roll_sum(x_, d), _roll_sum(y_, d), _roll_sum(x_ * y_, d)
    return ((sxy - sx * sy / d) / d).to_numpy()


def adv(close: np.ndarray, volume: np.ndarray, d: int) -> np.ndarray:
    """Average daily dollar volume over ``d`` days (the paper's ``adv{d}``)."""
    dollar = np.asarray(close, dtype=np.float64) * np.asarray(volume, dtype=np.float64)
    return _df(dollar).rolling(d, min_periods=d).mean().to_numpy()


def returns(close: np.ndarray) -> np.ndarray:
    """Daily close-to-close simple return (the paper's ``returns``). Causal."""
    c = np.asarray(close, dtype=np.float64)
    return c / delay(c, 1) - 1.0
