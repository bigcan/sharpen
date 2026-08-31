"""Causal defensive / betting-against-beta (BAB) conviction — candidate 3rd linear-core sleeve.

The generalized Frazzini-Pedersen BAB premium (leverage-constrained investors bid up high-beta
assets → low-beta assets earn higher risk-adjusted returns). Built as the low-correlation
DEFENSIVE diversifier to the two live sleeves (cross-asset TSMOM net SR 0.601, rates-carry 0.467):
a within-asset-class, dollar-neutral **long low-beta / short high-beta** conviction the base-sleeve
loop vol-scales into a book, scored by its net-deflated MARGINAL contribution to the combined book
(NOT its standalone Sharpe — the portfolio bar, per the add-uncorrelated-sleeves thesis).

Construction (S553-cont-95):
  * **Market proxy** = the cross-sectional equal-weight return of the panel (a data-driven,
    universe-agnostic risk factor; no external index needed). Beta is measured against it.
  * **Causal rolling beta** ``β_i(t) = cov(r_i, r_m)/var(r_m)`` over a trailing window, then
    ``.shift(1)`` so row ``t`` reads returns ``<= t-1`` (hence close ``<= t-1``) — strictly causal,
    even one bar tighter than the momentum sleeve's ``close[<= t-skip]``.
  * **Within-class rank** removes the cross-class duration confound: ranking beta INSIDE each
    ``asset_class`` bucket and going long-low / short-high is genuine BAB (low-beta equity vs
    high-beta equity, …), not a disguised "long bonds, short stocks" risk-off bet. Centered to
    mean-zero within class → dollar-neutral within class. Frazzini-Pedersen apply BAB within each
    asset class exactly this way.

Causality (LEAK-2): every conviction value at ``t`` is a pure function of close ``<= t-1``; the
base-sleeve loop then earns the 1-day forward return ``t -> t+1``. Guarded by :func:`assert_causal`
(future-bar sweep + current-bar crash), mirroring ``cross_asset_signals.assert_causal``.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("defensive_signals")

# Trailing beta window; ~1y with a ~6m warmup floor. Not a tuned free parameter — the standard
# BAB beta horizon. Kept explicit so the causality/warmup contract is legible.
DEFAULT_BETA_WINDOW: int = 252
DEFAULT_BETA_MIN_PERIODS: int = 126
# Current-bar tripwire perturbation (mirrors cross_asset_signals): a 99% crash at bar t. With the
# beta ``.shift(1)`` row t reads returns <= t-1, so it is invariant; a shift-0 read would trip.
_CURRENT_BAR_CRASH: float = 0.01


def _causal_beta(
    close: pd.DataFrame, beta_window: int, min_periods: int
) -> pd.DataFrame:
    """``(T, N)`` causal rolling beta of each name vs the equal-weight market proxy.

    ``r_i = close.pct_change()``; market ``r_m`` = cross-sectional mean of ``r_i`` (skips inactive
    names). ``β_i(t) = rolling_cov(r_i, r_m)/rolling_var(r_m)`` over ``beta_window`` (``min_periods``
    warmup), then ``.shift(1)`` so row ``t`` uses returns ``<= t-1`` only. NaN through warmup (never
    back-filled) and wherever ``var(r_m)`` is undefined.
    """
    rets = close.pct_change()
    mkt = rets.mean(axis=1, skipna=True)                       # EW market return (contemporaneous)
    var_m = mkt.rolling(beta_window, min_periods=min_periods).var()
    beta = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        cov_im = rets[col].rolling(beta_window, min_periods=min_periods).cov(mkt)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta[col] = cov_im / var_m
    return beta.shift(1)                                       # CAUSAL: row t reads returns <= t-1


def _within_class_lowbeta_rank(
    beta: pd.DataFrame, asset_class: Mapping[str, str]
) -> pd.DataFrame:
    """Within-class, dollar-neutral **low-beta** conviction in ``[-1, 1]`` (long low-beta).

    Mirrors ``cross_asset_signals._xs_rank_within_class`` but on BETA and NEGATED: within each
    asset-class bucket, centered rank ``2·(rank-1)/(n-1) - 1`` maps high-beta → +1; we return its
    negation so LOW-beta → +1 (the long leg). Single-name classes (N<2) get 0 (no cross-section).
    NaN where beta is undefined (warmup) — never forward-filled.
    """
    out = pd.DataFrame(np.nan, index=beta.index, columns=beta.columns)
    classes: dict[str, list[str]] = {}
    for tk in beta.columns:
        classes.setdefault(asset_class.get(tk, "_unknown"), []).append(tk)

    for _cls, tickers in classes.items():
        sub = beta[tickers]
        if len(tickers) < 2:
            out[tickers] = sub.where(sub.isna(), 0.0)          # no cross-section in a 1-name class
            continue
        ranks = sub.rank(axis=1)                               # 1..n_valid per row (NaN preserved)
        n_valid = sub.notna().sum(axis=1)
        denom = (n_valid - 1).replace(0, np.nan)
        norm = (ranks.sub(1).div(denom, axis=0) * 2.0 - 1.0)   # high beta -> +1
        out[tickers] = -norm                                   # NEGATE: long low-beta
    return out


def defensive_conviction(
    close: pd.DataFrame,
    asset_class: Mapping[str, str],
    *,
    beta_window: int = DEFAULT_BETA_WINDOW,
    min_periods: int = DEFAULT_BETA_MIN_PERIODS,
) -> pd.DataFrame:
    """Causal within-class betting-against-beta conviction, ``(T, N)`` in ``[-1, 1]``.

    Args:
        close: wide daily close, DatetimeIndex ascending, columns = tickers (DATA-CLEAN upstream).
        asset_class: ticker → asset-class bucket (the within-class neutralization group). A bucket
            with a single name contributes 0 (no cross-section).
        beta_window / min_periods: trailing causal beta window and warmup floor.

    Returns:
        ``(len(close), len(close.columns))`` DataFrame indexed like ``close``, columns in input
        order. ``> 0`` = long (low-beta within its class), ``< 0`` = short (high-beta). Warmup rows
        carry NaN (the sleeve loop treats NaN as flat). Strictly causal: value at ``t`` is a pure
        function of close ``<= t-1``. Guarded by :func:`assert_causal`.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close.index must be a DatetimeIndex")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index must be sorted ascending (causal shift assumes order)")
    if close.isna().all(axis=None):
        raise ValueError("close is entirely NaN")

    beta = _causal_beta(close, beta_window, min_periods)
    conv = _within_class_lowbeta_rank(beta, asset_class)
    log.info("defensive_conviction: %d dates × %d tickers (beta_window=%d, min_periods=%d, "
             "classes=%d)", close.shape[0], close.shape[1], beta_window, min_periods,
             len(set(asset_class.get(tk, "_unknown") for tk in close.columns)))
    return conv


def assert_causal(
    close: pd.DataFrame,
    asset_class: Mapping[str, str],
    *,
    perturb_frac: float = 0.5,
    bump: float = 1.5,
    atol: float = 1e-9,
    **kwargs,
) -> None:
    """Look-ahead TRIPWIRE (LEAK-2). Two perturbations, each asserting the conviction at every date
    ``<= perturb_date`` is byte-identical:

      1. **future-bar sweep** — bump ALL bars strictly after ``t`` (catches a beta window reaching
         into the future).
      2. **current-bar** — crash bar ``t`` itself (catches a beta that reads its own bar; with the
         ``.shift(1)`` row ``t`` reads returns ``<= t-1`` and is invariant, a shift-0 read trips).

    Raises ``AssertionError`` on leak; returns ``None`` on pass.
    """
    base = defensive_conviction(close, asset_class, **kwargs)
    t_idx = int(len(close) * perturb_frac)
    perturb_date = close.index[t_idx]
    b = base.loc[base.index <= perturb_date]

    def _assert_past_unchanged(perturbed: pd.DataFrame, label: str) -> None:
        after = defensive_conviction(perturbed, asset_class, **kwargs)
        a = after.loc[after.index <= perturb_date]
        diff = (b.fillna(-999.0).to_numpy() - a.fillna(-999.0).to_numpy())
        max_diff = float(np.abs(diff).max()) if diff.size else 0.0
        if max_diff > atol:
            raise AssertionError(
                f"LEAK-2 VIOLATION ({label}): perturbing the {label} around "
                f"{perturb_date.date()} changed a conviction at <= {perturb_date.date()} by "
                f"{max_diff:.3e}. The BAB beta is reading {label} data — fix the shift before build."
            )

    fut = close.copy()
    fut.iloc[t_idx + 1:] = fut.iloc[t_idx + 1:] * bump
    _assert_past_unchanged(fut, "future-bar")

    cur = close.copy()
    cur.iloc[t_idx] = cur.iloc[t_idx] * _CURRENT_BAR_CRASH
    _assert_past_unchanged(cur, "current-bar")

    log.info("defensive_signals.assert_causal PASS: 0 leak (future-bar + current-bar) across "
             "%d past rows (perturb @ %s)", len(b), perturb_date.date())
