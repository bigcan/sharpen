"""Financing leg for self-funded allocator books (TAILWIND Tier-2 roadmap N2, 2026-09-29).

A self-funded ETF book pays for its longs out of cash and has its short proceeds credited to
cash, so it holds ``(1 - net)`` of equity in cash. Its return in excess of the T-bill rate is

    r_excess = sum_i w_i r_i  -  net * rf  -  S * b

where ``net = sum_i w_i``, ``S`` is the short notional (a fraction of equity), ``rf`` is the
per-bar cash rate and ``b`` is the per-bar borrow fee on shorts.

The allocator env and :class:`~sharpen.paper.paper_state.PaperState` accrue ``carry_ary`` on
open positions: a long earns it and a short pays it. So ``carry_ary = -rf`` on every asset gives
the ``-net * rf`` term exactly, and ``borrow_ary = b`` (charged on short notional only) gives
``-S * b``. The book's P&L is then its excess-of-T-bill P&L, which is the estimand of every
certifying Sharpe, DSR and P(pass) for a net-long book. For a prop CFD account, where the account
balance earns no interest, the same figure is a FLOOR on the financing drag. Venue swap markups
are roadmap item X4, not modelled here.

Rates are causal (LEAK-2):
- The interval ``(t-1, t]`` accrues at the cash yield known at close ``t-1``, read as-of from the
  curve.
- It accrues over the calendar days in that interval, act/``day_count``. 360 is the default, the
  money-market convention the 3m bill yield (^IRX) is quoted on.
- A position's financing is charged on the notional it carried INTO the interval (marked at close
  ``t-1``), which is how a broker charges daily interest on the balance held overnight.
- Row 0 of a calendar closes no interval, so its rate is 0.

Opt-in: a config with no ``financing:`` block, or ``model: none``, builds ``carry_ary = 0``
and no ``borrow_ary``, which is byte-identical to the unfinanced path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

FINANCING_MODELS = ("none", "tbill")
DAY_COUNTS = (360, 365)
_KEYS = frozenset({"model", "tenor", "day_count", "short_borrow_bps"})
_NS_PER_DAY = 86_400 * 10**9


@dataclass(frozen=True)
class FinancingSpec:
    """The resolved ``financing:`` config block."""

    model: str = "none"            # "none" (carry 0, legacy) | "tbill"
    tenor: str = "3m"              # Treasury-curve tenor used as the cash rate (PERCENT)
    day_count: int = 360           # act/day_count accrual on calendar days
    short_borrow_bps: float = 0.0  # annual borrow fee on short notional, basis points

    @property
    def enabled(self) -> bool:
        return self.model != "none"


def financing_spec(config: Mapping) -> FinancingSpec:
    """Resolve ``config['financing']`` into a :class:`FinancingSpec`, failing closed.

    If the block is absent the model is ``none``. An unknown key, an unknown model, a day count
    other than 360/365, or a negative or non-finite borrow fee raises: a misspelled key would
    otherwise leave the book silently unfinanced. A borrow fee without a financing model also
    raises, because the declaration is ambiguous."""
    raw = config.get("financing")
    if raw is None:
        return FinancingSpec()
    block = dict(raw)
    unknown = sorted(set(block) - _KEYS)
    if unknown:
        raise ValueError(f"financing: unknown key(s) {unknown}; allowed {sorted(_KEYS)}")
    model = str(block.get("model", "none")).lower()
    if model not in FINANCING_MODELS:
        raise ValueError(f"financing.model must be one of {FINANCING_MODELS}, got {model!r}")
    day_count = int(block.get("day_count", 360))
    if day_count not in DAY_COUNTS:
        raise ValueError(f"financing.day_count must be one of {DAY_COUNTS}, got {day_count}")
    bps = float(block.get("short_borrow_bps", 0.0))
    if not np.isfinite(bps) or bps < 0.0:
        raise ValueError(f"financing.short_borrow_bps must be finite and >= 0, got {bps}")
    if model == "none" and bps > 0.0:
        raise ValueError("financing.short_borrow_bps > 0 needs financing.model: tbill")
    return FinancingSpec(model=model, tenor=str(block.get("tenor", "3m")),
                         day_count=day_count, short_borrow_bps=bps)


def _interval_days(dates: pd.DatetimeIndex) -> np.ndarray:
    """Calendar days in each interval ``(t-1, t]``; row 0 closes no interval (0 days)."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) and (not dates.is_monotonic_increasing or dates.has_duplicates):
        raise ValueError("financing: the bar calendar must be strictly increasing")
    days = np.zeros(len(dates), dtype=np.float64)
    if len(dates) > 1:
        days[1:] = np.diff(dates.normalize().asi8) / _NS_PER_DAY
    return days


def per_bar_cash_rate(rf_pct: pd.Series, dates: pd.DatetimeIndex, *, day_count: int) -> pd.Series:
    """Per-bar cash rate (fraction) for the interval ending at each bar of ``dates``.

    ``rate[t] = y(t-1) / 100 * days(t-1, t) / day_count``, where ``y(t-1)`` is the last yield
    printed at or before close ``t-1`` (PERCENT). ``rate[0] = 0``. Raises if a bar that closes
    an interval has no yield on or before its previous close, so a gap in the curve fails
    instead of silently financing at 0."""
    if day_count not in DAY_COUNTS:
        raise ValueError(f"day_count must be one of {DAY_COUNTS}, got {day_count}")
    dates = pd.DatetimeIndex(dates)
    days = _interval_days(dates)
    y = rf_pct.dropna().sort_index().reindex(dates, method="ffill").to_numpy(np.float64)
    rate = np.zeros(len(dates), dtype=np.float64)
    if len(dates) > 1:
        prev_y = y[:-1]
        missing = np.isnan(prev_y)
        if missing.any():
            first = dates[int(np.argmax(missing))]
            raise ValueError(f"financing: no cash yield on or before {first.date()} "
                             f"(curve starts {rf_pct.dropna().index.min()})")
        rate[1:] = prev_y / 100.0 * days[1:] / float(day_count)
    return pd.Series(rate, index=dates, name="cash_rate")


def per_bar_borrow_rate(dates: pd.DatetimeIndex, *, short_borrow_bps: float,
                        day_count: int) -> pd.Series:
    """Per-bar borrow fee (fraction of short notional) for the interval ending at each bar:
    ``bps / 1e4 * days(t-1, t) / day_count``, with ``rate[0] = 0``."""
    if day_count not in DAY_COUNTS:
        raise ValueError(f"day_count must be one of {DAY_COUNTS}, got {day_count}")
    days = _interval_days(pd.DatetimeIndex(dates))
    return pd.Series(float(short_borrow_bps) / 1e4 * days / float(day_count),
                     index=pd.DatetimeIndex(dates), name="borrow_rate")


def financing_rates(curve: Mapping[str, pd.Series], dates: pd.DatetimeIndex,
                    spec: FinancingSpec) -> dict:
    """Per-bar financing rates on the FULL bar calendar ``dates``, to be sliced per window.

    Returns ``{"spec", "cash_rate", "borrow_rate"}``. ``borrow_rate`` is None when the spec
    charges no borrow. Rates are computed on the full calendar, not per window, so that a
    window's first interval reads the bar before the window, the same way the signals do."""
    if not spec.enabled:
        raise ValueError("financing_rates called with financing.model: none")
    if spec.tenor not in curve:
        raise ValueError(f"financing.tenor {spec.tenor!r} not in curve tenors {sorted(curve)}")
    cash = per_bar_cash_rate(curve[spec.tenor], dates, day_count=spec.day_count)
    borrow = (per_bar_borrow_rate(dates, short_borrow_bps=spec.short_borrow_bps,
                                  day_count=spec.day_count)
              if spec.short_borrow_bps > 0.0 else None)
    return {"spec": spec, "cash_rate": cash, "borrow_rate": borrow}


def financing_arrays(financing: Mapping | None, wdates: pd.DatetimeIndex, n_assets: int) -> dict:
    """The env/PaperState financing arrays over window ``wdates`` for ``n_assets`` assets.

    With ``financing=None`` it returns ``{"carry_ary": zeros}``, the unfinanced path. Otherwise
    it returns ``carry_ary = -cash_rate`` on every asset, plus ``borrow_ary = borrow_rate`` if
    the spec charges borrow. Both arrays have shape ``(T, n_assets)``."""
    T = len(wdates)
    if financing is None:
        return {"carry_ary": np.zeros((T, n_assets), dtype=np.float64)}
    cash = financing["cash_rate"].reindex(pd.DatetimeIndex(wdates))
    if cash.isna().any():
        raise ValueError("financing: window bars missing from the financing calendar")
    col = cash.to_numpy(np.float64)[:, None]
    out = {"carry_ary": np.repeat(-col, n_assets, axis=1)}
    borrow = financing.get("borrow_rate")
    if borrow is not None:
        b = borrow.reindex(pd.DatetimeIndex(wdates))
        if b.isna().any():
            raise ValueError("financing: window bars missing from the borrow calendar")
        out["borrow_ary"] = np.repeat(b.to_numpy(np.float64)[:, None], n_assets, axis=1)
    return out


def excess_returns(returns: pd.Series, held: pd.DataFrame, cash_rate: pd.Series,
                   borrow_rate: pd.Series | None = None) -> pd.Series:
    """Excess-of-cash return of a research weight book: ``r - net * rf - S * b``.

    ``held`` holds the weights in force over each interval ``(t-1, t]``, i.e. the research
    backtest's lagged daily weights (``xsec_momentum_falsification.held_weights``). ``returns``
    is that book's net return on the same calendar. The rates must cover every bar."""
    idx = returns.index
    h = held.reindex(idx)
    if h.isna().any().any():
        raise ValueError("excess_returns: held weights do not cover the return calendar")
    rf = cash_rate.reindex(idx)
    if rf.isna().any():
        raise ValueError("excess_returns: cash rate does not cover the return calendar")
    out = returns - h.sum(axis=1) * rf
    if borrow_rate is not None:
        b = borrow_rate.reindex(idx)
        if b.isna().any():
            raise ValueError("excess_returns: borrow rate does not cover the return calendar")
        out = out - h.clip(upper=0.0).abs().sum(axis=1) * b
    return out.rename(returns.name)
