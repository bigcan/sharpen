"""Causal options-volatility features for the crypto-options VRP strategy.

Mirrors the design of ``sharpen/features/cross_asset_signals.compute`` (the
cross-asset allocator's signal module): every column at row ``t`` is a pure
function of data ``<= t`` (LEAK-2). The VRP edge signal is the spread between
the *forward-looking* implied vol (DVOL, 30-day constant maturity) and the
*trailing* realized vol — both observable at ``t``. The realized vol that the
short-vol book actually earns against is a future outcome computed by the
simulator, NOT a feature here.

Aligns the three Deribit series (DVOL, perp, funding — see
``deribit_options_loader``) onto a common UTC **calendar-date** grid:
  * DVOL stamps at 00:00 UTC, perp daily candles at 08:00 UTC, funding hourly —
    all are floored to date and joined, which is exact for a daily cadence.
  * Daily funding carry = mean(interest_8h over the day) * 3  (3 funding
    periods/day), the conventional perpetual daily carry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUALIZATION_DAYS = 365  # crypto trades 24/7/365


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Floor a UTC DatetimeIndex to calendar date, keeping the last row per date."""
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).floor("D")
    out = out[~out.index.duplicated(keep="last")]
    return out


def _realized_vol(close: pd.Series, window: int, ann: int = ANNUALIZATION_DAYS) -> pd.Series:
    """Trailing close-to-close annualized realized vol ending at each bar (causal).

    rv[t] uses log returns r_{t-window+1..t}; r_t = ln(S_t/S_{t-1}) is known at t,
    so rv[t] depends only on data <= t. ``min_periods == window`` => NaN warmup
    (no partial-window leakage of a shorter sample masquerading as the signal).
    """
    logret = np.log(close / close.shift(1))
    return logret.rolling(window, min_periods=window).std() * np.sqrt(ann)


def _daily_funding(funding: pd.DataFrame) -> pd.Series:
    """Aggregate hourly funding to a daily carry rate (fraction): mean(8h) * 3."""
    if funding.empty or "interest_8h" not in funding.columns:
        return pd.Series(dtype=float)
    by_date = funding.copy()
    by_date.index = pd.DatetimeIndex(by_date.index).floor("D")
    fund = by_date.groupby(level=0)["interest_8h"].mean() * 3.0
    fund.name = "funding_daily"
    return fund


def compute_asset(dvol: pd.DataFrame, perp: pd.DataFrame, funding: pd.DataFrame,
                  *, rv_windows=(10, 30, 90), skip_bars: int = 0,
                  iv_rv_ref_window: int = 30) -> pd.DataFrame:
    """Per-asset causal feature panel on a common daily date index.

    Columns: spot, atm_iv, rv_<w> for each window, iv_rv_spread, funding_daily.
    ``atm_iv`` and ``iv_rv_spread`` are in annualized **fractions** (0.55 = 55%).
    ``skip_bars`` optionally lags the *signal* columns (not spot) by N bars to
    avoid same-bar microstructure; spot stays contemporaneous for PnL.
    """
    dvol_d = _to_date_index(dvol)
    perp_d = _to_date_index(perp)

    df = pd.DataFrame(index=perp_d.index.union(dvol_d.index).sort_values())
    df["spot"] = perp_d["close"].reindex(df.index)
    df["atm_iv"] = (dvol_d["close"].reindex(df.index)) / 100.0  # vol points -> fraction

    for w in rv_windows:
        df[f"rv_{w}"] = _realized_vol(df["spot"], w)

    df["iv_rv_spread"] = df["atm_iv"] - df[f"rv_{iv_rv_ref_window}"]

    fund = _daily_funding(funding)
    df["funding_daily"] = fund.reindex(df.index).fillna(0.0)

    # Drop rows lacking the core series (warmup / gaps).
    df = df.dropna(subset=["spot", "atm_iv"])

    if skip_bars > 0:
        signal_cols = ["atm_iv"] + [f"rv_{w}" for w in rv_windows] + ["iv_rv_spread"]
        df[signal_cols] = df[signal_cols].shift(skip_bars)

    return df


def compute(raw, *, rv_windows=(10, 30, 90), skip_bars: int = 0,
            iv_rv_ref_window: int = 30) -> dict[str, pd.DataFrame]:
    """Compute per-asset feature panels for every asset in a RawOptionsData.

    ``raw`` is a ``deribit_options_loader.RawOptionsData`` (duck-typed: needs
    ``.dvol``, ``.perp``, ``.funding`` dicts keyed by currency).
    """
    out: dict[str, pd.DataFrame] = {}
    for ccy in raw.dvol:
        out[ccy] = compute_asset(
            raw.dvol[ccy], raw.perp[ccy], raw.funding.get(ccy, pd.DataFrame()),
            rv_windows=rv_windows, skip_bars=skip_bars, iv_rv_ref_window=iv_rv_ref_window,
        )
        logger.info("Features %s: %d rows, IV-RV spread mean=%.4f",
                    ccy, len(out[ccy]), out[ccy]["iv_rv_spread"].mean())
    return out
