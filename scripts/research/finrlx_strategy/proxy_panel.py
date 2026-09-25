"""Assemble the 18 ETF proxies + cash into one panel on the US trading calendar, behind the seal.

Every value passes through the connectors' own ``load_*(window)`` (seal.dev_view / seal.backward_view), so the sealed
window stays sealed here too. Calendar = the SPY proxy's dates (NYSE sessions). Other series are forward-filled onto it
for at most ``FFILL_LIMIT`` sessions (foreign holidays, London vs New York), never before their own first value.
"""
from __future__ import annotations

import importlib
import logging

import pandas as pd

from .paths import TICKERS

log = logging.getLogger("finrlx.proxy_panel")
FFILL_LIMIT = 5
AREAS = {"equity": "load_equity", "rates": "load_rates", "fx": "load_fx", "commodity": "load_commodity"}


def _area_frames(window: str) -> dict[str, tuple[pd.DataFrame, dict]]:
    out = {}
    for area, fn in AREAS.items():
        mod = importlib.import_module(f"{__package__}.data_{area}")
        out[area] = (getattr(mod, fn)(window), dict(getattr(mod, "RECOMMENDED", {})))
    return out


def _pick(frames: dict[str, tuple[pd.DataFrame, dict]], ticker: str) -> tuple[pd.Series | None, str | None]:
    for area, (df, rec) in frames.items():
        col = ticker if ticker in df.columns else rec.get(ticker)
        if col is not None and col in df.columns:
            return df[col], f"{area}:{col}"
    return None, None


def load(window: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """(close levels [TICKERS], daily cash return, provenance). ``window`` is 'dev' or 'backward'."""
    frames = _area_frames(window)
    spy, src = _pick(frames, "SPY")
    if spy is None:
        raise KeyError("no SPY proxy")
    cal = spy.dropna().index
    cols, prov = {}, {}
    for t in TICKERS:
        s, src = _pick(frames, t)
        if s is None:
            cols[t] = pd.Series(float("nan"), index=cal)
            prov[t] = {"source": None, "first": None}
            continue
        s = s.dropna()
        first = s.index.min()
        aligned = s.reindex(cal.union(s.index)).ffill(limit=FFILL_LIMIT).reindex(cal)
        aligned[aligned.index < first] = float("nan")
        cols[t] = aligned
        prov[t] = {"source": src, "first": str(first.date()) if pd.notna(first) else None,
                   "n_ffilled": int(aligned.notna().sum() - s.reindex(cal).notna().sum())}
    close = pd.DataFrame(cols)[TICKERS]
    cash, csrc = _pick(frames, "CASH")
    if cash is None:
        raise KeyError("no CASH series")
    rf = cash.reindex(cal.union(cash.index)).ffill().reindex(cal).pct_change(fill_method=None).fillna(0.0)
    prov["CASH"] = {"source": csrc}
    return close, rf, prov
