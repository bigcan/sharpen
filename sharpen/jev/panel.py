"""The ATL x Jev screening price panel (Phase 2 step 6; architecture ADR-3, ADR-8).

Built once from the cached yfinance fetch of the S&P 500 point-in-time union and saved with a manifest; the
evaluation reads the saved panel and never repairs data in memory.

* **Firewall first (ADR-3).** Rows after ``screening_window[1]`` are dropped before anything else runs, so no
  cleaning window, ADV or repair can read a 2025+ price.
* **DATA-CLEAN (ADR-8).** The project cleaner (``cross_asset_loader._clean_wide``: outlier repair plus the
  stale-print scan) runs on the truncated prices. The frozen Tier-0 allows zero OHLC violations, so the remaining
  bars whose high or low excludes the open or close are bracketed, counted with the funnel's own
  :func:`~sharpen.signals.features.ohlc_violations` before and after.
* **Universe.** ``active`` = S&P 500 member (the audited as-of join) and priced. Trailing ADV excludes the
  current bar. Sectors are current GICS from the constituents file; former members share an Unknown bucket.
* **Honest metadata.** ``survivorship_free`` is False: the fetch could not price 270 delisted members, so every
  screening result is an UPPER BOUND.
* The trading calendar returned alongside the panel keeps the warm-up rows before ``screening_window[0]``, so a
  filing released before the first evaluated row is placed on its true release row, not on the first row.
"""
from __future__ import annotations

import csv
import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.crucible.data.us_equity_panel import _membership_matrix, _trailing_adv
from sharpen.data import cross_asset_loader as cal
from sharpen.signals.features import Panel, ohlc_violations

_FIELDS = ("open", "high", "low", "close", "volume")


def _rows(p: Panel, sel: np.ndarray) -> Panel:
    return dataclasses.replace(p, dates=p.dates[sel], open=p.open[sel], high=p.high[sel], low=p.low[sel],
                               close=p.close[sel], volume=p.volume[sel], active=p.active[sel],
                               adv_usd=p.adv_usd[sel])


def bracket_ohlc(p: Panel) -> tuple[Panel, int]:
    """Widen high/low to bracket open and close on the active, finite bars that violate it."""
    fin = np.isfinite(p.open) & np.isfinite(p.high) & np.isfinite(p.low) & np.isfinite(p.close) & p.active
    hi, lo = p.high.copy(), p.low.copy()
    top, bot = np.maximum(p.open, p.close), np.minimum(p.open, p.close)
    fix_hi, fix_lo = fin & (hi < top), fin & (lo > bot)
    hi[fix_hi], lo[fix_lo] = top[fix_hi], bot[fix_lo]
    return dataclasses.replace(p, high=hi, low=lo), int(fix_hi.sum() + fix_lo.sum())


def sector_ids(tickers: tuple[str, ...], constituents_csv: str | Path) -> tuple[np.ndarray, list[str]]:
    """Current GICS sector per ticker; tickers not in the constituents file share one Unknown bucket."""
    gics: dict[str, str] = {}
    with open(constituents_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gics[r["Symbol"].strip().upper().replace(".", "-")] = r["GICS Sector"].strip()
    names = sorted({s for s in gics.values() if s}) + ["Unknown"]
    idx = {s: i for i, s in enumerate(names)}
    return np.array([idx[gics.get(t, "Unknown") or "Unknown"] for t in tickers], dtype=int), names


def screening_panel(raw: Panel, phase1: Mapping, constituents_csv: str | Path) -> tuple[Panel, np.ndarray, dict]:
    """(screening panel, trading calendar through the screening end, build report)."""
    start, end = (np.datetime64(str(x), "D") for x in phase1["screening_window"])
    days = raw.dates.astype("datetime64[D]")
    upto = _rows(raw, days <= end)                      # firewall before any computation
    calendar = upto.dates.astype("datetime64[D]")

    wide = {f: pd.DataFrame(getattr(upto, f), index=pd.DatetimeIndex(upto.dates), columns=list(upto.tickers))
            for f in _FIELDS}
    wide, clean_report = cal._clean_wide(wide)
    arr = {f: np.asarray(wide[f].to_numpy(), dtype=np.float64) for f in _FIELDS}

    member = _membership_matrix(upto.dates, tuple(upto.tickers))
    priced = np.isfinite(arr["close"]) & (arr["close"] > 0)
    sid, sectors = sector_ids(tuple(upto.tickers), constituents_csv)
    meta = {**dict(raw.meta), "survivorship_free": False, "verdict_cap": "PROMISING",
            "universe_def": "S&P 500 PIT members (as-of join), priced", "screening_window": [str(start), str(end)],
            "data_clean": "cross_asset_loader._clean_wide + OHLC bracketing", "sectors": sectors}
    cleaned = Panel(upto.dates, tuple(upto.tickers), arr["open"], arr["high"], arr["low"], arr["close"],
                    arr["volume"], member & priced, _trailing_adv(arr["close"], arr["volume"]), sid, meta)
    evaluated = _rows(cleaned, calendar >= start)
    before = ohlc_violations(evaluated)
    evaluated, n_bracketed = bracket_ohlc(evaluated)
    after = ohlc_violations(evaluated)
    report = {"rows": int(evaluated.T), "tickers": int(evaluated.N), "calendar_days": int(calendar.size),
              "first_row": str(evaluated.dates[0])[:10], "last_row": str(evaluated.dates[-1])[:10],
              "active_names_per_day": round(float(evaluated.active.sum(1).mean()), 1),
              "clean": {"outliers_repaired": int(sum(r.get("outliers_repaired", 0) for r in clean_report.values())),
                        "stale_flagged": sorted(t for t, r in clean_report.items() if r.get("stale_flagged"))},
              "ohlc_violations_before": before, "ohlc_bracketed": n_bracketed, "ohlc_violations_after": after,
              "unknown_sector_names": int((sid == len(sectors) - 1).sum())}
    return evaluated, calendar, report
