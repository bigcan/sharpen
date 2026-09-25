"""Commodity-area long-history proxies for the FinRL-X clean-window test (GLD, SLV, USO, DBC; DBA uncovered).

Builds DAILY TOTAL-RETURN INDEX LEVELS (base 1.0) on a weekday grid that stand in for the commodity ETFs
before they existed. Every source is free and needs no key:

  * LBMA daily fixes, JSON (``prices.lbma.org.uk``), USD per troy ounce: gold PM from 1968-04-01, gold AM
    (cross-check only) and silver from 1968-01-02. ``v`` = [USD, GBP, EUR].
  * EIA daily NYMEX futures, contracts 1 and 2 (nearest and next delivery): WTI (``RCLC1`` 1983-04-04,
    ``RCLC2`` 1985-01-02), NY Harbor heating oil (C1 1980, C2 1994-02-02), RBOB gasoline (2005-10-03),
    Henry Hub natural gas (1994-01-12); plus the Cushing WTI spot (``RWTC``, 1986-01-02). EIA stopped
    publishing futures after 2024-04-05, so every futures-based proxy ends there (the dev window has the ETFs).
  * Yahoo Finance chart API: ``^SPGSCI`` is the S&P GSCI **SPOT** index (no roll yield; its level equals the
    CME GSCI future GD=F). Verified on the dev window: it beats GSG (a GSCI TR tracker) by +10..15 bps/day on
    exactly business days 5-9, the GSCI roll window, i.e. it omits several %/yr of roll cost. Yahoo has no
    history for ^SPGSCITR / ^SPGSCIP (ER) or any agriculture index. ``GSG`` (2006+) is verification only.
  * Ken French daily ``RF`` (1-month T-bill, DECIMAL) from ``paths.FF_DAILY`` for futures collateral.

Proxies (the bare ticker column is the recommended candidate; alternatives are ``TICKER__candidate``):
  GLD  LBMA gold PM fix, price return, minus the ETF fee (GLD holds bullion: no carry).
  SLV  LBMA silver fix, price return, minus the ETF fee.
  USO  front-month WTI futures EXCESS return (EIA C1/C2), rolled into the next contract at the close
       ``ROLL_BDAYS_BEFORE_LTD`` NYMEX days before the front contract's last trading day (USO's own pre-2020
       practice: roll ~2 weeks before expiry), PLUS the daily T-bill return, minus the fee.
  USO__spot  Cushing spot, price return only. NOT a return proxy (no roll yield, no collateral).
  DBC  DBC-weighted composite of the components that have honest daily data: rolled front-month futures
       (+ T-bill) for WTI (also carrying Brent's weight), heating oil, RBOB and natural gas, and LBMA gold /
       silver bullion, at the DBIQ Optimum Yield Diversified base weights (``DBC_WEIGHTS``), rebalanced daily.
       A missing energy leg's weight moves to WTI (1985-01: WTI + gold + silver; 1994-02: + heating oil and
       natural gas; 2005-10: + RBOB). Industrial metals (12.5%) and grains/softs (22.5%) have no free daily
       history and are dropped, the rest renormalised: vol is ~1.5-1.9x DBC's.
  DBC__wti_prec  the 1985 composition held fixed throughout (WTI 84.6%, gold 12.3%, silver 3.1%).
  DBC__gsci_spot S&P GSCI spot + T-bill - fee. NOT honest (omits roll yield); diagnostic only.
  DBA  NOT BUILT (see ``UNCOVERED``).

Fees: ``FEE_BPS_YR`` equals ``configs/finrlx_strategy.yaml: costs.proxy_expense_bps_yr``, and ``pnl.py``
charges that expense on proxies too. Levels are NET of fees by default (``net_of_fees=True``, matching the
ETF prices for fidelity); a book that charges ``proxy_expense_bps_yr`` itself must load
``net_of_fees=False`` or it pays the fee twice.

THE SEAL. Dates <= ``seal.SEAL_END`` are sealed. This module stores full histories but never computes a
return, drift, volatility or trend statistic on sealed dates: the sealed part is touched only by parsing,
unit, date and contract-identity plumbing checks and by ``seal.hygiene`` counts. Levels are cumulated only
AFTER a seal view has been taken, so the dev window is rebased to 1.0 at its own first date and never
carries sealed growth. ``fidelity_commodity`` runs on the dev window only.

Run: ``python -m research.finrlx_strategy.data_commodity [--force]`` from ``scripts/``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from research.finrlx_strategy import paths, seal

logger = logging.getLogger(__name__)

AREA = "commodity"
RAW_DIR: Path = paths.DATA_ROOT / AREA
REPORT_FILE = "report_commodity.json"

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25
ROLL_BDAYS_BEFORE_LTD = 10              # USO rolled over 4 days starting ~2 weeks before expiry (pre-2020)
USO_RESTRUCTURE = pd.Timestamp("2020-04-01")   # USO left pure front-month in April 2020
RF_DAILY_MAX = 0.002                    # a DECIMAL daily bill return above this (~50%/yr) means percent units
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")
YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}?period1=0&period2=9999999999&interval=1d"
EIA_PET = "https://www.eia.gov/dnav/pet/hist_xls/{key}d.xls"
EIA_NG = "https://www.eia.gov/dnav/ng/hist_xls/{key}d.xls"

# Same values as configs/finrlx_strategy.yaml costs.proxy_expense_bps_yr (see module docstring).
FEE_BPS_YR: dict[str, float] = {"GLD": 40.0, "SLV": 50.0, "USO": 70.0, "DBC": 85.0}

# DBIQ Optimum Yield Diversified Commodity Index base weights (%), for the legs with free daily data.
# WTI carries Brent's 12.375 too. Not available anywhere free and daily: aluminium, zinc, copper (12.5) and
# corn, wheat, soybeans, sugar (22.5); they are dropped and the rest renormalised.
DBC_WEIGHTS: dict[str, float] = {"wti": 24.75, "ho": 12.375, "rbob": 12.375, "ng": 5.5, "gold": 8.0,
                                 "silver": 2.0}
ENERGY = ("wti", "ho", "rbob", "ng")
ENERGY_FALLBACK = "wti"
# futures leg -> (contract-1 source, contract-2 source, last-trading-day rule)
FUTURES: dict[str, tuple[str, str, str]] = {
    "wti": ("eia_wti_c1", "eia_wti_c2", "cl"),
    "ho": ("eia_ho_c1", "eia_ho_c2", "month_last"),
    "rbob": ("eia_rbob_c1", "eia_rbob_c2", "month_last"),
    "ng": ("eia_ng_c1", "eia_ng_c2", "ng"),
}

RECOMMENDED: dict[str, str] = {"GLD": "GLD", "SLV": "SLV", "USO": "USO", "DBC": "DBC"}
UNCOVERED: dict[str, str] = {
    "DBA": ("No free daily agriculture proxy before 2007: Yahoo has no history for S&P GSCI Agriculture "
            "(^SPGSAG/^SPGSAGTR/^SPGSAGP) or Dow Jones/Bloomberg ag indices; EIA has no agricultural futures; "
            "Yahoo continuous ag futures (ZC=F...) start 2000 and are unadjusted front-contract splices (roll "
            "gaps are fake returns); Stooq and Nasdaq Data Link CHRIS are behind bot walls; World Bank CMO is "
            "monthly AVERAGES (smoothed, spuriously autocorrelated) and must not be used as a return proxy. "
            "DBA has no honest daily proxy before its inception (2007-01-05)."),
}

# Plausible-range UNIT guards. A value outside means the series is in the wrong unit (cents, per kg, GBP,
# percent...) or mis-parsed; they are never used as data filters.
PRICE_RANGE: dict[str, tuple[float, float]] = {
    "lbma_gold_pm": (20.0, 20000.0), "lbma_gold_am": (20.0, 20000.0), "lbma_silver": (0.5, 1000.0),
    "eia_wti_spot": (-100.0, 500.0), "eia_wti_c1": (-100.0, 500.0), "eia_wti_c2": (-100.0, 500.0),
    "eia_ho_c1": (0.05, 20.0), "eia_ho_c2": (0.05, 20.0), "eia_rbob_c1": (0.05, 20.0),
    "eia_rbob_c2": (0.05, 20.0), "eia_ng_c1": (0.3, 100.0), "eia_ng_c2": (0.3, 100.0),
    "yahoo_spgsci": (10.0, 20000.0), "yahoo_gsg": (1.0, 1000.0),
}


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    kind: str        # lbma | eia | yahoo
    notes: str
    eia_key: str = ""
    eia_unit: str = ""


def _eia(key: str, eia_key: str, unit: str, notes: str, *, ng: bool = False) -> Source:
    url = (EIA_NG if ng else EIA_PET).format(key=eia_key)
    return Source(key, url, f"eia_{eia_key}d.xls", "eia", notes, eia_key=eia_key, eia_unit=unit)


_BBL, _GAL, _MMBTU = "Dollars per Barrel", "Dollars per Gallon", "Dollars per Million Btu"
_END = " EIA stopped publishing futures after 2024-04-05."
SOURCES: dict[str, Source] = {s.key: s for s in [
    Source("lbma_gold_pm", "https://prices.lbma.org.uk/json/gold_pm.json", "lbma_gold_pm.json", "lbma",
           "LBMA Gold Price PM (London 15:00), USD/oz = v[0]. GLD's NAV is struck on this price."),
    Source("lbma_gold_am", "https://prices.lbma.org.uk/json/gold_am.json", "lbma_gold_am.json", "lbma",
           "LBMA Gold Price AM (London 10:30), USD/oz = v[0]. Cross-check of the PM series only."),
    Source("lbma_silver", "https://prices.lbma.org.uk/json/silver.json", "lbma_silver.json", "lbma",
           "LBMA Silver Price (London 12:00), USD/oz = v[0]; USD missing on FX-closure days 1968-1973."),
    _eia("eia_wti_spot", "RWTC", _BBL, "Cushing OK WTI spot FOB. 2020-04-20 is negative (real)."),
    _eia("eia_wti_c1", "RCLC1", _BBL, "NYMEX WTI futures contract 1. 2020-04-20 is negative (real)." + _END),
    _eia("eia_wti_c2", "RCLC2", _BBL, "NYMEX WTI futures contract 2, from 1985-01-02." + _END),
    _eia("eia_ho_c1", "EER_EPD2F_PE1_Y35NY_DPG", _GAL, "NY Harbor No.2 heating oil futures contract 1 (1980+)."
         " Only rolled together with contract 2." + _END),
    _eia("eia_ho_c2", "EER_EPD2F_PE2_Y35NY_DPG", _GAL, "NY Harbor No.2 heating oil futures contract 2, from "
         "1994-02-02." + _END),
    _eia("eia_rbob_c1", "EER_EPMRR_PE1_Y35NY_DPG", _GAL, "NY Harbor RBOB gasoline futures contract 1, from "
         "2005-10-03 (conventional-gasoline futures are not on EIA's site)." + _END),
    _eia("eia_rbob_c2", "EER_EPMRR_PE2_Y35NY_DPG", _GAL, "NY Harbor RBOB gasoline futures contract 2." + _END),
    _eia("eia_ng_c1", "RNGC1", _MMBTU, "Henry Hub natural gas futures contract 1, from 1994-01-13." + _END,
         ng=True),
    _eia("eia_ng_c2", "RNGC2", _MMBTU, "Henry Hub natural gas futures contract 2, from 1994-01-12." + _END,
         ng=True),
    Source("yahoo_spgsci", YAHOO.format(sym="%5ESPGSCI"), "yahoo_SPGSCI.json", "yahoo",
           "Yahoo ^SPGSCI = S&P GSCI SPOT index (no roll yield): +10..15 bps/day vs GSG on GSCI roll days "
           "5-9 (dev window). History from 1984-01-03; the index launched 1991, earlier values are backfill."),
    Source("yahoo_gsg", YAHOO.format(sym="GSG"), "yahoo_GSG.json", "yahoo",
           "iShares S&P GSCI Commodity-Indexed Trust (tracks GSCI TR, fee 0.75%). Dev-window verification "
           "only; not a proxy."),
]}

# Clearly-wrong raw values, removed at parse time and recorded in the source manifest. Never a clamp.
FIXES: dict[str, list[dict[str, str]]] = {
    "lbma_silver": [{
        "date": "1983-02-05", "action": "drop",
        "reason": ("Saturday row (the London fixing never runs on weekends). USD 7.54 / GBP 4.097 are ~0.53x "
                   "the neighbouring fixes (USD 14.152 on Fri 02-04, 13.69 on Mon 02-07) and reverse the next "
                   "day: a keying error, not a price. Found via seal.hygiene big-move count."),
    }],
}


# ----------------------------------------------------------------------------------------------- fetch

def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _manifest_path(src: Source) -> Path:
    return RAW_DIR / f"{src.filename}.manifest.json"


def fetch_source(key: str, *, force: bool = False) -> dict[str, Any]:
    """Download one raw file into ``RAW_DIR`` (skipped when cached unless ``force``) and (re)write its manifest."""
    src = SOURCES[key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path, man_path = RAW_DIR / src.filename, _manifest_path(src)
    if raw_path.exists() and man_path.exists() and not force:
        raw = raw_path.read_bytes()
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if _sha256(raw) != man.get("sha256"):
            raise RuntimeError(f"{raw_path} changed on disk since it was fetched (sha256 mismatch); refetch "
                               "with force=True")
        logger.info("%s: cached (%s, fetched %s)", key, raw_path.name, man.get("fetched_at"))
    else:
        logger.info("%s: downloading %s", key, src.url)
        resp = requests.get(src.url, headers={"User-Agent": USER_AGENT}, timeout=120)
        resp.raise_for_status()
        raw = resp.content
        raw_path.write_bytes(raw)
        man = {"key": key, "url": src.url, "file": src.filename,
               "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "sha256": _sha256(raw), "bytes": len(raw), "http_status": resp.status_code}
    s = parse_source(key, raw, fetched_at=man["fetched_at"])
    man.update({"parsed_rows": int(len(s)), "first_date": str(s.index.min().date()),
                "last_date": str(s.index.max().date()), "units": _units(src), "notes": src.notes,
                "fixes": FIXES.get(key, []), "n_nonpositive": int((s <= 0).sum())})
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def fetch_all(*, force: bool = False) -> dict[str, dict[str, Any]]:
    return {k: fetch_source(k, force=force) for k in SOURCES}


def _units(src: Source) -> str:
    return {"lbma": "USD per troy ounce", "eia": src.eia_unit, "yahoo": "index points / USD price"}[src.kind]


# ----------------------------------------------------------------------------------------------- parse

def _check_dates(idx: pd.DatetimeIndex, what: str) -> None:
    if idx.has_duplicates:
        raise ValueError(f"{what}: duplicate dates {list(idx[idx.duplicated()][:5])}")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{what}: dates not increasing")


def parse_lbma(raw: bytes) -> pd.DataFrame:
    """LBMA fix JSON -> DataFrame[usd, gbp, eur] (per troy ounce); nulls become NaN."""
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("LBMA: expected a non-empty JSON list")
    recs = []
    for r in rows:
        v = r.get("v")
        if not isinstance(v, list) or not v:
            raise ValueError(f"LBMA: row without price vector: {r}")
        v = (list(v) + [None, None, None])[:3]
        recs.append((r["d"], *v))
    df = pd.DataFrame(recs, columns=["date", "usd", "gbp", "eur"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index("date")[["usd", "gbp", "eur"]].apply(pd.to_numeric, errors="coerce").astype(float)
    _check_dates(df.index, "LBMA")
    return df


def parse_eia_frame(frame: pd.DataFrame, eia_key: str, unit: str = _BBL) -> pd.Series:
    """EIA ``Data 1`` sheet read with ``header=None`` -> float Series in ``unit``.

    Layout: row 0 title, row 1 ``['Sourcekey', KEY]``, row 2 ``['Date', '<name> (<unit>)']``.
    """
    got = str(frame.iat[1, 1]).strip()
    if got != eia_key:
        raise ValueError(f"EIA: sourcekey {got!r} != expected {eia_key!r}")
    label = str(frame.iat[2, 1])
    if f"({unit})" not in label:
        raise ValueError(f"EIA {eia_key}: unit label {label!r} is not ({unit})")
    body = frame.iloc[3:, :2].dropna(how="all")
    idx = pd.DatetimeIndex(pd.to_datetime(body.iloc[:, 0]), name="date")
    s = pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").to_numpy(dtype=float), index=idx,
                  name=eia_key).dropna()
    _check_dates(s.index, f"EIA {eia_key}")
    return s


def parse_eia_xls(raw: bytes, eia_key: str, unit: str = _BBL) -> pd.Series:
    frame = pd.read_excel(io.BytesIO(raw), sheet_name="Data 1", header=None)
    return parse_eia_frame(frame, eia_key, unit)


def parse_yahoo_chart(raw: bytes, *, drop_on_or_after: pd.Timestamp | None = None) -> pd.Series:
    """Yahoo v8 chart JSON -> daily close keyed by the EXCHANGE-local date.

    ``drop_on_or_after`` (the fetch date) removes the in-progress session Yahoo appends as a live snapshot.
    """
    res = json.loads(raw)["chart"]["result"][0]
    meta = res["meta"]
    if meta.get("currency") not in (None, "USD"):
        raise ValueError(f"Yahoo {meta.get('symbol')}: currency {meta.get('currency')} is not USD")
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    ts = res.get("timestamp") or []
    close = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    s = pd.Series(pd.to_numeric(pd.Series(close, dtype="object"), errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(idx, name="date")).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if drop_on_or_after is not None:
        s = s[s.index < pd.Timestamp(drop_on_or_after).normalize()]
    return s


def _apply_fixes(s: pd.Series, fixes: list[dict[str, str]]) -> pd.Series:
    for fx in fixes:
        d = pd.Timestamp(fx["date"])
        if fx["action"] != "drop":
            raise ValueError(f"unknown fix action {fx['action']!r}")
        if d in s.index:
            s = s.drop(d)
    return s


def _check_range(s: pd.Series, key: str) -> None:
    lo, hi = PRICE_RANGE[key]
    bad = s[(s < lo) | (s > hi)]
    if len(bad):
        raise ValueError(f"{key}: {len(bad)} values outside unit range [{lo}, {hi}] (first {bad.index[0].date()}"
                         f" = {bad.iloc[0]}): wrong units or mis-parse")


def parse_source(key: str, raw: bytes, *, fetched_at: str | None = None) -> pd.Series:
    """Raw bytes -> clean price Series (USD units), with ``FIXES`` applied and unit ranges checked."""
    src = SOURCES[key]
    if src.kind == "lbma":
        s = parse_lbma(raw)["usd"].dropna()
    elif src.kind == "eia":
        s = parse_eia_xls(raw, src.eia_key, src.eia_unit)
    elif src.kind == "yahoo":
        cut = pd.Timestamp(fetched_at).tz_convert("America/New_York").tz_localize(None) if fetched_at else None
        s = parse_yahoo_chart(raw, drop_on_or_after=cut)
    else:
        raise ValueError(f"unknown source kind {src.kind}")
    s = _apply_fixes(s.rename(key), FIXES.get(key, []))
    _check_range(s, key)
    return s


def read_source(key: str) -> pd.Series:
    """Parsed price series of one source, full history (fetches it first if it is not cached)."""
    src = SOURCES[key]
    raw_path, man_path = RAW_DIR / src.filename, _manifest_path(src)
    if not (raw_path.exists() and man_path.exists()):
        fetch_source(key)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    return parse_source(key, raw_path.read_bytes(), fetched_at=man.get("fetched_at"))


# ------------------------------------------------------------------------------------------ return math

def load_rf(path: Path = paths.FF_DAILY) -> pd.Series:
    rf = pd.read_parquet(path)["RF"].astype(float)
    rf.index = pd.DatetimeIndex(rf.index, name="date")
    check_rf_units(rf)
    return rf


def check_rf_units(rf: pd.Series) -> None:
    """Ken French daily RF must be DECIMAL; the raw library files are PERCENT (100x too large)."""
    if (rf.abs() > RF_DAILY_MAX).any():
        raise ValueError(f"RF has |value| > {RF_DAILY_MAX}: looks like PERCENT units, expected decimal")


def rf_on_index(rf: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily bill return on a weekday grid: 0 on weekdays Ken French skips (US holidays: the month's bill
    return is spread over trading days), the last value carried past the file's end (dev tail only)."""
    check_rf_units(rf)
    out = rf.reindex(index).astype(float)
    inside = (index >= rf.index.min()) & (index <= rf.index.max())
    out[inside & out.isna().to_numpy()] = 0.0
    out[index > rf.index.max()] = float(rf.iloc[-1])
    return out


def price_returns(p: pd.Series) -> pd.Series:
    """Simple returns between consecutive POSITIVE observations (a non-positive print has no return)."""
    p = p[p > 0]
    return p.pct_change(fill_method=None).iloc[1:]


def place_returns(r_obs: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Book each return on its observation date of the weekday grid; grid days with no new observation get 0
    (level carried, no look-ahead); NaN outside [first, last] observation."""
    if len(r_obs) and (r_obs.index.dayofweek > 4).any():
        raise ValueError(f"observation on a weekend: {list(r_obs.index[r_obs.index.dayofweek > 4][:3])}")
    missing = ~r_obs.index.isin(index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} observations fall outside the grid")
    out = r_obs.reindex(index).astype(float)
    if len(r_obs):
        inside = (index >= r_obs.index.min()) & (index <= r_obs.index.max())
        out[inside & ~index.isin(r_obs.index)] = 0.0
    return out


def apply_fee(r: pd.Series, fee_bps_yr: float) -> pd.Series:
    """Deduct an annual fee accrued on calendar days: (1 + r) * (1 - fee)^(days / 365.25) - 1."""
    if fee_bps_yr == 0:
        return r
    days = r.index.to_series().diff().dt.days.fillna(1.0).to_numpy(dtype=float)
    factor = (1.0 - fee_bps_yr / 1e4) ** (days / DAYS_PER_YEAR)
    return (1.0 + r) * factor - 1.0


# ------------------------------------------------------------------------------- futures roll calendars

def _ext_calendar(trading_days: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(observed trading days, observed + plain weekdays for 45 days past the end) so the last month resolves."""
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days))))
    ext = days.union(pd.bdate_range(days[-1] + pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=45)))
    return days, ext


def cl_last_trade_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """NYMEX WTI (CL) last trading day in each month of the calendar.

    Rule: trading ends 3 business days before the 25th calendar day of the month preceding delivery; if the
    25th is not a business day, 3 business days before the last business day preceding the 25th. So
    D = last business day <= 25th and LTD = D minus 3 business days (business days = observed NYMEX days).
    """
    days, ext = _ext_calendar(trading_days)
    out = []
    for per in pd.period_range(days[0], days[-1], freq="M"):
        d25 = pd.Timestamp(per.year, per.month, 25)
        pos = int(ext.searchsorted(d25, side="right")) - 1
        if pos - 3 < 0 or ext[pos].to_period("M") != per:
            continue
        if days[0] <= ext[pos - 3]:
            out.append(ext[pos - 3])
    return pd.DatetimeIndex(out, name="ltd")


def nth_last_bday_dates(trading_days: pd.DatetimeIndex,
                        n_before_last: int | tuple[tuple[str | None, int], ...]) -> pd.DatetimeIndex:
    """Last trading day = the (n+1)-th last business day of each month. n=0: NY Harbor heating oil and RBOB
    (last business day of the month before delivery); n=2: Henry Hub natural gas today (3 business days
    before the first calendar day of the delivery month). ``n_before_last`` may be a regime table
    ``((last_month_inclusive | None, n), ...)`` for rules that changed over time."""
    days, ext = _ext_calendar(trading_days)
    regimes = ((None, n_before_last),) if isinstance(n_before_last, int) else n_before_last
    s = ext.to_series()
    out = []
    for per, sub in s.groupby(s.index.to_period("M")):
        n = next(k for end, k in regimes if end is None or per <= pd.Period(end, freq="M"))
        if per > days[-1].to_period("M") or len(sub) <= n:
            continue
        ltd = sub.iloc[-1 - n]
        if days[0] <= ltd:
            out.append(ltd)
    return pd.DatetimeIndex(out, name="ltd")


# Henry Hub natural gas expiry, identified from the EIA contract-1/2 continuity (no rulebook history found):
# 6 business days before the delivery month for expiries through 1995-12, 5 through 1997-01, 3 since (the
# current NYMEX rule). Sealed roll-day continuity with a >3% C1-C2 gap: 23/46 under the 3-day rule
# throughout vs 43/50 with this table (dev window 54/66 either way). Contract identity only: no returns.
NG_LTD_REGIMES: tuple[tuple[str | None, int], ...] = (("1995-12", 5), ("1997-01", 4), (None, 2))


def futures_ltd(rule: str, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if rule == "cl":
        return cl_last_trade_dates(trading_days)
    if rule == "month_last":
        return nth_last_bday_dates(trading_days, 0)
    if rule == "ng":
        return nth_last_bday_dates(trading_days, NG_LTD_REGIMES)
    raise ValueError(f"unknown LTD rule {rule!r}")


def rolled_front_returns(c1: pd.Series, c2: pd.Series, *, roll_bdays_before_ltd: int,
                         ltd: pd.DatetimeIndex | None = None) -> pd.Series:
    """Daily EXCESS return of holding the front future, switched into the next contract at the close
    ``roll_bdays_before_ltd`` exchange days before the front contract's last trading day (0 = at the LTD).

    Contract identity: on date t, C1 is the first contract whose LTD >= t, C2 the one after. The held contract
    at the close of t-1 is priced at t through whichever of C1/C2 it is on t, so a roll is never booked as a
    price move. Returns are on the dates where both C1 and C2 print (gaps merge two days into one return).
    ``ltd`` defaults to the WTI (CL) calendar.
    """
    cal = c1.index.union(c2.index)
    if ltd is None:
        ltd = cl_last_trade_dates(cal)
    ltd = pd.DatetimeIndex(sorted(ltd))
    both = pd.concat({"c1": c1, "c2": c2}, axis=1).dropna()
    t = both.index
    front = np.searchsorted(ltd.values, t.values, side="left")        # number of LTDs strictly before t
    cal_ext = cal.union(ltd)
    pos_t = cal_ext.get_indexer(t)
    has_ltd = front < len(ltd)
    cur_ltd = np.where(has_ltd, ltd.values[np.minimum(front, len(ltd) - 1)], np.datetime64("NaT"))
    pos_ltd = np.where(has_ltd, cal_ext.searchsorted(pd.DatetimeIndex(cur_ltd).fillna(t[-1])),
                       np.iinfo(np.int64).max)
    next_t = np.append(t.values[1:], np.datetime64("NaT"))
    expires_before_next = has_ltd & (next_t > cur_ltd) & ~pd.isna(next_t)
    hold_next = has_ltd & ((pos_t >= pos_ltd - roll_bdays_before_ltd) | expires_before_next)
    held = front + hold_next.astype(int)
    px = both[["c1", "c2"]].to_numpy(dtype=float)
    j_prev = held[:-1] - front[:-1]
    j_now = held[:-1] - front[1:]
    ok = (j_now >= 0) & (j_now <= 1)
    rows = np.arange(1, len(t))
    p_prev = px[rows - 1, j_prev]
    p_now = np.where(ok, px[rows, np.clip(j_now, 0, 1)], np.nan)
    good = ok & (p_prev > 0) & (p_now > 0)
    r = np.where(good, p_now / np.where(p_prev > 0, p_prev, np.nan) - 1.0, np.nan)
    n_bad = int((~good).sum())
    if n_bad:
        logger.warning("rolled_front_returns: %d days without a valid held-contract price (NaN)", n_bad)
    return pd.Series(r, index=t[1:], name="front_er")


def roll_alignment_counts(c1: pd.Series, c2: pd.Series, ltd: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    """Contract-identity plumbing check (COUNTS only): on a day the calendar says C1 changed contract, the new
    C1 should continue yesterday's C2 (|ln C1_t - ln C2_t-1| < |ln C1_t - ln C1_t-1|); on other days C1 should
    continue C1. Reported for the calendar as computed and with every LTD shifted by -1/+1 trading day; the
    test is only sharp when the C1-C2 gap is wide, so counts with a >3% gap are reported separately."""
    both = pd.concat({"c1": c1, "c2": c2}, axis=1).dropna()
    both = both[(both > 0).all(axis=1)]
    cal = c1.index.union(c2.index)
    base = cl_last_trade_dates(cal) if ltd is None else pd.DatetimeIndex(ltd)
    lc1, lc2 = np.log(both["c1"].to_numpy()), np.log(both["c2"].to_numpy())
    cross = np.abs(lc1[1:] - lc2[:-1])
    same = np.abs(lc1[1:] - lc1[:-1])
    wide = np.abs(lc2[:-1] - lc1[:-1]) > 0.03
    t = both.index
    sealed = (t[1:] <= seal.SEAL_END)
    out: dict[str, Any] = {}
    for name, shift in (("as_computed", 0), ("ltd_minus_1", -1), ("ltd_plus_1", 1)):
        pos = cal.get_indexer(base)
        pos = pos[pos >= 0] + shift
        pos = pos[(pos >= 0) & (pos < len(cal))]
        lt = cal[pos]
        front = np.searchsorted(lt.values, t.values, side="left")
        rolled = np.diff(front) == 1
        for part, m in (("sealed", sealed), ("dev", ~sealed)):
            p = f"{name}.{part}"
            out[f"{p}.n_roll_days"] = int((rolled & m).sum())
            out[f"{p}.n_roll_days_consistent"] = int((rolled & m & (cross < same)).sum())
            out[f"{p}.n_roll_days_gap_gt_3pct"] = int((rolled & m & wide).sum())
            out[f"{p}.n_roll_days_gap_gt_3pct_consistent"] = int((rolled & m & wide & (cross < same)).sum())
            out[f"{p}.n_other_days"] = int((~rolled & m).sum())
            out[f"{p}.n_other_days_consistent"] = int((~rolled & m & (same < cross)).sum())
    return out


def futures_excess_returns(leg: str) -> pd.Series:
    """Rolled front-month excess return of one ``FUTURES`` leg (dates where both EIA contracts print)."""
    k1, k2, rule = FUTURES[leg]
    c1, c2 = read_source(k1), read_source(k2)
    ltd = futures_ltd(rule, c1.index.union(c2.index))
    return rolled_front_returns(c1, c2, roll_bdays_before_ltd=ROLL_BDAYS_BEFORE_LTD, ltd=ltd).rename(leg)


# ------------------------------------------------------------------------------------------- proxies

def composite_returns(components: pd.DataFrame, weights: dict[str, float] = DBC_WEIGHTS) -> pd.Series:
    """Daily-rebalanced weighted return of total-return ``components`` (one column per ``weights`` key).

    On each day a missing energy leg's weight moves to ``ENERGY_FALLBACK`` (WTI); any other missing leg is
    dropped and the rest renormalised. NaN when the fallback leg itself is missing.
    """
    r = components[list(weights)]
    w = pd.DataFrame({k: np.full(len(r), v, dtype=float) for k, v in weights.items()}, index=r.index)
    for k in weights:
        if k == ENERGY_FALLBACK:
            continue
        miss = r[k].isna().to_numpy()
        if k in ENERGY:
            w.loc[miss, ENERGY_FALLBACK] += w.loc[miss, k]
        w.loc[miss, k] = 0.0
    out = (r.fillna(0.0) * w).sum(axis=1) / w.sum(axis=1)
    out[r[ENERGY_FALLBACK].isna()] = np.nan
    return out


def _component_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-history daily TOTAL returns of every building block on one weekday grid, plus the raw-price
    return columns used by diagnostics. Row-wise only: nothing is cumulated or summarised here."""
    gold, silver, spot = read_source("lbma_gold_pm"), read_source("lbma_silver"), read_source("eia_wti_spot")
    gsci = read_source("yahoo_spgsci")
    fut = {leg: futures_excess_returns(leg) for leg in FUTURES}
    grid = pd.bdate_range(min(gold.index.min(), silver.index.min()),
                          max(s.index.max() for s in (gold, silver, spot, gsci)), name="date")
    rf = rf_on_index(load_rf(), grid)
    comp = {leg: place_returns(r, grid) + rf for leg, r in fut.items()}     # futures: ER + collateral
    comp["gold"] = place_returns(price_returns(gold), grid)                 # bullion: price return
    comp["silver"] = place_returns(price_returns(silver), grid)
    diag = {"wti_spot": place_returns(price_returns(spot), grid),
            "gsci_spot_tr": place_returns(price_returns(gsci), grid) + rf}
    return pd.DataFrame(comp, index=grid), pd.DataFrame(diag, index=grid)


def _daily_returns(*, net_of_fees: bool = True) -> pd.DataFrame:
    """Full-history daily total returns of every proxy column. PRIVATE: callers must go through a seal view
    (``load_commodity``); nothing here cumulates or summarises returns."""
    comp, diag = _component_returns()

    def fee(t: str) -> float:
        return FEE_BPS_YR[t] if net_of_fees else 0.0

    fixed_1985 = comp.copy()
    fixed_1985[["ho", "rbob", "ng"]] = np.nan
    out = {
        "GLD": apply_fee(comp["gold"], fee("GLD")),
        "SLV": apply_fee(comp["silver"], fee("SLV")),
        "USO": apply_fee(comp["wti"], fee("USO")),
        "USO__spot": diag["wti_spot"],
        "DBC": apply_fee(composite_returns(comp), fee("DBC")),
        "DBC__wti_prec": apply_fee(composite_returns(fixed_1985), fee("DBC")),
        "DBC__gsci_spot": apply_fee(diag["gsci_spot_tr"], fee("DBC")),
    }
    return pd.DataFrame(out, index=comp.index)


def levels_from_returns(r: pd.DataFrame) -> pd.DataFrame:
    """Index levels = 1.0 on each column's first valid date of ``r`` (its return that day is not used)."""
    out = {}
    for c in r.columns:
        s = r[c].copy()
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None:
            out[c] = s
            continue
        s.loc[fv] = 0.0
        lvl = (1.0 + s).cumprod()
        lvl[(s.index < fv) | (s.index > lv)] = np.nan
        out[c] = lvl
    return pd.DataFrame(out, index=r.index)


def load_commodity(window: str, *, net_of_fees: bool = True) -> pd.DataFrame:
    """Proxy total-return index levels for ``window`` in {"dev", "backward"}.

    ``dev``: dates after SEAL_END, rebased to 1.0 at each column's first dev date. ``backward``: dates
    <= SEAL_END, base 1.0 at each column's first date; raises ``seal.SealedError`` until pre-registration.
    """
    if window not in ("dev", "backward"):
        raise ValueError(f"window must be 'dev' or 'backward', got {window!r}")
    rets = _daily_returns(net_of_fees=net_of_fees)
    view = seal.dev_view(rets) if window == "dev" else seal.backward_view(rets)
    return levels_from_returns(view)


# ------------------------------------------------------------------------------------ hygiene, fidelity

def hygiene_commodity() -> pd.DataFrame:
    """``seal.hygiene`` COUNTS for every raw source, full history. Non-positive prints (WTI 2020-04-20) are
    masked for the big-move count (``seal.hygiene`` would switch to dollar differences) and counted in
    ``n_nonpositive_raw``."""
    rows = []
    for key in SOURCES:
        s = read_source(key)
        h = seal.hygiene(s.where(s > 0).rename(key))
        h["n_nonpositive_raw"] = int((s <= 0).sum())
        h["n_fixes"] = len(FIXES.get(key, []))
        rows.append(h)
    return pd.concat(rows)


def gold_fix_crosscheck_counts(threshold: float = 0.05) -> dict[str, int]:
    """Same-day plumbing check (COUNTS only): days where the PM and AM gold fixes differ by more than
    ``threshold``. A keying error in one fix shows up here; a real intraday move rarely exceeds 5%."""
    pm, am = read_source("lbma_gold_pm"), read_source("lbma_gold_am")
    j = pd.concat({"pm": pm, "am": am}, axis=1).dropna()
    bad = (j["pm"] / j["am"] - 1.0).abs() > threshold
    sealed = j.index <= seal.SEAL_END
    return {"n_common_days": int(len(j)), "n_sealed_gt": int((bad & sealed).sum()),
            "n_dev_gt": int((bad & ~sealed).sum())}


def _etf_dev_closes(tickers: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(paths.ETF_PANEL)
    close = df[df["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="close").sort_index()
    close.index = pd.DatetimeIndex(close.index, name="date")
    return seal.dev_view(close)


def fidelity_metrics(proxy: pd.Series, etf: pd.Series) -> dict[str, Any]:
    """Proxy-vs-ETF tracking on their common dates (dev window only: raises on any sealed row)."""
    j = pd.concat({"p": proxy, "e": etf}, axis=1).dropna()
    seal.assert_no_sealed_rows(j)
    if len(j) < 30:
        raise ValueError(f"only {len(j)} common dates")
    d = j.pct_change(fill_method=None).dropna()
    w = j.resample("W-FRI").last().dropna().pct_change(fill_method=None).dropna()
    vol_p = float(d["p"].std() * np.sqrt(TRADING_DAYS))
    vol_e = float(d["e"].std() * np.sqrt(TRADING_DAYS))
    growth = float((j["p"].iloc[-1] / j["p"].iloc[0]) / (j["e"].iloc[-1] / j["e"].iloc[0]))
    years = (j.index[-1] - j.index[0]).days / DAYS_PER_YEAR
    return {"start": str(j.index[0].date()), "end": str(j.index[-1].date()), "n_days": int(len(d)),
            "n_weeks": int(len(w)), "corr_daily": float(d["p"].corr(d["e"])),
            "corr_weekly": float(w["p"].corr(w["e"])), "vol_proxy": vol_p, "vol_etf": vol_e,
            "vol_ratio": vol_p / vol_e, "te_ann": float((d["p"] - d["e"]).std() * np.sqrt(TRADING_DAYS)),
            "growth_ratio": growth, "ann_log_gap": float(np.log(growth) / years)}


def _gsci_roll_window_test(gsci_tr: pd.Series, gsg: pd.Series) -> dict[str, Any]:
    """Dev window, 2006-2019 contango era: mean daily (GSCI-spot TR - GSG) return in bps on GSCI roll days
    (business days 5-9 of the month) vs all other days. A spot index shows its missing roll yield there."""
    j = pd.concat({"g": gsci_tr, "s": gsg}, axis=1).dropna()
    seal.assert_no_sealed_rows(j)
    r = j.pct_change(fill_method=None).dropna().loc[:"2019-12-31"]
    d = (r["g"] - r["s"]) * 1e4
    bday = r.groupby([r.index.year, r.index.month]).cumcount().to_numpy() + 1
    roll = (bday >= 5) & (bday <= 9)
    return {"mean_bps_roll_days_5_9": float(d[roll].mean()), "mean_bps_other_days": float(d[~roll].mean()),
            "n_roll_days": int(roll.sum()), "n_other_days": int((~roll).sum())}


def fidelity_commodity() -> dict[str, Any]:
    """Dev-window fidelity of every proxy column vs its ETF, plus verification diagnostics."""
    lv = load_commodity("dev", net_of_fees=True)
    etf = _etf_dev_closes(["GLD", "SLV", "USO", "DBC", "DBA"])
    out: dict[str, Any] = {}
    for col in lv.columns:
        tk = col.split("__")[0]
        rec = {"vs": tk, "full_dev": fidelity_metrics(lv[col], etf[tk])}
        if tk == "USO":
            rec["dev_before_2020_04"] = fidelity_metrics(lv[col].loc[:USO_RESTRUCTURE - pd.Timedelta(days=1)],
                                                         etf[tk])
            rec["dev_from_2020_04"] = fidelity_metrics(lv[col].loc[USO_RESTRUCTURE:], etf[tk])
        out[col] = rec
    comp, diag = _component_returns()
    comp, diag = seal.dev_view(comp), seal.dev_view(diag)
    # DBC with the composition it has in 1994-2005 (no RBOB yet), measured on the dev window.
    set_1994 = comp.copy()
    set_1994["rbob"] = np.nan
    r94 = apply_fee(composite_returns(set_1994), FEE_BPS_YR["DBC"]).to_frame("x")
    out["DBC__diag_1994_2005_composition"] = {"vs": "DBC",
                                              "full_dev": fidelity_metrics(levels_from_returns(r94)["x"],
                                                                           etf["DBC"])}
    # DBA has no proxy: show what the DBC composite would give, as evidence it is not a stand-in.
    out["DBA__diag_dbc_composite"] = {"vs": "DBA", "full_dev": fidelity_metrics(lv["DBC"], etf["DBA"])}
    # GSCI identification: spot + T-bill - GSG's 0.75% fee vs GSG (a GSCI TR tracker).
    gsg = seal.dev_view(read_source("yahoo_gsg").to_frame("GSG"))["GSG"]
    gsci = levels_from_returns(apply_fee(diag["gsci_spot_tr"], 75.0).to_frame("x"))["x"]
    out["GSCI_spot_tr_vs_GSG"] = {"vs": "GSG", "full_dev": fidelity_metrics(gsci, gsg),
                                  "roll_window_test": _gsci_roll_window_test(gsci, gsg)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download even when cached")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manifests = fetch_all(force=args.force)
    hyg = hygiene_commodity()
    roll = {}
    for leg, (k1, k2, rule) in FUTURES.items():
        c1, c2 = read_source(k1), read_source(k2)
        roll[leg] = roll_alignment_counts(c1, c2, futures_ltd(rule, c1.index.union(c2.index)))
    rets = _daily_returns()
    comp, _ = _component_returns()
    coverage = {c: str(rets[c].first_valid_index().date()) for c in rets.columns}
    coverage.update({f"leg:{c}": str(comp[c].first_valid_index().date()) for c in comp.columns})
    last = {c: str(rets[c].last_valid_index().date()) for c in rets.columns}
    fid = fidelity_commodity()
    report = {"area": AREA, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "recommended": RECOMMENDED, "uncovered": UNCOVERED, "fee_bps_yr": FEE_BPS_YR,
              "dbc_weights": DBC_WEIGHTS, "roll_bdays_before_ltd": ROLL_BDAYS_BEFORE_LTD,
              "first_daily_return": coverage, "last_daily_return": last, "manifests": manifests,
              "hygiene": json.loads(hyg.to_json(orient="index", date_format="iso")),
              "roll_alignment_counts": roll, "gold_pm_am_crosscheck": gold_fix_crosscheck_counts(),
              "fidelity_dev": fid}
    (RAW_DIR / REPORT_FILE).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        logger.info("hygiene (counts only):\n%s", hyg)
    for leg, c in roll.items():
        logger.info("roll alignment %s: %s", leg, {k: v for k, v in c.items() if "gap_gt" in k})
    logger.info("first daily return: %s", coverage)
    logger.info("last daily return: %s", last)
    for k, v in fid.items():
        for part, m in v.items():
            if isinstance(m, dict) and "corr_daily" in m:
                logger.info("fidelity %-32s vs %-4s %-20s corr_d=%.3f corr_w=%.3f vol_ratio=%.3f te=%.3f "
                            "growth=%.3f gap=%+.4f/yr n=%d", k, v["vs"], part, m["corr_daily"],
                            m["corr_weekly"], m["vol_ratio"], m["te_ann"], m["growth_ratio"],
                            m["ann_log_gap"], m["n_days"])
            elif isinstance(m, dict):
                logger.info("fidelity %-32s %s %s", k, part, m)
    logger.info("report written to %s", RAW_DIR / REPORT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
