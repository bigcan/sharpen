"""Equity-area long-history proxies for the FinRL-X clean-window test.

Builds DAILY TOTAL-RETURN INDEX LEVELS (base 1.0) that stand in for SPY, QQQ, IWM, EFA and EEM before
those ETFs existed, plus the benchmark ``SPX_TR`` (S&P 500 total return). Every source is free:

  * Ken French data library (CRSP / Bloomberg based, daily, PERCENT units, -99.99 / -999 missing codes);
  * Yahoo Finance chart API (index levels are PRICE-ONLY; the VEIEX mutual fund NAV is rebuilt from its
    close plus its dividend events);
  * Robert Shiller's monthly S&P 500 dividend series (``ie_data.xls``), used to accrue dividends onto the
    price-only ^GSPC with a one-month publication lag (the yield KNOWN at the date, no look-ahead).

THE SEAL. Dates <= ``seal.SEAL_END`` are sealed. This module downloads and stores full histories but never
computes a return, drift, volatility or trend statistic on sealed dates. Sealed data is touched only by
parsing / unit / date plumbing checks and ``seal.hygiene`` counts. ``fidelity_equity`` runs on the dev
window only. ``load_equity`` returns data only through ``seal.dev_view`` / ``seal.backward_view``.

Column naming: ``TICKER__candidate`` for each candidate; the bare ``TICKER`` column is the recommended
candidate (see ``RECOMMENDED``). ``SPY`` is the same series as ``SPX_TR``.

Recommendation rule (declared before any fidelity number was computed): among a ticker's candidates that
cover the whole clean window (first valid <= ``CLEAN_START``), take the lowest dev-window annualised daily
tracking error vs the ETF; if no candidate covers the clean window, take the earliest-starting one.
``SPX_TR`` is S&P 500 by definition, so it is restricted to S&P-based candidates (``gspc_shiller``);
``kf_mkt`` (CRSP value-weighted total market, "KF_MKT_TR") is kept as the alternative.

Run: ``python -m research.finrlx_strategy.data_equity [--force]`` from ``scripts/``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from research.finrlx_strategy import paths, seal

logger = logging.getLogger(__name__)

AREA = "equity"
RAW_DIR: Path = paths.DATA_ROOT / AREA
LEVELS_FILE = "levels_equity.parquet"
REPORT_FILE = "report_equity.json"

CLEAN_START = pd.Timestamp("1973-01-02")   # first NYSE session of the clean window
KF_PCT = 100.0                              # Ken French files are in percent
KF_MISSING = (-99.99, -999.0)
TRADING_DAYS = 252
DIV_LAG_MONTHS = 1                          # dividend yield applied in month m is Shiller's month m-1
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")

KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
KF_SOURCES: dict[str, str] = {
    "kf_ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "kf_ind5_daily": "5_Industry_Portfolios_daily_CSV.zip",
    "kf_me_daily": "Portfolios_Formed_on_ME_Daily_CSV.zip",
    "kf_devxus3_daily": "Developed_ex_US_3_Factors_Daily_CSV.zip",
}
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
YAHOO_SOURCES: dict[str, str] = {
    "yh_gspc": "^GSPC", "yh_sp500tr": "^SP500TR", "yh_ndx": "^NDX", "yh_ixic": "^IXIC",
    "yh_rut": "^RUT", "yh_veiex": "VEIEX",
}
YAHOO_PERIOD1 = -2208988800                 # 1900-01-01; Yahoo returns from each symbol's first bar
SHILLER_PAGE = "https://shillerdata.com/"
SHILLER_FALLBACK = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"   # frozen at 2023-09

VW_SECTION = "Average Value Weighted Returns -- Daily"
ME_CANDIDATES: dict[str, str] = {        # declared set for IWM (value-weighted, NYSE breakpoints)
    "IWM__kf_lo30": "Lo 30", "IWM__kf_qnt2": "Qnt 2", "IWM__kf_dec2": "Dec 2",
    "IWM__kf_dec3": "Dec 3", "IWM__kf_dec4": "Dec 4", "IWM__kf_dec5": "Dec 5",
}
ETF_OF_BENCH = {"SPX_TR": "SPY"}

# Recommended candidate per ticker (derived by the rule in the module docstring; see report_equity.json
# "recommended_by_rule", which re-derives it from the numbers so a drift is visible).
RECOMMENDED: dict[str, str] = {
    "SPX_TR": "SPX_TR__gspc_shiller",
    "SPY": "SPX_TR__gspc_shiller",
    "QQQ": "QQQ__ixic",
    "IWM": "IWM__kf_qnt2",
    "EFA": "EFA__kf_devxus",
    "EEM": "EEM__veiex",
}

CANDIDATE_NOTES: dict[str, str] = {
    "SPX_TR__gspc_shiller": "^GSPC price return + Shiller D/P (month m-1) / 252 per trading day. "
                            "Monthly-smoothed dividends (no ex-date lumpiness); no fees.",
    "SPX_TR__kf_mkt": "KF_MKT_TR = KF Mkt-RF + RF: CRSP value-weighted US total market (NYSE/AMEX/NASDAQ), "
                      "total return. Broader than the S&P 500 (includes mid/small caps).",
    "SPX_TR__sp500tr": "Yahoo ^SP500TR (official S&P 500 total return), 1988+. Reference only.",
    "QQQ__kf_hitec": "KF 5-industry HiTec value-weighted total return (business equipment, telecom). "
                     "Excludes Nasdaq-100 retail/biotech/consumer names (AMZN, COST...), includes NYSE tech.",
    "QQQ__ndx": "^NDX Nasdaq-100 PRICE-ONLY (no dividends), from 1985-10-01.",
    "QQQ__ixic": "^IXIC Nasdaq Composite PRICE-ONLY (no dividends), from 1971-02-05. Pre-1990s it is a "
                 "broad OTC small/mid-cap index, not a mega-cap tech index.",
    "IWM__rut": "^RUT Russell 2000 PRICE-ONLY, from 1987-09-10. Reference only.",
    "EFA__kf_devxus": "KF Developed ex US Mkt-RF + RF (USD, total return, Bloomberg data), from 1990-07. "
                      "Includes Canada (EAFE does not); local-close timing (non-synchronous with EFA).",
    "EEM__veiex": "Vanguard Emerging Markets Stock Index fund (VEIEX) NAV + Yahoo dividend events, from "
                  "1994-05. Net of fund expenses (~0.3-0.6%/yr); tracked MSCI EM until 2013 (FTSE since).",
}
for _c, _lbl in ME_CANDIDATES.items():
    CANDIDATE_NOTES[_c] = (f"KF Portfolios_Formed_on_ME daily, value-weighted '{_lbl}' (NYSE breakpoints, "
                           "rebalanced end-June), total return. Not the Russell 2000 membership rule.")

# Clear data errors found by the hygiene investigation, applied at build time and logged in the manifest.
# Each entry: (column, date, action, reason). Empty: the 2026-09-25 investigation proved no level error.
FIXES: list[tuple[str, str, str, str]] = []

# Suspicious-but-not-fixed findings (stale prints move a return by a day; the level path is intact from the
# next observation, and the true value is unknown, so nothing is overwritten). Written into the manifests.
DATA_FLAGS: list[dict[str, str]] = [
    {"source": "yh_ndx", "column": "QQQ__ndx", "date": "2000-12-18",
     "issue": "stale print: 0.00% while ^IXIC -1.08% and KF HiTec -1.91%; the move lands on 2000-12-19. "
              "Not fixed (level path intact from 2000-12-19)."},
    {"source": "yh_ndx", "column": "QQQ__ndx", "date": "1988-10-19,1989-09-25,2001-07-02",
     "issue": "isolated 0.00% prints on days the S&P moved >0.5%; small sibling moves; not fixed."},
    {"source": "yh_ixic", "column": "QQQ__ixic",
     "date": "1972-08-22,1976-05-05,1984-11-09,1987-06-15,1987-12-01,1987-12-02,1988-06-07,1988-07-01,"
             "1990-10-23",
     "issue": "isolated 0.00% prints (max run 2) on days the S&P moved 0.5-0.8%; likely stale prints of "
              "one day; not fixed."},
    {"source": "yh_rut", "column": "IWM__rut", "date": "1988-07-18,1990-03-21,2002-04-09,2018-05-15,2019-07-01",
     "issue": "isolated 0.00% prints; 2018-05-15 matches IWM (+0.02%), 2019-07-01 does not (IWM +0.39%); "
              "not fixed (reference column only)."},
    {"source": "yh_veiex", "column": "EEM__veiex", "date": "1995-09-05..1995-09-08",
     "issue": "4-day stale NAV run (0.00% x4 while KF Developed ex US moved every day); the move lands on "
              "1995-09-11. Other 0.00% days (6-18/yr pre-2006) match the 2-decimal NAV rounding. Not fixed."},
    {"source": "kf_ff3_daily,kf_ind5_daily,kf_me_daily,kf_devxus3_daily", "column": "all KF columns",
     "date": "-", "issue": "exact-zero days (44-155 per series, max run 2) are consistent with the 0.01% "
                           "rounding of the percent values; not stale."},
]


# ----------------------------------------------------------------------------------------------------------
# download + manifest plumbing
# ----------------------------------------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _http_get(url: str, timeout: int = 90) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.content


def _manifest_path(raw_dir: Path, source: str) -> Path:
    return raw_dir / f"{source}.manifest.json"


def _write_manifest(raw_dir: Path, source: str, fields: dict[str, Any]) -> None:
    p = _manifest_path(raw_dir, source)
    p.write_text(json.dumps(fields, indent=2, default=str), encoding="utf-8")


def _read_manifest(raw_dir: Path, source: str) -> dict[str, Any] | None:
    p = _manifest_path(raw_dir, source)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cached(raw_dir: Path, source: str, raw_name: str) -> Path | None:
    """Return the cached raw path if it and its manifest exist and the bytes match the manifest sha256."""
    raw, man = raw_dir / raw_name, _read_manifest(raw_dir, source)
    if not raw.exists() or man is None:
        return None
    got = _sha256(raw.read_bytes())
    if got != man.get("sha256"):
        raise RuntimeError(f"{source}: cached {raw_name} sha256 {got[:12]} != manifest "
                           f"{str(man.get('sha256'))[:12]}; re-fetch with force=True")
    return raw


def _date_span(df: pd.DataFrame | pd.Series) -> tuple[str | None, str | None]:
    if len(df) == 0:
        return None, None
    return str(df.index.min().date()), str(df.index.max().date())


def _store(raw_dir: Path, source: str, raw_name: str, url: str, content: bytes,
           parsed: pd.DataFrame, notes: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / raw_name
    raw.write_bytes(content)
    first, last = _date_span(parsed)
    _write_manifest(raw_dir, source, {
        "source": source, "url": url, "fetched_at": _utcnow(), "sha256": _sha256(content),
        "raw_file": raw_name, "bytes": len(content), "parsed_rows": int(len(parsed)),
        "first_date": first, "last_date": last, "notes": notes, "fixes": [],
    })
    logger.info("fetched %s: %d rows %s..%s (%s)", source, len(parsed), first, last, url)
    return raw


def fetch_kf(source: str, *, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    raw_name = f"{source}.zip"
    if not force and (hit := _cached(raw_dir, source, raw_name)) is not None:
        return hit
    url = KF_BASE + KF_SOURCES[source]
    content = _http_get(url)
    sections = parse_kf_csv(_kf_text(content))
    first = next(iter(sections.values()))
    notes = (f"Ken French {KF_SOURCES[source]}; sections={list(sections)}; values PERCENT in the file, "
             f"stored parse divides by {KF_PCT:g}; missing codes {KF_MISSING} -> NaN")
    return _store(raw_dir, source, raw_name, url, content, first, notes)


def fetch_yahoo(source: str, *, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    raw_name = f"{source}.json"
    if not force and (hit := _cached(raw_dir, source, raw_name)) is not None:
        return hit
    sym = YAHOO_SOURCES[source]
    period2 = int(datetime.now(timezone.utc).timestamp())
    url = (YAHOO_CHART.format(sym=requests.utils.quote(sym)) +
           f"?period1={YAHOO_PERIOD1}&period2={period2}&interval=1d&events=div%2Csplit%2CcapitalGains")
    content = _http_get(url)
    parsed = parse_yahoo_chart(content)
    notes = (f"Yahoo chart API {sym}; close is split-adjusted, NOT dividend-adjusted; "
             f"dividend events={int((parsed['dividend'] > 0).sum())}; bars stamped by exchange-local date")
    return _store(raw_dir, source, raw_name, url, content, parsed, notes)


def _shiller_url() -> str:
    """The live ie_data.xls link is on shillerdata.com (its ?ver= changes); fall back to the frozen Yale copy."""
    try:
        html = _http_get(SHILLER_PAGE).decode("utf-8", "replace")
        m = re.search(r'(//img1\.wsimg\.com/[^"\']+/ie_data\.xls[^"\']*)', html)
        if m:
            return "https:" + m.group(1)
    except requests.RequestException as exc:
        logger.warning("shillerdata.com unreachable (%s); using the frozen Yale copy", exc)
    return SHILLER_FALLBACK


def fetch_shiller(*, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    source, raw_name = "shiller_ie_data", "shiller_ie_data.xls"
    if not force and (hit := _cached(raw_dir, source, raw_name)) is not None:
        return hit
    url = _shiller_url()
    content = _http_get(url)
    parsed = parse_shiller_xls(content)
    last_d = parsed["D"].last_valid_index()
    notes = (f"Shiller monthly S&P composite P (monthly avg of daily closes) and D (annualised dividends, "
             f"index points). Last month with D: {last_d.date() if last_d is not None else None}. "
             f"Date column is YYYY.MM (1871.1 = October).")
    return _store(raw_dir, source, raw_name, url, content, parsed, notes)


def fetch_all(*, force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for s in KF_SOURCES:
        out[s] = fetch_kf(s, force=force, raw_dir=raw_dir)
    for s in YAHOO_SOURCES:
        out[s] = fetch_yahoo(s, force=force, raw_dir=raw_dir)
    out["shiller_ie_data"] = fetch_shiller(force=force, raw_dir=raw_dir)
    return out


# ----------------------------------------------------------------------------------------------------------
# parsers (pure)
# ----------------------------------------------------------------------------------------------------------
def _kf_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the Ken French zip, got {z.namelist()}")
        return z.read(names[0]).decode("latin-1")


def parse_kf_csv(text: str) -> dict[str, pd.DataFrame]:
    """Split a Ken French CSV into its stacked tables; keep DAILY (YYYYMMDD) rows only; return DECIMALS.

    A table starts at a header line beginning with ',' and runs to the next blank line. Its title is the
    line directly above the header ('' when that line is blank, as in the single-table factor files). Monthly (YYYYMM) or annual (YYYY) tables are
    dropped. -99.99 / -999 become NaN before the percent-to-decimal division.
    """
    lines = text.splitlines()
    out: dict[str, pd.DataFrame] = {}
    i, last_text = 0, ""
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(","):
            cols = [c.strip() for c in line.split(",")[1:]]
            rows: list[list[str]] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                rows.append([t.strip() for t in lines[i].split(",")])
                i += 1
            daily = [r for r in rows if re.fullmatch(r"\d{8}", r[0])]
            if daily:
                if any(len(r) != len(cols) + 1 for r in daily):
                    raise ValueError(f"ragged rows in KF table '{last_text}'")
                idx = pd.to_datetime([r[0] for r in daily], format="%Y%m%d")
                vals = np.array([[float(v) for v in r[1:]] for r in daily], dtype=float)
                for code in KF_MISSING:
                    vals[np.isclose(vals, code)] = np.nan
                df = pd.DataFrame(vals / KF_PCT, index=idx, columns=cols)
                df.index.name = "date"
                if not df.index.is_monotonic_increasing or df.index.has_duplicates:
                    raise ValueError(f"KF table '{last_text}' dates not strictly increasing")
                if last_text in out:
                    raise ValueError(f"duplicate KF table title '{last_text}'")
                out[last_text] = df
            last_text = ""
            continue
        last_text = line                     # a title must sit directly above its header
        i += 1
    if not out:
        raise ValueError("no daily table found in Ken French CSV")
    return out


def parse_yahoo_chart(raw: bytes) -> pd.DataFrame:
    """Yahoo v8 chart JSON -> DataFrame[close, adjclose, dividend] indexed by exchange-local session date."""
    j = json.loads(raw)
    if j.get("chart", {}).get("error"):
        raise ValueError(f"Yahoo error: {j['chart']['error']}")
    res = j["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    idx = pd.DatetimeIndex(ts.tz_localize(None).normalize(), name="date")
    close = np.asarray(res["indicators"]["quote"][0]["close"], dtype=float)
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({"close": close,
                       "adjclose": np.asarray(adj, dtype=float) if adj is not None else np.nan},
                      index=idx)
    df["dividend"] = 0.0
    for ev in (res.get("events") or {}).get("dividends", {}).values():
        d = pd.Timestamp(pd.to_datetime(int(ev["date"]), unit="s", utc=True).tz_convert(tz)
                         .tz_localize(None).normalize())
        if d in df.index:
            df.loc[d, "dividend"] += float(ev["amount"])
        else:
            logger.warning("Yahoo dividend on %s has no bar; dropped", d.date())
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"].notna()].sort_index()
    # A snapshot taken during a session carries that session's in-progress bar: drop it (live-bar leak).
    meta = res["meta"]
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    rmt = meta.get("regularMarketTime")
    if rmt is not None and reg and reg.get("start", 0) <= rmt < reg.get("end", 0):
        d = pd.Timestamp(pd.to_datetime(rmt, unit="s", utc=True).tz_convert(tz).tz_localize(None)
                         .normalize())
        if d in df.index:
            logger.warning("dropping in-progress Yahoo bar %s", d.date())
            df = df.drop(index=d)
    return df


def _shiller_date(x: float) -> pd.Timestamp:
    """Shiller's YYYY.MM float: 1871.1 is OCTOBER (1871.10), 1871.01 is January."""
    year = int(np.floor(x))
    month = int(round((x - year) * 100))
    if not 1 <= month <= 12:
        raise ValueError(f"bad Shiller date {x!r}")
    return pd.Timestamp(year=year, month=month, day=1)


def parse_shiller_xls(raw: bytes) -> pd.DataFrame:
    """Shiller ie_data.xls 'Data' sheet -> monthly DataFrame[P, D] indexed by month start."""
    d = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None)
    hdr = d.index[(d[0].astype(str).str.strip() == "Date") & (d[1].astype(str).str.strip() == "P")]
    if len(hdr) != 1:
        raise ValueError("could not find the Shiller 'Date / P' header row")
    body = d.loc[hdr[0] + 1:, [0, 1, 2]]
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    idx = pd.DatetimeIndex([_shiller_date(float(v)) for v in body[0]], name="month")
    out = pd.DataFrame({"P": pd.to_numeric(body[1], errors="coerce").to_numpy(),
                        "D": pd.to_numeric(body[2], errors="coerce").to_numpy()}, index=idx)
    expect = pd.date_range(idx[0], idx[-1], freq="MS")
    if not idx.equals(expect):
        raise ValueError("Shiller months are not a gap-free monthly sequence")
    return out


# ----------------------------------------------------------------------------------------------------------
# construction (pure)
# ----------------------------------------------------------------------------------------------------------
def check_units(r: pd.Series, name: str) -> None:
    """Plumbing: a DECIMAL daily return of a diversified index is never beyond +/-100%.

    Counts only (allowed on the sealed window). A percent-for-decimal slip makes every >1% day a >100% day.
    """
    v = r.dropna()
    n_bad = int((v.abs() >= 1.0).sum())
    if n_bad:
        raise ValueError(f"{name}: {n_bad} daily returns with |r| >= 100%: units are not decimal")


def levels_from_returns(r: pd.Series) -> pd.Series:
    """Index level with base 1.0 on the first valid date; the first date's own return is the base."""
    v = r.loc[r.first_valid_index():] if r.first_valid_index() is not None else r.iloc[:0]
    check_units(v, str(r.name))
    n_gap = int(v.isna().sum())
    if n_gap:
        logger.warning("%s: %d missing daily returns inside the series (level set NaN, move lost)",
                       r.name, n_gap)
    lv = (1.0 + v.fillna(0.0)).cumprod()
    lv = lv / lv.iloc[0]
    lv[v.isna()] = np.nan
    return lv.rename(r.name)


def levels_from_prices(p: pd.Series) -> pd.Series:
    v = p.dropna()
    if (v <= 0).any():
        raise ValueError(f"{p.name}: non-positive price")
    return (v / v.iloc[0]).rename(p.name)


def kf_total_return(ff: pd.DataFrame) -> pd.Series:
    """Market total return = (Mkt - RF) + RF, both decimal."""
    return (ff["Mkt-RF"] + ff["RF"]).rename("kf_mkt")


def lagged_monthly_yield(shiller: pd.DataFrame, dates: pd.DatetimeIndex,
                         lag: int | None = None) -> pd.Series:
    """Dividend yield D/P KNOWN at each daily date: month m uses Shiller month m-lag (default 1).

    Months after the last published D forward-fill the last known yield (a flagged, dev-window-only fill).
    """
    lag = DIV_LAG_MONTHS if lag is None else lag
    y = (shiller["D"] / shiller["P"]).dropna()
    y.index = y.index + pd.DateOffset(months=lag)          # value becomes usable `lag` months later
    month = dates.to_period("M").to_timestamp()
    full = y.reindex(y.index.union(month.unique())).sort_index().ffill()
    n_fill = int((month > y.index.max()).sum())
    if n_fill:
        logger.info("dividend yield forward-filled past the last Shiller month on %d days (%s..)",
                    n_fill, dates[month > y.index.max()][0].date())
    return pd.Series(full.reindex(month).to_numpy(), index=dates, name="div_yield")


def price_plus_dividend_yield(close: pd.Series, shiller: pd.DataFrame) -> pd.Series:
    """Daily total return = price return + (lagged D/P) / 252."""
    c = close.dropna()
    y = lagged_monthly_yield(shiller, pd.DatetimeIndex(c.index))
    r = c.pct_change() + y / TRADING_DAYS
    return r.iloc[1:].rename(close.name)


def price_plus_cash_dividends(df: pd.DataFrame) -> pd.Series:
    """Fund total return from NAV and ex-date cash distributions: (close_t + div_t) / close_{t-1} - 1."""
    c, d = df["close"], df["dividend"].fillna(0.0)
    return ((c + d) / c.shift(1) - 1.0).iloc[1:]


NON_US_CALENDAR = ("EFA__kf_devxus",)      # Bloomberg ex-US data carries rows on US market holidays


def _us_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep weekday rows on which at least one US-calendar series has an observation.

    Levels are path-consistent, so dropping a row folds its move into the next kept row: a Saturday
    session (pre-1952) lands in Monday, and an ex-US index move on a US holiday lands in the next US
    session, which is when a US-listed ETF such as EFA would reprice.
    """
    us = [c for c in df.columns if c not in NON_US_CALENDAR]
    keep = (df.index.dayofweek < 5) & df[us].notna().any(axis=1).to_numpy()
    return df.loc[keep]


def construct_levels(src: dict[str, Any]) -> pd.DataFrame:
    """Full-history levels (sealed part included) from parsed sources. PRIVATE: callers go via load_equity."""
    cols: dict[str, pd.Series] = {}
    ff = src["kf_ff3_daily"][""]
    cols["SPX_TR__kf_mkt"] = levels_from_returns(kf_total_return(ff))
    cols["SPX_TR__gspc_shiller"] = levels_from_returns(
        price_plus_dividend_yield(src["yh_gspc"]["close"].rename("gspc"), src["shiller_ie_data"]))
    cols["SPX_TR__sp500tr"] = levels_from_prices(src["yh_sp500tr"]["close"])
    cols["QQQ__kf_hitec"] = levels_from_returns(src["kf_ind5_daily"][VW_SECTION]["HiTec"])
    cols["QQQ__ndx"] = levels_from_prices(src["yh_ndx"]["close"])
    cols["QQQ__ixic"] = levels_from_prices(src["yh_ixic"]["close"])
    me = src["kf_me_daily"][VW_SECTION]
    for name, lbl in ME_CANDIDATES.items():
        cols[name] = levels_from_returns(me[lbl])
    cols["IWM__rut"] = levels_from_prices(src["yh_rut"]["close"])
    cols["EFA__kf_devxus"] = levels_from_returns(kf_total_return(src["kf_devxus3_daily"][""]))
    cols["EEM__veiex"] = levels_from_returns(price_plus_cash_dividends(src["yh_veiex"]))
    lv = pd.DataFrame({k: v.rename(k) for k, v in cols.items()})
    for col, date, action, reason in FIXES:
        logger.warning("FIX %s %s: %s (%s)", col, date, action, reason)
        if action == "nan":
            lv.loc[pd.Timestamp(date), col] = np.nan
    lv = _us_session_rows(lv).dropna(how="all")
    for tkr, cand in RECOMMENDED.items():
        lv[tkr] = lv[cand]
    lv.index.name = "date"
    return lv


def load_sources(raw_dir: Path = RAW_DIR, *, fetch_missing: bool = True) -> dict[str, Any]:
    if fetch_missing:
        fetch_all(raw_dir=raw_dir)
    src: dict[str, Any] = {}
    for s in KF_SOURCES:
        src[s] = parse_kf_csv(_kf_text((raw_dir / f"{s}.zip").read_bytes()))
    for s in YAHOO_SOURCES:
        src[s] = parse_yahoo_chart((raw_dir / f"{s}.json").read_bytes())
    src["shiller_ie_data"] = parse_shiller_xls((raw_dir / "shiller_ie_data.xls").read_bytes())
    return src


_LEVELS_CACHE: dict[str, pd.DataFrame] = {}


def _levels_full(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    key = str(raw_dir)
    if key not in _LEVELS_CACHE:
        _LEVELS_CACHE[key] = construct_levels(load_sources(raw_dir))
    return _LEVELS_CACHE[key]


# ----------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------
def load_equity(window: str) -> pd.DataFrame:
    """Daily total-return index levels. window='dev' (after SEAL_END) or 'backward' (sealed; raises)."""
    lv = _levels_full()
    if window == "dev":
        return seal.dev_view(lv)
    if window == "backward":
        return seal.backward_view(lv)
    raise ValueError(f"window must be 'dev' or 'backward', got {window!r}")


def hygiene_equity() -> pd.DataFrame:
    """seal.hygiene counts over the full history of every column, plus clean-window coverage flags."""
    lv = _levels_full()
    h = seal.hygiene(lv)
    h["covers_clean_start"] = h["first_valid"] <= CLEAN_START
    return h


def big_move_diagnostics(lv: pd.DataFrame, thr: float = 0.10) -> pd.DataFrame:
    """Data-error triage for |day change| > thr: dates only plus error-signature FLAGS, no performance.

    reverses: the next observation moves back by at least 80% of the jump (spike-and-revert signature);
    echoed: at least one other column in the frame also moved > 5% that day (a real market event).
    """
    ch = lv.pct_change(fill_method=None)
    big = ch.abs() > thr
    rows = []
    for col in lv.columns:
        if "__" not in col:
            continue                                   # recommended aliases duplicate a candidate
        for d in ch.index[big[col].to_numpy()]:
            i = ch.index.get_loc(d)
            nxt = ch[col].iloc[i + 1:].dropna()
            rev = bool(len(nxt) and np.sign(nxt.iloc[0]) == -np.sign(ch.at[d, col])
                       and abs(nxt.iloc[0]) >= 0.8 * abs(ch.at[d, col]) / (1 + abs(ch.at[d, col])))
            others = ch.loc[d].drop(labels=[c for c in lv.columns if c == col or "__" not in c])
            rows.append({"column": col, "date": d, "sealed": bool(d <= seal.SEAL_END), "reverses": rev,
                         "echoed": bool((others.abs() > 0.05).any())})
    return pd.DataFrame(rows)


def _etf_levels() -> pd.DataFrame:
    p = pd.read_parquet(paths.ETF_PANEL, columns=["date", "ticker", "close"])
    p = p[p["ticker"].isin(paths.UNIVERSE[AREA])]
    w = p.pivot(index="date", columns="ticker", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index, name="date")
    return seal.dev_view(w)


def pair_metrics(p: pd.Series, e: pd.Series) -> dict[str, Any]:
    """Proxy-vs-ETF fidelity on their common dates. Callers must pass dev-window data only."""
    seal.assert_no_sealed_rows(p)
    seal.assert_no_sealed_rows(e)
    both = pd.concat([p.rename("p"), e.rename("e")], axis=1).dropna()
    if len(both) < 60:
        return {"n_days": int(max(len(both) - 1, 0))}
    rd = both.pct_change().dropna()
    wk = both.resample("W-FRI").last().dropna().pct_change().dropna()
    g = (both["p"].iloc[-1] / both["p"].iloc[0]) / (both["e"].iloc[-1] / both["e"].iloc[0])
    return {
        "first": str(both.index[0].date()), "last": str(both.index[-1].date()), "n_days": int(len(rd)),
        "corr_daily": float(rd["p"].corr(rd["e"])), "corr_weekly": float(wk["p"].corr(wk["e"])),
        "vol_ratio": float(rd["p"].std() / rd["e"].std()),
        "vol_ratio_weekly": float(wk["p"].std() / wk["e"].std()),
        "te_ann": float((rd["p"] - rd["e"]).std() * np.sqrt(TRADING_DAYS)),
        "te_ann_weekly": float((wk["p"] - wk["e"]).std() * np.sqrt(52)),
        "growth_ratio": float(g), "growth_gap_ann": float(g ** (TRADING_DAYS / len(rd)) - 1.0),
    }


def _etf_of(col: str) -> str:
    base = col.split("__")[0]
    return ETF_OF_BENCH.get(base, base)


def recommend_by_rule(fid: dict[str, dict[str, Any]], first_valid: pd.Series) -> dict[str, str]:
    """Re-derive RECOMMENDED from dev fidelity + coverage (rule in the module docstring)."""
    out: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for col in fid:
        groups.setdefault(col.split("__")[0], []).append(col)
    for base, cands in groups.items():
        if base == "SPX_TR":
            cands = [c for c in cands if c == "SPX_TR__gspc_shiller"]
        full = [c for c in cands if first_valid[c] <= CLEAN_START and "te_ann" in fid[c]]
        if full:
            out[base] = min(full, key=lambda c: fid[c]["te_ann"])
        else:
            out[base] = min(cands, key=lambda c: first_valid[c])
    return out


def fidelity_equity() -> dict[str, Any]:
    """Proxy vs ETF (and SPX_TR candidates vs ^SP500TR) on the DEV window only."""
    lv = seal.dev_view(_levels_full())
    etf = _etf_levels()
    cands = [c for c in lv.columns if "__" in c]
    vs_etf = {c: pair_metrics(lv[c], etf[_etf_of(c)]) for c in cands}
    vs_tr = {c: pair_metrics(lv[c], lv["SPX_TR__sp500tr"])
             for c in ("SPX_TR__gspc_shiller", "SPX_TR__kf_mkt")}
    first_valid = seal.hygiene(_levels_full()[cands])["first_valid"]
    return {
        "window": "dev (dates > %s)" % seal.SEAL_END.date(),
        "vs_etf": vs_etf, "spx_tr_vs_sp500tr": vs_tr,
        "recommended": dict(RECOMMENDED),
        "recommended_by_rule": recommend_by_rule(vs_etf, first_valid),
    }


SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "kf_ff3_daily": ("SPX_TR__kf_mkt",), "kf_ind5_daily": ("QQQ__kf_hitec",),
    "kf_me_daily": tuple(ME_CANDIDATES), "kf_devxus3_daily": ("EFA__kf_devxus",),
    "yh_gspc": ("SPX_TR__gspc_shiller",), "shiller_ie_data": ("SPX_TR__gspc_shiller",),
    "yh_sp500tr": ("SPX_TR__sp500tr",), "yh_ndx": ("QQQ__ndx",), "yh_ixic": ("QQQ__ixic",),
    "yh_rut": ("IWM__rut",), "yh_veiex": ("EEM__veiex",),
}


def _columns_of(source: str) -> tuple[str, ...]:
    return SOURCE_COLUMNS.get(source, ())


def build_report(raw_dir: Path = RAW_DIR) -> dict[str, Any]:
    """Build, write the levels parquet + report JSON into the data dir, return the report."""
    lv = _levels_full(raw_dir)
    h = hygiene_equity()
    fid = fidelity_equity()
    diag = big_move_diagnostics(lv)
    lv.to_parquet(raw_dir / LEVELS_FILE)
    for source in list(KF_SOURCES) + list(YAHOO_SOURCES) + ["shiller_ie_data"]:
        man = _read_manifest(raw_dir, source)
        if man is not None:
            man["data_flags"] = [f for f in DATA_FLAGS if source in f["source"].split(",")]
            man["fixes"] = [list(f) for f in FIXES if f[0] in _columns_of(source)]
            _write_manifest(raw_dir, source, man)
    shiller_man = _read_manifest(raw_dir, "shiller_ie_data") or {}
    report = {
        "built_at": _utcnow(), "levels_file": str(raw_dir / LEVELS_FILE),
        "levels_sha256": _sha256((raw_dir / LEVELS_FILE).read_bytes()),
        "columns": list(lv.columns), "candidate_notes": CANDIDATE_NOTES, "fixes": FIXES,
        "data_flags": DATA_FLAGS,
        "hygiene": json.loads(h.to_json(orient="index", date_format="iso")),
        "big_moves": json.loads(diag.to_json(orient="records", date_format="iso")),
        "fidelity": fid, "shiller_url": shiller_man.get("url"),
    }
    (raw_dir / REPORT_FILE).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download every source")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetch_all(force=args.force)
    rep = build_report()
    logger.info("wrote %s and %s", rep["levels_file"], RAW_DIR / REPORT_FILE)


if __name__ == "__main__":
    main()
