"""FX-area long-history proxies for the FinRL-X clean-window test.

Builds DAILY TOTAL-RETURN INDEX LEVELS (base 1.0) that stand in for FXE, FXY, FXB, FXA and UUP before those
ETFs existed. Every source is free:

  * Federal Reserve H.10 (noon buying rates in New York), full-history SDMX XML bundle: EUR (1999+), JPY,
    GBP, AUD, CAD, SEK, CHF daily from 1971-01-04, plus the Fed dollar indexes used only as dev-window
    cross-checks (legacy Major Currencies index, AFE index);
  * Federal Reserve H.10 historical pages for the legacy euro-area currencies (DEM, FRF, ITL, NLG, BEF),
    1971-01-04 .. 1998-12-31, one HTML table per currency and vintage;
  * OECD Financial Market statistics (SDMX): monthly 3-month interbank rates (IR3TIB) and call-money
    rates (IRSTCI), percent per annum;
  * BIS central bank policy rates (bulk CSV), daily, percent per annum -- the last-resort foreign rate;
  * Ken French daily RF (1-month T-bill, DECIMAL per trading day), the cached project file
    ``paths.FF_DAILY`` -- the US rate for UUP.

QUOTING. H.10 series named ``RXI$US_*`` are USD per unit (EUR, GBP, AUD); ``RXI_*`` and every legacy page
are units per USD. Everything is normalised to ``units per USD`` (a rise = the dollar strengthens).

CONSTRUCTION (one daily step from business day t-1 to t, d = calendar days between them):
  * FX trust proxy (FXE/FXY/FXB/FXA): hold one unit of foreign currency on deposit.
        r_t = (S_t / S_{t-1}) * (1 + r_f(t-1) * d / basis) - 1 - fee * d / 365
    with S = USD per unit. ``TICKER__spot`` is the spot-only alternative (no carry, no fee).
    FXE before 1999: the German mark at the irrevocable 1.95583 DEM/EUR, German 3-month rate.
  * UUP proxy: ICE US Dollar Index (DXY) rebuilt from H.10 with the geometric formula
    (1999+: EUR 0.576, JPY 0.136, GBP 0.119, CAD 0.091, SEK 0.042, CHF 0.036 on units-per-USD; before
    1999 the ten-currency basket DEM 0.208, FRF 0.131, ITL 0.090, NLG 0.083, BEF 0.064 in place of EUR),
    then UUP = fully collateralised long USDX futures:
        r_t = DXY_t / DXY_{t-1} - 1 + (rf_US - rf_basket) [futures carry] + rf_US [T-bill collateral] - fee
    ``UUP__excess`` omits the collateral T-bill leg (the futures excess return); ``UUP__spot`` is DXY
    alone.

CAUSALITY (LEAK-2). A monthly rate averages the whole month, so the value for month m is used only for
accruals starting in month m+1; a daily policy rate is used from its own date. The accrual over
(t-1, t] uses the rate known at t-1. Ken French RF for month m is the 1-month bill yield at the end of
month m-1, so it is used as published.

THE SEAL. Dates <= ``seal.SEAL_END`` are sealed. This module downloads and stores full histories but never
computes a return, drift, volatility or trend statistic on sealed dates. Sealed data is touched only by
parsing / unit / date / join plumbing checks and ``seal.hygiene`` counts (plus dates of suspected data
errors). ``fidelity_fx`` runs on the dev window only. ``load_fx`` returns data only through
``seal.dev_view`` (re-based to 1.0 on the first dev day, so a dev level never encodes sealed growth) or
``seal.backward_view``.

Recommendation rule (declared before any fidelity number was computed): the bare ``TICKER`` column is
the carry candidate for FX trusts (the trusts earn foreign deposit interest) and the funded candidate
for UUP (the fund holds T-bills as collateral). Fidelity is reported for every candidate.

Run: ``python -m research.finrlx_strategy.data_fx [--force]`` from ``scripts/``.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from research.finrlx_strategy import paths, seal

logger = logging.getLogger(__name__)

AREA = "fx"
RAW_DIR: Path = paths.DATA_ROOT / AREA
LEVELS_FILE = "levels_fx.parquet"
REPORT_FILE = "report_fx.json"
BUILD_MANIFEST = "fx_build.manifest.json"
TICKERS = ("FXE", "FXY", "FXB", "FXA", "UUP")

CLEAN_START = pd.Timestamp("1973-01-02")
TRADING_DAYS = 252
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")

# ---------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------
H10_XML_URL = "https://www.federalreserve.gov/releases/h10/data/FRB_h10_xml.zip"
H10_LEGACY_URL = "https://www.federalreserve.gov/releases/h10/hist/dat{vintage}_{code}.htm"
LEGACY_CODES = {"DEM": "ge", "FRF": "fr", "ITL": "it", "NLG": "ne", "BEF": "be"}
LEGACY_VINTAGES = ("89", "96")          # 1971-01-04..1989-12-29 and 1990-01-01..1998-12-31
LEGACY_TITLE_UNIT = {"DEM": "DM/US$", "FRF": "FRANCS/US$", "ITL": "LIRA/US$",
                     "NLG": "GUILDERS/US$", "BEF": "FRANCS/US$"}
OECD_AREAS = ("AUS", "BEL", "CAN", "CHE", "DEU", "EA20", "FRA", "GBR", "ITA", "JPN", "NLD", "SWE", "USA")
OECD_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
            + "+".join(OECD_AREAS) + ".M.IR3TIB+IRSTCI.PA.....?startPeriod=1960-01&format=csv")
BIS_CBPOL_URL = "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip"
BIS_AREAS = ("AU", "BE", "CA", "CH", "DE", "FR", "GB", "IT", "JP", "NL", "SE", "US", "XM")

# H.10 daily series -> (currency, quote convention). "usd_per_unit" series are inverted to units per USD.
H10_FX_SERIES: dict[str, tuple[str, str]] = {
    "EUR": ("RXI$US_N.B.EU", "usd_per_unit"),
    "GBP": ("RXI$US_N.B.UK", "usd_per_unit"),
    "AUD": ("RXI$US_N.B.AL", "usd_per_unit"),
    "JPY": ("RXI_N.B.JA", "units_per_usd"),
    "CAD": ("RXI_N.B.CA", "units_per_usd"),
    "SEK": ("RXI_N.B.SD", "units_per_usd"),
    "CHF": ("RXI_N.B.SZ", "units_per_usd"),
}
H10_INDEX_SERIES = {"FED_MAJOR_LEGACY": "V0.JRXWTFN_N.B",   # Major Currencies (goods), Mar-1973=100, to 2019
                    "FED_AFE": "JRXWTFN_N.B",               # Advanced Foreign Economies, 2006+
                    "FED_BROAD": "JRXWTFB_N.B"}             # Broad, 2006+
H10_ND = -9999.0

# Wide plausibility bands, units per USD, over 1971-2026 -- a units check, not a statistic.
UNITS_PER_USD_BANDS: dict[str, tuple[float, float]] = {
    "EUR": (0.5, 1.5), "GBP": (0.25, 1.2), "AUD": (0.6, 2.2), "JPY": (70.0, 400.0),
    "CAD": (0.9, 1.7), "SEK": (3.5, 12.5), "CHF": (0.6, 4.5), "DEM": (1.3, 4.0), "FRF": (3.8, 11.5),
    "ITL": (500.0, 2500.0), "NLG": (1.4, 4.3), "BEF": (25.0, 75.0),
}

EURO_START = pd.Timestamp("1999-01-04")            # first H.10 euro quote
EURO_CONVERSION = {"DEM": 1.95583, "FRF": 6.55957, "ITL": 1936.27, "NLG": 2.20371, "BEF": 40.3399}

# ICE U.S. Dollar Index. 1999+ exponents are from the ICE USDX FAQ; the pre-1999 legacy split is the
# original (Fed, 1973, 1972-76 trade) basket and must sum to the euro's 0.576.
DXY_CONST = 50.14348112
DXY_WEIGHTS_EURO = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}
DXY_WEIGHTS_LEGACY = {"DEM": 0.208, "JPY": 0.136, "FRF": 0.131, "GBP": 0.119, "CAD": 0.091,
                      "ITL": 0.090, "NLG": 0.083, "BEF": 0.064, "SEK": 0.042, "CHF": 0.036}

# Foreign short rate, priority order (first available wins on each date).
# ("oecd", AREA, MEASURE) monthly percent p.a.; ("bis", AREA) daily percent p.a.
RATE_CHAIN: dict[str, list[tuple[str, ...]]] = {
    "EUR_PRE": [("oecd", "DEU", "IR3TIB"), ("oecd", "DEU", "IRSTCI"), ("bis", "DE")],
    "EUR_POST": [("oecd", "EA20", "IR3TIB"), ("oecd", "EA20", "IRSTCI"), ("bis", "XM")],
    "DEM": [("oecd", "DEU", "IR3TIB"), ("oecd", "DEU", "IRSTCI"), ("bis", "DE")],
    "FRF": [("oecd", "FRA", "IR3TIB"), ("oecd", "FRA", "IRSTCI"), ("bis", "FR")],
    "ITL": [("oecd", "ITA", "IR3TIB"), ("bis", "IT")],
    "NLG": [("oecd", "NLD", "IR3TIB"), ("bis", "NL")],
    "BEF": [("oecd", "BEL", "IR3TIB"), ("bis", "BE")],
    "JPY": [("oecd", "JPN", "IR3TIB"), ("oecd", "JPN", "IRSTCI"), ("bis", "JP")],
    "GBP": [("oecd", "GBR", "IR3TIB"), ("oecd", "GBR", "IRSTCI"), ("bis", "GB")],
    "AUD": [("oecd", "AUS", "IR3TIB"), ("oecd", "AUS", "IRSTCI"), ("bis", "AU")],
    "CAD": [("oecd", "CAN", "IR3TIB"), ("oecd", "CAN", "IRSTCI"), ("bis", "CA")],
    "SEK": [("oecd", "SWE", "IR3TIB"), ("oecd", "SWE", "IRSTCI"), ("bis", "SE")],
    "CHF": [("oecd", "CHE", "IR3TIB"), ("oecd", "CHE", "IRSTCI"), ("bis", "CH")],
}
DAY_BASIS = {"EUR": 360, "DEM": 360, "FRF": 360, "ITL": 360, "NLG": 360, "BEF": 360, "CHF": 360,
             "SEK": 360, "JPY": 365, "GBP": 365, "AUD": 365, "CAD": 365}
MONTHLY_MAX_STALE_DAYS = 40      # a month-m value (usable from day 1 of m+1) may serve at most ~1 month
DAILY_MAX_STALE_DAYS = 10
SPOT_FFILL_LIMIT = 5             # H.10 "ND" (New York holidays): carry the last quote at most 5 bdays
# Real market closures longer than the limit: the last quote is carried (no spot return accrues until
# the market reopens; the reopening day carries the whole move). All before the 1973 clean window.
KNOWN_MARKET_CLOSURES: list[tuple[str, str, str]] = [
    ("1971-08-16", "1971-08-31", "Nixon shock: FX markets closed / no NY noon quotes (JPY 11 bdays)"),
]
RATE_PCT_BAND = (-5.0, 60.0)     # percent p.a. plausibility band (units check, not an outlier filter)

# ETF sponsor fees, annual (ASSUMED from prospectus headline expense ratios).
FEES = {"FXE": 0.0040, "FXY": 0.0040, "FXB": 0.0040, "FXA": 0.0040, "UUP": 0.0078}
FX_TRUSTS = {"FXE": "EUR", "FXY": "JPY", "FXB": "GBP", "FXA": "AUD"}
RECOMMENDED = {"FXE": "carry", "FXY": "carry", "FXB": "carry", "FXA": "carry", "UUP": "funded"}

# Clear data errors found by the hygiene investigation (``find_isolated_spikes``): a one-day jump of
# >= 3% (log) in ONE currency that no other H.10 currency (NZD included) shares -- the largest
# same-direction move elsewhere is < 1/3 of it -- and that the next quote reverses by >= 70%. Each looks
# like a single mis-keyed digit. The quote is dropped (NaN) and the previous quote carried, so the move
# is realised on the next valid day. Real one-day jumps (devaluations, the Jan-2015 SNB unpeg, the
# Oct-2008 AUD/NZD moves, the 1971 Nixon-shock reopening) are NOT touched. JPY 1978-03-31 passes the
# screen but is NOT fixed: it is a Japanese fiscal year-end (JPY 2000-03-31 shows the same shape at 2%),
# so a real year-end flow cannot be ruled out. Every fix below sits inside a peg or band (Smithsonian,
# snake, EMS, SEK basket) or has a same-bloc cousin moving the other way. Applied at build time and
# logged in the build manifest.
_SPIKE = "isolated one-day spike reversed next day; no other currency co-moved (suspected keying error)"
DATA_FIXES: list[dict[str, str]] = [
    {"series": "CHF", "date": "1972-03-13", "action": "nan", "reason": _SPIKE},
    {"series": "DEM", "date": "1972-03-17", "action": "nan", "reason": _SPIKE},
    {"series": "CHF", "date": "1972-04-10", "action": "nan", "reason": _SPIKE},
    {"series": "SEK", "date": "1973-02-16", "action": "nan", "reason": _SPIKE},
    {"series": "SEK", "date": "1974-01-11", "action": "nan", "reason": _SPIKE + "; SEK was in the snake"},
    {"series": "BEF", "date": "1980-08-22", "action": "nan", "reason": _SPIKE + "; EMS band vs DEM"},
    {"series": "ITL", "date": "1982-02-03", "action": "nan", "reason": _SPIKE + "; EMS band vs DEM"},
    {"series": "FRF", "date": "1982-09-02", "action": "nan", "reason": _SPIKE + "; EMS band vs DEM"},
    {"series": "AUD", "date": "1988-12-08", "action": "nan", "reason": _SPIKE + "; NZD moved opposite"},
    {"series": "SEK", "date": "1991-02-13", "action": "nan", "reason": _SPIKE + "; SEK basket peg"},
]


# ---------------------------------------------------------------------------------------------------
# download + manifest plumbing
# ---------------------------------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _http_get(url: str, timeout: int = 180, tries: int = 3) -> bytes:
    last: Exception | None = None
    for _ in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            if not r.content:
                raise RuntimeError(f"empty body from {url}")
            return r.content
        except Exception as e:  # retried, then re-raised
            last = e
            logger.warning("GET %s failed: %s", url, e)
    raise RuntimeError(f"GET {url} failed after {tries} tries") from last


def _manifest_path(raw_dir: Path, source: str) -> Path:
    return raw_dir / f"{source}.manifest.json"


def _read_manifest(raw_dir: Path, source: str) -> dict[str, Any] | None:
    p = _manifest_path(raw_dir, source)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cached(raw_dir: Path, source: str, raw_name: str) -> Path | None:
    """The cached raw path if it and its manifest exist and the bytes match the manifest sha256."""
    raw, man = raw_dir / raw_name, _read_manifest(raw_dir, source)
    if not raw.exists() or man is None:
        return None
    got = _sha256(raw.read_bytes())
    if got != man.get("sha256"):
        raise RuntimeError(f"{source}: cached {raw_name} sha256 {got[:12]} != manifest "
                           f"{str(man.get('sha256'))[:12]} -- refetch with force=True")
    return raw


def _date_span(obj: pd.Series | pd.DataFrame) -> tuple[str | None, str | None]:
    idx = obj.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    if len(idx) == 0:
        return None, None
    return str(pd.Timestamp(idx.min()).date()), str(pd.Timestamp(idx.max()).date())


def _fetch(source: str, url: str, raw_name: str, parser: Any, notes: str, *, force: bool,
           raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        hit = _cached(raw_dir, source, raw_name)
        if hit is not None:
            logger.info("%s: cached %s", source, hit.name)
            return hit
    content = _http_get(url)
    parsed = parser(content)
    first, last = _date_span(parsed)
    (raw_dir / raw_name).write_bytes(content)
    man = {"source": source, "url": url, "fetched_at": _utcnow(), "sha256": _sha256(content),
           "bytes": len(content), "raw_file": raw_name, "parsed_rows": int(len(parsed)),
           "first_date": first, "last_date": last, "notes": notes}
    _manifest_path(raw_dir, source).write_text(json.dumps(man, indent=2), encoding="utf-8")
    logger.info("%s: fetched %d bytes, %d parsed rows, %s..%s", source, len(content), len(parsed),
                first, last)
    return raw_dir / raw_name


def fetch_h10_xml(*, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    return _fetch("h10_xml", H10_XML_URL, "FRB_h10_xml.zip", lambda b: parse_h10_xml(b)[0],
                  "Fed H.10 full-history SDMX bundle; daily (FREQ=9) series used; ND coded -9999",
                  force=force, raw_dir=raw_dir)


def fetch_h10_legacy(ccy: str, vintage: str, *, force: bool = False,
                     raw_dir: Path = RAW_DIR) -> Path:
    code = LEGACY_CODES[ccy]
    url = H10_LEGACY_URL.format(vintage=vintage, code=code)
    return _fetch(f"h10_legacy_{ccy}_{vintage}", url, f"h10_dat{vintage}_{code}.htm",
                  lambda b: parse_h10_legacy_html(b, ccy),
                  f"Fed H.10 historical page, {ccy} units per USD, 'ND' = no data",
                  force=force, raw_dir=raw_dir)


def fetch_oecd(*, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    return _fetch("oecd_finmark", OECD_URL, "oecd_finmark_ir.csv", parse_oecd_csv,
                  "OECD DF_FINMARK monthly IR3TIB (3m interbank) + IRSTCI (call money), percent p.a.",
                  force=force, raw_dir=raw_dir)


def fetch_bis(*, force: bool = False, raw_dir: Path = RAW_DIR) -> Path:
    return _fetch("bis_cbpol", BIS_CBPOL_URL, "WS_CBPOL_csv_flat.zip", parse_bis_cbpol,
                  "BIS central bank policy rates, daily, percent p.a.; last-resort foreign rate",
                  force=force, raw_dir=raw_dir)


def fetch_all(*, force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    out = {"h10_xml": fetch_h10_xml(force=force, raw_dir=raw_dir),
           "oecd_finmark": fetch_oecd(force=force, raw_dir=raw_dir),
           "bis_cbpol": fetch_bis(force=force, raw_dir=raw_dir)}
    for ccy in LEGACY_CODES:
        for v in LEGACY_VINTAGES:
            out[f"h10_legacy_{ccy}_{v}"] = fetch_h10_legacy(ccy, v, force=force, raw_dir=raw_dir)
    return out


# ---------------------------------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_h10_xml(raw: bytes) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Daily (FREQ=9) H.10 series as they are quoted, NaN for 'ND'. Accepts the zip or the bare XML.

    Returns (frame indexed by date, one column per SERIES_NAME; attributes per series).
    """
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.endswith("_data.xml"))
            raw = z.read(name)
    cols: dict[str, dict[pd.Timestamp, float]] = {}
    attrs: dict[str, dict[str, str]] = {}
    for _, el in ET.iterparse(io.BytesIO(raw), events=("end",)):
        if _local(el.tag) != "Series":
            continue
        a = dict(el.attrib)
        if a.get("FREQ") == "9":
            name = a["SERIES_NAME"]
            vals: dict[pd.Timestamp, float] = {}
            for ob in el:
                if _local(ob.tag) != "Obs":
                    continue
                v = float(ob.attrib["OBS_VALUE"])
                nd = ob.attrib.get("OBS_STATUS") == "ND" or v == H10_ND
                vals[pd.Timestamp(ob.attrib["TIME_PERIOD"])] = np.nan if nd else v
            cols[name] = vals
            desc = ""
            for ann in el.iter():
                if _local(ann.tag) == "AnnotationText":
                    desc = ann.text or ""
                    break
            attrs[name] = {**a, "description": desc}
        el.clear()
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.DatetimeIndex(df.index, name="date")
    return df, attrs


_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _legacy_date(s: str) -> pd.Timestamp:
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})", s)
    if not m:
        raise ValueError(f"bad H.10 legacy date {s!r}")
    yy = int(m.group(3))
    year = 1900 + yy if yy >= 50 else 2000 + yy
    return pd.Timestamp(year=year, month=_MONTHS[m.group(2).upper()], day=int(m.group(1)))


def parse_h10_legacy_html(raw: bytes, ccy: str) -> pd.Series:
    """One H.10 historical page -> Series of units per USD (NaN for 'ND'). Checks the page's unit title."""
    s = raw.decode("utf-8", errors="replace")
    title = re.search(r'summary="([^"]*)"', s)
    if not title or LEGACY_TITLE_UNIT[ccy] not in title.group(1):
        raise ValueError(f"{ccy}: legacy page title {title.group(1) if title else None!r} does not "
                         f"say {LEGACY_TITLE_UNIT[ccy]!r} -- quoting convention unverified")
    rows = re.findall(r'<th[^>]*axis="Date"[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>', s, flags=re.S)
    if not rows:
        raise ValueError(f"{ccy}: no rows parsed from legacy page")
    out: dict[pd.Timestamp, float] = {}
    for d, v in rows:
        d = html.unescape(re.sub(r"<[^>]+>", "", d)).replace("\xa0", "").strip()
        v = html.unescape(re.sub(r"<[^>]+>", "", v)).replace("\xa0", "").strip()
        out[_legacy_date(d)] = np.nan if v in ("ND", "") else float(v)
    ser = pd.Series(out, name=ccy, dtype=float).sort_index()
    ser.index = pd.DatetimeIndex(ser.index, name="date")
    if ser.index.has_duplicates:
        raise ValueError(f"{ccy}: duplicate dates in legacy page")
    return ser


def parse_oecd_csv(raw: bytes) -> pd.DataFrame:
    """OECD SDMX csv -> monthly frame (PeriodIndex 'M'), columns 'AREA:MEASURE', DECIMAL per annum."""
    d = pd.read_csv(io.BytesIO(raw))
    need = {"REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"}
    if not need <= set(d.columns):
        raise ValueError(f"OECD csv missing columns {need - set(d.columns)}")
    d = d[(d["FREQ"] == "M") & d["MEASURE"].isin(["IR3TIB", "IRSTCI"])]
    if not (d["UNIT_MEASURE"] == "PA").all():
        raise ValueError("OECD rates not all in percent per annum (UNIT_MEASURE != PA)")
    if "UNIT_MULT" in d.columns and not (d["UNIT_MULT"].fillna(0) == 0).all():
        raise ValueError("OECD UNIT_MULT != 0")
    d = d.assign(key=d["REF_AREA"] + ":" + d["MEASURE"],
                 period=pd.PeriodIndex(d["TIME_PERIOD"], freq="M"))
    wide = d.pivot_table(index="period", columns="key", values="OBS_VALUE", aggfunc="first")
    wide = wide.sort_index().astype(float)
    check_rate_units_pct(wide, "OECD")
    return wide / 100.0


def _col(d: pd.DataFrame, prefix: str) -> str:
    for c in d.columns:
        if c.split(":")[0] == prefix:
            return c
    raise ValueError(f"BIS csv has no {prefix} column")


def parse_bis_cbpol(raw: bytes) -> pd.DataFrame:
    """BIS flat csv (zip or bare) -> daily frame, columns = BIS area code, DECIMAL per annum."""
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), low_memory=False)
    fq, ar, tp = _col(d, "FREQ"), _col(d, "REF_AREA"), _col(d, "TIME_PERIOD")
    ov, um = _col(d, "OBS_VALUE"), _col(d, "UNIT_MEASURE")
    d = d[d[fq].astype(str).str.split(":").str[0].str.strip() == "D"].copy()
    d["area"] = d[ar].astype(str).str.split(":").str[0].str.strip()
    d = d[d["area"].isin(BIS_AREAS)]
    if not d[um].astype(str).str.contains("Per cent", case=False).all():
        raise ValueError("BIS policy rates not all 'Per cent per year'")
    d["date"] = pd.to_datetime(d[tp])
    wide = d.pivot_table(index="date", columns="area", values=ov, aggfunc="first").sort_index()
    wide = wide.astype(float)
    check_rate_units_pct(wide, "BIS")
    return wide / 100.0


def check_rate_units_pct(wide: pd.DataFrame, name: str) -> None:
    """Units check on a raw source: values must look like PERCENT per annum.

    Not an outlier filter: real crisis prints (Riksbank marginal rate 500% in Sep 1992) pass. Fails on
    DECIMAL input (most |v| below 0.5) or basis points (more than 1% of |v| above 100, or any above 1000).
    """
    v = wide.stack().dropna().abs()
    if not len(v):
        return
    if v.max() > 1000 or (v > 100).mean() > 0.01:
        raise ValueError(f"{name}: rates look like BASIS POINTS, expected percent p.a.")
    if len(v) > 50 and (v > 0.5).mean() < 0.5:
        raise ValueError(f"{name}: most rates below 0.5 -- looks like DECIMAL, expected percent")


def check_rate_units_decimal(r: pd.Series | pd.DataFrame, name: str) -> None:
    """Units check for a rate already converted to DECIMAL per annum."""
    v = r.stack().dropna() if isinstance(r, pd.DataFrame) else r.dropna()
    if len(v) and (v.min() < RATE_PCT_BAND[0] / 100 or v.max() > RATE_PCT_BAND[1] / 100):
        raise ValueError(f"{name}: decimal rate outside band -- percent/decimal units error?")


# ---------------------------------------------------------------------------------------------------
# spot panel
# ---------------------------------------------------------------------------------------------------
def to_units_per_usd(h10: pd.DataFrame) -> pd.DataFrame:
    """H.10 daily frame -> units of each currency per USD (inverts the USD-per-unit series)."""
    out = {}
    for ccy, (series, quote) in H10_FX_SERIES.items():
        s = h10[series].astype(float)
        out[ccy] = 1.0 / s if quote == "usd_per_unit" else s
    return pd.DataFrame(out)


def check_units_per_usd(upu: pd.DataFrame) -> None:
    """Units check: every quote inside its wide band (catches an inverted quote or a scale change)."""
    for c in upu.columns:
        v = upu[c].dropna()
        lo, hi = UNITS_PER_USD_BANDS[c]
        bad = v[(v < lo) | (v > hi)]
        if len(bad):
            raise ValueError(f"{c}: {len(bad)} quotes outside {lo}..{hi} units/USD, first "
                             f"{bad.index[0].date()} -- quoting or units error")


def business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, name="date")


def fill_holidays(s: pd.Series, idx: pd.DatetimeIndex, limit: int = SPOT_FFILL_LIMIT,
                  closures: list[tuple[str, str, str]] = KNOWN_MARKET_CLOSURES) -> pd.Series:
    """Reindex a quote to business days; carry the last quote across 'ND' days, at most ``limit``
    (without limit inside a documented market closure).

    Raises if a gap longer than ``limit`` business days remains inside the series' span.
    """
    v = s.dropna()
    out = v.reindex(idx).ffill(limit=limit)
    full = v.reindex(idx).ffill()
    for start, end, _ in closures:
        inside_closure = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        out[inside_closure] = full[inside_closure]
    inside = out.loc[v.index.min():v.index.max()]
    if inside.isna().any():
        first = inside.index[inside.isna()][0]
        raise ValueError(f"{s.name}: gap longer than {limit} business days at {first.date()}")
    return out


def euro_splice(upu: pd.DataFrame, legacy: dict[str, pd.Series]) -> pd.DataFrame:
    """Add the legacy currencies and a spliced EUR column (DEM at 1.95583 before 1999).

    After ``EURO_START`` each legacy currency is implied from the euro at its irrevocable rate, so the
    legacy DXY formula and the euro formula agree exactly from 1999.
    """
    out = upu.copy()
    eur = upu["EUR"]
    for ccy, s in legacy.items():
        k = EURO_CONVERSION[ccy]
        pre = s.loc[s.index < EURO_START]
        post = (eur * k).loc[eur.index >= EURO_START]
        out[ccy] = pd.concat([pre, post]).reindex(out.index)
    dem_as_eur = legacy["DEM"].loc[legacy["DEM"].index < EURO_START] / EURO_CONVERSION["DEM"]
    out["EUR"] = pd.concat([dem_as_eur, eur.loc[eur.index >= EURO_START]]).reindex(out.index)
    return out


def check_euro_join(upu: pd.DataFrame, tol: float = 0.05) -> None:
    """Plumbing check on the 1998-12-31 -> 1999-01-04 join: no units jump (no value is logged)."""
    for c in ["EUR", *EURO_CONVERSION]:
        s = upu[c].dropna()
        before = s.loc[s.index < EURO_START]
        after = s.loc[s.index >= EURO_START]
        if len(before) and len(after) and abs(np.log(after.iloc[0] / before.iloc[-1])) > tol:
            raise ValueError(f"{c}: level jump across the euro join exceeds {tol:.0%} -- units error")


# ---------------------------------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------------------------------
def monthly_to_daily_causal(m: pd.Series, idx: pd.DatetimeIndex,
                            max_stale: int = MONTHLY_MAX_STALE_DAYS) -> pd.Series:
    """Month-m average usable from day 1 of month m+1 (LEAK-2); stale beyond ``max_stale`` -> NaN."""
    m = m.dropna()
    if m.empty:
        return pd.Series(np.nan, index=idx)
    avail = (m.index.to_timestamp(how="start") + pd.offsets.MonthBegin(1))
    s = pd.Series(m.to_numpy(), index=pd.DatetimeIndex(avail))
    return _asof(s, idx, max_stale)


def daily_to_daily_causal(d: pd.Series, idx: pd.DatetimeIndex,
                          max_stale: int = DAILY_MAX_STALE_DAYS) -> pd.Series:
    return _asof(d.dropna(), idx, max_stale)


def _asof(s: pd.Series, idx: pd.DatetimeIndex, max_stale: int) -> pd.Series:
    if s.empty:
        return pd.Series(np.nan, index=idx)
    s = s.sort_index()
    stamp = pd.Series(s.index, index=s.index)
    val = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    st = stamp.reindex(stamp.index.union(idx)).ffill().reindex(idx)
    stale = (idx - pd.DatetimeIndex(st)).days > max_stale
    val[np.asarray(stale) | st.isna().to_numpy()] = np.nan
    return val


def rate_chain(ccy_key: str, oecd: pd.DataFrame, bis: pd.DataFrame,
               idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Causal daily foreign rate (decimal p.a.) and the source tag used on each date."""
    rate = pd.Series(np.nan, index=idx)
    tag = pd.Series("", index=idx, dtype=object)
    for src in RATE_CHAIN[ccy_key]:
        if src[0] == "oecd":
            col = f"{src[1]}:{src[2]}"
            cand = (monthly_to_daily_causal(oecd[col], idx) if col in oecd.columns
                    else pd.Series(np.nan, index=idx))
        else:
            cand = (daily_to_daily_causal(bis[src[1]], idx) if src[1] in bis.columns
                    else pd.Series(np.nan, index=idx))
        fill = rate.isna() & cand.notna()
        rate[fill] = cand[fill]
        tag[fill] = ":".join(src)
    check_rate_units_decimal(rate, ccy_key)
    return rate, tag


def foreign_rates(oecd: pd.DataFrame, bis: pd.DataFrame,
                  idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily causal foreign rates for every currency; EUR = German rate before 1999, euro area after."""
    rates, tags = {}, {}
    for ccy in RATE_CHAIN:
        if ccy.startswith("EUR_"):
            continue
        rates[ccy], tags[ccy] = rate_chain(ccy, oecd, bis, idx)
    pre, tpre = rate_chain("EUR_PRE", oecd, bis, idx)
    post, tpost = rate_chain("EUR_POST", oecd, bis, idx)
    is_post = idx >= EURO_START
    rates["EUR"] = pre.where(~is_post, post)
    tags["EUR"] = tpre.where(~is_post, tpost)
    return pd.DataFrame(rates), pd.DataFrame(tags)


def rate_segments(tags: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Metadata: which source serves each currency over which date range (no rate values)."""
    out: dict[str, list[dict[str, str]]] = {}
    for c in tags.columns:
        t = tags[c].fillna("")
        brk = t.ne(t.shift())
        segs = []
        for start in t.index[brk]:
            src = t.loc[start]
            nxt = t.index[brk & (t.index > start)]
            end = t.index[t.index < nxt[0]][-1] if len(nxt) else t.index[-1]
            segs.append({"source": src or "NONE (spot only)", "first": str(start.date()),
                         "last": str(end.date())})
        out[c] = segs
    return out


def us_rate_daily(ff: pd.DataFrame, idx: pd.DatetimeIndex) -> tuple[pd.Series, str | None]:
    """Ken French RF (DECIMAL per trading day) on the business-day grid.

    Non-trading weekdays get 0 (French spreads each month's bill return over its trading days, so the
    monthly total is preserved). After the file ends the last value is carried (dev window only).
    """
    rf = ff["RF"].astype(float)
    if rf.abs().max() > 0.01:
        raise ValueError("FF RF above 1% per day -- percent/decimal units error")
    out = rf.reindex(idx)
    last = rf.index.max()
    carried = None
    if idx.max() > last:
        out.loc[idx > last] = rf.iloc[-1]
        carried = str(last.date())
    return out.fillna(0.0), carried


# ---------------------------------------------------------------------------------------------------
# return construction
# ---------------------------------------------------------------------------------------------------
def levels_from_returns(r: pd.Series) -> pd.Series:
    """Index level base 1.0 on the first date (whose return is ignored)."""
    r = r.copy()
    r.iloc[0] = 0.0
    if r.isna().any():
        raise ValueError(f"{r.name}: NaN return inside the series")
    return (1.0 + r).cumprod()


def day_counts(idx: pd.DatetimeIndex) -> pd.Series:
    d = pd.Series(idx, index=idx).diff().dt.days
    return d


def fx_trust_returns(upu: pd.Series, rate: pd.Series, basis: int, fee: float,
                     carry: bool = True) -> pd.Series:
    """Daily total return of holding one foreign-currency unit on deposit, in USD.

    ``upu`` is units per USD, so the USD value of one unit is S = 1/upu. The accrual over (t-1, t] uses
    the rate at t-1. With ``carry=False`` the result is the spot return only (no interest, no fee).
    """
    s = 1.0 / upu
    spot = s / s.shift(1)
    if not carry:
        return spot - 1.0
    d = day_counts(upu.index)
    acc = rate.shift(1) * d / basis
    return spot * (1.0 + acc) - 1.0 - fee * d / 365.0


def dxy_log_returns(upu: pd.DataFrame) -> pd.Series:
    """Daily log return of the ICE DXY: legacy ten-currency formula through 1999-01-04, euro after."""
    lr = np.log(upu / upu.shift(1))
    legacy = sum(w * lr[c] for c, w in DXY_WEIGHTS_LEGACY.items())
    euro = sum(w * lr[c] for c, w in DXY_WEIGHTS_EURO.items())
    return legacy.where(upu.index <= EURO_START, euro).rename("DXY")


def dxy_level(upu: pd.DataFrame) -> pd.Series:
    """ICE DXY level from the euro-era formula (valid from 1999-01-04)."""
    x = upu.loc[upu.index >= EURO_START]
    lvl = DXY_CONST * np.exp(sum(w * np.log(x[c]) for c, w in DXY_WEIGHTS_EURO.items()))
    return lvl.rename("DXY_level")


def basket_rate_accrual(rates: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """DXY-weighted foreign interest accrual over (t-1, t] (rates at t-1, per-currency day basis)."""
    d = day_counts(idx)
    legacy = sum(w * rates[c].shift(1) * d / DAY_BASIS[c] for c, w in DXY_WEIGHTS_LEGACY.items())
    euro = sum(w * rates[c].shift(1) * d / DAY_BASIS[c] for c, w in DXY_WEIGHTS_EURO.items())
    return legacy.where(idx <= EURO_START, euro)


def uup_returns(upu: pd.DataFrame, rates: pd.DataFrame, us_daily: pd.Series,
                fee: float) -> dict[str, pd.Series]:
    """UUP candidates: spot (DXY), excess (futures ER), funded (ER + T-bill collateral) - fee."""
    idx = upu.index
    d = day_counts(idx)
    spot = np.exp(dxy_log_returns(upu)) - 1.0
    fb = basket_rate_accrual(rates, idx)
    fee_dt = fee * d / 365.0
    excess = spot + us_daily - fb - fee_dt
    funded = spot + 2.0 * us_daily - fb - fee_dt
    return {"spot": spot, "excess": excess, "funded": funded}


# ---------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------
def apply_fixes(upu: pd.DataFrame, fixes: list[dict[str, str]]) -> pd.DataFrame:
    """Apply logged data-error fixes: action 'nan' drops a bad quote (then holiday-filled)."""
    out = upu.copy()
    for f in fixes:
        if f["action"] != "nan":
            raise ValueError(f"unknown fix action {f['action']!r}")
        d = pd.Timestamp(f["date"])
        if d not in out.index or pd.isna(out.at[d, f["series"]]):
            raise ValueError(f"fix {f} targets a date with no quote -- stale fix list?")
        out.at[d, f["series"]] = np.nan
    return out


def load_sources(raw_dir: Path = RAW_DIR, *, fetch_missing: bool = True) -> dict[str, Any]:
    if fetch_missing:
        fetch_all(raw_dir=raw_dir)
    h10, attrs = parse_h10_xml((raw_dir / "FRB_h10_xml.zip").read_bytes())
    legacy: dict[str, pd.Series] = {}
    for ccy, code in LEGACY_CODES.items():
        parts = [parse_h10_legacy_html((raw_dir / f"h10_dat{v}_{code}.htm").read_bytes(), ccy)
                 for v in LEGACY_VINTAGES]
        s = pd.concat(parts).sort_index()
        if s.index.has_duplicates:
            raise ValueError(f"{ccy}: legacy vintages overlap")
        legacy[ccy] = s
    oecd = parse_oecd_csv((raw_dir / "oecd_finmark_ir.csv").read_bytes())
    bis = parse_bis_cbpol((raw_dir / "WS_CBPOL_csv_flat.zip").read_bytes())
    ff = pd.read_parquet(paths.FF_DAILY)
    return {"h10": h10, "h10_attrs": attrs, "legacy": legacy, "oecd": oecd, "bis": bis, "ff": ff}


def spot_panel(src: dict[str, Any], fixes: list[dict[str, str]] = DATA_FIXES) -> pd.DataFrame:
    """Units per USD for every currency on the business-day grid (holidays carried, fixes applied)."""
    raw = to_units_per_usd(src["h10"])
    legacy = {k: v.copy() for k, v in src["legacy"].items()}
    raw_all = pd.concat([raw, pd.DataFrame(legacy)], axis=1)
    raw_all = apply_fixes(raw_all, fixes)
    check_units_per_usd(raw_all)
    start = raw_all.dropna(how="all").index.min()
    idx = business_days(start, raw_all.index.max())
    filled = pd.DataFrame({c: fill_holidays(raw_all[c].rename(c), idx) for c in raw_all.columns})
    spliced = euro_splice(filled[list(H10_FX_SERIES)],
                          {c: raw_all[c].dropna() for c in EURO_CONVERSION})
    for c in EURO_CONVERSION:
        spliced[c] = fill_holidays(spliced[c].rename(c), idx)
    spliced["EUR"] = fill_holidays(spliced["EUR"].rename("EUR"), idx)
    check_euro_join(spliced)
    return spliced


def construct_levels(src: dict[str, Any], fixes: list[dict[str, str]] = DATA_FIXES
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All candidate levels (full history, sealed part included) plus build metadata (no statistics)."""
    upu = spot_panel(src, fixes)
    idx = upu.index
    rates, tags = foreign_rates(src["oecd"], src["bis"], idx)
    us, us_carried = us_rate_daily(src["ff"], idx)
    cols: dict[str, pd.Series] = {}
    for tk, ccy in FX_TRUSTS.items():
        cols[f"{tk}__carry"] = fx_trust_returns(upu[ccy], rates[ccy], DAY_BASIS[ccy], FEES[tk])
        cols[f"{tk}__spot"] = fx_trust_returns(upu[ccy], rates[ccy], DAY_BASIS[ccy], 0.0,
                                               carry=False)
    for name, r in uup_returns(upu, rates, us, FEES["UUP"]).items():
        cols[f"UUP__{name}"] = r
    rets = pd.DataFrame(cols)
    # a candidate starts once every input it needs exists (first date's return is ignored)
    levels = {}
    for c in rets.columns:
        r = rets[c]
        first = r.first_valid_index()
        if first is None:
            continue
        start = r.index[r.index.get_loc(first) - 1]
        levels[c] = levels_from_returns(r.loc[start:].rename(c))
    lv = pd.DataFrame(levels).reindex(idx)
    for tk in TICKERS:
        lv[tk] = lv[f"{tk}__{RECOMMENDED[tk]}"]
    lv = lv[[*TICKERS, *sorted(c for c in lv.columns if "__" in c)]]
    meta = {"rate_segments": rate_segments(tags),
            "days_without_foreign_rate": {c: int((tags[c] == "").sum()) for c in tags.columns},
            "us_rate": {"source": "Ken French daily RF (1-month T-bill), paths.FF_DAILY",
                        "carried_forward_after": us_carried},
            "first_level_date": {c: str(lv[c].first_valid_index().date()) for c in lv.columns}}
    return lv, meta


def _levels_full(raw_dir: Path = RAW_DIR, *, fetch_missing: bool = True) -> pd.DataFrame:
    return construct_levels(load_sources(raw_dir, fetch_missing=fetch_missing))[0]


# ---------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------
def rebase(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its first valid value, so a view never carries growth from before it."""
    return df.apply(lambda s: s / s.dropna().iloc[0] if s.notna().any() else s)


def load_fx(window: str, *, raw_dir: Path = RAW_DIR, gates_path: Path = seal.GATES_PATH,
            levels: pd.DataFrame | None = None) -> pd.DataFrame:
    """FX proxy levels. window='dev' (after the seal, re-based to 1.0) or 'backward' (sealed)."""
    if window not in ("dev", "backward"):
        raise ValueError(f"window must be 'dev' or 'backward', got {window!r}")
    if window == "backward":
        if levels is None:
            # check the seal BEFORE building anything from sealed data
            seal.backward_view(pd.DataFrame(index=pd.DatetimeIndex([seal.SEAL_END])),
                               gates_path=gates_path)
        lv = levels if levels is not None else _levels_full(raw_dir)
        return seal.backward_view(lv, gates_path=gates_path)
    lv = levels if levels is not None else _levels_full(raw_dir)
    return rebase(seal.dev_view(lv))


def hygiene_fx(raw_dir: Path = RAW_DIR, *, big_move: float = 0.10,
               src: dict[str, Any] | None = None,
               fixes: list[dict[str, str]] = DATA_FIXES) -> pd.DataFrame:
    """seal.hygiene COUNTS for the proxy levels and the raw (unfilled) spot quotes, sealed part included."""
    src = src if src is not None else load_sources(raw_dir)
    lv, _ = construct_levels(src, fixes)
    raw = pd.concat([to_units_per_usd(src["h10"]), pd.DataFrame(src["legacy"])], axis=1)
    raw.columns = [f"spot_{c}" for c in raw.columns]
    return pd.concat([seal.hygiene(lv, big_move=big_move), seal.hygiene(raw, big_move=big_move)])


def find_spike_reversals(s: pd.Series, thresh: float = 0.03, revert: float = 0.5) -> list[str]:
    """Dates where a quote jumps by more than ``thresh`` (log) and the next quote reverses at least
    ``revert`` of it -- the signature of a one-day keying error. Returns DATES only."""
    v = np.log(s.dropna())
    r = v.diff()
    nxt = r.shift(-1)
    hit = (r.abs() > thresh) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > revert * r.abs())
    return [str(d.date()) for d in r.index[hit.fillna(False)]]


def find_isolated_spikes(ref: pd.DataFrame, thresh: float = 0.03, revert: float = 0.7,
                         co_move: float = 1 / 3) -> list[dict[str, Any]]:
    """Suspected keying errors: a one-day log jump > ``thresh`` in one column, reversed by at least
    ``revert`` at the next valid quote, while the largest same-direction move of every OTHER column that
    day is below ``co_move`` times it. ``ref`` is units per USD (a reference-only column such as NZD
    may be included). Returns currency, date and the co-move ratio -- no return values."""
    lr = pd.DataFrame({c: np.log(ref[c].dropna()).diff() for c in ref.columns})
    out: list[dict[str, Any]] = []
    for c in ref.columns:
        r = lr[c].dropna()
        nxt = r.shift(-1)
        hit = (r.abs() > thresh) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= revert * r.abs())
        for d in r.index[hit.fillna(False)]:
            others = lr.loc[d].drop(c).dropna()
            same = others[np.sign(others) == np.sign(r[d])].abs()
            ratio = float(same.max() / abs(r[d])) if len(same) else 0.0
            if ratio < co_move:
                out.append({"series": c, "date": str(d.date()), "co_move_ratio": round(ratio, 2)})
    return out


def find_stale_runs(s: pd.Series, min_len: int = 10) -> list[str]:
    """Start dates of runs of >= ``min_len`` identical consecutive quotes (DATES only)."""
    v = s.dropna()
    same = v.diff().eq(0)
    grp = (~same).cumsum()
    runs = same.groupby(grp).sum()
    starts = [v.index[grp == g][0] for g, n in runs.items() if n + 1 >= min_len]
    return [str(d.date()) for d in starts]


def etf_closes(tickers: tuple[str, ...] = TICKERS) -> pd.DataFrame:
    p = pd.read_parquet(paths.ETF_PANEL, columns=["date", "ticker", "close"])
    p = p[p["ticker"].isin(tickers)]
    w = p.pivot(index="date", columns="ticker", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w


def fidelity_pair(proxy: pd.Series, etf: pd.Series) -> dict[str, Any]:
    """Proxy-vs-ETF fidelity on their common DEV dates (the sealed part is cut before anything)."""
    df = seal.dev_view(pd.concat([proxy.rename("p"), etf.rename("e")], axis=1)).dropna()
    seal.assert_no_sealed_rows(df)
    if len(df) < 30:
        return {"n_days": int(len(df))}
    r = df.pct_change().dropna()
    wk = df.resample("W-FRI").last().dropna().pct_change().dropna()
    te = (r["p"] - r["e"]).std() * np.sqrt(TRADING_DAYS)
    return {
        "first": str(df.index[0].date()), "last": str(df.index[-1].date()), "n_days": int(len(r)),
        "daily_corr": float(r["p"].corr(r["e"])),
        "daily_corr_proxy_lead1": float(r["p"].corr(r["e"].shift(-1))),
        "daily_corr_proxy_lag1": float(r["p"].corr(r["e"].shift(1))),
        "weekly_corr": float(wk["p"].corr(wk["e"])),
        "vol_ratio": float(r["p"].std() / r["e"].std()),
        "te_ann": float(te),
        "growth_ratio": float((df["p"].iloc[-1] / df["p"].iloc[0])
                              / (df["e"].iloc[-1] / df["e"].iloc[0])),
    }


def fidelity_fx(levels: pd.DataFrame | None = None, etf: pd.DataFrame | None = None,
                raw_dir: Path = RAW_DIR, src: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dev-window fidelity of every candidate vs its ETF, plus DXY vs Fed dollar indexes."""
    if levels is None:
        src = src if src is not None else load_sources(raw_dir)
        levels = construct_levels(src)[0]
    lv = rebase(seal.dev_view(levels))
    etf = seal.dev_view(etf if etf is not None else etf_closes())
    out: dict[str, Any] = {"window": f"> {seal.SEAL_END.date()}", "proxies": {}}
    for c in lv.columns:
        tk = c.split("__")[0]
        if tk in etf.columns:
            out["proxies"][c] = fidelity_pair(lv[c], etf[tk])
    if src is not None:
        h10 = seal.dev_view(src["h10"])
        upu = seal.dev_view(spot_panel(src))
        dxy = dxy_level(upu)
        out["dxy_vs_fed"] = {}
        for name, series in H10_INDEX_SERIES.items():
            if series in h10.columns:
                out["dxy_vs_fed"][name] = fidelity_pair(dxy, h10[series].dropna())
        out["dxy_level_check"] = {
            "min_2008": float(dxy.loc["2008"].min()), "min_2008_date": str(dxy.loc["2008"].idxmin().date()),
            "max_2022": float(dxy.loc["2022"].max()), "max_2022_date": str(dxy.loc["2022"].idxmax().date()),
            "note": "published DXY: all-time low ~71 (Mar 2008), 20-yr high ~114 (Sep 2022)"}
    return out


def fidelity_gate(fid: dict[str, Any], columns: list[str], min_weekly_corr: float = 0.8) -> None:
    """Fail loudly if a proxy does not track its ETF (catches sign flips, inversions, misalignment)."""
    bad = {c: fid["proxies"][c].get("weekly_corr") for c in columns
           if fid["proxies"].get(c, {}).get("weekly_corr", -1.0) < min_weekly_corr}
    if bad:
        raise ValueError(f"fidelity gate failed (weekly corr < {min_weekly_corr}): {bad}")


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------
def _investigate(src: dict[str, Any]) -> dict[str, Any]:
    """Dates (only) of suspected data errors in the raw quotes, sealed part included."""
    raw = pd.concat([to_units_per_usd(src["h10"]), pd.DataFrame(src["legacy"])], axis=1)
    ref = raw.assign(NZD=1.0 / src["h10"]["RXI$US_N.B.NZ"])
    fixed = {(f["series"], f["date"]) for f in DATA_FIXES}
    iso3 = [x for x in find_isolated_spikes(ref, 0.03) if x["series"] != "NZD"]
    iso2 = [x for x in find_isolated_spikes(ref, 0.02) if x["series"] != "NZD"]
    return {"isolated_spikes_3pct": iso3,
            "isolated_spikes_3pct_not_fixed": [x for x in iso3
                                               if (x["series"], x["date"]) not in fixed],
            "isolated_spikes_2pct_not_fixed": [x for x in iso2
                                               if (x["series"], x["date"]) not in fixed],
            "per_series": {c: {"spike_reversals_3pct": find_spike_reversals(raw[c], 0.03),
                               "stale_runs_10": find_stale_runs(raw[c], 10)} for c in raw.columns}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetch_all(force=a.force, raw_dir=a.raw_dir)
    src = load_sources(a.raw_dir, fetch_missing=False)
    lv, meta = construct_levels(src)
    lv.to_parquet(a.raw_dir / LEVELS_FILE)
    hyg = hygiene_fx(a.raw_dir, src=src)
    hyg5 = hygiene_fx(a.raw_dir, big_move=0.05, src=src)
    fid = fidelity_fx(lv, raw_dir=a.raw_dir, src=src)
    fidelity_gate(fid, list(TICKERS))
    ff_sha = _sha256(Path(paths.FF_DAILY).read_bytes())
    build = {"built_at": _utcnow(), "levels_file": LEVELS_FILE, "recommended": RECOMMENDED,
             "fees": FEES, "day_basis": DAY_BASIS, "dxy_weights_euro": DXY_WEIGHTS_EURO,
             "dxy_weights_legacy": DXY_WEIGHTS_LEGACY, "euro_conversion": EURO_CONVERSION,
             "data_fixes": DATA_FIXES, "known_market_closures_carried": KNOWN_MARKET_CLOSURES,
             "ff_daily_sha256": ff_sha, **meta,
             "investigation": _investigate(src)}
    (a.raw_dir / BUILD_MANIFEST).write_text(json.dumps(build, indent=2), encoding="utf-8")
    report = {"hygiene_big10": json.loads(hyg.to_json(orient="index", date_format="iso")),
              "hygiene_big5": json.loads(hyg5.to_json(orient="index", date_format="iso")),
              "fidelity_dev": fid}
    (a.raw_dir / REPORT_FILE).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for c, f in fid["proxies"].items():
        logger.info("dev fidelity %-14s %s", c, {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in f.items()})
    logger.info("wrote %s, %s, %s", LEVELS_FILE, REPORT_FILE, BUILD_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
