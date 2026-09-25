"""Rates-area long-history proxies: TLT, IEF, LQD and CASH from Federal Reserve H.15 yields.

Sources (Federal Reserve Data Download Program, release H.15):
  * the whole release as SDMX XML (zip) -- Treasury constant-maturity (CMT) yields, the 3-month T-bill
    secondary-market rate (discount basis) and Moody's seasoned Aaa / Baa corporate yields (daily);
  * the preformatted "Treasury Constant Maturities" CSV package -- a second, independent encoding of the
    CMT yields, used only as a parsing cross-check (the two must agree to the published 2 decimals);
  * the project's cached Ken French daily factors (``paths.FF_DAILY``) -- RF, a CASH candidate.

Construction (Swinkels 2019, "Data: International government bond returns since 1947"): hold a par bond of
constant maturity ``M`` at yesterday's yield and re-price it at today's yield,

    r_t = y_{t-1} * dt  -  D(y_{t-1}, M) * dy  +  0.5 * C(y_{t-1}, M) * dy**2,

with ``D`` / ``C`` the modified duration / convexity of a par bond with semi-annual coupons and
``dt`` = calendar days since the previous observation / 365 (weekend and holiday carry accrues on the next
observation). Yields enter as DECIMALS; every public path converts H.15's percent exactly once.

Splicing without jumps: a proxy is an ordered list of *legs* (a yield series each). A leg's return is
computed only between two of ITS OWN observations, so ``dy`` is never taken across two different series.
On each date the highest-priority leg whose previous observation is the proxy's previous date is used.

Declared candidate set: yield family x maturity M (``CANDIDATES``); ``...roll`` families add the roll-down
term D * slope * dt with the slope read off the curve at t-1. Selection (``select_candidate``, dev window
only): lowest TE_ann + |ln(fee-adjusted growth ratio)| / years, i.e. annualised noise error plus annualised
drift error. (A TE-only rule was tried first; it ignores drift and picked a constant-maturity IEF that lags
IEF by ~0.5%/yr from omitted roll-down, so the drift term was added after seeing dev results.) Columns are
``TICKER__candidate``; the recommended one is also exposed as ``TICKER``. Levels are GROSS of ETF fees.

THE SEAL: every date <= 2005-12-31 is sealed. This module computes return *levels* over the full history
(plumbing) but never a statistic of them there; ``load_rates`` returns data only through
``seal.dev_view`` / ``seal.backward_view``; ``fidelity_rates`` reads the dev window only; ``hygiene_rates``
returns counts only.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths, seal

log = logging.getLogger(__name__)

AREA = "rates"
RAW_DIR = paths.DATA_ROOT / AREA
USER_AGENT = "Mozilla/5.0 (FinRL-Pro_DS research data fetch)"

H15_ZIP_URL = "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&filetype=zip"
# Preformatted package, all observations (1962-01-02 onward). A non-empty ``from=`` returns 0 bytes for
# ``type=package`` (verified 2026-09-25), so the date range is left open.
H15_CMT_CSV_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=bf17364827e38702b42a58cf8eaa3f78"
    "&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package"
)
H15_ZIP_FILE = "h15_all_sdmx.zip"
H15_CMT_CSV_FILE = "h15_cmt_package.csv"

# H.15 business-day series (percent per year, as published).
SERIES: dict[str, str] = {
    "y5": "RIFLGFCY05_N.B",      # 5y CMT, investment (bond-equivalent) basis
    "y7": "RIFLGFCY07_N.B",      # 7y CMT
    "y10": "RIFLGFCY10_N.B",     # 10y CMT
    "y20": "RIFLGFCY20_N.B",     # 20y CMT (no data 1987-01 .. 1993-09)
    "y30": "RIFLGFCY30_N.B",     # 30y CMT (from 1977-02-15; no data 2002-02 .. 2006-02)
    "cmt3m": "RIFLGFCM03_N.B",   # 3m CMT, investment basis (from 1981) -- units cross-check only
    "tb3m": "RIFSGFSM03_N.B",    # 3m T-bill secondary market, DISCOUNT basis (from 1954)
    "aaa": "RIMLPAAAR_N.B",      # Moody's seasoned Aaa, all industries (daily 1983-01-03 .. 2016-10-07)
    "baa": "RIMLPBAAR_N.B",      # Moody's seasoned Baa, all industries (daily 1986-01-02 .. 2016-10-07)
}
CMT_KEYS = ("y5", "y7", "y10", "y20", "y30", "cmt3m")

TBILL_DAYS = 91          # 3-month bill tenor used in the discount -> bond-equivalent conversion
MAX_LEG_GAP_DAYS = 10    # a leg return spanning more calendar days than this is not bridged
MAX_DECIMAL_YIELD = 0.50  # a "decimal" yield above 50% means percent units leaked in

# Data errors found by investigation and corrected before construction. Each entry is logged in the
# manifest. Format: (series key, ISO date, raw published value in percent, action, reason).
DATA_FIXES: tuple[tuple[str, str, float, str, str], ...] = ()
# What the 2026-09-25 hygiene investigation found (recorded in the manifest; nothing met the bar for a fix).
INVESTIGATION = (
    "One-day reversal spikes (|dy|>=25bp both ways, net <=5bp): 37 flags, 1975-2020. Every flag on a series "
    "used by a proxy co-moves with neighbouring maturities on the same day (1980-82 Volcker volatility, "
    "2008-09-29, 2009-05-29 Aaa with Baa and 10y, 2020-03-09) => genuine, kept. The 1981-09..1982-03 3m CMT "
    "flags diverge from the 3m bill (e.g. 1982-01-14 CMT +47bp vs bill -8bp) and look like errors, but the 3m "
    "CMT is used only as a dev-window units check, so they are left as published. Stale runs: 1962-1966 long "
    "CMTs repeat for up to 21 days (y20, 1965-02) -- pre-1973, outside the test window; 1973-2005 runs are "
    "<= 8 days. Bill non-positive values (21, min -0.05%) are all 2008-2020 and real. No fixes applied."
)

# Recommended candidate per ticker = ``select_candidate`` on the dev window (re-checked, with a warning on
# drift, by ``fidelity_rates()['recommendation_check']``). CASH is chosen by argument, not by the score.
RECOMMENDED: dict[str, str] = {
    "TLT": "TLT__blendroll_m25",
    "IEF": "IEF__avg710roll_m8.5",
    "LQD": "LQD__tsyspread_m10",
    "CASH": "CASH__tbill3m",
}


# ----------------------------------------------------------------------------------------------------
# Fetch + manifests
# ----------------------------------------------------------------------------------------------------
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _download(url: str, dest: Path, *, force: bool = False, timeout: int = 600) -> tuple[bytes, bool]:
    """Return (bytes, fetched_now). Uses the cached file unless ``force``."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("cache hit %s", dest)
        return dest.read_bytes(), False
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        body = resp.read()
    if not body:
        raise RuntimeError(f"empty response from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return body, True


def _manifest_path(name: str) -> Path:
    return RAW_DIR / f"manifest_{name}.json"


def _write_manifest(name: str, meta: dict) -> Path:
    p = _manifest_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return p


def _read_manifest(name: str) -> dict:
    p = _manifest_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _coverage(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for c in df.columns:
        v = df[c].dropna()
        out[c] = {"n_obs": int(len(v)),
                  "first": str(v.index.min().date()) if len(v) else None,
                  "last": str(v.index.max().date()) if len(v) else None}
    return out


def fetch_rates(*, force: bool = False) -> dict[str, Path]:
    """Download the H.15 sources into ``DATA_ROOT/rates`` and write one manifest per source."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    zdest = RAW_DIR / H15_ZIP_FILE
    body, fresh = _download(H15_ZIP_URL, zdest, force=force)
    ylds = parse_h15_sdmx(_sdmx_text(body), SERIES)
    old = _read_manifest("h15_sdmx")
    _write_manifest("h15_sdmx", {
        "source": "Federal Reserve Board DDP, H.15 Selected Interest Rates, whole release (SDMX XML in zip)",
        "url": H15_ZIP_URL,
        "file": str(zdest),
        "fetched_at": _now_utc() if fresh or not old.get("fetched_at") else old["fetched_at"],
        "sha256": _sha256(body),
        "bytes": len(body),
        "parsed_rows": int(ylds.notna().any(axis=1).sum()),
        "first_date": str(ylds.dropna(how="all").index.min().date()),
        "last_date": str(ylds.dropna(how="all").index.max().date()),
        "series": SERIES,
        "coverage": _coverage(ylds),
        "units": "percent per year as published; CMT on investment (bond-equivalent) basis; tb3m on "
                 "DISCOUNT basis; OBS_STATUS != 'A' (ND / NA, encoded -9999) parsed as missing",
        "fixes": [dict(zip(("series", "date", "raw_value_pct", "action", "reason"), f)) for f in DATA_FIXES],
        "investigation": INVESTIGATION,
        "notes": "Moody's daily Aaa starts 1983-01-03, Baa 1986-01-02, both end 2016-10-07 (H.15 "
                 "discontinued them); only monthly averages exist before 1983 (not used: a monthly average "
                 "is smoothed and would plant autocorrelation). 20y CMT has no data 1987-01..1993-09; 30y "
                 "CMT starts 1977-02-15 and has no data 2002-02-19..2006-02-08.",
    })
    out["h15_sdmx"] = zdest

    cdest = RAW_DIR / H15_CMT_CSV_FILE
    body, fresh = _download(H15_CMT_CSV_URL, cdest, force=force)
    cmt = parse_ddp_csv(body.decode("utf-8-sig"))
    old = _read_manifest("h15_cmt_csv")
    _write_manifest("h15_cmt_csv", {
        "source": "Federal Reserve Board DDP, H.15 preformatted package 'Treasury Constant Maturities' (CSV)",
        "url": H15_CMT_CSV_URL,
        "file": str(cdest),
        "fetched_at": _now_utc() if fresh or not old.get("fetched_at") else old["fetched_at"],
        "sha256": _sha256(body),
        "bytes": len(body),
        "parsed_rows": int(len(cmt)),
        "first_date": str(cmt.index.min().date()),
        "last_date": str(cmt.index.max().date()),
        "coverage": _coverage(cmt),
        "units": "percent per year; 'ND' and blanks parsed as missing",
        "notes": "used only to cross-check the SDMX parse of the CMT series",
    })
    out["h15_cmt_csv"] = cdest

    ffb = paths.FF_DAILY.read_bytes()
    rf = _load_ff_rf()
    _write_manifest("ff_rf", {
        "source": "Ken French data library, 5-factor 2x3 daily (cached by scripts/research/"
                  "etf_outperformance_factors.py)",
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip (via cache)",
        "file": str(paths.FF_DAILY),
        "fetched_at": _now_utc(),
        "sha256": _sha256(ffb),
        "parsed_rows": int(rf.notna().sum()),
        "first_date": str(rf.index.min().date()),
        "last_date": str(rf.index.max().date()),
        "units": "decimal per trading day (cache divided the published percent by 100)",
        "notes": "RF is the 1-month T-bill return spread over trading days. The published file prints RF "
                 "in percent with 2 decimals, so the daily rate is quantised to 0.0001 (1 bp/day ~ 2.5%/yr "
                 "steps) -- verified on the dev window, where it takes only the values 0, 1e-4, 2e-4.",
    })
    out["ff_rf"] = paths.FF_DAILY
    return out


# ----------------------------------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------------------------------
def _sdmx_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith("_data.xml"))
        return z.read(name).decode("utf-8")


_SERIES_RE = re.compile(r"<kf:Series\b([^>]*)>(.*?)</kf:Series>", re.S)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_OBS_RE = re.compile(r"<frb:Obs\b([^>]*)/>")


def parse_h15_sdmx(xml_text: str, series: dict[str, str]) -> pd.DataFrame:
    """Parse the H.15 SDMX-compact XML into a percent-valued frame (columns = keys of ``series``).

    Only observations with ``OBS_STATUS="A"`` are kept; ND / NA carry the sentinel -9999 and become NaN.
    """
    want = {v: k for k, v in series.items()}
    cols: dict[str, pd.Series] = {}
    for m in _SERIES_RE.finditer(xml_text):
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        key = want.get(attrs.get("SERIES_NAME", ""))
        if key is None:
            continue
        dates, vals = [], []
        for o in _OBS_RE.finditer(m.group(2)):
            a = dict(_ATTR_RE.findall(o.group(1)))
            dates.append(a["TIME_PERIOD"])
            vals.append(float(a["OBS_VALUE"]) if a.get("OBS_STATUS") == "A" else np.nan)
        s = pd.Series(vals, index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"), name=key)
        if s.index.has_duplicates:
            raise ValueError(f"duplicate dates in {attrs['SERIES_NAME']}")
        cols[key] = s.sort_index()
    missing = sorted(set(series) - set(cols))
    if missing:
        raise KeyError(f"series not found in SDMX: {missing}")
    return pd.DataFrame(cols)[list(series)]


def parse_ddp_csv(text: str) -> pd.DataFrame:
    """Parse a DDP 'seriescolumn' CSV (5 label rows, then 'Time Period' header) into percent values.

    Columns are renamed to the keys of ``SERIES`` when the identifier is known; 'ND' / blanks -> NaN.
    """
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith('"Time Period"'))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), na_values=["ND", "NA", ""], keep_default_na=False)
    df = df.rename(columns={"Time Period": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index("date").sort_index()
    inv = {v: k for k, v in SERIES.items()}
    df = df.rename(columns={c: inv.get(c, c) for c in df.columns})
    return df.apply(pd.to_numeric, errors="raise").astype(float)


def pct_to_decimal(y_pct: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """H.15 publishes percent per year; construction uses decimals. The ONLY units conversion."""
    return y_pct / 100.0


def _require_decimal(y: pd.Series | pd.DataFrame, what: str) -> None:
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v)]
    if v.size and (np.abs(v).max() > MAX_DECIMAL_YIELD):
        raise ValueError(f"{what}: |yield| {np.abs(v).max():.4g} > {MAX_DECIMAL_YIELD} -- percent units or a "
                         "sentinel (-9999) leaked into a decimal-yield path")


def _apply_fixes(y_pct: pd.DataFrame, fixes: Sequence[tuple[str, str, float, str, str]]) -> pd.DataFrame:
    y = y_pct.copy()
    for key, day, raw, action, reason in fixes:
        ts = pd.Timestamp(day)
        got = y.at[ts, key]
        if not np.isclose(got, raw):
            raise ValueError(f"fix {key}@{day}: expected raw {raw}, found {got} (source changed?)")
        if action != "set_missing":
            raise ValueError(f"unknown fix action {action}")
        y.at[ts, key] = np.nan
        log.warning("data fix: %s %s raw=%s -> missing (%s)", key, day, raw, reason)
    return y


def _load_ff_rf() -> pd.Series:
    ff = pd.read_parquet(paths.FF_DAILY)
    rf = ff["RF"].astype(float)
    rf.index = pd.DatetimeIndex(rf.index, name="date")
    _require_decimal(rf * 252, "FF RF (annualised)")  # daily decimal * 252 must look like a decimal yield
    return rf.sort_index()


@lru_cache(maxsize=1)
def load_raw_yields() -> pd.DataFrame:
    """Cached H.15 yields in PERCENT (after DATA_FIXES), cross-checked against the CSV package."""
    zpath, cpath = RAW_DIR / H15_ZIP_FILE, RAW_DIR / H15_CMT_CSV_FILE
    if not (zpath.exists() and cpath.exists()):
        fetch_rates()
    y = parse_h15_sdmx(_sdmx_text(zpath.read_bytes()), SERIES)
    cmt = parse_ddp_csv(cpath.read_text(encoding="utf-8-sig"))
    _cross_check_cmt(y, cmt)
    return _apply_fixes(y, DATA_FIXES)


def _cross_check_cmt(sdmx: pd.DataFrame, csv: pd.DataFrame) -> None:
    """Plumbing: the two encodings of the CMT series must agree on every common date (to 1e-9)."""
    for k in CMT_KEYS:
        a, b = sdmx[k].dropna(), csv[k].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 0.99 * min(len(a), len(b)):
            raise ValueError(f"{k}: SDMX/CSV date sets disagree ({len(a)} vs {len(b)}, {len(common)} common)")
        bad = (a.loc[common] - b.loc[common]).abs() > 1e-9
        if bad.any():
            raise ValueError(f"{k}: SDMX/CSV values disagree on {int(bad.sum())} dates, first {bad.idxmax()}")
    log.info("SDMX vs CSV CMT cross-check OK for %s", ",".join(CMT_KEYS))


# ----------------------------------------------------------------------------------------------------
# Bond math
# ----------------------------------------------------------------------------------------------------
def par_duration(y: np.ndarray | float, m: float) -> np.ndarray:
    """Modified duration of a par bond (coupon = yield y, decimal), maturity m years, semi-annual coupons."""
    y = np.asarray(y, dtype=float)
    return (1.0 - (1.0 + y / 2.0) ** (-2.0 * m)) / y


def par_convexity(y: np.ndarray | float, m: float) -> np.ndarray:
    """Convexity of a par bond, semi-annual coupons (Swinkels 2019, eq. 3)."""
    y = np.asarray(y, dtype=float)
    z = 1.0 + y / 2.0
    return (2.0 / y**2) * (1.0 - z ** (-2.0 * m)) - (2.0 * m) / (y * z ** (2.0 * m + 1.0))


def bond_return(y_prev: np.ndarray | float, y: np.ndarray | float, dt: np.ndarray | float, m: float) -> np.ndarray:
    """One-period total return of a constant-maturity par bond: carry - D*dy + 0.5*C*dy^2 (decimals)."""
    y_prev = np.asarray(y_prev, dtype=float)
    dy = np.asarray(y, dtype=float) - y_prev
    return y_prev * np.asarray(dt, dtype=float) - par_duration(y_prev, m) * dy + 0.5 * par_convexity(y_prev, m) * dy**2


def coupon_bond_price(coupon: float, y: float, m: float) -> float:
    """Exact price (per 1 face) of a bond with ``2m`` whole semi-annual periods left. Reference for tests."""
    n = int(round(2 * m))
    if abs(n - 2 * m) > 1e-12:
        raise ValueError("coupon_bond_price needs 2m to be an integer")
    z = 1.0 + y / 2.0
    return float(coupon / 2.0 * (1.0 - z ** (-n)) / (y / 2.0) + z ** (-n))


def tbill_discount_to_bey(d: np.ndarray | pd.Series | float, days: int = TBILL_DAYS) -> np.ndarray | pd.Series:
    """Bank-discount bill rate (decimal, 360-day) -> bond-equivalent (investment) yield, 365-day basis."""
    return 365.0 * d / (360.0 - d * days)


# ----------------------------------------------------------------------------------------------------
# Legs and composites
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    name: str
    y: pd.Series                        # decimal yield (NaN where not published)
    slope: pd.Series | None = None      # curve slope dy/dM (decimal per year of maturity) for roll-down
    before: pd.Timestamp | None = None  # leg usable only on dates strictly before this (backfill legs)

    def valid(self) -> pd.DataFrame:
        """Columns y, s (slope, 0 when the leg has none) on the leg's own valid dates."""
        s = self.slope if self.slope is not None else pd.Series(0.0, index=self.y.index)
        d = pd.concat([self.y.rename("y"), s.rename("s")], axis=1).dropna()
        return d.loc[d.index < self.before] if self.before is not None else d


@dataclass
class Composite:
    returns: pd.Series                  # on the composite grid; first date NaN (level anchor)
    source: pd.Series                   # leg name used per date ('' where unbridged)
    n_unbridged: int = 0
    notes: list[str] = field(default_factory=list)


# fn(y_prev, y, dt_years, slope_prev) -> simple return
ReturnFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def leg_returns(leg: Leg, fn: ReturnFn, *, max_gap_days: int = MAX_LEG_GAP_DAYS) -> pd.DataFrame:
    """Return per valid date t of the leg: r (from its own previous valid date) and that previous date."""
    d = leg.valid()
    _require_decimal(d["y"], f"leg {leg.name}")
    prev_date = d.index.to_series().shift(1)
    gap = (d.index.to_series() - prev_date).dt.days
    r = fn(d["y"].shift(1).to_numpy(), d["y"].to_numpy(), (gap / 365.0).to_numpy(), d["s"].shift(1).to_numpy())
    r = pd.Series(r, index=d.index)
    r[gap > max_gap_days] = np.nan
    return pd.DataFrame({"r": r, "prev": prev_date})


def composite(legs: Sequence[Leg], fn: ReturnFn) -> Composite:
    """Splice legs in priority order; a leg's return is used only if its previous observation is the
    composite's previous date, so no return is double-counted and no ``dy`` spans two series."""
    per_leg = {lg.name: leg_returns(lg, fn) for lg in legs}
    grid = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in per_leg.values()])), name="date")
    grid_prev = pd.Series(grid, index=grid).shift(1)
    out = pd.Series(np.nan, index=grid)
    src = pd.Series("", index=grid, dtype=object)
    for name in (lg.name for lg in legs):
        d = per_leg[name].reindex(grid)
        ok = d["r"].notna() & (d["prev"] == grid_prev) & out.isna()
        out[ok] = d["r"][ok]
        src[ok] = name
    unbridged = out.isna()
    unbridged.iloc[0] = False
    n_unb = int(unbridged.sum())
    if n_unb:
        log.warning("composite %s: %d dates with no bridging leg (return set to 0): %s",
                    "/".join(lg.name for lg in legs), n_unb, [str(t.date()) for t in grid[unbridged][:10]])
    out[unbridged] = 0.0
    out.iloc[0] = np.nan
    return Composite(out, src, n_unb)


def levels_from_returns(r: pd.Series, *, bdays: pd.DatetimeIndex | None = None) -> pd.Series:
    """Index level = 1.0 on the first date, compounded after; forward-filled onto business days inside
    coverage (bond-market holidays keep the level flat; the carry accrues on the next observation)."""
    r = r.copy()
    lvl = (1.0 + r.fillna(0.0)).cumprod()
    lvl.iloc[0] = 1.0
    lvl = lvl / lvl.iloc[0]
    if bdays is None:
        bdays = pd.bdate_range(lvl.index.min(), lvl.index.max(), name="date")
    inside = (bdays >= lvl.index.min()) & (bdays <= lvl.index.max())
    out = lvl.reindex(bdays).ffill()
    out[~inside] = np.nan
    return out


def _mean2(a: pd.Series, b: pd.Series) -> pd.Series:
    """Average of two yields, defined only where BOTH exist (never a silent single-series fallback)."""
    return (a + b) / 2.0


def roll_down_return(y_prev: np.ndarray | float, slope_prev: np.ndarray | float, dt: np.ndarray | float,
                     m: float) -> np.ndarray:
    """Roll-down of a bond held for dt on an unchanged curve: its maturity shortens by dt, so its yield
    moves by -slope*dt and its price by +D*slope*dt (slope = dy/dM at t-1; decimal per year)."""
    return par_duration(y_prev, m) * np.asarray(slope_prev, dtype=float) * np.asarray(dt, dtype=float)


def _par_fn(m: float) -> ReturnFn:
    def fn(yp: np.ndarray, y: np.ndarray, dt: np.ndarray, sp: np.ndarray) -> np.ndarray:
        if np.any(yp[np.isfinite(yp)] <= 0):
            raise ValueError("par-bond duration needs a positive yield")
        return bond_return(yp, y, dt, m) + roll_down_return(yp, sp, dt, m)
    return fn


def _cash_fn(yp: np.ndarray, y: np.ndarray, dt: np.ndarray, sp: np.ndarray) -> np.ndarray:
    return yp * dt


# ----------------------------------------------------------------------------------------------------
# Candidate declarations
# ----------------------------------------------------------------------------------------------------
TLT_M = (20.0, 25.0, 30.0)
IEF_M = (7.5, 8.5, 9.5)
LQD_M = (10.0, 12.0)


def _fmt_m(m: float) -> str:
    return f"{m:g}"


def candidate_legs(yd: pd.DataFrame) -> dict[str, list[Leg]]:
    """Declared candidates -> ordered legs. ``yd``: decimal yields, columns = SERIES keys."""
    y7, y10, y20, y30, aaa, baa = (yd[k] for k in ("y7", "y10", "y20", "y30", "aaa", "baa"))
    long_legs = [("b", _mean2(y20, y30)), ("30", y30), ("20", y20)]
    # the prefill leg bridges up to AND INCLUDING the first Moody's date (a Moody's leg has no return there)
    prefill_end = aaa.first_valid_index() + pd.Timedelta(days=1)
    out: dict[str, list[Leg]] = {}
    # TLT: 20-30y bucket. blend = mean(20y,30y) where both exist, else the one that exists.
    out["blend"] = [Leg("y20y30", _mean2(y20, y30)), Leg("y30", y30), Leg("y20", y20)]
    out["pref30"] = [Leg("y30", y30), Leg("y20", y20)]
    # blendroll: + roll-down with the local slope (y30-y20)/10. Single-series fallback legs carry NO roll-down:
    # the chord to the 10y overstates the 20-30y slope (dev window: 0.81 / 0.50 vs 0.20 %/yr of roll), while
    # zero understates it by ~0.2 %/yr -- the smaller error.
    out["blendroll"] = [Leg("y20y30", _mean2(y20, y30), slope=(y30 - y20) / 10.0), Leg("y30", y30), Leg("y20", y20)]
    # IEF: 7-10y bucket.
    out["avg710"] = [Leg("y7y10", _mean2(y7, y10)), Leg("y10", y10)]
    out["y10"] = [Leg("y10", y10)]
    # avg710roll: + roll-down, slope (y10-y7)/3; the pre-1969 10y-only fallback leg carries none.
    out["avg710roll"] = [Leg("y7y10", _mean2(y7, y10), slope=(y10 - y7) / 3.0), Leg("y10", y10)]
    # LQD: IG corporate. avgab = mean(Aaa, Baa) where both exist (Baa daily only from 1986), else Aaa.
    out["avgab"] = [Leg("aaabaa", _mean2(aaa, baa)), Leg("aaa", aaa), Leg("baa", baa)]
    out["baa"] = [Leg("baa", baa)]
    # tsyspread: 10y CMT + (Moody's IG yield - long Treasury): intermediate curve point + IG spread.
    ts: list[Leg] = []
    tsr: list[Leg] = []
    for cname, cy in (("aaabaa", _mean2(aaa, baa)), ("aaa", aaa), ("baa", baa)):
        for lname, ly in long_legs:
            ts.append(Leg(f"y10+{cname}-y{lname}", y10 + cy - ly))
            tsr.append(Leg(f"y10+{cname}-y{lname}", y10 + cy - ly, slope=(y10 - y7) / 3.0))
    out["tsyspread"] = ts
    out["tsyspreadroll"] = tsr
    # tsyspread_tsyfill: tsyspread, with pre-Moody's history back-filled by the 10y CMT, i.e. the credit
    # spread is held CONSTANT before 1983 (no daily corporate yields exist). Not recommended; opt-in only.
    out["tsyspread_tsyfill"] = ts + [Leg("y10_prefill", y10, before=prefill_end)]
    return out


CANDIDATES: dict[str, list[tuple[str, tuple[float, ...]]]] = {
    "TLT": [("blend", TLT_M), ("pref30", TLT_M), ("blendroll", TLT_M)],
    "IEF": [("avg710", IEF_M), ("y10", IEF_M), ("avg710roll", IEF_M)],
    "LQD": [("avgab", LQD_M), ("baa", LQD_M), ("tsyspread", LQD_M), ("tsyspreadroll", LQD_M),
            ("tsyspread_tsyfill", LQD_M)],
}
# Backfill candidates are excluded from selection (identical to their family on the dev window).
NOT_SELECTABLE = ("tsyfill",)
# ETF expense ratios (ASSUMED from issuer fact sheets, not fetched). Levels are GROSS of fees; the ratio is
# used only for the fee-adjusted growth ratio in the selection tie-break.
EXPENSE_RATIO: dict[str, float] = {"TLT": 0.0015, "IEF": 0.0015, "LQD": 0.0014}


def build_rates(y_pct: pd.DataFrame, rf: pd.Series | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Construct all candidate level columns (plus recommended aliases) from PERCENT yields.

    Returns (levels on a business-day index, per-column construction info).
    """
    yd = pct_to_decimal(y_pct)
    for k in ("y7", "y10", "y20", "y30", "aaa", "baa", "tb3m"):
        _require_decimal(yd[k], k)
    legs = candidate_legs(yd)
    bdays = pd.bdate_range(yd.dropna(how="all").index.min(), yd.dropna(how="all").index.max(), name="date")
    cols: dict[str, pd.Series] = {}
    info: dict[str, dict] = {}
    for tkr, specs in CANDIDATES.items():
        for rule, ms in specs:
            for m in ms:
                name = f"{tkr}__{rule}_m{_fmt_m(m)}"
                comp = composite(legs[rule], _par_fn(m))
                cols[name] = levels_from_returns(comp.returns, bdays=bdays)
                used = comp.source[comp.source != ""]
                info[name] = {"maturity": m, "legs": [lg.name for lg in legs[rule]],
                              "n_unbridged": comp.n_unbridged,
                              "leg_days": used.value_counts().to_dict(),
                              "first": str(comp.returns.index.min().date()),
                              "last": str(comp.returns.index.max().date())}
    # CASH: 3m bill, discount -> bond-equivalent, simple accrual y_{t-1} * dt.
    bey = tbill_discount_to_bey(yd["tb3m"]).rename("tb3m_bey")
    comp = composite([Leg("tb3m_bey", bey)], _cash_fn)
    cols["CASH__tbill3m"] = levels_from_returns(comp.returns, bdays=bdays)
    info["CASH__tbill3m"] = {"legs": ["tb3m_bey"], "n_unbridged": comp.n_unbridged}
    if "cmt3m" in yd:
        comp = composite([Leg("cmt3m", yd["cmt3m"])], _cash_fn)
        cols["CASH__cmt3m"] = levels_from_returns(comp.returns, bdays=bdays)
        info["CASH__cmt3m"] = {"legs": ["cmt3m"], "n_unbridged": comp.n_unbridged,
                               "note": "units cross-check only (3m CMT starts 1981)"}
    if rf is not None:
        r = rf.copy()
        r.iloc[0] = np.nan  # the first day's RF is not compounded into the anchor level
        cols["CASH__ff_rf"] = levels_from_returns(r, bdays=bdays)
        info["CASH__ff_rf"] = {"legs": ["ff_rf"], "note": "quantised to 1e-4/day at source"}
    out = pd.DataFrame(cols)
    for tkr, cand in RECOMMENDED.items():
        if cand in out:
            out[tkr] = out[cand]
            info[tkr] = {"alias_of": cand}
    return out, info


@lru_cache(maxsize=1)
def _build_cached() -> tuple[pd.DataFrame, dict[str, dict]]:
    return build_rates(load_raw_yields(), _load_ff_rf())


# ----------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------
def load_rates(window: str) -> pd.DataFrame:
    """Daily total-return index levels (start 1.0). ``window`` is 'dev' (> 2005-12-31, always readable)
    or 'backward' (<= 2005-12-31; raises ``seal.SealedError`` until the pre-registration is committed)."""
    if window not in ("dev", "backward"):
        raise ValueError(f"window must be 'dev' or 'backward', got {window!r}")
    levels = _build_cached()[0]
    if window == "dev":
        return seal.dev_view(levels)
    return seal.backward_view(levels)


def hygiene_rates() -> pd.DataFrame:
    """Coverage / outlier COUNTS for the raw yields and the constructed levels (sealed part included)."""
    y = load_raw_yields()
    levels = _build_cached()[0]
    hy = seal.hygiene(y)
    hy = hy.join(yield_outlier_counts(y))
    hy.index = [f"yield:{c}" for c in hy.index]
    hl = seal.hygiene(levels, big_move=0.03)
    hl.index = [f"level:{c}" for c in hl.index]
    return pd.concat([hy, hl])


def yield_outlier_counts(y_pct: pd.DataFrame, *, big_bp: float = 50.0, spike_bp: float = 25.0,
                         stale_run: int = 5) -> pd.DataFrame:
    """Counts only: |dy| > big_bp; one-day spikes that reverse (|dy_t|,|dy_t+1| >= spike_bp, opposite sign,
    |y_t+1 - y_t-1| <= spike_bp/5); runs of >= stale_run identical consecutive published values."""
    rows = {}
    for c in y_pct.columns:
        v = y_pct[c].dropna()
        d = v.diff() * 100.0
        nxt = d.shift(-1)
        spike = (d.abs() >= spike_bp) & (nxt.abs() >= spike_bp) & (np.sign(d) != np.sign(nxt)) \
            & ((v.shift(-1) - v.shift(1)).abs() * 100.0 <= spike_bp / 5.0)
        same = v.diff() == 0
        run_id = (~same).cumsum()
        runs = same.groupby(run_id).sum()
        rows[c] = {f"n_abs_dy_gt_{big_bp:g}bp": int((d.abs() > big_bp).sum()),
                   "n_spike_reversals": int(spike.sum()),
                   f"n_stale_runs_ge{stale_run}": int((runs + 1 >= stale_run).sum())}
    return pd.DataFrame(rows).T


def suspicious_dates(y_pct: pd.DataFrame, *, spike_bp: float = 25.0) -> dict[str, list[str]]:
    """Dates of one-day reversal spikes, for manual diagnosis of data errors (dates only)."""
    out: dict[str, list[str]] = {}
    for c in y_pct.columns:
        v = y_pct[c].dropna()
        d = v.diff() * 100.0
        nxt = d.shift(-1)
        spike = (d.abs() >= spike_bp) & (nxt.abs() >= spike_bp) & (np.sign(d) != np.sign(nxt)) \
            & ((v.shift(-1) - v.shift(1)).abs() * 100.0 <= spike_bp / 5.0)
        if spike.any():
            out[c] = [str(t.date()) for t in v.index[spike.to_numpy()]]
    return out


def etf_levels(tickers: Sequence[str]) -> pd.DataFrame:
    """Dividend-adjusted ETF closes (the dev-window reference), wide, from ``paths.ETF_PANEL``."""
    p = pd.read_parquet(paths.ETF_PANEL, columns=["date", "ticker", "close"])
    p = p[p["ticker"].isin(list(tickers))]
    w = p.pivot(index="date", columns="ticker", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w


def fidelity_pair(proxy: pd.Series, etf: pd.Series, *, expense_ratio: float = 0.0) -> dict[str, float | int | str]:
    """Proxy-vs-ETF fidelity on the DEV window only (common dates after SEAL_END).

    ``growth_ratio_fee_adj`` charges the gross proxy the ETF's expense ratio before comparing growth.
    """
    p, e = seal.dev_view(proxy.dropna()), seal.dev_view(etf.dropna())
    common = p.index.intersection(e.index)
    p, e = p.loc[common], e.loc[common]
    seal.assert_no_sealed_rows(p)
    rp, re_ = p.pct_change().dropna(), e.pct_change().dropna()
    wp = p.resample("W-FRI").last().dropna().pct_change().dropna()
    we = e.resample("W-FRI").last().dropna().pct_change().dropna()
    wc = wp.index.intersection(we.index)
    diff = rp - re_
    return {
        "start": str(common.min().date()), "end": str(common.max().date()), "n_days": int(len(rp)),
        "years": float((common.max() - common.min()).days / 365.25),
        "daily_corr": float(rp.corr(re_)),
        "weekly_corr": float(wp.loc[wc].corr(we.loc[wc])),
        "vol_ratio": float(rp.std() / re_.std()),
        "tracking_error_ann": float(diff.std() * np.sqrt(252.0)),
        "growth_ratio": float((p.iloc[-1] / p.iloc[0]) / (e.iloc[-1] / e.iloc[0])),
        "growth_ratio_fee_adj": float((p.iloc[-1] / p.iloc[0]) / (e.iloc[-1] / e.iloc[0])
                                      * np.exp(-expense_ratio * (common.max() - common.min()).days / 365.25)),
    }


def selection_score(pair: dict) -> float:
    """Annualised noise error + annualised drift error: TE + |ln(fee-adjusted growth ratio)| / years."""
    return float(pair["tracking_error_ann"] + abs(np.log(pair["growth_ratio_fee_adj"])) / pair["years"])


def select_candidate(pairs: dict[str, dict]) -> str:
    """Selection rule: lowest ``selection_score`` (ties by name)."""
    return min(pairs, key=lambda k: (selection_score(pairs[k]), k))


def fidelity_rates() -> dict:
    """Dev-window fidelity of every candidate vs its ETF, the CASH cross-checks, and a check that
    ``RECOMMENDED`` is still what ``select_candidate`` picks per ticker."""
    lv = load_rates("dev")
    etf = seal.dev_view(etf_levels(["TLT", "IEF", "LQD"]))
    out: dict = {"pairs": {}, "cash": {}, "recommendation_check": {}}
    for c in lv.columns:
        tkr = c.split("__")[0]
        if tkr in etf.columns:
            out["pairs"][c] = fidelity_pair(lv[c], etf[tkr], expense_ratio=EXPENSE_RATIO[tkr])
    for tkr in ("TLT", "IEF", "LQD"):
        cands = {k: v for k, v in out["pairs"].items()
                 if k.startswith(f"{tkr}__") and not any(s in k for s in NOT_SELECTABLE)}
        pick = select_candidate(cands)
        out["recommendation_check"][tkr] = {"recommended": RECOMMENDED[tkr], "selected_by_rule": pick,
                                            "agrees": pick == RECOMMENDED[tkr]}
        if pick != RECOMMENDED[tkr]:
            log.warning("%s: RECOMMENDED %s but the declared rule now selects %s", tkr, RECOMMENDED[tkr], pick)
    # CASH: compare annualised accrual rates on the dev window (no ETF).
    y = seal.dev_view(load_raw_yields())
    bey = tbill_discount_to_bey(y["tb3m"] / 100.0)
    both = pd.concat([bey.rename("bey"), (y["cmt3m"] / 100.0).rename("cmt")], axis=1).dropna()
    out["cash"]["tbill_bey_minus_cmt3m_bp_mean"] = float((both["bey"] - both["cmt"]).mean() * 1e4)
    out["cash"]["tbill_bey_minus_cmt3m_bp_mad"] = float((both["bey"] - both["cmt"]).abs().mean() * 1e4)
    out["cash"]["tbill_raw_discount_minus_cmt3m_bp_mean"] = float(
        (y["tb3m"] / 100.0 - y["cmt3m"] / 100.0).dropna().mean() * 1e4)
    for c in ("CASH__tbill3m", "CASH__cmt3m", "CASH__ff_rf"):
        s = lv[c].dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out["cash"][f"{c}_ann_growth"] = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0)
    a, b = lv["CASH__tbill3m"], lv["CASH__ff_rf"]
    cm = a.dropna().index.intersection(b.dropna().index)
    out["cash"]["growth_ratio_tbill3m_over_ff_rf"] = float(
        (a.loc[cm].iloc[-1] / a.loc[cm].iloc[0]) / (b.loc[cm].iloc[-1] / b.loc[cm].iloc[0]))
    ma = a.loc[cm].resample("ME").last().pct_change().dropna()
    mb = b.loc[cm].resample("ME").last().pct_change().dropna()
    out["cash"]["monthly_corr_tbill3m_vs_ff_rf"] = float(ma.corr(mb))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetch_rates()
    lv, info = _build_cached()
    log.info("built %d columns; unbridged: %s", lv.shape[1],
             {k: v.get("n_unbridged") for k, v in info.items() if v.get("n_unbridged")})
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        log.info("hygiene:\n%s", hygiene_rates().to_string())
    log.info("fidelity (dev window):\n%s", json.dumps(fidelity_rates(), indent=1))


if __name__ == "__main__":
    main()
