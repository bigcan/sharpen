"""Deribit options-VRP data loader (Phase 0 of the crypto-options volatility
risk-premium strategy — see ``.agent/artifacts/options_vol_harvest_architecture.md``).

Free, single-venue, point-in-time loader built on Deribit's public REST API
(no auth, no third-party data vendor). It fetches the three series the Phase-1
linear falsification needs:

  * DVOL  — Deribit's 30-day constant-maturity implied-volatility index
            (``get_volatility_index_data``; BTC/ETH; history from 2021-04-01).
            This is the causal ATM-IV signal (the "sell overpriced vol" leg).
  * PERP  — the perpetual OHLC (``get_tradingview_chart_data`` on
            ``<CCY>-PERPETUAL``) — the spot/hedge price path used for realized
            vol and the delta-hedge leg.
  * FUND  — perpetual funding history (``get_funding_rate_history``) — the
            carry cost of holding the delta hedge.

Why DVOL rather than a reconstructed option chain for the *gate*: DVOL is a
published, model-free 30-day IV index (VIX-analogue). It carries **no
survivorship risk** (it is an index, not a set of expiring instruments) and is
directly causal, so the cheapest honest VRP falsification can be run from it
before we ever pay for / reconstruct a full point-in-time chain. The
higher-fidelity per-instrument path (expired-chain OHLC via
``get_tradingview_chart_data`` per option, for skew/strangle and the RL env) is
a documented Phase-3 extension and is intentionally **not** built here.

Network access runs on the operator machine; this module is import-safe (the
HTTP calls only fire when a fetch function is invoked or via ``__main__``).
Uses only the standard library for HTTP (urllib) so it adds no dependency.

Invariants honoured: DATA-CLEAN (validation + ``.manifest.json``),
no ``print`` (logging), LEAK-2 (the series are point-in-time; causality of
*derived* features is enforced in ``options_vol_features``).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DERIBIT_BASE = "https://www.deribit.com/api/v2/public/"
USER_AGENT = "finrl-pro-ds/vrp-harvest (Phase0 loader)"

# DVOL index history begins 2021-04-01 on Deribit's public endpoint.
DVOL_INCEPTION = "2021-04-01"

# Per-request safety chunks (calendar days) — Deribit caps points/response.
_DVOL_CHUNK_DAYS = 300
_CHART_CHUNK_DAYS = 900
_FUNDING_CHUNK_DAYS = 25   # funding endpoint caps ~744 hourly points (~31d)

_REQUEST_DELAY_S = 0.15    # polite pacing between paginated requests
_MAX_RETRIES = 5

MS_PER_DAY = 86_400_000

DEFAULT_CACHE_DIR = Path("data/processed/deribit")


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------
def _to_ms(ts: str | int | datetime) -> int:
    """Normalize a date/datetime/epoch-ms into integer epoch milliseconds (UTC)."""
    if isinstance(ts, int):
        return ts
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    # string date / datetime
    dt = pd.Timestamp(ts, tz="UTC").to_pydatetime()
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _get(method: str, params: dict) -> dict:
    """Call a Deribit public endpoint with retries; return the ``result`` payload."""
    url = DERIBIT_BASE + method + "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload and payload["error"]:
                raise RuntimeError(f"Deribit API error for {method}: {payload['error']}")
            return payload["result"]
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError) as exc:
            last_err = exc
            backoff = _REQUEST_DELAY_S * (2 ** attempt)
            logger.warning("Deribit %s attempt %d/%d failed: %s (retry in %.1fs)",
                           method, attempt + 1, _MAX_RETRIES, exc, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"Deribit {method} failed after {_MAX_RETRIES} retries: {last_err}")


def _chunks(start_ms: int, end_ms: int, chunk_days: int):
    """Yield (chunk_start_ms, chunk_end_ms) windows covering [start, end]."""
    step = chunk_days * MS_PER_DAY
    a = start_ms
    while a < end_ms:
        b = min(a + step, end_ms)
        yield a, b
        a = b


# ---------------------------------------------------------------------------
# Endpoint fetchers (each returns a UTC-indexed DataFrame)
# ---------------------------------------------------------------------------
def fetch_dvol(currency: str, start: str | int | datetime, end: str | int | datetime,
               resolution: str = "1D") -> pd.DataFrame:
    """DVOL implied-vol index OHLC. Columns: open/high/low/close (vol points, e.g. 55.0)."""
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows: list[list[float]] = []
    for a, b in _chunks(start_ms, end_ms, _DVOL_CHUNK_DAYS):
        res = _get("get_volatility_index_data", {
            "currency": currency, "start_timestamp": a, "end_timestamp": b,
            "resolution": resolution,
        })
        rows.extend(res.get("data", []))
        time.sleep(_REQUEST_DELAY_S)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["open", "high", "low", "close"]].astype(float)


def fetch_perp_chart(currency: str, start: str | int | datetime, end: str | int | datetime,
                     resolution: str = "1D") -> pd.DataFrame:
    """Perpetual OHLCV from get_tradingview_chart_data on ``<CCY>-PERPETUAL``."""
    instrument = f"{currency}-PERPETUAL"
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    frames: list[pd.DataFrame] = []
    for a, b in _chunks(start_ms, end_ms, _CHART_CHUNK_DAYS):
        res = _get("get_tradingview_chart_data", {
            "instrument_name": instrument, "start_timestamp": a, "end_timestamp": b,
            "resolution": resolution,
        })
        if res.get("status") == "no_data" or not res.get("ticks"):
            continue
        frames.append(pd.DataFrame({
            "timestamp": res["ticks"], "open": res["open"], "high": res["high"],
            "low": res["low"], "close": res["close"], "volume": res["volume"],
        }))
        time.sleep(_REQUEST_DELAY_S)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_funding(currency: str, start: str | int | datetime, end: str | int | datetime) -> pd.DataFrame:
    """Perpetual funding history. Column ``interest_8h`` (per-8h funding rate, fraction)."""
    instrument = f"{currency}-PERPETUAL"
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows: list[dict] = []
    for a, b in _chunks(start_ms, end_ms, _FUNDING_CHUNK_DAYS):
        res = _get("get_funding_rate_history", {
            "instrument_name": instrument, "start_timestamp": a, "end_timestamp": b,
        })
        rows.extend(res if isinstance(res, list) else [])
        time.sleep(_REQUEST_DELAY_S)
    if not rows:
        return pd.DataFrame(columns=["interest_8h", "index_price"])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    keep = [c for c in ("interest_8h", "interest_1h", "index_price") if c in df.columns]
    return df[keep].astype(float)


# ---------------------------------------------------------------------------
# Validation (DATA-CLEAN)
# ---------------------------------------------------------------------------
def _validate(df: pd.DataFrame, name: str, price_cols: tuple[str, ...] = ("close",),
              *, require_positive: bool = True, ohlc: bool = False,
              check_gaps: bool = False, max_gap_days: float = 4.0,
              stale_max: int = 15) -> list[str]:
    """Return a list of DATA-CLEAN validation issues (empty == clean). Never raises.

    Covers (CLAUDE.md DATA-CLEAN): monotonic/duplicate index, NaN + (optional)
    non-positivity, OHLC invariants (``high >= max(O,C)``, ``low <= min(O,C)``,
    ``high >= low``), daily gap/continuity, and a stale-feed run check. ``load`` then
    raises in strict mode — the prior loader merely *warned* and the issues never
    gated, so a corrupt ``--refresh`` would have graded silently (V1-06)."""
    issues: list[str] = []
    if df.empty:
        issues.append(f"{name}: EMPTY")
        return issues
    if not df.index.is_monotonic_increasing:
        issues.append(f"{name}: index not monotonic")
    if df.index.has_duplicates:
        issues.append(f"{name}: duplicate timestamps")
    for c in price_cols:
        if c not in df.columns:
            issues.append(f"{name}: missing column {c}")
            continue
        col = df[c]
        if col.isna().any():
            issues.append(f"{name}: {int(col.isna().sum())} NaNs in {c}")
        if require_positive and (col.dropna() <= 0).any():
            issues.append(f"{name}: non-positive values in {c}")
    # OHLC invariants — a decimal-shift / aggregation artifact violates these
    if ohlc and {"open", "high", "low", "close"}.issubset(df.columns):
        o_a = df["open"].to_numpy(dtype=float)
        h_a = df["high"].to_numpy(dtype=float)
        l_a = df["low"].to_numpy(dtype=float)
        c_a = df["close"].to_numpy(dtype=float)
        max_oc = np.maximum(o_a, c_a)
        min_oc = np.minimum(o_a, c_a)
        n_h = int((h_a < max_oc - 1e-9).sum())
        n_l = int((l_a > min_oc + 1e-9).sum())
        n_hl = int((h_a < l_a - 1e-9).sum())
        if n_h:
            issues.append(f"{name}: {n_h} bars with high < max(open,close)")
        if n_l:
            issues.append(f"{name}: {n_l} bars with low > min(open,close)")
        if n_hl:
            issues.append(f"{name}: {n_hl} bars with high < low")
    # Gap / continuity (daily cadence): flag any calendar gap exceeding max_gap_days
    if check_gaps and len(df) > 1:
        gap_days = np.diff(df.index.asi8) / (MS_PER_DAY * 1_000_000)  # ns -> days
        big = int((gap_days > max_gap_days).sum())
        if big:
            issues.append(f"{name}: {big} gap(s) > {max_gap_days:g}d (max {gap_days.max():.1f}d)")
    # Stale feed: a long run of identical closes => a frozen series, not real ticks
    if "close" in df.columns and len(df) > stale_max:
        close_a = df["close"].to_numpy(dtype=float)
        longest = run = 1
        for k in range(1, len(close_a)):
            run = run + 1 if close_a[k] == close_a[k - 1] else 1
            longest = max(longest, run)
        if longest > stale_max:
            issues.append(f"{name}: stale close run of {longest} (> {stale_max})")
    return issues


# ---------------------------------------------------------------------------
# Top-level load + cache
# ---------------------------------------------------------------------------
@dataclass
class RawOptionsData:
    """Point-in-time per-asset Deribit series (DVOL + perp + funding)."""
    dvol: dict[str, pd.DataFrame] = field(default_factory=dict)      # ccy -> OHLC vol points
    perp: dict[str, pd.DataFrame] = field(default_factory=dict)      # ccy -> OHLCV
    funding: dict[str, pd.DataFrame] = field(default_factory=dict)   # ccy -> interest_8h
    manifest: dict = field(default_factory=dict)


def load(config: dict, *, cache_dir: Path | str = DEFAULT_CACHE_DIR,
         use_cache: bool = True, refresh: bool = False,
         strict: bool = True) -> RawOptionsData:
    """Fetch (or load cached) DVOL + perp + funding for the configured universe.

    ``config`` keys used (with sensible defaults):
        universe.assets   : list[str]  (subset of {"BTC","ETH"}; default both)
        data.start_date   : str        (default DVOL inception 2021-04-01)
        data.end_date     : str|None   (default = now)
        data.resolution   : str        (default "1D")

    ``strict`` (default True): DATA-CLEAN gate — raise ``RuntimeError`` if any frame
    fails ``_validate`` (OHLC invariants, gaps, stale runs, NaN/non-positive). The
    issues are always recorded in the manifest first; ``strict=False`` downgrades to
    a warning for exploratory fetches.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    assets = config.get("universe", {}).get("assets", ["BTC", "ETH"])
    start = config.get("data", {}).get("start_date") or DVOL_INCEPTION
    end = config.get("data", {}).get("end_date") or _now_ms()
    resolution = config.get("data", {}).get("resolution", "1D")

    out = RawOptionsData()
    issues: dict[str, list[str]] = {}

    for ccy in assets:
        dvol_p = cache_dir / f"dvol_{ccy}_{resolution}.parquet"
        perp_p = cache_dir / f"perp_{ccy}_{resolution}.parquet"
        fund_p = cache_dir / f"funding_{ccy}.parquet"

        if use_cache and not refresh and dvol_p.exists() and perp_p.exists() and fund_p.exists():
            logger.info("Loading cached Deribit data for %s", ccy)
            dvol_df = pd.read_parquet(dvol_p)
            perp_df = pd.read_parquet(perp_p)
            fund_df = pd.read_parquet(fund_p)
        else:
            logger.info("Fetching Deribit data for %s (%s -> %s, res=%s)", ccy, start, end, resolution)
            # Back up the existing graded parquets before a refresh overwrites them, so a
            # bad fetch never silently destroys the data that produced the verdict (V1-07).
            for p in (dvol_p, perp_p, fund_p):
                if p.exists():
                    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            dvol_df = fetch_dvol(ccy, start, end, resolution)
            perp_df = fetch_perp_chart(ccy, start, end, resolution)
            fund_df = fetch_funding(ccy, start, end)
            dvol_df.to_parquet(dvol_p)
            perp_df.to_parquet(perp_p)
            fund_df.to_parquet(fund_p)

        issues[ccy] = (
            _validate(dvol_df, f"dvol_{ccy}", ("open", "high", "low", "close"),
                      ohlc=True, check_gaps=True)
            + _validate(perp_df, f"perp_{ccy}", ("open", "high", "low", "close"),
                        ohlc=True, check_gaps=True)
            # funding can legitimately be negative (perp carry) — no positivity check
            + _validate(fund_df, f"funding_{ccy}", ("interest_8h",), require_positive=False)
        )
        out.dvol[ccy] = dvol_df
        out.perp[ccy] = perp_df
        out.funding[ccy] = fund_df

    out.manifest = {
        "source": "deribit_public_api",
        "endpoints": ["get_volatility_index_data", "get_tradingview_chart_data",
                      "get_funding_rate_history"],
        "assets": assets,
        "start_date": str(start),
        "end_date": str(end),
        "resolution": resolution,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {ccy: {"dvol": int(len(out.dvol[ccy])), "perp": int(len(out.perp[ccy])),
                       "funding": int(len(out.funding[ccy]))} for ccy in assets},
        "validation_issues": issues,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(out.manifest, indent=2))
    n_issues = sum(len(v) for v in issues.values())
    if n_issues:
        logger.warning("Deribit load completed with %d validation issue(s): %s", n_issues, issues)
        if strict:
            raise RuntimeError(
                f"Deribit DATA-CLEAN failed with {n_issues} issue(s) (see manifest "
                f"{manifest_path}): {issues}")
    else:
        logger.info("Deribit load clean for assets %s", assets)
    return out


def _cli() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch + cache Deribit DVOL/perp/funding")
    ap.add_argument("--assets", default="BTC,ETH")
    ap.add_argument("--start", default=DVOL_INCEPTION)
    ap.add_argument("--end", default=None)
    ap.add_argument("--resolution", default="1D")
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-strict", dest="no_strict", action="store_true",
                    help="downgrade DATA-CLEAN failures from raise to warn (exploratory)")
    args = ap.parse_args()

    cfg = {
        "universe": {"assets": args.assets.split(",")},
        "data": {"start_date": args.start, "end_date": args.end, "resolution": args.resolution},
    }
    data = load(cfg, cache_dir=args.cache_dir, refresh=args.refresh, strict=not args.no_strict)
    logger.info("Manifest: %s", json.dumps(data.manifest["rows"], indent=2))


if __name__ == "__main__":
    _cli()
