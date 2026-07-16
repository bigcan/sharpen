"""Taiwan market data ingestion via FinMind → cleaned Parquet + manifest (PoC).

Step one of the "less-liquid-market" thesis test (S553-cont, research artifact
``.agent/artifacts/taiwan_market_data_sourcing_research_s553.md``). Pulls FREE-tier
FinMind daily bars for TWSE/TPEx stocks and TAIFEX futures, runs the canonical
DATA-CLEAN, and writes a v2 manifest whose ``status`` is EARNED from the stale-print
scan — mirroring :mod:`finrl_pro_ds.data.cross_asset_loader` so downstream gates and
``protocol_v2`` treat Taiwan data identically to ETF/crypto data.

Why the HTTP REST route (not the FinMind SDK): the endpoint + dataset names are the
stable, verified contract (https://finmind.github.io/tutor/TaiwanMarket/Technical),
need only ``requests`` (no extra install), and don't pin us to SDK method names.

DESIGN INVARIANTS (see the research artifact §5):
  - **Raw, never pre-adjusted (LEAK-2).** We store FinMind ``TaiwanStockPrice`` (raw,
    unadjusted) as the training series. Back-adjusted prices (``TaiwanStockPriceAdj``,
    a PAID tier) bake *future* split/dividend factors into *past* bars — a look-ahead
    leak of the exact X2-coarse-bar shape. If you later license adjusted data, keep it
    in a SEPARATE table and adjust causally in the feature layer, never here.
  - **Survivorship.** Official/free daily feeds list only currently-trading tickers.
    ``--delisting`` fetches the delisting table (dataset id parameterised — confirm the
    name against your FinMind tier) so the backtest universe can be reconstructed
    point-in-time. Training only on survivors silently conditions on survival.
  - **Dates are already Gregorian.** FinMind returns ISO ``YYYY-MM-DD`` — NO ROC/民國
    +1911 conversion (that gotcha is only for the raw TWSE/TPEx/TAIFEX OpenAPIs).
    Daily bars are tz-naive calendar dates; GMT+8 only matters once you add intraday.
  - **Ground-truth cross-check (PF-XCHECK-style).** ``--xcheck`` samples a few
    ticker-days and compares FinMind close to the official TWSE OpenAPI
    ``STOCK_DAY_ALL``; a >tol divergence HALTS (it usually means an adjustment or
    units mismatch you must understand before trusting a backtest).

Usage:
    # Free tier (no token, ~300 req/hr). Token raises the quota (set FINMIND_TOKEN).
    python scripts/data/fetch_taiwan_finmind.py \
        --stocks 2330,2317,0050 --futures TX,MTX \
        --start 2015-01-01 --out data/taiwan

    python scripts/data/fetch_taiwan_finmind.py --stocks 2330 --start 2020-01-01 \
        --delisting --xcheck --dry_run

Datasets (FinMind, verified names): TaiwanStockPrice (raw daily, free, 1994-10-01→),
TaiwanFuturesDaily (free, 1998-07-01→), TaiwanStockPriceAdj (adjusted, PAID).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("fetch_taiwan_finmind")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

# Manifest schema version — keep in lockstep with cross_asset_loader's contract so a
# downstream gate can read either loader's manifest. v2 == the stale-print-scan era.
LOADER_MANIFEST_VERSION = 2

# Canonical project OHLCV contract (lowercase) — matches cross_asset_loader._OHLCV.
_OHLCV = ["open", "high", "low", "close", "volume"]

# FinMind field → our contract. FinMind uses `max`/`min` for high/low, and a different
# volume column for stocks vs futures (handled in _normalize).
_STOCK_MAP = {"open": "open", "max": "high", "min": "low", "close": "close",
              "Trading_Volume": "volume"}
_FUT_MAP = {"open": "open", "max": "high", "min": "low", "close": "close",
            "volume": "volume"}


# --------------------------------------------------------------------------- #
# FinMind HTTP
# --------------------------------------------------------------------------- #
# Transient network failures worth retrying with capped exponential backoff. These are raised by
# `requests.get()` BEFORE any HTTP status is returned (a WinError 10054 reset mid-run, a DNS blip, a
# read timeout), so the status-code retry loop below can never see them. Without an explicit catch a
# single transient reset propagates as a non-RuntimeError and crashes a multi-hour fetch — S553-cont-133
# saw one ConnectionError after 4,262 clean calls discard an in-progress channel because callers only
# `except RuntimeError`. A 402 quota block or a schema/tier error is NOT transient: those raise
# RuntimeError immediately and are never retried here.
_TRANSIENT_NET = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _finmind_get(dataset: str, data_id: str | None, start: str, end: str | None,
                 token: str, *, max_retries: int = 3, net_retries: int = 5) -> pd.DataFrame:
    """One FinMind dataset pull → DataFrame. Surfaces the 402 quota error explicitly.

    Retries two disjoint classes of transient failure, each capped:
      - retryable HTTP *status* codes (non-200, non-402) up to ``max_retries`` (linear backoff);
      - requests-level network errors (``_TRANSIENT_NET``) up to ``net_retries`` (exponential backoff,
        capped at 30 s) — raised before any status, so the status loop cannot catch them. On final
        give-up they become a ``RuntimeError`` so callers that ``except RuntimeError`` (per-name skip
        paths) degrade gracefully instead of crashing the whole run.
    A 402 quota block or a schema/tier ``RuntimeError`` is NOT transient and propagates immediately.
    """
    params = {"dataset": dataset, "start_date": start}
    if data_id:
        params["data_id"] = data_id
    if end:
        params["end_date"] = end
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _get_once() -> requests.Response:
        """`requests.get`, retrying only TRANSIENT network errors with capped exponential backoff."""
        for net_attempt in range(1, net_retries + 1):
            try:
                return requests.get(FINMIND_URL, headers=headers, params=params, timeout=60)
            except _TRANSIENT_NET as e:
                if net_attempt >= net_retries:
                    raise RuntimeError(
                        f"FinMind network error on {dataset}/{data_id} after {net_retries} "
                        f"retries: {type(e).__name__}: {e}") from e
                log.warning("%s/%s transient net error (attempt %d/%d): %s — backing off",
                            dataset, data_id, net_attempt, net_retries, type(e).__name__)
                time.sleep(min(2 ** net_attempt, 30))
        raise RuntimeError(f"FinMind network: unreachable retry fallthrough on {dataset}/{data_id}")

    for attempt in range(1, max_retries + 1):
        resp = _get_once()
        if resp.status_code == 402:
            raise RuntimeError(
                f"FinMind quota exhausted (HTTP 402) on {dataset}/{data_id}. "
                "Set FINMIND_TOKEN for the higher free quota, or slow down."
            )
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("status") not in (200, None):
                raise RuntimeError(f"FinMind error on {dataset}/{data_id}: {payload.get('msg')}")
            return pd.DataFrame(payload.get("data", []))
        if attempt < max_retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"FinMind HTTP {resp.status_code} on {dataset}/{data_id}: {resp.text[:200]}")


def _normalize(df: pd.DataFrame, ticker: str, field_map: dict) -> pd.DataFrame:
    """FinMind frame → tidy ``[date, ticker, open, high, low, close, volume]`` (ascending)."""
    if df.empty:
        return df
    missing = [c for c in field_map if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: FinMind frame missing {missing} (got {list(df.columns)})")
    out = df.rename(columns=field_map)[["date", *_OHLCV]].copy()
    out["date"] = pd.to_datetime(out["date"])          # FinMind dates are ISO/Gregorian
    out["ticker"] = str(ticker)
    out = out[["date", "ticker", *_OHLCV]].sort_values("date").reset_index(drop=True)
    for c in _OHLCV:                                    # FinMind sometimes ships numerics as str
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# DATA-CLEAN — single source of truth (lazy import, exactly as cross_asset_loader)
# --------------------------------------------------------------------------- #
def _clean_long(long: pd.DataFrame, threshold: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Per-ticker outlier detect/repair + stale-print scan via scripts.clean_ohlcv."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean_ohlcv import detect_outliers, detect_stale_runs, repair_outliers

    report: dict[str, dict] = {}
    cleaned: list[pd.DataFrame] = []
    for tk, g in long.groupby("ticker", sort=True):
        df = g.set_index("date")[_OHLCV].copy()
        valid = df[["open", "high", "low", "close"]].dropna(how="all")
        if df.empty or valid.empty:
            report[tk] = {"rows": 0, "outliers_repaired": 0, "stale_suspect_frac": 0.0,
                          "stale_pnl_share": 0.0, "flat_ohlc_spikes": 0, "stale_flagged": False}
            cleaned.append(g)
            continue
        det = detect_outliers(df, threshold=threshold)
        n_bad = int((det["bad_high"] | det["bad_low"]).sum())
        if n_bad:
            df = repair_outliers(df, det, threshold=threshold)
        stale = detect_stale_runs(df)                  # POST-repair, DatetimeIndex
        report[str(tk)] = {
            "rows": int(len(df)), "outliers_repaired": n_bad,
            "stale_suspect_frac": stale["stale_suspect_frac"],
            "stale_pnl_share": stale["stale_pnl_share"],
            "flat_ohlc_spikes": stale["flat_ohlc_spikes"],
            "stale_flagged": bool(stale["flagged"]),
        }
        out = df.reset_index()
        out["ticker"] = str(tk)
        cleaned.append(out[["date", "ticker", *_OHLCV]])
    long_clean = pd.concat(cleaned, ignore_index=True).sort_values(
        ["date", "ticker"]).reset_index(drop=True)
    return long_clean, report


# --------------------------------------------------------------------------- #
# Ground-truth cross-check (PF-XCHECK-style)
# --------------------------------------------------------------------------- #
def _xcheck_against_twse(long: pd.DataFrame, tol: float = 0.30) -> dict:
    """Compare FinMind close to the official TWSE OpenAPI for the latest snapshot day.

    STOCK_DAY_ALL is a latest-trading-day snapshot, so we can only cross-check tickers
    whose FinMind series reaches that day. A >tol relative divergence HALTS — it usually
    means raw-vs-adjusted or a units mismatch you must resolve before trusting a backtest.
    """
    resp = requests.get(TWSE_STOCK_DAY_ALL, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    twse = {r["Code"]: r for r in rows}
    # STOCK_DAY_ALL is a single-day snapshot; its `Date` is ROC YYMMDD (+1911). We can only
    # compare FinMind rows stamped that SAME day — comparing a 2024 FinMind bar to today's
    # snapshot is a date mismatch, not a data error (the bug the first smoke test caught).
    snap_date = None
    if rows and rows[0].get("Date"):
        roc = str(rows[0]["Date"])                       # e.g. "1150626" → 2026-06-26
        snap_date = pd.Timestamp(year=int(roc[:-4]) + 1911, month=int(roc[-4:-2]),
                                 day=int(roc[-2:])).normalize()
    checked, diverged = [], []
    if snap_date is None:
        return {"checked": [], "diverged": [], "tol": tol, "halt": False,
                "skipped": "no snapshot date in TWSE response"}
    on_snap = long[long["date"].dt.normalize() == snap_date]
    for _, row in on_snap.iterrows():
        code = str(row["ticker"])
        ref = twse.get(code)
        if not ref:
            continue
        try:
            off_close = float(str(ref["ClosingPrice"]).replace(",", ""))
        except (ValueError, KeyError):
            continue
        fin_close = float(row["close"])
        rel = abs(fin_close - off_close) / off_close if off_close else 0.0
        rec = {"ticker": code, "finmind_close": fin_close, "twse_close": off_close,
               "rel_div": round(rel, 4), "date": str(snap_date.date())}
        checked.append(rec)
        if rel > tol:
            diverged.append(rec)
    out = {"checked": checked, "diverged": diverged, "tol": tol, "halt": bool(diverged),
           "snapshot_date": str(snap_date.date())}
    if not checked:
        out["skipped"] = (f"no FinMind rows on TWSE snapshot day {snap_date.date()} "
                          "(fetch through today to enable the cross-check)")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Fetch + DATA-CLEAN Taiwan daily bars via FinMind")
    ap.add_argument("--stocks", default="", help="Comma TWSE/TPEx stock ids, e.g. 2330,2317,0050")
    ap.add_argument("--futures", default="", help="Comma FinMind futures_id, e.g. TX,MTX,TMF")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/taiwan", help="Output dir (under project root)")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--delisting", action="store_true",
                    help="Also fetch the delisting table (survivorship). Confirm dataset id.")
    ap.add_argument("--delisting-dataset", default="TaiwanStockDelisting")
    ap.add_argument("--xcheck", action="store_true", help="Cross-check vs official TWSE OpenAPI")
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between FinMind calls")
    ap.add_argument("--dry_run", action="store_true", help="Fetch + clean + report, don't write")
    args = ap.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    futures = [f.strip() for f in args.futures.split(",") if f.strip()]
    if not stocks and not futures:
        ap.error("provide at least one of --stocks / --futures")

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fetch (raw) ---
    frames: list[pd.DataFrame] = []
    for sid in stocks:
        log.info("FinMind TaiwanStockPrice %s %s..%s", sid, args.start, args.end or "latest")
        frames.append(_normalize(
            _finmind_get("TaiwanStockPrice", sid, args.start, args.end, args.token),
            sid, _STOCK_MAP))
        time.sleep(args.sleep)
    for fid in futures:
        log.info("FinMind TaiwanFuturesDaily %s %s..%s", fid, args.start, args.end or "latest")
        fut = _finmind_get("TaiwanFuturesDaily", fid, args.start, args.end, args.token)
        # Futures table carries many contract_dates per day; keep the front/near continuous
        # series only if a contract_date column exists (else it's already a single series).
        if not fut.empty and "contract_date" in fut.columns:
            fut = fut.sort_values(["date", "contract_date"]).groupby("date", as_index=False).first()
        frames.append(_normalize(fut, fid, _FUT_MAP))
        time.sleep(args.sleep)

    frames = [f for f in frames if not f.empty]
    if not frames:
        log.error("No data returned for any instrument — aborting.")
        return 2
    raw_long = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "ticker"]).reset_index(drop=True)
    log.info("Fetched %d rows across %d instruments (%s..%s)", len(raw_long),
             raw_long["ticker"].nunique(), raw_long["date"].min().date(),
             raw_long["date"].max().date())

    # --- Optional cross-check (halt-on-divergence) ---
    xcheck = None
    if args.xcheck and stocks:
        xcheck = _xcheck_against_twse(raw_long, tol=0.30)
        if xcheck["halt"]:
            log.error("PF-XCHECK HALT — FinMind vs TWSE OpenAPI divergence >30%%: %s",
                      xcheck["diverged"])
            log.error("This usually means raw-vs-adjusted or a units mismatch. Resolve first.")
            return 3
        if xcheck.get("skipped"):
            log.info("PF-XCHECK skipped — %s", xcheck["skipped"])
        else:
            log.info("PF-XCHECK ok — %d tickers within tol vs TWSE snapshot %s",
                     len(xcheck["checked"]), xcheck.get("snapshot_date"))

    # --- DATA-CLEAN ---
    clean_long, report = _clean_long(raw_long, threshold=args.threshold)

    # --- Optional delisting (survivorship) ---
    delisting_rows = 0
    if args.delisting:
        try:
            dl = _finmind_get(args.delisting_dataset, None, "1990-01-01", args.end, args.token)
            delisting_rows = int(len(dl))
            if not args.dry_run and delisting_rows:
                dl.to_parquet(out_dir / "delisting.parquet", index=False)
            log.info("Delisting table (%s): %d rows", args.delisting_dataset, delisting_rows)
        except RuntimeError as e:  # wrong dataset id / paid tier — surface, don't crash the pull
            log.warning("Delisting fetch failed (%s) — confirm dataset id / tier: %s",
                        args.delisting_dataset, e)

    # --- EARN manifest status from the stale-print scan (P1-05) ---
    flagged = sorted(tk for tk, r in report.items() if r.get("stale_flagged"))
    max_pnl_share = max((r.get("stale_pnl_share", 0.0) for r in report.values()), default=0.0)
    total_outliers = int(sum(r["outliers_repaired"] for r in report.values()))
    status = "FAIL" if max_pnl_share >= 0.02 else ("WARN" if flagged else "PASS")
    payload_hash = hashlib.sha256(
        pd.util.hash_pandas_object(clean_long, index=True).values.tobytes()).hexdigest()[:16]

    manifest = {
        "stage": "data-prep",
        "status": status,
        "loader_manifest_version": LOADER_MANIFEST_VERSION,
        "source": "finmind_http_free",
        "adjusted": False,                       # raw prices — adjust causally downstream (LEAK-2)
        "date_encoding": "gregorian_iso",        # NOT ROC — FinMind already converts
        "timezone": "Asia/Taipei (GMT+8); daily bars tz-naive calendar dates",
        "frequency": "1d",
        "datasets": {"stocks": "TaiwanStockPrice", "futures": "TaiwanFuturesDaily",
                     "delisting": args.delisting_dataset if args.delisting else None},
        "stocks": stocks, "futures": futures,
        "n_instruments": int(clean_long["ticker"].nunique()),
        "start": args.start, "end": args.end,
        "n_rows": int(len(clean_long)),
        "date_min": str(clean_long["date"].min().date()),
        "date_max": str(clean_long["date"].max().date()),
        "clean_threshold": args.threshold,
        "total_outliers_repaired": total_outliers,
        "delisting_rows": delisting_rows,
        "stale_scan": {"flagged_tickers": flagged,
                       "max_stale_pnl_share": round(float(max_pnl_share), 5),
                       "stale_pnl_fail_threshold": 0.02},
        "xcheck": xcheck,
        "per_ticker": report,
        "content_sha256_16": payload_hash,
    }

    if args.dry_run:
        log.info("DRY RUN — status=%s, %d rows, %d outliers repaired. No files written.",
                 status, len(clean_long), total_outliers)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    raw_path = out_dir / "taiwan_daily_raw.parquet"     # the .bak (pre-clean, raw)
    clean_path = out_dir / "taiwan_daily.parquet"
    manifest_path = out_dir / "taiwan_daily.manifest.json"
    raw_long.to_parquet(raw_path, index=False)
    clean_long.to_parquet(clean_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if status != "PASS":
        log.warning("Manifest status=%s (stale-flagged: %s, max_stale_pnl_share=%.4f)",
                    status, flagged, max_pnl_share)
    log.info("Wrote %s (%d rows) + raw + manifest [%s]", clean_path, len(clean_long), status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
