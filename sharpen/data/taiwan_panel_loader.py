"""Taiwan cross-asset ETF ``Panel`` loader for the alpha-generation harness (TAIEX substrate).

The Taiwan analog of :mod:`cross_asset_panel_loader`: a liquid Taiwan-listed ETF cross-section
(equity / bond / commodity classes) the C3 miner ranks into a dollar-neutral candidate sleeve,
scored by its net-deflated uplift to the Taiwan base book (TX/TE/TF TSMOM — step 2). Universe +
class map live in ``configs/taiwan_cross_asset.yaml`` (scope artifact
``.agent/artifacts/taiex_mining_universe_scope_s553.md``).

Two deliberate differences vs the US cross-asset loader:
  * **Fetch backend is FinMind**, not yfinance — daily bars via the verified HTTP contract in
    ``scripts/data/fetch_taiwan_finmind`` (``TaiwanStockPrice`` for ETFs). Cleaning + the
    stale-print scan + the manifest status are the SAME single source of truth
    (:func:`cross_asset_loader._clean_wide`), so downstream gates treat Taiwan data identically.
  * ``meta.survivorship_free = False`` (an honest DOWNGRADE, not the US cell's ``True``). The
    free FinMind feed lists currently-trading tickers, and Taiwan ETFs do get merged/delisted, so
    the panel is NOT delisting-free — results here are an UPPER BOUND on survivorship grounds.

Returns are **causal total return** (H1 fix): FinMind ships raw (price-only) bars, which on a
distribution-heavy universe (0056/00878 high-div, bond ETFs) book each ex-dividend price drop as a
spurious ~-2% loss and distort every downstream return. We add each cash distribution back on its
ex-date, **forward-accumulated** (:func:`_apply_causal_total_return`), so ``close[t+1]/close[t]``
is the total return. This is the opposite of the PAID ``TaiwanStockPriceAdj`` / yfinance
back-adjustment, which bakes FUTURE split/div factors into PAST bars (LEAK-2): our factor starts at
1 and only ratchets UP at an ex-date, so no bar is ever rewritten by a later dividend. The Panel
carries no forward-looking field; ``forward_returns`` is a LABEL, and the DSL alphas evaluated on
this panel are causal-by-construction (tripwired).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.data import cross_asset_loader as cal
from sharpen.data.cross_asset_panel_loader import _class_ids
from sharpen.signals.features import Panel

log = logging.getLogger("taiwan_panel_loader")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "taiwan_cross_asset.yaml"
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "taiwan_panel"


def load_taiwan_universe(
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Return ``(etfs, {etf: asset_class}, futures)`` from the Taiwan cross-asset config.

    ``etfs`` are the panel (rank-L/S) cross-section; ``futures`` is the base-sleeve substrate
    (TX/TE/TF), returned here so a single config is the source of truth but NOT loaded into the
    panel — the base sleeves fetch them on their own universe (step 2)."""
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    uni = cfg.get("universe", {})
    etfs = [str(a) for a in uni.get("assets", [])]
    class_map = {str(k): str(v) for k, v in dict(uni.get("asset_class", {})).items()}
    futures = [str(f) for f in uni.get("futures", [])]
    if not etfs:
        raise ValueError(f"no universe.assets in {config_path}")
    return etfs, class_map, futures


# --------------------------------------------------------------------------- #
# FinMind fetch → wide (date × ticker) frames — the yfinance-shaped contract
# --------------------------------------------------------------------------- #
def fetch_taiwan_wide(
    etfs: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    futures: list[str] | tuple[str, ...] = (),
    token: str | None = None,
    sleep: float = 0.4,
) -> dict[str, pd.DataFrame]:
    """Daily OHLCV via FinMind → ``{open,high,low,close,volume}`` wide frames (date × ticker).

    Reuses the verified FinMind HTTP + normalize helpers from ``scripts.data.fetch_taiwan_finmind``
    (single source of truth for the endpoint/field contract), lazy-imported so this library module
    stays import-clean (the same pattern ``_clean_wide`` uses for ``scripts.clean_ohlcv``). ETFs pull
    ``TaiwanStockPrice``; ``futures`` pull ``TaiwanFuturesDaily`` (front/near continuous — first
    contract_date per day). Columns are reindexed to ``etfs + futures`` order, index ascending.
    Positional ``(etfs, start, end)`` matches the injectable ``fetch_fn`` signature used offline.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.data.fetch_taiwan_finmind import (  # noqa: PLC0415 (lazy, single-source contract)
        _FUT_MAP,
        _STOCK_MAP,
        _finmind_get,
        _normalize,
    )

    token = token if token is not None else os.environ.get("FINMIND_TOKEN", "")
    frames: list[pd.DataFrame] = []
    for sid in etfs:
        frames.append(_normalize(_finmind_get("TaiwanStockPrice", sid, start, end, token),
                                 sid, _STOCK_MAP))
        time.sleep(sleep)
    for fid in futures:
        fut = _finmind_get("TaiwanFuturesDaily", fid, start, end, token)
        if not fut.empty and "contract_date" in fut.columns:  # keep the near/front continuous series
            fut = fut.sort_values(["date", "contract_date"]).groupby("date", as_index=False).first()
        frames.append(_normalize(fut, fid, _FUT_MAP))
        time.sleep(sleep)

    frames = [f for f in frames if not f.empty]
    ids = list(etfs) + list(futures)
    if not frames:
        raise RuntimeError(f"FinMind returned no data for {ids} ({start}..{end})")
    long = pd.concat(frames, ignore_index=True)
    return {c: (long.pivot(index="date", columns="ticker", values=c)
                .reindex(columns=ids).sort_index()) for c in cal._OHLCV}


# --------------------------------------------------------------------------- #
# Causal total-return adjustment (H1) — add each cash distribution back on its ex-date
# --------------------------------------------------------------------------- #
def fetch_taiwan_dividends(
    ids: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    token: str | None = None,
    sleep: float = 0.35,
) -> dict[str, pd.DataFrame]:
    """Per-ticker realized ex-dividend events → ``{ticker: DataFrame[ex_date, amount]}`` via FinMind.

    Uses ``TaiwanStockDividendResult`` (the REALIZED ex-dividend event, not the forward-looking policy
    table): its ``date`` is the ex-dividend trading date and ``amount = before_price - after_price`` is
    the exact reference-price adjustment (for this cash-paying ETF universe, the cash distribution).
    Causal — an ex-dividend event is stamped on its own ex-date. Futures / dividend-less names return
    an empty frame (no records, or a dataset miss on a non-stock id), never fatal.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.data.fetch_taiwan_finmind import _finmind_get  # noqa: PLC0415 (lazy, single-source)

    token = token if token is not None else os.environ.get("FINMIND_TOKEN", "")
    out: dict[str, pd.DataFrame] = {}
    empty = pd.DataFrame({"ex_date": pd.Series([], dtype="datetime64[ns]"),
                          "amount": pd.Series([], dtype=float)})
    for tk in ids:
        try:
            df = _finmind_get("TaiwanStockDividendResult", tk, start, end, token)
        except RuntimeError as e:             # dataset miss (e.g. a future id) — no dividends, skip
            log.info("taiwan_panel_loader: no dividend data for %s (%s)", tk, e)
            out[tk] = empty.copy()
            time.sleep(sleep)
            continue
        if df.empty or not {"date", "before_price", "after_price"} <= set(df.columns):
            out[tk] = empty.copy()
        else:
            amt = (pd.to_numeric(df["before_price"], errors="coerce")
                   - pd.to_numeric(df["after_price"], errors="coerce"))
            ev = pd.DataFrame({"ex_date": pd.to_datetime(df["date"]), "amount": amt}).dropna()
            out[tk] = ev[ev["amount"] > 0].sort_values("ex_date").reset_index(drop=True)
        time.sleep(sleep)
    return out


def _apply_causal_total_return(
    wide: dict[str, pd.DataFrame], dividends: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Turn raw price-only OHLC into a CAUSAL total-return series (H1 fix) → ``(adjusted, n_events)``.

    Per ticker, build a per-date multiplier ``m[t]=1`` except ``m[ex] = 1 + D/close_raw[ex]`` on each
    ex-dividend bar, and scale O/H/L/C (NOT volume) by ``cf = m.cumprod()``. Then
    ``adj_close[t]/adj_close[t-1] == (close_raw[t] + D_t)/close_raw[t-1]`` (the total return), while
    ``cf`` starts at 1 and only RATCHETS UP at an ex-date — **bars before a dividend are byte-for-byte
    unchanged** (forward, not back, adjustment → LEAK-2 clean). All four OHLC fields share one factor,
    preserving intraday ratios for OHLC-using DSL alphas.
    """
    close = wide["close"]
    idx = close.index
    factors = pd.DataFrame(1.0, index=idx, columns=close.columns)
    applied: dict[str, int] = {tk: 0 for tk in close.columns}
    for tk in close.columns:
        ev = dividends.get(tk)
        if ev is None or not len(ev):
            continue
        col = close[tk]
        jcol = factors.columns.get_loc(tk)
        for ex_date, amount in zip(ev["ex_date"], ev["amount"]):
            pos = int(idx.searchsorted(pd.Timestamp(ex_date), side="left"))  # first bar on/after ex
            if pos >= len(idx):
                continue                       # ex-date beyond the loaded window — no effect here
            p = col.iloc[pos]
            if not np.isfinite(p) or p <= 0:   # name not yet listed / bad print — skip this event
                continue
            factors.iloc[pos, jcol] *= (1.0 + float(amount) / float(p))
            applied[tk] += 1
    cf = factors.cumprod(axis=0)
    adj = dict(wide)
    for field in ("open", "high", "low", "close"):
        adj[field] = wide[field] * cf
    return adj, applied


# --------------------------------------------------------------------------- #
# Fetch + DATA-CLEAN + cache + manifest — mirrors cross_asset_loader.fetch_and_clean
# with the FinMind backend (cleaning/scan/status are the shared single source of truth).
# --------------------------------------------------------------------------- #
def fetch_and_clean_taiwan(
    etfs: list[str],
    start: str,
    end: str | None,
    *,
    futures: list[str] | tuple[str, ...] = (),
    cache_dir: Path = DEFAULT_CACHE_DIR,
    token: str | None = None,
    threshold: float = 0.05,
    force_refetch: bool = False,
    require_fresh: bool = False,
    freshness_tol_days: int = 5,
    stale_pnl_fail_threshold: float = 0.02,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fetch FinMind OHLCV, DATA-CLEAN it, cache raw + cleaned + manifest → ``(wide, manifest)``.

    Cache reuse requires the cached asset superset AND coverage of BOTH window ends
    (:func:`cross_asset_loader._cache_covers_end` and ``_cache_covers_start``) AND a v2 stale-scan
    block — identical policy to the US loader, so a scheduled run cannot silently freeze on a
    frozen cache and a wider-history request cannot silently receive a narrower cached window.
    ``status`` is EARNED from the stale-print
    scan (``FAIL`` if any ticker's stale-print P&L share >= ``stale_pnl_fail_threshold`` — the
    gmgp1-gold class — ``WARN`` if flagged below, else ``PASS``).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "ohlcv_daily_raw.parquet"
    clean_path = cache_dir / "ohlcv_daily.parquet"
    manifest_path = cache_dir / "ohlcv_daily.manifest.json"
    ids = list(etfs) + list(futures)

    if clean_path.exists() and manifest_path.exists() and not force_refetch:
        manifest = json.loads(manifest_path.read_text())
        assets_ok = set(manifest.get("assets", [])) >= set(ids)
        fresh_ok = cal._cache_covers_end(manifest, end, require_fresh=require_fresh,
                                         tol_days=freshness_tol_days)
        start_ok = cal._cache_covers_start(manifest, start)
        scan_ok = ("stale_scan" in manifest
                   and manifest.get("loader_manifest_version", 1) >= cal.LOADER_MANIFEST_VERSION)
        if assets_ok and fresh_ok and scan_ok and start_ok:
            log.info("taiwan_panel_loader: using cached clean OHLCV (%s, window=%s..%s)",
                     clean_path, manifest.get("start"), manifest.get("date_max"))
            long = pd.read_parquet(clean_path)
            long["date"] = pd.to_datetime(long["date"])
            wide = {c: long.pivot(index="date", columns="ticker", values=c)
                    .reindex(columns=ids).sort_index() for c in cal._OHLCV}
            return cal._clip_window(wide, start, end), manifest
        log.info("taiwan_panel_loader: cache stale (assets_ok=%s fresh_ok=%s scan_ok=%s "
                 "start_ok=%s, cached_window=%s..%s, requested=%s..%s) → refetching",
                 assets_ok, fresh_ok, scan_ok, start_ok, manifest.get("start"),
                 manifest.get("date_max"), start, end)

    log.info("taiwan_panel_loader: fetching %d ids %s..%s via FinMind", len(ids), start, end)
    wide = fetch_taiwan_wide(list(etfs), start, end, futures=futures, token=token)
    cal._wide_to_long(wide).to_parquet(raw_path, index=False)  # .bak (raw, pre-clean)

    wide, clean_report = cal._clean_wide(wide, threshold=threshold)  # shared DATA-CLEAN + stale scan

    # H1: causal total-return. The stale/outlier scan above ran on RAW prices (correct — an ex-div
    # gap is OHLC-internally-consistent, not a fat-finger outlier), so we now add each cash
    # distribution back on its ex-date to get the total-return series downstream returns use. Only the
    # ETF ids can pay distributions; futures ids resolve to empty and pass through with cf==1.
    dividends = fetch_taiwan_dividends(list(etfs), start, end, token=token)
    wide, div_applied = _apply_causal_total_return(wide, dividends)

    long = cal._wide_to_long(wide)
    long.to_parquet(clean_path, index=False)

    idx = wide["close"].index
    total_outliers = int(sum(r["outliers_repaired"] for r in clean_report.values()))
    flagged_tickers = sorted(tk for tk, r in clean_report.items() if r.get("stale_flagged"))
    max_stale_pnl_share = max((r.get("stale_pnl_share", 0.0) for r in clean_report.values()),
                              default=0.0)
    if max_stale_pnl_share >= stale_pnl_fail_threshold:
        status = "FAIL"
    elif flagged_tickers:
        status = "WARN"
    else:
        status = "PASS"
    payload_hash = hashlib.sha256(
        pd.util.hash_pandas_object(long, index=True).values.tobytes()).hexdigest()[:16]
    manifest = {
        "stage": "data-prep",
        "status": status,
        "loader_manifest_version": cal.LOADER_MANIFEST_VERSION,
        "source": "finmind_http",
        "adjusted": True,                        # H1: causal total-return (raw + forward div add-back)
        "adjust_method": "causal_total_return_forward",
        "dividend_source": "finmind_TaiwanStockDividendResult",
        "dividend_events_applied": div_applied,  # {ticker: n ex-div events folded in}
        "frequency": "1d",
        "assets": list(ids),                     # superset the cache-hit check keys on
        "etfs": list(etfs),
        "futures": list(futures),
        "n_assets": len(ids),
        "start": str(start),
        "end": str(end) if end else None,
        "n_rows": int(len(idx)),
        "date_min": str(idx.min().date()) if len(idx) else None,
        "date_max": str(idx.max().date()) if len(idx) else None,
        "clean_threshold": threshold,
        "total_outliers_repaired": total_outliers,
        "stale_scan": {
            "flagged_tickers": flagged_tickers,
            "max_stale_pnl_share": round(float(max_stale_pnl_share), 5),
            "stale_pnl_fail_threshold": stale_pnl_fail_threshold,
        },
        "per_ticker": clean_report,
        "raw_cache": str(raw_path),
        "clean_cache": str(clean_path),
        "content_sha256_16": payload_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if status != "PASS":
        log.warning("taiwan_panel_loader: manifest status=%s (stale-flagged: %s, "
                    "max_stale_pnl_share=%.4f)", status, flagged_tickers, max_stale_pnl_share)
    log.info("taiwan_panel_loader: cleaned %d ids × %d rows (%d outliers repaired) → %s [%s]",
             len(ids), len(idx), total_outliers, clean_path, status)
    return wide, manifest


# --------------------------------------------------------------------------- #
# Panel builder
# --------------------------------------------------------------------------- #
def load_taiwan_panel(
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    adv_window: int = 20,
    universe: tuple[list[str], dict[str, str]] | None = None,
    token: str | None = None,
    fetch_fn=None,
    clean_fn=None,
    require_fresh: bool = False,
) -> Panel:
    """Build the Taiwan ETF cross-section ``Panel`` with an EARNED DATA-CLEAN status.

    Default path fetches via FinMind (:func:`fetch_and_clean_taiwan`) — canonical DATA-CLEAN,
    cached raw/clean/manifest, and the gmgp1-gold stale-print gate (``FAIL`` REFUSES to build).
    ``universe`` / ``fetch_fn`` / ``clean_fn`` are injectable for offline tests (that path derives
    an equivalent status from the clean report), mirroring :func:`load_cross_asset_panel`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if universe is not None:
        etfs, class_map = universe
    else:
        etfs, class_map, _futures = load_taiwan_universe(config_path)

    if fetch_fn is None and clean_fn is None:
        wide, manifest = fetch_and_clean_taiwan(
            etfs, start, end, cache_dir=cache_dir, token=token, require_fresh=require_fresh)
        status = str(manifest.get("status", "UNKNOWN"))
        scan = dict(manifest.get("stale_scan", {}))
        max_stale = float(scan.get("max_stale_pnl_share", 0.0))
        stale_flagged = int(len(scan.get("flagged_tickers", [])))
        date_max = manifest.get("date_max")
        content_sha = manifest.get("content_sha256_16")
        adjusted = bool(manifest.get("adjusted", False))
        adjust_method = manifest.get("adjust_method")
    else:                                        # injected (offline test) path — derive the status
        fetch = fetch_fn or fetch_taiwan_wide
        clean = clean_fn or cal._clean_wide
        wide = fetch(etfs, start, end)
        wide, clean_report = clean(wide)
        stale_flagged = sum(1 for r in clean_report.values() if r.get("stale_flagged"))
        max_stale = max((r.get("stale_pnl_share", 0.0) for r in clean_report.values()), default=0.0)
        status = "FAIL" if max_stale >= 0.02 else ("WARN" if stale_flagged else "PASS")
        date_max, content_sha = None, None
        adjusted, adjust_method = False, None    # offline synthetic wide is raw (no dividend fold-in)

    if status == "FAIL":                         # gmgp1-gold stale-print gate
        raise ValueError(
            f"taiwan panel DATA-CLEAN status=FAIL (max_stale_pnl_share={max_stale:.4f}) — "
            "refusing to build a panel on stale-print-contaminated data.")

    close_df = wide["close"]
    dates = close_df.index.to_numpy(dtype="datetime64[ns]")
    cols = list(close_df.columns)                # == etfs (reindexed by fetch)

    def arr(field: str) -> np.ndarray:
        return np.asarray(wide[field].to_numpy(), dtype=np.float64)

    open_, high, low, close, volume = (arr(f) for f in ("open", "high", "low", "close", "volume"))
    active = np.isfinite(close) & (close > 0)    # PIT membership: a name is tradeable once it lists
    # `adv_usd` is dollar-VOLUME (here TWD notional — FinMind volume is shares, close is TWD); used
    # only as a relative cost/size proxy, so the currency label is immaterial to the ranking.
    dollar = close * np.where(np.isfinite(volume), volume, np.nan)
    adv_usd = pd.DataFrame(dollar).rolling(adv_window, min_periods=1).mean().to_numpy()
    class_id = _class_ids(cols, class_map)

    meta = {
        # Honest DOWNGRADE vs the US ETF cell: Taiwan ETFs get merged/delisted and the free FinMind
        # feed lists current tickers only, so the panel is NOT delisting-free.
        "survivorship_free": False,
        "survivorship_note": ("Taiwan-listed-ETF UPPER BOUND — free FinMind feed lists currently-"
                              "trading tickers; ETF mergers/delistings not PIT-modeled. Can only "
                              "over-state, never falsely downgrade a verdict."),
        "source": "finmind_http+taiwan_etf",
        # H1: causal total-return on the real path (raw + forward dividend add-back); the offline
        # synthetic path is raw (no distributions to fold in).
        "adjusted": adjusted,
        "adjust_method": adjust_method,
        "universe_def": "10 liquid TW-listed ETFs × {tw_equity, bond, commodity} (taiwan_cross_asset.yaml)",
        "n_universe": len(cols),
        "start": start,
        "end": end or "latest",
        "data_clean_status": status,             # EARNED from the stale-print scan
        "max_stale_pnl_share": round(float(max_stale), 5),
        "stale_flagged_tickers": int(stale_flagged),
        "date_max": str(date_max) if date_max else (end or "latest"),
        "content_sha256_16": content_sha,
    }
    return Panel(dates, tuple(cols), open_, high, low, close, volume, active,
                 adv_usd, class_id, meta)
