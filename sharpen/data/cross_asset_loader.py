"""Cross-asset daily OHLCV loader → env arrays for the MultiAssetAllocatorEnv.

Phase 4 of the cross-sectional pivot (S553-cont-34). Closes the data gap left by
Phases 1-3: the cached ``results/xsec_momentum/prices_daily.parquet`` is
**close-only** (``raw["Close"]`` in the falsification), so the env's ``volume_ary``
(slippage) and a DATA-CLEAN'd OHLC could not be built from it. This module:

  1. **Re-fetches full OHLCV** (open/high/low/close/volume) via yfinance
     (``auto_adjust=True`` — identical adjustment to the validated falsification's
     close), caches the raw pull (the ``.bak`` equivalent) and a cleaned copy.
  2. **DATA-CLEAN** — runs the canonical ``scripts/clean_ohlcv`` detect/repair per
     ticker (single source of truth, lazy-imported so this library module stays
     import-clean) and writes a ``.manifest.json`` recording the clean step.
  3. **Builds the env arrays** via :mod:`sharpen.features.cross_asset_signals`:
     ``price_ary, tech_ary, vol_ary, carry_ary, volume_ary, timestamps`` plus a
     ``conviction_ary`` (raw ``trend_conviction``) so the RL-beats-linear gate can
     drive the env as the frozen linear core.

Causality / invariants:
  - **LEAK-2**: signals are causal-by-construction (see ``cross_asset_signals``);
    ``vol_ary`` is the raw causal realized vol the env vol-scales with (uses
    returns ``<= t-1``); the env applies weights T+1. A future-bar perturbation
    tripwire is in ``tests/data/test_cross_asset_loader.py``.
  - **LEAK-1**: the obs ``tech_ary`` is z-scored with **window-local** statistics
    only (the renorm runs on the sliced window). Only the unbounded ``vol`` column
    is normalized; the bounded signal columns (signs ∈ {-1,0,1}, conviction/rank
    ∈ [-1,1], and the linear-core ``baseline_weight`` which is a *position*) pass
    through untouched — z-scoring ``baseline_weight`` would destroy ADR-3 (the RL
    must be able to copy it to match the core).
  - ``vol_ary`` is NEVER normalized — it is the vol-scaling denominator, not an obs
    feature (the obs *does* carry a separate, normalized ``vol`` column).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from sharpen.data import treasury_curve_loader as tcl
from sharpen.features import cross_asset_signals as cas
from sharpen.features import defensive_signals as dfs
from sharpen.features import rates_carry as rc

log = logging.getLogger("cross_asset_loader")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "results" / "xsec_momentum"

# Bump when the manifest schema changes in a way a downstream gate depends on. v2 = the
# stale-print scan era: a cached manifest WITHOUT a `stale_scan` block (or below this version)
# predates the scan and is REFUSED on cache-hit so the scan is re-earned on active data
# (P1-03 — the committed run reused a pre-stale-scan cache).
LOADER_MANIFEST_VERSION = 2
# The rates curve must reach (within this tolerance) the ETF data's date_max, else the live
# book serves stale curve weights past the curve's end (P1-05; the audit saw a 6-day desync).
_CURVE_ETF_DESYNC_TOL_DAYS = 5

# yfinance field (level-0 column) -> our lowercase OHLCV name.
_YF_FIELDS = {"Open": "open", "High": "high", "Low": "low",
              "Close": "close", "Volume": "volume"}
_OHLCV = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Tech-column contract (what the RL sees per asset)
# --------------------------------------------------------------------------- #
def tech_cols_for(lookbacks: Sequence[int]) -> list[str]:
    """Ordered per-asset obs feature columns. Order IS the contract (the env's
    ``tech_ary`` layout and the obs unit tests depend on it). ``baseline_weight``
    is included per ADR-3 (the RL learns a residual over the linear core)."""
    return [f"sig_tsmom_{int(L)}" for L in lookbacks] + [
        "trend_conviction", "vol", "xs_rank", "baseline_weight",
    ]


# Only this column is unbounded → window-local z-scored in the obs. Every other
# tech column is bounded and passes through unnormalized (see module docstring).
_NORMALIZE_TECH = {"vol"}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_ohlcv_wide(
    assets: Sequence[str],
    start: str,
    end: str | None,
    *,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV via yfinance → dict of wide (date × ticker) frames.

    Keys: ``open, high, low, close, volume``. Columns are reindexed to ``assets``
    order; index is an ascending ``DatetimeIndex``. ``auto_adjust=True`` matches
    the validated falsification (split/dividend-adjusted close)."""
    import yfinance as yf

    assets = list(assets)
    raw = yf.download(
        assets, start=start, end=end, progress=False, auto_adjust=auto_adjust,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {assets} ({start}..{end})")

    # Multi-ticker → MultiIndex (field, ticker); single-ticker → flat columns.
    out: dict[str, pd.DataFrame] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    level0 = set(raw.columns.get_level_values(0)) if multi else set(raw.columns)
    for field, name in _YF_FIELDS.items():
        if field not in level0:
            raise RuntimeError(f"yfinance result missing field '{field}' (got {sorted(level0)})")
        wide = raw[field] if multi else raw[[field]].rename(columns={field: assets[0]})
        wide = wide.reindex(columns=assets).sort_index()
        out[name] = wide
    return out


# --------------------------------------------------------------------------- #
# Clean (DATA-CLEAN) + manifest
# --------------------------------------------------------------------------- #
def _clean_wide(
    wide: dict[str, pd.DataFrame], threshold: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run the canonical OHLCV outlier detect/repair AND the stale-print scan per ticker.

    Lazy-imports ``scripts.clean_ohlcv`` (single source of truth — no duplicated
    detection logic that could silently drift from the project cleaner) so this
    library module imports cleanly without ``scripts/`` on the path.

    Beyond outlier detect/repair, each (post-repair) ticker is scanned with
    ``detect_stale_runs`` — the exact flat-close / flat-OHLC detector for the
    gmgp1-gold stale-print class that voided "directional RL almost worked on gold"
    (DATA-CLEAN, P1-03). Per-ticker ``stale_*`` metrics + a ``stale_flagged`` flag are
    recorded so the manifest ``status`` can be EARNED from the scan rather than asserted
    (P1-05). On daily ETF bars the run threshold auto-scales (needs ~30 consecutive
    identical closes) so normal data is not false-flagged.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean_ohlcv import detect_outliers, detect_stale_runs, repair_outliers

    tickers = list(wide["close"].columns)
    report: dict[str, dict] = {}
    for tk in tickers:
        df = pd.DataFrame({c: wide[c][tk] for c in _OHLCV}).dropna(how="all")
        if df.empty or df[["open", "high", "low", "close"]].dropna(how="all").empty:
            report[tk] = {"rows": 0, "outliers_repaired": 0, "stale_suspect_frac": 0.0,
                          "stale_pnl_share": 0.0, "flat_ohlc_spikes": 0, "stale_flagged": False}
            continue
        det = detect_outliers(df, threshold=threshold)
        n_bad = int((det["bad_high"] | det["bad_low"]).sum())
        if n_bad:
            df = repair_outliers(df, det, threshold=threshold)
            for c in ("open", "high", "low", "close"):
                wide[c].loc[df.index, tk] = df[c].to_numpy()
        stale = detect_stale_runs(df)     # scan the POST-repair series (DatetimeIndex)
        report[tk] = {
            "rows": int(len(df)), "outliers_repaired": n_bad,
            "stale_suspect_frac": stale["stale_suspect_frac"],
            "stale_pnl_share": stale["stale_pnl_share"],
            "flat_ohlc_spikes": stale["flat_ohlc_spikes"],
            "stale_flagged": bool(stale["flagged"]),
        }
    return wide, report


def _wide_to_long(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide per-field frames → tidy long ``[date, ticker, open..volume]``."""
    long = pd.concat(
        {name: wide[name].stack(future_stack=True) for name in _OHLCV}, axis=1,
    ).rename_axis(index=["date", "ticker"]).reset_index()
    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


def _cache_covers_end(manifest: Mapping, end: str | None, *,
                      require_fresh: bool, tol_days: int) -> bool:
    """Does the cached manifest's ``date_max`` reach the requested window end (P1-02)?

    The old cache-hit branch keyed ONLY on the asset superset and ignored ``date_max``, so a
    scheduled live run silently reused a frozen cache and never advanced. Now:
      - a finite ``end`` is covered iff ``date_max >= end - tol_days`` (always enforced — a
        request for data beyond the cache must refetch; safe for backtests whose cache covers
        their end);
      - ``end is None`` (rolling-to-latest) is trusted by default, but when ``require_fresh``
        (the live scheduler) it is covered only if ``date_max`` is within ``tol_days`` of today.
    """
    dm = manifest.get("date_max")
    if not dm:
        return False
    date_max = pd.Timestamp(dm)
    if end is None:
        if not require_fresh:
            return True
        target = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        target = pd.Timestamp(end)
    return date_max >= (target - pd.Timedelta(days=int(tol_days)))


def _cache_covers_start(manifest: Mapping, start: str) -> bool:
    """Does the cached fetch window BEGIN at or before the requested ``start``?

    The other half of window coverage, and for a long time the missing half. The cache-hit branch
    tested the asset superset and :func:`_cache_covers_end` only, so a cache built from a NARROWER
    window was served for a WIDER request and the caller silently received a truncated panel — with
    ``end=None`` the freshness leg returns True unconditionally, so a request for more HISTORY hit
    a clean cache-hit path and logged "using cached clean OHLCV". Measured on the Taiwan loader
    (2026-09-02): a cache built at ``start=2018-01-01`` answered a ``start=2010-01-01`` request with
    **713 bars instead of 4085**, moving every downstream statistic with no warning anywhere. Same
    failure class as the flags-silently-change-the-subject rule in the standing record.

    Compares the manifest's recorded fetch-window ``start`` — the start REQUESTED by the run that
    built the cache — and deliberately NOT the observed ``date_min``. A name that listed after the
    window opens (00891 lists 2021 on a 2010 request) legitimately gives ``date_min > start``, so
    testing ``date_min`` would declare a correct cache stale and refetch on every single call.

    A manifest with no ``start`` key predates this contract: report NOT covered, which triggers one
    refetch that rewrites the manifest with the key. Self-healing, and it cannot mask a real gap.
    """
    cached = manifest.get("start")
    if not cached:
        return False
    return pd.Timestamp(cached) <= pd.Timestamp(start)


def _clip_window(wide: dict, start: str, end: str | None) -> dict:
    """Clip cached wide frames to the REQUESTED ``[start, end]`` window.

    The cache-hit path used to return whatever the cache held, while a fresh fetch returns exactly
    the requested window — so one call site got different data depending on cache state, which is
    the same class of silent subject-change as the start-coverage gap above and is what makes a
    "reproduce this window" request unreproducible. Measured before this fix: a request for
    2018-01-01..2020-12-31 against a 2010..2026 cache returned all 4085 bars, not the 713 asked for.

    Clipping here makes the two paths agree. ``end=None`` means rolling-to-latest, so only the
    lower bound is applied.
    """
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end is not None else None
    out = {}
    for name, frame in wide.items():
        idx = frame.index
        mask = idx >= lo if hi is None else (idx >= lo) & (idx <= hi)
        out[name] = frame.loc[mask]
    return out


def fetch_and_clean(
    assets: Sequence[str],
    start: str,
    end: str | None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    threshold: float = 0.05,
    force_refetch: bool = False,
    require_fresh: bool = False,
    freshness_tol_days: int = 5,
    stale_pnl_fail_threshold: float = 0.02,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fetch full OHLCV, DATA-CLEAN it, cache raw + cleaned + manifest.

    Returns ``(wide, manifest)`` where ``wide`` is the cleaned dict of wide frames.
    Caches: ``ohlcv_daily_raw.parquet`` (the ``.bak``), ``ohlcv_daily.parquet``
    (cleaned), ``ohlcv_daily.manifest.json``.

    The cache is reused only when it covers the requested asset superset AND BOTH ENDS of the
    requested window — ``end`` via :func:`_cache_covers_end` (P1-02), so a scheduled run asking
    for fresher data than the cache holds refetches instead of silently freezing, and ``start``
    via :func:`_cache_covers_start`, so a run asking for more HISTORY than the cache was built
    with refetches instead of silently receiving a truncated panel. The
    manifest ``status`` is EARNED from the stale-print scan (P1-05): ``FAIL`` if any ticker's
    stale-print P&L share crosses ``stale_pnl_fail_threshold`` (the gmgp1-gold class),
    ``WARN`` if any ticker is flagged below that, else ``PASS``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "ohlcv_daily_raw.parquet"
    clean_path = cache_dir / "ohlcv_daily.parquet"
    manifest_path = cache_dir / "ohlcv_daily.manifest.json"

    if clean_path.exists() and manifest_path.exists() and not force_refetch:
        manifest = json.loads(manifest_path.read_text())
        assets_ok = set(manifest.get("assets", [])) >= set(assets)
        fresh_ok = _cache_covers_end(manifest, end, require_fresh=require_fresh,
                                     tol_days=freshness_tol_days)
        start_ok = _cache_covers_start(manifest, start)
        # Refuse a cache that predates the stale-print scan (P1-03): no `stale_scan` block, or
        # an older manifest schema, means the scan never ran on this data — refetch so the
        # status is EARNED on active data instead of silently reusing un-scanned parquet.
        scan_ok = ("stale_scan" in manifest
                   and manifest.get("loader_manifest_version", 1) >= LOADER_MANIFEST_VERSION)
        if assets_ok and fresh_ok and scan_ok and start_ok:
            log.info("cross_asset_loader: using cached clean OHLCV (%s, window=%s..%s)",
                     clean_path, manifest.get("start"), manifest.get("date_max"))
            long = pd.read_parquet(clean_path)
            long["date"] = pd.to_datetime(long["date"])
            wide = {c: long.pivot(index="date", columns="ticker", values=c)
                    .reindex(columns=list(assets)).sort_index() for c in _OHLCV}
            return _clip_window(wide, start, end), manifest
        log.info("cross_asset_loader: cache stale (assets_ok=%s fresh_ok=%s scan_ok=%s "
                 "start_ok=%s, cached_window=%s..%s, requested=%s..%s) → refetching",
                 assets_ok, fresh_ok, scan_ok, start_ok, manifest.get("start"),
                 manifest.get("date_max"), start, end)

    log.info("cross_asset_loader: fetching %d tickers %s..%s", len(assets), start, end)
    wide = fetch_ohlcv_wide(assets, start, end)
    _wide_to_long(wide).to_parquet(raw_path, index=False)  # .bak (raw, pre-clean)

    wide, clean_report = _clean_wide(wide, threshold=threshold)
    long = _wide_to_long(wide)
    long.to_parquet(clean_path, index=False)

    idx = wide["close"].index
    total_outliers = int(sum(r["outliers_repaired"] for r in clean_report.values()))
    # EARN the manifest status from the stale-print scan (P1-05) — never a hardcoded literal.
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
        pd.util.hash_pandas_object(long, index=True).values.tobytes(),
    ).hexdigest()[:16]
    manifest = {
        "stage": "data-prep",
        "status": status,
        "loader_manifest_version": LOADER_MANIFEST_VERSION,
        "source": "yfinance_etf",
        "auto_adjust": True,
        "frequency": "1d",
        "assets": list(assets),
        "n_assets": len(assets),
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
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if status != "PASS":
        log.warning("cross_asset_loader: manifest status=%s (stale-flagged: %s, "
                    "max_stale_pnl_share=%.4f)", status, flagged_tickers, max_stale_pnl_share)
    log.info(
        "cross_asset_loader: cleaned %d tickers × %d rows (%d outliers repaired) → %s [%s]",
        len(assets), len(idx), total_outliers, clean_path, status,
    )
    return wide, manifest


# --------------------------------------------------------------------------- #
# Window-local z-score renorm (LEAK-1) — pure helper, copied from
# crypto_array_builder._renormalize_within_window to avoid cross-package coupling
# of the allocator to the crypto pipeline (the function is generic & has no
# crypto-specific behaviour). Kept byte-equivalent.
# --------------------------------------------------------------------------- #
def _renormalize_within_window(
    tech_ary: np.ndarray, norm_window: int, clip: float = 5.0,
    passthrough: set[int] | None = None,
) -> np.ndarray:
    """Window-local rolling z-score of feature columns (LEAK-1). ``passthrough``
    column indices are copied through unscaled."""
    T, n_cols = tech_ary.shape
    result = np.empty_like(tech_ary, dtype=np.float64)
    min_periods = max(24, norm_window // 10)
    for col_idx in range(n_cols):
        if passthrough and col_idx in passthrough:
            result[:, col_idx] = tech_ary[:, col_idx]
            continue
        col = pd.Series(tech_ary[:, col_idx], dtype=np.float64)
        roll_mean = col.rolling(window=norm_window, min_periods=min_periods).mean()
        roll_std = col.rolling(window=norm_window, min_periods=min_periods).std()
        z = (col - roll_mean) / (roll_std + 1e-6)
        result[:, col_idx] = z.fillna(0.0).clip(-clip, clip).to_numpy()
    return result.astype(np.float32)


# --------------------------------------------------------------------------- #
# Array builder
# --------------------------------------------------------------------------- #
def build_allocator_arrays(
    signals_long: pd.DataFrame,
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    assets: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    lookbacks: Sequence[int] = cas.DEFAULT_LOOKBACKS,
    norm_window: int = 252,
) -> dict:
    """Slice a [start_ts, end_ts] window and build the env arrays.

    Signals are computed ONCE on the full series (causal → early-window rows
    legitimately use pre-window *past* data) and sliced here; only the obs
    normalization is window-local (LEAK-1).

    Returns dict with keys: ``price_ary (T,N), tech_ary (T, N*tech_dim),
    vol_ary (T,N), carry_ary (T,N), volume_ary (T,N), timestamps (T,),
    conviction_ary (T,N), tech_cols, assets``.
    """
    assets = list(assets)
    n = len(assets)
    tcols = tech_cols_for(lookbacks)

    needed = set(tcols) | {"trend_conviction", "vol"}
    piv = {
        col: signals_long.pivot(index="date", columns="ticker", values=col)
        .reindex(columns=assets).sort_index()
        for col in needed
    }

    dates = close_wide.index
    mask = (dates >= start_ts) & (dates <= end_ts)
    wdates = dates[mask]
    if len(wdates) == 0:
        raise ValueError(f"empty window {start_ts}..{end_ts}")

    price_ary = close_wide.loc[wdates, assets].ffill().fillna(0.0).to_numpy(np.float64)
    # F1 (Fable 2026-06-11): the env charges slippage participation =
    # order_notional / volume_ary, so volume_ary MUST be DOLLAR volume (shares ×
    # price), not raw share count. Raw share volume made participation
    # dollars-per-share — overstating impact by ~price (≈90× for a $90 ETF) and
    # mis-ranking per-asset cost. dollar_volume[t,i] = share_volume[t,i] × close[t,i]
    # (close is the price basis used throughout this module; same wdates slice).
    share_volume = volume_wide.loc[wdates, assets].ffill().fillna(0.0).to_numpy(np.float64)
    volume_ary = share_volume * price_ary

    # Raw causal realized vol — the vol-scaling denominator. NEVER normalized.
    # Warmup NaN → 0; the env treats vol <= vol_floor as flat (weight 0), exactly
    # reproducing the falsification's warmup skip.
    vol_ary = piv["vol"].reindex(wdates).to_numpy(np.float64)
    np.nan_to_num(vol_ary, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Raw conviction for the frozen-linear-core baseline drive (RL-beats-linear gate).
    conviction_ary = piv["trend_conviction"].reindex(wdates).fillna(0.0).to_numpy(np.float64)

    carry_ary = np.zeros((len(wdates), n), dtype=np.float64)  # v1 TSMOM-only (ADR-6)

    # tech_ary: per-asset blocks in `tcols` order, concatenated.
    blocks = []
    for a in assets:
        block = pd.concat([piv[col][a].rename(col) for col in tcols], axis=1)
        block = block.reindex(wdates).ffill().fillna(0.0)
        blocks.append(block.to_numpy(np.float64))
    tech_ary = np.hstack(blocks)  # (T, N * tech_dim)

    n_tech = len(tcols)
    norm_local = {tcols.index(c) for c in _NORMALIZE_TECH if c in tcols}
    passthrough = {a * n_tech + j for a in range(n) for j in range(n_tech)
                   if j not in norm_local}
    tech_ary = _renormalize_within_window(tech_ary, norm_window, passthrough=passthrough)

    timestamps = (wdates.asi8 // 10**9).astype(np.int64)

    return {
        "price_ary": price_ary,
        "tech_ary": tech_ary,
        "vol_ary": vol_ary,
        "carry_ary": carry_ary,
        "volume_ary": volume_ary,
        "timestamps": timestamps,
        "conviction_ary": conviction_ary,
        "tech_cols": tcols,
        "assets": assets,
    }


# --------------------------------------------------------------------------- #
# Top-level loader (mirrors crypto_backtest_runner.prepare_data contract)
# --------------------------------------------------------------------------- #
def load_cross_asset_data(config: Mapping, *, force_refetch: bool = False,
                          require_fresh: bool = False) -> dict:
    """Fetch+clean OHLCV and compute causal signals for the whole series.

    Returns dict: ``signals (long DF), close (wide), volume (wide), assets,
    asset_class, lookbacks, manifest``. Per-window arrays are built lazily via
    :func:`build_allocator_arrays` (mirrors the crypto build_env_arrays pattern).
    """
    uni = config["universe"]
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})

    assets = list(uni["assets"])
    asset_class = dict(uni.get("asset_class", {a: "all" for a in assets}))
    lookbacks = list(feat_cfg.get("lookbacks", cas.DEFAULT_LOOKBACKS))
    skip = int(feat_cfg.get("skip", cas.DEFAULT_SKIP))
    vol_window = int(feat_cfg.get("vol_window", cas.DEFAULT_VOL_WINDOW))
    target_vol = float(config.get("env", {}).get("target_vol_asset", cas.DEFAULT_TARGET_VOL_ASSET))
    lev_cap = float(config.get("env", {}).get("lev_cap", cas.DEFAULT_LEV_CAP))

    wide, manifest = fetch_and_clean(
        assets,
        start=data_cfg.get("start_date", "2006-01-01"),
        end=data_cfg.get("end_date"),
        cache_dir=Path(data_cfg.get("cache_dir", DEFAULT_CACHE_DIR)),
        force_refetch=force_refetch,
        require_fresh=require_fresh,
    )

    close = wide["close"]
    # Assert causality of the signal stack once at load (cheap tripwire, LEAK-2).
    cas.assert_causal(
        close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
        target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=asset_class,
    )
    signals = cas.compute(
        close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
        target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=asset_class,
    )

    return {
        "signals": signals,
        "close": close,
        "volume": wide["volume"],
        "assets": assets,
        "asset_class": asset_class,
        "lookbacks": lookbacks,
        "manifest": manifest,
    }


# --------------------------------------------------------------------------- #
# Two-sleeve (momentum + rates-carry) loader / array builders
# --------------------------------------------------------------------------- #
# The paper executor shadows a FUND-OF-FUNDS: the validated momentum sleeve
# (cross_asset_signals TSMOM over the 18 ETFs) and the validated rates-carry sleeve
# (Treasury-curve carry+roll over {SHY,IEF,TLT,LQD}, net SR 0.467, corr-to-mom 0.014),
# each driven through the env independently and risk-parity-combined (allocator_factory).
# These builders fetch the UNION universe ONCE (so all three array sets share one
# calendar) and expose per-sleeve env arrays + a union price/volume set for the combined
# PaperState replay. The momentum sleeve arrays are byte-identical to the single-sleeve
# path (signals computed on the 18-asset subset only → XS-rank-within-class unchanged; SHY
# never enters momentum).


def _window_mask(dates: pd.DatetimeIndex, start_ts, end_ts) -> pd.DatetimeIndex:
    mask = (dates >= start_ts) & (dates <= end_ts)
    wdates = dates[mask]
    if len(wdates) == 0:
        raise ValueError(f"empty window {start_ts}..{end_ts}")
    return wdates


def _dollar_volume_and_price(
    close_wide: pd.DataFrame, volume_wide: pd.DataFrame,
    assets: Sequence[str], wdates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """``(volume_ary, price_ary)`` over ``assets`` on ``wdates``. ``volume_ary`` is DOLLAR
    volume (shares×price, F1) — the env/SimFillEngine slippage-participation denominator."""
    assets = list(assets)
    price_ary = close_wide.loc[wdates, assets].ffill().fillna(0.0).to_numpy(np.float64)
    share_volume = volume_wide.loc[wdates, assets].ffill().fillna(0.0).to_numpy(np.float64)
    return share_volume * price_ary, price_ary


def _causal_vol_ary(
    close_wide: pd.DataFrame, assets: Sequence[str], wdates: pd.DatetimeIndex,
    *, vol_window: int,
) -> np.ndarray:
    """Causal annualized realized vol over ``assets`` on ``wdates`` — the env's
    vol-scaling denominator. Uses the SAME formula as the momentum ``vol_ary``
    (``cross_asset_signals._realized_vol``: rolling std then ``.shift(1)``, so row ``t``
    uses returns ``<= t-1``) so the two sleeves vol-scale identically. Warmup NaN → 0
    (env treats vol ≤ vol_floor as flat)."""
    rets = close_wide[list(assets)].pct_change()
    vol = cas._realized_vol(rets, vol_window, cas.ANN)
    arr = vol.reindex(wdates)[list(assets)].to_numpy(np.float64)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_rates_carry_arrays(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    curve: Mapping[str, "pd.Series"],
    rates_assets: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    tenor_map: Mapping[str, str] = rc.DEFAULT_RATES_TENOR,
    financing_tenor: str = rc.DEFAULT_FINANCING_TENOR,
    tanh_scale: float = rc.DEFAULT_TANH_SCALE,
    vol_window: int = cas.DEFAULT_VOL_WINDOW,
) -> dict:
    """Env arrays for the RATES-CARRY sleeve drive (``linear_core_trajectory`` over the
    rates universe). ``conviction_ary`` is the causal daily carry+roll conviction
    (:func:`rates_carry.rates_carry_conviction`), NOT trend momentum; everything else
    mirrors :func:`build_allocator_arrays` (dollar volume F1, causal vol, carry_ary=0).

    ``tech_ary`` is a minimal placeholder (the conviction itself, ``tech_dim=1``): the
    frozen-linear-core drive ignores the observation (``action_at_step`` reads the
    conviction directly), so only its SHAPE must be valid for the env constructor.
    """
    rates_assets = list(rates_assets)
    n = len(rates_assets)
    wdates = _window_mask(close_wide.index, start_ts, end_ts)

    volume_ary, _price = _dollar_volume_and_price(close_wide, volume_wide, rates_assets, wdates)
    vol_ary = _causal_vol_ary(close_wide, rates_assets, wdates, vol_window=vol_window)
    conviction_ary = rc.daily_conviction_array(
        curve, wdates, rates_assets, tenor_map=tenor_map,
        financing_tenor=financing_tenor, tanh_scale=tanh_scale)
    carry_ary = np.zeros((len(wdates), n), dtype=np.float64)  # position-carry accrual = 0 (ADR-6)
    timestamps = (wdates.asi8 // 10**9).astype(np.int64)

    return {
        "price_ary": _price,
        "tech_ary": conviction_ary.astype(np.float32),   # placeholder (tech_dim=1); drive ignores obs
        "vol_ary": vol_ary,
        "carry_ary": carry_ary,
        "volume_ary": volume_ary,
        "timestamps": timestamps,
        "conviction_ary": conviction_ary,
        "tech_cols": ["rates_carry_conviction"],
        "assets": rates_assets,
    }


def build_defensive_arrays(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    defensive_assets: Sequence[str],
    asset_class: Mapping[str, str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    beta_window: int = dfs.DEFAULT_BETA_WINDOW,
    min_periods: int = dfs.DEFAULT_BETA_MIN_PERIODS,
    vol_window: int = cas.DEFAULT_VOL_WINDOW,
) -> dict:
    """Env arrays for the DEFENSIVE / betting-against-beta (BAB) sleeve drive
    (``linear_core_trajectory`` over the defensive universe). ``conviction_ary`` is the
    causal within-class long-low-beta / short-high-beta conviction
    (:func:`defensive_signals.defensive_conviction`), NOT trend momentum; everything else
    mirrors :func:`build_rates_carry_arrays` (dollar volume F1, causal vol, carry_ary=0).

    The conviction is computed on the FULL series (causal → early-window rows legitimately
    use pre-window past data for the trailing beta) then sliced to the window; warmup NaN →
    0 (the env treats a flat/NaN conviction as no position). The market proxy is the
    cross-sectional equal-weight of ``defensive_assets`` (the sleeve's OWN universe, matching
    ``signals.generation.base_sleeves.defensive_sleeve_returns``). ``tech_ary`` is a minimal
    placeholder (the conviction itself, ``tech_dim=1``): the frozen-linear-core drive reads
    the conviction directly, so only its SHAPE must be valid for the env constructor.
    """
    defensive_assets = list(defensive_assets)
    n = len(defensive_assets)
    wdates = _window_mask(close_wide.index, start_ts, end_ts)

    volume_ary, _price = _dollar_volume_and_price(close_wide, volume_wide, defensive_assets, wdates)
    vol_ary = _causal_vol_ary(close_wide, defensive_assets, wdates, vol_window=vol_window)
    sub_class = {a: asset_class.get(a, "all") for a in defensive_assets}
    conv_full = dfs.defensive_conviction(
        close_wide[defensive_assets], sub_class,
        beta_window=beta_window, min_periods=min_periods)
    conviction_ary = (conv_full.reindex(wdates)[defensive_assets]
                      .fillna(0.0).to_numpy(np.float64))
    carry_ary = np.zeros((len(wdates), n), dtype=np.float64)  # position-carry accrual = 0 (ADR-6)
    timestamps = (wdates.asi8 // 10**9).astype(np.int64)

    return {
        "price_ary": _price,
        "tech_ary": conviction_ary.astype(np.float32),   # placeholder (tech_dim=1); drive ignores obs
        "vol_ary": vol_ary,
        "carry_ary": carry_ary,
        "volume_ary": volume_ary,
        "timestamps": timestamps,
        "conviction_ary": conviction_ary,
        "tech_cols": ["defensive_conviction"],
        "assets": defensive_assets,
    }


def build_union_arrays(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    union_assets: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    """Price / dollar-volume / carry / timestamps over the UNION universe (momentum 18 +
    SHY → 19) for the combined PaperState replay (the risk-parity-combined target weights
    are booked here). No signals — accounting only; the combined weights come from the two
    sleeve drives + the risk-parity combine (allocator_factory)."""
    union_assets = list(union_assets)
    wdates = _window_mask(close_wide.index, start_ts, end_ts)
    volume_ary, price_ary = _dollar_volume_and_price(close_wide, volume_wide, union_assets, wdates)
    return {
        "price_ary": price_ary,
        "volume_ary": volume_ary,
        "carry_ary": np.zeros((len(wdates), len(union_assets)), dtype=np.float64),
        "timestamps": (wdates.asi8 // 10**9).astype(np.int64),
        "assets": union_assets,
    }


# Sleeve name → default signal kind when the sleeve spec omits an explicit ``signal:`` key
# (back-compat: the pre-tailwind {momentum, rates_carry} configs carry no ``signal`` on the
# rates sleeve). ``tsmom``/``rates_carry``/``defensive`` are the three wired allocator drives.
_SIGNAL_BY_NAME = {"momentum": "tsmom", "rates_carry": "rates_carry", "defensive": "defensive"}
_KNOWN_SIGNALS = frozenset({"tsmom", "rates_carry", "defensive"})


def _sleeve_specs(config: Mapping) -> list[tuple[str, str, dict]]:
    """Ordered ``(name, signal, spec)`` for the ALLOCATOR-family sleeves (weights over the
    ETF union), signal-dispatched. ``return_stream`` sleeves (VRP) are excluded — the
    portfolio executor combines those at the return level. The signal is ``spec['signal']``
    if present, else inferred from the sleeve NAME (``_SIGNAL_BY_NAME``); ``momentum``/``mom``
    normalize to ``tsmom``. Back-compat: a classic ``{momentum, rates_carry}`` config (or an
    empty ``sleeves`` block) resolves to exactly ``[(momentum, tsmom), (rates_carry,
    rates_carry)]`` — byte-identical to the pre-tailwind two-sleeve path."""
    sleeves = dict(config.get("sleeves", {}))
    if not sleeves:
        return [("momentum", "tsmom", {}), ("rates_carry", "rates_carry", {})]
    specs: list[tuple[str, str, dict]] = []
    for name, raw in sleeves.items():
        spec = dict(raw or {})
        if str(spec.get("type", "allocator")) == "return_stream":
            continue
        signal = str(spec.get("signal") or _SIGNAL_BY_NAME.get(name, "")).lower()
        if signal in ("momentum", "mom"):
            signal = "tsmom"
        if signal not in _KNOWN_SIGNALS:
            raise ValueError(
                f"sleeve {name!r}: unresolved signal {signal!r} — set sleeves.{name}.signal "
                f"to one of {sorted(_KNOWN_SIGNALS)} (or name it momentum/rates_carry/defensive)")
        specs.append((name, signal, spec))
    if not specs:
        raise KeyError("config.sleeves has no allocator sleeves (all return_stream?)")
    return specs


def load_two_sleeve_data(config: Mapping, *, force_refetch: bool = False,
                         require_fresh: bool = False) -> dict:
    """Fetch+clean the UNION OHLCV once and prepare EVERY allocator sleeve's inputs.

    Signal-dispatched over :func:`_sleeve_specs` (``tsmom`` / ``rates_carry`` / ``defensive``),
    so it wires the classic momentum+rates-carry book AND the TAILWIND momentum+defensive(BAB)
    book (and a momentum-only challenge book) from one loader. The rates curve is loaded ONLY
    if a ``rates_carry`` sleeve is present; each sleeve's signal stack gets a LEAK-2 causality
    tripwire at load (``cas`` / ``rc`` / ``dfs`` ``assert_causal``).

    Returns a payload consumed by :func:`build_two_sleeve_arrays`. Momentum signals are
    computed on each tsmom sleeve's OWN asset subset (so XS-rank-within-class and the validated
    baseline are byte-identical to the single-sleeve path). Back-compat aliases
    (``mom_signals``/``mom_assets``/``rates_assets``/``curve``/``curve_manifest``) are populated
    for the classic two-sleeve config; external callers read only ``close``/``union_assets``/
    ``manifest``/``curve_manifest``.
    """
    specs = _sleeve_specs(config)
    signals_present = {sig for _, sig, _ in specs}
    uni = config["universe"]
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})

    union_assets = list(uni["assets"])
    asset_class = dict(uni.get("asset_class", {a: "all" for a in union_assets}))

    lookbacks = list(feat_cfg.get("lookbacks", cas.DEFAULT_LOOKBACKS))
    skip = int(feat_cfg.get("skip", cas.DEFAULT_SKIP))
    vol_window = int(feat_cfg.get("vol_window", cas.DEFAULT_VOL_WINDOW))
    target_vol = float(config.get("env", {}).get("target_vol_asset", cas.DEFAULT_TARGET_VOL_ASSET))
    lev_cap = float(config.get("env", {}).get("lev_cap", cas.DEFAULT_LEV_CAP))

    for name, sig, spec in specs:
        s_assets = list(spec.get("assets", union_assets if sig != "rates_carry"
                                  else rc.rates_universe()))
        missing = [a for a in s_assets if a not in union_assets]
        if missing:
            raise ValueError(
                f"sleeve {name!r} assets {missing} not in universe.assets (union must cover every sleeve)")

    wide, manifest = fetch_and_clean(
        union_assets,
        start=data_cfg.get("start_date", "2006-01-01"),
        end=data_cfg.get("end_date"),
        cache_dir=Path(data_cfg.get("cache_dir", DEFAULT_CACHE_DIR)),
        force_refetch=force_refetch,
        require_fresh=require_fresh,
    )
    close = wide["close"]

    payload: dict = {
        "close": close,
        "volume": wide["volume"],
        "union_assets": union_assets,
        "asset_class": asset_class,
        "lookbacks": lookbacks,
        "vol_window": vol_window,
        "manifest": manifest,
        "sleeve_specs": specs,
        "sleeve_signals": {},
        "curve_manifest": None,
    }

    # --- tsmom sleeves: compute the validated TSMOM signal stack on each sleeve's subset ---
    for name, sig, spec in specs:
        if sig != "tsmom":
            continue
        s_assets = list(spec["assets"])
        s_close = close[s_assets]
        s_class = {a: asset_class.get(a, "all") for a in s_assets}
        cas.assert_causal(
            s_close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
            target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=s_class,
        )
        signals = cas.compute(
            s_close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
            target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=s_class,
        )
        payload["sleeve_signals"][name] = signals
        payload.setdefault("mom_signals", signals)     # back-compat alias (first tsmom sleeve)
        payload.setdefault("mom_assets", s_assets)

    # --- rates_carry sleeves: load the Treasury curve ONCE, assert causal per tenor map ---
    if "rates_carry" in signals_present:
        curve, curve_manifest = tcl.load_treasury_curve_with_manifest(
            cache_dir=data_cfg.get("cache_dir"),
            force_refetch=force_refetch, require_fresh=require_fresh)
        for name, sig, spec in specs:
            if sig != "rates_carry":
                continue
            rc.assert_causal(
                curve, close.index,
                tenor_map=dict(spec.get("tenor_map", rc.DEFAULT_RATES_TENOR)),
                financing_tenor=str(spec.get("financing_tenor", rc.DEFAULT_FINANCING_TENOR)),
                tanh_scale=float(spec.get("tanh_scale", rc.DEFAULT_TANH_SCALE)))
            payload.setdefault("rates_assets",
                               list(spec.get("assets", rc.rates_universe())))

        # P1-05: the curve must reach (within tol) the ETF data's date_max, else the live book
        # serves stale curve weights past the curve's end. Surface as a WARN on the manifest.
        etf_date_max = close.index.max()
        curve_dm = pd.Timestamp(curve_manifest["date_max"]) if curve_manifest.get("date_max") else None
        if curve_dm is not None:
            desync_days = int((etf_date_max - curve_dm).days)
            if desync_days > _CURVE_ETF_DESYNC_TOL_DAYS:
                curve_manifest["calendar_desync_days"] = desync_days
                curve_manifest["etf_date_max"] = str(etf_date_max.date())
                if curve_manifest["status"] == "PASS":
                    curve_manifest["status"] = "WARN"
                log.warning("cross_asset_loader: curve date_max %s is %d days behind ETF date_max "
                            "%s (stale curve weights past the curve end)",
                            curve_manifest.get("date_max"), desync_days, etf_date_max.date())
                # The desync is cross-dataset (only known here), so the curve loader's sidecar was
                # written without it — keep it consistent so a sidecar reader can't see a stale PASS.
                cdir = data_cfg.get("cache_dir")
                if cdir:
                    (Path(cdir) / "treasury_curve.manifest.json").write_text(
                        json.dumps(curve_manifest, indent=2))
        payload["curve"] = curve
        payload["curve_manifest"] = curve_manifest

    # --- defensive (BAB) sleeves: LEAK-2 tripwire at load; conviction is built lazily in
    #     build_defensive_arrays (a pure function of the sleeve's close, needs no curve). ---
    for name, sig, spec in specs:
        if sig != "defensive":
            continue
        s_assets = list(spec.get("assets", union_assets))
        s_class = {a: asset_class.get(a, "all") for a in s_assets}
        dfs.assert_causal(
            close[s_assets], s_class,
            beta_window=int(spec.get("beta_window", dfs.DEFAULT_BETA_WINDOW)),
            min_periods=int(spec.get("min_periods", dfs.DEFAULT_BETA_MIN_PERIODS)))

    return payload


def build_two_sleeve_arrays(data: Mapping, start_ts, end_ts) -> dict:
    """Build the per-sleeve env arrays + the union accounting arrays for ``[start_ts,
    end_ts]`` from a :func:`load_two_sleeve_data` payload, signal-dispatched over
    ``data['sleeve_specs']``. All arrays share one calendar (single union fetch), so step
    index ``k`` aligns across sleeves and the union book.

    Returns ``{<sleeve_name>: arrays, ..., "union": union_arrays}`` — for a classic
    ``{momentum, rates_carry}`` config this is byte-identical to the pre-tailwind
    ``{"momentum", "rates_carry", "union"}``; for TAILWIND it is
    ``{"momentum", "defensive", "union"}``.
    """
    close, volume, asset_class = data["close"], data["volume"], data["asset_class"]
    out: dict = {}
    for name, sig, spec in data["sleeve_specs"]:
        if sig == "tsmom":
            out[name] = build_allocator_arrays(
                data["sleeve_signals"][name], close, volume, list(spec["assets"]),
                start_ts, end_ts, lookbacks=data["lookbacks"],
            )
        elif sig == "rates_carry":
            out[name] = build_rates_carry_arrays(
                close, volume, data["curve"], list(spec.get("assets", rc.rates_universe())),
                start_ts, end_ts,
                tenor_map=dict(spec.get("tenor_map", rc.DEFAULT_RATES_TENOR)),
                financing_tenor=str(spec.get("financing_tenor", rc.DEFAULT_FINANCING_TENOR)),
                tanh_scale=float(spec.get("tanh_scale", rc.DEFAULT_TANH_SCALE)),
                vol_window=data["vol_window"],
            )
        elif sig == "defensive":
            out[name] = build_defensive_arrays(
                close, volume, list(spec.get("assets", data["union_assets"])), asset_class,
                start_ts, end_ts,
                beta_window=int(spec.get("beta_window", dfs.DEFAULT_BETA_WINDOW)),
                min_periods=int(spec.get("min_periods", dfs.DEFAULT_BETA_MIN_PERIODS)),
                vol_window=data["vol_window"],
            )
    out["union"] = build_union_arrays(
        close, volume, data["union_assets"], start_ts, end_ts)
    return out
