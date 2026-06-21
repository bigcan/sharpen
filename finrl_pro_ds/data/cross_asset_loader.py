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
  3. **Builds the env arrays** via :mod:`finrl_pro_ds.features.cross_asset_signals`:
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

from finrl_pro_ds.data import treasury_curve_loader as tcl
from finrl_pro_ds.features import cross_asset_signals as cas
from finrl_pro_ds.features import rates_carry as rc

log = logging.getLogger("cross_asset_loader")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "results" / "xsec_momentum"

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

    The cache is reused only when it covers BOTH the requested asset superset AND the
    requested window ``end`` (:func:`_cache_covers_end`, P1-02) — a scheduled run that asks
    for fresher data than the cache holds refetches instead of silently freezing. The
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
        if assets_ok and fresh_ok:
            log.info("cross_asset_loader: using cached clean OHLCV (%s, date_max=%s)",
                     clean_path, manifest.get("date_max"))
            long = pd.read_parquet(clean_path)
            long["date"] = pd.to_datetime(long["date"])
            wide = {c: long.pivot(index="date", columns="ticker", values=c)
                    .reindex(columns=list(assets)).sort_index() for c in _OHLCV}
            return wide, manifest
        log.info("cross_asset_loader: cache stale (assets_ok=%s fresh_ok=%s, date_max=%s, "
                 "requested_end=%s) → refetching", assets_ok, fresh_ok,
                 manifest.get("date_max"), end)

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


def _sleeve_cfg(config: Mapping) -> tuple[dict, dict]:
    sleeves = dict(config.get("sleeves", {}))
    if "momentum" not in sleeves or "rates_carry" not in sleeves:
        raise KeyError("two-sleeve config needs sleeves.{momentum,rates_carry}")
    return dict(sleeves["momentum"]), dict(sleeves["rates_carry"])


def load_two_sleeve_data(config: Mapping, *, force_refetch: bool = False,
                         require_fresh: bool = False) -> dict:
    """Fetch+clean the UNION OHLCV once and prepare BOTH sleeves' inputs.

    Returns dict: ``close (wide), volume (wide), mom_signals (long), mom_assets,
    rates_assets, union_assets, asset_class, curve, lookbacks, sleeve cfgs, manifest``.
    Per-window env arrays are built lazily via :func:`build_two_sleeve_arrays`.

    Momentum signals are computed on the 18-asset SUBSET (so XS-rank-within-class and the
    validated baseline are byte-identical to the single-sleeve path; SHY never enters the
    momentum cross-section). The rates curve is loaded once and a LEAK-2 causality tripwire
    is asserted on it at load (mirrors ``cas.assert_causal``).
    """
    mom_cfg, rc_cfg = _sleeve_cfg(config)
    uni = config["universe"]
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})

    union_assets = list(uni["assets"])
    asset_class = dict(uni.get("asset_class", {a: "all" for a in union_assets}))
    mom_assets = list(mom_cfg["assets"])
    rates_assets = list(rc_cfg.get("assets", rc.rates_universe()))
    tenor_map = dict(rc_cfg.get("tenor_map", rc.DEFAULT_RATES_TENOR))
    financing_tenor = str(rc_cfg.get("financing_tenor", rc.DEFAULT_FINANCING_TENOR))
    tanh_scale = float(rc_cfg.get("tanh_scale", rc.DEFAULT_TANH_SCALE))

    lookbacks = list(feat_cfg.get("lookbacks", cas.DEFAULT_LOOKBACKS))
    skip = int(feat_cfg.get("skip", cas.DEFAULT_SKIP))
    vol_window = int(feat_cfg.get("vol_window", cas.DEFAULT_VOL_WINDOW))
    target_vol = float(config.get("env", {}).get("target_vol_asset", cas.DEFAULT_TARGET_VOL_ASSET))
    lev_cap = float(config.get("env", {}).get("lev_cap", cas.DEFAULT_LEV_CAP))

    missing = [a for a in mom_assets + rates_assets if a not in union_assets]
    if missing:
        raise ValueError(f"sleeve assets {missing} not in universe.assets (union must cover both sleeves)")

    wide, manifest = fetch_and_clean(
        union_assets,
        start=data_cfg.get("start_date", "2006-01-01"),
        end=data_cfg.get("end_date"),
        cache_dir=Path(data_cfg.get("cache_dir", DEFAULT_CACHE_DIR)),
        force_refetch=force_refetch,
        require_fresh=require_fresh,
    )
    close = wide["close"]

    mom_close = close[mom_assets]
    mom_asset_class = {a: asset_class.get(a, "all") for a in mom_assets}
    cas.assert_causal(
        mom_close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
        target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=mom_asset_class,
    )
    mom_signals = cas.compute(
        mom_close, lookbacks=lookbacks, skip=skip, vol_window=vol_window,
        target_vol_asset=target_vol, lev_cap=lev_cap, asset_class=mom_asset_class,
    )

    curve = tcl.load_treasury_curve(cache_dir=data_cfg.get("cache_dir"))
    rc.assert_causal(curve, close.index, tenor_map=tenor_map,
                     financing_tenor=financing_tenor, tanh_scale=tanh_scale)

    return {
        "close": close,
        "volume": wide["volume"],
        "mom_signals": mom_signals,
        "mom_assets": mom_assets,
        "rates_assets": rates_assets,
        "union_assets": union_assets,
        "asset_class": asset_class,
        "curve": curve,
        "lookbacks": lookbacks,
        "vol_window": vol_window,
        "tenor_map": tenor_map,
        "financing_tenor": financing_tenor,
        "tanh_scale": tanh_scale,
        "manifest": manifest,
    }


def build_two_sleeve_arrays(data: Mapping, start_ts, end_ts) -> dict:
    """Build the per-sleeve env arrays + the union accounting arrays for ``[start_ts,
    end_ts]`` from a :func:`load_two_sleeve_data` payload. All three share one calendar
    (single union fetch), so step index ``k`` aligns across sleeves and the union book.

    Returns ``{"momentum": mom_arrays, "rates_carry": rates_arrays, "union": union_arrays}``.
    """
    mom_arrays = build_allocator_arrays(
        data["mom_signals"], data["close"], data["volume"], data["mom_assets"],
        start_ts, end_ts, lookbacks=data["lookbacks"],
    )
    rates_arrays = build_rates_carry_arrays(
        data["close"], data["volume"], data["curve"], data["rates_assets"],
        start_ts, end_ts, tenor_map=data["tenor_map"],
        financing_tenor=data["financing_tenor"], tanh_scale=data["tanh_scale"],
        vol_window=data["vol_window"],
    )
    union_arrays = build_union_arrays(
        data["close"], data["volume"], data["union_assets"], start_ts, end_ts)
    return {"momentum": mom_arrays, "rates_carry": rates_arrays, "union": union_arrays}
