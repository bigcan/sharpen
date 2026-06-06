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

from finrl_pro_ds.features import cross_asset_signals as cas

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
    """Run the canonical OHLCV outlier detect/repair per ticker (DATA-CLEAN).

    Lazy-imports ``scripts.clean_ohlcv`` (single source of truth — no duplicated
    detection logic that could silently drift from the project cleaner) so this
    library module imports cleanly without ``scripts/`` on the path.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean_ohlcv import detect_outliers, repair_outliers

    tickers = list(wide["close"].columns)
    report: dict[str, dict] = {}
    for tk in tickers:
        df = pd.DataFrame({c: wide[c][tk] for c in _OHLCV}).dropna(how="all")
        if df.empty or df[["open", "high", "low", "close"]].dropna(how="all").empty:
            report[tk] = {"rows": 0, "outliers_repaired": 0}
            continue
        det = detect_outliers(df, threshold=threshold)
        n_bad = int((det["bad_high"] | det["bad_low"]).sum())
        if n_bad:
            fixed = repair_outliers(df, det, threshold=threshold)
            for c in ("open", "high", "low", "close"):
                wide[c].loc[fixed.index, tk] = fixed[c].to_numpy()
        report[tk] = {"rows": int(len(df)), "outliers_repaired": n_bad}
    return wide, report


def _wide_to_long(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide per-field frames → tidy long ``[date, ticker, open..volume]``."""
    long = pd.concat(
        {name: wide[name].stack(future_stack=True) for name in _OHLCV}, axis=1,
    ).rename_axis(index=["date", "ticker"]).reset_index()
    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


def fetch_and_clean(
    assets: Sequence[str],
    start: str,
    end: str | None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    threshold: float = 0.05,
    force_refetch: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fetch full OHLCV, DATA-CLEAN it, cache raw + cleaned + manifest.

    Returns ``(wide, manifest)`` where ``wide`` is the cleaned dict of wide frames.
    Caches: ``ohlcv_daily_raw.parquet`` (the ``.bak``), ``ohlcv_daily.parquet``
    (cleaned), ``ohlcv_daily.manifest.json``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "ohlcv_daily_raw.parquet"
    clean_path = cache_dir / "ohlcv_daily.parquet"
    manifest_path = cache_dir / "ohlcv_daily.manifest.json"

    if clean_path.exists() and manifest_path.exists() and not force_refetch:
        manifest = json.loads(manifest_path.read_text())
        if set(manifest.get("assets", [])) >= set(assets):
            log.info("cross_asset_loader: using cached clean OHLCV (%s)", clean_path)
            long = pd.read_parquet(clean_path)
            long["date"] = pd.to_datetime(long["date"])
            wide = {c: long.pivot(index="date", columns="ticker", values=c)
                    .reindex(columns=list(assets)).sort_index() for c in _OHLCV}
            return wide, manifest

    log.info("cross_asset_loader: fetching %d tickers %s..%s", len(assets), start, end)
    wide = fetch_ohlcv_wide(assets, start, end)
    _wide_to_long(wide).to_parquet(raw_path, index=False)  # .bak (raw, pre-clean)

    wide, clean_report = _clean_wide(wide, threshold=threshold)
    long = _wide_to_long(wide)
    long.to_parquet(clean_path, index=False)

    idx = wide["close"].index
    total_outliers = int(sum(r["outliers_repaired"] for r in clean_report.values()))
    payload_hash = hashlib.sha256(
        pd.util.hash_pandas_object(long, index=True).values.tobytes(),
    ).hexdigest()[:16]
    manifest = {
        "stage": "data-prep",
        "status": "PASS",
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
        "per_ticker": clean_report,
        "raw_cache": str(raw_path),
        "clean_cache": str(clean_path),
        "content_sha256_16": payload_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(
        "cross_asset_loader: cleaned %d tickers × %d rows (%d outliers repaired) → %s",
        len(assets), len(idx), total_outliers, clean_path,
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
    volume_ary = volume_wide.loc[wdates, assets].ffill().fillna(0.0).to_numpy(np.float64)

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
def load_cross_asset_data(config: Mapping, *, force_refetch: bool = False) -> dict:
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
