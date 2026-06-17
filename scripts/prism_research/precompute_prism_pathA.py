"""Path-A (in-process, walk-forward SAFE) PRISM feature precompute — multi-asset daily.

Replaces the leaky REST precompute (`precompute_prism_data.py`) for the BTC / gold /
EURUSD daily PRISM re-eval. Uses the verified-clean in-process SAFFS provider
(`compute_prism_features` -> `PRISMFeatureProvider`), which fits GAHMM on
``iloc[:bar+1]`` and forecasts Chronos on ``context <= bar`` — i.e. **no full-window
GAHMM fit and no future-of-eval leak** (the Path-B defect found in S553-cont-48,
`market_data_service.refit` fitting once on ``[now-730d, now]``).

Critical runtime gotcha: the provider lives in ``C:\\FinRL\\SAFFS\\exports``, NOT
``C:\\FinRL\\PRISM`` (which is only the REST client). `prism_features.py` defaults
``PRISM_ROOT`` to the PRISM dir, so the import fails unless ``PRISM_ROOT`` points at
SAFFS. This script auto-points it at SAFFS when needed (before the import).

Walk-forward proof: `tests/prism_research/test_pathA_walk_forward.py` asserts features
at bar t are byte-identical whether the provider is handed the full frame or a frame
truncated at t (future bars cannot change past features).

Usage:
    # Fetch daily OHLCV via yfinance, clean, compute features for all 3 assets:
    python scripts/prism_research/precompute_prism_pathA.py --fetch \
        --start 2015-01-01 --end 2026-06-17

    # Use a pre-built clean long-format parquet instead of fetching:
    python scripts/prism_research/precompute_prism_pathA.py \
        --ohlcv-parquet data/prism_research/daily_ohlcv_long.parquet

    # Single asset / CPU Chronos:
    python scripts/prism_research/precompute_prism_pathA.py --fetch --assets btc \
        --chronos-device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.precompute_pathA")

# ---------------------------------------------------------------------------
# Asset spec: our label -> yfinance ticker. The label is the `ticker` value the
# provider keys on, and the output filename stem.
# ---------------------------------------------------------------------------
ASSET_TICKERS = {
    "btc": "BTC-USD",
    "gold": "GC=F",
    "eurusd": "EURUSD=X",
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research"

# Provider 13-col output -> legacy parquet schema consumed by regime_sanity_check.py
# / l2_backtest.py. We keep BOTH the raw provider columns and the legacy columns.
_PROB_RENAME = {
    "gahmm_price_bear": "price_bear_prob",
    "gahmm_price_neutral": "price_neutral_prob",
    "gahmm_price_bull": "price_bull_prob",
    "gahmm_vol_low": "vol_low_prob",
    "gahmm_vol_normal": "vol_normal_prob",
    "gahmm_vol_high": "vol_high_prob",
}


def ensure_prism_root() -> str:
    """Point PRISM_ROOT at the SAFFS dir holding the in-process provider.

    The provider package is ``<SAFFS>/exports/finrl_feature_provider.py``. The repo
    default (``C:\\FinRL\\PRISM``) is the REST client only and has no ``exports/`` —
    importing the provider from there fails. We only override when the current
    PRISM_ROOT cannot satisfy the import, and only if a SAFFS dir with exports/ exists.
    """
    candidates = []
    env_root = os.environ.get("PRISM_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    # Common SAFFS locations
    if sys.platform == "win32":
        candidates.append(Path(r"C:\FinRL\SAFFS"))
    candidates.append(Path.home() / "FinRL" / "SAFFS")

    for cand in candidates:
        if (cand / "exports" / "finrl_feature_provider.py").is_file():
            os.environ["PRISM_ROOT"] = str(cand)
            # The in-process provider computes GAHMM/Chronos locally and never touches
            # the Postgres regime store, but SAFFS `config/settings.py` raises at import
            # time if PRISM_USE_DB=true (its default) without a DB password. Disable the
            # DB path for this process so the import is self-contained.
            if os.environ.setdefault("PRISM_USE_DB", "false").lower() == "true":
                logger.warning("PRISM_USE_DB=true is set; the in-process provider does "
                               "not use the DB but SAFFS settings will require a password.")
            logger.info("PRISM_ROOT -> %s (in-process provider; PRISM_USE_DB=%s)",
                        cand, os.environ["PRISM_USE_DB"])
            return str(cand)

    raise FileNotFoundError(
        "Could not locate the SAFFS in-process provider "
        "(exports/finrl_feature_provider.py). Set PRISM_ROOT to your SAFFS root."
    )


def fetch_daily_ohlcv(label: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV via yfinance, returned as a tidy frame for one asset.

    Columns: [timestamp, ticker, open, high, low, close, volume]. `timestamp` is
    tz-naive (normalized to date). Raises on empty download.
    """
    import yfinance as yf

    logger.info("Fetching %s (%s) daily %s -> %s ...", label, ticker, start, end)
    raw = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    # yfinance may return a column MultiIndex (field, ticker) for single tickers too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df = df.reset_index().rename(columns={df.reset_index().columns[0]: "timestamp"})
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df.insert(1, "ticker", label)
    logger.info("  %s: %d daily bars (%s -> %s)",
                label, len(df), df["timestamp"].iloc[0].date(), df["timestamp"].iloc[-1].date())
    return df


def repair_ohlc_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Conservatively clamp impossible-OHLC bars: high>=max(O,C), low<=min(O,C).

    A SAME-BAR transform (causal, no look-ahead) that never touches open/close — the
    signal basis (Chronos log-returns, GAHMM price returns) is preserved exactly. It
    only repairs OHLC-consistency violations (e.g. yfinance BTC-USD daily glitches:
    high<close, low>open). high=max(O,H,C) >= max(O,C) >= min(O,C) >= min(O,L,C)=low,
    so no bar is left crossed. Returns (repaired_frame, n_bars_changed).
    """
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    hi_fix = pd.concat([o, h, c], axis=1).max(axis=1)
    lo_fix = pd.concat([o, low, c], axis=1).min(axis=1)
    changed = int(((hi_fix != h) | (lo_fix != low)).sum())
    out = df.copy()
    out["high"] = hi_fix
    out["low"] = lo_fix
    return out, changed


def clean_asset(df: pd.DataFrame, label: str, repair_ohlc: bool = False) -> pd.DataFrame:
    """Run the DATA-CLEAN validator (incl. stale / flat-OHLC scan). WARN-only.

    `validate_ohlcv` returns a bool; a False return flags suspect data (stale runs,
    flat-OHLC teleport spikes, OHLC violations). We log loudly but do not drop bars —
    a bug-free PRISM eval requires the operator to see data-quality flags, not have
    them silently swallowed. With ``repair_ohlc`` we first clamp OHLC-consistency
    violations (conservative, causal) so the validator reflects the repaired frame.
    """
    from scripts.clean_ohlcv import validate_ohlcv

    if repair_ohlc:
        df, n_fixed = repair_ohlc_consistency(df)
        if n_fixed:
            logger.warning("OHLC-repair: clamped %d impossible-OHLC bar(s) for %s "
                           "(high>=max(O,C), low<=min(O,C); open/close untouched).",
                           n_fixed, label)
        else:
            logger.info("OHLC-repair: %s had no OHLC-consistency violations.", label)

    ok = validate_ohlcv(df.drop(columns=["ticker"]), name=f"{label}-daily")
    if not ok:
        logger.warning("DATA-CLEAN flagged %s daily OHLCV — inspect before trusting "
                       "the eval (stale/flat-OHLC/OHLC-violation).", label)
    else:
        logger.info("DATA-CLEAN: %s daily OHLCV clean.", label)
    return df


def _legacy_schema(feats: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Map provider 13-col output -> legacy parquet schema + attach close.

    composite_code = round(gahmm_composite_code * 8)  (provider stores code/8 in [0,1]);
    price_regime = code // 3, vol_regime = code % 3  (CompositeRegime: price*3 + vol).
    confidence = mean of the two max regime probabilities (proxy; legacy field only).
    """
    df = feats.copy()
    # Probability column renames (keep originals too for provenance).
    for src, dst in _PROB_RENAME.items():
        df[dst] = df[src]

    code = (df["gahmm_composite_code"] * 8.0).round().clip(0, 8).astype(int)
    df["composite_code"] = code
    df["price_regime"] = (code // 3).astype(int)
    df["vol_regime"] = (code % 3).astype(int)

    price_max = df[["price_bear_prob", "price_neutral_prob", "price_bull_prob"]].max(axis=1)
    vol_max = df[["vol_low_prob", "vol_normal_prob", "vol_high_prob"]].max(axis=1)
    df["confidence"] = (price_max + vol_max) / 2.0

    # Attach close (reference price for downstream eval) keyed on timestamp.
    close_map = ohlcv.set_index("timestamp")["close"]
    df["close"] = df["timestamp"].map(close_map)
    return df


def chronos_is_live(df: pd.DataFrame) -> bool:
    """Heuristic: did Chronos actually produce forecasts, or is it in fallback?

    Fallback (transformers/chronos/weights/network unavailable) emits all-zero
    quantiles. If every chronos column is ~0 across all bars, forecasts are inert.
    """
    chronos_cols = ["chronos_p10", "chronos_p30", "chronos_p50",
                    "chronos_p70", "chronos_p90", "chronos_spread"]
    present = [c for c in chronos_cols if c in df.columns]
    if not present:
        return False
    return bool(np.any(np.abs(df[present].to_numpy()) > 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Path-A walk-forward-safe PRISM feature precompute (multi-asset daily)")
    parser.add_argument("--assets", nargs="+", default=list(ASSET_TICKERS),
                        choices=list(ASSET_TICKERS),
                        help="Which assets to compute (default: btc gold eurusd)")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch daily OHLCV via yfinance (else use --ohlcv-parquet)")
    parser.add_argument("--ohlcv-parquet", type=str, default=None,
                        help="Pre-built clean long-format OHLCV parquet "
                             "[timestamp,ticker,open,high,low,close,volume]")
    parser.add_argument("--start", default="2015-01-01", help="Fetch start (inclusive)")
    parser.add_argument("--end", default="2026-06-17", help="Fetch end (exclusive)")
    parser.add_argument("--chronos-device", default="cuda",
                        help="Chronos device (cuda/cpu); GAHMM is CPU regardless")
    parser.add_argument("--gahmm-refit-every", type=int, default=252,
                        help="Refit GAHMM every N bars (daily default ~1yr)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--save-ohlcv", type=str, default=None,
                        help="Optional path to dump the fetched clean long OHLCV parquet")
    parser.add_argument("--repair-ohlc", action="store_true",
                        help="Conservatively clamp impossible-OHLC bars (high>=max(O,C), "
                             "low<=min(O,C)); open/close untouched. Fixes yfinance glitches.")
    args = parser.parse_args()

    saffs_root = ensure_prism_root()
    # Import only AFTER PRISM_ROOT is set.
    from finrl_pro_ds.crypto.features.prism_features import (
        compute_prism_features,
        get_prism_feature_cols,
        unload_prism,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- assemble long-format clean daily OHLCV ----
    if args.ohlcv_parquet:
        long_df = pd.read_parquet(args.ohlcv_parquet)
        long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
        long_df = long_df[long_df["ticker"].isin(args.assets)].copy()
        logger.info("Loaded OHLCV from %s (%d rows, assets=%s)",
                    args.ohlcv_parquet, len(long_df), sorted(long_df["ticker"].unique()))
        cleaned = [clean_asset(long_df[long_df["ticker"] == label], label, args.repair_ohlc)
                   for label in args.assets]
        long_df = pd.concat(cleaned, ignore_index=True)
    elif args.fetch:
        frames = []
        for label in args.assets:
            raw = fetch_daily_ohlcv(label, ASSET_TICKERS[label], args.start, args.end)
            frames.append(clean_asset(raw, label, args.repair_ohlc))
        long_df = pd.concat(frames, ignore_index=True)
        if args.save_ohlcv:
            Path(args.save_ohlcv).parent.mkdir(parents=True, exist_ok=True)
            long_df.to_parquet(args.save_ohlcv, engine="pyarrow")
            logger.info("Saved clean long OHLCV -> %s", args.save_ohlcv)
    else:
        parser.error("Provide --fetch or --ohlcv-parquet")

    # ---- compute Path-A walk-forward-safe features ----
    prism_cfg = {
        "chronos_device": args.chronos_device,
        "gahmm_refit_every": args.gahmm_refit_every,
    }
    start_ts = long_df["timestamp"].min()
    end_ts = long_df["timestamp"].max()
    logger.info("Computing PRISM features (Path A, in-process) for %s over %s -> %s ...",
                args.assets, start_ts.date(), end_ts.date())
    feats = compute_prism_features(
        ohlcv_df=long_df,
        assets=args.assets,
        start_ts=start_ts,
        end_ts=end_ts,
        config=prism_cfg,
    )
    feature_cols = get_prism_feature_cols()
    logger.info("Provider returned %d rows, %d feature cols", len(feats), len(feature_cols))

    chronos_live = chronos_is_live(feats)
    if not chronos_live:
        logger.warning("Chronos appears to be in FALLBACK mode (all-zero forecasts). "
                       "GAHMM regime features are still valid; Chronos forecast features "
                       "are inert. Install the `chronos` package + weights for the full eval.")

    # ---- per-asset: map to legacy schema, save ----
    written = []
    for label in args.assets:
        a_feats = feats[feats["ticker"] == label].sort_values("timestamp")
        a_ohlcv = long_df[long_df["ticker"] == label]
        out = _legacy_schema(a_feats, a_ohlcv)
        out = out.set_index(pd.DatetimeIndex(out["timestamp"], name="date")).drop(columns=["timestamp"])
        out_path = output_dir / f"prism_features_{label}_daily.parquet"
        out.to_parquet(out_path, engine="pyarrow")
        written.append(out_path)
        comp_dist = out["composite_code"].value_counts().sort_index().to_dict()
        logger.info("Saved %s: %d rows %s -> %s | composite_code dist=%s | chronos_live=%s",
                    out_path.name, len(out), out.index[0].date(), out.index[-1].date(),
                    comp_dist, chronos_live)

    unload_prism()
    logger.info("Done. SAFFS=%s. Wrote: %s", saffs_root, [p.name for p in written])
    logger.info("Next: python scripts/prism_research/prism_predictive_eval.py "
                "--assets %s", " ".join(args.assets))


if __name__ == "__main__":
    main()
