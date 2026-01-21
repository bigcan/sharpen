"""Pro loader: build/fetch features and assemble env arrays (Option A).

This module wires FinRL Pro feature config into a concrete array assembly
compatible with the upstream StockTrading env (fixed 7 tech indicators).

It supports snapshot-backed datasets: given a `snapshot_id` and `features:`
config, it will (a) resolve family indicators, (b) compute optional advanced
features PIT-safely, (c) register/fetch features from the DB feature store,
and (d) assemble `price_ary`, `tech_ary` (7 features per ticker), and an
optional `turbulence_ary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd

from finrl_pro_ds.data.cache import feature_cache_key
from finrl_pro_ds.data.db import DatabaseClient, MarketBar
from finrl_pro_ds.features.families import resolve_indicator_list
from finrl_pro_ds.features.custom_features import (
    build_features,
    FracDiffConfig,
    WaveletConfig,
    RegimeConfig,
)


DEFAULT_TECH7 = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])  # type: ignore[assignment]
    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def _bars_to_df(bars: Sequence[MarketBar]) -> pd.DataFrame:
    rows = [{
        "date": pd.to_datetime(b.timestamp),
        "tic": b.ticker,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
        "source": b.source,
        "vendor_rev": b.vendor_rev,
    } for b in bars]
    return _ensure_dt(pd.DataFrame(rows))


def _add_stockstats(df: pd.DataFrame, indicators: Sequence[str]) -> pd.DataFrame:
    from stockstats import StockDataFrame as Sdf

    df = _ensure_dt(df)
    stock = Sdf.retype(df.copy())
    unique = stock.tic.unique()
    for ind in indicators:
        if ind in df.columns:
            continue
        ind_df = pd.DataFrame()
        for t in unique:
            try:
                tmp = stock[stock.tic == t][ind]
                tmp = pd.DataFrame(tmp)
                tmp["tic"] = t
                tmp["date"] = df[df.tic == t]["date"].to_list()
                ind_df = pd.concat([ind_df, tmp], axis=0, ignore_index=True)
            except Exception:
                continue
        df = df.merge(ind_df[["tic", "date", ind]], on=["tic", "date"], how="left")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def _calculate_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    # Based on upstream calculate_turbulence (PIT-safe)
    d = _ensure_dt(df)
    piv = d.pivot(index="date", columns="tic", values="close").pct_change()
    uniq = d["date"].drop_duplicates().sort_values().to_numpy()
    start = 252
    idx_vals: list[float] = [0.0] * min(start, len(uniq))
    count = 0
    for i in range(start, len(uniq)):
        cur = piv[piv.index == uniq[i]]
        hist = piv[(piv.index < uniq[i]) & (piv.index >= uniq[i - 252])]
        hist = hist.iloc[hist.isna().sum().min() :].dropna(axis=1)
        if hist.empty or cur.empty:
            idx_vals.append(0.0)
            continue
        cov = hist.cov()
        cur_tmp = cur[[x for x in hist]].values - np.mean(hist.values, axis=0)
        try:
            temp = cur_tmp.dot(np.linalg.pinv(cov)).dot(cur_tmp.T)
            val = float(temp[0][0]) if temp.size else 0.0
            if val > 0:
                count += 1
                idx_vals.append(val if count > 2 else 0.0)
            else:
                idx_vals.append(0.0)
        except Exception:
            idx_vals.append(0.0)
    return pd.DataFrame({"date": uniq, "turbulence": idx_vals})


def _select_tech7(all_columns: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    # Keep the first 7 from preferred_order that exist; if fewer, pad with remaining available
    chosen: List[str] = [c for c in preferred_order if c in all_columns][:7]
    if len(chosen) < 7:
        for c in all_columns:
            if c not in chosen:
                chosen.append(c)
                if len(chosen) == 7:
                    break
    if len(chosen) < 7:
        # Pad with zeros later if needed, but try to avoid
        pass
    return chosen[:7]


@dataclass
class Assembly:
    price_ary: np.ndarray
    tech_ary: np.ndarray
    turbulence_ary: np.ndarray
    tickers: List[str]
    dates: List[pd.Timestamp]
    feature_list: List[str]
    feature_set_id: str


class ProFeatureAssembler:
    def __init__(self, dsn: str | None = None) -> None:
        self._db = DatabaseClient(dsn=dsn)

    def _process_and_assemble_df(
        self,
        bars_df: pd.DataFrame,
        features_cfg: Mapping[str, object],
        preferred_order: Sequence[str],
        dataset_hash: str,
    ) -> Assembly:
        """Helper to process a DataFrame and assemble the final arrays."""
        
        # Check for Log Baseline Mode
        if bool(features_cfg.get("log_baseline", False)):
            from finrl_pro_ds.features.custom_features import add_log_features
            bars_df = add_log_features(bars_df)
            # Override indicators to use only log features
            ind_list = []
            preferred_order = ["log_close", "log_open", "log_high", "log_low", "log_volume", "log_sma_50", "log_sma_200"]
        elif bool(features_cfg.get("hybrid_baseline", False)):
            from finrl_pro_ds.features.custom_features import add_hybrid_features
            bars_df = add_hybrid_features(bars_df)
            ind_list = []
            preferred_order = ["macd", "rsi_14", "vwap_ratio", "atr_norm", "log_volume"]
        else:
            ind_list = resolve_indicator_list(
                families=dict(features_cfg.get("families", {})),
                overrides=list(features_cfg.get("stockstats_overrides", []) or [])
            ) or list(DEFAULT_TECH7)

        bars_df = _add_stockstats(bars_df, ind_list)

        # Advanced features
        adv = dict(features_cfg.get("advanced", {}) or {})
        fd_cfg = adv.get("fracdiff")
        wl_cfg = adv.get("wavelet")
        reg_dict = adv.get("regime") or features_cfg.get("regime")
        
        frac = None
        wav = None
        reg = None

        if fd_cfg and fd_cfg.get("enable"):
            frac = FracDiffConfig(
                cols=tuple(fd_cfg.get("cols", ["close"])),
                d=float(fd_cfg.get("d", 0.5)),
                window=int(fd_cfg.get("window", 256)),
                min_weight=float(fd_cfg.get("min_weight", 1e-5)),
            )
        if wl_cfg and wl_cfg.get("enable"):
            wav = WaveletConfig(
                cols=tuple(wl_cfg.get("cols", ["close"])),
                wavelet=str(wl_cfg.get("wavelet", "db4")),
                level=int(wl_cfg.get("level", 3)),
                window=int(wl_cfg.get("window", 256)),
            )
        if reg_dict:
             reg = RegimeConfig(
                 method=str(reg_dict.get("method", "hmm")),
                 benchmark_tic=str(reg_dict.get("benchmark_tic", "SPY")),
                 source=str(reg_dict.get("source", "close")),
                 window=int(reg_dict.get("window", 252)),
                 n_components=int(reg_dict.get("n_components", 3)),
             )

        if frac or wav or reg:
            bars_df = build_features(bars_df, fracdiff=frac, wavelet=wav, regime=reg)

        # Instead of storing in DB, directly assemble arrays
        tickers = sorted(bars_df["tic"].unique().tolist())
        dates = sorted(bars_df["date"].unique().tolist())

        # Select 7 indicators in a stable order
        base_cols = ["date", "tic", "open", "high", "low", "close", "volume", "source", "vendor_rev"]
        feature_cols = [c for c in bars_df.columns if c not in base_cols]
        # Re-add base cols if they are in preferred_order (e.g. for raw price features)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in preferred_order and c in bars_df.columns:
                feature_cols.append(c)

        tech7 = _select_tech7(feature_cols, preferred_order=preferred_order)
        
        # Ensure market_regime is included if present
        if "market_regime" in bars_df.columns:
            # We append it to the end of the feature list
            if "market_regime" not in tech7:
                tech7.append("market_regime")

        # Compute turbulence if requested
        turbulence = np.zeros(len(dates), dtype=float)
        if bool(features_cfg.get("use_turbulence", False)):
            tdf = _calculate_turbulence(bars_df)
            tdf = tdf.set_index("date").reindex(dates).fillna(0.0).reset_index()
            turbulence = tdf["turbulence"].to_numpy(dtype=float)

        # Build arrays
        price_piv = bars_df.pivot(index="date", columns="tic", values="close").reindex(dates).reindex(columns=tickers)
        price_ary = price_piv.to_numpy(dtype=float)

        tech_blocks: List[np.ndarray] = []
        for f in tech7:
            blk = bars_df.pivot(index="date", columns="tic", values=f).reindex(dates).reindex(columns=tickers).to_numpy(dtype=float)
            tech_blocks.append(blk)
        tech_stack = np.stack(tech_blocks, axis=2)  # (T, stock_dim, 7)
        tech_ary = tech_stack.reshape((tech_stack.shape[0], -1))  # (T, stock_dim*7)

        # Fill NaNs by forward-fill along time per column (like upstream fill)
        def ffill2d(a: np.ndarray) -> np.ndarray:
            out = a.copy()
            for j in range(out.shape[1]):
                col = out[:, j]
                mask = np.isnan(col)
                if mask.all():
                    continue
                idx = np.where(~mask, np.arange(len(col)), 0)
                np.maximum.accumulate(idx, out=idx)
                out[:, j] = col[idx]
            return out
            
        # Simple backfill for leading NaNs
        def bfill2d(a: np.ndarray) -> np.ndarray:
            out = a.copy()
            for j in range(out.shape[1]):
                col = out[:, j]
                mask = np.isnan(col)
                if mask.all():
                    continue
                # Find first valid index
                first_valid = np.where(~mask)[0][0]
                if first_valid > 0:
                    out[:first_valid, j] = out[first_valid, j]
            return out

        price_ary = ffill2d(price_ary)
        price_ary = bfill2d(price_ary) # Backfill leading NaNs
        
        tech_ary = ffill2d(tech_ary)
        tech_ary = bfill2d(tech_ary) # Backfill leading NaNs

        local_cache_key = feature_cache_key(dataset_hash, dict(features_cfg))

        return Assembly(
            price_ary=price_ary,
            tech_ary=tech_ary,
            turbulence_ary=turbulence,
            tickers=tickers,
            dates=dates,
            feature_list=tech7,
            feature_set_id=local_cache_key, # Use local_cache_key as feature_set_id
        )

    def assemble_from_df(
        self,
        *,
        df: pd.DataFrame,
        features_cfg: Mapping[str, object],
        dataset_hash: str,
        start: str | None = None,
        end: str | None = None,
    ) -> Assembly:
        """Build features and assemble arrays from a DataFrame."""
        fams = dict(features_cfg.get("families", {})) if features_cfg else {}
        overrides = list(features_cfg.get("stockstats_overrides", []) or [])
        preferred_order = overrides or DEFAULT_TECH7

        # Filter by start/end dates if provided
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]

        return self._process_and_assemble_df(
            bars_df=df,
            features_cfg=features_cfg,
            preferred_order=preferred_order,
            dataset_hash=dataset_hash,
        )

    def assemble_from_snapshot(
        self,
        *,
        snapshot_id: str,
        features_cfg: Mapping[str, object],
        start: str | None = None,
        end: str | None = None,
    ) -> Assembly:
        """Build or fetch features and assemble arrays (Option A: 7 tech)."""
        # Resolve indicator families
        fams = dict(features_cfg.get("families", {})) if features_cfg else {}
        overrides = list(features_cfg.get("stockstats_overrides", []) or [])
        ind_list = resolve_indicator_list(families=fams, overrides=overrides) or list(DEFAULT_TECH7)
        # Ensure 7 will be selectable later
        preferred_order = overrides or DEFAULT_TECH7

        # Compute cache_key against dataset hash if provided, else snapshot id
        dataset_hash = str(features_cfg.get("dataset_hash_source", snapshot_id))
        cache_key = feature_cache_key(dataset_hash, dict(features_cfg))

        self._db.init_schema()
        self._db.init_feature_store()

        fs_id = self._db.get_feature_set(snapshot_id=snapshot_id, cache_key=cache_key)
        if not fs_id:
            # Load raw bars
            bars = self._db.load_snapshot(snapshot_id)
            if not bars:
                raise RuntimeError(f"No bars for snapshot {snapshot_id}")
            bars_df = _bars_to_df(bars)

            # Process and assemble DF using the new helper
            assembly = self._process_and_assemble_df(
                bars_df=bars_df,
                features_cfg=features_cfg,
                preferred_order=preferred_order,
                dataset_hash=dataset_hash,
            )

            # Register feature set (wide → long rows)
            from uuid import uuid4
            fs_id = str(uuid4())
            self._db.insert_feature_set(
                feature_set_id=fs_id,
                snapshot_id=snapshot_id,
                cache_key=cache_key,
                config_json=pd.Series(dict(features_cfg)).to_json(),
                code_hash="unknown",
                lib_versions_json=pd.Series({}).to_json(),
            )
            # Prepare rows for upsert: convert to long
            base_cols = ["date", "tic", "open", "high", "low", "close", "volume", "source", "vendor_rev"]
            feat_cols = [c for c in bars_df.columns if c not in base_cols]
            def gen_rows():
                for _, r in bars_df.iterrows():
                    for name in feat_cols:
                        val = r[name]
                        if pd.isna(val):
                            continue
                        yield {
                            "timestamp": pd.to_datetime(r["date"]).to_pydatetime(),
                            "ticker": str(r["tic"]),
                            "feature_set_id": fs_id,
                            "name": str(name),
                            "value": float(val),
                        }
            self._db.upsert_feature_values(fs_id, gen_rows())
        
            # Update fs_id in the returned assembly
            assembly.feature_set_id = fs_id
            return assembly

        # Fetch features and assemble arrays
        # Determine tickers from snapshot assets and time window
        bars = self._db.load_snapshot(snapshot_id)
        bars_df = _bars_to_df(bars)
        if start:
            bars_df = bars_df[bars_df["date"] >= pd.to_datetime(start)]
        if end:
            bars_df = bars_df[bars_df["date"] <= pd.to_datetime(end)]

        tickers = sorted(bars_df["tic"].unique().tolist())
        dates = sorted(bars_df["date"].unique().tolist())

        feat_df = self._db.fetch_features(feature_set_id=fs_id, tickers=tickers, start=start, end=end)
        if feat_df.empty:
            raise RuntimeError("Feature fetch returned empty set")
        # Pivot to wide format: date,tic columns for each feature name
        feat_df["date"] = pd.to_datetime(feat_df["timestamp"])  # align names
        wide = feat_df.pivot_table(index=["date", "tic"], columns="name", values="value").reset_index()
        # Select 7 indicators in a stable order
        feature_cols = [c for c in wide.columns if c not in ["date", "tic"]]
        tech7 = _select_tech7(feature_cols, preferred_order=preferred_order)
        # Merge with bars for prices
        merged = bars_df.merge(wide[["date", "tic", *tech7]], on=["date", "tic"], how="left")
        merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)
        # Compute turbulence if requested
        turbulence = np.zeros(len(dates), dtype=float)
        if bool(features_cfg.get("use_turbulence", False)):
            tdf = _calculate_turbulence(bars_df)
            tdf = tdf.set_index("date").reindex(dates).fillna(0.0).reset_index()
            turbulence = tdf["turbulence"].to_numpy(dtype=float)

        # Build arrays
        price_piv = merged.pivot(index="date", columns="tic", values="close").reindex(dates).reindex(columns=tickers)
        price_ary = price_piv.to_numpy(dtype=float)

        tech_blocks: List[np.ndarray] = []
        for f in tech7:
            blk = merged.pivot(index="date", columns="tic", values=f).reindex(dates).reindex(columns=tickers).to_numpy(dtype=float)
            tech_blocks.append(blk)
        tech_stack = np.stack(tech_blocks, axis=2)  # (T, stock_dim, 7)
        tech_ary = tech_stack.reshape((tech_stack.shape[0], -1))  # (T, stock_dim*7)

        # Fill NaNs by forward-fill along time per column (like upstream fill)
        def ffill2d(a: np.ndarray) -> np.ndarray:
            out = a.copy()
            for j in range(out.shape[1]):
                col = out[:, j]
                mask = np.isnan(col)
                if mask.all():
                    continue
                idx = np.where(~mask, np.arange(len(col)), 0)
                np.maximum.accumulate(idx, out=idx)
                out[:, j] = col[idx]
            return out

        price_ary = ffill2d(price_ary)
        tech_ary = ffill2d(tech_ary)

        return Assembly(
            price_ary=price_ary,
            tech_ary=tech_ary,
            turbulence_ary=turbulence,
            tickers=tickers,
            dates=dates,
            feature_list=tech7,
            feature_set_id=fs_id,
        )
