"""CMGP1 X2-leak falsification probe (free, no-GPU first read).

Quantifies how much FORWARD-RETURN information the X2 coarse-bar look-ahead
injected into CMGP1's multiscale coarse features, and whether that information
collapses once the de-leak (multiscale_crypto_handler.py:255, the scale_ns
subtraction) is applied.

This is the project's "cheap linear falsification BEFORE any RL build" gate (the
same doctrine that gated the cross-sectional lever GO and the R0-regime NO-GO).
It is NECESSARY-not-sufficient: a coarse-feature IC that collapses leaky->causal
confirms the coarse signal was substantially leak-borne (the sg1/gmgp1-btc
fingerprint). It does NOT by itself prove the causal base-scale signal has zero
edge — that remains the RL canary's job.

Method: instantiate the REAL MultiScaleCryptoHandler on the cached crypto OHLCV.
For each base bar t and asset a, take the coarse-scale feature mapped to t by
(i) the FIXED causal map and (ii) a reconstructed LEAKY map, and measure the
rank-IC (Spearman) of that feature against the forward 1h base return
r_{t->t+1}. log_return (feat 0) is the most direct: the in-progress coarse bar's
log_return literally contains the current+future move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finrl_pro_ds.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler

CACHE = "data/crypto_cache/silver_ohlcv.parquet"
SCALES = [1, 4, 24]
FEATS = {0: "log_return", 6: "close_z"}
# Config OOS test window + full sample.
WINDOWS = {
    "test_2025Q4": ("2025-10-01", "2025-12-31"),
    "full_sample": (None, None),
}


def _ic(feat: np.ndarray, fwd: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(feat) & np.isfinite(fwd)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    rho, _ = spearmanr(feat[m], fwd[m])
    return float(rho), int(m.sum())


def main() -> None:
    df = pd.read_parquet(CACHE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    assets = sorted(df["ticker"].unique().tolist())
    print(f"universe ({len(assets)}): {assets}")
    print(f"range: {df['timestamp'].min()} -> {df['timestamp'].max()}  rows={len(df)}\n")

    feature_config = {
        "scales": SCALES,
        "window_size": 30,
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "norm_span": 120,
        "feature_set_version": "v1",
    }
    h = MultiScaleCryptoHandler(
        ohlcv_df=df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]],
        funding_df=None,
        assets=assets,
        feature_config=feature_config,
    )

    base_ts = np.asarray(h._base_timestamps).astype("datetime64[ns]")
    base_ns = base_ts.astype("int64")
    base_close = h._base_close  # (T, N)
    active = h._base_active     # (T, N)
    T, N = base_close.shape

    # Forward 1h base return per (t, a); valid only where t and t+1 both active.
    fwd = np.full((T, N), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = base_close[1:] / np.where(base_close[:-1] > 0, base_close[:-1], np.nan)
    fwd[:-1] = ratio - 1.0
    both_active = active[:-1] & active[1:]
    fwd[:-1][~both_active] = np.nan

    base_scale = min(SCALES)
    for wname, (s, e) in WINDOWS.items():
        if s is None:
            wmask = np.ones(T, dtype=bool)
        else:
            wmask = (base_ts >= np.datetime64(s)) & (base_ts <= np.datetime64(e))
        wmask[-1] = False  # last bar has no forward return
        print(f"=== window {wname}  ({int(wmask.sum())} base bars) ===")
        print(f"{'scale':>6} {'feat':>10} {'IC_leaky':>10} {'IC_causal':>11} "
              f"{'|delta|':>9} {'n':>8}")
        for scale in SCALES:
            if scale == base_scale:
                continue
            coarse_ns = np.asarray(h._scale_timestamps[scale]).astype(
                "datetime64[ns]").astype("int64")
            causal_map = np.asarray(h._scale_index_map[scale])  # FIXED (de-leaked)
            leaky_map = np.clip(
                np.searchsorted(coarse_ns, base_ns, side="right") - 1,
                0, len(coarse_ns) - 1)  # reconstructed OLD leaky map
            feats = h._scale_features[scale]  # (T_coarse, N, 8)
            for fi, fname in FEATS.items():
                leaky_vals = feats[leaky_map, :, fi]    # (T, N)
                causal_vals = feats[causal_map, :, fi]  # (T, N)
                fl = leaky_vals[wmask].reshape(-1)
                fc = causal_vals[wmask].reshape(-1)
                fr = fwd[wmask].reshape(-1)
                ic_l, n_l = _ic(fl, fr)
                ic_c, n_c = _ic(fc, fr)
                print(f"{scale:>5}h {fname:>10} {ic_l:>10.4f} {ic_c:>11.4f} "
                      f"{abs(ic_l - ic_c):>9.4f} {n_c:>8}")
        print()


if __name__ == "__main__":
    main()
