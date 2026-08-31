"""CMGP1 X2-leak falsification probe v2 — tighten the necessary-not-sufficient gap.

v1 (cmgp1_x2_leak_ic_probe.py) showed the raw coarse `last`-value forward-IC
collapses leaky->causal (+0.31 -> -0.04 @4h). That tests ONE of the things the RL
policy sees. v2 closes the gap the single-feature probe leaves open by testing the
dimensions the policy actually exploits, ALL no-GPU:

  (A) OBS-STAT IC  — the policy consumes mean/std/last over a 30-bar window per
      scale (summary_stats), not the raw per-bar feature. Measure each stat's
      forward-IC, leaky vs causal. The window-MEAN of de-leaked coarse log_return
      is momentum over the last W CLOSED coarse bars — the one place a REAL
      (causal) TSMOM signal could survive the de-leak.
  (B) BASE-SCALE causal IC — scale=1 is 1:1 (no leak); does the causal base/coarse
      signal carry ANY forward edge once de-leaked?
  (C) MULTI-HORIZON — forward returns at 1h/4h/24h. Each scale's window captures a
      different horizon; a 1h-forward test understates the 1D-regime scale. Fair
      test = match the horizon to where momentum would live.
  (D) CROSS-SECTIONAL rank-IC — CMGP1 is multi-asset; the policy can play relative
      value. At each bar, rank assets by the causal feature vs forward-return ranks
      (the standard XS IC_IR). This is ALSO the repurposing question: a positive
      causal XS momentum IC points at the GO'd cross-sectional allocator.

Decision logic: if EVERY causal signal (base+coarse, pooled+cross-sectional, all
horizons) is ~0 -> decisive NO-GO, skip the GPU canary. If de-leaked window-MEAN
momentum or XS rank-IC is materially non-zero -> residual causal edge -> the RL
canary (or a cross-sectional reframe) is warranted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sharpen.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler

CACHE = "data/crypto_cache/silver_ohlcv.parquet"
SCALES = [1, 4, 24]
W = 30  # window_size (matches config)
HORIZONS = [1, 4, 24]  # forward base-bar (1h) horizons
# OOS test window (config) + full sample.
WINDOWS = {"test_2025Q4": ("2025-10-01", "2025-12-31"), "full_sample": (None, None)}
FEAT_LOGRET, FEAT_CLOSEZ = 0, 6


def _pooled_ic(feat: np.ndarray, fwd: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(feat) & np.isfinite(fwd)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    rho, _ = spearmanr(feat[m], fwd[m])
    return float(rho), int(m.sum())


def _xs_ic(feat: np.ndarray, fwd: np.ndarray) -> tuple[float, float, int]:
    """Cross-sectional rank-IC: per-bar Spearman across assets, then mean + IC_IR.

    feat, fwd: (T, N). Returns (mean_xs_ic, ic_ir, n_bars_used).
    """
    ics = []
    for t in range(feat.shape[0]):
        fr = feat[t]
        rr = fwd[t]
        m = np.isfinite(fr) & np.isfinite(rr)
        if m.sum() < 4:  # need >=4 assets to rank cross-sectionally
            continue
        if np.unique(fr[m]).size < 3 or np.unique(rr[m]).size < 3:
            continue
        rho, _ = spearmanr(fr[m], rr[m])
        if np.isfinite(rho):
            ics.append(rho)
    if len(ics) < 20:
        return float("nan"), float("nan"), len(ics)
    ics = np.asarray(ics)
    mean_ic = float(ics.mean())
    ic_ir = float(mean_ic / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else float("nan")
    return mean_ic, ic_ir, len(ics)


def _rolling_stat(coarse_feat: np.ndarray, stat: str) -> np.ndarray:
    """Rolling W-window stat over the coarse grid, per asset. (T_coarse, N)."""
    s = pd.DataFrame(coarse_feat)
    r = s.rolling(W, min_periods=1)
    if stat == "mean":
        return r.mean().values
    if stat == "std":
        return r.std().fillna(0.0).values
    raise ValueError(stat)


def main() -> None:
    df = pd.read_parquet(CACHE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    assets = sorted(df["ticker"].unique().tolist())
    print(f"universe ({len(assets)}): {assets}")
    print(f"NOTE: cache = funding-arb universe, not CMGP1's exact 10 — leak is a\n"
          f"      construction property, universe-agnostic. W={W}, scales={SCALES}\n")

    fcfg = {"scales": SCALES, "window_size": W, "obs_mode": "summary_stats",
            "summary_feature_indices": [0, 1, 2, 6, 7], "norm_span": 120,
            "feature_set_version": "v1"}
    h = MultiScaleCryptoHandler(
        ohlcv_df=df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]],
        funding_df=None, assets=assets, feature_config=fcfg)

    base_ts = np.asarray(h._base_timestamps).astype("datetime64[ns]")
    base_ns = base_ts.astype("int64")
    bclose, active = h._base_close, h._base_active
    T, N = bclose.shape
    base_scale = min(SCALES)

    # Forward returns at each horizon, gated on both endpoints active.
    fwd = {}
    for hz in HORIZONS:
        f = np.full((T, N), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = bclose[hz:] / np.where(bclose[:-hz] > 0, bclose[:-hz], np.nan)
        f[:-hz] = ratio - 1.0
        f[:-hz][~(active[:-hz] & active[hz:])] = np.nan
        fwd[hz] = f

    # Pre-compute coarse maps + rolling stats per scale.
    maps = {}      # scale -> {'causal':(T,), 'leaky':(T,)}
    rollstats = {}  # scale -> {feat -> {'mean','std'}} on coarse grid
    lastfeat = {}   # scale -> feat -> (T_coarse,N)
    for scale in SCALES:
        if scale == base_scale:
            causal = leaky = np.arange(T)
        else:
            cns = np.asarray(h._scale_timestamps[scale]).astype("datetime64[ns]").astype("int64")
            causal = np.asarray(h._scale_index_map[scale])
            leaky = np.clip(np.searchsorted(cns, base_ns, side="right") - 1, 0, len(cns) - 1)
        maps[scale] = {"causal": causal, "leaky": leaky}
        feats = h._scale_features[scale]
        rollstats[scale] = {}
        lastfeat[scale] = {}
        for fi in (FEAT_LOGRET, FEAT_CLOSEZ):
            rollstats[scale][fi] = {"mean": _rolling_stat(feats[:, :, fi], "mean"),
                                    "std": _rolling_stat(feats[:, :, fi], "std")}
            lastfeat[scale][fi] = feats[:, :, fi]

    def obs(scale, fi, stat, which, t_idx):
        idx = maps[scale][which][t_idx]
        if stat == "last":
            return lastfeat[scale][fi][idx]
        return rollstats[scale][fi][stat][idx]

    for wname, (s, e) in WINDOWS.items():
        wmask = (np.ones(T, bool) if s is None
                 else (base_ts >= np.datetime64(s)) & (base_ts <= np.datetime64(e)))
        wmask[-max(HORIZONS):] = False
        widx = np.where(wmask)[0]
        print(f"################ window {wname}  ({len(widx)} base bars) ################")

        # ---- (A)/(B) POOLED IC: stat x feature x scale, leaky vs causal, fwd-1h ----
        print("\n[A] Pooled forward-1h IC  (leaky vs causal)  — the obs the policy sees")
        print(f"{'scale':>6} {'feat':>9} {'stat':>5} {'IC_leaky':>9} {'IC_causal':>10} {'|d|':>7}")
        for scale in SCALES:
            tag = "(base,1:1)" if scale == base_scale else ""
            for fi in (FEAT_LOGRET, FEAT_CLOSEZ):
                fname = "logret" if fi == FEAT_LOGRET else "close_z"
                for stat in ("mean", "std", "last"):
                    fl = obs(scale, fi, stat, "leaky", widx).reshape(-1)
                    fc = obs(scale, fi, stat, "causal", widx).reshape(-1)
                    fr = fwd[1][widx].reshape(-1)
                    icl, _ = _pooled_ic(fl, fr)
                    icc, n = _pooled_ic(fc, fr)
                    print(f"{scale:>5}h {fname:>9} {stat:>5} {icl:>9.4f} {icc:>10.4f} "
                          f"{abs(icl - icc):>7.4f} {tag}")

        # ---- (C) MULTI-HORIZON causal momentum (window-MEAN logret) ----
        print("\n[C] Causal window-MEAN logret (momentum) — pooled IC by fwd horizon")
        print(f"{'scale':>6} " + " ".join(f"{'fwd'+str(hz)+'h':>9}" for hz in HORIZONS))
        for scale in SCALES:
            row = f"{scale:>5}h "
            for hz in HORIZONS:
                fc = obs(scale, FEAT_LOGRET, "mean", "causal", widx).reshape(-1)
                fr = fwd[hz][widx].reshape(-1)
                icc, _ = _pooled_ic(fc, fr)
                row += f"{icc:>9.4f} "
            print(row)

        # ---- (D) CROSS-SECTIONAL rank-IC (causal), momentum + close_z ----
        print("\n[D] Cross-sectional rank-IC (causal)  mean_xs_ic [IC_IR]  by fwd horizon")
        for fi, fname in ((FEAT_LOGRET, "MEAN-logret"), (FEAT_CLOSEZ, "last-close_z")):
            stat = "mean" if fi == FEAT_LOGRET else "last"
            print(f"  feature={fname} (stat={stat})")
            print(f"{'scale':>8} " + " ".join(f"{'fwd'+str(hz)+'h':>16}" for hz in HORIZONS))
            for scale in SCALES:
                row = f"{scale:>7}h "
                for hz in HORIZONS:
                    fc = obs(scale, fi, stat, "causal", widx)  # (len(widx), N)
                    fr = fwd[hz][widx]
                    mic, ir, nb = _xs_ic(fc, fr)
                    row += f"{mic:>8.4f}[{ir:>5.1f}] "
                print(row)
        print()


if __name__ == "__main__":
    main()
