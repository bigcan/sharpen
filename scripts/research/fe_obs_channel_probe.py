"""FE observation-channel Stage-0 probe — FFD price channel + per-timeframe norm_span.

Spec (pre-registered, gates FROZEN before any result):
    docs/research/fe_obs_channel_preregistration_2026-07-12.md

This is the CPU-only ($0 GPU) incremental-information gate. It measures whether a
new obs channel adds forward-return-predictive information *beyond* the incumbent
8-feature / norm_span=120 observation. The gated quantity is the DIFFERENCE
(delta-IC of augmented-vs-baseline on identical folds), never the augmented
model's absolute score — a channel that merely re-expresses information already
present cannot pass.

Forks scripts/research/r1_illiquidity_probe.py (same de-leaked MultiScaleOHLCV
path, same WF/bootstrap/FDR machinery). Feature construction is consumed directly
from finrl_pro_ds.data.multiscale_handler — no reimplementation. The FFD channel
reuses _fractional_diff (crypto_features, FE-08-fixed) + the handler's own
_symlog -> _ema_zscore_tanh normalization.

Stages:
    python scripts/research/fe_obs_channel_probe.py --stage qc        # read-only data QC + window coverage
    python scripts/research/fe_obs_channel_probe.py --stage tripwire  # TW-1/TW-2 (HALT on fail)
    python scripts/research/fe_obs_channel_probe.py --stage probe     # base + FFD + norm_span, all cells
    python scripts/research/fe_obs_channel_probe.py --stage verdict   # frozen gates -> verdict.json
    python scripts/research/fe_obs_channel_probe.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crypto.features.crypto_features import _fractional_diff  # noqa: E402
from finrl_pro_ds.data.multiscale_handler import (  # noqa: E402
    MultiScaleOHLCVHandler,
    _ema_zscore_tanh,
    _symlog,
)

OUT_DIR = ROOT / "results" / "fe_obs_channel"

# ----------------------------------------------------------------- spec ----
# Cells: substrate file -> scales. Roles decide the verdict family.
#   primary            = window-complete gold cells (live SG-1 XAUUSD substrate)
#   secondary_reduced  = GC CME proxy, 2025-only data (contributes 2025 folds)
#   control            = BTC (canary IC ~ 0; excluded from discovery family)
CELLS = {
    "xauusd_15m": {"file": "data/oanda/xauusd_m1_m.parquet", "ticker": "XAUUSD",
                   "scales": [15, 60, 240], "role": "primary"},
    "xauusd_3m":  {"file": "data/oanda/xauusd_m1_m.parquet", "ticker": "XAUUSD",
                   "scales": [3, 15, 60], "role": "primary"},
    "gc_15m":     {"file": "data/cme/gc_2025_lob1_1min_stitched.parquet", "ticker": "GC",
                   "scales": [15, 60, 240], "role": "secondary_reduced"},
    "btc_15m":    {"file": "data/btc_usdt_1min_bybit.parquet", "ticker": "BTC",
                   "scales": [15, 60, 240], "role": "control"},
    "btc_3m":     {"file": "data/btc_usdt_1min_bybit.parquet", "ticker": "BTC",
                   "scales": [3, 15, 60], "role": "control"},
}

WINDOW = 30
N_FEATS = 8
HORIZONS = [1, 4]              # base bars; primary = 1
EMBARGO_BARS = 16
INCUMBENT_SPAN = 120

# frozen grids (spec sections 2-3) — NO post-hoc extension may rescue a NO-GO
FFD_DS = [0.3, 0.4, 0.5]
FFD_WINDOW = 100
NORM_SPANS_ALT = [60, 240, 480]   # vs incumbent 120

OOS_FOLD_STARTS = ["2025-06-01", "2025-08-01", "2025-10-01",
                   "2025-12-01", "2026-02-01", "2026-04-01"]
OOS_END = "2026-06-01"

RIDGE_ALPHAS = [1e2, 1e3, 1e4]
MAX_TRAIN_ROWS = 150_000
GBM_MAX_ROWS = 100_000
SUMMARY_FEATS = [0, 1, 2, 6, 7]   # of the base 8 (GBM reduced view)
DEADBAND_PCTL = 60.0

N_BOOT = 2_000
BOOT_CHUNK = 100
RNG_SEED = 20260712

# one-way cost (spread/2 + fee/slip); only affects the RELATIVE pf_net check (S0-B)
COST_ONEWAY = {"XAUUSD": 2.0 / 1e4, "GC": 2.0 / 1e4, "BTC": 10.5 / 1e4}

# Gate thresholds (spec section 5 — FROZEN)
S0A_DELTA_IC = 0.010
S0A_FDR_Q = 0.10
S0B_GROSS_PF = 1.20
S0C_COLLINEAR_RHO = 0.95
TW1_MIN_IC = 0.10
TW3_MAX_BTC_IC = 0.01
QC_STALE_FRAC = 0.02
QC_STALE_PNL_SHARE = 0.05


# ------------------------------------------------------------------- qc ----

def stale_scan(df: pd.DataFrame) -> dict:
    """Read-only stale-print scan (spec Gate D; clean_ohlcv NOT run — see deviation note)."""
    close = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    o, h, lo = (df[c].to_numpy(np.float64) for c in ("open", "high", "low"))
    n = len(close)
    same = np.zeros(n, dtype=bool)
    same[1:] = close[1:] == close[:-1]
    starts_all = np.flatnonzero(~same)
    ends_all = np.r_[starts_all[1:] - 1, n - 1] if len(starts_all) else np.array([], dtype=int)
    lens_all = ends_all - starts_all + 1
    volsum_all = np.add.reduceat(vol, starts_all) if len(starts_all) else np.array([])
    sus_mask = (lens_all >= 30) & (volsum_all > 0)
    suspect = np.zeros(n, dtype=bool)
    for i0, i1 in zip(starts_all[sus_mask], ends_all[sus_mask]):
        suspect[i0:i1 + 1] = True
    logret = np.zeros(n)
    logret[1:] = np.log(np.where(close[1:] > 0, close[1:], np.nan)
                        / np.where(close[:-1] > 0, close[:-1], np.nan))
    logret = np.nan_to_num(logret)
    total_abs = np.abs(logret).sum()
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = suspect[1:] != suspect[:-1]
    stale_pnl_share = float(np.abs(logret[boundary]).sum() / total_abs) if total_abs > 0 else 0.0
    frac = float(suspect.mean())
    return {
        "rows_1min": int(n),
        "stale_suspect_frac": round(frac, 5),
        "stale_pnl_share": round(stale_pnl_share, 5),
        "zero_vol_frac": round(float((vol == 0).mean()), 5),
        "untestable": bool(frac > QC_STALE_FRAC or stale_pnl_share > QC_STALE_PNL_SHARE),
    }


def run_qc() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qc: dict = {}
    seen_files: dict[str, dict] = {}
    for cell, meta in CELLS.items():
        f = ROOT / meta["file"]
        if not f.exists():
            qc[cell] = {"status": "FILE_MISSING", "file": meta["file"]}
            continue
        if meta["file"] not in seen_files:
            df = pd.read_parquet(f)
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            scan = stale_scan(df)
            scan["ts_min"] = str(ts.min())
            scan["ts_max"] = str(ts.max())
            # which frozen folds this file's window can populate (test side)
            folds_ok = []
            bounds = [pd.Timestamp(b) for b in OOS_FOLD_STARTS] + [pd.Timestamp(OOS_END)]
            for k in range(len(OOS_FOLD_STARTS)):
                f0, f1 = bounds[k], bounds[k + 1]
                if ((ts >= f0) & (ts < f1)).sum() > 200:
                    folds_ok.append(OOS_FOLD_STARTS[k])
            scan["folds_populated"] = folds_ok
            scan["status"] = "UNTESTABLE" if scan["untestable"] else "CLEAN"
            seen_files[meta["file"]] = scan
        s = dict(seen_files[meta["file"]])
        s["role"] = meta["role"]
        qc[cell] = s
        print(f"QC {cell:12s} [{meta['role']:17s}] {s['status']} "
              f"stale={s['stale_suspect_frac']} folds={len(s['folds_populated'])} "
              f"{s['ts_min']}->{s['ts_max']}", flush=True)
    (OUT_DIR / "data_qc.json").write_text(json.dumps(qc, indent=2))
    print("QC done. NOTE: clean_ohlcv NOT invoked — production files are already under "
          "the DATA-CLEAN invariant from their own pipelines; a research probe must not "
          "mutate live-strategy data. Read-only stale-scan substitutes for Gate D.", flush=True)


# ------------------------------------------------------- feature builder ----

def _norm_ts(base_ts) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(base_ts)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _tw4_assert(handler, scales, base, asset):
    for s in scales:
        if s == base:
            continue
        idx_map = handler._scale_index_map[s]
        coarse_ts = handler._scale_timestamps[s].astype("int64")
        bts = handler._base_timestamps.astype("int64")
        closes_at = coarse_ts[idx_map] + s * 60 * 1_000_000_000
        viol = closes_at > bts
        first_valid = np.argmax(~viol) if (~viol).any() else len(bts)
        if viol.any() and viol[max(first_valid, WINDOW):].any():
            raise AssertionError(f"TW-4 FAIL {asset} scale={s}: look-ahead coarse-bar map")


def _warmup_t0(handler, scales, base):
    starts = []
    for s in scales:
        if s == base:
            starts.append(WINDOW)
        else:
            ok = np.where(handler._scale_index_map[s] >= WINDOW)[0]
            starts.append(int(ok[0]) if len(ok) else len(handler._base_close))
    return max(starts)


def _gather(handler, scales, base, t0, n_eff, feat_by_scale, n_feats):
    X = np.empty((n_eff, len(scales), WINDOW, n_feats), dtype=np.float32)
    offs = np.arange(-(WINDOW - 1), 1)
    for si, s in enumerate(scales):
        feats = feat_by_scale[s]
        idx = (np.arange(t0, t0 + n_eff) if s == base
               else handler._scale_index_map[s][t0:t0 + n_eff])
        gather = idx[:, None] + offs[None, :]
        np.clip(gather, 0, len(feats) - 1, out=gather)
        X[:, si, :, :] = feats[gather]
    return X


def _ffd_channel_by_scale(handler, scales, d):
    """FFD(log close) -> _symlog -> _ema_zscore_tanh(INCUMBENT_SPAN), per scale.

    Aligned row-for-row with handler._scale_features[s] (same untrimmed resampled df).
    """
    chan = {}
    for s in scales:
        close = handler._scale_dfs[s]["close"].to_numpy(np.float64)
        logc = np.log(np.where(close > 0, close, np.nan))
        logc = pd.Series(logc).ffill().fillna(0.0)
        ffd = _fractional_diff(logc, d=d, window=FFD_WINDOW).to_numpy(np.float64)
        col = _ema_zscore_tanh(_symlog(ffd), INCUMBENT_SPAN)
        chan[s] = col.reshape(-1, 1).astype(np.float32)
    return chan


def build_cell(file_path, ticker, scales, norm_span, add_ffd=False, ffd_d=0.4,
               leak_shift=False):
    """Return (X, ys, ts, base, ffd_last) for one cell configuration."""
    handler = MultiScaleOHLCVHandler(
        file_path=str(ROOT / file_path), ticker=ticker,
        feature_config={"scales": scales, "window_size": WINDOW, "norm_span": norm_span},
    )
    base = min(scales)
    _tw4_assert(handler, scales, base, ticker)
    close = handler._base_close
    n = len(close)
    t0 = _warmup_t0(handler, scales, base)
    n_eff = n - t0 - max(HORIZONS)
    if n_eff < 5_000:
        raise RuntimeError(f"{ticker}/{scales}: insufficient bars after warmup ({n_eff})")

    feat_base = {s: handler._scale_features[s] for s in scales}
    X = _gather(handler, scales, base, t0, n_eff, feat_base, N_FEATS)
    ffd_last = None
    if add_ffd:
        chan = _ffd_channel_by_scale(handler, scales, ffd_d)
        Xf = _gather(handler, scales, base, t0, n_eff, chan, 1)
        X = np.concatenate([X, Xf], axis=-1)
        ffd_last = Xf[:, :, -1, 0]  # (n_eff, n_scales) last-bar channel per scale

    logc = np.log(np.where(close > 0, close, np.nan))
    ys = {h: np.nan_to_num(logc[t0 + h: t0 + h + n_eff] - logc[t0: t0 + n_eff]).astype(np.float32)
          for h in HORIZONS}
    ts = _norm_ts(handler._base_timestamps[t0: t0 + n_eff])
    if leak_shift:                       # TW-1: current target sees NEXT bar's features
        X = X[1:]
        ys = {h: v[:-1] for h, v in ys.items()}
        ts = ts[:-1]
        if ffd_last is not None:
            ffd_last = ffd_last[1:]
    del handler
    gc.collect()
    return X, ys, ts, base, ffd_last


def flatten_views(X: np.ndarray):
    """(n, n_scales, WINDOW, F) -> ridge view (n, n_scales*WINDOW*F) and GBM reduced view."""
    n, ns, w, F = X.shape
    ridge_v = X.reshape(n, -1)
    sel_idx = SUMMARY_FEATS + ([F - 1] if F > N_FEATS else [])   # include FFD channel if present
    sel = X[:, :, :, sel_idx]
    mean = sel.mean(axis=2)
    std = sel.std(axis=2)
    last = X[:, :, -1, :]
    gbm_v = np.concatenate([mean.reshape(n, -1), std.reshape(n, -1), last.reshape(n, -1)], axis=1)
    return ridge_v, gbm_v


# ------------------------------------------------------------ WF models ----

def _stride_cap(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.arange(0, n, int(np.ceil(n / cap)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy(np.float64)
    rb = pd.Series(b).rank().to_numpy(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def wf_predict(view, y, ts, model, rng):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    spacing = np.median(np.diff(ts.values).astype("timedelta64[s]").astype(np.int64))
    embargo = pd.Timedelta(seconds=int(spacing) * EMBARGO_BARS)
    pred = np.full(len(y), np.nan, dtype=np.float32)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    taus: list[float] = []
    bounds = [pd.Timestamp(b) for b in OOS_FOLD_STARTS] + [pd.Timestamp(OOS_END)]
    for k in range(len(OOS_FOLD_STARTS)):
        f0, f1 = bounds[k], bounds[k + 1]
        tr = np.where(ts < (f0 - embargo))[0]
        te = np.where((ts >= f0) & (ts < f1))[0]
        if len(tr) < 5_000 or len(te) < 200:
            continue
        cap = GBM_MAX_ROWS if model == "gbm" else MAX_TRAIN_ROWS
        tr = tr[_stride_cap(len(tr), cap)]
        Xtr, ytr = view[tr], y[tr]
        if model == "ridge":
            scaler = StandardScaler().fit(Xtr)
            Xs = scaler.transform(Xtr).astype(np.float32)
            n_inner = int(len(tr) * 0.8)
            best_alpha, best_ic = RIDGE_ALPHAS[0], -np.inf
            for alpha in RIDGE_ALPHAS:
                m = Ridge(alpha=alpha).fit(Xs[:n_inner], ytr[:n_inner])
                ic = _spearman(m.predict(Xs[n_inner:]), ytr[n_inner:])
                if ic > best_ic:
                    best_ic, best_alpha = ic, alpha
            m = Ridge(alpha=best_alpha).fit(Xs, ytr)
            p_te = m.predict(scaler.transform(view[te]).astype(np.float32))
            p_tr = m.predict(Xs)
        else:
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                              early_stopping=False, random_state=RNG_SEED)
            m.fit(Xtr, ytr)
            p_te = m.predict(view[te])
            p_tr = m.predict(Xtr)
        pred[te] = p_te.astype(np.float32)
        fold_id[te] = k
        taus.append(float(np.percentile(np.abs(p_tr), DEADBAND_PCTL)))
    return pred, fold_id, taus


# ----------------------------------------------------------- statistics ----

def _bars_per_day(ts):
    return max(int(round(86_400 / np.median(
        np.diff(ts.values).astype("timedelta64[s]").astype(np.int64)))), 1)


def paired_delta_ic_boot(pred_b, pred_a, y, ts, rng):
    """Circular day-block bootstrap of delta-IC = IC(aug) - IC(base), paired on same rows."""
    n = len(y)
    block = min(_bars_per_day(ts), n)
    rb = pd.Series(pred_b).rank().to_numpy(np.float32)
    ra = pd.Series(pred_a).rank().to_numpy(np.float32)
    ry = pd.Series(y).rank().to_numpy(np.float32)
    n_blocks = int(np.ceil(n / block))

    def _rc(x_idx, y_idx):
        xm = x_idx.mean(1, keepdims=True)
        ym = y_idx.mean(1, keepdims=True)
        cov = ((x_idx - xm) * (y_idx - ym)).mean(1)
        den = x_idx.std(1) * y_idx.std(1)
        return np.where(den > 0, cov / den, 0.0)

    deltas = np.empty(N_BOOT, dtype=np.float64)
    done = 0
    while done < N_BOOT:
        b = min(BOOT_CHUNK, N_BOOT - done)
        starts = rng.randint(0, n, size=(b, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        idx = idx.reshape(b, -1)[:, :n]
        ic_b = _rc(rb[idx], ry[idx])
        ic_a = _rc(ra[idx], ry[idx])
        deltas[done:done + b] = ic_a - ic_b
        done += b
    return {
        "delta_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_ci_hi": float(np.percentile(deltas, 97.5)),
        "delta_p_one_sided": float((deltas <= 0).mean()),  # H1: delta > 0
    }


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        prev = min(prev, p[i] * m / (r + 1))
        q[i] = prev
    return q.tolist()


def trade_sim(pred, fold_id, taus, y1, cost_oneway):
    valid = fold_id >= 0
    simple_ret = np.expm1(y1)
    seen = sorted(set(fold_id[valid].tolist()))
    tau_by_fold = {k: (taus[j] if j < len(taus) else (taus[-1] if taus else 0.0))
                   for j, k in enumerate(seen)}
    pnl_fr, pnl_net = [], []
    fold_sums: dict[int, float] = {}
    pos_prev = 0.0
    idxs = np.where(valid)[0]
    for prev_i, i in zip(np.r_[-1, idxs[:-1]], idxs):
        if prev_i >= 0 and fold_id[prev_i] != fold_id[i]:
            pos_prev = 0.0
        tau = tau_by_fold[int(fold_id[i])]
        p = pred[i]
        pos = float(np.sign(p)) if abs(p) >= tau else 0.0
        gross = pos * simple_ret[i]
        net = gross - cost_oneway * abs(pos - pos_prev)
        pnl_fr.append(gross)
        pnl_net.append(net)
        fold_sums[int(fold_id[i])] = fold_sums.get(int(fold_id[i]), 0.0) + net
        pos_prev = pos

    def pf(arr):
        a = np.asarray(arr)
        up = a[a > 0].sum()
        dn = -a[a < 0].sum()
        return float(up / dn) if dn > 0 else float("inf")
    return {"pf_frictionless": round(pf(pnl_fr), 4), "pf_net": round(pf(pnl_net), 4),
            "pos_folds": int(sum(1 for v in fold_sums.values() if v > 0))}


# ----------------------------------------------------------- probe stage ----

def _ic_and_sim(view, y1, ts, model, rng, cost):
    pred, fold_id, taus = wf_predict(view, y1, ts, model, rng)
    valid = fold_id >= 0
    if valid.sum() < 1_000:
        return None
    ic = _spearman(pred[valid], y1[valid])
    sim = trade_sim(pred, fold_id, taus, y1, cost)
    return {"pred": pred, "fold_id": fold_id, "valid": valid, "ic": ic,
            "n_oos": int(valid.sum()), **sim}


def probe_cell(cell, meta, rng):
    file, ticker, scales, role = meta["file"], meta["ticker"], meta["scales"], meta["role"]
    cost = COST_ONEWAY[ticker]
    rows = []

    # --- baseline (8 feats, span 120) ---
    Xb, ys, ts, base, _ = build_cell(file, ticker, scales, INCUMBENT_SPAN)
    ridge_b, gbm_b = flatten_views(Xb)
    y1 = ys[1]
    base_res = {}
    for model, view in (("ridge", ridge_b), ("gbm", gbm_b)):
        r = _ic_and_sim(view, y1, ts, model, rng, cost)
        base_res[model] = r
        if r:
            rows.append({"cell": cell, "role": role, "part": "base", "variant": "span120",
                         "model": model, "ic_base": round(r["ic"], 5), "ic_var": round(r["ic"], 5),
                         "delta_ic": 0.0, "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
                         "delta_p_one_sided": 1.0, "pf_net_base": r["pf_net"],
                         "pf_net_var": r["pf_net"], "pf_fr_var": r["pf_frictionless"],
                         "tw5_max_collinear": "", "n_oos": r["n_oos"]})
    del Xb, ridge_b, gbm_b
    gc.collect()

    # --- Part A: FFD channel (9th feature) ---
    for d in FFD_DS:
        Xa, _, ts_a, _, ffd_last = build_cell(file, ticker, scales, INCUMBENT_SPAN,
                                              add_ffd=True, ffd_d=d)
        # TW-5 redundancy: FFD last-bar vs base last-bar log_ret(0) and close_z(6), pooled per scale
        # recompute base last-bar features quickly from Xa? Xa has 9 feats incl base 0..7.
        collin = 0.0
        for si in range(len(scales)):
            lr = Xa[:, si, -1, 0]
            cz = Xa[:, si, -1, 6]
            ff = Xa[:, si, -1, 8]
            collin = max(collin, abs(_spearman(ff, lr)), abs(_spearman(ff, cz)))
        ridge_a, gbm_a = flatten_views(Xa)
        for model, view in (("ridge", ridge_a), ("gbm", gbm_a)):
            br = base_res.get(model)
            r = _ic_and_sim(view, y1, ts_a, model, rng, cost)
            if not (r and br):
                continue
            boot = paired_delta_ic_boot(br["pred"][br["valid"]], r["pred"][r["valid"]],
                                        y1[r["valid"]], ts_a[r["valid"]], rng)
            rows.append({"cell": cell, "role": role, "part": "A_ffd", "variant": f"d{d}",
                         "model": model, "ic_base": round(br["ic"], 5), "ic_var": round(r["ic"], 5),
                         "delta_ic": round(r["ic"] - br["ic"], 5),
                         "delta_ci_lo": round(boot["delta_ci_lo"], 5),
                         "delta_ci_hi": round(boot["delta_ci_hi"], 5),
                         "delta_p_one_sided": boot["delta_p_one_sided"],
                         "pf_net_base": br["pf_net"], "pf_net_var": r["pf_net"],
                         "pf_fr_var": r["pf_frictionless"], "tw5_max_collinear": round(collin, 4),
                         "n_oos": r["n_oos"]})
        del Xa, ridge_a, gbm_a
        gc.collect()

    # --- Part B: per-timeframe norm_span ---
    for span in NORM_SPANS_ALT:
        Xs, _, ts_s, _, _ = build_cell(file, ticker, scales, span)
        ridge_s, gbm_s = flatten_views(Xs)
        for model, view in (("ridge", ridge_s), ("gbm", gbm_s)):
            br = base_res.get(model)
            r = _ic_and_sim(view, y1, ts_s, model, rng, cost)
            if not (r and br):
                continue
            boot = paired_delta_ic_boot(br["pred"][br["valid"]], r["pred"][r["valid"]],
                                        y1[r["valid"]], ts_s[r["valid"]], rng)
            rows.append({"cell": cell, "role": role, "part": "B_span", "variant": f"span{span}",
                         "model": model, "ic_base": round(br["ic"], 5), "ic_var": round(r["ic"], 5),
                         "delta_ic": round(r["ic"] - br["ic"], 5),
                         "delta_ci_lo": round(boot["delta_ci_lo"], 5),
                         "delta_ci_hi": round(boot["delta_ci_hi"], 5),
                         "delta_p_one_sided": boot["delta_p_one_sided"],
                         "pf_net_base": br["pf_net"], "pf_net_var": r["pf_net"],
                         "pf_fr_var": r["pf_frictionless"], "tw5_max_collinear": "",
                         "n_oos": r["n_oos"]})
        del Xs, ridge_s, gbm_s
        gc.collect()
    return rows


def run_probe(cell_filter=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RNG_SEED)
    qc = json.loads((OUT_DIR / "data_qc.json").read_text()) if (OUT_DIR / "data_qc.json").exists() else {}
    all_rows: list[dict] = []
    out_csv = OUT_DIR / (f"incremental_delta_{cell_filter}.csv" if cell_filter else "incremental_delta.csv")
    for cell, meta in CELLS.items():
        if cell_filter and cell != cell_filter:
            continue
        if qc.get(cell, {}).get("status") == "UNTESTABLE":
            print(f"skip {cell} (QC UNTESTABLE)", flush=True)
            continue
        print(f"probe {cell} ...", flush=True)
        try:
            rows = probe_cell(cell, meta, rng)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows = [{"cell": cell, "role": meta["role"], "part": "-", "variant": "-",
                     "model": "-", "status": f"ERROR: {str(exc)[:200]}"}]
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"  {cell}: {len([r for r in rows if 'delta_ic' in r])} variant-rows", flush=True)
    print("probe stage done.", flush=True)


def run_tripwires():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RNG_SEED)
    res = {}
    # TW-1: +1-bar feature leak on a gold cell base view must light up
    print("TW-1 leak-shift on xauusd_15m ...", flush=True)
    m = CELLS["xauusd_15m"]
    X, ys, ts, _, _ = build_cell(m["file"], m["ticker"], m["scales"], INCUMBENT_SPAN,
                                 leak_shift=True)
    ridge_v, _ = flatten_views(X)
    r = _ic_and_sim(ridge_v, ys[1], ts, "ridge", rng, COST_ONEWAY[m["ticker"]])
    res["tw1"] = {"ic_leak_shift": round(r["ic"], 5), "pass": bool(r["ic"] >= TW1_MIN_IC)}
    print(f"  TW-1 ic={res['tw1']['ic_leak_shift']} pass={res['tw1']['pass']}", flush=True)
    del X, ridge_v
    gc.collect()
    # TW-2: day-block shuffled target -> base IC dead AND paired delta-IC CI contains 0
    print("TW-2 shuffle-null on xauusd_15m ...", flush=True)
    Xb, ys, ts, _, _ = build_cell(m["file"], m["ticker"], m["scales"], INCUMBENT_SPAN)
    Xa, _, ts_a, _, _ = build_cell(m["file"], m["ticker"], m["scales"], INCUMBENT_SPAN,
                                   add_ffd=True, ffd_d=0.4)
    days = pd.Series(ts.normalize())
    uniq = days.unique()
    perm = rng.permutation(len(uniq))
    mp = {d: i for i, d in enumerate(uniq)}
    order = np.argsort([perm[mp[d]] for d in days], kind="stable")
    y_sh = ys[1][order]
    rb = _ic_and_sim(flatten_views(Xb)[0], y_sh, ts, "ridge", rng, COST_ONEWAY[m["ticker"]])
    ra = _ic_and_sim(flatten_views(Xa)[0], y_sh, ts_a, "ridge", rng, COST_ONEWAY[m["ticker"]])
    boot = paired_delta_ic_boot(rb["pred"][rb["valid"]], ra["pred"][ra["valid"]],
                                y_sh[ra["valid"]], ts_a[ra["valid"]], rng)
    res["tw2"] = {"base_ic_shuffled": round(rb["ic"], 5),
                  "delta_ci_lo": round(boot["delta_ci_lo"], 5),
                  "delta_ci_hi": round(boot["delta_ci_hi"], 5),
                  "pass": bool(abs(rb["ic"]) < 0.015 and boot["delta_ci_lo"] <= 0 <= boot["delta_ci_hi"])}
    print(f"  TW-2 base_ic={res['tw2']['base_ic_shuffled']} "
          f"dCI=[{res['tw2']['delta_ci_lo']},{res['tw2']['delta_ci_hi']}] "
          f"pass={res['tw2']['pass']}", flush=True)
    res["tw4"] = {"note": "asserted inside build_cell on every gather", "pass": True}
    res["tw3"] = {"note": "asserted in verdict: BTC control base IC <= +0.01"}
    (OUT_DIR / "tripwires.json").write_text(json.dumps(res, indent=2))
    if not (res["tw1"]["pass"] and res["tw2"]["pass"]):
        raise SystemExit("TRIPWIRE FAILURE — HALT (fix harness before reading results)")
    print("Tripwires TW-1/TW-2 PASS.", flush=True)


def run_verdict():
    parts = [pd.read_csv(p) for p in sorted(OUT_DIR.glob("incremental_delta*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.get("part").isin(["base", "A_ffd", "B_span"])].copy()
    df = df.drop_duplicates(subset=["cell", "part", "variant", "model"], keep="last")
    var = df[df["part"].isin(["A_ffd", "B_span"])].copy()
    # Spec §0/§5: the verdict FAMILY is PRIMARY (window-complete gold) cells ONLY.
    # gc_15m is secondary/robustness (2025-only proxy) — reported separately, it
    # "cannot count toward or against the frozen verdict". BTC = control (excluded).
    var["is_primary"] = var["role"] == "primary"

    # TW-3: BTC control base IC
    btc = df[(df.cell == "btc_15m") & (df.part == "base") & (df.model == "ridge")]["ic_base"]
    tw3_pass = bool(len(btc) and btc.iloc[0] <= TW3_MAX_BTC_IC)

    # BH-FDR within each part's family (PRIMARY cells only)
    fam = var[var["is_primary"]].copy()
    fam["q_fdr"] = np.nan
    for part in ("A_ffd", "B_span"):
        m = fam["part"] == part
        if m.any():
            fam.loc[m, "q_fdr"] = bh_fdr(fam.loc[m, "delta_p_one_sided"].tolist())

    fam["s0a"] = (fam["delta_ic"] >= S0A_DELTA_IC) & (fam["delta_ci_lo"] > 0) & (fam["q_fdr"] < S0A_FDR_Q)
    fam["s0b"] = (fam["pf_net_var"] >= fam["pf_net_base"]) & (fam["pf_fr_var"] >= S0B_GROSS_PF)
    # S0-C only applies to Part A
    def _collin_ok(r):
        if r["part"] != "A_ffd":
            return True
        try:
            return float(r["tw5_max_collinear"]) < S0C_COLLINEAR_RHO
        except (TypeError, ValueError):
            return True
    fam["s0c"] = fam.apply(_collin_ok, axis=1)
    fam["pass_all"] = fam["s0a"] & fam["s0b"] & fam["s0c"]

    passers = fam[fam["pass_all"]][["cell", "part", "variant", "model", "delta_ic",
                                    "delta_ci_lo", "q_fdr", "pf_net_base", "pf_net_var"]].to_dict("records")
    a_only = fam[fam["s0a"]][["cell", "part", "variant", "model", "delta_ic",
                              "delta_ci_lo", "q_fdr"]].to_dict("records")

    # Secondary/robustness (gc_15m): reported, NON-triggering per spec §0.
    sec = var[var["role"] == "secondary_reduced"].copy()
    sec_survivors = sec[(sec["delta_ic"] >= S0A_DELTA_IC) & (sec["delta_ci_lo"] > 0)][
        ["cell", "part", "variant", "model", "delta_ic", "delta_ci_lo",
         "pf_net_base", "pf_net_var"]].to_dict("records")

    if passers:
        verdict = "GO"
    elif a_only:
        verdict = "WEAK/AMBIGUOUS"
    else:
        verdict = "NO-GO"

    best = var.loc[var["delta_ic"].idxmax()] if len(var) else None
    out = {
        "spec": "docs/research/fe_obs_channel_preregistration_2026-07-12.md",
        "stage": "Stage-0 (CPU incremental-information gate)",
        "verdict": verdict,
        "tw3_btc_control_pass": tw3_pass,
        "s0a_s0b_s0c_passers_PRIMARY": passers,
        "s0a_only_passers_PRIMARY": a_only,
        "secondary_robustness_survivors_NONTRIGGERING": sec_survivors,
        "gates": {"S0A_DELTA_IC": S0A_DELTA_IC, "S0A_FDR_Q": S0A_FDR_Q,
                  "S0B_GROSS_PF": S0B_GROSS_PF, "S0C_COLLINEAR_RHO": S0C_COLLINEAR_RHO},
        "n_variant_cells": int(len(var)),
        "max_delta_ic": float(var["delta_ic"].max()) if len(var) else None,
        "max_delta_ic_where": (f"{best['cell']}|{best['part']}|{best['variant']}|{best['model']}"
                               if best is not None else None),
        "median_delta_ic_primary": float(fam["delta_ic"].median()) if len(fam) else None,
    }
    (OUT_DIR / "verdict.json").write_text(json.dumps(out, indent=2, default=str))
    fam.to_csv(OUT_DIR / "gated_family.csv", index=False)
    print(json.dumps(out, indent=2, default=str), flush=True)
    if not tw3_pass:
        print("WARNING: TW-3 failed — BTC control shows positive base IC; verdict INVALID "
              "until explained.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["qc", "tripwire", "probe", "verdict", "all"], default="all")
    ap.add_argument("--cell", choices=list(CELLS.keys()), default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("qc", "all"):
        run_qc()
    if args.stage in ("tripwire", "all"):
        run_tripwires()
    if args.stage in ("probe", "all"):
        run_probe(args.cell)
    if args.stage in ("verdict", "all"):
        run_verdict()


if __name__ == "__main__":
    main()
