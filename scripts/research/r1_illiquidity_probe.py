"""R1-illiquidity probe — QC, IC / walk-forward linear probe, programmatic verdict.

Spec (pre-registered, gates frozen): docs/research/r1_illiquidity_probe_spec_2026-06-12.md
Commit of record: a6336235 (spec committed before any result was computed).

Stages:
  python scripts/research/r1_illiquidity_probe.py --stage qc       # Gate D + liquidity
  python scripts/research/r1_illiquidity_probe.py --stage tripwire # TW-1..TW-4 (HALT on fail)
  python scripts/research/r1_illiquidity_probe.py --stage probe    # IC + WF probe, all cells
  python scripts/research/r1_illiquidity_probe.py --stage verdict  # gates -> verdict.json
  python scripts/research/r1_illiquidity_probe.py --stage all

Feature fidelity: features and causal coarse-bar maps are consumed directly from
sharpen.data.multiscale_handler.MultiScaleOHLCVHandler (the de-leaked X2
code path) — no reimplementation. TW-4 re-asserts the X2 causality condition on
every gathered window.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler  # noqa: E402

DATA_DIR = ROOT / "data" / "r1_illiquidity"
OUT_DIR = ROOT / "results" / "r1_illiquidity"

# ----------------------------------------------------------------- spec ----
START = pd.Timestamp("2024-09-01")
END = pd.Timestamp("2026-06-01")
OOS_FOLD_STARTS = ["2025-06-01", "2025-08-01", "2025-10-01",
                   "2025-12-01", "2026-02-01", "2026-04-01"]
OOS_END = "2026-06-01"

CELLS = {"gmgp1_15m": [15, 60, 240], "sg1_3m": [3, 15, 60]}
WINDOW = 30
NORM_SPAN = 120
N_FEATS = 8
HORIZONS = [1, 4, 16]          # base bars; primary = 1
EMBARGO_BARS = 16              # >= max horizon

RIDGE_ALPHAS = [1e2, 1e3, 1e4]
MAX_TRAIN_ROWS = 150_000       # uniform stride cap (ridge)
GBM_MAX_ROWS = 100_000         # uniform stride cap (GBM)
SUMMARY_FEATS = [0, 1, 2, 6, 7]

N_BOOT = 2_000
BOOT_CHUNK = 250
RNG_SEED = 20260612

# Gate thresholds (spec section 9 — FROZEN)
GATE_A_IC = 0.015
GATE_A_FDR_Q = 0.10
GATE_B_P = 0.05
GATE_C_NET_PF = 1.10
GATE_C_GROSS_PF = 1.20
GATE_C_MIN_POS_FOLDS = 4
GATE_D_STALE_FRAC = 0.02
GATE_D_STALE_PNL_SHARE = 0.05
DEADBAND_PCTL = 60.0

CRYPTO_TIER_SLIP_BPS = {"T0": 5.0, "T1": 7.5, "T2": 12.5, "T3": 20.0}
CRYPTO_FEE_BPS = 5.5
OANDA_COST_BPS = {
    "EUR_USD": 1.0, "USD_MXN": 8.0, "USD_ZAR": 10.0, "USD_TRY": 30.0,
    "XCU_USD": 5.0, "JP225_USD": 3.0, "WTICO_USD": 4.0,
}

TW1_MIN_IC = 0.10
TW3_MAX_BTC_IC = 0.01


# ------------------------------------------------------------- universe ----

def load_universe() -> list[dict]:
    """Assets from the fetch manifests, with venue/tier/cost metadata."""
    assets: list[dict] = []
    cm = json.loads((DATA_DIR / "fetch_manifest_crypto.json").read_text())
    for sym, info in cm["crypto"].items():
        if info.get("status") != "OK":
            assets.append({"asset": sym, "venue": "crypto", "status": "FETCH_FAIL",
                           "reason": info.get("reason", "")})
            continue
        slip = CRYPTO_TIER_SLIP_BPS[info["tier"]]
        assets.append({
            "asset": sym, "venue": "crypto", "tier": info["tier"],
            "file": str(ROOT / info["file"]), "status": "OK",
            "cost_oneway": (CRYPTO_FEE_BPS + slip) / 1e4,
            "cost_harsh": (CRYPTO_FEE_BPS + 2 * slip) / 1e4,
        })
    om = json.loads((DATA_DIR / "fetch_manifest_oanda.json").read_text())
    for inst, info in om["oanda"].items():
        if info.get("status") != "OK":
            assets.append({"asset": inst, "venue": "oanda", "status": "FETCH_FAIL",
                           "reason": info.get("reason", "")})
            continue
        bps = OANDA_COST_BPS[inst]
        assets.append({
            "asset": inst, "venue": "oanda", "tier": "FX",
            "file": str(ROOT / info["file"]), "status": "OK",
            "cost_oneway": bps / 1e4, "cost_harsh": 2 * bps / 1e4,
        })
    return assets


# ------------------------------------------------------------------- qc ----

def stale_scan(df: pd.DataFrame) -> dict:
    """Spec section 4 stale-print scan on 1-min data."""
    close = df["close"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    o, h, lo = (df[c].to_numpy(np.float64) for c in ("open", "high", "low"))
    n = len(close)

    same = np.zeros(n, dtype=bool)
    same[1:] = close[1:] == close[:-1]
    # vectorized run-length encoding of identical-close runs
    starts_all = np.flatnonzero(~same)               # each new-price run start
    ends_all = np.r_[starts_all[1:] - 1, n - 1]
    lens_all = ends_all - starts_all + 1
    volsum_all = np.add.reduceat(vol, starts_all)
    sus_mask = (lens_all >= 30) & (volsum_all > 0)
    suspect = np.zeros(n, dtype=bool)
    run_starts = starts_all[sus_mask].tolist()
    for i0, i1 in zip(starts_all[sus_mask], ends_all[sus_mask]):
        suspect[i0:i1 + 1] = True

    logret = np.zeros(n)
    logret[1:] = np.log(np.where(close[1:] > 0, close[1:], np.nan)
                        / np.where(close[:-1] > 0, close[:-1], np.nan))
    logret = np.nan_to_num(logret)
    total_abs = np.abs(logret).sum()
    # |return| at suspect-run boundaries (entry + exit bar of each run)
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = suspect[1:] != suspect[:-1]
    stale_pnl_share = float(np.abs(logret[boundary]).sum() / total_abs) if total_abs > 0 else 0.0

    # daily flat-fraction anomaly
    flat = (o == h) & (h == lo) & (lo == close)
    day = pd.to_datetime(df["timestamp"]).dt.normalize()
    daily_flat = pd.Series(flat).groupby(day.values).mean()
    anom_days = 0
    dfv = daily_flat.values
    for i in range(1, len(dfv) - 1):
        if dfv[i] > 0.5 and dfv[i - 1] < 0.1 and dfv[i + 1] < 0.1:
            anom_days += 1

    # post-flat jump: |r| > 5 sigma right after a >=30-bar flat run
    sigma = pd.Series(logret).rolling(1440, min_periods=100).std().to_numpy()
    jumps = 0
    for i1 in ends_all[sus_mask]:
        nxt = i1 + 1
        if nxt < n and sigma[nxt] and not np.isnan(sigma[nxt]):
            if np.abs(logret[nxt]) > 5 * sigma[nxt]:
                jumps += 1

    frac = float(suspect.mean())
    return {
        "stale_suspect_frac": round(frac, 5),
        "stale_pnl_share": round(stale_pnl_share, 5),
        "stale_runs": len(run_starts),
        "post_flat_jumps": int(jumps),
        "flat_anomaly_days": int(anom_days),
        "zero_vol_frac": round(float((vol == 0).mean()), 5),
        "untestable": bool(frac > GATE_D_STALE_FRAC or stale_pnl_share > GATE_D_STALE_PNL_SHARE),
    }


def run_qc() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    qc: dict = {}
    liq: dict = {}
    for a in universe:
        if a["status"] != "OK":
            qc[a["asset"]] = {"status": "FETCH_FAIL", "reason": a.get("reason", "")}
            continue
        print(f"QC {a['asset']} ...", flush=True)
        # DATA-CLEAN invariant: clean_ohlcv pass (creates .bak, repairs H/L)
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "clean_ohlcv.py"),
             "--input", a["file"]],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), timeout=600,
        )
        if r.returncode != 0:
            qc[a["asset"]] = {"status": "CLEAN_FAIL", "stderr": r.stderr[-500:]}
            continue
        df = pd.read_parquet(a["file"])
        scan = stale_scan(df)
        scan["status"] = "UNTESTABLE" if scan["untestable"] else "CLEAN"
        scan["rows_1min"] = int(len(df))
        qc[a["asset"]] = scan
        # liquidity proxy
        day = pd.to_datetime(df["timestamp"]).dt.normalize()
        if a["venue"] == "crypto":
            dollar = (df["volume"].astype(float) * df["close"].astype(float))
            med = float(dollar.groupby(day.values).sum().median())
            liq[a["asset"]] = {"venue": "crypto", "median_daily_dollar_vol": med,
                               "log10_liq": float(np.log10(max(med, 1.0)))}
        else:
            ticks = df["volume"].astype(float)
            med = float(ticks.groupby(day.values).sum().median())
            liq[a["asset"]] = {"venue": "oanda", "median_daily_ticks": med,
                               "log10_liq": float(np.log10(max(med, 1.0)))}
        print(f"  {a['asset']}: {scan['status']} stale_frac={scan['stale_suspect_frac']} "
              f"pnl_share={scan['stale_pnl_share']}", flush=True)
    (OUT_DIR / "data_qc.json").write_text(json.dumps(qc, indent=2))
    (OUT_DIR / "liquidity.json").write_text(json.dumps(liq, indent=2))
    print("QC done.", flush=True)


# ------------------------------------------------------- feature builder ----

def build_cell(file_path: str, asset: str, scales: list[int]):
    """Build (X, views, y, timestamps) for one (asset, cell) via the de-leaked handler."""
    handler = MultiScaleOHLCVHandler(
        file_path=file_path, ticker=asset,
        feature_config={"scales": scales, "window_size": WINDOW, "norm_span": NORM_SPAN},
    )
    base = min(scales)
    base_ts = handler._base_timestamps
    close = handler._base_close
    n = len(close)

    # TW-4: X2 causality assert — coarse bar must be CLOSED at base bar time.
    for s in scales:
        if s == base:
            continue
        idx_map = handler._scale_index_map[s]
        coarse_ts = handler._scale_timestamps[s].astype("int64")
        bts = base_ts.astype("int64")
        closes_at = coarse_ts[idx_map] + s * 60 * 1_000_000_000
        viol = int((closes_at > bts).sum())
        # first window_size bars may clip to coarse bar 0 (pre-history padding)
        first_valid = np.argmax(closes_at <= bts) if (closes_at <= bts).any() else n
        if viol and (closes_at[max(first_valid, WINDOW):] > bts[max(first_valid, WINDOW):]).any():
            raise AssertionError(f"TW-4 FAIL {asset} scale={s}: {viol} look-ahead mappings")

    # start after warmup: every scale has >= WINDOW closed coarse bars
    starts = []
    for s in scales:
        if s == base:
            starts.append(WINDOW)
        else:
            idx_map = handler._scale_index_map[s]
            ok = np.where(idx_map >= WINDOW)[0]
            starts.append(int(ok[0]) if len(ok) else n)
    t0 = max(starts)

    n_eff = n - t0 - max(HORIZONS)
    if n_eff < 5_000:
        raise RuntimeError(f"{asset}: insufficient bars after warmup ({n_eff})")

    n_scales = len(scales)
    X = np.empty((n_eff, n_scales, WINDOW, N_FEATS), dtype=np.float32)
    for si, s in enumerate(scales):
        feats = handler._scale_features[s]
        idx = (np.arange(t0, t0 + n_eff) if s == base
               else handler._scale_index_map[s][t0:t0 + n_eff])
        # gather windows [idx-29, idx]
        offs = np.arange(-(WINDOW - 1), 1)
        gather = idx[:, None] + offs[None, :]
        np.clip(gather, 0, len(feats) - 1, out=gather)
        X[:, si, :, :] = feats[gather]

    logc = np.log(np.where(close > 0, close, np.nan))
    ys = {}
    for h in HORIZONS:
        y = logc[t0 + h: t0 + h + n_eff] - logc[t0: t0 + n_eff]
        ys[h] = np.nan_to_num(y).astype(np.float32)

    ts = pd.DatetimeIndex(base_ts[t0: t0 + n_eff])
    del handler
    gc.collect()
    return X, ys, ts, base


def flatten_views(X: np.ndarray):
    """(n, 3, 30, 8) -> ridge view (n, 720) and GBM view (n, 69)."""
    n = X.shape[0]
    ridge_v = X.reshape(n, -1)
    sel = X[:, :, :, SUMMARY_FEATS]                      # (n, 3, 30, 5)
    mean = sel.mean(axis=2)                              # (n, 3, 5)
    std = sel.std(axis=2)
    last = X[:, :, -1, :]                                # (n, 3, 8)
    gbm_v = np.concatenate(
        [mean.reshape(n, -1), std.reshape(n, -1), last.reshape(n, -1)], axis=1)
    return ridge_v, gbm_v


# ------------------------------------------------------------ WF models ----

def _stride_cap(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    stride = int(np.ceil(n / cap))
    return np.arange(0, n, stride)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy(np.float64)
    rb = pd.Series(b).rank().to_numpy(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0

def wf_predict(view: np.ndarray, y: np.ndarray, ts: pd.DatetimeIndex,
               model: str, rng: np.random.RandomState):
    """Expanding-window WF predictions. Returns (pred, mask_oos, fold_id, taus)."""
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
            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, early_stopping=False,
                random_state=RNG_SEED)
            m.fit(Xtr, ytr)
            p_te = m.predict(view[te])
            p_tr = m.predict(Xtr)

        pred[te] = p_te.astype(np.float32)
        fold_id[te] = k
        taus.append(float(np.percentile(np.abs(p_tr), DEADBAND_PCTL)))
    return pred, fold_id, taus


# ----------------------------------------------------------- statistics ----

def block_bootstrap_ic(pred: np.ndarray, y: np.ndarray, ts: pd.DatetimeIndex,
                       rng: np.random.RandomState):
    """Circular day-block bootstrap of Spearman IC (rank-pearson approximation)."""
    n = len(pred)
    bars_per_day = max(int(round(86_400 / np.median(
        np.diff(ts.values).astype("timedelta64[s]").astype(np.int64)))), 1)
    block = min(bars_per_day, n)
    rp = pd.Series(pred).rank().to_numpy(np.float32)
    ry = pd.Series(y).rank().to_numpy(np.float32)
    n_blocks = int(np.ceil(n / block))
    ics = np.empty(N_BOOT, dtype=np.float64)
    done = 0
    while done < N_BOOT:
        b = min(BOOT_CHUNK, N_BOOT - done)
        starts = rng.randint(0, n, size=(b, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        idx = idx.reshape(b, -1)[:, :n]
        xs = rp[idx]; ys_ = ry[idx]
        xm = xs.mean(axis=1, keepdims=True); ym = ys_.mean(axis=1, keepdims=True)
        cov = ((xs - xm) * (ys_ - ym)).mean(axis=1)
        den = xs.std(axis=1) * ys_.std(axis=1)
        ics[done:done + b] = np.where(den > 0, cov / den, 0.0)
        done += b
    se = float(ics.std())
    return {
        "ci_lo": float(np.percentile(ics, 2.5)),
        "ci_hi": float(np.percentile(ics, 97.5)),
        "se": se,
    }


def one_sided_p(ic: float, se: float) -> float:
    from math import erf, sqrt
    if se <= 0:
        return 1.0
    z = ic / se
    return float(1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        i = order[rank_i]
        val = p[i] * m / (rank_i + 1)
        prev = min(prev, val)
        q[i] = prev
    return q.tolist()


def trade_sim(pred: np.ndarray, fold_id: np.ndarray, taus: list[float],
              y1: np.ndarray, cost_oneway: float):
    """Deadband sign strategy per spec section 6. Returns dict of PF metrics."""
    valid = fold_id >= 0
    pnl_fr = []; pnl_net = []; fold_sums: dict[int, float] = {}
    pos_prev = 0.0
    simple_ret = np.expm1(y1)
    tau_by_fold = {}
    seen = sorted(set(fold_id[valid].tolist()))
    for j, k in enumerate(seen):
        tau_by_fold[k] = taus[j] if j < len(taus) else (taus[-1] if taus else 0.0)
    idxs = np.where(valid)[0]
    for prev_i, i in zip(np.r_[-1, idxs[:-1]], idxs):
        if prev_i >= 0 and fold_id[prev_i] != fold_id[i]:
            pos_prev = 0.0  # reset at fold boundary
        tau = tau_by_fold[int(fold_id[i])]
        p = pred[i]
        pos = float(np.sign(p)) if abs(p) >= tau else 0.0
        gross = pos * simple_ret[i]
        cost = cost_oneway * abs(pos - pos_prev)
        pnl_fr.append(gross)
        pnl_net.append(gross - cost)
        fold_sums[int(fold_id[i])] = fold_sums.get(int(fold_id[i]), 0.0) + (gross - cost)
        pos_prev = pos

    def pf(arr):
        a = np.asarray(arr)
        up = a[a > 0].sum(); dn = -a[a < 0].sum()
        return float(up / dn) if dn > 0 else float("inf")

    a_net = np.asarray(pnl_net)
    return {
        "pf_frictionless": round(pf(pnl_fr), 4),
        "pf_net": round(pf(pnl_net), 4),
        "net_total_ret_pct": round(float(a_net.sum()) * 100, 2),
        "pos_folds": int(sum(1 for v in fold_sums.values() if v > 0)),
        "n_folds": len(fold_sums),
        "time_in_market": round(float(np.mean(np.asarray(pnl_fr) != 0.0)), 4),
    }


# ----------------------------------------------------------- probe stage ----

def probe_asset_cell(a: dict, cell: str, scales: list[int],
                     rng: np.random.RandomState,
                     leak_shift: bool = False, shuffle_y: bool = False):
    X, ys_, ts, base = build_cell(a["file"], a["asset"], scales)
    if leak_shift:
        # TW-1: use NEXT bar's features for the current bar's target
        X = X[1:]
        ys_ = {h: v[:-1] for h, v in ys_.items()}
        ts = ts[:-1]
    ridge_v, gbm_v = flatten_views(X)
    del X
    gc.collect()
    y1 = ys_[1]
    if shuffle_y:
        # TW-2: day-block permutation of the target
        days = pd.Series(ts.normalize())
        uniq = days.unique()
        perm = rng.permutation(len(uniq))
        mapping = {d: i for i, d in enumerate(uniq)}
        order = np.argsort([perm[mapping[d]] for d in days], kind="stable")
        y1 = y1[order]

    rows = []
    for model, view in (("ridge", ridge_v), ("gbm", gbm_v)):
        if shuffle_y and model == "gbm":
            continue  # TW-2 is ridge-only per spec implementation note
        pred, fold_id, taus = wf_predict(view, y1, ts, model, rng)
        valid = fold_id >= 0
        if valid.sum() < 1_000:
            rows.append({"asset": a["asset"], "cell": cell, "model": model,
                         "status": "INSUFFICIENT_OOS"})
            continue
        ic = _spearman(pred[valid], y1[valid])
        boot = block_bootstrap_ic(pred[valid], y1[valid], ts[valid], rng)
        p = one_sided_p(ic, boot["se"])
        sims = {}
        for label, cost in (("frictionless", 0.0), ("tier", a["cost_oneway"]),
                            ("harsh", a["cost_harsh"]),
                            ("half", a["cost_oneway"] * 0.5),
                            ("double", a["cost_oneway"] * 2.0)):
            sims[label] = trade_sim(pred, fold_id, taus, y1, cost)
        ic4 = _spearman(pred[valid], ys_[4][valid])
        rows.append({
            "asset": a["asset"], "cell": cell, "model": model, "status": "OK",
            "venue": a["venue"], "tier": a.get("tier", ""),
            "n_oos": int(valid.sum()),
            "ic_h1": round(ic, 5), "ic_h1_ci_lo": round(boot["ci_lo"], 5),
            "ic_h1_ci_hi": round(boot["ci_hi"], 5), "ic_h1_se": round(boot["se"], 6),
            "p_one_sided": p, "ic_h4": round(ic4, 5),
            "pf_frictionless": sims["frictionless"]["pf_frictionless"],
            "pf_net_tier": sims["tier"]["pf_net"],
            "pf_net_harsh": sims["harsh"]["pf_net"],
            "pf_net_half": sims["half"]["pf_net"],
            "pf_net_double": sims["double"]["pf_net"],
            "net_ret_pct_tier": sims["tier"]["net_total_ret_pct"],
            "pos_folds_tier": sims["tier"]["pos_folds"],
            "n_folds": sims["tier"]["n_folds"],
            "time_in_market": sims["tier"]["time_in_market"],
        })
    del ridge_v, gbm_v
    gc.collect()
    return rows


def run_tripwires() -> None:
    qc = json.loads((OUT_DIR / "data_qc.json").read_text())
    universe = {a["asset"]: a for a in load_universe() if a["status"] == "OK"}
    rng = np.random.RandomState(RNG_SEED)
    res: dict = {}

    btc = universe["BTCUSDT"]
    # TW-1: +1-bar feature leak must light up (ridge, 15m cell)
    print("TW-1 leak-shift on BTCUSDT 15m ...", flush=True)
    rows = probe_asset_cell(btc, "gmgp1_15m", CELLS["gmgp1_15m"], rng, leak_shift=True)
    ic_leak = [r["ic_h1"] for r in rows if r["model"] == "ridge"][0]
    res["tw1"] = {"ic_leak_shift": ic_leak, "pass": bool(ic_leak >= TW1_MIN_IC)}
    print(f"  TW-1 ic={ic_leak} pass={res['tw1']['pass']}", flush=True)

    # TW-2: day-block shuffled target -> dead (ridge, 15m, 3 representative assets)
    tw2 = {}
    for sym in ("BTCUSDT", "SOLUSDT", "GALAUSDT"):
        if sym not in universe or qc.get(sym, {}).get("status") != "CLEAN":
            continue
        print(f"TW-2 shuffle on {sym} 15m ...", flush=True)
        rows = probe_asset_cell(universe[sym], "gmgp1_15m", CELLS["gmgp1_15m"],
                                rng, shuffle_y=True)
        ic_sh = [r["ic_h1"] for r in rows if r["model"] == "ridge"][0]
        tw2[sym] = ic_sh
    res["tw2"] = {"ics": tw2, "pass": bool(all(abs(v) < 0.015 for v in tw2.values()))}
    print(f"  TW-2 {tw2} pass={res['tw2']['pass']}", flush=True)

    # TW-3 is evaluated inside the probe stage (BTC control IC), recorded here
    res["tw3"] = {"note": "asserted post-probe: BTC 15m ridge IC <= +0.01"}
    res["tw4"] = {"note": "asserted inside build_cell on every gather", "pass": True}
    (OUT_DIR / "tripwires.json").write_text(json.dumps(res, indent=2))
    if not (res["tw1"]["pass"] and res["tw2"]["pass"]):
        raise SystemExit("TRIPWIRE FAILURE — HALT (fix harness before reading results)")
    print("Tripwires TW-1/TW-2 PASS.", flush=True)


def run_probe(cell_filter: str | None = None) -> None:
    qc = json.loads((OUT_DIR / "data_qc.json").read_text())
    universe = [a for a in load_universe() if a["status"] == "OK"]
    rng = np.random.RandomState(RNG_SEED)
    cells = {k: v for k, v in CELLS.items() if cell_filter in (None, k)}
    suffix = f"_{cell_filter}" if cell_filter else ""
    out_csv = OUT_DIR / f"ic_table{suffix}.csv"
    all_rows: list[dict] = []
    for a in universe:
        if qc.get(a["asset"], {}).get("status") != "CLEAN":
            print(f"skip {a['asset']} (QC: {qc.get(a['asset'], {}).get('status')})", flush=True)
            continue
        for cell, scales in cells.items():
            print(f"probe {a['asset']} {cell} ...", flush=True)
            try:
                rows = probe_asset_cell(a, cell, scales, rng)
            except Exception as exc:  # noqa: BLE001
                rows = [{"asset": a["asset"], "cell": cell, "model": "-",
                         "status": f"ERROR: {str(exc)[:200]}"}]
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print("probe stage done.", flush=True)


def run_verdict() -> None:
    qc = json.loads((OUT_DIR / "data_qc.json").read_text())
    liq = json.loads((OUT_DIR / "liquidity.json").read_text())
    parts = [pd.read_csv(p) for p in sorted(OUT_DIR.glob("ic_table*.csv"))
             if "gated" not in p.name]
    df = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["asset", "cell", "model"], keep="last")
    ok = df[df["status"] == "OK"].copy()

    # TW-3: BTC control consistency
    btc_ic = ok[(ok.asset == "BTCUSDT") & (ok.cell == "gmgp1_15m")
                & (ok.model == "ridge")]["ic_h1"]
    tw3_pass = bool(len(btc_ic) and btc_ic.iloc[0] <= TW3_MAX_BTC_IC)

    # Gate A: FDR within primary family
    ok["q_fdr"] = bh_fdr(ok["p_one_sided"].tolist())
    ok["gate_a"] = ((ok["ic_h1"] >= GATE_A_IC) & (ok["ic_h1_ci_lo"] > 0)
                    & (ok["q_fdr"] < GATE_A_FDR_Q))
    # Gate C (evaluated for all, gated on A)
    ok["gate_c"] = ((ok["pf_net_tier"] >= GATE_C_NET_PF)
                    & (ok["pf_frictionless"] >= GATE_C_GROSS_PF)
                    & (ok["pos_folds_tier"] >= GATE_C_MIN_POS_FOLDS))
    ok["gate_ac"] = ok["gate_a"] & ok["gate_c"]

    # Gate B: dose-response within crypto venue
    rng = np.random.RandomState(RNG_SEED)
    dose: dict = {}
    cr = ok[ok["venue"] == "crypto"]
    for (cell, model), grp in cr.groupby(["cell", "model"]):
        xs = np.array([liq[a]["log10_liq"] for a in grp["asset"]])
        ys_ = grp["ic_h1"].to_numpy()
        if len(xs) < 6:
            continue
        rho = _spearman(xs, ys_)
        perms = np.array([_spearman(xs, rng.permutation(ys_)) for _ in range(10_000)])
        p_perm = float((perms <= rho).mean())  # one-sided: theory predicts rho < 0
        dose[f"{cell}|{model}"] = {"rho": round(rho, 4), "p_one_sided_neg": p_perm,
                                   "n_assets": int(len(xs)),
                                   "pass": bool(rho < 0 and p_perm < GATE_B_P)}
    gate_b_pass = any(v["pass"] for v in dose.values())

    a_set = ok[ok["gate_a"]][["asset", "cell", "model", "ic_h1", "q_fdr"]].to_dict("records")
    ac_set = ok[ok["gate_ac"]][["asset", "cell", "model", "ic_h1", "pf_net_tier",
                                "pf_frictionless"]].to_dict("records")
    untestable = [k for k, v in qc.items() if v.get("status") in ("UNTESTABLE", "CLEAN_FAIL",
                                                                  "FETCH_FAIL")]

    if ac_set:
        verdict = "GO"
    elif a_set or gate_b_pass:
        verdict = "WEAK-GO"
    else:
        verdict = "NO-GO"

    out = {
        "spec": "docs/research/r1_illiquidity_probe_spec_2026-06-12.md",
        "spec_commit": "a6336235",
        "verdict": verdict,
        "tw3_btc_control_pass": tw3_pass,
        "gate_a_passers": a_set,
        "gate_ac_passers": ac_set,
        "gate_b_dose_response": dose,
        "gate_b_pass": gate_b_pass,
        "untestable_assets": untestable,
        "n_cells_tested": int(len(ok)),
        "max_ic": float(ok["ic_h1"].max()),
        "max_ic_cell": ok.loc[ok["ic_h1"].idxmax(),
                              ["asset", "cell", "model"]].to_dict(),
        "median_ic": float(ok["ic_h1"].median()),
    }
    def _np(o):
        return o.item() if hasattr(o, "item") else str(o)

    (OUT_DIR / "verdict.json").write_text(json.dumps(out, indent=2, default=_np))
    ok.to_csv(OUT_DIR / "ic_table_gated.csv", index=False)
    (OUT_DIR / "dose_response.json").write_text(json.dumps(dose, indent=2, default=_np))
    print(json.dumps(out, indent=2, default=_np), flush=True)
    if not tw3_pass:
        print("WARNING: TW-3 failed — BTC control shows positive IC; "
              "treat verdict as INVALID until explained.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["qc", "tripwire", "probe", "verdict", "all"],
                        default="all")
    parser.add_argument("--cell", choices=list(CELLS.keys()), default=None,
                        help="probe stage: restrict to one cell (for parallel runs)")
    args = parser.parse_args()
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
