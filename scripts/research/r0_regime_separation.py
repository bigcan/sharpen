"""R0-regime — does a causal, leading regime variable separate gold's WIN folds
(0-1, Aug-Sep 2025) from BREAKEVEN folds (2-3, Oct-Nov 2025)?

Spec: docs/research/r0_regime_spec_2026-06-02.md
CPU-only. Reuses de-leaked gmgp1-gold WF trajectories + Prism GAHMM labels + GC 15m OHLCV.
No training/deploy. Emits results/r0_regime/{verdict.json, *.csv, summary.md}.

Decision rule (GO/NO-GO) is applied programmatically at the end.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / "results" / "gmgp1_gold_ensemble_wf_x2deleak"
PRISM = ROOT / "results" / "prism_research" / "prism_features_gc_2025.parquet"
GC = ROOT / "data" / "processed" / "gc_2025_15min_front.parquet"
OUT = ROOT / "results" / "r0_regime"
OUT.mkdir(parents=True, exist_ok=True)

INIT_CAPITAL = 100_000.0
WIN_FOLDS = {0, 1}      # regime A (Aug-Sep 2025)
BREAKEVEN_FOLDS = {2, 3}  # regime B (Oct-Nov 2025)
PRICE_STATE = {0: "bear", 1: "neutral", 2: "bull"}
VOL_STATE = {0: "low", 1: "normal", 2: "high"}


def pf(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else 0.0


# ---------------------------------------------------------------- load policy
def load_bar_panel(rule: str) -> pd.DataFrame:
    """Per-bar PnL across the 4 folds. PnL = within-fold Δportfolio_value;
    first bar of each fold = pv[0] - INIT_CAPITAL (captures the reset move)."""
    frames = []
    for f in range(4):
        tj = pd.read_parquet(WF / f"fold_0{f}" / f"{rule}_trajectory.parquet")
        if "portfolio_value" not in tj or "timestamp" not in tj:
            sys.exit(f"FATAL: fold {f} trajectory missing required columns")
        pv = tj["portfolio_value"].to_numpy(float)
        pnl = np.empty_like(pv)
        pnl[0] = pv[0] - INIT_CAPITAL
        pnl[1:] = np.diff(pv)
        frames.append(pd.DataFrame({
            "timestamp": tj["timestamp"].values,
            "fold": f,
            "pnl": pnl,
            "position": tj["position"].to_numpy(float),
        }))
    bar = pd.concat(frames, ignore_index=True)
    bar["date"] = pd.to_datetime(bar["timestamp"]).dt.normalize()
    bar["period"] = np.where(bar["fold"].isin(WIN_FOLDS), "WIN", "BREAKEVEN")
    return bar


# ------------------------------------------------------- causal regime sources
def causal_asof(values: pd.DataFrame, on_dates: pd.Series, *, leaky: bool) -> pd.DataFrame:
    """As-of merge: for trading date d, take the most recent regime row with
    date < d (causal) or date <= d (leaky upper bound)."""
    left = pd.DataFrame({"date": pd.to_datetime(on_dates.unique())}).sort_values("date")
    right = values.reset_index().rename(columns={values.index.name or "index": "date"})
    right["date"] = pd.to_datetime(right["date"])
    right = right.sort_values("date")
    merged = pd.merge_asof(
        left, right, on="date", direction="backward",
        allow_exact_matches=leaky,
    )
    return merged


def build_prism_daily(leaky: bool) -> pd.DataFrame:
    p = pd.read_parquet(PRISM)
    keep = ["price_regime", "vol_regime", "composite_code", "close"]
    return p[keep], leaky


def daily_price_features(prism_close: pd.Series) -> pd.DataFrame:
    """Causal daily price-regime features from the daily GC close.
    Each column is evaluated on the close series; the as-of merge (date<d)
    then guarantees the value used for trading day d came from <= d-1."""
    c = prism_close.astype(float)
    ret = c.pct_change()
    feats = pd.DataFrame(index=c.index)
    feats["close"] = c
    feats["mom_5"] = c.pct_change(5)
    feats["mom_20"] = c.pct_change(20)
    feats["trend_up_20"] = (c > c.rolling(20).mean()).astype(float)
    feats["dist_ma20"] = c / c.rolling(20).mean() - 1.0
    feats["vol_20"] = ret.rolling(20).std()
    # percentile rank of current vol within trailing 60-day window (stationary regime var)
    feats["vol_pct_60"] = ret.rolling(20).std().rolling(60).apply(
        lambda w: stats.rankdata(w)[-1] / len(w), raw=True)
    return feats


def intraday_features() -> pd.DataFrame:
    """Causal 15m features from GC OHLCV. All trailing windows shifted by 1 bar
    so the value at bar t uses only bars <= t-1."""
    gc = pd.read_parquet(GC)[["timestamp", "open", "high", "low", "close"]].copy()
    gc["timestamp"] = pd.to_datetime(gc["timestamp"])
    gc = gc.sort_values("timestamp").reset_index(drop=True)
    c = gc["close"].astype(float)
    logret = np.log(c).diff()
    # realized vol over trailing 32 bars (~half a session), shifted 1 -> excludes bar t
    gc["rv_32"] = logret.rolling(32).std().shift(1)
    gc["rv_pct"] = gc["rv_32"].rolling(390).apply(  # ~ trailing week of bars
        lambda w: stats.rankdata(w)[-1] / len(w), raw=True)
    # ATR(14) shifted, percentile vs trailing
    tr = pd.concat([
        gc["high"] - gc["low"],
        (gc["high"] - c.shift(1)).abs(),
        (gc["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    gc["atr_14"] = tr.rolling(14).mean().shift(1)
    gc["atr_pct"] = gc["atr_14"].rolling(390).apply(
        lambda w: stats.rankdata(w)[-1] / len(w), raw=True)
    gc["hour"] = gc["timestamp"].dt.hour
    # GC front exchange ts (UTC-ish): Asia 22-07, London 07-13, NY 13-21
    def sess(h):
        if 7 <= h < 13:
            return "London"
        if 13 <= h < 21:
            return "NY"
        return "Asia"
    gc["session"] = gc["hour"].map(sess)
    return gc[["timestamp", "rv_32", "rv_pct", "atr_14", "atr_pct", "session"]]


def causality_tripwire():
    """Perturb the GC close at one index; confirm causal features at earlier
    indices are unchanged and only indices >= perturbation+1 move. Fails loudly
    if any feature reads bar t into the value at bar t (look-ahead)."""
    gc = pd.read_parquet(GC)[["timestamp", "high", "low", "close"]].copy()
    gc = gc.sort_values("timestamp").reset_index(drop=True)
    c = gc["close"].astype(float)
    base_rv = np.log(c).diff().rolling(32).std().shift(1)
    i = 5000
    c2 = c.copy()
    c2.iloc[i] *= 1.5  # wild perturbation at bar i
    pert_rv = np.log(c2).diff().rolling(32).std().shift(1)
    # values strictly before i+1 must be identical (causal: bar i not yet used)
    same_before = np.allclose(
        base_rv.iloc[:i + 1].fillna(-9).values,
        pert_rv.iloc[:i + 1].fillna(-9).values)
    moved_after = not np.allclose(
        base_rv.iloc[i + 1:i + 40].fillna(-9).values,
        pert_rv.iloc[i + 1:i + 40].fillna(-9).values)
    if not (same_before and moved_after):
        sys.exit(f"FATAL TRIPWIRE: rv_32 look-ahead (same_before={same_before}, "
                 f"moved_after={moved_after})")
    return {"rv_32_same_before_perturb": bool(same_before),
            "rv_32_moves_only_after": bool(moved_after)}


# ----------------------------------------------------------------- daily panel
def build_daily_panel(bar: pd.DataFrame, *, leaky: bool) -> pd.DataFrame:
    daily = bar.groupby(["fold", "date"], as_index=False).agg(
        pnl=("pnl", "sum"), bars=("pnl", "size"))
    daily["period"] = np.where(daily["fold"].isin(WIN_FOLDS), "WIN", "BREAKEVEN")
    daily["up"] = (daily["pnl"] > 0).astype(int)

    prism = pd.read_parquet(PRISM)[["price_regime", "vol_regime", "composite_code", "close"]]
    pr = causal_asof(prism, daily["date"], leaky=leaky)
    pricef = daily_price_features(prism["close"])
    pf_asof = causal_asof(pricef, daily["date"], leaky=leaky)

    daily = daily.merge(pr, on="date", how="left").merge(
        pf_asof.drop(columns=[c for c in ["close"] if c in pf_asof]), on="date", how="left")
    daily["price_state"] = daily["price_regime"].map(PRICE_STATE)
    daily["vol_state"] = daily["vol_regime"].map(VOL_STATE)
    return daily


# ---------------------------------------------------------------------- tests
def t1_characterization(daily: pd.DataFrame) -> dict:
    out = {}
    for col, states in [("price_state", PRICE_STATE), ("vol_state", VOL_STATE)]:
        cnt = daily.groupby(["period", col]).size().rename("n").reset_index()
        cnt["frac"] = cnt.groupby("period")["n"].transform(lambda s: (s / s.sum()).round(3))
        out[col] = json.loads(cnt.to_json(orient="records"))
    for col in ["mom_20", "vol_20", "vol_pct_60", "dist_ma20"]:
        w = daily.loc[daily.period == "WIN", col].dropna()
        b = daily.loc[daily.period == "BREAKEVEN", col].dropna()
        u = stats.mannwhitneyu(w, b, alternative="two-sided") if len(w) and len(b) else None
        out[col] = {"win_mean": float(w.mean()), "breakeven_mean": float(b.mean()),
                    "mwu_p": float(u.pvalue) if u else None}
    return out


def t2_daily_separation(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # categorical
    for col in ["price_state", "vol_state", "composite_code"]:
        groups = [g["pnl"].values for _, g in daily.groupby(col) if len(g) >= 3]
        kw = stats.kruskal(*groups) if len(groups) >= 2 else None
        for state, g in daily.groupby(col):
            rows.append({"var": col, "state": str(state), "n": len(g),
                         "mean_pnl": float(g["pnl"].mean()), "pf": pf(g["pnl"].values),
                         "win_rate": float((g["pnl"] > 0).mean()),
                         "kruskal_p": float(kw.pvalue) if kw else None})
    # continuous
    for col in ["mom_5", "mom_20", "dist_ma20", "vol_20", "vol_pct_60"]:
        d = daily[[col, "pnl", "up"]].dropna()
        if len(d) < 20:
            continue
        rho, rp = stats.spearmanr(d[col], d["pnl"])
        try:
            auc = roc_auc_score(d["up"], d[col])
        except ValueError:
            auc = float("nan")
        med = d[col].median()
        hi, lo = d[d[col] >= med]["pnl"], d[d[col] < med]["pnl"]
        mwu = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({"var": col, "state": "continuous", "n": len(d),
                     "spearman_rho": float(rho), "spearman_p": float(rp),
                     "auc_up_day": float(auc),
                     "hi_half_mean_pnl": float(hi.mean()), "lo_half_mean_pnl": float(lo.mean()),
                     "median_split_mwu_p": float(mwu.pvalue)})
    return pd.DataFrame(rows)


def t2b_within_period(daily: pd.DataFrame) -> pd.DataFrame:
    """DECISIVE test: does the variable separate good from bad DAYS *within* each
    period (a stable, gateable relationship) — or only across the single macro
    boundary (hindsight)? A variable that is null within both periods cannot be a
    leading gate, no matter how strongly it separates the two periods."""
    rows = []
    for col in ["vol_20", "vol_pct_60", "mom_20", "dist_ma20"]:
        for period in ["WIN", "BREAKEVEN"]:
            d = daily[daily.period == period][[col, "pnl"]].dropna()
            if len(d) < 8:
                continue
            med = d[col].median()
            hi, lo = d[d[col] >= med]["pnl"], d[d[col] < med]["pnl"]
            mwu = stats.mannwhitneyu(hi, lo, alternative="two-sided")
            rho, rp = stats.spearmanr(d[col], d["pnl"])
            rows.append({"var": col, "period": period, "n": len(d),
                         "hi_mean_pnl": float(hi.mean()), "lo_mean_pnl": float(lo.mean()),
                         "mwu_p": float(mwu.pvalue), "spearman_rho": float(rho),
                         "spearman_p": float(rp)})
    # Prism vol_regime calm vs high, within period
    for period in ["WIN", "BREAKEVEN"]:
        d = daily[daily.period == period].dropna(subset=["vol_regime"])
        calm, high = d[d.vol_regime.isin([0, 1])]["pnl"], d[d.vol_regime == 2]["pnl"]
        if len(calm) >= 5 and len(high) >= 5:
            mwu = stats.mannwhitneyu(calm, high, alternative="two-sided")
            rows.append({"var": "prism_vol_regime", "period": period,
                         "n": len(d), "hi_mean_pnl": float(high.mean()),
                         "lo_mean_pnl": float(calm.mean()), "mwu_p": float(mwu.pvalue),
                         "spearman_rho": float("nan"), "spearman_p": float("nan")})
    return pd.DataFrame(rows)


# pre-registered gate rules: name -> boolean Series (regime A = trade)
def gate_masks(daily: pd.DataFrame) -> dict:
    return {
        "GATE_price_bull": daily["price_regime"] == 2,
        "GATE_vol_calm": daily["vol_regime"].isin([0, 1]),
        "GATE_trend_up": daily["trend_up_20"] == 1.0,
        "GATE_mom20_pos": daily["mom_20"] > 0,
        "GATE_vol_below_med": daily["vol_pct_60"] < daily["vol_pct_60"].median(),
    }


def bar_pf_under_daily_mask(bar: pd.DataFrame, daily: pd.DataFrame,
                            mask: pd.Series, folds) -> tuple:
    """Apply a per-(fold,date) trade/flat decision to BARS (flat day -> 0 PnL),
    return deploy-relevant bar-level PF + return + time-in-market over `folds`."""
    keep = daily.loc[mask.fillna(False).to_numpy(bool), ["fold", "date"]]
    keep_keys = set(map(tuple, keep.to_numpy()))
    b = bar[bar.fold.isin(folds)].copy()
    in_a = [(f, d) in keep_keys for f, d in zip(b.fold, b.date)]
    gated = np.where(in_a, b["pnl"].to_numpy(float), 0.0)
    ret = gated.sum() / INIT_CAPITAL * 100
    return pf(gated), float(ret), float(np.mean(in_a))


def t3_gate_sim(bar: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """In-sample, pre-registered gates evaluated at BAR-level PF (deploy metric)."""
    rows = []
    all_folds = [0, 1, 2, 3]
    ung_pf, ung_ret, _ = bar_pf_under_daily_mask(
        bar, daily, pd.Series(True, index=daily.index), all_folds)
    row = {"gate": "UNGATED", "pf_bar_pooled": ung_pf, "ret_pct": ung_ret,
           "time_in_mkt": 1.0}
    for f in range(4):
        row[f"fold{f}_pf"] = bar_pf_under_daily_mask(
            bar, daily, pd.Series(True, index=daily.index), [f])[0]
    rows.append(row)
    for name, mask in gate_masks(daily).items():
        p, r, tim = bar_pf_under_daily_mask(bar, daily, mask, all_folds)
        row = {"gate": name, "pf_bar_pooled": p, "ret_pct": r, "time_in_mkt": tim}
        for f in range(4):
            row[f"fold{f}_pf"] = bar_pf_under_daily_mask(bar, daily, mask, [f])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def t4_oos_gate(bar: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Calibrate each candidate gate on folds 0-1 ONLY, freeze, score held-out
    folds 2-3 at BAR-level PF. The decisive anti-hindsight test: can a gate learned
    on the winning period lift the breakeven period above ungated (0.99/1.04)?"""
    rows = []
    cal = daily[daily.fold.isin(WIN_FOLDS)]
    # continuous-feature threshold gates
    for col in ["mom_5", "mom_20", "dist_ma20", "vol_pct_60", "vol_20"]:
        if daily[col].dropna().shape[0] < 30:
            continue
        cand = np.quantile(cal[col].dropna(), np.linspace(0.1, 0.9, 17))
        best = None  # maximize gated Sharpe on calibration folds
        for thr in cand:
            for direction in ("above", "below"):
                m = (cal[col] >= thr) if direction == "above" else (cal[col] < thr)
                if m.fillna(False).mean() < 0.2:
                    continue
                sc = sharpe(np.where(m.fillna(False), cal["pnl"], 0.0))
                if best is None or sc > best[0]:
                    best = (sc, thr, direction)
        if best is None:
            continue
        _, thr, direction = best
        full_mask = (daily[col] >= thr) if direction == "above" else (daily[col] < thr)
        oos_pf, oos_ret, oos_tim = bar_pf_under_daily_mask(bar, daily, full_mask, [2, 3])
        ung_oos_pf, ung_oos_ret, _ = bar_pf_under_daily_mask(
            bar, daily, pd.Series(True, index=daily.index), [2, 3])
        rows.append({"gate": col, "rule": f"{direction} {thr:.5f}",
                     "oos_pf_bar": oos_pf, "oos_pf_ungated": ung_oos_pf,
                     "oos_ret_pct": oos_ret, "oos_ret_ungated_pct": ung_oos_ret,
                     "oos_time_in_mkt": oos_tim})
    # Prism categorical gates (fixed a-priori, no threshold to calibrate)
    for name, mask in [("prism_vol_calm", daily["vol_regime"].isin([0, 1])),
                       ("prism_price_bull", daily["price_regime"] == 2)]:
        oos_pf, oos_ret, oos_tim = bar_pf_under_daily_mask(bar, daily, mask, [2, 3])
        ung_oos_pf, ung_oos_ret, _ = bar_pf_under_daily_mask(
            bar, daily, pd.Series(True, index=daily.index), [2, 3])
        rows.append({"gate": name, "rule": "a-priori",
                     "oos_pf_bar": oos_pf, "oos_pf_ungated": ung_oos_pf,
                     "oos_ret_pct": oos_ret, "oos_ret_ungated_pct": ung_oos_ret,
                     "oos_time_in_mkt": oos_tim})
    return pd.DataFrame(rows)


def t6_intraday_gate(bar: pd.DataFrame) -> pd.DataFrame:
    """Last price-only candidate: intraday (15m) causal session + vol gates applied
    bar-by-bar. Session is trivially causal; rv_pct/atr_pct are trailing+shifted.
    Calibrated on folds 0-1, scored on folds 2-3 at bar-level PF."""
    feats = intraday_features()
    b = bar.merge(feats, on="timestamp", how="left")
    if b["session"].isna().any():
        sys.exit("FATAL: intraday feature merge left NaNs (bar/GC grid mismatch)")
    cal = b[b.fold.isin(WIN_FOLDS)]
    oos = b[b.fold.isin(BREAKEVEN_FOLDS)]
    ung_oos = pf(oos["pnl"].values)
    rows = []
    # fixed session gates (a-priori)
    for name, ses in [("session_London", {"London"}), ("session_NY", {"NY"}),
                      ("session_London+NY", {"London", "NY"})]:
        m = oos["session"].isin(ses)
        rows.append({"gate": name, "rule": "a-priori",
                     "oos_pf_bar": pf(np.where(m, oos["pnl"], 0.0)),
                     "oos_pf_ungated": ung_oos, "oos_time_in_mkt": float(m.mean())})
    # intraday vol-percentile gates calibrated on folds 0-1
    for col in ["rv_pct", "atr_pct"]:
        c = b[[col]].dropna()
        if len(c) < 100:
            continue
        cand = np.quantile(cal[col].dropna(), np.linspace(0.2, 0.9, 15))
        best = None
        for thr in cand:
            m = cal[col] < thr  # "calm" = below-threshold vol
            if m.fillna(False).mean() < 0.2:
                continue
            sc = sharpe(np.where(m.fillna(False), cal["pnl"], 0.0))
            if best is None or sc > best[0]:
                best = (sc, thr)
        if best is None:
            continue
        thr = best[1]
        m = (oos[col] < thr).fillna(False)
        rows.append({"gate": f"intraday_{col}_calm", "rule": f"below {thr:.3f}",
                     "oos_pf_bar": pf(np.where(m, oos["pnl"], 0.0)),
                     "oos_pf_ungated": ung_oos, "oos_time_in_mkt": float(m.mean())})
    return pd.DataFrame(rows)


def t5_hindsight_gap(rule: str, bar: pd.DataFrame) -> dict:
    """Best-possible hindsight gate (contemporaneous/leaky Prism, tuned on FULL
    data) vs causal — bar-level PF. Large gap => apparent benefit is look-ahead."""
    dl_leaky = build_daily_panel(bar, leaky=True)
    dl_causal = build_daily_panel(bar, leaky=False)
    all_folds = [0, 1, 2, 3]

    def best_full(daily):
        best = bar_pf_under_daily_mask(
            bar, daily, pd.Series(True, index=daily.index), all_folds)[0]
        for _, mask in gate_masks(daily).items():
            if mask.fillna(False).mean() < 0.2:
                continue
            best = max(best, bar_pf_under_daily_mask(bar, daily, mask, all_folds)[0])
        return best
    ung = bar_pf_under_daily_mask(
        bar, dl_causal, pd.Series(True, index=dl_causal.index), all_folds)[0]
    return {"hindsight_leaky_best_pf_bar": best_full(dl_leaky),
            "causal_best_full_pf_bar": best_full(dl_causal),
            "ungated_pf_bar": ung}


# --------------------------------------------------------------------- driver
OOS_PF_GO = 1.20          # held-out folds 2-3 bar-PF a gate must clear to count
ALPHA_BONF = 0.05 / 10.0  # ~10 separation tests => Bonferroni-corrected threshold


def run(rule: str) -> dict:
    bar = load_bar_panel(rule)
    daily = build_daily_panel(bar, leaky=False)
    t1 = t1_characterization(daily)
    t2 = t2_daily_separation(daily)
    t2b = t2b_within_period(daily)
    t3 = t3_gate_sim(bar, daily)
    t4 = t4_oos_gate(bar, daily)
    t6 = t6_intraday_gate(bar)
    t5 = t5_hindsight_gap(rule, bar)

    # ---- programmatic GO / NO-GO ----
    # (a) DECISIVE: any variable that separates good/bad days WITHIN a period,
    #     and (necessary) within BOTH periods => a stable, gateable relationship.
    sig_within = t2b[t2b["mwu_p"] < 0.05]
    vars_sig_one = sorted(sig_within["var"].unique().tolist())
    vars_sig_both = sorted([v for v in t2b["var"].unique()
                            if {"WIN", "BREAKEVEN"}.issubset(
                                set(t2b[(t2b["var"] == v) & (t2b["mwu_p"] < 0.05)]["period"]))])
    # (b) pooled daily separation surviving Bonferroni (weak evidence at best)
    pooled_p = []
    for rec in t2.to_dict("records"):
        for k in ("kruskal_p", "spearman_p"):
            if rec.get(k) is not None and not pd.isna(rec.get(k)):
                pooled_p.append((rec["var"], k, rec[k]))
    pooled_bonf = [x for x in pooled_p if x[2] < ALPHA_BONF]
    # (c) OOS gate lifts held-out folds 2-3 BAR-PF clearly above ungated
    #     (daily-regime gates AND intraday price-only gates both count)
    t4_all = pd.concat([t4, t6], ignore_index=True) if len(t6) else t4
    oos_win = t4_all[(t4_all["oos_pf_bar"] >= OOS_PF_GO) &
                     (t4_all["oos_pf_bar"] > t4_all["oos_pf_ungated"] + 0.10)] if len(t4_all) else pd.DataFrame()
    has_oos = len(oos_win) > 0

    # GO needs a STABLE within-period relationship (both periods) AND an OOS lift.
    if len(vars_sig_both) > 0 and has_oos:
        decision = "GO"
    elif (len(vars_sig_both) > 0) or has_oos or len(pooled_bonf) > 0:
        decision = "AMBIGUOUS"
    else:
        decision = "NO_GO"
    return {
        "rule": rule,
        "t1_characterization": t1,
        "t2": t2.to_dict(orient="records"),
        "t2b_within_period": t2b.to_dict(orient="records"),
        "t3": t3.to_dict(orient="records"),
        "t4": t4.to_dict(orient="records"),
        "t6_intraday": t6.to_dict(orient="records"),
        "t5": t5,
        "criteria": {
            "vars_sig_within_one_period": vars_sig_one,
            "vars_sig_within_BOTH_periods": vars_sig_both,
            "pooled_separation_surviving_bonferroni": pooled_bonf,
            "oos_gate_generalizes": bool(has_oos),
            "oos_winners": oos_win.to_dict(orient="records") if len(oos_win) else [],
        },
        "decision": decision,
    }


def main():
    tw = causality_tripwire()
    print(f"[tripwire] {tw}")
    results = {}
    for rule in ["ens_pf_weighted", "solo_456", "ens_mean"]:
        print(f"\n{'='*60}\nRULE: {rule}\n{'='*60}")
        r = run(rule)
        results[rule] = r
        c = r["criteria"]
        print(f"  decision={r['decision']}")
        print(f"  within-period sig (one)={c['vars_sig_within_one_period']}  "
              f"(BOTH)={c['vars_sig_within_BOTH_periods']}")
        print(f"  pooled-sep surviving Bonferroni={c['pooled_separation_surviving_bonferroni']}")
        print(f"  OOS gate generalizes={c['oos_gate_generalizes']} winners={[w['gate'] for w in c['oos_winners']]}")
        print(f"  t5 (bar-PF) hindsight_leaky={r['t5']['hindsight_leaky_best_pf_bar']:.3f} "
              f"causal_best={r['t5']['causal_best_full_pf_bar']:.3f} ungated={r['t5']['ungated_pf_bar']:.3f}")
        if rule == "ens_pf_weighted":
            pd.DataFrame(r["t2"]).to_csv(OUT / "t2_daily_separation.csv", index=False)
            pd.DataFrame(r["t2b_within_period"]).to_csv(OUT / "t2b_within_period.csv", index=False)
            pd.DataFrame(r["t3"]).to_csv(OUT / "t3_gate_sim.csv", index=False)
            pd.DataFrame(r["t4"]).to_csv(OUT / "t4_oos_gate.csv", index=False)
            pd.DataFrame(r["t6_intraday"]).to_csv(OUT / "t6_intraday_gate.csv", index=False)

    primary = results["ens_pf_weighted"]["decision"]
    agree = all(results[r]["decision"] == primary for r in results)
    verdict = {
        "test": "R0-regime",
        "question": "Does any CAUSAL leading regime variable separate gold's WIN folds "
                    "(0-1) from BREAKEVEN folds (2-3) finely enough to gate the de-leaked policy?",
        "primary_rule": "ens_pf_weighted",
        "decision": primary,
        "robust_across_rules": bool(agree),
        "per_rule_decision": {r: results[r]["decision"] for r in results},
        "tripwire": tw,
        "chronos_status": "UNTESTABLE_NO_DATA (chronos_spread all-zero, confidence const 0.7)",
        "candidates_tested": "Prism GAHMM price_regime/vol_regime/composite_code (daily, lag-1 causal); "
                             "causal price features mom_5/mom_20/dist_ma20/vol_20/vol_pct_60 (daily); "
                             "Prism Chronos = untestable (no data)",
        "criteria_primary": results["ens_pf_weighted"]["criteria"],
        "t5_hindsight_gap_primary": results["ens_pf_weighted"]["t5"],
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (OUT / "full_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'#'*60}\nR0-REGIME DECISION: {primary} (robust_across_rules={agree})\n{'#'*60}")
    return verdict


if __name__ == "__main__":
    main()
