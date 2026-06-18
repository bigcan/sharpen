"""Cross-asset VALUE falsification (the missing uncorrelated leg).

Pre-registered spec: docs/research/value_falsification_spec_2026-06-18.md
Research artifact:    .agent/artifacts/cross_asset_value_sleeve_research.md

The cheap, CPU-only, linear falsification the redesign protocol requires BEFORE any
RL/env work: does a cross-asset long-horizon-reversal VALUE book (AMP-2013 "value"
leg) (a) survive realistic costs, and (b) ADD to the validated TSMOM momentum core
via its negative correlation? Mirrors xsec_momentum_falsification.py verbatim in
construction/costs/metrics so the two books are directly comparable.

Value signal (price-only, causal): value_raw(i,t) = logP(t-1260) - logP(t-252)
  = -(cumulative return over the (t-5y, t-1y) window); positive => asset FELL => cheap
  => LONG. The 1-year skip makes the value window non-overlapping with the entire
  momentum window [t-252, t-5] => orthogonal-to-momentum by construction.

LEAK-2: weights computed at rebalance t from data <= t, applied to returns from t+1.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MOM_CACHE = ROOT / "results" / "xsec_momentum" / "prices_daily.parquet"
OUT = ROOT / "results" / "value_falsification"
OUT.mkdir(parents=True, exist_ok=True)

# Universe identical to the momentum core (apples-to-apples combination).
UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
ALL_TICKERS = [t for v in UNIVERSE.values() for t in v]

TARGET_VOL_ASSET = 0.10
TARGET_VOL_PORT = 0.10
LEV_CAP = 2.0
VOL_WIN = 63
# momentum (for the same-window baseline) — identical to xsec_momentum_falsification
LOOKBACKS = [63, 126, 252]
SKIP = 5
# value (pre-registered): 5y long, 1y skip
L_LONG = 1260
L_SKIP = 252
COST_MODELS = {"frictionless": 0.0, "standard_2bps": 0.0002, "harsh_10bps": 0.0010}
ANN = 252
SUBPERIODS = {  # pre-registered (combined book persistence + post-2017 decay stress)
    "2011-13": ("2011-01-01", "2013-12-31"),
    "2014-16": ("2014-01-01", "2016-12-31"),
    "2017-20": ("2017-01-01", "2020-12-31"),
    "2021-26": ("2021-01-01", "2026-12-31"),
}
POST2017 = ["2017-20", "2021-26"]


# ----------------------------- metrics (verbatim) -----------------------------
def sharpe(daily: pd.Series) -> float:
    d = daily.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(ANN)) if s > 0 else 0.0


def pf(daily: pd.Series) -> float:
    d = daily.dropna().to_numpy()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def max_dd(daily: pd.Series) -> float:
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def vol_norm(daily: pd.Series) -> pd.Series:
    """Single constant scalar to TARGET_VOL_PORT (Sharpe-invariant)."""
    d = daily.dropna()
    raw_vol = float(d.std() * np.sqrt(ANN))
    k = (TARGET_VOL_PORT / raw_vol) if raw_vol > 0 else 1.0
    return d * k


def metrics(daily: pd.Series, turnover_ann: float, bench: pd.Series) -> dict:
    d = daily.dropna()
    raw_vol = float(d.std() * np.sqrt(ANN))
    dn = vol_norm(d)
    return {
        "sharpe": round(sharpe(d), 3),
        "pf": round(pf(d), 3),
        "ann_ret_pct@10vol": round(float(dn.mean() * ANN * 100), 2),
        "max_dd_pct@10vol": round(max_dd(dn) * 100, 2),
        "raw_ann_vol_pct": round(raw_vol * 100, 1),
        "turnover_ann": round(turnover_ann, 1),
        "corr_SPY": round(float(d.corr(bench.reindex(d.index))), 3),
        "skew": round(float(d.skew()), 3),
        "cvar95_pct": round(float(d[d <= d.quantile(0.05)].mean() * 100), 3),
        "n_days": int(len(d)),
    }


def last_trading_of_period(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    key = index.to_period("W" if freq == "weekly" else "M")
    last = pd.Series(index, index=index).groupby(key).max().values
    return pd.DatetimeIndex(last)


# ----------------------------- signals -----------------------------
def tsmom_signal(close: pd.DataFrame, rebal: pd.DatetimeIndex) -> pd.DataFrame:
    """Momentum (verbatim from xsec_momentum_falsification): sign-ensemble trend."""
    sig = pd.DataFrame(index=rebal, columns=close.columns, dtype=float)
    for dt in rebal:
        upto = close.loc[:dt]
        if len(upto) < max(LOOKBACKS) + SKIP + 5:
            continue
        ref = upto.iloc[-1 - SKIP]
        scores = []
        for L in LOOKBACKS:
            if len(upto) < L + SKIP + 1:
                continue
            past = upto.iloc[-1 - SKIP - L]
            scores.append(np.sign(ref / past - 1.0))
        if scores:
            sig.loc[dt] = np.nanmean(scores, axis=0)
    return sig


def value_raw_row(upto: pd.DataFrame) -> pd.Series | None:
    """value_raw = logP(t-L_LONG) - logP(t-L_SKIP); positive = fell = cheap = long."""
    if len(upto) < L_LONG + 2:
        return None
    p_long = upto.iloc[-1 - L_LONG]
    p_skip = upto.iloc[-1 - L_SKIP]
    return np.log(p_long) - np.log(p_skip)


def vol_scaled_weights(sig: pd.DataFrame, rets: pd.DataFrame,
                       rebal: pd.DatetimeIndex) -> pd.DataFrame:
    realized = rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(index=rebal, columns=sig.columns, dtype=float)
    for dt in rebal:
        rv = realized.loc[:dt]
        if rv.empty:
            continue
        rv_t = rv.iloc[-1]
        scale = (TARGET_VOL_ASSET / rv_t).clip(upper=LEV_CAP)
        w.loc[dt] = (sig.loc[dt] * scale).clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


def xs_value_weights(close: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex,
                     tickers: list) -> pd.DataFrame:
    """Cross-sectional value WITHIN class: rank by value_raw, long top third (cheapest,
    fell most), short bottom third (most expensive), vol-scaled. Mirrors xsmom_weights."""
    realized = rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(index=rebal, columns=tickers, dtype=float)
    for dt in rebal:
        upto = close.loc[:dt, tickers]
        vraw = value_raw_row(upto)
        if vraw is None:
            continue
        vraw = vraw.dropna()
        if len(vraw) < 3:
            continue
        ranks = vraw.rank()
        n = len(vraw)
        k = max(1, n // 3)
        longs = ranks.nlargest(k).index       # highest value_raw = cheapest
        shorts = ranks.nsmallest(k).index
        rv_t = realized.loc[:dt].iloc[-1]
        row = pd.Series(0.0, index=tickers)
        for tk in longs:
            row[tk] = (TARGET_VOL_ASSET / rv_t[tk]) if rv_t[tk] > 0 else 0.0
        for tk in shorts:
            row[tk] = -(TARGET_VOL_ASSET / rv_t[tk]) if rv_t[tk] > 0 else 0.0
        w.loc[dt] = row.clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


def ts_value_weights(close: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex,
                     tickers: list) -> pd.DataFrame:
    """Time-series value: sign(value_raw - class_mean(value_raw)), vol-scaled
    (demeaned within class => relative, avoids secular-trend short-bleed)."""
    realized = rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(index=rebal, columns=tickers, dtype=float)
    for dt in rebal:
        upto = close.loc[:dt, tickers]
        vraw = value_raw_row(upto)
        if vraw is None:
            continue
        vraw = vraw.dropna()
        if len(vraw) < 3:
            continue
        sig = np.sign(vraw - vraw.mean())
        rv_t = realized.loc[:dt].iloc[-1]
        row = pd.Series(0.0, index=tickers)
        for tk in vraw.index:
            row[tk] = (sig[tk] * TARGET_VOL_ASSET / rv_t[tk]) if rv_t[tk] > 0 else 0.0
        w.loc[dt] = row.clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


# ----------------------------- backtest (verbatim + lag param) -----------------------------
def backtest(weights_rebal: pd.DataFrame, rets: pd.DataFrame, lag: int = 1) -> tuple:
    daily_idx = rets.index
    w_daily = weights_rebal.reindex(daily_idx).ffill().fillna(0.0)
    w_eff = w_daily.shift(lag).fillna(0.0)             # lag=1 causal; lag=0 same-day (leak probe)
    gross = (w_eff * rets).sum(axis=1)
    dw = weights_rebal.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = weights_rebal.iloc[0].abs().sum()
    years = (daily_idx[-1] - daily_idx[0]).days / 365.25
    turnover_ann = float(dw.sum() / years)
    dw_daily = dw.reindex(daily_idx).fillna(0.0)
    cost_daily = {name: dw_daily * c for name, c in COST_MODELS.items()}
    return gross, cost_daily, turnover_ann


def run_book(name: str, weights_rebal: pd.DataFrame, rets: pd.DataFrame,
             bench: pd.Series, lag: int = 1) -> dict:
    gross, cost_daily, turn = backtest(weights_rebal, rets, lag=lag)
    out = {}
    net_std = None
    for cm in COST_MODELS:
        net = gross - cost_daily[cm]
        out[cm] = metrics(net, turn, bench)
        if cm == "standard_2bps":
            net_std = net
    return {"book": name, "turnover_ann": round(turn, 1), "by_cost": out,
            "_net_standard": net_std}


# ----------------------------- IC / leak diagnostics -----------------------------
def pooled_rank_ic(close: pd.DataFrame, rebal: pd.DatetimeIndex,
                   horizon: int = 21) -> tuple:
    """Pooled Spearman rank-IC between value_raw and the forward `horizon`-day return,
    plus a shuffle-null distribution. value_raw>0 (cheap) should predict positive fwd
    return => positive IC if the reversal premium is real."""
    sig_vals, fwd_vals = [], []
    px = close
    for dt in rebal:
        upto = px.loc[:dt]
        vraw = value_raw_row(upto)
        if vraw is None:
            continue
        pos = px.index.get_loc(dt)
        if pos + horizon >= len(px.index):
            continue
        fwd = px.iloc[pos + horizon] / px.iloc[pos] - 1.0
        for tk in vraw.index:
            if np.isfinite(vraw[tk]) and np.isfinite(fwd[tk]):
                sig_vals.append(vraw[tk])
                fwd_vals.append(fwd[tk])
    s = pd.Series(sig_vals)
    f = pd.Series(fwd_vals)
    ic = float(s.rank().corr(f.rank()))
    # shuffle null
    rng = np.random.default_rng(7)
    null = []
    fr = f.rank().to_numpy()
    sr = s.rank().to_numpy()
    for _ in range(500):
        perm = rng.permutation(fr)
        c = np.corrcoef(sr, perm)[0, 1]
        null.append(c)
    null = np.array(null)
    return ic, float(np.quantile(np.abs(null), 0.95)), int(len(s))


def risk_parity_combine(mom_net: pd.Series, val_net: pd.Series) -> pd.Series:
    """Each sleeve scaled to 10% vol over the common window, equal-weight (= risk parity
    for 2 sleeves), then summed. Sharpe is scale-invariant."""
    idx = mom_net.dropna().index.intersection(val_net.dropna().index)
    m, v = mom_net.reindex(idx), val_net.reindex(idx)
    km = TARGET_VOL_PORT / (m.std() * np.sqrt(ANN))
    kv = TARGET_VOL_PORT / (v.std() * np.sqrt(ANN))
    return 0.5 * km * m + 0.5 * kv * v


def main():
    if not MOM_CACHE.exists():
        raise FileNotFoundError(f"price cache missing: {MOM_CACHE} (run xsec_momentum first)")
    close = pd.read_parquet(MOM_CACHE)
    close = close[[t for t in ALL_TICKERS if t in close.columns]].sort_index()
    missing = [t for t in ALL_TICKERS if t not in close.columns]
    if missing:
        raise KeyError(f"universe tickers missing from cache: {missing}")
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    print(f"Loaded {close.shape[1]} tickers, {close.shape[0]} days, "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    results = {"spec": "docs/research/value_falsification_spec_2026-06-18.md",
               "params": {"L_long": L_LONG, "L_skip": L_SKIP, "vol_win": VOL_WIN,
                          "lookbacks_mom": LOOKBACKS, "lev_cap": LEV_CAP,
                          "cost_models": COST_MODELS}, "books": {}}

    rebal = last_trading_of_period(close.index, "monthly")
    rebal_val = rebal[rebal >= close.index[L_LONG + VOL_WIN]]   # value needs 5y history
    rebal_mom = rebal[rebal >= close.index[max(LOOKBACKS) + SKIP + VOL_WIN]]

    # ---- momentum pooled (full + same-window) ----
    sig_mom = tsmom_signal(close, rebal_mom)
    w_mom = vol_scaled_weights(sig_mom, rets, rebal_mom)
    book_mom = run_book("TSMOM_pooled_monthly", w_mom, rets, bench)
    results["books"]["TSMOM_pooled_monthly"] = {k: v for k, v in book_mom.items()
                                                if k != "_net_standard"}

    # ---- XS-VALUE per class + pooled ----
    xs_books = []
    for cls, tickers in UNIVERSE.items():
        tk = [t for t in tickers if t in close.columns]
        w_cls = xs_value_weights(close, rets, rebal_val, tk).reindex(
            columns=close.columns).fillna(0.0)
        b = run_book(f"XSVALUE_{cls}_monthly", w_cls, rets, bench)
        results["books"][f"XSVALUE_{cls}_monthly"] = {k: v for k, v in b.items()
                                                      if k != "_net_standard"}
        xs_books.append(w_cls)
    w_xsval = sum(xs_books)
    book_xsval = run_book("XSVALUE_pooled_monthly", w_xsval, rets, bench)
    results["books"]["XSVALUE_pooled_monthly"] = {k: v for k, v in book_xsval.items()
                                                  if k != "_net_standard"}

    # ---- TS-VALUE pooled (secondary) ----
    ts_books = []
    for cls, tickers in UNIVERSE.items():
        tk = [t for t in tickers if t in close.columns]
        ts_books.append(ts_value_weights(close, rets, rebal_val, tk).reindex(
            columns=close.columns).fillna(0.0))
    w_tsval = sum(ts_books)
    book_tsval = run_book("TSVALUE_pooled_monthly", w_tsval, rets, bench)
    results["books"]["TSVALUE_pooled_monthly"] = {k: v for k, v in book_tsval.items()
                                                  if k != "_net_standard"}

    # ---- leak probe: same-day vs causal XS-VALUE ----
    book_xsval_leak = run_book("XSVALUE_pooled_monthly_SAMEDAY", w_xsval, rets, bench, lag=0)
    leak_gap = round(book_xsval_leak["by_cost"]["standard_2bps"]["sharpe"]
                     - book_xsval["by_cost"]["standard_2bps"]["sharpe"], 3)

    # ---- IC + shuffle-null ----
    ic, null95, n_ic = pooled_rank_ic(close, rebal_val)

    # ---- common-window combination (additivity) ----
    mom_net = book_mom["_net_standard"]
    val_net = book_xsval["_net_standard"]
    first_val = w_xsval[(w_xsval.abs().sum(axis=1) > 0)].index.min()
    common = mom_net.index[mom_net.index >= first_val]
    mom_w = mom_net.reindex(common).dropna()
    val_w = val_net.reindex(common).dropna()
    cidx = mom_w.index.intersection(val_w.index)
    mom_w, val_w = mom_w.reindex(cidx), val_w.reindex(cidx)
    combined = risk_parity_combine(mom_w, val_w)

    corr_vm = round(float(val_w.corr(mom_w)), 3)
    sh_mom = round(sharpe(mom_w), 3)
    sh_val = round(sharpe(val_w), 3)
    sh_comb = round(sharpe(combined), 3)
    dd_mom = round(max_dd(vol_norm(mom_w)) * 100, 2)
    dd_comb = round(max_dd(vol_norm(combined)) * 100, 2)

    # subperiods on the combined + momentum books
    sub = {}
    for name, (a, b) in SUBPERIODS.items():
        cm = combined.loc[a:b]
        mm = mom_w.loc[a:b]
        sub[name] = {"combined_sharpe": round(sharpe(cm), 3) if len(cm) > 20 else None,
                     "momentum_sharpe": round(sharpe(mm), 3) if len(mm) > 20 else None,
                     "n_days": int(len(cm))}

    # ----------------------------- GATES -----------------------------
    g1a = sh_val >= 0.20
    cls_val_sharpe = {c: results["books"][f"XSVALUE_{c}_monthly"]["by_cost"]
                      ["standard_2bps"]["sharpe"] for c in UNIVERSE}
    g1b = sum(1 for s in cls_val_sharpe.values() if s > 0) >= 2
    fric_val = results["books"]["XSVALUE_pooled_monthly"]["by_cost"]["frictionless"]["sharpe"]
    g1c = (fric_val - sh_val) <= 0.10
    gate1 = g1a and g1b and g1c

    g2a = corr_vm < 0.10
    g2b = sh_comb >= sh_mom + 0.05
    g2c = dd_comb <= dd_mom   # less negative or equal (dd values are negative %)
    gate2 = g2a and g2b and g2c

    sub_pos = sum(1 for v in sub.values() if v["combined_sharpe"] and v["combined_sharpe"] > 0)
    post17_adds = all(
        (sub[p]["combined_sharpe"] is not None and sub[p]["momentum_sharpe"] is not None
         and sub[p]["combined_sharpe"] > sub[p]["momentum_sharpe"]) for p in POST2017)
    g3a = (sub_pos >= 3) and post17_adds
    leak_ok = abs(leak_gap) < 0.10

    if gate1 and gate2:
        decision = "GO"
    elif gate1 and not gate2:
        decision = "NO_GO_value_redundant"
    else:
        decision = "NO_GO_value_weak"

    verdict = {
        "test": "cross-asset VALUE falsification (linear, pre-RL)",
        "decision": decision,
        "common_window": [str(cidx.min().date()), str(cidx.max().date())],
        "standalone": {
            "xsvalue_pooled_net_sharpe": sh_val,
            "xsvalue_pooled_frictionless_sharpe": fric_val,
            "frictionless_minus_net_gap": round(fric_val - sh_val, 3),
            "per_class_net_sharpe": cls_val_sharpe,
            "tsvalue_pooled_net_sharpe":
                results["books"]["TSVALUE_pooled_monthly"]["by_cost"]["standard_2bps"]["sharpe"],
        },
        "additivity": {
            "corr_value_momentum": corr_vm,
            "momentum_alone_sharpe_samewindow": sh_mom,
            "value_alone_sharpe_samewindow": sh_val,
            "combined_riskparity_sharpe": sh_comb,
            "uplift_vs_momentum": round(sh_comb - sh_mom, 3),
            "momentum_maxdd_pct@10vol": dd_mom,
            "combined_maxdd_pct@10vol": dd_comb,
        },
        "persistence_subperiods": sub,
        "leak": {"causal_sharpe": sh_val,
                 "sameday_sharpe": book_xsval_leak["by_cost"]["standard_2bps"]["sharpe"],
                 "gap": leak_gap, "ok": leak_ok},
        "ic": {"value_rank_ic": round(ic, 4), "shuffle_null_95": round(null95, 4),
               "ic_escapes_null": abs(ic) > null95, "n": n_ic},
        "gates": {
            "G1_standalone": {"pass": gate1, "1a_sharpe>=0.20": g1a,
                              "1b_>=2_classes_pos": g1b, "1c_cost_gap<=0.10": g1c},
            "G2_additivity": {"pass": gate2, "2a_corr<0.10": g2a,
                              "2b_combined>=mom+0.05": g2b, "2c_dd<=mom": g2c},
            "G3_persistence": {"pass": g3a, "3a_>=3of4_and_post2017_adds": g3a,
                               "subperiods_positive": sub_pos},
            "leak_tripwire": leak_ok,
        },
        "interpretation": (
            "Value earns a sleeve slot only if it is non-positively correlated to "
            "momentum AND lifts the combined book. Standalone strength is secondary "
            "(value's job is diversification via the AMP -0.5 correlation)."),
    }
    results["verdict"] = verdict

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    # ---- console + summary ----
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)
    p("=" * 70)
    p("CROSS-ASSET VALUE FALSIFICATION")
    p(f"common window: {verdict['common_window']}")
    p("-" * 70)
    p(f"XS-VALUE pooled net Sharpe (2bps): {sh_val}  (frictionless {fric_val}, "
      f"gap {round(fric_val - sh_val, 3)}, turnover {book_xsval['turnover_ann']}/yr)")
    p(f"  per-class net Sharpe: {cls_val_sharpe}")
    p(f"  TS-VALUE pooled net Sharpe: {verdict['standalone']['tsvalue_pooled_net_sharpe']}")
    p(f"value rank-IC {round(ic,4)} vs shuffle-null95 {round(null95,4)} "
      f"=> escapes_null={abs(ic) > null95} (n={n_ic})")
    p(f"leak: causal {sh_val} vs same-day "
      f"{book_xsval_leak['by_cost']['standard_2bps']['sharpe']} gap {leak_gap} ok={leak_ok}")
    p("-" * 70)
    p("ADDITIVITY (common window, risk-parity combine):")
    p(f"  corr(value, momentum)        = {corr_vm}")
    p(f"  momentum-alone Sharpe        = {sh_mom}   maxDD@10vol {dd_mom}%")
    p(f"  value-alone Sharpe           = {sh_val}")
    p(f"  COMBINED Sharpe              = {sh_comb}   maxDD@10vol {dd_comb}%")
    p(f"  uplift vs momentum           = {round(sh_comb - sh_mom, 3)}")
    p("  subperiods (combined / momentum):")
    for name, v in sub.items():
        p(f"    {name}: {v['combined_sharpe']} / {v['momentum_sharpe']}")
    p("-" * 70)
    p(f"G1 standalone : {gate1}  (1a {g1a} 1b {g1b} 1c {g1c})")
    p(f"G2 additivity : {gate2}  (2a corr {g2a} 2b uplift {g2b} 2c dd {g2c})")
    p(f"G3 persistence: {g3a}  (>=3/4 pos & post-2017 adds; pos={sub_pos}/4)")
    p(f"leak tripwire : {leak_ok}")
    p("=" * 70)
    p(f"DECISION: {decision}")
    p("=" * 70)
    (OUT / "summary.md").write_text("```\n" + "\n".join(lines) + "\n```\n")
    return verdict


if __name__ == "__main__":
    main()
