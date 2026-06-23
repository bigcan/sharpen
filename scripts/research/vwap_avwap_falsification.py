"""VWAP / Anchored-VWAP falsification (pre-RL linear gate).

THE GATE. Before any RL or sleeve build, this answers: does a VWAP / anchored-VWAP
*signal* (not execution benchmark) convert into a real, net-of-cost, no-look-ahead,
regime-robust edge -- AND does volume-weighting add anything over the SAME-lookback
time-weighted moving average that the TSMOM book already owns?

Research artifact: .agent/artifacts/vwap_avwap_research.md (4/4 agents NO-GO prior).
The literature (Heston-Korajczyk-Sadka JF2010; Della Corte-Kosowski; Brock 1992;
Han-Zhou-Zhu JFE2016; Berkowitz-Logue-Noser JF1988; crypto-micro arXiv 2602.00776)
says VWAP is an EXECUTION benchmark and the only robust price-vs-average alpha is
TIME-weighted trend. So the decisive, never-published controlled experiment is:

    matched-lookback  VWAP-arm  vs  time-weighted-control-arm,  net of cost.

Every VWAP book here is paired with its OWN control (rolling-VWAP vs SMA of typical
price; VWAP-bands vs Bollinger; anchored-VWAP vs anchored-SMA). The verdict needs
BOTH (a) the absolute deploy gate AND (b) VWAP beating its matched control by the
pre-registered uplift -- else VWAP is a cosmetic re-skin of an MA (redundant with
TSMOM), not a new sleeve.

Universe = the 18-ETF, 4-asset-class breadth panel (the daily-timeframe BREADTH
exception to the no-4H/daily ban). Data = results/xsec_momentum/ohlcv_daily.parquet
(DATA-CLEAN'd, SHA-provenanced). Carry/intraday VWAP-reversion are out of scope here
(intraday MR already falsified on BTC -- gmgp1_btc_meanrev_maker_probe).

Causality (LEAK-2): every reference line at date t uses ONLY tp/volume <= t; the
signal formed from close[t] is applied to returns from t+1 (weights shifted 1 day).
Anchored VWAP re-anchors only at FIXED CALENDAR boundaries (causal) -- swing-pivot
anchors REPAINT and are deliberately excluded (a LEAK-2 violation by construction).

Run:
    python scripts/research/vwap_avwap_falsification.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "xsec_momentum" / "ohlcv_daily.parquet"
OUT = ROOT / "results" / "vwap_avwap"
OUT.mkdir(parents=True, exist_ok=True)
GATES_FILE = ROOT / "configs" / "vwap_avwap.gates.yaml"

# Universe / classes mirror the momentum panel (single source = the cached OHLCV).
UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
ALL_TICKERS = [t for v in UNIVERSE.values() for t in v]

ANN = 252
TARGET_VOL_ASSET = 0.10          # 10% annualized per-asset target (matches momentum harness)
TARGET_VOL_PORT = 0.10
LEV_CAP = 2.0
VOL_WIN = 63                     # realized-vol window for sizing
LOOKBACKS = [21, 63]             # 1m / 3m swing reference windows
ANCHOR_BLOCKS = {"A63": 63, "A252": 252}   # causal periodic re-anchor (quarterly, ~annual)
REV_THRESH = 1.0                 # band entry: act only when |z| > 1 sigma
REV_CAP = 3.0                    # clip z at +/-3 sigma
# 3 subperiods for regime robustness (recent = trailing window below)
SUBPERIODS = [("2008-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31"),
              ("2020-01-01", "2026-12-31")]
RECENT_YEARS = 3


# ---------------------------------------------------------------------------
# Gates (pre-registered; never hardcoded -- CLAUDE.md invariant)
# ---------------------------------------------------------------------------
def load_gates(gates_file: Path = GATES_FILE) -> dict:
    raw = yaml.safe_load(Path(gates_file).read_text(encoding="utf-8"))
    g = raw.get("gates", {})
    p1 = g.get("phase1_linear", {})
    add = g.get("vwap_adds_edge", {})
    div = g.get("g_diversification", {})
    return {
        "min_net_sharpe": float(p1.get("min_net_sharpe", 0.50)),
        "min_net_pf": float(p1.get("min_net_pf", 1.10)),
        "min_subperiods_positive": int(p1.get("min_subperiods_positive", 2)),
        "n_subperiods": int(p1.get("n_subperiods", 3)),
        "min_recent_oos_sharpe": float(p1.get("min_recent_oos_sharpe", 0.0)),
        "max_worst_subperiod_dd_pct": float(p1.get("max_worst_subperiod_dd_pct", 40.0)),
        "min_uplift_vs_matched_control": float(add.get("min_uplift_vs_matched_control", 0.10)),
        "max_corr_to_spy": float(div.get("max_corr_to_spy", 0.40)),
        "cost_models": raw.get("cost_models",
                               {"frictionless": 0.0, "standard_2bps": 0.0002,
                                "harsh_10bps": 0.0010}),
    }


GATES = load_gates()
COST_MODELS = GATES["cost_models"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_panel() -> dict:
    """Long OHLCV -> wide per-field frames (date x ticker). typical = (H+L+C)/3."""
    df = pd.read_parquet(DATA)
    df = df[df["ticker"].isin(ALL_TICKERS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = {f: df.pivot(index="date", columns="ticker", values=f).sort_index()
            for f in ["open", "high", "low", "close", "volume"]}
    cols = [t for t in ALL_TICKERS if t in wide["close"].columns]
    wide = {f: w[cols] for f, w in wide.items()}
    wide["typical"] = (wide["high"] + wide["low"] + wide["close"]) / 3.0
    sha = hashlib.sha256(pd.util.hash_pandas_object(wide["close"].fillna(0)).values
                         .tobytes()).hexdigest()[:16]
    wide["_sha"] = sha
    return wide


# ---------------------------------------------------------------------------
# Reference lines (all causal: rolling/expanding over data <= t)
# ---------------------------------------------------------------------------
def rolling_vwap(tp: pd.DataFrame, vol: pd.DataFrame, n: int) -> pd.DataFrame:
    num = (tp * vol).rolling(n, min_periods=n).sum()
    den = vol.rolling(n, min_periods=n).sum()
    return num / den


def rolling_vwap_std(tp: pd.DataFrame, vol: pd.DataFrame, n: int, vwap: pd.DataFrame) -> pd.DataFrame:
    # volume-weighted variance = E_v[x^2] - (E_v[x])^2 over the trailing window
    e_x2 = (tp * tp * vol).rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()
    var = (e_x2 - vwap * vwap).clip(lower=0.0)
    return np.sqrt(var)


def anchored_vwap(tp: pd.DataFrame, vol: pd.DataFrame, block: int) -> pd.DataFrame:
    """Causal anchored VWAP: re-anchor every `block` rows at fixed calendar
    boundaries; within a block, cumulative (tp*vol)/vol from the anchor forward."""
    pos = np.arange(len(tp))
    blk = pd.Series(pos // block, index=tp.index)
    pv = (tp * vol)
    num = pv.groupby(blk).cumsum()
    den = vol.groupby(blk).cumsum()
    return num / den


def anchored_sma(tp: pd.DataFrame, block: int) -> pd.DataFrame:
    """Matched time-weighted control: cumulative (expanding) mean of tp within block."""
    pos = np.arange(len(tp))
    blk = pd.Series(pos // block, index=tp.index)
    csum = tp.groupby(blk).cumsum()
    cnt = tp.notna().groupby(blk).cumsum()
    return csum / cnt


# ---------------------------------------------------------------------------
# Metrics (leverage-invariant Sharpe/PF; return/DD reported @ 10% vol target)
# ---------------------------------------------------------------------------
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


def metrics(net: pd.Series, turnover_ann: float, bench: pd.Series) -> dict:
    d = net.dropna()
    raw_vol = float(d.std() * np.sqrt(ANN))
    k = (TARGET_VOL_PORT / raw_vol) if raw_vol > 0 else 1.0
    dn = d * k
    return {
        "sharpe": round(sharpe(d), 3),
        "pf": round(pf(d), 3),
        "ann_ret_pct@10vol": round(float(dn.mean() * ANN * 100), 2),
        "max_dd_pct@10vol": round(max_dd(dn) * 100, 2),
        "raw_ann_vol_pct": round(raw_vol * 100, 1),
        "turnover_ann": round(turnover_ann, 1),
        "corr_SPY": round(float(d.corr(bench.reindex(d.index))), 3),
        "n_days": int(len(d)),
    }


def subperiod_sharpes(net: pd.Series) -> list:
    out = []
    for a, b in SUBPERIODS:
        seg = net.loc[a:b].dropna()
        out.append(round(sharpe(seg), 3) if len(seg) > 60 else None)
    return out


def recent_sharpe(net: pd.Series) -> float:
    d = net.dropna()
    if d.empty:
        return 0.0
    cut = d.index.max() - pd.Timedelta(days=365 * RECENT_YEARS)
    return round(sharpe(d.loc[cut:]), 3)


def worst_subperiod_dd_pct(net: pd.Series) -> float:
    worst = 0.0
    for a, b in SUBPERIODS:
        seg = net.loc[a:b].dropna()
        if len(seg) > 60:
            raw_vol = seg.std() * np.sqrt(ANN)
            k = (TARGET_VOL_PORT / raw_vol) if raw_vol > 0 else 1.0
            worst = min(worst, max_dd(seg * k))
    return round(worst * 100, 2)


# ---------------------------------------------------------------------------
# Weight construction + backtest (cloned causal/cost machinery)
# ---------------------------------------------------------------------------
def last_trading_of_period(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "daily":
        return index
    key = index.to_period("W" if freq == "weekly" else "M")
    last = pd.Series(index, index=index).groupby(key).max().values
    return pd.DatetimeIndex(last)


def vol_scale(rets: pd.DataFrame) -> pd.DataFrame:
    return (rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN))


def signal_to_weights(sig_daily: pd.DataFrame, rets: pd.DataFrame,
                      rebal: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-asset weight = signal[t] * (target_vol / realized_vol[t-1]) clipped, at
    each rebalance date t (signal uses data <= t). Causal."""
    realized = vol_scale(rets)
    w = pd.DataFrame(index=rebal, columns=sig_daily.columns, dtype=float)
    for dt in rebal:
        if dt not in sig_daily.index:
            continue
        s = sig_daily.loc[dt]
        rv = realized.loc[:dt]
        if rv.empty:
            continue
        rv_t = rv.iloc[-1]
        scale = (TARGET_VOL_ASSET / rv_t).clip(upper=LEV_CAP)
        w.loc[dt] = (s * scale).clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


def backtest(weights_rebal: pd.DataFrame, rets: pd.DataFrame) -> tuple:
    daily_idx = rets.index
    w_daily = weights_rebal.reindex(daily_idx).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)               # causal execution lag
    gross = (w_eff * rets).sum(axis=1)
    dw = weights_rebal.fillna(0.0).diff().abs().sum(axis=1)
    if len(weights_rebal):
        dw.iloc[0] = weights_rebal.iloc[0].abs().sum()
    years = (daily_idx[-1] - daily_idx[0]).days / 365.25
    turnover_ann = float(dw.sum() / years) if years > 0 else 0.0
    dw_daily = dw.reindex(daily_idx).fillna(0.0)
    cost_daily = {name: dw_daily * c for name, c in COST_MODELS.items()}
    return gross, cost_daily, turnover_ann


def run_book(weights_rebal: pd.DataFrame, rets: pd.DataFrame, bench: pd.Series) -> dict:
    gross, cost_daily, turn = backtest(weights_rebal, rets)
    by_cost = {cm: metrics(gross - cost_daily[cm], turn, bench) for cm in COST_MODELS}
    net_std = gross - cost_daily["standard_2bps"]
    return {
        "turnover_ann": round(turn, 1),
        "by_cost": by_cost,
        "subperiod_sharpes_net2bps": subperiod_sharpes(net_std),
        "recent3y_sharpe_net2bps": recent_sharpe(net_std),
        "worst_subperiod_dd_pct_net2bps": worst_subperiod_dd_pct(net_std),
        "_net_std": net_std,
    }


# ---------------------------------------------------------------------------
# Signal families (VWAP arm + matched time-weighted control share the SAME logic;
# only the reference line differs -> isolates the value of volume-weighting)
# ---------------------------------------------------------------------------
def trend_ts(close: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Long above the reference, short below (the classic price-vs-VWAP rule)."""
    return np.sign(close - ref)


def trend_xs(close: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional: long top third / short bottom third by (close/ref - 1)."""
    score = close / ref - 1.0
    out = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for dt, row in score.iterrows():
        v = row.dropna()
        if len(v) < 6:
            continue
        k = max(1, len(v) // 3)
        out.loc[dt, v.nlargest(k).index] = 1.0
        out.loc[dt, v.nsmallest(k).index] = -1.0
    return out


def reversion(close: pd.DataFrame, ref: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """Fade deviation: short when price >> ref, long when price << ref. Act only
    outside the +/-REV_THRESH band (a band trade, like Bollinger reversion)."""
    z = (close - ref) / std.replace(0.0, np.nan)
    sig = -z.clip(-REV_CAP, REV_CAP) / REV_CAP
    sig = sig.where(z.abs() > REV_THRESH, 0.0)
    return sig


def build_books(w: dict, rets: pd.DataFrame, bench: pd.Series) -> dict:
    """Full matched grid. Each VWAP book is paired with its time-weighted control
    via the naming convention '<family>__VWAP' / '<family>__CTRL'."""
    close, tp, vol = w["close"], w["typical"], w["volume"]
    books: dict = {}

    def add(name, sig_daily, rets_, freq):
        rebal = last_trading_of_period(rets_.index, freq)
        wts = signal_to_weights(sig_daily.reindex(rets_.index), rets_, rebal)
        books[name] = run_book(wts, rets_, bench)

    # ---- ROLLING reference: trend (TS + XS) ----
    for n in LOOKBACKS:
        vwap = rolling_vwap(tp, vol, n)
        sma = tp.rolling(n, min_periods=n).mean()
        for freq in ["monthly", "weekly"]:
            add(f"trend_ts_n{n}_{freq}__VWAP", trend_ts(close, vwap), rets, freq)
            add(f"trend_ts_n{n}_{freq}__CTRL", trend_ts(close, sma), rets, freq)
            add(f"trend_xs_n{n}_{freq}__VWAP", trend_xs(close, vwap), rets, freq)
            add(f"trend_xs_n{n}_{freq}__CTRL", trend_xs(close, sma), rets, freq)

    # ---- ROLLING reference: band reversion (VWAP-bands vs Bollinger) ----
    for n in LOOKBACKS:
        vwap = rolling_vwap(tp, vol, n)
        vstd = rolling_vwap_std(tp, vol, n, vwap)
        sma = tp.rolling(n, min_periods=n).mean()
        sstd = tp.rolling(n, min_periods=n).std()
        for freq in ["weekly", "daily"]:
            add(f"rev_n{n}_{freq}__VWAP", reversion(close, vwap, vstd), rets, freq)
            add(f"rev_n{n}_{freq}__CTRL", reversion(close, sma, sstd), rets, freq)

    # ---- ANCHORED reference: trend (causal periodic re-anchor) ----
    for name, blk in ANCHOR_BLOCKS.items():
        avwap = anchored_vwap(tp, vol, blk)
        asma = anchored_sma(tp, blk)
        add(f"anchored_trend_{name}_monthly__VWAP", trend_ts(close, avwap), rets, "monthly")
        add(f"anchored_trend_{name}_monthly__CTRL", trend_ts(close, asma), rets, "monthly")

    return books


# ---------------------------------------------------------------------------
# Deflated-Sharpe expected-max-under-null benchmark (multiplicity awareness)
# ---------------------------------------------------------------------------
def expected_max_sharpe(book_sharpes: list, n_trials: int) -> float:
    """Bailey & Lopez de Prado expected max Sharpe of N independent null trials,
    scaled by the cross-book Sharpe dispersion. A best-of-N winner should clear this."""
    s = np.array([x for x in book_sharpes if x is not None], dtype=float)
    if len(s) < 2 or n_trials < 2:
        return 0.0
    sigma = float(s.std(ddof=1))
    gamma = 0.5772156649
    from statistics import NormalDist
    nd = NormalDist()
    emax = ((1 - gamma) * nd.inv_cdf(1 - 1.0 / n_trials)
            + gamma * nd.inv_cdf(1 - 1.0 / (n_trials * np.e)))
    return round(sigma * emax, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    w = load_panel()
    close = w["close"]
    rets = close.pct_change()
    bench = close["SPY"].pct_change() if "SPY" in close else rets.mean(axis=1)
    print(f"Loaded {close.shape[1]} tickers, {close.shape[0]} days, "
          f"{close.index.min().date()} -> {close.index.max().date()} (sha {w['_sha']})")

    books = build_books(w, rets, bench)

    # split into VWAP books and their matched controls
    vwap_books = {k: v for k, v in books.items() if k.endswith("__VWAP")}
    cost = "standard_2bps"

    # best VWAP book by net Sharpe
    best_name = max(vwap_books, key=lambda k: vwap_books[k]["by_cost"][cost]["sharpe"])
    best = vwap_books[best_name]
    ctrl_name = best_name.replace("__VWAP", "__CTRL")
    ctrl = books[ctrl_name]

    best_sh = best["by_cost"][cost]["sharpe"]
    ctrl_sh = ctrl["by_cost"][cost]["sharpe"]
    uplift = round(best_sh - ctrl_sh, 3)

    # uplift across EVERY matched pair (is there ANY book where VWAP beats control?)
    pair_uplifts = {}
    for vk in vwap_books:
        ck = vk.replace("__VWAP", "__CTRL")
        if ck in books:
            pair_uplifts[vk] = round(
                vwap_books[vk]["by_cost"][cost]["sharpe"] - books[ck]["by_cost"][cost]["sharpe"], 3)
    best_uplift_name = max(pair_uplifts, key=pair_uplifts.get)
    best_uplift = pair_uplifts[best_uplift_name]

    # subperiod robustness for the best VWAP book
    sp = best["subperiod_sharpes_net2bps"]
    sp_pos = sum(1 for x in sp if x is not None and x > 0)
    recent = best["recent3y_sharpe_net2bps"]
    worst_dd = best["worst_subperiod_dd_pct_net2bps"]
    best_pf = best["by_cost"][cost]["pf"]
    fric = best["by_cost"]["frictionless"]["sharpe"]
    corr_spy = best["by_cost"][cost]["corr_SPY"]

    # deflated-Sharpe expected-max benchmark over all VWAP trials
    all_vwap_sh = [vwap_books[k]["by_cost"][cost]["sharpe"] for k in vwap_books]
    emax = expected_max_sharpe(all_vwap_sh, len(all_vwap_sh))

    # ---- gate evaluation ----
    pass_abs = (
        best_sh >= GATES["min_net_sharpe"]
        and best_pf >= GATES["min_net_pf"]
        and sp_pos >= GATES["min_subperiods_positive"]
        and recent > GATES["min_recent_oos_sharpe"]
        and abs(worst_dd) < GATES["max_worst_subperiod_dd_pct"]
    )
    pass_uplift = uplift >= GATES["min_uplift_vs_matched_control"]
    decision = "GO" if (pass_abs and pass_uplift) else "NO_GO"
    if not pass_abs and not pass_uplift:
        reason = "fails absolute deploy gate AND redundant with time-weighted MA"
    elif not pass_abs:
        reason = "fails absolute deploy gate (net Sharpe/PF/subperiod/recent/DD)"
    elif not pass_uplift:
        reason = "redundant_with_time_weighted_MA (no uplift over matched control)"
    else:
        reason = "clears both absolute gate and matched-control uplift"

    verdict = {
        "test": "VWAP / anchored-VWAP falsification (linear, pre-RL)",
        "data_sha16": w["_sha"],
        "n_vwap_books": len(vwap_books),
        "best_vwap_book": best_name,
        "best_vwap_net_sharpe_2bps": best_sh,
        "best_vwap_net_pf_2bps": best_pf,
        "matched_control_book": ctrl_name,
        "matched_control_net_sharpe_2bps": ctrl_sh,
        "uplift_vwap_minus_control": uplift,
        "best_uplift_any_pair": {best_uplift_name: best_uplift},
        "frictionless_sharpe": fric,
        "frictionless_minus_net_gap": round(fric - best_sh, 3),
        "subperiod_sharpes": sp,
        "subperiods_positive": sp_pos,
        "recent3y_net_sharpe": recent,
        "worst_subperiod_dd_pct": worst_dd,
        "corr_SPY": corr_spy,
        "deflated_sharpe_expected_max_null": emax,
        "best_clears_deflated_benchmark": bool(best_sh > emax),
        "gate_abs_pass": bool(pass_abs),
        "gate_uplift_pass": bool(pass_uplift),
        "decision": decision,
        "reason": reason,
        "gate_def": ("best VWAP net Sharpe>=%.2f AND net PF>=%.2f AND >=%d/%d subperiods net+ "
                     "AND recent-3y net Sharpe>0 AND worst-subperiod DD<%.0f%% AND uplift-vs-"
                     "matched-control>=%.2f" % (
                         GATES["min_net_sharpe"], GATES["min_net_pf"],
                         GATES["min_subperiods_positive"], GATES["n_subperiods"],
                         GATES["max_worst_subperiod_dd_pct"],
                         GATES["min_uplift_vs_matched_control"])),
        "expectation_note": ("Research prior = NO-GO: VWAP is an execution benchmark; the only "
                             "robust price-vs-average alpha is time-weighted trend (already in TSMOM). "
                             "Volume-weighting expected to add ~0 over the matched control."),
    }

    # serialize (drop heavy series)
    def slim(bk):
        return {"turnover_ann": bk["turnover_ann"], "by_cost": bk["by_cost"],
                "subperiod_sharpes_net2bps": bk["subperiod_sharpes_net2bps"],
                "recent3y_sharpe_net2bps": bk["recent3y_sharpe_net2bps"],
                "worst_subperiod_dd_pct_net2bps": bk["worst_subperiod_dd_pct_net2bps"]}
    payload = {"verdict": verdict, "gates": GATES,
               "books": {k: slim(v) for k, v in books.items()},
               "pair_uplifts": pair_uplifts}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- console summary ----
    print(f"\n{'='*78}\nMATCHED VWAP vs CONTROL (net 2bps Sharpe | PF | uplift)\n{'='*78}")
    print(f"{'book':34s} {'VWAP_Sh':>8s} {'CTRL_Sh':>8s} {'uplift':>7s} "
          f"{'VWAP_PF':>8s} {'turn':>6s}")
    for vk in sorted(vwap_books):
        ck = vk.replace("__VWAP", "__CTRL")
        vb, cb = vwap_books[vk], books[ck]
        base = vk.replace("__VWAP", "")
        print(f"{base:34s} {vb['by_cost'][cost]['sharpe']:>8.2f} "
              f"{cb['by_cost'][cost]['sharpe']:>8.2f} {pair_uplifts[vk]:>7.2f} "
              f"{vb['by_cost'][cost]['pf']:>8.2f} {vb['turnover_ann']:>6.0f}")

    print(f"\n{'#'*78}")
    print(f"BEST VWAP book: {best_name}")
    print(f"  net Sharpe(2bps)={best_sh}  PF={best_pf}  frictionless={fric} (gap {fric-best_sh:.3f})")
    print(f"  matched control={ctrl_name}  net Sharpe={ctrl_sh}  => UPLIFT={uplift}")
    print(f"  best uplift of ANY pair: {best_uplift_name} = {best_uplift}")
    print(f"  subperiods(net2bps)={sp} (positive {sp_pos}/{GATES['n_subperiods']})  "
          f"recent3y={recent}  worstDD%={worst_dd}  corrSPY={corr_spy}")
    print(f"  deflated-Sharpe expected-max-null={emax} (best clears: {best_sh > emax})")
    print(f"  gate_abs={pass_abs}  gate_uplift={pass_uplift}")
    print(f"DECISION: {decision}  ({reason})")
    print(f"{'#'*78}")
    return verdict


if __name__ == "__main__":
    main()
