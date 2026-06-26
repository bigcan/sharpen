"""Falsification of two YouTube scalping videos (both SMB-Capital content):

  V1  pImpnT4Bqj0  "You will never look at scalping the same way again"
      -> SMB top scalp setups. From the authentic SMB "Offsides Scalping" cheat
         sheet (FX Replay):
         * VWAP FLIP: market entry on a *decisive* candle close that flips to the
           other side of session VWAP; stop at the recent swing H/L before the
           flip; FIXED 1R take-profit; no management. (SMB: beginners use ONLY this.)
         * OFFSIDES REVERSAL: trend-exhaustion after price deviates from VWAP +
           volume divergence (new price extreme on fading volume); stop-limit
           reversal entry past the recent H/L; stop at the H/L midpoint; FIXED 2R.
      Params: 5m, YM (Dow future). "May also work with other assets/timeframes."

  V2  t7XAyKAUvFA  "Very High Win-Rate MACD Scalping Strategy (Simple Trade)"
      -> MACD line/signal CROSSOVER scalp (+ zero-line/trend filter), exit on the
         opposite cross or a fixed R bracket. 5m.

Both are sold on WIN RATE. The decisive question (cont-66/67 doctrine) is whether
high win rate survives as positive NET EXPECTANCY after realistic intraday cost,
and whether it beats simply HOLDING the index intraday. A high win rate with <=0
expectancy is the small-win/large-loss negative-skew trap.

Method == the project's falsify-first / cost-aware / vs-baseline harness.
Testbed: ES (E-mini S&P) 1-min 2025 resampled to 5-min -- the faithful, liquid
equity-index-future home of an SMB YM scalp -- with GC (gold) and BTC 5m as
cross-checks. ES is also the CHEAPEST-to-trade case (1 tick ~ 0.4bps), so if a
scalp cannot survive cost on ES it cannot survive anywhere (cont-67 logic).

Causality (LEAK-2): every signal is computed from bar t's CLOSE and acted on at
bar t+1's OPEN (entries .shift one bar). Bracket TP/stop fill intrabar from t+1
forward; if a bar straddles both, the STOP fills first (adverse/conservative).
Session VWAP resets at the active-window open each day; the book is forced FLAT
at the active-window close (no overnight carry).

Run:  python scripts/research/smb_macd_scalp_eval.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "scalp_eval"
OUT.mkdir(parents=True, exist_ok=True)
GATES_FILE = ROOT / "configs" / "scalp_eval.gates.yaml"

ASSETS = {
    # name: (path, is_1min_needs_resample, is_24_7)
    "ES_5m": (ROOT / "data" / "cme" / "es_2025_ohlcv_1min.parquet", True, False),
    "GC_5m": (ROOT / "data" / "cme" / "gc_2025_ohlcv_1min.parquet", True, False),
    "BTC_5m": (ROOT / "data" / "processed" / "btc_bitfinex_2025_full_year_5min.parquet", False, True),
}
BARS_PER_DAY_5M_RTH = 78          # ~6.5h US cash session in 5m bars (annualization helper)
GAMMA = 0.5772156649             # Euler-Mascheroni (deflated-Sharpe expected-max)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_5m(path: Path, needs_resample: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"].fillna(0) > 0].reset_index(drop=True)
    if needs_resample:
        g = df.set_index("timestamp").resample("5min")
        df = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(),
            "low": g["low"].min(), "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }).dropna(subset=["open", "high", "low", "close"]).reset_index()
        df = df[df["volume"].fillna(0) > 0].reset_index(drop=True)
    # OHLC sanity (cheap data-integrity tripwire)
    bad = ((df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) |
           (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)).sum()
    assert bad == 0, f"{path.name}: {bad} OHLC-inconsistent bars"
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def active_window(df: pd.DataFrame, is_24_7: bool) -> pd.Series:
    """Boolean mask of the liquid intraday window. For 24/7 assets => all bars.
    For session assets => hours whose mean volume exceeds the median hourly mean
    (captures the active US window without needing exact tz)."""
    if is_24_7:
        return pd.Series(True, index=df.index)
    hv = df.groupby("hour")["volume"].mean()
    keep = set(hv[hv > hv.median()].index)
    return df["hour"].isin(keep)


def session_vwap(df: pd.DataFrame, active: pd.Series) -> pd.Series:
    """Causal session VWAP, reset at each day's first ACTIVE bar; NaN outside the
    active window. cumsum(tp*vol)/cumsum(vol) within (date, active)."""
    a = df[active].copy()
    cv = a.groupby("date")["volume"].cumsum()
    cnum = a.assign(x=a["tp"] * a["volume"]).groupby("date")["x"].cumsum()
    vw = cnum / cv
    out = pd.Series(np.nan, index=df.index)
    out.loc[a.index] = vw.values
    return out


# --------------------------------------------------------------------------- #
# bracket / signal-exit trade simulator
# --------------------------------------------------------------------------- #
def simulate(df: pd.DataFrame, signals: list[dict], sess_id: np.ndarray,
             sess_last: np.ndarray) -> pd.DataFrame:
    """signals: list of {i (signal bar), side, stop, target|None}. Enter at i+1
    open; walk forward; STOP-before-target on a straddling bar; exit at session's
    last active bar close if neither hits. Returns one row per executed trade."""
    o, h, lw, c = (df["open"].to_numpy(), df["high"].to_numpy(),
                   df["low"].to_numpy(), df["close"].to_numpy())
    n = len(df)
    rows = []
    for s in signals:
        i = s["i"]
        e = i + 1
        if e >= n or sess_id[e] != sess_id[i]:      # no next bar in same session
            continue
        side, stop, target = s["side"], s["stop"], s["target"]
        entry = o[e]
        r_risk = abs(entry - stop) / entry
        if r_risk <= 0 or not np.isfinite(r_risk):
            continue
        exit_px, exit_j, reason = None, None, None
        for j in range(e, n):
            if side > 0 and lw[j] <= stop:                      # stop first (adverse)
                exit_px, exit_j, reason = stop, j, "stop"
            elif side > 0 and target is not None and h[j] >= target:
                exit_px, exit_j, reason = target, j, "target"
            elif side < 0 and h[j] >= stop:
                exit_px, exit_j, reason = stop, j, "stop"
            elif side < 0 and target is not None and lw[j] <= target:
                exit_px, exit_j, reason = target, j, "target"
            elif sess_last[j]:                                  # flat at session end
                exit_px, exit_j, reason = c[j], j, "session_end"
            if exit_px is not None:
                break
        if exit_px is None:
            exit_px, exit_j, reason = c[n - 1], n - 1, "data_end"
        gross = side * (exit_px - entry) / entry
        rows.append({"entry_i": e, "exit_i": exit_j, "side": side, "entry": entry,
                     "exit": exit_px, "stop": stop, "r_risk": r_risk,
                     "gross_ret": gross, "r_mult": gross / r_risk,
                     "bars_held": exit_j - e, "reason": reason,
                     "ts": df["timestamp"].iloc[e]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# strategy signal generators (all causal: use info up to bar t close)
# --------------------------------------------------------------------------- #
def sig_vwap_flip(df, vwap, active, swing=10, body_k=0.0, tp_mult=1.0) -> list[dict]:
    """Decisive close flips across session VWAP. body_k = min body/ATR ('decisive'
    filter; 0 = any flip). Stop = recent swing H/L over prior `swing` bars;
    TP = tp_mult * R (tp_mult<1 => tight TP / wide stop = the high-win-rate trap)."""
    c, o = df["close"].to_numpy(), df["open"].to_numpy()
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    vw = vwap.to_numpy()
    act = active.to_numpy()
    body = np.abs(c - o)
    atr = pd.Series(np.maximum.reduce([hi - lo,
                    np.abs(hi - np.r_[c[0], c[:-1]]),
                    np.abs(lo - np.r_[c[0], c[:-1]])])).rolling(14, min_periods=5).mean().to_numpy()
    sigs = []
    for t in range(swing + 1, len(df) - 1):
        if not (act[t] and np.isfinite(vw[t]) and np.isfinite(vw[t - 1])):
            continue
        if atr[t] > 0 and body[t] < body_k * atr[t]:
            continue
        up = c[t] > vw[t] and c[t - 1] <= vw[t - 1]     # flip up
        dn = c[t] < vw[t] and c[t - 1] >= vw[t - 1]     # flip down
        e = o[t + 1]                                     # causal next-bar entry
        if up:
            stop = lo[t - swing:t + 1].min()
            if stop < e:
                sigs.append({"i": t, "side": 1, "stop": stop, "target": e + tp_mult * (e - stop)})
        elif dn:
            stop = hi[t - swing:t + 1].max()
            if stop > e:
                sigs.append({"i": t, "side": -1, "stop": stop, "target": e - tp_mult * (stop - e)})
    return sigs


def sig_offsides_reversal(df, vwap, active, swing=10, dev_k=1.0, vol_win=10) -> list[dict]:
    """Approximation of SMB Offsides Reversal: price extended from VWAP + volume
    divergence (new N-bar extreme on below-average volume) -> fade. Stop-entry past
    recent H/L, stop at H/L midpoint, TP=2R."""
    c, o, hi, lo, v = (df["close"].to_numpy(), df["open"].to_numpy(),
                       df["high"].to_numpy(), df["low"].to_numpy(),
                       df["volume"].to_numpy())
    vw = vwap.to_numpy()
    act = active.to_numpy()
    vstd = pd.Series(c).rolling(20, min_periods=10).std().to_numpy()
    vma = pd.Series(v).rolling(vol_win, min_periods=3).mean().to_numpy()
    sigs = []
    for t in range(swing + 1, len(df) - 1):
        if not (act[t] and np.isfinite(vw[t]) and vstd[t] > 0):
            continue
        dev = (c[t] - vw[t]) / vstd[t]
        new_hi = hi[t] >= hi[t - swing:t].max()
        new_lo = lo[t] <= lo[t - swing:t].min()
        vol_div = v[t] < vma[t]                                   # fading volume
        rec_hi, rec_lo = hi[t - swing:t + 1].max(), lo[t - swing:t + 1].min()
        mid = 0.5 * (rec_hi + rec_lo)
        e = o[t + 1]                                              # causal next-bar entry
        if dev > dev_k and new_hi and vol_div and mid > e:        # uptrend exhaustion -> short
            sigs.append({"i": t, "side": -1, "stop": mid, "target": e - 2 * (mid - e)})
        elif dev < -dev_k and new_lo and vol_div and mid < e:     # downtrend exhaustion -> long
            sigs.append({"i": t, "side": 1, "stop": mid, "target": e + 2 * (e - mid)})
    return sigs


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def sig_macd(df, active, fast=12, slow=26, signal=9, mode="pure",
             zero_filter=False, atr_mult=1.5, rr=1.5, swing=10) -> list[dict]:
    """MACD line/signal crossover. mode='pure' (exit handled separately by caller
    via opposite-cross synthetic targets) or 'bracket' (stop=swing H/L, TP=rr*R).
    Here we emit bracket signals; pure-crossover handled by sig_macd_pure."""
    c, o, hi, lo = (df["close"].to_numpy(), df["open"].to_numpy(),
                    df["high"].to_numpy(), df["low"].to_numpy())
    macd = _ema(c, fast) - _ema(c, slow)
    sigl = _ema(macd, signal)
    act = active.to_numpy()
    cross_up = (macd > sigl) & (np.r_[False, macd[:-1] <= sigl[:-1]])
    cross_dn = (macd < sigl) & (np.r_[False, macd[:-1] >= sigl[:-1]])
    sigs = []
    for t in range(slow + signal, len(df) - 1):
        if not act[t]:
            continue
        e = o[t + 1]
        if cross_up[t] and (not zero_filter or macd[t] > 0):
            stop = lo[t - swing:t + 1].min()
            if stop < e:
                sigs.append({"i": t, "side": 1, "stop": stop, "target": e + rr * (e - stop)})
        elif cross_dn[t] and (not zero_filter or macd[t] < 0):
            stop = hi[t - swing:t + 1].max()
            if stop > e:
                sigs.append({"i": t, "side": -1, "stop": stop, "target": e - rr * (stop - e)})
    return sigs


def macd_pure_positions(df, active, sess_id, sess_last, fast=12, slow=26,
                        signal=9, zero_filter=False) -> pd.Series:
    """Pure MACD crossover as a continuous long/short position (exit only on the
    opposite cross), flat outside the active window and at session end. Used to
    test the 'hold until opposite signal' MACD variant as a position stream."""
    c = df["close"].to_numpy()
    macd = _ema(c, fast) - _ema(c, slow)
    sigl = _ema(macd, signal)
    pos = np.where(macd > sigl, 1.0, -1.0)
    if zero_filter:
        pos = np.where(macd > sigl, np.where(macd > 0, 1.0, 0.0),
                       np.where(macd < 0, -1.0, 0.0))
    pos = pd.Series(pos, index=df.index).shift(1).fillna(0.0)     # act next bar
    pos = pos.where(active, 0.0)
    pos = pos.where(~pd.Series(sess_last, index=df.index), 0.0)
    return pos


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def expected_max_sr(k: int) -> float:
    if k < 2:
        return 0.0
    return ((1 - GAMMA) * _z(1 - 1.0 / k) + GAMMA * _z(1 - 1.0 / (k * math.e)))


def _z(p: float) -> float:
    # inverse standard normal CDF (Acklam approximation)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def trade_metrics(trades: pd.DataFrame, cost: float, years: float, df: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n": 0}
    net = trades["gross_ret"].to_numpy() - cost          # round-trip cost per trade
    n = len(net)
    wins, losses = net[net > 0], net[net < 0]
    exp_bps = float(net.mean() * 1e4)
    sd = net.std(ddof=1)
    tpy = n / years if years > 0 else np.nan
    sharpe = float(net.mean() / sd * math.sqrt(tpy)) if sd > 0 and tpy > 0 else 0.0
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"n": n, "win_rate": round(float((net > 0).mean()), 3),
            "avg_R_net": round(float((trades["r_mult"]).mean() - cost / trades["r_risk"].mean()), 3),
            "expectancy_bps": round(exp_bps, 3), "net_sharpe": round(sharpe, 3),
            "net_pf": round(pf, 3), "max_dd_pct": round(dd * 100, 2),
            "trades_per_yr": round(float(tpy), 0), "avg_bars_held": round(float(trades["bars_held"].mean()), 1),
            "_net": net}


def pos_metrics(df, pos, cost_oneway, years) -> dict:
    """For the continuous pure-MACD position stream."""
    ret = df["close"].pct_change().to_numpy()
    p = pos.to_numpy()
    # pos is ALREADY shifted in macd_pure_positions (pos[t] = decision from close[t-1]),
    # so it earns bar-t's return ret[t] = (c[t]-c[t-1])/c[t-1]. Multiplying by ret (NOT
    # ret[1:]) avoids a double-lag; single-bar causal, no look-ahead.
    pnl = p * np.nan_to_num(ret)
    turn = np.abs(np.diff(p, prepend=0.0))
    net = pnl - turn * cost_oneway
    net = net[np.isfinite(net)]
    sd = net.std(ddof=1)
    bpy = len(df) / years
    sharpe = float(net.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0
    pos_s, neg_s = net[net > 0].sum(), -net[net < 0].sum()
    pf = float(pos_s / neg_s) if neg_s > 0 else float("inf")
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"net_sharpe": round(sharpe, 3), "net_pf": round(pf, 3),
            "max_dd_pct": round(dd * 100, 2), "turnover_per_yr": round(float(turn.sum() / years), 0),
            "_net": net}


def intraday_bh(df, active, sess_id, sess_last, cost, years) -> dict:
    """Benchmark: long 1 unit from each session's first active bar open to its
    last active bar close (hold the index intraday), net of one round trip."""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    act = active.to_numpy()
    rows = []
    cur_date = None
    for t in range(len(df)):
        if not act[t]:
            continue
        if sess_id[t] != cur_date:
            entry_i, cur_date = t, sess_id[t]
        if sess_last[t]:
            rows.append((c[t] - o[entry_i]) / o[entry_i] - cost)
    arr = np.array(rows)
    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
    spd = len(arr) / years
    sharpe = float(arr.mean() / sd * math.sqrt(spd)) if sd > 0 else 0.0
    return {"net_sharpe": round(sharpe, 3), "mean_bps": round(float(arr.mean() * 1e4), 2),
            "n_sessions": len(arr), "_net": arr}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_gates() -> dict:
    raw = yaml.safe_load(GATES_FILE.read_text(encoding="utf-8"))
    return raw


def run_asset(name: str, path: Path, needs_resample: bool, is_24_7: bool,
              costs: dict) -> dict:
    df = load_5m(path, needs_resample)
    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25
    active = active_window(df, is_24_7)
    vwap = session_vwap(df, active)
    sess_id = df["date"].to_numpy()
    # last active bar per day
    sess_last = np.zeros(len(df), dtype=bool)
    af = df[active]
    last_idx = af.groupby("date").tail(1).index
    sess_last[last_idx] = True

    variants = {
        "vwap_flip_1R_anybody": sig_vwap_flip(df, vwap, active, swing=10, body_k=0.0),
        "vwap_flip_1R_decisive": sig_vwap_flip(df, vwap, active, swing=10, body_k=1.0),
        "vwap_flip_TP0.5R_hiwin": sig_vwap_flip(df, vwap, active, swing=10, body_k=1.0, tp_mult=0.5),
        "vwap_flip_TP0.33R_hiwin": sig_vwap_flip(df, vwap, active, swing=10, body_k=1.0, tp_mult=0.33),
        "offsides_rev_2R": sig_offsides_reversal(df, vwap, active, swing=10, dev_k=1.0),
        "macd_12_26_9_bracket1.5R": sig_macd(df, active, 12, 26, 9, rr=1.5, swing=10),
        "macd_5_13_9_bracket1.5R": sig_macd(df, active, 5, 13, 9, rr=1.5, swing=10),
        "macd_8_17_9_bracket1R": sig_macd(df, active, 8, 17, 9, rr=1.0, swing=10),
    }

    res = {"_meta": {"bars": len(df), "years": round(years, 2),
                     "start": str(df["timestamp"].iloc[0]), "end": str(df["timestamp"].iloc[-1]),
                     "active_frac": round(float(active.mean()), 3)}}
    # bracket strategies (per-trade)
    res["bench_intraday_bh"] = {ck: intraday_bh(df, active, sess_id, sess_last, costs[ck], years)
                                for ck in costs}
    for vname, sigs in variants.items():
        tr = simulate(df, sigs, sess_id, sess_last)
        res[vname] = {ck: trade_metrics(tr, costs[ck], years, df) for ck in costs}
    # pure MACD crossover (continuous position) at the standard 2.5bp one-way
    for tag, (f, s, g, zf) in {"macd_pure_12_26_9": (12, 26, 9, False),
                               "macd_pure_5_13_9": (5, 13, 9, False),
                               "macd_pure_12_26_9_zfilt": (12, 26, 9, True)}.items():
        pos = macd_pure_positions(df, active, sess_id, sess_last, f, s, g, zf)
        res[tag] = {ck: pos_metrics(df, pos, costs[ck], years) for ck in costs}
    return res


def main() -> dict:
    gates = load_gates()
    costs = gates["cost_models"]
    realistic = "es_realistic_2p5bp"
    fr = "frictionless"

    all_res = {}
    best = {"net_sharpe": -9, "name": None, "asset": None, "expectancy_bps": -9}
    deflate_pool = []      # (name, asset, per-trade net array) at realistic cost
    for name, (path, rs, t247) in ASSETS.items():
        if not path.exists():
            all_res[name] = {"error": "missing"}
            continue
        r = run_asset(name, path, rs, t247, costs)
        all_res[name] = r
        bh_ns = r["bench_intraday_bh"][realistic]["net_sharpe"]    # this asset's intraday hold
        for vname, byc in r.items():
            if vname.startswith("_") or vname == "bench_intraday_bh":
                continue
            m = byc.get(realistic, {})
            ns = m.get("net_sharpe", -9)
            # deflation pool: PER-TRADE bracket variants only (consistent SR units;
            # exclude the continuous pure-MACD streams whose _net is per-bar)
            if "expectancy_bps" in m and "_net" in m and len(np.atleast_1d(m["_net"])) > 5:
                deflate_pool.append((f"{name}:{vname}", m["_net"]))
            if ns > best["net_sharpe"]:
                best = {"net_sharpe": ns, "name": vname, "asset": name,
                        "expectancy_bps": m.get("expectancy_bps", None),
                        "net_pf": m.get("net_pf", None), "win_rate": m.get("win_rate", None),
                        "frictionless_sharpe": byc.get(fr, {}).get("net_sharpe", None),
                        "asset_bh_sharpe": bh_ns, "beats_bh": bool(ns > bh_ns)}

    # deflated Sharpe across all variants/assets searched (per-trade units)
    defl = {}
    if deflate_pool:
        srs = []
        for _, net in deflate_pool:
            sd = np.std(net, ddof=1)
            srs.append(net.mean() / sd if sd > 0 else 0.0)
        srs = np.array(srs)
        k = len(srs)
        best_i = int(np.argmax(srs))
        sr_best = float(srs[best_i])
        net_best = deflate_pool[best_i][1]
        T = len(net_best)
        sk = float(pd.Series(net_best).skew())
        ku = float(pd.Series(net_best).kurt()) + 3.0
        sr0 = float(np.std(srs, ddof=1) * expected_max_sr(k)) if k > 1 else 0.0
        denom = math.sqrt(max(1e-9, 1 - sk * sr_best + (ku - 1) / 4 * sr_best ** 2))
        dsr = _phi((sr_best - sr0) * math.sqrt(max(1, T - 1)) / denom)
        defl = {"n_variants_searched": k, "best_variant": deflate_pool[best_i][0],
                "best_per_trade_SR": round(sr_best, 4), "expected_max_SR_null": round(sr0, 4),
                "deflated_prob_best_gt_0": round(float(dsr), 4), "n_trades_best": T}

    g = gates["gates"]["phase1_linear"]
    decision = "GO" if (
        best["net_sharpe"] >= g["min_net_sharpe"] and
        (best.get("expectancy_bps") or -9) >= g["min_net_expectancy_bps"] and
        (best.get("net_pf") or 0) >= g["min_net_pf"] and
        best.get("beats_bh", False) and        # cont-66 gate: must beat intraday buy-and-hold
        defl.get("deflated_prob_best_gt_0", 0) >= gates["gates"]["g_deflation"]["min_deflated_prob_best_gt_0"]
    ) else "NO_GO"

    verdict = {
        "test": "SMB VWAP-scalp (V1) + MACD-scalp (V2) falsification, net of cost, vs intraday B&H",
        "best_variant": f"{best['asset']}:{best['name']}",
        "best_net_sharpe_realistic": best["net_sharpe"],
        "best_expectancy_bps_realistic": best.get("expectancy_bps"),
        "best_net_pf_realistic": best.get("net_pf"),
        "best_win_rate": best.get("win_rate"),
        "best_frictionless_sharpe": best.get("frictionless_sharpe"),
        "best_asset_bh_sharpe": best.get("asset_bh_sharpe"),
        "best_beats_buy_and_hold": best.get("beats_bh"),
        "deflation": defl,
        "gate": f"net Sharpe>={g['min_net_sharpe']} AND expectancy>0 AND PF>={g['min_net_pf']} "
                f"AND beats intraday B&H AND DSR>=0.95",
        "decision": decision,
    }
    payload = {"verdict": verdict, "by_asset": _strip(all_res)}
    (OUT / "scalp_results.json").write_text(json.dumps(payload, indent=2, default=str))
    _print(all_res, verdict, realistic, fr)
    return verdict


def _strip(d):
    """Drop the heavy _net arrays before JSON dump."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k != "_net"}
    return d


def _print(all_res, verdict, realistic, fr):
    print(f"\n{'='*92}\nSMB VWAP-SCALP (V1) + MACD-SCALP (V2) FALSIFICATION  (net of cost, vs intraday B&H)\n{'='*92}")
    for name, r in all_res.items():
        if "error" in r:
            print(f"\n{name}: MISSING")
            continue
        m = r["_meta"]
        bh = r["bench_intraday_bh"][realistic]
        print(f"\n### {name}  ({m['bars']} 5m bars, {m['years']}y, {m['start'][:10]}->{m['end'][:10]}, active {m['active_frac']*100:.0f}%)")
        print(f"    intraday B&H net Sharpe={bh['net_sharpe']}  ({bh['n_sessions']} sessions, {bh['mean_bps']}bps/session)")
        print(f"    {'variant':28s} {'N':>5s} {'winRt':>6s} {'expBPS':>7s} {'frSh':>6s} {'netSh':>6s} {'netPF':>6s} {'DD%':>7s} {'t/yr':>6s}")
        for vname, byc in r.items():
            if vname.startswith("_") or vname == "bench_intraday_bh":
                continue
            net = byc.get(realistic, {})
            frm = byc.get(fr, {})
            if "expectancy_bps" in net:        # bracket strategy
                print(f"    {vname:28s} {net.get('n',0):>5d} {net.get('win_rate',0):>6.2f} "
                      f"{net.get('expectancy_bps',0):>7.2f} {frm.get('net_sharpe',0):>6.2f} "
                      f"{net.get('net_sharpe',0):>6.2f} {net.get('net_pf',0):>6.2f} "
                      f"{net.get('max_dd_pct',0):>7.1f} {net.get('trades_per_yr',0):>6.0f}")
            else:                              # continuous pure-MACD
                print(f"    {vname:28s} {'--':>5s} {'--':>6s} {'--':>7s} {frm.get('net_sharpe',0):>6.2f} "
                      f"{net.get('net_sharpe',0):>6.2f} {net.get('net_pf',0):>6.2f} "
                      f"{net.get('max_dd_pct',0):>7.1f} {net.get('turnover_per_yr',0):>6.0f}")
    v = verdict
    print(f"\n{'#'*92}")
    print(f"BEST variant (realistic cost): {v['best_variant']}")
    print(f"  net Sharpe={v['best_net_sharpe_realistic']}  expectancy={v['best_expectancy_bps_realistic']}bps  "
          f"net PF={v['best_net_pf_realistic']}  win-rate={v['best_win_rate']}  frictionless Sharpe={v['best_frictionless_sharpe']}")
    if v["deflation"]:
        d = v["deflation"]
        print(f"  deflation: {d['n_variants_searched']} variants searched; best per-trade SR={d['best_per_trade_SR']}; "
              f"E[max SR|null]={d['expected_max_SR_null']}; P(best>0 deflated)={d['deflated_prob_best_gt_0']}")
    print(f"DECISION: {v['decision']}")
    print(f"{'#'*92}")


if __name__ == "__main__":
    main()
