"""Intraday session-VWAP falsification (the classic day-trading VWAP claim).

Companion to vwap_avwap_falsification.py (swing/daily breadth). This tests the
SPECIFIC strategy retail VWAP lore is about: intraday SESSION VWAP (resets each
trading day), traded as (a) reversion to VWAP and (b) price-above/below-VWAP
trend, FLAT overnight, net of cost -- with the same matched time-weighted control
(session SMA / Bollinger) so volume-weighting is isolated.

Testbed = OANDA XAUUSD M15 (~2.4y, 2024-2026). Gold is chosen deliberately: it is
the project's flagship intraday asset AND its ~2bps fee is the MOST FAVORABLE case
for an intraday strategy to survive costs. If session-VWAP fails on cheap, liquid,
long-history gold, the intraday case is closed. BTC M15 (24/7, rolling 24h VWAP,
6.5bps) is a secondary cross-check.

Prior: intraday mean-reversion already falsified on BTC even with passive-maker
execution (gmgp1_btc_meanrev_maker_probe, S553-cont-59); research 4/4 NO-GO.

Causality (LEAK-1/2): session VWAP at bar t uses only bars from the session open
..t; the position formed at t earns the t->t+1 return (pos.shift(1)); VWAP resets
each session and the book is forced FLAT at each session's last bar (no overnight
carry, no cross-session normalization).

Run:
    python scripts/research/vwap_intraday_falsification.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "vwap_avwap"
OUT.mkdir(parents=True, exist_ok=True)
GATES_FILE = ROOT / "configs" / "vwap_avwap.gates.yaml"

# (path, fee_per_side, session_mode)  session: 'cal'=calendar-day reset; 'roll96'=24h rolling
ASSETS = {
    "XAUUSD_M15": (ROOT / "data" / "oanda" / "xauusd_m15_m.parquet", 0.0002, "cal"),
    "BTC_M15": (ROOT / "data" / "processed" / "btc_bitfinex_2025_15min.parquet", 0.00065, "roll96"),
}
ROLL = 96                 # 24h rolling VWAP window for 24/7 assets
REV_THRESH = 1.0
REV_CAP = 3.0
MIN_BARS = 5              # skip first N bars of a session (VWAP/std undefined/noisy)


def load_gates() -> dict:
    raw = yaml.safe_load(Path(GATES_FILE).read_text(encoding="utf-8"))
    g = raw.get("gates", {})
    p1, add = g.get("phase1_linear", {}), g.get("vwap_adds_edge", {})
    return {
        "min_net_sharpe": float(p1.get("min_net_sharpe", 0.50)),
        "min_net_pf": float(p1.get("min_net_pf", 1.10)),
        "min_uplift": float(add.get("min_uplift_vs_matched_control", 0.10)),
    }


GATES = load_gates()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"].fillna(0) > 0].reset_index(drop=True)
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def session_key(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "cal":
        return df["timestamp"].dt.date
    return pd.Series(np.arange(len(df)) // ROLL, index=df.index)  # block proxy (unused for roll)


def session_refs(df: pd.DataFrame, mode: str) -> dict:
    """Causal session VWAP/SMA + their dispersion. For 'cal', expanding within the
    calendar day (resets each session). For 'roll96', trailing 24h rolling."""
    tp, vol = df["tp"], df["volume"]
    if mode == "roll96":
        num = (tp * vol).rolling(ROLL, min_periods=MIN_BARS).sum()
        den = vol.rolling(ROLL, min_periods=MIN_BARS).sum()
        vwap = num / den
        e_x2 = (tp * tp * vol).rolling(ROLL, min_periods=MIN_BARS).sum() / den
        vstd = np.sqrt((e_x2 - vwap * vwap).clip(lower=0))
        sma = tp.rolling(ROLL, min_periods=MIN_BARS).mean()
        sstd = tp.rolling(ROLL, min_periods=MIN_BARS).std()
        sess = pd.Series(0, index=df.index)  # single stream; flatten handled by roll
        return {"vwap": vwap, "vstd": vstd, "sma": sma, "sstd": sstd, "sess": sess}
    sess = session_key(df, mode)
    g = df.groupby(sess)
    cv = g["volume"].cumsum()
    vwap = g.apply(lambda x: (x["tp"] * x["volume"]).cumsum()).reset_index(level=0, drop=True) / cv
    ex2 = g.apply(lambda x: (x["tp"] ** 2 * x["volume"]).cumsum()).reset_index(level=0, drop=True) / cv
    vstd = np.sqrt((ex2 - vwap * vwap).clip(lower=0))
    cnt = g.cumcount() + 1
    csum = g["tp"].cumsum()
    sma = csum / cnt
    # expanding std within session
    csum2 = g.apply(lambda x: (x["tp"] ** 2).cumsum()).reset_index(level=0, drop=True)
    var = (csum2 / cnt - sma * sma).clip(lower=0) * (cnt / (cnt - 1).clip(lower=1))
    sstd = np.sqrt(var)
    return {"vwap": vwap, "vstd": vstd, "sma": sma, "sstd": sstd, "sess": pd.Series(
        pd.factorize(sess)[0], index=df.index)}


def positions(df: pd.DataFrame, refs: dict, kind: str, use: str) -> pd.Series:
    close = df["close"]
    ref = refs["vwap"] if use == "vwap" else refs["sma"]
    std = refs["vstd"] if use == "vwap" else refs["sstd"]
    sess = refs["sess"]
    if kind == "rev":
        z = (close - ref) / std.replace(0, np.nan)
        pos = (-z.clip(-REV_CAP, REV_CAP) / REV_CAP).where(z.abs() > REV_THRESH, 0.0)
    else:  # trend
        pos = np.sign(close - ref)
    pos = pd.Series(pos, index=df.index).fillna(0.0)
    # skip first MIN_BARS of each session; force flat at the last bar of each session
    bar_in_sess = df.groupby(sess).cumcount()
    pos = pos.where(bar_in_sess >= MIN_BARS, 0.0)
    last = df.groupby(sess).cumcount(ascending=False) == 0
    pos = pos.where(~last, 0.0)
    return pos


def evaluate(df: pd.DataFrame, pos: pd.Series, fee: float, bars_per_year: float) -> dict:
    ret = df["close"].pct_change()
    pnl = pos.shift(1) * ret                       # position at t-1 earns t-1->t return
    cost = pos.diff().abs().fillna(pos.abs()) * fee
    net = (pnl - cost).dropna()
    gross = pnl.dropna()
    s = net.std()
    sharpe = float(net.mean() / s * np.sqrt(bars_per_year)) if s > 0 else 0.0
    gs = gross.std()
    gsharpe = float(gross.mean() / gs * np.sqrt(bars_per_year)) if gs > 0 else 0.0
    n = net.to_numpy()
    poss, negs = n[n > 0].sum(), -n[n < 0].sum()
    pf = float(poss / negs) if negs > 0 else float("inf")
    eq = (1 + net.fillna(0)).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    turn = float(pos.diff().abs().sum() / (len(df) / bars_per_year))
    return {"net_sharpe": round(sharpe, 3), "gross_sharpe": round(gsharpe, 3),
            "net_pf": round(pf, 3), "max_dd_pct": round(dd * 100, 2),
            "turnover_per_yr": round(turn, 0), "n": int(len(net))}


def run_asset(name: str, path: Path, fee: float, mode: str) -> dict:
    df = load(path)
    span_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days or 1
    bpy = len(df) / (span_days / 365.25)
    refs = session_refs(df, mode)
    out = {"_meta": {"bars": len(df), "start": str(df["timestamp"].iloc[0]),
                     "end": str(df["timestamp"].iloc[-1]), "bars_per_year": round(bpy)}}
    for kind in ["rev", "trend"]:
        v = evaluate(df, positions(df, refs, kind, "vwap"), fee, bpy)
        c = evaluate(df, positions(df, refs, kind, "sma"), fee, bpy)
        out[kind] = {"VWAP": v, "CTRL": c, "uplift": round(v["net_sharpe"] - c["net_sharpe"], 3)}
    return out


def main() -> dict:
    results = {}
    best_sharpe, best_uplift = -9.0, -9.0
    for name, (path, fee, mode) in ASSETS.items():
        if not path.exists():
            results[name] = {"error": "missing"}
            continue
        r = run_asset(name, path, fee, mode)
        results[name] = r
        for kind in ["rev", "trend"]:
            best_sharpe = max(best_sharpe, r[kind]["VWAP"]["net_sharpe"])
            best_uplift = max(best_uplift, r[kind]["uplift"])

    decision = "GO" if (best_sharpe >= GATES["min_net_sharpe"] and
                        best_uplift >= GATES["min_uplift"]) else "NO_GO"
    verdict = {
        "test": "intraday session-VWAP falsification (reversion + trend, net of cost)",
        "best_vwap_net_sharpe": best_sharpe,
        "best_uplift_vs_control": best_uplift,
        "gate": f"net Sharpe>={GATES['min_net_sharpe']} AND uplift-vs-control>={GATES['min_uplift']}",
        "decision": decision,
    }
    payload = {"verdict": verdict, "by_asset": results}
    (OUT / "intraday_results.json").write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n{'='*72}\nINTRADAY SESSION-VWAP vs session-SMA control (net of cost)\n{'='*72}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name}: MISSING")
            continue
        m = r["_meta"]
        print(f"\n{name}  ({m['bars']} bars, {m['start'][:10]}->{m['end'][:10]}, ~{m['bars_per_year']}/yr, fee {ASSETS[name][1]*1e4:.1f}bps)")
        print(f"  {'strat':6s} {'VWAP_netSh':>10s} {'gross':>7s} {'CTRL_netSh':>10s} {'uplift':>7s} {'VWAP_PF':>8s} {'DD%':>7s} {'turn':>7s}")
        for kind in ["rev", "trend"]:
            v, c = r[kind]["VWAP"], r[kind]["CTRL"]
            print(f"  {kind:6s} {v['net_sharpe']:>10.2f} {v['gross_sharpe']:>7.2f} {c['net_sharpe']:>10.2f} "
                  f"{r[kind]['uplift']:>7.2f} {v['net_pf']:>8.2f} {v['max_dd_pct']:>7.1f} {v['turnover_per_yr']:>7.0f}")
    print(f"\n{'#'*72}")
    print(f"best VWAP net Sharpe={best_sharpe}  best uplift-vs-control={best_uplift}")
    print(f"DECISION: {decision}")
    print(f"{'#'*72}")
    return verdict


if __name__ == "__main__":
    main()
