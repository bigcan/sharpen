"""GMGP1-BTC conditional-alpha (conviction) probe — STEP 0 reward-redesign gate.

The clean canary (project_gmgp1_btc_clean_canary_s553) falsified UNCONDITIONAL
directional alpha on BTC 15m: realized IC negative in 25/25 trajectories, gross PF
~1.07-1.09 even frictionless. A reward function cannot create information that is
not in the observation. The ONLY way "take only high-confidence trades" can extract
alpha is if CONDITIONAL alpha exists — a subset of bars, selected by a CAUSAL
confidence proxy, where the directional signal has positive IC and net-of-cost
PF > 1.2 even though the all-bars signal is ~breakeven.

This probe answers that go/no-go question CHEAPLY (no GPU, no RL) before any reward
code is written. For a panel of standard causal primary signals x causal confidence
proxies, on the canary's walk-forward test folds, it compares the GATED subset to the
ALL-bars baseline, and computes an ORACLE upper bound (perfect-confidence gating). The
oracle is the crux:

  DECISION MATRIX (per primary x confidence pair, aggregated over folds)
  ---------------------------------------------------------------------------
  causal-gated net PF | oracle-gated net PF | conclusion
  > 1.2 in >=3/5 folds | (any)               | GO   — exploitable conditional alpha
                       |                     |        -> build the Selective Conviction Reward
  <= 1.2               | > 1.2               | conditional alpha EXISTS but our confidence
                       |                     |        features miss it -> search better features,
                       |                     |        do NOT build the reward yet
  <= 1.2               | <= 1.2              | NO-GO — no exploitable conditional alpha even
                       |                     |        under PERFECT gating; reward redesign is
                       |                     |        futile on this signal family

LEAK-2 discipline: every primary signal and confidence proxy uses ONLY data <= t.
Forward return uses t+h (it is the LABEL, never a feature). Confidence-gate thresholds
are fit on an EXPANDING TRAIN window that ENDS at each fold's start (never on test).
15m bars are built with label='right'/closed='right' (bar t closes at t; decision at
t, PnL realized over (t, t+h]).

Cost model matches the env / canary reverify EXACTLY:
  FEE_FRAC = taker 0.00055 + slippage 0.0005 (one-way), charged on |d position|.
PF convention matches reverify (pf over per-bar net pnl).

Multiple-testing: this scans 4 primaries x 4 confidence proxies x 5 folds. A handful
of cells WILL look good by chance. The verdict requires the SAME (primary, confidence)
pair to clear the bar in a MAJORITY of folds (>=3/5), and reports how many pairs clear
under the null expectation, so a single lucky cell cannot manufacture a GO.

Strict mode: any missing data, empty fold, or OHLC violation in a test window raises.

Run:  python scripts/research/gmgp1_btc_conviction_probe.py
Out:  results/gmgp1_btc_conviction_probe/probe_verdict.json  (+ stdout report)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_usdt_1min_bybit.parquet"
OUT_DIR = ROOT / "results" / "gmgp1_btc_conviction_probe"

# Cost: taker 5.5 bps + slippage 5.0 bps, one-way (matches reverify_gmgp1_btc_canary).
FEE_FRAC = 0.00055 + 0.0005

# Decision-scale = 15m (the env's decision bar). Sub-signals at 1h/4h built from it.
BAR = "15min"
TRAIN_START = "2024-06-01"   # expanding-train start for gate-threshold fitting
GATE_Q = 0.90                # top-decile confidence gate (fit on train only)
FWD_H = 1                    # forward holding horizon in 15m bars (next-bar return)

# Canary WF test folds + the virgin April-2026 bull month (+12.6%, all seeds lost).
FOLDS = {
    "2025-12":        ("2025-12-01", "2026-01-01"),
    "2026-01":        ("2026-01-01", "2026-02-01"),
    "2026-02":        ("2026-02-01", "2026-03-01"),
    "2026-03":        ("2026-03-01", "2026-04-01"),
    "2026-04_virgin": ("2026-04-01", "2026-05-01"),
}

PF_BAR = 1.2          # net-PF GO bar (canary kill criterion CONFIRM level)
MAJORITY = 3          # of 5 folds


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pf_bar(pnl: np.ndarray) -> float:
    """Profit factor over per-bar net pnl (zeros ignored)."""
    pnl = pnl[np.isfinite(pnl)]
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def ic(sig: np.ndarray, fwd: np.ndarray) -> float:
    m = np.isfinite(sig) & np.isfinite(fwd)
    if m.sum() < 30 or np.unique(sig[m]).size < 3:
        return float("nan")
    rho, _ = spearmanr(sig[m], fwd[m])
    return float(rho)


def strat(sig_sign: np.ndarray, fwd: np.ndarray, gate: np.ndarray, fee: float) -> dict:
    """Faithful gated long/short strategy metrics over a test fold.

    pos = sign(signal) on gated bars, 0 elsewhere. Cost on |d pos| (entry AND exit).
    pf_net over the FULL fold series (so exit-bar costs count); pf_fric over gated bars.
    """
    pos = np.where(gate, sig_sign, 0.0).astype(float)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(pos - pos_prev) * fee
    gross = pos * fwd
    net = gross - cost
    g = gate & np.isfinite(fwd)
    n = int(g.sum())
    hit = float((np.sign(sig_sign[g]) == np.sign(fwd[g])).mean()) if n else float("nan")
    return {
        "n_trades": n,
        "frac_bars": float(gate.mean()),
        "pf_fric": pf_bar(gross[g]),     # over gated bars only (gross pnl)
        "pf_net": pf_bar(net),           # over full fold (entry AND exit costs count)
        "hit_rate": hit,
        "mean_net_bps": float(np.nanmean(net[net != 0.0]) * 1e4) if (net != 0).any() else 0.0,
    }


# --------------------------------------------------------------------------- #
# data + causal features
# --------------------------------------------------------------------------- #
def load_15m() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    o = df["open"].resample(BAR, label="right", closed="right").first()
    h = df["high"].resample(BAR, label="right", closed="right").max()
    low = df["low"].resample(BAR, label="right", closed="right").min()
    c = df["close"].resample(BAR, label="right", closed="right").last()
    v = df["volume"].resample(BAR, label="right", closed="right").sum()
    bars = pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": v}).dropna()
    # OHLC sanity (DATA-CLEAN spirit): crash on impossible bars.
    bad = ((bars["high"] < bars["low"]) | (bars["high"] < bars["close"])
           | (bars["low"] > bars["close"]) | (bars[["open", "high", "low", "close"]] <= 0).any(axis=1))
    if bad.any():
        raise ValueError(f"OHLC violations in {int(bad.sum())} resampled 15m bars")
    return bars


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All columns are causal (use data <= t). Forward return uses t+h (label only)."""
    c = bars["close"]
    logret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=bars.index)

    # ---- forward return (LABEL) — close[t+h]/close[t]-1, NaN on the last h bars ----
    f["fwd"] = c.shift(-FWD_H) / c - 1.0

    # ---- multi-scale causal momentum (sign = direction, magnitude = strength) ----
    f["mom_1h"] = c / c.shift(4) - 1.0           # 4 x 15m = 1h
    f["mom_4h"] = c / c.shift(16) - 1.0          # 16 x 15m = 4h
    ema_f, ema_s = c.ewm(span=16).mean(), c.ewm(span=64).mean()
    f["ema_cross"] = (ema_f - ema_s) / c
    roll_m = c.rolling(16).mean()
    roll_sd = c.rolling(16).std()
    f["meanrev_1h"] = -(c - roll_m) / roll_sd.replace(0, np.nan)   # contrarian z-score

    # ---- confidence proxies (causal; higher = more confident) ----
    # self-strength is per-signal (|standardized signal|), built in the loop.
    s15 = np.sign(c / c.shift(1) - 1.0)
    s1h = np.sign(f["mom_1h"])
    s4h = np.sign(f["mom_4h"])
    f["conf_xscale_agree"] = (s15 + s1h + s4h).abs()             # 0..3 scales aligned
    rv = logret.rolling(16).std()
    f["conf_low_vol"] = 1.0 / rv.replace(0, np.nan)              # low realized vol
    dir1h = np.sign(f["mom_1h"])
    f["conf_trend_persist"] = (np.sign(logret).rolling(8).apply(
        lambda x: np.mean(x == x[-1]), raw=True))                # run-consistency 8 bars
    return f


PRIMARIES = ["mom_1h", "mom_4h", "ema_cross", "meanrev_1h"]
PROXIES = ["self_strength", "conf_xscale_agree", "conf_low_vol", "conf_trend_persist"]


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def bh_benchmark(bars: pd.DataFrame, s: str, e: str) -> dict:
    c = bars["close"]
    seg = c[(c.index >= pd.Timestamp(s, tz="UTC")) & (c.index < pd.Timestamp(e, tz="UTC"))]
    r = seg.pct_change().dropna().values
    return {"pf": pf_bar(r), "total_return_pct": float((seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0)}


def run() -> dict:
    bars = load_15m()
    feats = build_features(bars)
    print(f"15m bars: {len(bars)}  {bars.index[0]} -> {bars.index[-1]}  cost(one-way)={FEE_FRAC*1e4:.1f}bps\n")

    idx = feats.index
    results: dict = {"config": {"bar": BAR, "fee_frac": FEE_FRAC, "gate_q": GATE_Q,
                                "fwd_h": FWD_H, "pf_bar": PF_BAR, "folds": FOLDS},
                     "benchmark": {}, "cells": {}, "oracle": {}}

    for fname, (s, e) in FOLDS.items():
        results["benchmark"][fname] = bh_benchmark(bars, s, e)

    # ---- per (primary, confidence) cell, per fold: gated vs all ----
    for prim in PRIMARIES:
        sig = feats[prim]
        sig_sign = np.sign(sig.values)
        # causal standardized strength for the self_strength proxy
        strength = (sig / sig.rolling(1500, min_periods=200).std()).abs()
        for proxy in PROXIES:
            conf = strength if proxy == "self_strength" else feats[proxy]
            cell_key = f"{prim}|{proxy}"
            results["cells"][cell_key] = {}
            for fname, (s, e) in FOLDS.items():
                ts, te = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
                test_m = (idx >= ts) & (idx < te)
                train_m = (idx >= pd.Timestamp(TRAIN_START, tz="UTC")) & (idx < ts)
                # gate threshold fit on TRAIN only (OOS-honest)
                ctrain = conf.values[train_m]
                ctrain = ctrain[np.isfinite(ctrain)]
                if ctrain.size < 500:
                    raise ValueError(f"{cell_key} {fname}: train conf has {ctrain.size} finite (<500)")
                thr = float(np.quantile(ctrain, GATE_Q))

                fwd = feats["fwd"].values[test_m]
                ss = sig_sign[test_m]
                sc = sig.values[test_m]
                cf = conf.values[test_m]
                if test_m.sum() < 100:
                    raise ValueError(f"{cell_key} {fname}: only {int(test_m.sum())} test bars")

                gate_all = np.isfinite(sc) & np.isfinite(fwd)
                gate_top = gate_all & (cf >= thr)

                m_all = strat(ss, fwd, gate_all, FEE_FRAC)
                m_top = strat(ss, fwd, gate_top, FEE_FRAC)
                results["cells"][cell_key][fname] = {
                    "thr": thr,
                    "ic_all": ic(sc[gate_all], fwd[gate_all]),
                    "ic_gated": ic(sc[gate_top], fwd[gate_top]),
                    "pf_net_all": m_all["pf_net"],
                    "pf_net_gated": m_top["pf_net"],
                    "pf_fric_gated": m_top["pf_fric"],
                    "hit_gated": m_top["hit_rate"],
                    "n_gated": m_top["n_trades"],
                    "frac_gated": m_top["frac_bars"],
                    "mean_net_bps_gated": m_top["mean_net_bps"],
                }

    # ---- oracle upper bounds per primary (confidence-proxy-independent) ----
    for prim in PRIMARIES:
        sig = feats[prim]
        sig_sign = np.sign(sig.values)
        results["oracle"][prim] = {}
        for fname, (s, e) in FOLDS.items():
            ts, te = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
            test_m = (idx >= ts) & (idx < te)
            fwd = feats["fwd"].values[test_m]
            ss = sig_sign[test_m]
            fin = np.isfinite(fwd) & np.isfinite(ss)
            # oracle A: keep only bars where the signal's direction turns out CORRECT
            oracle_correct = fin & (np.sign(ss) == np.sign(fwd)) & (ss != 0)
            # oracle B: keep only the top-decile |forward move| bars (perfect big-move detector)
            big_thr = np.nanquantile(np.abs(fwd[fin]), GATE_Q)
            oracle_big = fin & (np.abs(fwd) >= big_thr)
            results["oracle"][prim][fname] = {
                "pf_net_oracle_correct": strat(ss, fwd, oracle_correct, FEE_FRAC)["pf_net"],
                "pf_net_oracle_bigmove": strat(ss, fwd, oracle_big, FEE_FRAC)["pf_net"],
                "frac_correct": float(oracle_correct.mean()),
            }

    return results


def verdict(res: dict) -> dict:
    """Aggregate the decision matrix across folds and pick the global verdict."""
    fold_names = list(FOLDS.keys())
    cell_pass = {}
    best_cell, best_score = None, -1
    for cell, byfold in res["cells"].items():
        n_pass = sum(
            1 for f in fold_names
            if np.isfinite(byfold[f]["pf_net_gated"]) and byfold[f]["pf_net_gated"] > PF_BAR
            and np.isfinite(byfold[f]["ic_gated"]) and byfold[f]["ic_gated"] > 0
        )
        cell_pass[cell] = n_pass
        if n_pass > best_score:
            best_cell, best_score = cell, n_pass

    # Oracle ceilings. NOTE: `oracle_correct` (keep only bars where sign(signal) turns
    # out right) is TAUTOLOGICAL — gross pnl = |fwd|>0 by construction, so it is
    # profitable whenever winners' moves exceed cost. It conditions on the OUTCOME, not
    # on any forecastable feature, so it is NOT a valid "can a confidence feature be
    # found" ceiling. The HONEST ceiling is `oracle_bigmove`: perfect knowledge of which
    # bars MOVE MOST (volatility is forecastable) — trade the signal there. If even that
    # fails, no causal confidence proxy can rescue the signal's DIRECTION.
    oracle_bigmove = {}
    oracle_correct_taut = {}
    for prim, byfold in res["oracle"].items():
        oracle_bigmove[prim] = sum(
            1 for f in fold_names
            if np.isfinite(byfold[f]["pf_net_oracle_bigmove"]) and byfold[f]["pf_net_oracle_bigmove"] > PF_BAR)
        oracle_correct_taut[prim] = sum(
            1 for f in fold_names
            if np.isfinite(byfold[f]["pf_net_oracle_correct"]) and byfold[f]["pf_net_oracle_correct"] > PF_BAR)
    oracle_best = max(oracle_bigmove.values()) if oracle_bigmove else 0

    # Side-finding: signals with genuine positive ALL-BARS IC in a majority of folds
    # (predictive structure exists even if cost-blocked). ic_all is identical across the
    # 4 proxies of a primary (same all-bars subset), so read it from any cell.
    positive_ic = {}
    for prim in PRIMARIES:
        any_cell = next(c for c in res["cells"] if c.startswith(prim + "|"))
        byfold = res["cells"][any_cell]
        n_pos = sum(1 for f in fold_names
                    if np.isfinite(byfold[f]["ic_all"]) and byfold[f]["ic_all"] > 0)
        ics = [byfold[f]["ic_all"] for f in fold_names if np.isfinite(byfold[f]["ic_all"])]
        positive_ic[prim] = {"folds_ic_positive": n_pos, "mean_ic_all": float(np.mean(ics))}

    causal_go = best_score >= MAJORITY
    oracle_go = oracle_best >= MAJORITY
    if causal_go:
        decision = ("GO — exploitable conditional alpha found CAUSALLY "
                    "(build the Selective Conviction Reward)")
    elif oracle_go:
        decision = ("SEARCH-BETTER-FEATURES — a forecastable move-size detector would "
                    "clear the bar; current causal proxies miss it (do NOT build reward yet)")
    else:
        decision = ("NO-GO — no exploitable conditional alpha even with a PERFECT "
                    "move-size detector; confidence-gating reward redesign is futile on "
                    "this signal family")

    cost_blocked = [p for p, d in positive_ic.items() if d["folds_ic_positive"] >= MAJORITY]
    return {
        "decision": decision,
        "causal_go": causal_go,
        "oracle_go": oracle_go,
        "best_cell": best_cell,
        "best_cell_folds_passed": f"{best_score}/{len(fold_names)}",
        "cell_pass_counts": cell_pass,
        "honest_oracle_bigmove_counts": oracle_bigmove,
        "tautological_oracle_correct_counts": oracle_correct_taut,
        "positive_ic_signals": positive_ic,
        "cost_blocked_positive_ic_signals": cost_blocked,
        "n_cells_scanned": len(cell_pass),
        "majority_required": MAJORITY,
    }


def report(res: dict, vd: dict) -> None:
    fold_names = list(FOLDS.keys())
    print("=" * 100)
    print("BUY-AND-HOLD benchmark per fold (PF, total return %):")
    for f in fold_names:
        b = res["benchmark"][f]
        print(f"  {f:>16}: PF {b['pf']:.3f}   ret {b['total_return_pct']:+.2f}%")

    print("\n" + "=" * 100)
    print("CAUSAL gated cells — net PF (gated) by fold, with all-bars net PF for contrast")
    print(f"{'primary|confidence':>34} " + " ".join(f"{f[:7]:>9}" for f in fold_names) + "   pass")
    for cell, byfold in res["cells"].items():
        row = f"{cell:>34} "
        for f in fold_names:
            pf = byfold[f]["pf_net_gated"]
            row += f"{pf:>9.3f}" if np.isfinite(pf) else f"{'nan':>9}"
        row += f"   {vd['cell_pass_counts'][cell]}/{len(fold_names)}"
        print(row)

    print("\nIC (gated subset) by fold — is the signal more predictive when 'confident'?")
    print(f"{'primary|confidence':>34} " + " ".join(f"{f[:7]:>9}" for f in fold_names))
    for cell, byfold in res["cells"].items():
        row = f"{cell:>34} "
        for f in fold_names:
            v = byfold[f]["ic_gated"]
            row += f"{v:>9.4f}" if np.isfinite(v) else f"{'nan':>9}"
        print(row)

    print("\n" + "=" * 100)
    print("ORACLE upper bounds — net PF by fold")
    print("  bigmove  = HONEST ceiling: perfect detector of top-decile move SIZE, trade the signal")
    print("  correct  = TAUTOLOGICAL: keep only outcome-correct bars (profitable by construction)")
    print(f"{'primary':>14} {'mode':>16} " + " ".join(f"{f[:7]:>9}" for f in fold_names))
    for prim, byfold in res["oracle"].items():
        for mode in ("pf_net_oracle_bigmove", "pf_net_oracle_correct"):
            row = f"{prim:>14} {mode.replace('pf_net_oracle_',''):>16} "
            for f in fold_names:
                v = byfold[f][mode]
                row += f"{v:>9.3f}" if np.isfinite(v) else f"{'nan':>9}"
            print(row)

    print("\nALL-BARS IC (predictive structure, pre-gating) — folds positive / mean IC:")
    for prim, d in vd["positive_ic_signals"].items():
        flag = "  <-- POSITIVE-IC SIGNAL (cost-blocked)" if d["folds_ic_positive"] >= MAJORITY else ""
        print(f"  {prim:>14}: {d['folds_ic_positive']}/{len(fold_names)} folds IC>0   "
              f"mean_ic={d['mean_ic_all']:+.4f}{flag}")

    print("\n" + "=" * 100)
    print(f"VERDICT: {vd['decision']}")
    print(f"  best causal cell : {vd['best_cell']}  ({vd['best_cell_folds_passed']} folds clear "
          f"PF_net>{PF_BAR} & IC>0; need {MAJORITY})")
    print(f"  honest oracle    : best {max(vd['honest_oracle_bigmove_counts'].values())}/{len(fold_names)} "
          f"folds clear under a PERFECT move-size detector (bigmove)")
    if vd["cost_blocked_positive_ic_signals"]:
        print(f"  SIDE-FINDING     : {vd['cost_blocked_positive_ic_signals']} carry genuine positive "
              f"OOS IC but are cost-blocked (edge < {FEE_FRAC*2*1e4:.0f}bps round-trip)")
    print(f"  cells scanned    : {vd['n_cells_scanned']} (multiple-testing: a single lucky cell "
          f"cannot meet the >={MAJORITY}/5 majority bar)")
    print("=" * 100)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run()
    vd = verdict(res)
    report(res, vd)
    out = OUT_DIR / "probe_verdict.json"
    out.write_text(json.dumps({"verdict": vd, "results": res}, indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
