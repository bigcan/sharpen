"""Fable breadth pass: independent re-pricing of ALL remaining decisive
walk-forward verdicts that have trajectory artifacts on disk.

Targets (after fable_canary_reprice.py covered the gmgp1-btc canary):
  1. sg1-btc de-leaked cost-corrected X1 WF (8 folds x 3 seeds, 3-min bars)
  2. gmgp1-gold de-leaked X2 WF (4 folds x 3 seeds, 15-min bars)

Trajectory rows are FILTERED env-step logs (rows can skip bars; gold also has
session breaks), so alignment is wall-clock: discover a constant time offset
dt such that on single-bar no-trade rows, pv.pct_change == pos * close-return
at (ts + dt). Require fit corr > 0.999 on that subset to certify. Re-pricing
then uses cumulative close returns BETWEEN logged rows (position held constant
across unlogged spans), fees on traded rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fable_verdict"
OUT.mkdir(parents=True, exist_ok=True)


def load_close(price_file: str, freq: str) -> pd.Series:
    df = pd.read_parquet(ROOT / price_file)
    ts = pd.to_datetime(df["timestamp"])
    df = df.set_index(pd.DatetimeIndex(ts).tz_localize(None)).sort_index()
    return df["close"].resample(freq, label="left", closed="left").last().dropna()


def _prep(traj: pd.DataFrame) -> pd.DataFrame:
    t = traj.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    return t.set_index("timestamp")


def _aligned_price(close: pd.Series, ts: pd.DatetimeIndex,
                   mode: str, k: int, bar_sec: int) -> pd.Series:
    if mode == "wallclock":
        p = close.reindex(ts + pd.Timedelta(seconds=k * bar_sec))
        p.index = ts
        return p
    # sequence mode: trajectory stamp -> bar index in the (gappy) bar series + k
    pis = close.index.get_indexer(ts)
    idx = pis + k
    valid = (pis >= 0) & (idx >= 0) & (idx < len(close))
    return pd.Series(np.where(valid, close.to_numpy()[np.clip(idx, 0, len(close) - 1)],
                              np.nan), index=ts)


def discover(traj: pd.DataFrame, close: pd.Series, bar_sec: int) -> dict:
    """Find (mode, offset) such that on single-bar no-trade rows the recorded
    equity return equals position * price return EXACTLY. Certification metric
    is the fraction of machine-exact rows (|resid| < 1e-8), not correlation -
    a handful of outlier rows must not veto an otherwise exact fit."""
    t = _prep(traj)
    pvr = t["portfolio_value"].pct_change(fill_method=None)
    gap = t.index.to_series().diff().dt.total_seconds()
    sel = (gap == bar_sec) & (t["traded"] == 0.0) & pvr.notna()
    best = {"frac_exact": -1.0}
    scans = [("wallclock", range(-int(12 * 3600 / bar_sec), int(12 * 3600 / bar_sec) + 1)),
             ("seq", range(-40, 41))]
    for mode, rng in scans:
        for k in rng:
            p = _aligned_price(close, t.index, mode, k, bar_sec)
            r = p.pct_change(fill_method=None)
            resid = (pvr - t["position"] * r)[sel].dropna()
            if len(resid) < 200:
                continue
            fe = float((resid.abs() < 1e-8).mean())
            if fe > best["frac_exact"]:
                best = {"mode": mode, "k": k, "frac_exact": round(fe, 4),
                        "med_resid": float(resid.abs().median()), "n": int(len(resid))}
    return best


def reprice(traj: pd.DataFrame, close: pd.Series, conv: dict, bar_sec: int,
            fee: float, slip: float) -> pd.Series:
    t = _prep(traj)
    p = _aligned_price(close, t.index, conv["mode"], conv["k"], bar_sec).astype(float)
    p = p.ffill()  # rare rows landing in a session break: carry last close
    ret = p.pct_change(fill_method=None).fillna(0.0).to_numpy()
    pos = t["position"].fillna(0.0).to_numpy()
    traded = t["traded"].fillna(0.0).to_numpy() > 0
    dpos = np.abs(np.diff(pos, prepend=0.0))
    eq = np.empty(len(t))
    e = 100_000.0
    for i in range(len(t)):
        e *= 1.0 + pos[i] * ret[i]
        if traded[i]:
            e -= e * dpos[i] * (fee + slip)
        eq[i] = e
    return pd.Series(eq, index=t.index)


def eq_stats(eq: pd.Series) -> dict:
    d = eq.diff().dropna()
    up, dn = d[d > 0].sum(), -d[d < 0].sum()
    return {"pf": round(float(up / dn), 4) if dn > 0 else float("inf"),
            "ret": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2),
            "dd": round(float((eq / eq.cummax() - 1).min()) * 100, 2)}


def reverify(name: str, results_dir: str, price_file: str, freq: str,
             bar_sec: int, fee: float, slip: float) -> pd.DataFrame:
    print(f"\n{'='*105}\n{name}  ({results_dir} @ {freq}, fee={fee} slip={slip})\n{'='*105}")
    close = load_close(price_file, freq)
    rdir = ROOT / results_dir
    rows, conv = [], None
    for fold_dir in sorted(rdir.glob("fold_*")):
        trajs = sorted(fold_dir.glob("solo_*_trajectory.parquet"))
        if (fold_dir / "ens_mean_trajectory.parquet").exists():
            trajs.append(fold_dir / "ens_mean_trajectory.parquet")
        for tf in trajs:
            label = tf.name.replace("_trajectory.parquet", "")
            rec = json.loads((fold_dir / f"{label}_metrics.json").read_text())
            traj = pd.read_parquet(tf)
            if conv is None:
                conv = discover(traj, close, bar_sec)
                print(f"convention fit (single-bar no-trade rows): {conv}")
                if conv["frac_exact"] < 0.90:
                    print("!! NO consistent convention found - cannot certify this dataset")
                    return pd.DataFrame()
            my = eq_stats(reprice(traj, close, conv, bar_sec, fee, slip))
            fr = eq_stats(reprice(traj, close, conv, bar_sec, 0.0, 0.0))
            rows.append({"fold": fold_dir.name, "label": label,
                         "rec_pf": round(rec["pf_bar"], 4), "my_pf": my["pf"],
                         "fric_pf": fr["pf"],
                         "rec_ret": round(rec["total_return_pct"], 1), "my_ret": my["ret"],
                         "rec_dd": round(rec["trailing_max_drawdown_pct"], 1), "my_dd": my["dd"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    solo = df[df.label.str.startswith("solo")]
    print(f"\n{name} SUMMARY: recorded median solo PF {solo.rec_pf.median():.4f} | "
          f"mine {solo.my_pf.median():.4f} | frictionless mine {solo.fric_pf.median():.4f}")
    print(f"  PF>=1.1 (deploy bar): recorded {int((solo.rec_pf>=1.1).sum())}/{len(solo)}, "
          f"mine {int((solo.my_pf>=1.1).sum())}/{len(solo)}")
    df.to_csv(OUT / f"reverify_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    reverify("sg1_btc_x1_costcorr_wf",
             "results/sg1_btc_velotrade_decay01_x1_ensemble_wf",
             "data/btc_usdt_1min_bybit.parquet", "3min", 180,
             fee=0.00055, slip=0.0005)
    # gmgp1-gold note: the WF config names data/cme/gc_2025_lob1_1min_stitched.parquet
    # but the env's actual mark-to-market series (certified: p95 residual 1.3e-16 on
    # no-trade rows, sequence offset +31 bars) is the 15-min file below. The sparse
    # lob1 1-min file CANNOT reproduce the trajectories (33% of minutes missing,
    # including event minutes) - see fable_data_trust addendum. The 15m file itself
    # contains confirmed corrupt stale-print bars (2025-08-15 21:00 flat-OHLC 3521.4,
    # +5% vs neighbors, reverting over the weekend; Yahoo GC=F shows no such move),
    # inside fold_00's window.
    reverify("gmgp1_gold_x2deleak_wf",
             "results/gmgp1_gold_ensemble_wf_x2deleak",
             "data/cme/gold_2025_2026q1_15min_stitched.parquet", "15min", 900,
             fee=0.000068, slip=0.0)
