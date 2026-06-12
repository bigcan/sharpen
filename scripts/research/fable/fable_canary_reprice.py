"""Fable independent re-pricing of the gmgp1-btc clean-canary 'kill'.

Claim under test (results/gmgp1_btc_canary_costcorr_wf/verdict.json):
  OVERALL FAIL - every solo seed PF < 1.0 in all 4 folds, ens 0.893,
  worst stress DD ~ -33%, and (per the Tier-2 audit) no edge even
  frictionless. This is the falsification that killed the V7 directional
  program. If the eval harness was broken, the kill is unreliable; if
  independent re-pricing reproduces it, the kill stands on market data.

Empirically discovered trajectory convention (offset scan, corr=1.00000 on
2,745 no-trade bars): trajectory timestamps are bar-END stamps in UTC+8
(= +31 x 15min vs my UTC bar-START labels) and `position` is LEVERAGE
(notional/equity), so P&L compounds multiplicatively:
    eq[t] = eq[t-1] * (1 + pos[t] * ret[t]) - fees[t]
    fees[t] = eq[t-1] * |dpos[t]| * (taker + slip)   only on traded bars
Fees: taker 0.00055 + slippage 5bps (the costcorr config cost layer).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CAN = ROOT / "results" / "gmgp1_btc_canary_costcorr_wf"
OUT = ROOT / "results" / "fable_verdict"
OUT.mkdir(parents=True, exist_ok=True)

FEE = 0.00055
SLIP = 0.0005
K_OFFSET = 31  # discovered: traj ts + 31 bars = my UTC bar-start label


def load_close_15m() -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "btc_usdt_1min_bybit.parquet")
    ts = pd.to_datetime(df["timestamp"])
    df = df.set_index(pd.DatetimeIndex(ts).tz_localize(None)).sort_index()
    return df["close"].resample("15min", label="left", closed="left").last().dropna()


def reprice_weight_traj(traj: pd.DataFrame, close15: pd.Series,
                        fee: float, slip: float) -> pd.Series:
    t = traj.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t.set_index("timestamp")
    px = close15.shift(-K_OFFSET).reindex(t.index).astype(float)
    ret = px.pct_change(fill_method=None).fillna(0.0).to_numpy()
    pos = t["position"].fillna(0.0).to_numpy()
    traded = t["traded"].fillna(0.0).to_numpy() > 0
    dpos = np.abs(np.diff(pos, prepend=0.0))

    eq = np.empty(len(t))
    e = 100_000.0
    for i in range(len(t)):
        e = e * (1.0 + pos[i] * ret[i])
        if traded[i]:
            e -= e * dpos[i] * (fee + slip)
        eq[i] = e
    return pd.Series(eq, index=t.index)


def eq_stats(eq: pd.Series) -> dict:
    d = eq.diff().dropna()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return {
        "pf": round(float(pos / neg), 4) if neg > 0 else float("inf"),
        "ret_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2),
        "dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
    }


def main():
    close15 = load_close_15m()
    rows = []
    for fold_dir in sorted(CAN.glob("fold_*")):
        for mfile in sorted(fold_dir.glob("solo_*_metrics.json")) + \
                     [fold_dir / "ens_mean_metrics.json"]:
            label = mfile.name.replace("_metrics.json", "")
            rec = json.loads(mfile.read_text())
            traj = pd.read_parquet(fold_dir / f"{label}_trajectory.parquet")

            my = eq_stats(reprice_weight_traj(traj, close15, FEE, SLIP))
            fr = eq_stats(reprice_weight_traj(traj, close15, 0.0, 0.0))

            # recorded-path tracking error (with my fee model)
            t = traj.copy()
            t["timestamp"] = pd.to_datetime(t["timestamp"])
            myeq = reprice_weight_traj(traj, close15, FEE, SLIP)
            dev = (t.set_index("timestamp")["portfolio_value"] - myeq).abs()
            rows.append({
                "fold": fold_dir.name, "label": label,
                "rec_pf": round(rec["pf_bar"], 4), "my_pf": my["pf"],
                "fric_pf": fr["pf"],
                "rec_ret": round(rec["total_return_pct"], 2), "my_ret": my["ret_pct"],
                "rec_dd": round(rec["trailing_max_drawdown_pct"], 2), "my_dd": my["dd_pct"],
                "max_dev_$": round(float(dev.max()), 0),
                "end_dev_pct_of_eq": round(float(dev.iloc[-1] / myeq.iloc[-1] * 100), 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "canary_reprice.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))

    solo = df[df.label.str.startswith("solo")]
    print("\nSUMMARY (20 solo seed-folds):")
    print(f"  PF      recorded median {solo.rec_pf.median():.4f} | mine {solo.my_pf.median():.4f} | frictionless mine {solo.fric_pf.median():.4f}")
    print(f"  PF<1.0  recorded {int((solo.rec_pf<1).sum())}/20 | mine {int((solo.my_pf<1).sum())}/20 | frictionless {int((solo.fric_pf<1).sum())}/20")
    print(f"  worst DD recorded {solo.rec_dd.min():.1f}% | mine {solo.my_dd.min():.1f}%")
    print(f"  path tracking: median max-dev ${df['max_dev_$'].median():,.0f} on $100k")


if __name__ == "__main__":
    main()
