"""
Discovery pass (folds 0-1, Dec-Jan ONLY). Condition net bar-PF on each pre-specified
causal axis to locate WHERE gmgp1-btc loses. Folds 2-3 held out for confirmation.

Decisive metric = PF (ratio). A zero-edge position loses more *dollars* in high vol but
that is not tradeable; only a regime whose PF is meaningfully >1 (or a loss regime whose
removal lifts the remainder's PF) is a candidate for an external overlay.
"""
import numpy as np
import pandas as pd

pool = pd.read_parquet("results/gmgp1_btc_canary_costcorr_wf/_loss_regime_pool.parquet")
DISC = pool[pool.phase == "discovery"].copy()


def pf(x):
    x = x[~np.isnan(x)]
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return pos / neg if neg > 0 else np.inf


def tab(df, col, bins=None, labels=None, qcut=None):
    d = df.copy()
    if qcut is not None:
        d["_b"] = pd.qcut(d[col], qcut, duplicates="drop")
    elif bins is not None:
        d["_b"] = pd.cut(d[col], bins=bins, labels=labels)
    else:
        d["_b"] = d[col]
    rows = []
    for b, g in d.groupby("_b", observed=True):
        x = g["pv_ret"].values
        rows.append({
            "bucket": str(b), "n": len(g),
            "PF": round(pf(x), 3),
            "mean_bp": round(np.nanmean(x) * 1e4, 2),
            "sum_pct": round(np.nansum(x) * 100, 1),
            "win%": round(float((x > 0).mean()) * 100, 1),
        })
    return pd.DataFrame(rows)


print(f"DISCOVERY rows={len(DISC)}  overall PF={pf(DISC['pv_ret'].values):.4f}  "
      f"mean={np.nanmean(DISC['pv_ret'])*1e4:.2f}bp\n")

print("=== A1. Trailing realized vol rv_32 (8h), quintiles ===")
print(tab(DISC, "rv_32", qcut=5).to_string(index=False))
print("\n=== A1b. rv_16 (4h) quintiles ===")
print(tab(DISC, "rv_16", qcut=5).to_string(index=False))

print("\n=== A2. Trailing volume ratio (2h/24h), quintiles  (<1 = quiet) ===")
print(tab(DISC, "vol_ratio", qcut=5).to_string(index=False))
print("\n=== A2b. Prior-bar raw volume, quintiles ===")
print(tab(DISC, "vol_lag1", qcut=5).to_string(index=False))

print("\n=== A3. Session (UTC) ===")
print(tab(DISC, "session").to_string(index=False))
print("\n=== A3b. Hour-of-day (UTC) ===")
print(tab(DISC, "hour").to_string(index=False))

print("\n=== A4. Weekend (Sat/Sun UTC) vs weekday ===")
print(tab(DISC, "weekend").to_string(index=False))
print("\n=== A4b. Day-of-week (0=Mon) ===")
print(tab(DISC, "dow").to_string(index=False))

print("\n=== A5. Direction (pos held into bar) ===")
print(tab(DISC, "dir").to_string(index=False))

print("\n=== A6. Trailing 4h momentum mom_16 quintiles (trend vs chop) ===")
print(tab(DISC, "mom_16", qcut=5).to_string(index=False))

print("\n=== A7. Position magnitude |pos_prev| quintiles (leverage) ===")
DISC["abs_pos"] = DISC["pos_prev"].abs()
print(tab(DISC, "abs_pos", qcut=5).to_string(index=False))
