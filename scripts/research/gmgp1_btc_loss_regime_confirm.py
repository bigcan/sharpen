"""
Confirmation pass. Thresholds FROZEN from discovery (folds 0-1); applied to held-out
confirmation (folds 2-3, Feb-Mar). Success criterion (pre-registered):
  a causal filter must lift confirmation PF to >= 1.10 with >= 30% activity retained
  AND the per-bin sign identified in discovery must persist OOS.
Plus: consecutive-loss / circuit-breaker (episode-level) analysis.
"""
import numpy as np
import pandas as pd

pool = pd.read_parquet("results/gmgp1_btc_canary_costcorr_wf/_loss_regime_pool.parquet")
DISC = pool[pool.phase == "discovery"]
CONF = pool[pool.phase == "confirm"].copy()


def pf(x):
    x = x[~np.isnan(x)]
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return pos / neg if neg > 0 else np.inf


def block_boot_pf(x, nb=2000, block=64, seed=0):
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    nblk = int(np.ceil(n / block))
    out = np.empty(nb)
    starts_pool = n - block
    for i in range(nb):
        starts = rng.integers(0, max(1, starts_pool), nblk)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        out[i] = pf(x[idx])
    return np.nanpercentile(out, [5, 50, 95])


# ============ FROZEN thresholds from discovery ============
LOSS_HOURS = [1, 9, 13, 16]   # discovery PF < 0.80
WIN_HOURS = [20, 21]          # discovery PF > 1.15
VOL_RATIO_BUSY = 1.528        # discovery top-quintile lower edge (busiest = worst)

print(f"CONFIRM rows={len(CONF)}  overall PF={pf(CONF['pv_ret'].values):.4f}  "
      f"mean={np.nanmean(CONF['pv_ret'])*1e4:.2f}bp")
print(f"  block-boot PF [5,50,95]: {np.round(block_boot_pf(CONF['pv_ret'].values),3)}\n")

print("=== C1. Do discovery WIN/LOSS hours persist OOS? (per-hour confirm PF) ===")
hh = CONF.groupby("hour")["pv_ret"].agg(n="size", PF=lambda s: pf(s.values),
                                          mean_bp=lambda s: np.nanmean(s)*1e4).round(3)
for h in WIN_HOURS + LOSS_HOURS:
    tag = "WIN(disc)" if h in WIN_HOURS else "LOSS(disc)"
    row = hh.loc[h]
    print(f"  hour {h:2d} [{tag}]: confirm PF={row.PF:.3f}  mean={row.mean_bp:+.2f}bp  n={int(row.n)}")

print("\n=== C2. FILTER A: drop discovery LOSS_HOURS {1,9,13,16} ===")
keepA = CONF[~CONF.hour.isin(LOSS_HOURS)]
print(f"  retained {len(keepA)/len(CONF):.1%}  confirm PF={pf(keepA['pv_ret'].values):.4f}  "
      f"(baseline {pf(CONF['pv_ret'].values):.4f})  boot{np.round(block_boot_pf(keepA['pv_ret'].values),3)}")

print("\n=== C3. FILTER B: trade ONLY discovery WIN_HOURS {20,21} ===")
keepB = CONF[CONF.hour.isin(WIN_HOURS)]
print(f"  retained {len(keepB)/len(CONF):.1%}  confirm PF={pf(keepB['pv_ret'].values):.4f}  "
      f"mean={np.nanmean(keepB['pv_ret'])*1e4:+.2f}bp  boot{np.round(block_boot_pf(keepB['pv_ret'].values),3)}")

print("\n=== C4. FILTER C: drop busiest-volume bars (vol_ratio>=1.528) ===")
keepC = CONF[CONF.vol_ratio < VOL_RATIO_BUSY]
print(f"  retained {len(keepC)/len(CONF):.1%}  confirm PF={pf(keepC['pv_ret'].values):.4f}  "
      f"(baseline {pf(CONF['pv_ret'].values):.4f})")

# ============ Consecutive-loss / circuit-breaker (episode level) ============
print("\n=== C5. Consecutive-loss structure (per fold x seed, position episodes) ===")
def episodes(df):
    """maximal same-sign holding runs; episode pnl = compounded pv_ret over the run."""
    d = df.sort_values("timestamp")
    sign = np.sign(d["pos_prev"].fillna(0)).values
    ret = d["pv_ret"].fillna(0).values
    eps, cur_sign, cur = [], None, []
    for s, rr in zip(sign, ret):
        if s == 0:
            continue
        if cur_sign is None or s == cur_sign:
            cur.append(rr); cur_sign = s
        else:
            eps.append(np.prod([1+x for x in cur]) - 1); cur = [rr]; cur_sign = s
    if cur:
        eps.append(np.prod([1+x for x in cur]) - 1)
    return np.array(eps)

all_eps = []
for (f, s), g in pool.groupby(["fold", "rule"]):
    e = episodes(g)
    all_eps.append(e)
ep = np.concatenate(all_eps)
losing = ep < 0
print(f"  episodes={len(ep)}  win-rate={np.mean(ep>0):.1%}  mean ep pnl={np.mean(ep)*100:.3f}%  "
      f"ep PF={pf(ep):.3f}")
# lag-1 sign autocorrelation: do losses cluster?
sgn = np.sign(ep)
ac1 = np.corrcoef(sgn[:-1], sgn[1:])[0, 1]
print(f"  episode-sign lag1 autocorr={ac1:+.3f}  (≈0 => losses do NOT cluster => circuit-breaker only reduces exposure)")
# conditional next-episode expectancy given current losing streak length
streak = 0; cond = {}
for i in range(1, len(ep)):
    if ep[i-1] < 0:
        streak += 1
    else:
        streak = 0
    cond.setdefault(min(streak, 5), []).append(ep[i])
print("  E[next episode pnl %] by current consecutive-loss streak:")
for k in sorted(cond):
    arr = np.array(cond[k])
    print(f"    after {k} losses: n={len(arr):4d}  mean={np.mean(arr)*100:+.3f}%  win%={np.mean(arr>0)*100:.1f}")
