"""
gmgp1-btc loss-regime analysis (S553-cont-62+).

Question: can an EXTERNAL, causal (ex-ante) risk overlay lift gmgp1-btc net PF by
avoiding identifiable loss regimes (low-vol / low-volume / weekend / specific hours /
consecutive-loss streaks)?

Foundation only here: build a merged per-bar dataset across the 4 cost-corrected WF
test folds x 5 seeds (+ ens_mean), aligned to the underlying Bybit 1-min OHLCV via the
verified +465-min (=+31-bar) trajectory-timestamp offset, with CAUSAL trailing regime
features (known at the start of each bar -> a filter built on them is live-implementable).

Self-validating tripwire: bar-level PF on ens_mean must reproduce the verdict's
graded ens_mean PF (~0.893) within tolerance, else the pipeline is misaligned and we HALT.

Discovery = folds 0,1 (Dec-Jan).  Confirmation = folds 2,3 (Feb-Mar). Hold out 2,3.
"""
import json
import numpy as np
import pandas as pd

OFFSET_MIN = 465  # verified: traj_ts + 465min -> price bar label (corr 0.9999); = +31 bars, UTC+8 bar-end
ROOT = "results/gmgp1_btc_canary_costcorr_wf"
DATA = "data/btc_usdt_1min_bybit.parquet"
SEEDS = [123, 456, 789, 1024, 2026]
FOLDS = [0, 1, 2, 3]
DISCOVERY_FOLDS = [0, 1]
CONFIRM_FOLDS = [2, 3]


def build_price_features() -> pd.DataFrame:
    """15-min naive-UTC bar-start price table with CAUSAL trailing regime features.
    Every rolling feature is .shift(1) so it excludes the current bar's own return/volume
    -> known at the START of the bar (decision time)."""
    o = pd.read_parquet(DATA)
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True).dt.tz_convert(None)
    o = o.set_index("timestamp").sort_index()
    r = (
        o.resample("15min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna()
    )
    r["ret"] = r["close"].pct_change()
    # --- causal trailing volatility (realized, on bar returns) ---
    r["rv_16"] = r["ret"].rolling(16).std().shift(1)   # ~4h
    r["rv_32"] = r["ret"].rolling(32).std().shift(1)   # ~8h
    r["rv_96"] = r["ret"].rolling(96).std().shift(1)   # ~24h
    # --- causal trailing volume regime ---
    r["vol_ma8"] = r["volume"].rolling(8).mean().shift(1)    # ~2h liquidity
    r["vol_ma96"] = r["volume"].rolling(96).mean().shift(1)  # ~24h baseline
    r["vol_ratio"] = r["vol_ma8"] / r["vol_ma96"]            # >1 busy, <1 quiet
    r["vol_lag1"] = r["volume"].shift(1)                     # prior bar raw volume
    # --- calendar (known ex-ante) ---
    r["hour"] = r.index.hour
    r["dow"] = r.index.dayofweek           # 0=Mon .. 6=Sun
    r["weekend"] = (r["dow"] >= 5).astype(int)
    # session buckets (UTC): Asia 00-07, Europe 07-13, US 13-21, LateUS/rollover 21-24
    def sess(h):
        if h < 7: return "Asia"
        if h < 13: return "Europe"
        if h < 21: return "US"
        return "LateUS"
    r["session"] = r["hour"].map(sess)
    # --- trailing trend / chop (causal) ---
    r["mom_16"] = (r["close"] / r["close"].shift(16) - 1).shift(1)  # 4h momentum, lagged
    return r


def load_trajectory(fold: int, rule: str) -> pd.DataFrame:
    p = f"{ROOT}/fold_{fold:02d}/{rule}_trajectory.parquet"
    t = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    t["pv_ret"] = t["portfolio_value"].pct_change()
    t["pos_prev"] = t["position"].shift(1)
    t["price_key"] = t["timestamp"] + pd.Timedelta(minutes=OFFSET_MIN)
    t["fold"] = fold
    t["rule"] = rule
    return t


def merge(t: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    feat_cols = ["close", "volume", "ret", "rv_16", "rv_32", "rv_96", "vol_ma8",
                 "vol_ma96", "vol_ratio", "vol_lag1", "hour", "dow", "weekend",
                 "session", "mom_16"]
    m = t.merge(r[feat_cols], left_on="price_key", right_index=True, how="left")
    return m


def pf(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else np.inf


def main():
    r = build_price_features()

    # ---- tripwire: ens_mean bar-PF must reproduce verdict graded PF per fold ----
    verdict = json.load(open(f"{ROOT}/verdict.json"))
    print("=== TRIPWIRE: ens_mean bar-PF vs stored fold metrics ===")
    all_rows = []
    for fold in FOLDS:
        em = merge(load_trajectory(fold, "ens_mean"), r)
        # join coverage check
        miss = em["close"].isna().mean()
        bar_pf = pf(em["pv_ret"].values)
        stored = json.load(open(f"{ROOT}/fold_{fold:02d}/ens_mean_metrics.json"))
        print(f"  fold{fold}: bar_pf={bar_pf:.4f}  stored_pf={stored.get('pf_bar', stored.get('profit_factor','?'))}  "
              f"join_miss={miss:.3%}  n={len(em)}")

    # ---- build pooled per-bar dataset across all solos + ens_mean ----
    frames = []
    for fold in FOLDS:
        for seed in SEEDS:
            frames.append(merge(load_trajectory(fold, f"solo_{seed}"), r))
    pool = pd.concat(frames, ignore_index=True)
    pool["dir"] = np.sign(pool["pos_prev"]).map({1.0: "long", -1.0: "short", 0.0: "flat"})
    pool["phase"] = np.where(pool["fold"].isin(DISCOVERY_FOLDS), "discovery", "confirm")
    # drop warmup NaNs in pv_ret / features
    pool = pool.dropna(subset=["pv_ret", "rv_32", "vol_ratio"]).reset_index(drop=True)
    print(f"\nPooled per-bar rows (solos): {len(pool)}  "
          f"discovery={int((pool.phase=='discovery').sum())}  confirm={int((pool.phase=='confirm').sum())}")
    print(f"overall pooled bar-PF (solos): {pf(pool['pv_ret'].values):.4f}")
    print(f"long-only bar-PF: {pf(pool.loc[pool.dir=='long','pv_ret'].values):.4f}  "
          f"short-only bar-PF: {pf(pool.loc[pool.dir=='short','pv_ret'].values):.4f}  "
          f"long share={float((pool.dir=='long').mean()):.2%}")

    pool.to_parquet("results/gmgp1_btc_canary_costcorr_wf/_loss_regime_pool.parquet")
    print("\nsaved pooled dataset -> results/gmgp1_btc_canary_costcorr_wf/_loss_regime_pool.parquet")


if __name__ == "__main__":
    main()
