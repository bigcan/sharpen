#!/usr/bin/env python3
"""SG-1 BTC — is vol-conditional position sizing a good idea? (cheap, no-GPU eval)

CONTEXT
-------
The binary `atr_cap` (continuous_swing_env.py:288-305) clamps |position| to 0.5
on the top-decile-vol bars; the live engine applies NO such throttle. The
operator's instinct is "volatility is our friend — don't cap." Before spending
GPU-days retraining under a *continuous* vol-target sizer (the principled
version of atr_cap), this script evaluates whether vol-conditional sizing is
even worth it FOR THIS STRATEGY, as a first-order screen on the already-recorded
cost-corrected WF trajectories.

It can confidently KILL the idea (cheap to falsify); it cannot CONFIRM it (a
differently-trained policy would act differently — only a retrain + WF confirms).
That asymmetry is the point: falsify first, spend GPU only if it survives.

METHOD (no agent, no checkpoint, no env change)
-----------------------------------------------
1. One env replay per (seed, fold) reproduces the env's per-bar internals
   deterministically (same harness as sg1_btc_atr_cap_parity_audit.py). We
   extract per bar: the policy INTENT (clip(action)*max_lev), `current_atr`, the
   CAUSAL trailing-200 ATR percentiles p50..p90, `current_close`, and the
   recorded realized position / portfolio_value.
   GUARD A (replay fidelity): env-replayed position must equal recorded position.

2. price_return_t = (close_t - close_{t-1}) / close_{t-1} (gap_detection is off
   for BTC 24/7). The fixed-notional identity (sg1_btc_fixed_notional_backtest.py)
   is r_t = position_t * price_return_t - fee_frac * |Δposition_t|, equity-
   independent. GUARD B (primitive fidelity): plugging the RECORDED positions
   into this identity must reproduce the recorded per-bar return diff(pv)/pv.

3. Given validated primitives, ANY sizer s(atr) is replayed analytically via the
   env's deadband loop:
       tgt_scaled = clip(intent * s_t, -max_lev, max_lev)
       trade iff |tgt_scaled - pos| >= deadband*max_lev  -> pos = tgt_scaled
       r_t = pos_t*price_return_t - fee_frac*|Δpos_t|
   GUARD C (sizer fidelity): the binary_cap sizer (env-exact ordering) must
   reproduce the recorded fixed-notional return/DD; the uncapped sizer must
   reproduce sg1_btc_atr_cap_parity_audit's uncapped numbers.

STEP 0 (premise test): bucket bars by current_atr decile; per bucket report the
policy's realized capture and its risk-adjustment under the UNCAPPED intent path.
If high-vol deciles show strong, clean risk-adjusted capture -> vol IS the
friend, throttling is a bad idea. If they show degraded ratio / fat left tails
-> there is something for a vol-target to harvest.

STEP 1 (sizer sweep): per-fold fixed-notional return + DD for
uncapped / binary_cap / continuous one-sided (de-lever only) / continuous
symmetric (also levers up in calm). Pre-registered GO/NO-GO vs the uncapped
endpoint, with special attention to the worst-vol fold (the tail the cap exists
to defend).

USAGE
-----
    python scripts/sg1_btc_vol_target_eval.py \
        --wf_config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml \
        --results_dir results/sg1_btc_ensemble_wf \
        --seeds 456 2025
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sharpen.hpo.env_factory import make_env  # noqa: E402
from sharpen.data.splitter import RollingWindowSplitter  # noqa: E402
from sharpen.config_utils import _prep_backtest_config  # noqa: E402
from scripts.sg1_btc_velotrade_ensemble_eval import _override_test_window  # noqa: E402

PCTS = [50, 60, 70, 80, 90]


def _inner_env(env):
    e, seen = env, 0
    while not hasattr(e, "_atr_buffer") and hasattr(e, "env") and seen < 10:
        e = e.env
        seen += 1
    if not hasattr(e, "_atr_buffer"):
        raise RuntimeError("could not unwrap to env with _atr_buffer")
    return e


def extract_fold(wf_cfg: dict, test_start: str, test_end: str, norm_cutoff: str,
                 traj_path: Path) -> dict:
    """Replay recorded actions; return validated per-bar primitive arrays."""
    traj = pd.read_parquet(traj_path)
    actions = traj["action_agg"].to_numpy(dtype=np.float64)
    rec_pos = traj["position"].to_numpy(dtype=np.float64)
    rec_pv = traj["portfolio_value"].to_numpy(dtype=np.float64)
    n = len(actions)

    fold_cfg = _override_test_window(wf_cfg, test_start, test_end, norm_cutoff)
    cfg = _prep_backtest_config(fold_cfg)
    dc = cfg["data"]
    env = make_env(cfg, start_date=dc["test_start_date"], end_date=dc["test_end_date"],
                   norm_cutoff_date=dc.get("val_end_date"))
    inner = _inner_env(env)

    max_lev = float(getattr(inner, "max_leverage", 1.0)) or 1.0
    deadband = float(getattr(inner, "deadband_threshold", 0.25))
    atr_cap = float(getattr(inner, "atr_cap_max_position", 0.5))
    taker = float(getattr(inner, "taker_fee", 0.0))
    slip = float(getattr(inner, "slippage_base_bps", 0.0))
    fee_frac = taker + slip / 10000.0           # cost per unit |Δposition|

    env.reset()
    intent = np.zeros(n)
    cur_atr = np.zeros(n)
    pctl = {p: np.zeros(n) for p in PCTS}
    close = np.zeros(n)
    mism = 0
    for t in range(n):
        a = actions[t]
        pos_before = float(inner.current_position)
        if np.isnan(a):
            a = pos_before / max_lev
        env.step(np.array([a], dtype=np.float64))
        cur_atr[t] = float(getattr(inner, "current_atr", 0.0))
        close[t] = float(getattr(inner, "current_close", 0.0))
        buf = list(getattr(inner, "_atr_buffer", []))
        roll = float(getattr(inner, "_atr_rolling_mean", 0.0))
        if roll > 0 and len(buf) >= 50:
            sb = sorted(buf)
            for p in PCTS:
                idx = min(int(len(sb) * p / 100.0), len(sb) - 1)
                pctl[p][t] = sb[idx]
        else:
            for p in PCTS:
                pctl[p][t] = np.nan          # warmup: no throttle reference yet
        intent[t] = float(np.clip(a, -1.0, 1.0) * max_lev)
        if abs(float(inner.current_position) - rec_pos[t]) > 1e-6:
            mism += 1
    env.close()

    # price_return from close (gap detection off for BTC 24/7)
    pr = np.zeros(n)
    pr[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] > 0, close[:-1], np.nan)
    pr = np.nan_to_num(pr, nan=0.0)

    # GUARD B: recorded positions -> identity reproduces recorded per-bar return
    rec_r = np.zeros(n)
    rec_r[1:] = np.diff(rec_pv) / rec_pv[:-1]
    dpos_rec = np.abs(np.diff(rec_pos, prepend=0.0))
    recon_r = rec_pos * pr - fee_frac * dpos_rec
    # bar 0 establishment cost not in diff(pv) convention -> compare from t>=1
    primitive_err = float(np.max(np.abs(recon_r[1:] - rec_r[1:]))) if n > 1 else 0.0

    return dict(n=n, intent=intent, cur_atr=cur_atr, pctl=pctl, pr=pr,
                rec_pos=rec_pos, rec_r=rec_r, max_lev=max_lev, deadband=deadband,
                atr_cap=atr_cap, fee_frac=fee_frac, mism=mism,
                primitive_err=primitive_err)


# ---- sizers: map (intent_t, atr arrays at t) -> scale in [0, +inf) ----------
def sizer_uncapped(d, t):
    return 1.0


def make_onesided(target_pct: int, k: float):
    def s(d, t):
        ref = d["pctl"][target_pct][t]
        atr = d["cur_atr"][t]
        if not np.isfinite(ref) or ref <= 0 or atr <= 0:
            return 1.0
        return float(min(1.0, (ref / atr) ** k))   # de-lever only when atr>ref
    return s


def make_symmetric(target_pct: int, lo: float, hi: float):
    def s(d, t):
        ref = d["pctl"][target_pct][t]
        atr = d["cur_atr"][t]
        if not np.isfinite(ref) or ref <= 0 or atr <= 0:
            return 1.0
        return float(np.clip(ref / atr, lo, hi))    # also levers up in calm
    return s


def replay_sizer(d, scale_fn, binary_cap_exact: bool = False) -> tuple[float, float, np.ndarray]:
    """Deadband-loop replay -> (fixed_return_pct, fixed_dd_of_initial_pct, r[])."""
    n, intent, pr = d["n"], d["intent"], d["pr"]
    max_lev, deadband, atr_cap, fee = d["max_lev"], d["deadband"], d["atr_cap"], d["fee_frac"]
    eff_db = deadband * max_lev
    pos = 0.0
    positions = np.zeros(n)
    for t in range(n):
        tgt = intent[t]
        if binary_cap_exact:
            # env-exact ordering: deadband first, then clamp-if-high-vol, re-deadband
            delta = tgt - pos
            if abs(delta) >= eff_db:
                ref = d["pctl"][90][t]
                high_vol = np.isfinite(ref) and d["cur_atr"][t] > ref
                if high_vol:
                    tgt = float(np.clip(tgt, -atr_cap, atr_cap))
                    delta = tgt - pos
                if abs(delta) >= eff_db:
                    pos = float(np.clip(pos + delta, -max_lev, max_lev))
        else:
            tgt_scaled = float(np.clip(tgt * scale_fn(d, t), -max_lev, max_lev))
            if abs(tgt_scaled - pos) >= eff_db:
                pos = tgt_scaled
        positions[t] = pos
    dpos = np.abs(np.diff(positions, prepend=0.0))
    r = positions * pr - fee * dpos
    cumret = np.cumsum(r[1:])                     # match diff(pv) convention (skip bar 0)
    fixed_ret = float(np.sum(r[1:]) * 100.0)
    if len(cumret):
        dd = float(np.min(cumret - np.maximum.accumulate(cumret)) * 100.0)
    else:
        dd = 0.0
    return fixed_ret, dd, r


def step0_buckets(folds_data: list[dict], n_buckets: int = 10) -> pd.DataFrame:
    """Pool bars across folds; bucket by current_atr; report uncapped-intent
    realized capture + risk-adjustment per bucket."""
    atr, pr, pos = [], [], []
    for d in folds_data:
        # uncapped-intent realized position path (deadband applied, no throttle)
        _, _, _ = replay_sizer(d, sizer_uncapped)
        # rebuild the position path for per-bar stats
        n, intent, max_lev, eff_db = d["n"], d["intent"], d["max_lev"], d["deadband"] * d["max_lev"]
        p = 0.0
        pp = np.zeros(n)
        for t in range(n):
            tgt = float(np.clip(intent[t], -max_lev, max_lev))
            if abs(tgt - p) >= eff_db:
                p = tgt
            pp[t] = p
        atr.append(d["cur_atr"][1:])
        pr.append(d["pr"][1:])
        pos.append(pp[1:])
    atr = np.concatenate(atr)
    pr = np.concatenate(pr)
    pos = np.concatenate(pos)
    bar_pnl = pos * pr                            # realized per-bar capture (pre-cost)

    qs = np.quantile(atr, np.linspace(0, 1, n_buckets + 1))
    qs[-1] = np.inf
    rows = []
    for b in range(n_buckets):
        m = (atr >= qs[b]) & (atr < qs[b + 1])
        if m.sum() < 20:
            continue
        x = bar_pnl[m]
        neg = x[x < 0]
        rows.append({
            "atr_decile": b + 1,
            "n_bars": int(m.sum()),
            "atr_lo": round(float(qs[b]), 6),
            "mean_pnl_bps": round(float(x.mean()) * 1e4, 3),
            "std_pnl_bps": round(float(x.std()) * 1e4, 3),
            "sharpe_bar": round(float(x.mean() / x.std()) if x.std() > 0 else 0.0, 4),
            "hit_rate": round(float((x > 0).mean()), 4),
            "cvar5_bps": round(float(np.mean(np.sort(x)[:max(1, int(0.05 * len(x)))])) * 1e4, 3),
            "mean_neg_bps": round(float(neg.mean()) * 1e4, 3) if len(neg) else 0.0,
            "mean_abs_pos": round(float(np.abs(pos[m]).mean()), 4),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf_config", default="configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml")
    ap.add_argument("--results_dir", default="results/sg1_btc_ensemble_wf")
    ap.add_argument("--seeds", nargs="*", default=["456", "2025"])
    args = ap.parse_args()

    with open(args.wf_config, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)
    sp, dc = wf_cfg.get("splitter", {}), wf_cfg.get("data", {})
    splitter = RollingWindowSplitter(
        train_months=sp.get("train_months", 18), val_months=sp.get("val_months", 1),
        test_months=sp.get("test_months", 1), step_months=sp.get("step_months", 1),
        buffer_days=sp.get("buffer_days", 0))
    folds = splitter.split(start_date=dc.get("start_date"), end_date=dc.get("end_date"))
    root = Path(args.results_dir)
    out_dir = root / "vol_target_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sizer panel
    sizers = {"uncapped": ("u", sizer_uncapped)}
    for tp in (50, 60, 70):
        for k in (1.0, 2.0):
            sizers[f"onesided_p{tp}_k{k}"] = ("c", make_onesided(tp, k))
    sizers["symmetric_p50_0.3-1.5"] = ("c", make_symmetric(50, 0.3, 1.5))
    sizers["symmetric_p50_0.3-2.0"] = ("c", make_symmetric(50, 0.3, 2.0))
    # Flat (vol-BLIND) de-leveraging — control: cuts every bar equally. If this
    # beats vol-targeting on return/DD, the right absolute-DD tool is a lower
    # max_leverage, not a vol-conditional throttle.
    for c in (0.7, 0.5):
        sizers[f"flat_{c}"] = ("f", (lambda cc: (lambda d, t: cc))(c))

    for seed in args.seeds:
        rule = f"solo_{seed}"
        folds_data, sizer_rows, guard = [], [], []
        for i, fold in enumerate(folds):
            tp = root / f"fold_{i:02d}" / f"{rule}_trajectory.parquet"
            if not tp.exists():
                print(f"[{rule} fold {i}] missing {tp}; skip")
                continue
            d = extract_fold(wf_cfg, fold["test"].start, fold["test"].end,
                             fold["val"].end, tp)
            d["fold"] = i
            folds_data.append(d)
            guard.append((i, d["mism"], d["primitive_err"]))

            # binary_cap (env-exact) + uncapped validation + continuous panel
            bret, bdd, _ = replay_sizer(d, None, binary_cap_exact=True)
            row = {"fold": i, "binary_cap_ret": round(bret, 2), "binary_cap_dd": round(bdd, 3)}
            for name, (_, fn) in sizers.items():
                rret, rdd, _ = replay_sizer(d, fn)
                row[f"{name}_ret"] = round(rret, 2)
                row[f"{name}_dd"] = round(rdd, 3)
            sizer_rows.append(row)

        if not folds_data:
            print(f"[{rule}] no folds; skip")
            continue

        # ---- guards ----
        max_mism = max(g[1] for g in guard)
        max_perr = max(g[2] for g in guard)
        print(f"\n===== {rule} =====")
        print(f"GUARD A replay-mismatch max = {max_mism} (must be 0)")
        print(f"GUARD B primitive-recon max abs err = {max_perr:.2e} fractional "
              f"(= {max_perr*1e4:.3f} bps/bar; tol 1 bp)")
        if max_mism or max_perr > 1e-4:
            print("  !! GUARD FAILED — primitives unreliable for this seed")

        # ---- Step 0 ----
        s0 = step0_buckets(folds_data)
        s0.to_csv(out_dir / f"step0_atr_buckets_{rule}.csv", index=False)
        print("\n-- STEP 0: realized capture by ATR decile (uncapped intent) --")
        print(s0.to_string(index=False))

        # ---- Step 1 ----
        s1 = pd.DataFrame(sizer_rows).sort_values("fold")
        s1.to_csv(out_dir / f"step1_sizer_perfold_{rule}.csv", index=False)

        # aggregate: median return, worst-fold DD, median return/DD
        agg = []
        cols = ["binary_cap"] + list(sizers.keys())
        worst_fold = int(s1.loc[s1["uncapped_dd"].idxmin(), "fold"])  # deepest-DD fold under uncapped
        for c in cols:
            rcol, dcol = s1[f"{c}_ret"], s1[f"{c}_dd"]
            rovers = (rcol / dcol.abs().replace(0, np.nan))
            wf_dd = float(s1.loc[s1["fold"] == worst_fold, f"{c}_dd"].iloc[0])
            agg.append({
                "sizer": c,
                "median_ret": round(float(rcol.median()), 2),
                "min_ret": round(float(rcol.min()), 2),
                "worst_dd": round(float(dcol.min()), 3),
                f"dd_fold{worst_fold}": round(wf_dd, 3),
                "median_ret_over_dd": round(float(rovers.median()), 1),
            })
        agg = pd.DataFrame(agg)
        agg.to_csv(out_dir / f"step1_sizer_agg_{rule}.csv", index=False)
        print(f"\n-- STEP 1: sizer aggregate (worst uncapped DD = fold {worst_fold}) --")
        print(agg.to_string(index=False))

        # ---- pre-registered verdict ----
        unc = agg[agg["sizer"] == "uncapped"].iloc[0]
        cont = agg[agg["sizer"].str.startswith(("onesided", "symmetric"))]
        # GO candidate: holds >=95% of uncapped median return AND improves worst-fold DD
        cand = cont[(cont["median_ret"] >= 0.95 * unc["median_ret"]) &
                    (cont[f"dd_fold{worst_fold}"] > unc[f"dd_fold{worst_fold}"])]
        print(f"\n-- VERDICT ({rule}) --")
        print(f"uncapped: median_ret={unc['median_ret']}%, "
              f"fold{worst_fold} DD={unc[f'dd_fold{worst_fold}']}%, "
              f"median ret/DD={unc['median_ret_over_dd']}x")
        if len(cand):
            best = cand.sort_values("median_ret_over_dd", ascending=False).iloc[0]
            print(f"GO-candidate: {best['sizer']} — median_ret={best['median_ret']}% "
                  f"(>=95% of uncapped), fold{worst_fold} DD={best[f'dd_fold{worst_fold}']}% "
                  f"(better), median ret/DD={best['median_ret_over_dd']}x")
            print("  => a continuous sizer pays for itself on first-order replay -> "
                  "GO to bootstrap (Step 2) then retrain.")
        else:
            print("NO continuous config holds >=95% return AND improves the worst-fold DD.")
            print("  => vol-targeting does not pay for itself here on first-order replay "
                  "-> NO-GO; keep uncapped, drop (b).")

    print(f"\nWrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
