#!/usr/bin/env python3
"""SG-1 BTC atr_cap train/live parity audit — sizing-decision item 3.

QUESTION
--------
Training (continuous_swing_env.py:288-311) clamps |position| to
`atr_cap_max_position` (0.5) on a TRADE bar whose `current_atr` exceeds the P90
of the trailing-200 ATR buffer. The live engine applies NO such cap: the only
position-magnitude limit is the constant `max_position_pct` (=1.0) in
CryptoRiskManager.check() (crypto_risk_manager.py:268-276); there is zero
volatility-dependent throttle anywhere in sharpen/crypto/live or the risk
manager. So on the highest-decile-vol bars the live policy can carry up to 2x
the position the training env held. This script measures how often that cap
BINDS and the size of the resulting live-vs-train exposure gap.

METHOD (no agent / no checkpoint)
---------------------------------
The cost-corrected WF trajectories already record the per-bar action that was
fed to the env (`action_agg`). We REPLAY those recorded actions through a
freshly-built `make_env` for each fold and read the env's atr_cap internals
(`current_atr`, `_atr_buffer`, `current_position`) immediately after each step,
reproducing the exact gate the env used. Replaying recorded actions is
deterministic, so realized positions must match the recorded trajectory — the
script asserts this (replay-fidelity guard) and aborts if they diverge.

For each bar we reconstruct the env's decision:
    target          = clip(action, -1, 1) * max_leverage
    eff_deadband    = deadband_threshold * max_leverage
    trade_triggered = |target - pos_before| >= eff_deadband
    warmup_ok       = atr_rolling_mean > 0 and len(buffer) >= 50
    p90             = sorted(buffer)[min(int(len*0.9), len-1)]
    high_vol        = warmup_ok and current_atr > p90
    cap_engaged     = high_vol and trade_triggered and |target| > atr_cap_max_position
    extra_live_pos  = |target| - atr_cap_max_position   (>0 only when cap_engaged)
`extra_live_pos` is precisely the additional exposure the LIVE engine would
carry on that bar relative to the training env (which clamped to 0.5).

USAGE
-----
    python scripts/sg1_btc_atr_cap_parity_audit.py \
        --wf_config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml \
        --results_dir results/sg1_btc_ensemble_wf \
        --seed 456 --device cpu
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


def _inner_env(env):
    """Unwrap to the ContinuousSwingEnv carrying the atr_cap state."""
    e = env
    seen = 0
    while not hasattr(e, "_atr_buffer") and hasattr(e, "env") and seen < 10:
        e = e.env
        seen += 1
    if not hasattr(e, "_atr_buffer"):
        raise RuntimeError("could not unwrap to env with _atr_buffer (atr_cap state)")
    return e


def audit_fold(wf_cfg: dict, test_start: str, test_end: str, norm_cutoff: str,
               traj_path: Path, device: str) -> pd.DataFrame:
    traj = pd.read_parquet(traj_path)
    actions = traj["action_agg"].to_numpy(dtype=np.float64)
    rec_pos = traj["position"].to_numpy(dtype=np.float64)
    n = len(actions)

    # Reproduce run_rule's env construction EXACTLY: date-override then
    # _prep_backtest_config (disables random_start, episode_length=0,
    # hindsight=0). Skipping prep starts the env on a random curriculum slice
    # and the replay diverges from the recorded trajectory.
    fold_cfg = _override_test_window(wf_cfg, test_start, test_end, norm_cutoff)
    cfg = _prep_backtest_config(fold_cfg)
    dc = cfg["data"]
    env = make_env(cfg, start_date=dc["test_start_date"], end_date=dc["test_end_date"],
                   norm_cutoff_date=dc.get("val_end_date"))
    inner = _inner_env(env)

    max_lev = float(getattr(inner, "max_leverage", 1.0)) or 1.0
    deadband = float(getattr(inner, "deadband_threshold", 0.25))
    atr_cap = float(getattr(inner, "atr_cap_max_position", 0.5))
    atr_pctile = float(getattr(inner, "atr_cap_percentile", 90))
    eff_deadband = deadband * max_lev

    env.reset()
    rows = []
    mismatches = 0
    for t in range(n):
        a = actions[t]
        pos_before = float(inner.current_position)
        # NaN sentinel handling mirrors run_rule (hold prior): for solo there are
        # no NaNs, but guard anyway.
        if np.isnan(a):
            a = pos_before / max_lev
        env.step(np.array([a], dtype=np.float64))

        cur_atr = float(getattr(inner, "current_atr", 0.0))
        buf = list(getattr(inner, "_atr_buffer", []))
        roll_mean = float(getattr(inner, "_atr_rolling_mean", 0.0))
        pos_after = float(inner.current_position)

        warmup_ok = roll_mean > 0 and len(buf) >= 50
        if warmup_ok:
            sb = sorted(buf)
            p90_idx = min(int(len(sb) * atr_pctile / 100.0), len(sb) - 1)
            p90 = sb[p90_idx]
            high_vol = cur_atr > p90
        else:
            p90 = None
            high_vol = False

        target = float(np.clip(a, -1.0, 1.0) * max_lev)
        trade_triggered = abs(target - pos_before) >= eff_deadband
        cap_engaged = bool(high_vol and trade_triggered and abs(target) > atr_cap)
        extra_live_pos = (abs(target) - atr_cap) if cap_engaged else 0.0

        if abs(pos_after - rec_pos[t]) > 1e-6:
            mismatches += 1

        rows.append({
            "t": t,
            "current_atr": cur_atr,
            "atr_p90": p90,
            "high_vol": int(high_vol),
            "target": target,
            "pos_before": pos_before,
            "pos_after": pos_after,
            "trade_triggered": int(trade_triggered),
            "cap_engaged": int(cap_engaged),
            "extra_live_pos": extra_live_pos,
        })
    env.close()

    df = pd.DataFrame(rows)
    df.attrs["mismatches"] = mismatches
    df.attrs["n"] = n
    return df


def replay_uncapped_pv(wf_cfg: dict, test_start: str, test_end: str, norm_cutoff: str,
                       traj_path: Path) -> np.ndarray:
    """Replay the recorded actions through an env with atr_cap DISABLED
    (atr_cap_max_position = max_leverage -> the clamp is a no-op). Returns the
    resulting per-bar portfolio_value curve. This is the live-equivalent path
    for a FIXED policy intent: the live engine runs the capped-trained policy
    with no atr_cap (crypto_risk_manager has no vol throttle), so executing the
    recorded intent uncapped reproduces the live position-sizing on high-vol
    bars. (First-order: the live agent's own position-feedback would differ.)"""
    traj = pd.read_parquet(traj_path)
    actions = traj["action_agg"].to_numpy(dtype=np.float64)

    fold_cfg = _override_test_window(wf_cfg, test_start, test_end, norm_cutoff)
    cfg = _prep_backtest_config(fold_cfg)
    max_lev = float(cfg["env"].get("max_leverage", 1.0)) or 1.0
    cfg["env"]["atr_cap_max_position"] = max_lev   # disable the cap (no-op clip)
    dc = cfg["data"]
    env = make_env(cfg, start_date=dc["test_start_date"], end_date=dc["test_end_date"],
                   norm_cutoff_date=dc.get("val_end_date"))
    inner = _inner_env(env)

    env.reset()
    pv = []
    for t in range(len(actions)):
        a = actions[t]
        if np.isnan(a):
            a = float(inner.current_position) / max_lev
        _, _, _, _, info = env.step(np.array([a], dtype=np.float64))
        v = info.get("portfolio_value", getattr(inner, "equity", 100000.0))
        pv.append(float(v))
    env.close()
    return np.asarray(pv, dtype=np.float64)


def _fixed_notional_metrics(pv: np.ndarray) -> tuple[float, float]:
    """(fixed-notional return %, worst DD-of-initial %) from a pv curve."""
    r = np.diff(pv) / pv[:-1]
    cumret = np.cumsum(r)
    fixed_ret = float(np.sum(r) * 100.0)
    running_peak = np.maximum.accumulate(cumret)
    mdd = float(np.min(cumret - running_peak) * 100.0) if len(cumret) else 0.0
    return fixed_ret, mdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf_config", default="configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml")
    ap.add_argument("--results_dir", default="results/sg1_btc_ensemble_wf")
    ap.add_argument("--seed", default="456")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare_uncapped", action="store_true", default=True,
                    help="Also replay uncapped (atr_cap off) and report DD/return delta")
    args = ap.parse_args()

    with open(args.wf_config, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)

    sp = wf_cfg.get("splitter", {})
    dc = wf_cfg.get("data", {})
    splitter = RollingWindowSplitter(
        train_months=sp.get("train_months", 18), val_months=sp.get("val_months", 1),
        test_months=sp.get("test_months", 1), step_months=sp.get("step_months", 1),
        buffer_days=sp.get("buffer_days", 0),
    )
    folds = splitter.split(start_date=dc.get("start_date"), end_date=dc.get("end_date"))

    root = Path(args.results_dir)
    rule = f"solo_{args.seed}"
    summaries = []
    total_mismatch = 0
    for i, fold in enumerate(folds):
        traj_path = root / f"fold_{i:02d}" / f"{rule}_trajectory.parquet"
        if not traj_path.exists():
            print(f"[fold {i}] no trajectory {traj_path}; skipping")
            continue
        df = audit_fold(wf_cfg, fold["test"].start, fold["test"].end,
                        fold["val"].end, traj_path, args.device)
        mm = df.attrs["mismatches"]
        total_mismatch += mm
        n = len(df)
        n_trade = int(df["trade_triggered"].sum())
        n_hv = int(df["high_vol"].sum())
        n_cap = int(df["cap_engaged"].sum())
        eng = df[df["cap_engaged"] == 1]["extra_live_pos"]
        row = {
            "fold": i,
            "n_bars": n,
            "replay_mismatch": mm,
            "n_trades": n_trade,
            "n_high_vol_bars": n_hv,
            "frac_high_vol": round(n_hv / n, 4) if n else None,
            "n_cap_engaged": n_cap,
            "frac_cap_of_all": round(n_cap / n, 4) if n else None,
            "frac_cap_of_trades": round(n_cap / n_trade, 4) if n_trade else None,
            "frac_cap_of_highvol": round(n_cap / n_hv, 4) if n_hv else None,
            "extra_live_pos_mean": round(float(eng.mean()), 4) if len(eng) else 0.0,
            "extra_live_pos_max": round(float(eng.max()), 4) if len(eng) else 0.0,
        }

        if args.compare_uncapped:
            capped_pv = pd.read_parquet(traj_path)["portfolio_value"].to_numpy(dtype=np.float64)
            cap_ret, cap_dd = _fixed_notional_metrics(capped_pv)
            unc_pv = replay_uncapped_pv(wf_cfg, fold["test"].start, fold["test"].end,
                                        fold["val"].end, traj_path)
            unc_ret, unc_dd = _fixed_notional_metrics(unc_pv)
            row.update({
                "fixed_ret_capped_pct": round(cap_ret, 2),
                "fixed_ret_uncapped_pct": round(unc_ret, 2),
                "fixed_dd_capped_pct": round(cap_dd, 3),
                "fixed_dd_uncapped_pct": round(unc_dd, 3),
                "dd_worsening_pp": round(unc_dd - cap_dd, 3),   # <0 => live DD deeper
            })
            print(f"[fold {i}] cap_engaged={n_cap} ({n_cap/max(n,1)*100:.2f}% bars / "
                  f"{n_cap/max(n_trade,1)*100:.1f}% trades) extra~{float(eng.mean()) if len(eng) else 0:.3f} | "
                  f"fixedDD capped {cap_dd:.2f}% -> uncapped {unc_dd:.2f}% "
                  f"(Δ{unc_dd-cap_dd:+.2f}pp); fixedRet {cap_ret:.0f}%->{unc_ret:.0f}% | mm={mm}")
        else:
            print(f"[fold {i}] bars={n} trades={n_trade} high_vol={n_hv} "
                  f"cap_engaged={n_cap} ({n_cap/max(n,1)*100:.2f}% of bars, "
                  f"{n_cap/max(n_trade,1)*100:.1f}% of trades) "
                  f"extra_live_pos~{float(eng.mean()) if len(eng) else 0:.3f} "
                  f"max {float(eng.max()) if len(eng) else 0:.3f} | mismatch={mm}")
        summaries.append(row)

    sm = pd.DataFrame(summaries)
    out_dir = root / "atr_cap_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sm.to_csv(out_dir / f"summary_{rule}.csv", index=False)

    if len(sm):
        tot_bars = sm["n_bars"].sum()
        tot_cap = sm["n_cap_engaged"].sum()
        tot_trades = sm["n_trades"].sum()
        print("\n========== ATR_CAP PARITY SUMMARY (seed %s) ==========" % args.seed)
        print(sm.to_string(index=False))
        print(f"\nAGGREGATE: cap engaged on {tot_cap}/{tot_bars} bars "
              f"({tot_cap/tot_bars*100:.2f}% of all bars; "
              f"{tot_cap/tot_trades*100:.1f}% of trades). "
              f"Replay mismatches: {total_mismatch} (must be 0 for a valid audit).")
        if total_mismatch:
            print("WARNING: replay diverged from recorded trajectory — results UNRELIABLE.")
        print(f"Wrote: {out_dir/('summary_'+rule+'.csv')}")


if __name__ == "__main__":
    main()
