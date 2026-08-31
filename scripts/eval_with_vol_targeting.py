#!/usr/bin/env python3
"""Phase 1b: vol-targeting overlay on SG-1-XAUUSD S488 ensemble (eval-only).

Tests whether dynamic position sizing via realized-vol scaling captures
the leverage benefit without retraining. If vol-targeting wins comparably
to constant-leverage HPO (Phase 1a), the constant-axis approach is
over-engineered.

For each annual vol target VT, position scaler at step t is:
    scaler_t = clip( target_per_bar / realized_vol_per_bar(t), [scale_min, scale_max] )
    action_scaled_t = ens_agreement(per-agent actions) × scaler_t

Then env clips to [-env.max_leverage, +env.max_leverage]. We set env.max_leverage
= scale_max at eval time so the policy (trained at 1x) can be scaled UP without
breaching env limits — tests "can a 1x-trained policy be safely deployed at >1x
via vol-aware sizing alone?"

Plan: ~/.claude/plans/plan-a-new-research-lexical-sunrise.md (Phase 1b)
Baseline: SG-1-XAUUSD S488 ensemble (PF 3.455, intraday DD 0.34% per memory)

Usage:
    python scripts/eval_with_vol_targeting.py
    python scripts/eval_with_vol_targeting.py --vol_targets 0.05 0.10 0.20 \\
        --scale_max 3.0 --vol_window 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sharpen.hpo.env_factory import make_env  # noqa: E402
from scripts.sg1_arm_gate_backtest import _prep_backtest_config  # noqa: E402
from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402
    SEED_CHECKPOINTS,
    _agg_agreement,
    _infer_actions,
    _load_agents_from_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vol-target")

# 3-min XAUUSD CFD bars per year. XAU CFD trades 23h/day Mon-Fri (closed
# Friday 22:00 UTC → Sunday 22:00 UTC) → ~23 active hours × ~252 trading
# days × (60/3) bars/hour = 115,920 bars/year.
# (B7 audit-fix 2026-04-27: previous 80,640 used 16 hr/day for an equity-
# style session, biased per-bar vol target ~20% high.)
BARS_PER_YEAR_3MIN_XAU = 252 * 23 * (60 // 3)  # = 115,920


def per_bar_realized_vol(returns: np.ndarray) -> float:
    if len(returns) < 5:
        return 0.0
    return float(np.std(returns, ddof=1))


def deannualize_vol(annual_vol: float, bars_per_year: int = BARS_PER_YEAR_3MIN_XAU) -> float:
    return annual_vol / np.sqrt(bars_per_year)


def _get_base_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def run_variant(
    config: dict,
    agents: dict,
    target_annual_vol: float | None,
    vol_window: int,
    scale_min: float,
    scale_max: float,
    env_max_leverage: float,
    device: str,
    out_dir: Path,
    tag: str,
) -> tuple[pd.DataFrame, dict]:
    """One eval variant: ens_agreement + optional vol-targeting overlay."""
    cfg = _prep_backtest_config(config)
    cfg["env"]["max_leverage"] = env_max_leverage
    data_cfg = cfg["data"]
    start, end = data_cfg["test_start_date"], data_cfg["test_end_date"]
    deadband = float(cfg.get("env", {}).get("deadband_threshold", 0.35))

    target_per_bar = deannualize_vol(target_annual_vol) if target_annual_vol else None
    log.info(
        f"[{tag}] test {start}->{end}  vol_target={target_annual_vol}  "
        f"per_bar={target_per_bar}  scale=[{scale_min},{scale_max}]  max_lev={env_max_leverage}"
    )

    env = make_env(
        cfg, start_date=start, end_date=end,
        norm_cutoff_date=data_cfg.get("val_end_date"),
    )
    base_env = _get_base_env(env)
    base_ts = getattr(base_env, "_base_timestamps", None)
    if base_ts is None and hasattr(base_env, "handler"):
        base_ts = getattr(base_env.handler, "_base_timestamps", None)

    obs, _ = env.reset()
    rows: list[dict] = []
    step = 0
    done = False

    return_window: deque = deque(maxlen=vol_window)
    last_close = float(getattr(base_env, "current_close", 0.0))

    while not done and step < 500000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)

        per_agent = _infer_actions(agents, scale_stack, priv, device)
        agg_action = _agg_agreement(per_agent, deadband)

        if target_per_bar is not None and len(return_window) >= 5:
            rv = per_bar_realized_vol(np.asarray(return_window))
            scaler = target_per_bar / max(rv, 1e-9)
            scaler = float(np.clip(scaler, scale_min, scale_max))
        else:
            scaler = 1.0

        scaled_action = agg_action * scaler
        obs, reward, terminated, truncated, info = env.step(scaled_action)
        done = terminated or truncated

        new_close = float(getattr(base_env, "current_close", last_close))
        if last_close > 0 and new_close > 0:
            return_window.append((new_close - last_close) / last_close)
        last_close = new_close

        cur = getattr(base_env, "current_step", None)
        ts = base_ts[cur - 1] if (
            base_ts is not None and cur is not None and 0 <= cur - 1 < len(base_ts)
        ) else None

        rv_now = per_bar_realized_vol(np.asarray(return_window)) if return_window else 0.0

        rows.append({
            "step": step,
            "timestamp": ts,
            "portfolio_value": float(info.get("portfolio_value", 100000.0)),
            "position": float(info.get("position", 0.0)),
            "traded": float(info.get("traded", 0.0)),
            "drawdown_pct": float(info.get("drawdown_pct", 0.0)),
            "reward": float(reward),
            "action_raw": float(agg_action[0]),
            "action_scaled": float(scaled_action[0]),
            "vol_scaler": scaler,
            "realized_vol_per_bar": rv_now,
        })
        if step % 5000 == 0:
            log.info(
                f"[{tag}] step={step} pv={rows[-1]['portfolio_value']:.2f} "
                f"scaler={scaler:.3f} rv={rv_now:.6f}"
            )
        step += 1

    env.close()
    df = pd.DataFrame(rows)
    traj_path = out_dir / f"{tag}_trajectory.parquet"
    df.to_parquet(traj_path)
    log.info(f"[{tag}] trajectory saved: {traj_path} ({len(df)} bars)")

    metrics = compute_metrics(df)
    log.info(
        f"[{tag}] PF={metrics['pf']:.3f} max_dd={metrics['max_dd']:.3%} "
        f"intraday_dd={metrics['intraday_dd']:.3%} return={metrics['total_return']:.3%} "
        f"trades={metrics['n_trades']} mean_scaler={metrics['mean_vol_scaler']:.3f}"
    )
    return df, metrics


def compute_metrics(df: pd.DataFrame) -> dict:
    pv = df["portfolio_value"].to_numpy()
    if len(pv) < 2:
        return {
            "pf": 0.0, "max_dd": 0.0, "intraday_dd": 0.0, "total_return": 0.0,
            "sortino": 0.0, "sharpe": 0.0, "n_trades": 0, "mean_vol_scaler": 0.0,
        }
    initial = float(pv[0])
    rets = np.diff(pv) / pv[:-1]

    profit = float(rets[rets > 0].sum())
    loss = float(-rets[rets < 0].sum())
    pf = profit / loss if loss > 0 else float("inf")

    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    max_dd = float(-dd.min())

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        dft = df.dropna(subset=["timestamp"]).copy()
        dft["date"] = pd.to_datetime(dft["timestamp"]).dt.date
        intra_dds = []
        for _, group in dft.groupby("date"):
            day_pv = group["portfolio_value"].to_numpy()
            if len(day_pv) > 1:
                day_peak = np.maximum.accumulate(day_pv)
                day_dd = (day_pv - day_peak) / day_peak
                intra_dds.append(-day_dd.min())
        intraday_dd = float(max(intra_dds)) if intra_dds else 0.0
    else:
        intraday_dd = max_dd

    total_return = float((pv[-1] - initial) / initial)
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else 0.0
    sharpe = float(rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
    n_trades = int(df["traded"].sum())
    mean_scaler = float(df["vol_scaler"].mean()) if "vol_scaler" in df.columns else 1.0

    return {
        "pf": pf, "max_dd": max_dd, "intraday_dd": intraday_dd,
        "total_return": total_return, "sortino": sortino, "sharpe": sharpe,
        "n_trades": n_trades, "mean_vol_scaler": mean_scaler,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/sg1_xauusd_ftmo_rehpo_l1_multiseed.yaml",
        help="L1 multiseed config — defines env + matches the S488 ensemble checkpoints",
    )
    parser.add_argument(
        "--vol_window", type=int, default=50,
        help="Rolling-bar window for realized-vol estimation (default 50 bars = ~2.5h on 3min)",
    )
    parser.add_argument(
        "--vol_targets", type=float, nargs="*",
        default=[0.05, 0.10, 0.15, 0.20],
        help="Annual vol targets (fraction). e.g. 0.10 = 10%% target annual vol",
    )
    parser.add_argument("--scale_min", type=float, default=0.25)
    parser.add_argument(
        "--scale_max", type=float, default=3.0,
        help="Position scaler ceiling — also passed as env.max_leverage at eval time",
    )
    parser.add_argument("--out_dir", default="results/sg1_xauusd_vol_targeting")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Skip vol-targeted variants (sanity check on baseline ensemble)",
    )
    parser.add_argument(
        "--constant_lev_arms", type=float, nargs="*",
        default=[3.0],
        help="B2 control: extra arms at fixed max_leverage with NO scaling. "
             "Decomposes 'vol-targeting wins' into 'leverage cap raised' vs "
             "'vol-aware sizing'. Default [3.0] matches scale_max so vol-target "
             "treatments share the same cap. Pass empty list to disable.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Device: {args.device}")
    log.info(f"Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log.info(f"Loading {len(SEED_CHECKPOINTS)} S488 ensemble checkpoints...")
    agents = _load_agents_from_paths(config, SEED_CHECKPOINTS, args.device)

    summary: list[dict] = []

    log.info("=" * 70)
    log.info("BASELINE: ens_agreement, no vol scaling, max_leverage=1.0")
    _, m = run_variant(
        config, agents, target_annual_vol=None,
        vol_window=args.vol_window, scale_min=1.0, scale_max=1.0,
        env_max_leverage=1.0, device=args.device, out_dir=out_dir, tag="baseline",
    )
    summary.append({"variant": "baseline", "vol_target": None, **m})

    if not args.baseline_only:
        # B2 control: constant-leverage arms with NO vol scaling. Isolates
        # "raised leverage cap" effect from "vol-aware sizing" — without these
        # the vol-target arms confound the two (they all run at scale_max env
        # cap while baseline runs at 1.0).
        for lev in args.constant_lev_arms:
            tag = f"constant_lev_{lev:g}x"
            log.info("=" * 70)
            log.info(f"CONTROL {tag}: max_leverage={lev}, scaler=1.0 fixed (no vol-scaling)")
            _, m = run_variant(
                config, agents, target_annual_vol=None,
                vol_window=args.vol_window, scale_min=1.0, scale_max=1.0,
                env_max_leverage=float(lev),
                device=args.device, out_dir=out_dir, tag=tag,
            )
            summary.append({"variant": tag, "vol_target": None, **m})

        for vt in args.vol_targets:
            tag = f"vol_target_{int(vt * 100):02d}pct"
            log.info("=" * 70)
            log.info(f"VARIANT {tag}: target annual vol = {vt:.1%}")
            _, m = run_variant(
                config, agents, target_annual_vol=vt,
                vol_window=args.vol_window,
                scale_min=args.scale_min, scale_max=args.scale_max,
                env_max_leverage=args.scale_max,
                device=args.device, out_dir=out_dir, tag=tag,
            )
            summary.append({"variant": tag, "vol_target": vt, **m})

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "vol_targeting_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_json = out_dir / "vol_targeting_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=str))

    log.info("=" * 70)
    log.info("SUMMARY (test window):")
    log.info(
        f"  {'variant':<25} {'PF':>6}  {'max_DD':>8}  {'intra_DD':>9}  "
        f"{'return':>8}  {'sortino':>8}  {'trades':>7}  {'mean_scl':>9}",
    )
    for row in summary:
        pf_s = f"{row['pf']:.3f}" if row['pf'] != float('inf') else "  inf"
        log.info(
            f"  {row['variant']:<25} {pf_s:>6}  {row['max_dd']:>7.2%}  "
            f"{row['intraday_dd']:>8.2%}  {row['total_return']:>7.2%}  "
            f"{row['sortino']:>8.3f}  {row['n_trades']:>7d}  {row['mean_vol_scaler']:>8.3f}",
        )
    log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
