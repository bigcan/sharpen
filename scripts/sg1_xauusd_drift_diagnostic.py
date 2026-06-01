#!/usr/bin/env python3
"""SG-1 XAUUSD drift diagnostic — replay the live ensemble (WF fold-07,
seeds 42/2025/3141) on out-of-fold OANDA bars and capture per-seed actions
to localize the 2026-04-30 drift_crit cause.

Hypotheses we are testing (cannot distinguish from the LIVE-run telemetry
because per-seed actions were not logged):
  A. Ensemble disagreement   (seeds split → ens_agreement → 0)
  B. Joint policy collapse   (each seed individually outputs in deadband)
  C. Observation drift       (input features moved out of training regime)

This script computes, for two windows:
  W1 = Mar 2026  (fold 7 test, the certified window the live deploy trusted)
  W2 = Apr 2026  (out-of-fold; closest proxy for live regime since OANDA
                   parquet ends 2026-04-23 and live drift onset was 04-28)

For each window:
  - per-seed action histograms        → tests B
  - inter-seed agreement rate         → tests A
  - aggregated ens_agreement output   → reproduces live dead_frac signal
  - private-state / observation summary stats   → tests C indirectly

Outputs: results/sg1_xauusd_drift_diagnostic/{window}_trajectory.parquet
         results/sg1_xauusd_drift_diagnostic/summary.json
"""
from __future__ import annotations
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.hpo.env_factory import make_env  # noqa: E402
from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402
    _agg_agreement,
    _infer_actions,
    _resolve_fold_checkpoints,
)
from scripts.sg1_arm_gate_backtest import _build_sac_agent, _prep_backtest_config  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("drift-diagnostic")

WINDOWS: Dict[str, Tuple[str, str]] = {
    # (test_start, test_end) — both inclusive at start, exclusive-ish at end
    "fold7_test_mar2026": ("2026-03-01", "2026-03-31"),
    "out_of_fold_apr2026_oanda": ("2026-04-01", "2026-04-23"),
    # Extended-parquet replay: OANDA + cTrader DEMO concat, covering the
    # actual live drift window (Apr 22-30). Same bars the live agent saw
    # (cTrader and OANDA are bar-by-bar comparable to <$0.50 close delta).
    "drift_window_apr22_30_ctrader": ("2026-04-22", "2026-04-30"),
}
# Override data path for windows that need the extended parquet
WINDOW_DATA_OVERRIDE = {
    "drift_window_apr22_30_ctrader": "data/oanda/xauusd_m1_m_ext_ctrader.parquet",
}
NORM_CUTOFF = "2026-02-28"  # fold-7 val end → freezes EMA-Z at deploy-time state
SEEDS = [42, 2025, 3141]
FOLD = 7
DEADBAND = 0.35  # ens_agreement deadband, matches live config


def build_window_config(base_config: dict, start: str, end: str) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("data", {})
    cfg["data"]["test_start_date"] = start
    cfg["data"]["test_end_date"] = end
    cfg["data"]["val_end_date"] = NORM_CUTOFF
    # _prep_backtest_config wants train/val markers; supply benign ones
    cfg["data"].setdefault("train_start_date", "2025-08-01")
    cfg["data"].setdefault("train_end_date", "2026-01-31")
    cfg["data"].setdefault("val_start_date", "2026-02-01")
    return cfg


def replay_window(label: str, cfg: dict, agents: Dict[int, object], device: str,
                  out_dir: Path) -> dict:
    cfg = _prep_backtest_config(cfg)
    start = cfg["data"]["test_start_date"]
    end = cfg["data"]["test_end_date"]
    log.info(f"[{label}] window {start} -> {end}  norm_cutoff={cfg['data']['val_end_date']}")

    env = make_env(cfg, start_date=start, end_date=end,
                   norm_cutoff_date=cfg["data"]["val_end_date"])

    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "timestamps"):
        base_env = base_env.env
    base_ts = getattr(base_env, "timestamps", None)
    if base_ts is None and hasattr(base_env, "handler"):
        base_ts = getattr(base_env.handler, "_base_timestamps", None)

    obs, _ = env.reset()
    rows = []
    step = 0
    done = False

    while not done and step < 500_000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)

        per_agent = _infer_actions(agents, scale_stack, priv, device)
        action_agg = _agg_agreement(per_agent, DEADBAND)
        obs, _, terminated, truncated, info = env.step(action_agg)
        done = terminated or truncated

        cur = getattr(base_env, "current_step", None)
        ts = base_ts[cur - 1] if (base_ts is not None and cur is not None and 0 <= cur - 1 < len(base_ts)) else None

        row = {
            "step": step, "timestamp": ts,
            "action_agg": float(action_agg[0]),
            "private_first": float(obs["private"][0]) if hasattr(obs, "__contains__") and "private" in obs else None,
        }
        for s, act in per_agent.items():
            row[f"action_seed_{s}"] = float(act[0])
        rows.append(row)
        step += 1
        if step % 5000 == 0:
            log.info(f"[{label}] step={step}")

    env.close()
    df = pd.DataFrame(rows)
    out_path = out_dir / f"{label}_trajectory.parquet"
    df.to_parquet(out_path)
    log.info(f"[{label}] trajectory saved: {out_path} ({len(df)} bars)")
    return summarize_window(label, df)


def summarize_window(label: str, df: pd.DataFrame) -> dict:
    """Compute per-seed + ensemble distribution stats + agreement rates."""
    out = {"label": label, "n_bars": int(len(df))}

    seed_cols = [c for c in df.columns if c.startswith("action_seed_")]
    seeds = [int(c.split("_")[-1]) for c in seed_cols]

    # Per-seed individual deadband fractions (would the seed alone HOLD?)
    for c, s in zip(seed_cols, seeds):
        a = df[c].values
        out[f"seed_{s}_dead_frac"] = float((np.abs(a) < DEADBAND).mean())
        out[f"seed_{s}_long_frac"] = float((a > DEADBAND).mean())
        out[f"seed_{s}_short_frac"] = float((a < -DEADBAND).mean())
        out[f"seed_{s}_sat_frac"] = float((np.abs(a) > 1 - DEADBAND).mean())
        out[f"seed_{s}_mean_abs"] = float(np.mean(np.abs(a)))
        out[f"seed_{s}_p25"] = float(np.percentile(a, 25))
        out[f"seed_{s}_p50"] = float(np.percentile(a, 50))
        out[f"seed_{s}_p75"] = float(np.percentile(a, 75))

    # Per-bar directional labels: -1/0/+1 (using ens_agreement deadband)
    labels = np.zeros((len(seed_cols), len(df)), dtype=int)
    for i, c in enumerate(seed_cols):
        a = df[c].values
        labels[i] = 0
        labels[i][a > DEADBAND] = 1
        labels[i][a < -DEADBAND] = -1

    # Inter-seed agreement diagnostics
    n_long = (labels == 1).sum(axis=0)
    n_short = (labels == -1).sum(axis=0)
    n_flat = (labels == 0).sum(axis=0)

    out["agree_3way_long_frac"] = float((n_long == 3).mean())
    out["agree_3way_short_frac"] = float((n_short == 3).mean())
    out["agree_3way_flat_frac"] = float((n_flat == 3).mean())   # all in deadband
    out["agree_2of3_directional_frac"] = float(((n_long >= 2) | (n_short >= 2)).mean())
    out["disagree_split_frac"] = float(
        ((n_long >= 1) & (n_short >= 1)).mean()
    )  # at least one long AND one short → ens_agreement → 0 (mixed signal)

    # Aggregated ensemble distribution (matches live observation)
    a_agg = df["action_agg"].values
    out["ensemble_dead_frac"] = float((np.abs(a_agg) < DEADBAND).mean())
    out["ensemble_long_frac"] = float((a_agg > DEADBAND).mean())
    out["ensemble_short_frac"] = float((a_agg < -DEADBAND).mean())
    out["ensemble_sat_frac"] = float((np.abs(a_agg) > 1 - DEADBAND).mean())

    # Decompose: when ensemble is in deadband, was it because all-flat OR mixed?
    in_dead = np.abs(a_agg) < DEADBAND
    if in_dead.any():
        out["dead_due_to_all_flat_frac"] = float(((labels == 0).all(axis=0) & in_dead).sum() / in_dead.sum())
        out["dead_due_to_split_frac"] = float(((n_long >= 1) & (n_short >= 1) & in_dead).sum() / in_dead.sum())
        out["dead_due_to_minority_directional_frac"] = float(
            (((n_long == 1) & (n_short == 0) | (n_short == 1) & (n_long == 0)) & in_dead).sum() / in_dead.sum()
        )
    else:
        out["dead_due_to_all_flat_frac"] = None
        out["dead_due_to_split_frac"] = None
        out["dead_due_to_minority_directional_frac"] = None

    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sg1_xauusd_ftmo_rehpo_wf_multiseed.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", default="results/sg1_xauusd_drift_diagnostic")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Resolve fold-07 checkpoints + load agents (single load, reused across windows)
    pattern = base_cfg["ensemble"]["checkpoint_pattern"]
    ckpts = _resolve_fold_checkpoints(pattern, SEEDS, FOLD)
    log.info(f"checkpoints resolved: {json.dumps(ckpts, indent=2)}")

    # Build a window config to give _build_sac_agent a complete cfg
    template_cfg = build_window_config(base_cfg, *WINDOWS["fold7_test_mar2026"])
    template_cfg = _prep_backtest_config(template_cfg)
    agents = {}
    for s, ckpt in ckpts.items():
        a = _build_sac_agent(template_cfg, args.device)
        a.load(ckpt)
        agents[s] = a
        log.info(f"seed {s}: loaded {ckpt}")

    summaries = []
    for label, (start, end) in WINDOWS.items():
        cfg = build_window_config(base_cfg, start, end)
        if label in WINDOW_DATA_OVERRIDE:
            cfg.setdefault("data", {})["file_path"] = WINDOW_DATA_OVERRIDE[label]
            log.info(f"[{label}] data path overridden: {cfg['data']['file_path']}")
        s = replay_window(label, cfg, agents, args.device, out_dir)
        summaries.append(s)
        log.info(f"[{label}] dead_frac={s['ensemble_dead_frac']:.3f}  "
                 f"agree_2of3={s['agree_2of3_directional_frac']:.3f}  "
                 f"disagree_split={s['disagree_split_frac']:.3f}  n={s['n_bars']}")

    # Compare to LIVE telemetry (from kill_file + WandB summary)
    live_ref = {
        "label": "live_2026-04-22_to_2026-04-30_kill_event",
        "ensemble_dead_frac_baseline_train": 0.5079,
        "ensemble_dead_frac_live_at_crit": 0.8096,
        "ensemble_sat_frac_baseline_train": 0.0,
        "ensemble_sat_frac_live_at_crit": 0.0,
        "n_bars_at_crit": 935,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "live_reference": live_ref,
            "windows": summaries,
            "config_used": args.config,
            "checkpoints": ckpts,
            "deadband": DEADBAND,
            "norm_cutoff": NORM_CUTOFF,
        }, f, indent=2, default=str)
    log.info(f"summary written: {summary_path}")

    # Pretty-print verdict
    print("\n========== DIAGNOSTIC VERDICT ==========")
    print(f"Training-time baseline: ensemble_dead_frac = {live_ref['ensemble_dead_frac_baseline_train']:.3f}")
    print(f"Live at crit (n=935):   ensemble_dead_frac = {live_ref['ensemble_dead_frac_live_at_crit']:.3f}  (delta = +0.302)")
    for s in summaries:
        print(f"  {s['label']:<40}  ensemble_dead_frac = {s['ensemble_dead_frac']:.3f}  "
              f"  per-seed dead = "
              f"{s['seed_42_dead_frac']:.2f}/{s['seed_2025_dead_frac']:.2f}/{s['seed_3141_dead_frac']:.2f}  "
              f"  disagree_split = {s['disagree_split_frac']:.3f}")


if __name__ == "__main__":
    main()
