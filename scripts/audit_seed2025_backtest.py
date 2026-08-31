"""Audit seed2025 backtest — replay user-chosen window with per-bar tracing.

Args (all optional — defaults match the original A1 test window):
    --start_date YYYY-MM-DD
    --end_date   YYYY-MM-DD
    --norm_cutoff YYYY-MM-DD
    --tag <slug>           # output filename suffix
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path("/workspace/DeepScalper")
sys.path.insert(0, str(REPO))
os.environ.setdefault("WANDB_MODE", "disabled")

from sharpen.hpo.env_factory import make_env  # noqa: E402

CONFIG = REPO / "configs/sg1_btc_velotrade_l1_multiseed_extended.yaml"
CHECKPOINT = REPO / "checkpoints/sg1-btc-velotrade-l1-multiseed-extended-seed2025_20260505_074754/checkpoint_final.pth"


def _build_backtest_config(cfg: dict) -> dict:
    """Mirror the run_full_pipeline.run_backtest() patches."""
    bt = copy.deepcopy(cfg)
    bt.setdefault("env", {})["private_state_augment_prob"] = 0.0
    bt["env"].setdefault("reward", {})["hindsight_weight"] = 0.0
    bt["env"]["episode_length"] = 0
    bt["env"]["random_start"] = False
    return bt


def _build_agent(cfg: dict, device: str):
    from sharpen.agents.sac.sac_agent import SACAgent

    network_config = dict(cfg.get("network", {}))
    scales = cfg.get("features", {}).get("scales", cfg.get("env", {}).get("scales", [3, 15, 60]))
    network_config["n_scales"] = len(scales)
    network_config["action_space_dims"] = (5, 9)  # SAC overrides; placeholder

    sac_cfg = cfg.get("agents", {}).get("sac", {})
    agent = SACAgent(
        network_config=network_config,
        lr_actor=sac_cfg.get("lr_actor", 3e-4),
        lr_critic=sac_cfg.get("lr_critic", 3e-4),
        lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
        gamma=sac_cfg.get("gamma", 0.99),
        tau=sac_cfg.get("tau", 0.005),
        batch_size=sac_cfg.get("batch_size", 256),
        buffer_size=100,
        initial_alpha=sac_cfg.get("initial_alpha", 0.2),
        device=device,
    )
    return agent


def _find_base_env(env):
    """Unwrap signal-gate / risk-shaping to reach ContinuousSwingEnv for diagnostics."""
    cur = env
    for _ in range(8):
        if cur.__class__.__name__ == "ContinuousSwingEnv":
            return cur
        cur = getattr(cur, "env", None)
        if cur is None:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", default=None)
    ap.add_argument("--end_date", default=None)
    ap.add_argument("--norm_cutoff", default=None)
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bt_cfg = _build_backtest_config(cfg)

    start = args.start_date or bt_cfg["data"]["test_start_date"]
    end = args.end_date or bt_cfg["data"]["test_end_date"]
    norm_cut = args.norm_cutoff or bt_cfg["data"]["val_end_date"]
    tag = args.tag

    print(f"[audit] device={device}  window={start} → {end}  norm_cutoff={norm_cut}  tag={tag}")
    env = make_env(bt_cfg, start_date=start, end_date=end, norm_cutoff_date=norm_cut)
    base_env = _find_base_env(env)

    agent = _build_agent(cfg, device)
    print(f"[audit] loading checkpoint: {CHECKPOINT}")
    agent.load(str(CHECKPOINT))

    obs, _ = env.reset()

    rows = []
    step = 0
    done = False
    cumulative_traded = 0
    cumulative_pos_change = 0.0

    while not done and step < 200_000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        pred = agent.predict(scale_stack, priv, deterministic=True)
        action = pred[0].cpu().numpy().reshape(-1)

        prev_close = float(getattr(base_env, "prev_close", 0.0)) if base_env else 0.0
        prev_position = float(getattr(base_env, "current_position", 0.0)) if base_env else 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        cur_close = float(getattr(base_env, "current_close", 0.0)) if base_env else 0.0
        cur_position = float(info.get("position", getattr(base_env, "current_position", 0.0)))
        equity = float(info.get("portfolio_value", 0.0))
        traded_flag = bool(info.get("traded", False) or info.get("switched", False))
        pos_delta = abs(cur_position - prev_position)
        if traded_flag and pos_delta > 1e-9:
            cumulative_traded += 1
            cumulative_pos_change += pos_delta

        price_return = ((cur_close - prev_close) / prev_close) if prev_close > 0 else 0.0
        bar_pnl_frac = cur_position * price_return  # fraction-of-equity bar return

        rows.append({
            "step": step,
            "action": float(action[0]) if action.size else 0.0,
            "position": cur_position,
            "prev_position": prev_position,
            "equity": equity,
            "prev_close": prev_close,
            "cur_close": cur_close,
            "price_return": price_return,
            "bar_pnl_frac": bar_pnl_frac,
            "abs_pnl_frac": abs(bar_pnl_frac),
            "traded": traded_flag,
            "pos_delta": pos_delta,
            "reward": float(reward),
            "atr": float(getattr(base_env, "current_atr", 0.0)) if base_env else 0.0,
        })

        if step % 1000 == 0:
            print(f"[audit] step={step:6d}  pos={cur_position:+.3f}  equity={equity:,.0f}  "
                  f"price_ret={price_return*1e4:+6.2f}bps  cum_trades={cumulative_traded}")
        step += 1

    df = pd.DataFrame(rows)
    out_csv = REPO / f"audit_seed2025_{tag}_trace.csv"
    df.to_csv(out_csv, index=False)
    print(f"[audit] wrote {len(df)} bars to {out_csv}")

    # Summary stats
    summary = {
        "n_bars": int(len(df)),
        "starting_equity": float(df["equity"].iloc[0]) if len(df) else None,
        "final_equity": float(df["equity"].iloc[-1]) if len(df) else None,
        "total_return_pct": float((df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100) if len(df) > 1 else None,
        "position_abs_max": float(df["position"].abs().max()),
        "position_abs_p99": float(df["position"].abs().quantile(0.99)),
        "position_abs_mean": float(df["position"].abs().mean()),
        "fraction_bars_in_market": float((df["position"].abs() > 1e-6).mean()),
        "fraction_bars_at_atr_cap": float((df["position"].abs() > (cfg["env"]["atr_cap_max_position"] - 0.01)).mean()),
        "trade_count": int(cumulative_traded),
        "cumulative_position_change": float(cumulative_pos_change),
        "price_return_per_bar_mean_bps": float(df["price_return"].mean() * 1e4),
        "price_return_per_bar_std_bps": float(df["price_return"].std() * 1e4),
        "bar_pnl_frac_mean_bps": float(df["bar_pnl_frac"].mean() * 1e4),
        "bar_pnl_frac_std_bps": float(df["bar_pnl_frac"].std() * 1e4),
        "bar_win_rate": float((df["bar_pnl_frac"] > 0).mean()),
        "bar_loss_rate": float((df["bar_pnl_frac"] < 0).mean()),
        "max_consecutive_winning_bars": int((df["bar_pnl_frac"] > 0).astype(int).groupby(
            (df["bar_pnl_frac"] <= 0).cumsum()).sum().max()),
        "max_consecutive_losing_bars": int((df["bar_pnl_frac"] < 0).astype(int).groupby(
            (df["bar_pnl_frac"] >= 0).cumsum()).sum().max()),
        # Equity ratchet: how often does equity hit a new high? a leak/lookahead would push this towards 1.0
        "fraction_bars_at_new_equity_peak": float((df["equity"] >= df["equity"].cummax()).mean()),
    }
    summary["window_start"] = start
    summary["window_end"] = end
    summary["norm_cutoff"] = norm_cut
    summary["tag"] = tag
    out_json = REPO / f"audit_seed2025_{tag}_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[audit] summary -> {out_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
