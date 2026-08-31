#!/usr/bin/env python
"""PPO-GAE (continuous PPO) vs SAC on gmgp1-btc — screen runner (S553-cont-54).

Trains a continuous PPO agent on the EXACT clean-canary train split, then
backtests on the 4 walk-forward OOS windows (2025-12 .. 2026-03) that SAC's
clean-canary verdict was computed on, plus the HPO val/test windows. Uses the
same V7 env, the same de-leaked + cost-corrected data, the same network encoder,
and PF/Sharpe/MaxDD computed identically to run_full_pipeline.run_backtest — so
the ONLY changed variable vs SAC is the RL algorithm.

SAC clean-canary baseline (results/gmgp1_btc_canary_costcorr_wf/verdict.json):
    best-solo median WF PF = 0.9774   ensemble = 0.8935   overall = FAIL
    worst fixed-lot trailing MDD = -33.1%

Pre-registered escalation gate (decided BEFORE seeing PPO numbers):
    median WF PF >= 1.10 AND test PF >= 1.0  -> escalate to full HPO + WF
    else                                     -> verdict: PPO does not rescue gmgp1-btc

Usage:
    python scripts/research/ppo_ge_gmgp1_btc_screen.py \
        --config configs/gmgp1_btc_ppoge_screen.yaml \
        --total_timesteps 1000000 --lr 3e-4 --ent_coef 0.0 \
        --max_leverage 2.0 --deadband 0.25 --seed 0 --tag a
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sharpen.agents.ppo_continuous.ppo_continuous_agent import PPOContinuousAgent  # noqa: E402
from sharpen.hpo.env_factory import create_vector_env, make_env  # noqa: E402
from sharpen.training.ppo_continuous_trainer import PPOContinuousTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ppo_ge_screen")

SAC_BASELINE = {
    "wf_best_solo_median_pf": 0.9774,
    "wf_ensemble_median_pf": 0.8935,
    "worst_fixed_lot_trailing_mdd_pct": -33.09,
    "overall": "FAIL",
}

# WF OOS windows that SAC's clean-canary verdict was computed on.
WF_WINDOWS = [
    ("wf_2025_12", "2025-12-01", "2026-01-01"),
    ("wf_2026_01", "2026-01-01", "2026-02-01"),
    ("wf_2026_02", "2026-02-01", "2026-03-01"),
    ("wf_2026_03", "2026-03-01", "2026-04-01"),
]


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backtest_window(config, checkpoint_path, device, start_date, end_date, prefix):
    """Deterministic backtest on [start_date, end_date].

    Mirrors run_full_pipeline.run_backtest: backtest-mode env overrides, the SAC
    obs path (stack scales + private), and identical metric computation.
    """
    bt_cfg = copy.deepcopy(config)
    bt_cfg.setdefault("env", {})
    bt_cfg["env"]["private_state_augment_prob"] = 0.0
    bt_cfg["env"].setdefault("reward", {})
    bt_cfg["env"]["reward"]["hindsight_weight"] = 0.0
    bt_cfg["env"]["episode_length"] = 0       # full window, sequential
    bt_cfg["env"]["random_start"] = False
    # Final fee from fee_schedule if present (canary uses fixed taker_fee — no-op).
    fee_schedule = bt_cfg.get("env", {}).get("fee_schedule")
    if fee_schedule:
        bt_cfg["env"]["taker_fee"] = fee_schedule[-1].get("ramp_to", fee_schedule[-1].get("taker_fee", 0.0))

    # LEAK-1: normalize causally within the window (reset at window start; no
    # cross-window normalization).
    env = make_env(bt_cfg, start_date=start_date, end_date=end_date, norm_cutoff_date=start_date)

    net_cfg = dict(config.get("network", {}))
    scales = config.get("features", {}).get("scales", config.get("env", {}).get("scales", [15, 60, 240]))
    net_cfg["n_scales"] = len(scales)
    ppo_cfg = config.get("agents", {}).get("ppoc", {})
    agent = PPOContinuousAgent(
        network_config=net_cfg,
        lr=ppo_cfg.get("learning_rate", 3e-4),
        gamma=ppo_cfg.get("gamma", 0.99),
        use_amp=False,
        device=device,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        logger.warning("No checkpoint — random policy backtest")

    obs, _info = env.reset()
    n_scales = sum(1 for i in range(100) if f"scale_{i}" in obs)
    portfolio_values, positions = [], []
    _trade_pnls, _prev_rpnl = [], 0.0
    done, step = False, 0
    while not done and step < 200000:
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        action = agent.predict(scale_stack, priv, deterministic=True)[0].cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info.get("portfolio_value", 100000))
        positions.append(info.get("position", info.get("inventory", 0)))
        switched = info.get("switched")
        if switched:
            rpnl = info.get("realized_pnl")
            if rpnl is not None:
                _trade_pnls.append(float(rpnl) - _prev_rpnl)
                _prev_rpnl = float(rpnl)
        step += 1
    env.close()

    pv = np.array(portfolio_values, dtype=np.float64)
    pos_arr = np.array(positions, dtype=np.float64)
    if len(pv) < 3:
        return {"prefix": prefix, "n_bars": int(len(pv)), "error": "too_few_bars"}
    returns = np.diff(pv) / pv[:-1]
    total_return = (pv[-1] - pv[0]) / pv[0]

    bar_minutes = scales[0] if scales else 15
    bars_per_year = 525600 / bar_minutes
    sharpe = 0.0
    if np.std(returns) > 1e-9:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)
    peak = np.maximum.accumulate(pv)
    max_dd = float(np.min(pv / np.maximum(peak, 1e-12)) - 1)

    # Bar-level PF (matches run_backtest BUG-14 cap convention).
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gp = float(np.sum(wins)) if len(wins) else 0.0
    gl = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = gp / gl if gl > 1e-12 else (10.0 if gp > 1e-12 else 0.0)

    tp = np.array(_trade_pnls)
    if len(tp) > 0:
        tpp = float(np.sum(tp[tp > 0]))
        tpl = float(abs(np.sum(tp[tp < 0])))
        pf_trade = tpp / tpl if tpl > 1e-12 else (10.0 if tpp > 1e-12 else 0.0)
    else:
        pf_trade = profit_factor

    base_count = int(np.sum(np.abs(np.diff(pos_arr)) > 1e-6))
    market_exposure = float(np.mean(np.abs(pos_arr) > 1e-6))

    return {
        "prefix": prefix,
        "n_bars": int(len(pv)),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "profit_factor": float(profit_factor),
        "profit_factor_trade": float(pf_trade),
        "trade_count": base_count,
        "market_exposure": market_exposure,
        "final_value": float(pv[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--total_timesteps", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ent_coef", type=float, default=None)
    ap.add_argument("--max_leverage", type=float, default=None)
    ap.add_argument("--deadband", type=float, default=None)
    ap.add_argument("--n_epochs", type=int, default=None)
    ap.add_argument("--rollout_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="a")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="results/ppo_ge_gmgp1_btc")
    ap.add_argument("--skip_train", action="store_true", help="Backtest only (needs existing checkpoint)")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--num_envs", type=int, default=None, help="Override training.num_envs")
    ap.add_argument("--use_sync", action="store_true", help="Use SyncVectorEnv (smoke tests / constrained boxes)")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("agents", {}).setdefault("ppoc", {})
    if args.lr is not None:
        config["agents"]["ppoc"]["learning_rate"] = args.lr
    if args.ent_coef is not None:
        config["agents"]["ppoc"]["ent_coef"] = args.ent_coef
    if args.n_epochs is not None:
        config["agents"]["ppoc"]["n_epochs"] = args.n_epochs
    if args.rollout_steps is not None:
        config["agents"]["ppoc"]["rollout_steps"] = args.rollout_steps
    config.setdefault("env", {})
    if args.max_leverage is not None:
        config["env"]["max_leverage"] = args.max_leverage
    if args.deadband is not None:
        config["env"]["deadband_threshold"] = args.deadband
    config.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    if args.num_envs is not None:
        config["training"]["num_envs"] = args.num_envs
    if args.use_sync:
        config["training"]["use_sync"] = True

    run_name = f"ppoge_gmgp1_btc_{args.tag}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"=== PPO-GAE screen | run={run_name} | device={args.device} ===")
    logger.info(f"HPs: lr={config['agents']['ppoc'].get('learning_rate')} "
                f"ent_coef={config['agents']['ppoc'].get('ent_coef')} "
                f"n_epochs={config['agents']['ppoc'].get('n_epochs')} "
                f"max_lev={config['env'].get('max_leverage')} "
                f"deadband={config['env'].get('deadband_threshold')} "
                f"steps={args.total_timesteps}")

    checkpoint = args.checkpoint
    if not args.skip_train:
        num_envs = config.get("training", {}).get("num_envs", 20)
        data = config["data"]
        env = create_vector_env(
            config, num_envs,
            start_date=data["train_start_date"], end_date=data["train_end_date"],
            use_sync=config.get("training", {}).get("use_sync", False),
        )
        try:
            trainer = PPOContinuousTrainer(env, config, device=args.device, run_name=run_name)
            t0 = time.time()
            checkpoint = trainer.train()
            logger.info(f"Training done in {(time.time()-t0)/60:.1f} min -> {checkpoint}")
        finally:
            try:
                env.close()
            except Exception:
                pass

    # ---- Backtest all OOS windows ----
    data = config["data"]
    windows = [
        ("val", data["val_start_date"], data["val_end_date"]),
        ("test", data["test_start_date"], data["test_end_date"]),
    ] + WF_WINDOWS

    results = {}
    for prefix, sd, ed in windows:
        logger.info(f"Backtest {prefix} [{sd} -> {ed}]")
        try:
            results[prefix] = backtest_window(config, checkpoint, args.device, sd, ed, prefix)
        except Exception as e:
            logger.error(f"Backtest {prefix} failed: {e}")
            results[prefix] = {"prefix": prefix, "error": str(e)}
        logger.info(f"  {prefix}: {results[prefix]}")

    wf_pfs = [results[w[0]].get("profit_factor") for w in WF_WINDOWS
              if isinstance(results.get(w[0]), dict) and results[w[0]].get("profit_factor") is not None]
    wf_median_pf = float(np.median(wf_pfs)) if wf_pfs else None
    test_pf = results.get("test", {}).get("profit_factor")

    escalate = (
        wf_median_pf is not None and test_pf is not None
        and wf_median_pf >= 1.10 and test_pf >= 1.0
    )

    summary = {
        "run_name": run_name,
        "hparams": {
            "lr": config["agents"]["ppoc"].get("learning_rate"),
            "ent_coef": config["agents"]["ppoc"].get("ent_coef"),
            "n_epochs": config["agents"]["ppoc"].get("n_epochs"),
            "rollout_steps": config["agents"]["ppoc"].get("rollout_steps"),
            "max_leverage": config["env"].get("max_leverage"),
            "deadband": config["env"].get("deadband_threshold"),
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
        },
        "checkpoint": checkpoint,
        "windows": results,
        "ppo_wf_median_pf": wf_median_pf,
        "ppo_test_pf": test_pf,
        "sac_baseline": SAC_BASELINE,
        "escalation_gate": "wf_median_pf>=1.10 AND test_pf>=1.0",
        "escalate": bool(escalate),
        "beats_sac_wf": (wf_median_pf is not None and wf_median_pf > SAC_BASELINE["wf_best_solo_median_pf"]),
    }
    out_path = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"PPO WF median PF = {wf_median_pf}  (SAC best-solo = {SAC_BASELINE['wf_best_solo_median_pf']})")
    logger.info(f"PPO test PF      = {test_pf}")
    logger.info(f"Escalate to full HPO+WF? {escalate}")
    logger.info(f"Wrote {out_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
