"""Phase 0B: Generate 5-seed baseline backtest traces for GMGP2-XAUUSD.

Runs deterministic backtest with the existing best checkpoint across 5 seeds,
capturing per-bar traces: (timestamp, close_price, position, portfolio_value, step_return).

These traces serve as the frozen A/B baseline for all PRISM research phases.

Usage:
    python scripts/prism_research/baseline_trace.py
    python scripts/prism_research/baseline_trace.py --checkpoint path/to/checkpoint.pth
    python scripts/prism_research/baseline_trace.py --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.agents.sac.sac_agent import SACAgent
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.baseline")

# ---------- defaults ----------
CONFIG_PATH = PROJECT_ROOT / "configs" / "gmgp2_xauusd_sac_15min.yaml"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "gmgp2-xauusd-l1-multiseed_20260403_083331"
    / "checkpoint_final.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research" / "baseline"
SEEDS = [42, 123, 456, 789, 2025]
BAR_MINUTES = 15


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_backtest_env(config: dict, start_date: str, end_date: str, norm_cutoff_date: str | None = None):
    """Create a single backtest env with proper settings."""
    from scripts.run_full_pipeline import make_env

    bt_config = copy.deepcopy(config)
    bt_config.setdefault("env", {})
    bt_config["env"]["private_state_augment_prob"] = 0.0
    bt_config["env"].setdefault("reward", {})
    bt_config["env"]["reward"]["hindsight_weight"] = 0.0
    bt_config["env"]["episode_length"] = 0      # Full dataset
    bt_config["env"]["random_start"] = False     # Sequential

    # Use final fee from fee_schedule (R2-AUD-03)
    fee_schedule = bt_config.get("env", {}).get("fee_schedule")
    if fee_schedule:
        final_tier = fee_schedule[-1]
        final_fee = final_tier.get("ramp_to", final_tier.get("taker_fee", 0.0))
        bt_config["env"]["taker_fee"] = final_fee

    return make_env(bt_config, start_date=start_date, end_date=end_date,
                    norm_cutoff_date=norm_cutoff_date)


def create_agent(config: dict, device: str) -> SACAgent:
    """Create SAC agent from config (mirrors run_full_pipeline logic)."""
    network_config = dict(config.get("network", {}))
    scales = config.get("features", {}).get("scales",
             config.get("env", {}).get("scales", [15, 60, 240]))
    network_config["n_scales"] = len(scales)
    network_config["action_space_dims"] = 1  # SAC continuous

    sac_cfg = config.get("agents", {}).get("sac", {})
    return SACAgent(
        network_config=network_config,
        lr_actor=sac_cfg.get("lr_actor", 3e-4),
        lr_critic=sac_cfg.get("lr_critic", 3e-4),
        lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
        gamma=sac_cfg.get("gamma", 0.99),
        tau=sac_cfg.get("tau", 0.005),
        batch_size=sac_cfg.get("batch_size", 256),
        buffer_size=100,  # Minimal for backtest
        initial_alpha=sac_cfg.get("initial_alpha", 0.2),
        device=device,
    )


def run_backtest_trace(
    config: dict,
    checkpoint_path: str,
    device: str,
    seed: int,
) -> pd.DataFrame:
    """Run a single deterministic backtest and return per-bar trace DataFrame."""
    data_cfg = config.get("data", {})
    start_date = data_cfg.get("test_start_date")
    end_date = data_cfg.get("test_end_date")
    norm_cutoff = data_cfg.get("train_end_date")

    logger.info(f"[Seed {seed}] Backtest: {start_date} → {end_date} "
                f"(norm cutoff: {norm_cutoff})")

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_backtest_env(config, start_date, end_date, norm_cutoff_date=norm_cutoff)

    agent = create_agent(config, device)
    agent.load(checkpoint_path)

    obs, info = env.reset()
    traces = []
    done = False
    step = 0
    n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)

    while not done and step < 200000:
        # SAC inference (summary_stats mode)
        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
        parts.append(obs["private"])
        flat_np = np.concatenate(parts)
        scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).unsqueeze(0).to(
            device, non_blocking=True,
        )
        pred = agent.predict(scale_stack, None, deterministic=True)
        action = pred[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Capture trace
        pv = info.get("portfolio_value", 100000.0)
        position = info.get("position", 0.0)
        # Get timestamp and close from the data handler
        handler = env.handler if hasattr(env, "handler") else env.unwrapped.handler
        ts = handler._base_timestamps[handler._ptr - 1]
        close = float(handler._base_close[handler._ptr - 1])

        traces.append({
            "step": step,
            "timestamp": pd.Timestamp(ts),
            "close": close,
            "position": position,
            "portfolio_value": pv,
        })

        if step % 5000 == 0:
            logger.info(f"[Seed {seed}] Step {step}: PV={pv:.2f}, pos={position:.4f}")
        step += 1

    env.close()

    df = pd.DataFrame(traces)
    # Compute per-bar returns
    pv = df["portfolio_value"].values
    returns = np.zeros(len(pv))
    returns[1:] = np.diff(pv) / pv[:-1]
    df["step_return"] = returns

    logger.info(f"[Seed {seed}] Done: {step} steps, "
                f"final PV={pv[-1]:.2f}, return={((pv[-1]/pv[0])-1)*100:.2f}%")
    return df


def compute_metrics(df: pd.DataFrame, bar_minutes: int = BAR_MINUTES) -> dict:
    """Compute standard metrics from a trace DataFrame."""
    returns = df["step_return"].values[1:]  # Skip first (always 0)
    pv = df["portfolio_value"].values
    pos = df["position"].values

    bars_per_year = 525600 / bar_minutes
    total_return = (pv[-1] - pv[0]) / pv[0]

    # Sharpe
    sharpe = 0.0
    if np.std(returns) > 1e-9:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    # MDD
    peak = np.maximum.accumulate(pv)
    max_dd = np.min(pv / np.maximum(peak, 1e-12)) - 1

    # Sortino
    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 0 and np.std(downside) > 1e-9:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(bars_per_year)

    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 1e-12 else float("inf")

    # Trade count (position changes past deadband)
    pos_deltas = np.abs(np.diff(pos))
    trade_count = int(np.sum(pos_deltas > 1e-6))

    # Market exposure
    exposure = float(np.mean(np.abs(pos) > 1e-6))

    # PyfolioAnalyzer for institutional metrics
    try:
        analyzer = PyfolioAnalyzer(pd.Series(returns), bar_minutes=bar_minutes)
        pyfolio = analyzer.get_audit_metrics()
    except Exception:
        pyfolio = {}

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "trade_count": trade_count,
        "market_exposure": exposure,
        "final_pv": float(pv[-1]),
        "n_steps": len(pv),
        **{f"pyfolio_{k}": v for k, v in pyfolio.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Research: Baseline trace generation")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Device: {args.device}")

    all_metrics = []

    for seed in args.seeds:
        logger.info(f"\n{'='*60}\nRunning seed {seed}\n{'='*60}")

        trace_df = run_backtest_trace(config, args.checkpoint, args.device, seed)

        # Save trace
        trace_path = output_dir / f"trace_seed_{seed}.csv"
        trace_df.to_csv(trace_path, index=False)
        logger.info(f"Saved trace: {trace_path}")

        # Compute metrics
        metrics = compute_metrics(trace_df)
        metrics["seed"] = seed
        all_metrics.append(metrics)

        logger.info(f"[Seed {seed}] Sharpe={metrics['sharpe']:.3f}, "
                    f"PF={metrics['profit_factor']:.3f}, "
                    f"MDD={metrics['max_drawdown']*100:.2f}%, "
                    f"Return={metrics['total_return']*100:.2f}%")

    # Aggregate stats
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    # Cross-seed statistics
    sharpes = metrics_df["sharpe"].values
    pfs = metrics_df["profit_factor"].values
    mdds = metrics_df["max_drawdown"].values

    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    cv_sharpe = std_sharpe / abs(mean_sharpe) if abs(mean_sharpe) > 1e-9 else float("inf")

    logger.info(f"\n{'='*60}")
    logger.info(f"BASELINE SUMMARY ({len(args.seeds)} seeds)")
    logger.info(f"{'='*60}")
    logger.info(f"Sharpe:  {mean_sharpe:.3f} +/- {std_sharpe:.3f} (CV={cv_sharpe*100:.1f}%)")
    logger.info(f"PF:      {np.mean(pfs):.3f} +/- {np.std(pfs):.3f}")
    logger.info(f"MDD:     {np.mean(mdds)*100:.2f}% +/- {np.std(mdds)*100:.2f}%")
    logger.info(f"Return:  {np.mean(metrics_df['total_return'])*100:.2f}% "
                f"+/- {np.std(metrics_df['total_return'])*100:.2f}%")

    # Gate check
    gate_pass = cv_sharpe < 0.30
    logger.info(f"\nGATE CHECK: CV(Sharpe) = {cv_sharpe*100:.1f}% "
                f"{'PASS' if gate_pass else 'FAIL'} (threshold: <30%)")

    if not gate_pass:
        logger.warning("Baseline is unstable. PRISM research should not proceed.")

    # Save summary
    summary = {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "cv_sharpe": cv_sharpe,
        "mean_pf": float(np.mean(pfs)),
        "mean_mdd": float(np.mean(mdds)),
        "mean_return": float(np.mean(metrics_df["total_return"])),
        "gate_pass": gate_pass,
        "n_seeds": len(args.seeds),
        "checkpoint": args.checkpoint,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "baseline_summary.csv", index=False)
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
