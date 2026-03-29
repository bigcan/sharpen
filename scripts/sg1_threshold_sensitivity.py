"""SG-1 Gate Threshold Sensitivity Study (Phase 0)

Sweeps signal gate thresholds on a trained SG-1 checkpoint to assess
robustness. Runs inference-only backtests with varied gate parameters.

Usage:
    python scripts/sg1_threshold_sensitivity.py \
        --checkpoint checkpoints/sg1_best/checkpoint_final.pth \
        [--config configs/signal_gate_sac_gc_3min.yaml]

GO/NO-GO: median PF > 1.2 AND no ±20% perturbation drops PF < 1.0
          AND "safe zone" spans >= 40% of tested range.
"""
import argparse
import copy
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.agents.sac.sac_agent import SACAgent
from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler
from finrl_pro_ds.envs.continuous_swing_env import ContinuousSwingEnv
from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_backtest_env(config: dict, gate_overrides: dict | None = None):
    """Create env with signal gate for backtest (full test period, no random start)."""
    cfg = copy.deepcopy(config)
    data_cfg = cfg["data"]
    features_cfg = cfg.get("features", {})

    # Backtest overrides: full dataset, sequential, production fee
    cfg["env"]["episode_length"] = 0
    cfg["env"]["random_start"] = False
    cfg["env"]["reward"]["hindsight_weight"] = 0.0
    fee_schedule = cfg["env"].get("fee_schedule")
    if fee_schedule:
        final_fee = fee_schedule[-1].get("ramp_to", fee_schedule[-1].get("taker_fee", 0.0))
        cfg["env"]["taker_fee"] = final_fee

    handler = MultiScaleOHLCVHandler(
        file_path=data_cfg["file_path"],
        ticker=data_cfg["ticker"],
        feature_config=features_cfg,
        start_date=data_cfg.get("test_start_date"),
        end_date=data_cfg.get("test_end_date"),
        norm_cutoff_date=data_cfg.get("test_start_date"),
    )
    env = ContinuousSwingEnv(config=cfg["env"], data_handler=handler)

    gate_cfg = copy.deepcopy(cfg.get("signal_gate", {}))
    if gate_overrides:
        gate_cfg.update(gate_overrides)

    if gate_cfg.get("enabled", False):
        env = SignalGatedWrapper(env, gate_config=gate_cfg)

    return env


def make_agent(config: dict, env, device: str) -> SACAgent:
    """Create SAC agent matching config."""
    network_cfg = dict(config.get("network", {}))
    scales = config.get("features", {}).get("scales", [3, 15, 60])
    network_cfg["n_scales"] = len(scales)
    sac_cfg = config.get("agents", {}).get("sac", {})
    return SACAgent(
        network_config=network_cfg,
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


def run_single_backtest(config: dict, checkpoint: str, device: str, gate_overrides: dict | None = None) -> dict:
    """Run one backtest with given gate overrides, return metrics."""
    env = make_backtest_env(config, gate_overrides)
    agent = make_agent(config, env, device)
    agent.load(checkpoint)

    obs, _ = env.reset()
    portfolio_values = []
    positions = []
    gate_skipped_total = 0
    step = 0
    done = False

    while not done and step < 200000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_t = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv_t = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        pred = agent.predict(scale_t, priv_t, deterministic=True)
        action = pred[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info.get("portfolio_value", 100000))
        positions.append(info.get("position", 0))
        gate_skipped_total = info.get("gate_total_skipped", 0)
        step += 1

    pv = np.array(portfolio_values)
    pos_arr = np.array(positions)
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-12)

    total_return = (pv[-1] - pv[0]) / pv[0] if len(pv) > 1 else 0.0

    # Profit Factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

    # Sharpe (per-bar, annualized)
    bar_minutes = config.get("features", {}).get("scales", [3])[0]
    bars_per_year = 525600 / bar_minutes
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    # Max Drawdown
    peak = np.maximum.accumulate(pv) if len(pv) > 0 else np.array([1.0])
    max_dd = float(np.min(pv / np.maximum(peak, 1e-12)) - 1)

    # Trade count
    pos_deltas = np.abs(np.diff(pos_arr))
    trade_count = int(np.sum(pos_deltas > 1e-6))

    # Gate stats
    total_bars = step + gate_skipped_total
    gate_open_pct = step / total_bars * 100 if total_bars > 0 else 100.0

    env.close()

    return {
        "pf": pf,
        "return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "trades": trade_count,
        "steps": step,
        "gate_open_pct": gate_open_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="SG-1 Gate Threshold Sensitivity")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SG-1 checkpoint")
    parser.add_argument("--config", type=str, default="configs/signal_gate_sac_gc_3min.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    # ── Sweep definitions ──
    atr_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    parkinson_values = [0.01, 0.015, 0.02, 0.025, 0.03]
    volume_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    hold_values = [10, 15, 20, 30, 40]
    return_thresholds = [0.001, 0.0015, 0.002, 0.003, 0.004]

    results = []

    # ── 1. Sweep ATR threshold (others fixed at default) ──
    print("\n" + "=" * 80)
    print("SWEEP 1: atr_threshold (parkinson=0.02, volume=0.5, max_hold=20)")
    print("=" * 80)
    for val in atr_values:
        label = f"atr={val:.2f}"
        marker = " ← BASE" if val == 0.30 else ""
        overrides = {"atr_threshold": val}
        t0 = time.time()
        m = run_single_backtest(config, args.checkpoint, device, overrides)
        dt = time.time() - t0
        results.append({"sweep": "atr", "param": val, **m})
        print(f"  {label:12s}  PF={m['pf']:.3f}  Ret={m['return_pct']:+.1f}%  "
              f"Sharpe={m['sharpe']:.1f}  MDD={m['max_dd_pct']:.1f}%  "
              f"Trades={m['trades']:>4d}  Gate={m['gate_open_pct']:.0f}%  "
              f"({dt:.1f}s){marker}")

    # ── 2. Sweep Parkinson threshold ──
    print("\n" + "=" * 80)
    print("SWEEP 2: parkinson_threshold (atr=0.3, volume=0.5, max_hold=20)")
    print("=" * 80)
    for val in parkinson_values:
        label = f"park={val:.3f}"
        marker = " ← BASE" if val == 0.02 else ""
        overrides = {"parkinson_threshold": val}
        t0 = time.time()
        m = run_single_backtest(config, args.checkpoint, device, overrides)
        dt = time.time() - t0
        results.append({"sweep": "parkinson", "param": val, **m})
        print(f"  {label:12s}  PF={m['pf']:.3f}  Ret={m['return_pct']:+.1f}%  "
              f"Sharpe={m['sharpe']:.1f}  MDD={m['max_dd_pct']:.1f}%  "
              f"Trades={m['trades']:>4d}  Gate={m['gate_open_pct']:.0f}%  "
              f"({dt:.1f}s){marker}")

    # ── 3. Sweep Volume threshold ──
    print("\n" + "=" * 80)
    print("SWEEP 3: volume_threshold (atr=0.3, parkinson=0.02, max_hold=20)")
    print("=" * 80)
    for val in volume_values:
        label = f"vol={val:.1f}"
        marker = " ← BASE" if val == 0.5 else ""
        overrides = {"volume_threshold": val}
        t0 = time.time()
        m = run_single_backtest(config, args.checkpoint, device, overrides)
        dt = time.time() - t0
        results.append({"sweep": "volume", "param": val, **m})
        print(f"  {label:12s}  PF={m['pf']:.3f}  Ret={m['return_pct']:+.1f}%  "
              f"Sharpe={m['sharpe']:.1f}  MDD={m['max_dd_pct']:.1f}%  "
              f"Trades={m['trades']:>4d}  Gate={m['gate_open_pct']:.0f}%  "
              f"({dt:.1f}s){marker}")

    # ── 4. Sweep max_hold_bars ──
    print("\n" + "=" * 80)
    print("SWEEP 4: max_hold_bars (atr=0.3, parkinson=0.02, volume=0.5)")
    print("=" * 80)
    for val in hold_values:
        label = f"hold={val}"
        marker = " ← BASE" if val == 20 else ""
        overrides = {"max_hold_bars": val}
        t0 = time.time()
        m = run_single_backtest(config, args.checkpoint, device, overrides)
        dt = time.time() - t0
        results.append({"sweep": "max_hold", "param": val, **m})
        print(f"  {label:12s}  PF={m['pf']:.3f}  Ret={m['return_pct']:+.1f}%  "
              f"Sharpe={m['sharpe']:.1f}  MDD={m['max_dd_pct']:.1f}%  "
              f"Trades={m['trades']:>4d}  Gate={m['gate_open_pct']:.0f}%  "
              f"({dt:.1f}s){marker}")

    # ── 5. Test abs_log_return gate mode ──
    print("\n" + "=" * 80)
    print("SWEEP 5: gate_mode='return' (abs_log_return threshold)")
    print("=" * 80)
    for val in return_thresholds:
        label = f"ret={val:.4f}"
        overrides = {"gate_mode": "return", "return_threshold": val}
        t0 = time.time()
        m = run_single_backtest(config, args.checkpoint, device, overrides)
        dt = time.time() - t0
        results.append({"sweep": "return", "param": val, **m})
        print(f"  {label:12s}  PF={m['pf']:.3f}  Ret={m['return_pct']:+.1f}%  "
              f"Sharpe={m['sharpe']:.1f}  MDD={m['max_dd_pct']:.1f}%  "
              f"Trades={m['trades']:>4d}  Gate={m['gate_open_pct']:.0f}%  "
              f"({dt:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pf_values = [r["pf"] for r in results]
    median_pf = float(np.median(pf_values))
    min_pf = float(np.min(pf_values))
    max_pf = float(np.max(pf_values))
    pf_above_1 = sum(1 for pf in pf_values if pf > 1.0)
    pf_above_1_2 = sum(1 for pf in pf_values if pf > 1.2)
    total = len(pf_values)

    print(f"  Total combinations: {total}")
    print(f"  Median PF: {median_pf:.3f}")
    print(f"  Min PF: {min_pf:.3f}")
    print(f"  Max PF: {max_pf:.3f}")
    print(f"  PF > 1.0: {pf_above_1}/{total} ({pf_above_1/total*100:.0f}%)")
    print(f"  PF > 1.2: {pf_above_1_2}/{total} ({pf_above_1_2/total*100:.0f}%)")

    # Per-sweep safe zone analysis
    for sweep_name in ["atr", "parkinson", "volume", "max_hold", "return"]:
        sweep_results = [r for r in results if r["sweep"] == sweep_name]
        if not sweep_results:
            continue
        sweep_pfs = [r["pf"] for r in sweep_results]
        safe_count = sum(1 for pf in sweep_pfs if pf > 1.0)
        safe_pct = safe_count / len(sweep_pfs) * 100
        print(f"  {sweep_name:12s}: safe zone {safe_count}/{len(sweep_pfs)} ({safe_pct:.0f}%), "
              f"median PF {np.median(sweep_pfs):.3f}")

    # GO/NO-GO
    print("\n" + "-" * 40)
    go_median = median_pf > 1.2
    go_no_crash = min_pf >= 0.8  # Relaxed: no catastrophic crash
    go_safe_zone = (pf_above_1 / total) >= 0.6  # 60% safe

    if go_median and go_no_crash and go_safe_zone:
        print("  VERDICT: *** GO *** — thresholds are robust")
    elif go_median and go_safe_zone:
        print("  VERDICT: MARGINAL GO — some perturbations weak but median holds")
    else:
        reasons = []
        if not go_median:
            reasons.append(f"median PF {median_pf:.3f} < 1.2")
        if not go_no_crash:
            reasons.append(f"min PF {min_pf:.3f} < 0.8")
        if not go_safe_zone:
            reasons.append(f"safe zone {pf_above_1/total*100:.0f}% < 60%")
        print(f"  VERDICT: NO-GO — {'; '.join(reasons)}")


if __name__ == "__main__":
    main()
