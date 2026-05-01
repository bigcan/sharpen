#!/usr/bin/env python3
"""SG-1 Arm A/B gate evaluator (Q1 2026 OANDA OOS).

Loads a checkpoint, runs the deterministic policy on the configured test
window, captures per-bar info (portfolio_value, daily_loss, eod_drawdown,
trade_count), and prints the three Arm B gate metrics:

  1. Test PF (bar-level + trade-level)
  2. Worst intraday daily drawdown (max of intra-day loss across sessions)
  3. FTMO compliance: active_days >= 4 AND max_single_day_share <= 0.50

Writes trajectory to results/gate_eval/<label>_trajectory.parquet for audit.

No WandB dependency (pure local eval).
"""
from __future__ import annotations
import argparse, os, sys, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.hpo.env_factory import make_env  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gate-eval")

MIN_ACTIVE_DAYS = 4
MAX_SINGLE_DAY_SHARE = 0.50


def _build_sac_agent(config, device):
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    network_config = dict(config.get("network", {}))
    if not network_config:
        raise ValueError("config missing 'network' section")
    scales = config.get("features", {}).get("scales", config.get("env", {}).get("scales", [3, 15, 60]))
    network_config["n_scales"] = len(scales)
    # Continuous Box(-1,1,(1,)) — encode as size_dims=1 style not needed; SAC reads private_dim + scale_encoder
    sac_cfg = config.get("agents", {}).get("sac", {})
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


# _prep_backtest_config consolidated to finrl_pro_ds.config_utils in S495-cont
# (prop-firm decoupling rev 2). Re-exported here under the original name so
# downstream scripts that import it from this module continue to work.
from finrl_pro_ds.config_utils import _prep_backtest_config  # noqa: E402,F401


def run_gate_backtest(
    config_path: str,
    checkpoint_path: str,
    label: str,
    device: str = "cuda",
    *,
    disable_profit_target: bool = True,
):
    """Run a single-checkpoint gate backtest.

    ``disable_profit_target`` — default True preserves the S466 FTMO gate
    behaviour (full-window evaluation needs the profit-target early-term
    disabled). The prop-firm A/B harness (ADR-6) sets this False for the
    control arm so the V7 adapter can terminate at +10% equity, which is
    exactly the behaviour the A/B is designed to measure.
    """
    out_dir = Path("results/gate_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = _prep_backtest_config(config, disable_profit_target=disable_profit_target)

    data_cfg = config["data"]
    start, end = data_cfg["test_start_date"], data_cfg["test_end_date"]
    log.info(f"[{label}] Test window: {start} -> {end}")
    log.info(f"[{label}] Checkpoint: {checkpoint_path}")

    env = make_env(config, start_date=start, end_date=end, norm_cutoff_date=data_cfg.get("val_end_date"))
    agent = _build_sac_agent(config, device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        log.info(f"[{label}] Checkpoint loaded")
    else:
        raise FileNotFoundError(checkpoint_path)

    # Expose base-scale timestamps for per-bar tagging
    base_ts = None
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "timestamps"):
        base_env = base_env.env
    if hasattr(base_env, "timestamps"):
        base_ts = base_env.timestamps
    if base_ts is None and hasattr(base_env, "handler") and hasattr(base_env.handler, "_base_timestamps"):
        base_ts = base_env.handler._base_timestamps
    log.info(f"[{label}] base_timestamps: {'present' if base_ts is not None else 'MISSING'} "
             f"({len(base_ts) if base_ts is not None else 0} bars)")

    obs, info = env.reset()
    rows = []
    step = 0
    done = False
    while not done and step < 500000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        pred = agent.predict(scale_stack, priv, deterministic=True)
        action = pred[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        def _scalar(k, default=0.0):
            v = info.get(k, default)
            if isinstance(v, np.ndarray):
                return float(v.flatten()[0]) if v.size else default
            return float(v) if v is not None else default

        # timestamp: pull from env's base-scale timestamps using current_step
        cur = getattr(base_env, "current_step", None)
        ts = None
        if base_ts is not None and cur is not None and 0 <= cur - 1 < len(base_ts):
            ts = base_ts[cur - 1]
        rows.append({
            "step": step,
            "timestamp": ts,
            "portfolio_value": _scalar("portfolio_value", 100000.0),
            "position": _scalar("position"),
            "traded": _scalar("traded"),
            "trade_count": _scalar("trade_count"),
            "realized_pnl": _scalar("realized_pnl"),
            "eod_drawdown": _scalar("eod_drawdown"),
            "drawdown_pct": _scalar("drawdown_pct"),
            "prop_firm_termination": info.get("prop_firm_termination"),
            "reward": float(reward),
        })
        if step % 20000 == 0:
            log.info(f"[{label}] step={step} pv={rows[-1]['portfolio_value']:.2f}")
        step += 1

    env.close()
    df = pd.DataFrame(rows)
    traj_path = out_dir / f"{label}_trajectory.parquet"
    df.to_parquet(traj_path)
    log.info(f"[{label}] trajectory saved: {traj_path} ({len(df)} bars)")
    return df


def compute_gate_metrics(df: pd.DataFrame, label: str) -> dict:
    """Compute PF / worst intraday daily DD / FTMO compliance from trajectory."""
    pv = df["portfolio_value"].to_numpy()
    returns = np.diff(pv) / pv[:-1]
    wins = returns[returns > 0]; losses = returns[returns < 0]
    gp = float(np.sum(wins)) if len(wins) else 0.0
    gl = float(abs(np.sum(losses))) if len(losses) else 0.0
    pf_bar = gp / gl if gl > 1e-12 else (10.0 if gp > 1e-12 else 0.0)

    # Trailing peak-to-trough (for reference)
    peak = np.maximum.accumulate(pv)
    trailing_mdd = float(np.min(pv / np.maximum(peak, 1e-12)) - 1) if len(pv) else 0.0

    # Daily grouping: need timestamps. If missing, fall back to bars-per-day inferred from DataFrame length vs test-window days.
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    has_ts = ts.notna().any()

    worst_intraday_dd = None
    active_days = 0
    max_single_day_share = None
    daily_profit = {}

    if has_ts:
        df2 = df.copy()
        df2["_utc_date"] = ts.dt.date
        # realized_pnl is cumulative — daily profit = end-of-day minus start-of-day
        # portfolio_value is session-level equity; use it as ground truth.
        df2["pv"] = df2["portfolio_value"]
        # Per-day: max intraday loss against that day's open equity
        day_open = df2.groupby("_utc_date")["pv"].first()
        day_close = df2.groupby("_utc_date")["pv"].last()
        day_min = df2.groupby("_utc_date")["pv"].min()
        # intra-day DD = 1 - min/open  (as fraction)
        intraday_dd_pct = 1.0 - (day_min / day_open)
        worst_intraday_dd = float(intraday_dd_pct.max())

        # Active days: days where position was non-flat at any point
        active_mask = df2["position"].abs() > 1e-9
        active_days = int(df2.loc[active_mask, "_utc_date"].nunique())

        # Daily profit (close-open)
        daily_profit_series = (day_close - day_open) / day_open
        daily_profit = {str(k): float(v) for k, v in daily_profit_series.items()}

        total_profit = sum(v for v in daily_profit.values() if v > 0)
        if total_profit > 0:
            max_pos_day = max((v for v in daily_profit.values() if v > 0), default=0.0)
            max_single_day_share = float(max_pos_day / total_profit)

    ftmo_ok = (active_days >= MIN_ACTIVE_DAYS and
               (max_single_day_share is None or max_single_day_share <= MAX_SINGLE_DAY_SHARE))

    result = {
        "label": label,
        "n_bars": int(len(df)),
        "pf_bar": pf_bar,
        "total_return_pct": float((pv[-1] - pv[0]) / pv[0] * 100) if len(pv) else 0.0,
        "trailing_max_drawdown_pct": trailing_mdd * 100,
        "worst_intraday_daily_dd_pct": None if worst_intraday_dd is None else worst_intraday_dd * 100,
        "active_days": active_days,
        "max_single_day_share": max_single_day_share,
        "ftmo_compliance_pass": bool(ftmo_ok),
        "trade_count": int(df["trade_count"].iloc[-1]) if len(df) else 0,
        "prop_firm_termination": (df["prop_firm_termination"].dropna().iloc[-1]
                                  if df["prop_firm_termination"].notna().any() else None),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = run_gate_backtest(args.config, args.checkpoint, args.label, device=args.device)
    metrics = compute_gate_metrics(df, args.label)
    out_path = Path("results/gate_eval") / f"{args.label}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))
    log.info(f"metrics written: {out_path}")


if __name__ == "__main__":
    main()
