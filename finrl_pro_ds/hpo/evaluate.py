"""HPO evaluation functions — shared between serial and distributed HPO.

Verbatim extraction from scripts/run_full_pipeline.py (lines 249-660).
"""
import logging

import numpy as np
import torch
import wandb

logger = logging.getLogger("FinRL.HPO")


def evaluate_for_hpo(env, agent, max_steps=5000, bar_minutes=1):
    """Evaluate agent for HPO — returns (profit_factor, trade_count).

    V4.2: Changed from raw Sharpe to profit_factor to prevent specification gaming.
    Also tracks trade_count for the activity constraint (min 100 trades).

    IMPORTANT: This function handles both single envs and VectorEnvs.
    VectorEnv returns info as a dict of arrays, or for newer Gymnasium versions,
    as a tuple (info_dict, final_info_dict).
    """
    all_returns = []
    positions = []  # V4.2: Track positions for trade counting
    action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
    trade_pnls = []  # FIX GMO1-06: Track PnL per completed swing

    def extract_portfolio_value(info, env_idx=0, default=100000.0):
        """Extract portfolio_value handling both single env and VectorEnv info structures."""
        if info is None:
            return default

        # Handle VectorEnv info structure (Gymnasium >= 0.26)
        pv = info.get("portfolio_value")

        if pv is not None:
            if hasattr(pv, "__len__") and not isinstance(pv, str):
                return float(pv[env_idx])
            return float(pv)

        if "_all_info" in info:
            per_env_info = info["_all_info"]
            if per_env_info and len(per_env_info) > env_idx:
                env_info = per_env_info[env_idx]
                if env_info and "portfolio_value" in env_info:
                    return float(env_info["portfolio_value"])

        if "final_info" in info:
            final = info["final_info"]
            if final and len(final) > env_idx and final[env_idx]:
                if "portfolio_value" in final[env_idx]:
                    return float(final[env_idx]["portfolio_value"])

        return default

    info_logged = False

    # BUG-B: Save and restore LSTM hidden state around eval.
    _saved_hidden = agent._hidden_state if hasattr(agent, '_hidden_state') else None
    if hasattr(agent, "reset_hidden_state"):
        agent.reset_hidden_state()

    # FIX GMO1-02: Detect discrete_dims BEFORE the eval loop.
    if hasattr(env.action_space, 'shape') and env.action_space.shape and not hasattr(env.action_space, 'n'):
        discrete_dims = -1  # V7: continuous action space
    elif hasattr(env.action_space, 'n'):
        discrete_dims = env.action_space.n
    elif hasattr(env.action_space, 'nvec'):
        discrete_dims = int(env.action_space.nvec[0])
    else:
        discrete_dims = 6

    try:
        obs, info = env.reset()
        done = False
        step = 0

        prev_val = extract_portfolio_value(info, env_idx=0, default=100000.0)
        prev_realized_pnl = 0.0
        current_direction = None

        while not done and step < max_steps:
            # Dispatch obs keys based on agent type
            if "scale_0" in obs:
                # SAC / V7 multi-scale obs
                n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
                sample = obs["scale_0"]
                obs_mode = getattr(agent, '_obs_mode', 'window')

                if obs_mode == "summary_stats":
                    if sample.ndim == 2:
                        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
                        parts.append(obs["private"])
                        flat_np = np.concatenate(parts, axis=1)
                        scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).to(
                            agent.device, non_blocking=True,
                        )
                    else:
                        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
                        parts.append(obs["private"])
                        flat_np = np.concatenate(parts)
                        scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).unsqueeze(0).to(
                            agent.device, non_blocking=True,
                        )
                    priv = None
                elif sample.ndim == 3:
                    # VectorEnv: obs["scale_i"] is (B,W,F)
                    scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=1)
                    scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).to(
                        agent.device, non_blocking=True,
                    )
                    priv = torch.as_tensor(obs["private"], dtype=torch.float32).to(
                        agent.device, non_blocking=True,
                    )
                else:
                    # Single env: obs["scale_i"] is (W,F)
                    scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
                    scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(
                        agent.device, non_blocking=True,
                    )
                    priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(
                        agent.device, non_blocking=True,
                    )
                # Optional LOB microstructure features (Market Making V8)
                lob = None
                if "lob" in obs:
                    lob_np = obs["lob"]
                    if lob_np.ndim == 2:
                        lob_np = lob_np[np.newaxis]
                    lob = torch.as_tensor(lob_np, dtype=torch.float32).to(
                        agent.device, non_blocking=True,
                    )
                pred = agent.predict(scale_stack, priv, deterministic=True, lob=lob)
            else:
                micro = torch.tensor(obs["micro"], dtype=torch.float32).to(agent.device, non_blocking=True)
                private = torch.tensor(obs["private"], dtype=torch.float32).to(agent.device, non_blocking=True)
                macro = torch.tensor(obs["macro"], dtype=torch.float32).to(agent.device, non_blocking=True)
                ctx = {"current_direction": current_direction} if current_direction is not None else None
                _eval_eps = getattr(agent, '_eval_epsilon', 0.0)
                pred = agent.predict(micro, private, macro, deterministic=True, context=ctx, eval_epsilon=_eval_eps)

            if isinstance(pred, tuple):
                action = pred[0]
            else:
                action = pred

            # Track action distribution
            first_action = action[0] if hasattr(action, 'shape') and len(action.shape) > 1 else action
            if discrete_dims == -1:
                act_val = float(first_action[0]) if hasattr(first_action, "__len__") else float(first_action)
                if act_val > 0.1:
                    action_counts[0] = action_counts.get(0, 0) + 1
                elif act_val < -0.1:
                    action_counts[2] = action_counts.get(2, 0) + 1
                else:
                    action_counts[1] = action_counts.get(1, 0) + 1
            else:
                direction = int(first_action[0]) if hasattr(first_action, "__len__") else int(first_action)
                action_counts[direction] = action_counts.get(direction, 0) + 1

            obs, reward, term, trunc, info = env.step(action)

            if not info_logged:
                info_keys = list(info.keys()) if isinstance(info, dict) else str(type(info))
                sample_values = {}
                if isinstance(info, dict):
                    for k, v in list(info.items())[:5]:
                        if hasattr(v, "shape"):
                            sample_values[k] = f"array{v.shape}"
                        elif hasattr(v, "__len__") and len(v) > 0:
                            sample_values[k] = f"list[{len(v)}]:{type(v[0]).__name__}"
                        else:
                            sample_values[k] = str(type(v).__name__)
                wandb.log({"_debug/info_keys": str(info_keys), "_debug/info_sample": str(sample_values)})
                info_logged = True

            t_val = term[0] if hasattr(term, "__len__") else term
            tr_val = trunc[0] if hasattr(trunc, "__len__") else trunc
            done = bool(t_val or tr_val)

            curr_val = extract_portfolio_value(info, env_idx=0, default=prev_val)

            dir_val = info.get("direction")
            if dir_val is not None:
                current_direction = np.array([float(dir_val[0])]) if hasattr(dir_val, "__len__") else np.array([float(dir_val)])

            pos = info.get("position", info.get("inventory"))
            if pos is not None:
                if hasattr(pos, "__len__") and not isinstance(pos, str):
                    p0 = pos[0]  # VectorEnv: take env 0
                    if hasattr(p0, "__len__") and not isinstance(p0, str):
                        # Multi-asset: store full position array
                        positions.append(np.asarray(p0, dtype=np.float64))
                    else:
                        positions.append(float(p0))
                else:
                    positions.append(float(pos))

            # FIX GMO1-06: Trade-level PF via realized_pnl and switched flag
            switched = info.get("switched")
            if switched is not None:
                is_switch = bool(switched[0]) if hasattr(switched, "__len__") and not isinstance(switched, str) else bool(switched)
                if is_switch:
                    rpnl = info.get("realized_pnl")
                    if rpnl is not None:
                        curr_rpnl = float(rpnl[0]) if hasattr(rpnl, "__len__") and not isinstance(rpnl, str) else float(rpnl)
                        trade_pnl = curr_rpnl - prev_realized_pnl
                        trade_pnls.append(trade_pnl)
                        prev_realized_pnl = curr_rpnl

            step_return = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
            all_returns.append(step_return)
            prev_val = curr_val
            step += 1

    except Exception as e:
        logger.error("Evaluation error: %s", e)
        import traceback
        wandb.log({"_debug/eval_error": str(e), "_debug/eval_traceback": traceback.format_exc()})
        if hasattr(agent, '_hidden_state'):
            agent._hidden_state = _saved_hidden
        return 0.0, 0

    returns = np.array(all_returns)

    # Build position array — handles both single-asset (scalar) and multi-asset (vector)
    if positions:
        if hasattr(positions[0], "__len__"):
            pos_arr = np.stack(positions)  # (T, n_assets)
        else:
            pos_arr = np.array(positions)  # (T,)
    else:
        pos_arr = np.array([0.0])

    # V4.2: Count trades (position changes) — axis=0 works for both 1D and 2D
    pos_deltas = np.abs(np.diff(pos_arr, axis=0))
    base_count = int(np.sum(pos_deltas > 1e-6))
    sign_flips = int(np.sum((pos_arr[:-1] * pos_arr[1:]) < -1e-9))
    if discrete_dims == -1:
        trade_count = base_count
    elif discrete_dims == 2:
        trade_count = sign_flips
    else:
        trade_count = base_count + sign_flips

    # FIX GMO1-06: Compute profit_factor based on trade-level PnL
    trade_pnls_arr = np.array(trade_pnls)
    if len(trade_pnls_arr) > 0:
        positive_pnls = trade_pnls_arr[trade_pnls_arr > 0]
        negative_pnls = trade_pnls_arr[trade_pnls_arr < 0]
        gross_profit = float(np.sum(positive_pnls))
        gross_loss = float(np.abs(np.sum(negative_pnls)))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)
    else:
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        gross_profit = float(np.sum(positive_returns)) if len(positive_returns) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

    # Diagnostic logging
    if discrete_dims == -1:
        action_labels = {0: "long", 1: "flat", 2: "short"}
    elif discrete_dims == 2:
        action_labels = {0: "long", 1: "short"}
    elif discrete_dims == 3:
        action_labels = {0: "taker_buy", 1: "hold", 2: "taker_sell"}
    else:
        action_labels = {0: "taker_buy", 1: "maker_buy", 2: "hold", 3: "cancel", 4: "maker_sell", 5: "taker_sell"}
    diag = {
        "_debug/eval_steps": step,
        "_debug/eval_returns_len": len(returns),
        "_debug/eval_returns_std": float(np.std(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_returns_mean": float(np.mean(returns)) if len(returns) > 0 else 0.0,
        "_debug/eval_final_pv": prev_val,
        "_debug/eval_trade_count": trade_count,
        "_debug/eval_profit_factor": profit_factor,
    }
    for idx, label in action_labels.items():
        diag[f"_debug/eval_action_{idx}_{label}"] = action_counts.get(idx, 0)
    wandb.log(diag)

    # Sharpe for research tracking (not used for HPO scoring)
    if len(returns) > 1 and np.std(returns) > 1e-9:
        raw_ratio = np.mean(returns) / np.std(returns)
        bars_per_year = 525600 / bar_minutes
        sharpe_minute = raw_ratio * np.sqrt(bars_per_year)
        n_per_hour = max(1, int(60 / bar_minutes))
        hourly_returns = np.add.reduceat(returns, np.arange(0, len(returns), n_per_hour))
        if len(returns) % n_per_hour != 0 and len(hourly_returns) > 1:
            hourly_returns = hourly_returns[:-1]
        if len(hourly_returns) > 1 and np.std(hourly_returns) > 1e-9:
            sharpe_hourly = (np.mean(hourly_returns) / np.std(hourly_returns)) * np.sqrt(365 * 24)
        else:
            sharpe_hourly = 0.0
        wandb.log({
            "_research/sharpe_minute": sharpe_minute,
            "_research/sharpe_hourly": sharpe_hourly,
            "_research/raw_ratio_per_step": raw_ratio,
        })
    else:
        logger.warning(
            "Zero Sharpe: steps=%d, len=%d, std=%s, actions=%s",
            step, len(returns),
            np.std(returns) if len(returns) > 0 else "N/A",
            action_counts,
        )

    # BUG-B: Restore training hidden state after eval
    if hasattr(agent, '_hidden_state'):
        agent._hidden_state = _saved_hidden

    return profit_factor, trade_count


def analyze_hpo_correlation(trial_records, agent_type):
    """Post-HPO analysis: Spearman correlation between training reward and validation PF.

    Logs a WandB table of all trials, computes reward-PF rank correlation,
    HP importance rankings, and a scatter plot.
    """
    if not trial_records:
        logger.warning("[HPO-Corr] No trial records to analyze.")
        return

    columns = ["trial", "mean_reward", "val_pf", "trade_count", "status"]
    table = wandb.Table(columns=columns)
    for rec in trial_records:
        table.add_data(rec["trial"], rec["mean_reward"], rec["val_pf"],
                       rec["trade_count"], rec["status"])
    wandb.log({"hpo/trial_details": table})

    valid = [r for r in trial_records
             if r["status"] == "completed"
             and not np.isnan(r["mean_reward"])
             and r["val_pf"] > 0]

    wandb.run.summary["hpo/n_valid_trials"] = len(valid)
    wandb.run.summary["hpo/n_total_trials"] = len(trial_records)

    if len(valid) < 5:
        logger.info("[HPO-Corr] Only %d valid trials (need >= 5). Skipping correlation.", len(valid))
        wandb.run.summary["hpo/reward_pf_spearman_rho"] = float('nan')
        wandb.run.summary["hpo/reward_pf_spearman_p"] = float('nan')
        wandb.run.summary["hpo/reward_pf_alignment"] = "INSUFFICIENT_DATA"
        return

    mean_rewards = np.array([r["mean_reward"] for r in valid])
    val_pfs = np.array([r["val_pf"] for r in valid])

    from scipy.stats import spearmanr
    rho, p_value = spearmanr(mean_rewards, val_pfs)
    wandb.run.summary["hpo/reward_pf_spearman_rho"] = float(rho)
    wandb.run.summary["hpo/reward_pf_spearman_p"] = float(p_value)

    abs_rho = abs(rho)
    if abs_rho < 0.3:
        alignment = "WEAK"
    elif abs_rho < 0.6:
        alignment = "MODERATE"
    else:
        alignment = "STRONG"
    if rho < 0:
        alignment = f"NEGATIVE_{alignment}"
    wandb.run.summary["hpo/reward_pf_alignment"] = alignment
    logger.info("[HPO-Corr] Reward-PF Spearman rho=%.3f (p=%.4f) → %s", rho, p_value, alignment)

    # HP importance: per-HP correlation with val_pf
    if valid[0].get("hps"):
        hp_names = [k for k, v in valid[0]["hps"].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for hp_name in hp_names:
            hp_vals = np.array([r["hps"].get(hp_name, float('nan')) for r in valid])
            if np.all(np.isnan(hp_vals)) or np.std(hp_vals) < 1e-12:
                continue
            hp_rho, _ = spearmanr(hp_vals, val_pfs)
            if not np.isnan(hp_rho):
                wandb.run.summary[f"hpo/hp_importance/{hp_name}_rho"] = float(hp_rho)

    # Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(mean_rewards, val_pfs, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.set_xlabel("Mean Training Reward")
        ax.set_ylabel("Validation Profit Factor")
        ax.set_title(f"Reward vs PF (Spearman rho={rho:.3f}, p={p_value:.4f})")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="PF=1.0 (breakeven)")
        ax.legend()
        fig.tight_layout()
        wandb.log({"hpo/reward_vs_pf_scatter": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        logger.warning("[HPO-Corr] Scatter plot failed: %s", e)
