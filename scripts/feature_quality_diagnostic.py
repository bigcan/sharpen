#!/usr/bin/env python3
"""
Feature Quality Diagnostic (FQ-1): Jacobian + Leave-One-Out analysis of trained SAC agent.

Experiment: fq-1
Computes feature attribution via:
  1. Jacobian analysis: dQ/ds (critic) and dmu/ds (actor) gradients
  2. Leave-One-Out: zero-mask each dim, measure Q-value degradation

Usage:
    python scripts/feature_quality_diagnostic.py \
      --checkpoint checkpoints/.../checkpoint_final.pth \
      --config configs/gmgp1v6_sac_gc_15min.yaml \
      --n_samples 5000 --device cuda --wandb
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.getcwd())
from finrl_pro_ds.agents.sac.sac_agent import SACAgent
from scripts.run_full_pipeline import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fq1")

EXPERIMENT_TAG = "fq-1"

# ---------------------------------------------------------------------------
# Feature naming
# ---------------------------------------------------------------------------

SCALE_LABELS = ["s0_15m", "s1_60m", "s2_240m"]
BASE_FEATURE_NAMES = ["log_return", "atr_norm", "parkinson_vol", "close_z", "volume_z"]
WINDOW_FEATURE_NAMES = [
    "log_return", "atr_norm", "parkinson_vol",
    "open_z", "high_z", "low_z", "close_z", "volume_z",
]
STAT_TYPES = ["mean", "std", "last"]
PRIVATE_NAMES = ["position", "pnl_proxy", "time_sin", "time_cos", "atr_ratio"]


def build_feature_names(
    n_scales: int = 3,
    base_features: list[str] | None = None,
    stat_types: list[str] | None = None,
    private_names: list[str] | None = None,
    obs_mode: str = "summary_stats",
    window_size: int = 30,
    features_per_scale: int = 8,
) -> list[str]:
    """Build human-readable names for each observation dimension.

    summary_stats mode (50 dims):
      [0:15]  scale_0: 5 features x 3 stats
      [15:30] scale_1: same
      [30:45] scale_2: same
      [45:50] private: 5 dims

    window mode (725 dims for 3 scales x 30 bars x 8 features + 5 private):
      Flattened as [scale_0(30*8), scale_1(30*8), scale_2(30*8), private(5)]
    """
    private_names = private_names or PRIVATE_NAMES

    if obs_mode == "summary_stats":
        base_features = base_features or BASE_FEATURE_NAMES
        stat_types = stat_types or STAT_TYPES
        names: list[str] = []
        for s in range(n_scales):
            label = SCALE_LABELS[s] if s < len(SCALE_LABELS) else f"s{s}"
            for stat in stat_types:
                for feat in base_features:
                    names.append(f"{label}_{stat}_{feat}")
        for pn in private_names:
            names.append(f"priv_{pn}")
        return names

    # Window mode: scale_i_bar_j_feature_k
    win_features = WINDOW_FEATURE_NAMES[:features_per_scale]
    names = []
    for s in range(n_scales):
        label = SCALE_LABELS[s] if s < len(SCALE_LABELS) else f"s{s}"
        for bar in range(window_size):
            for f_idx, feat in enumerate(win_features):
                names.append(f"{label}_b{bar:02d}_{feat}")
    for pn in private_names:
        names.append(f"priv_{pn}")
    return names


# ---------------------------------------------------------------------------
# Config + agent + env loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dates(config: dict, data_split: str) -> tuple[str, str, str]:
    """Return (start, end, norm_cutoff) for the requested split."""
    dc = config["data"]
    if data_split == "train":
        return dc["train_start_date"], dc["train_end_date"], dc["train_end_date"]
    if data_split == "val":
        return dc["val_start_date"], dc["val_end_date"], dc["train_end_date"]
    if data_split == "test":
        return dc["test_start_date"], dc["test_end_date"], dc["train_end_date"]
    raise ValueError(f"Unknown data_split: {data_split}")


def build_network_config(config: dict) -> dict:
    """Build network_config dict matching SACAgent constructor expectations."""
    net_cfg = config.get("network", {})
    feat_cfg = config.get("features", {})

    scale_encoder = dict(net_cfg.get("scale_encoder", {
        "input_size": 8, "channels": [32, 64, 64, 64],
        "kernel_size": 3, "output_dim": 64,
    }))

    obs_mode = net_cfg.get("obs_mode", feat_cfg.get("obs_mode", "window"))
    nc = {
        "obs_mode": obs_mode,
        "scale_encoder": scale_encoder,
        "private_dim": net_cfg.get("private_dim", 5),
        "fusion_dim": net_cfg.get("fusion_dim", 256),
        "window_size": net_cfg.get("window_size", feat_cfg.get("window_size", 30)),
        "n_scales": len(feat_cfg.get("scales", [15, 60, 240])),
        "action_dim": net_cfg.get("action_dim", 1),
    }
    if obs_mode == "summary_stats":
        nc["summary_input_dim"] = net_cfg.get("summary_input_dim", 50)
    return nc


def load_agent_and_env(args):
    """Load trained SAC agent + create environment for state collection."""
    config = load_config(args.config)

    # Override for analysis
    config["env"]["random_start"] = False
    config["env"]["episode_length"] = 0  # full rollout
    if "reward" in config.get("env", {}):
        config["env"]["reward"]["hindsight_weight"] = 0.0
    # Apply final fee from schedule
    fee_sched = config.get("env", {}).get("fee_schedule", [])
    if fee_sched:
        final_fee = fee_sched[-1].get("taker_fee", 0.0002)
        config["env"]["taker_fee"] = final_fee

    # Create env
    start_date, end_date, norm_cutoff = resolve_dates(config, args.data_split)
    env = make_env(config, start_date=start_date, end_date=end_date,
                   norm_cutoff_date=norm_cutoff)

    # Build agent (eager mode — no torch.compile)
    nc = build_network_config(config)
    agent_cfg = config.get("agents", {}).get("sac", {})
    agent = SACAgent(
        network_config=nc,
        lr_actor=agent_cfg.get("lr_actor", 3e-4),
        lr_critic=agent_cfg.get("lr_critic", 3e-4),
        lr_alpha=agent_cfg.get("lr_alpha", 3e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        tau=agent_cfg.get("tau", 0.005),
        batch_size=agent_cfg.get("batch_size", 256),
        buffer_size=1000,  # minimal — not training
        torch_compile=False,
        device=args.device,
    )
    agent.load(args.checkpoint)
    agent.actor.eval()
    agent.critic1.eval()
    agent.critic2.eval()
    log.info("Loaded checkpoint: %s (step %d)", args.checkpoint, agent.step_count)

    n_scales = nc.get("n_scales", 3)
    obs_mode = nc.get("obs_mode", "window")
    return agent, env, config, n_scales, obs_mode


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

def _flatten_obs(obs: dict, n_scales: int, obs_mode: str) -> np.ndarray:
    """Flatten Dict obs to (D,) vector matching replay buffer layout."""
    if obs_mode == "summary_stats":
        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
        parts.append(obs["private"])
        return np.concatenate(parts).astype(np.float32)
    # Window mode: flatten each scale's (W, F) window, then concat + private
    parts = [obs[f"scale_{i}"].flatten() for i in range(n_scales)]
    parts.append(obs["private"])
    return np.concatenate(parts).astype(np.float32)


def _obs_to_tensors(flat: np.ndarray, n_scales: int, obs_mode: str,
                    window_size: int, features_per_scale: int,
                    device: torch.device):
    """Convert flat obs to network-ready tensors.

    Returns (scale_input, private):
      summary_stats: ((1, D_flat), None)
      window: ((1, N, W, F), (1, private_dim))
    """
    flat_t = torch.as_tensor(flat, dtype=torch.float32)
    if obs_mode == "summary_stats":
        return flat_t.unsqueeze(0).to(device, non_blocking=True), None

    chunk = window_size * features_per_scale
    total_scale = n_scales * chunk
    scale_flat = flat_t[:total_scale]
    priv = flat_t[total_scale:]
    # Reshape to (N, W, F) then add batch dim
    scales = scale_flat.reshape(n_scales, window_size, features_per_scale)
    return (
        scales.unsqueeze(0).to(device, non_blocking=True),
        priv.unsqueeze(0).to(device, non_blocking=True),
    )


def collect_states(
    agent: SACAgent,
    env,
    n_samples: int,
    n_scales: int,
    obs_mode: str,
    window_size: int,
    features_per_scale: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Roll agent deterministically through env, collect (obs, action, Q) tuples."""
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    q1_list: list[float] = []
    q2_list: list[float] = []

    obs, _ = env.reset()
    collected = 0
    episodes = 0

    while collected < n_samples:
        flat = _flatten_obs(obs, n_scales, obs_mode)
        scale_input, priv = _obs_to_tensors(
            flat, n_scales, obs_mode, window_size, features_per_scale, device
        )

        with torch.no_grad():
            action, _ = agent.actor.sample(scale_input, priv, deterministic=True)
            q1 = agent.critic1(scale_input, private=priv, action=action)
            q2 = agent.critic2(scale_input, private=priv, action=action)

        obs_list.append(flat)
        action_list.append(action.cpu().numpy().squeeze(0))
        q1_list.append(q1.item())
        q2_list.append(q2.item())
        collected += 1

        action_np = action.cpu().numpy().squeeze(0)
        obs, _reward, terminated, truncated, _info = env.step(action_np)
        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1

    log.info("Collected %d states across %d episodes", collected, episodes + 1)
    return {
        "obs_flat": np.array(obs_list, dtype=np.float32),
        "actions": np.array(action_list, dtype=np.float32).reshape(-1, 1),
        "q1": np.array(q1_list, dtype=np.float32).reshape(-1, 1),
        "q2": np.array(q2_list, dtype=np.float32).reshape(-1, 1),
    }


# ---------------------------------------------------------------------------
# Jacobian analysis
# ---------------------------------------------------------------------------

def _batch_to_network(obs_batch_np: np.ndarray, obs_mode: str,
                      n_scales: int, window_size: int, features_per_scale: int,
                      device: torch.device, requires_grad: bool = False):
    """Convert flat obs batch to network-ready tensors with optional grad tracking.

    Returns (scale_input, private) matching network forward signature.
    For gradient computation, scale_input has requires_grad=True.
    """
    if obs_mode == "summary_stats":
        t = torch.as_tensor(obs_batch_np.copy(), dtype=torch.float32).to(
            device, non_blocking=True
        )
        if requires_grad:
            t.requires_grad_(True)
        return t, None, t  # (input, private, grad_leaf)

    B = obs_batch_np.shape[0]
    chunk = window_size * features_per_scale
    total_scale = n_scales * chunk

    # Build flat tensor for grad tracking
    flat_t = torch.as_tensor(obs_batch_np.copy(), dtype=torch.float32).to(
        device, non_blocking=True
    )
    if requires_grad:
        flat_t.requires_grad_(True)

    # Reshape scales from flat
    scale_flat = flat_t[:, :total_scale]
    priv = flat_t[:, total_scale:]
    scale_input = scale_flat.reshape(B, n_scales, window_size, features_per_scale)

    return scale_input, priv, flat_t  # grad_leaf is flat_t


def compute_jacobians(
    agent: SACAgent,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
    device: torch.device,
    obs_mode: str = "summary_stats",
    n_scales: int = 3,
    window_size: int = 30,
    features_per_scale: int = 8,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Compute per-sample Jacobians for critic and actor via batched autograd.

    Uses torch.autograd.grad on q.sum() / mu.sum() — valid because the forward
    pass has no cross-sample interactions, so grad[i] = per-sample Jacobian.

    Returns:
        critic_jacobian_mean_abs: (D,) mean |dQ/ds_i|
        critic_jacobian_std:      (D,) std of dQ/ds_i
        actor_jacobian_mean_abs:  (D,) mean |dmu/ds_i|
        actor_jacobian_std:       (D,) std of dmu/ds_i
        critic_jacobian_raw:      (N, D) full matrix
        actor_jacobian_raw:       (N, D) full matrix
    """
    N, D = obs_np.shape
    critic_grads: list[np.ndarray] = []
    actor_grads: list[np.ndarray] = []
    nan_count = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        act_t = torch.as_tensor(
            actions_np[start:end], dtype=torch.float32
        ).to(device, non_blocking=True)

        # --- Critic Jacobian: dQ/ds ---
        scale_c, priv_c, leaf_c = _batch_to_network(
            obs_np[start:end], obs_mode, n_scales, window_size,
            features_per_scale, device, requires_grad=True,
        )
        q = agent.critic1(scale_c, private=priv_c, action=act_t)
        (cg,) = torch.autograd.grad(q.sum(), leaf_c)
        cg_np = cg.detach().cpu().numpy()
        nan_mask = ~np.isfinite(cg_np)
        if nan_mask.any():
            nan_count += int(nan_mask.sum())
            cg_np = np.nan_to_num(cg_np, nan=0.0, posinf=0.0, neginf=0.0)
        critic_grads.append(cg_np)

        # --- Actor Jacobian: dmu/ds ---
        scale_a, priv_a, leaf_a = _batch_to_network(
            obs_np[start:end], obs_mode, n_scales, window_size,
            features_per_scale, device, requires_grad=True,
        )
        mu, _log_sigma = agent.actor.forward(scale_a, private=priv_a)
        (ag,) = torch.autograd.grad(mu.sum(), leaf_a)
        ag_np = ag.detach().cpu().numpy()
        nan_mask_a = ~np.isfinite(ag_np)
        if nan_mask_a.any():
            nan_count += int(nan_mask_a.sum())
            ag_np = np.nan_to_num(ag_np, nan=0.0, posinf=0.0, neginf=0.0)
        actor_grads.append(ag_np)

    if nan_count > 0:
        log.warning("Encountered %d NaN/Inf gradient values (replaced with 0)", nan_count)

    critic_jac = np.concatenate(critic_grads, axis=0)  # (N, D)
    actor_jac = np.concatenate(actor_grads, axis=0)    # (N, D)

    return {
        "critic_jacobian_mean_abs": np.mean(np.abs(critic_jac), axis=0),
        "critic_jacobian_std": np.std(critic_jac, axis=0),
        "actor_jacobian_mean_abs": np.mean(np.abs(actor_jac), axis=0),
        "actor_jacobian_std": np.std(actor_jac, axis=0),
        "critic_jacobian_raw": critic_jac,
        "actor_jacobian_raw": actor_jac,
    }


# ---------------------------------------------------------------------------
# Leave-One-Out analysis
# ---------------------------------------------------------------------------

def _forward_q_batch(agent, obs_np, actions_np, obs_mode, n_scales,
                     window_size, features_per_scale, device, batch_size):
    """Forward critic1 over obs_np in batches, return Q-values as (N,) numpy."""
    N = obs_np.shape[0]
    q_vals = np.zeros(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        scale_t, priv_t, _ = _batch_to_network(
            obs_np[start:end], obs_mode, n_scales, window_size,
            features_per_scale, device,
        )
        act_t = torch.as_tensor(actions_np[start:end], dtype=torch.float32).to(
            device, non_blocking=True
        )
        with torch.no_grad():
            q = agent.critic1(scale_t, private=priv_t, action=act_t)
        q_vals[start:end] = q.cpu().numpy().squeeze(-1)
    return q_vals


def compute_loo(
    agent: SACAgent,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
    device: torch.device,
    obs_mode: str = "summary_stats",
    n_scales: int = 3,
    window_size: int = 30,
    features_per_scale: int = 8,
    loo_mode: str = "zero",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Leave-One-Out: mask each dim, measure Q-value change.

    For window mode with many dims (725), only tests per-feature (8) and
    per-private (5) by masking all bars for each feature simultaneously.
    """
    N, D = obs_np.shape

    q_baseline = _forward_q_batch(
        agent, obs_np, actions_np, obs_mode, n_scales,
        window_size, features_per_scale, device, batch_size,
    )

    if loo_mode == "mean":
        replace_vals = obs_np.mean(axis=0)
    else:
        replace_vals = np.zeros(D, dtype=np.float32)

    # For window mode with many dims, do grouped LOO (per-feature across all bars)
    chunk = window_size * features_per_scale
    total_scale_dims = n_scales * chunk
    private_dim = D - total_scale_dims
    is_window = obs_mode != "summary_stats" and D > 100

    if is_window:
        # Group LOO: mask all bars of feature f across all scales
        n_groups = features_per_scale + private_dim
        delta_q_abs = np.zeros(n_groups, dtype=np.float32)
        delta_q_std = np.zeros(n_groups, dtype=np.float32)
        group_names: list[str] = []

        for f in range(features_per_scale):
            obs_masked = obs_np.copy()
            for s in range(n_scales):
                # Mask feature f at all bars in scale s
                indices = [s * chunk + bar * features_per_scale + f
                           for bar in range(window_size)]
                for idx in indices:
                    obs_masked[:, idx] = replace_vals[idx]
            q_masked = _forward_q_batch(
                agent, obs_masked, actions_np, obs_mode, n_scales,
                window_size, features_per_scale, device, batch_size,
            )
            diffs = q_baseline - q_masked
            delta_q_abs[f] = np.mean(np.abs(diffs))
            delta_q_std[f] = np.std(diffs)
            group_names.append(WINDOW_FEATURE_NAMES[f] if f < len(WINDOW_FEATURE_NAMES)
                               else f"feat_{f}")

        for p in range(private_dim):
            obs_masked = obs_np.copy()
            obs_masked[:, total_scale_dims + p] = replace_vals[total_scale_dims + p]
            q_masked = _forward_q_batch(
                agent, obs_masked, actions_np, obs_mode, n_scales,
                window_size, features_per_scale, device, batch_size,
            )
            diffs = q_baseline - q_masked
            idx = features_per_scale + p
            delta_q_abs[idx] = np.mean(np.abs(diffs))
            delta_q_std[idx] = np.std(diffs)
            group_names.append(PRIVATE_NAMES[p] if p < len(PRIVATE_NAMES)
                               else f"priv_{p}")

        q_base_mean_abs = np.mean(np.abs(q_baseline)) + 1e-8
        return {
            "loo_delta_q_abs": delta_q_abs,
            "loo_delta_q_rel": delta_q_abs / q_base_mean_abs,
            "loo_delta_q_std": delta_q_std,
            "loo_group_names": group_names,
            "loo_grouped": True,
        }

    # Standard per-dim LOO (summary_stats or small D)
    delta_q_abs = np.zeros(D, dtype=np.float32)
    delta_q_std = np.zeros(D, dtype=np.float32)

    for d in range(D):
        obs_masked = obs_np.copy()
        obs_masked[:, d] = replace_vals[d]
        q_masked = _forward_q_batch(
            agent, obs_masked, actions_np, obs_mode, n_scales,
            window_size, features_per_scale, device, batch_size,
        )
        diffs = q_baseline - q_masked
        delta_q_abs[d] = np.mean(np.abs(diffs))
        delta_q_std[d] = np.std(diffs)

    q_base_mean_abs = np.mean(np.abs(q_baseline)) + 1e-8
    return {
        "loo_delta_q_abs": delta_q_abs,
        "loo_delta_q_rel": delta_q_abs / q_base_mean_abs,
        "loo_delta_q_std": delta_q_std,
        "loo_grouped": False,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    feature_names: list[str],
    jacobian_results: dict[str, np.ndarray],
    loo_results: dict[str, np.ndarray],
    obs_mode: str = "summary_stats",
    n_scales: int = 3,
    window_size: int = 30,
    features_per_scale: int = 8,
) -> pd.DataFrame:
    """Combine Jacobian + LOO into ranked feature importance table.

    For window mode: aggregates 725-dim Jacobian to per-feature importance,
    then joins with grouped LOO results.
    """
    D = len(feature_names)
    is_window = obs_mode != "summary_stats" and D > 100

    if is_window:
        # Aggregate Jacobians by feature type (across all bars and scales)
        chunk = window_size * features_per_scale
        total_scale_dims = n_scales * chunk
        private_dim = D - total_scale_dims

        crit_raw = jacobian_results["critic_jacobian_mean_abs"]
        actor_raw = jacobian_results["actor_jacobian_mean_abs"]

        feat_names_agg: list[str] = []
        crit_agg: list[float] = []
        actor_agg: list[float] = []

        # Per-feature across all scales and bars
        for f in range(features_per_scale):
            indices = []
            for s in range(n_scales):
                for bar in range(window_size):
                    indices.append(s * chunk + bar * features_per_scale + f)
            crit_agg.append(float(np.mean(crit_raw[indices])))
            actor_agg.append(float(np.mean(actor_raw[indices])))
            feat_names_agg.append(
                WINDOW_FEATURE_NAMES[f] if f < len(WINDOW_FEATURE_NAMES)
                else f"feat_{f}"
            )

        # Private dims
        for p in range(private_dim):
            crit_agg.append(float(crit_raw[total_scale_dims + p]))
            actor_agg.append(float(actor_raw[total_scale_dims + p]))
            feat_names_agg.append(
                f"priv_{PRIVATE_NAMES[p]}" if p < len(PRIVATE_NAMES)
                else f"priv_{p}"
            )

        n_agg = len(feat_names_agg)
        loo_abs = loo_results.get("loo_delta_q_abs", np.zeros(n_agg))
        loo_rel = loo_results.get("loo_delta_q_rel", np.zeros(n_agg))
        loo_std = loo_results.get("loo_delta_q_std", np.zeros(n_agg))

        df = pd.DataFrame({
            "feature": feat_names_agg,
            "dim": list(range(n_agg)),
            "critic_grad": crit_agg,
            "actor_grad": actor_agg,
            "loo_dq": loo_abs[:n_agg],
            "loo_dq_rel": loo_rel[:n_agg],
            "loo_dq_std": loo_std[:n_agg],
        })

        # Also compute per-scale and per-bar aggregations
        scale_importance = []
        for s in range(n_scales):
            idx_start = s * chunk
            idx_end = idx_start + chunk
            scale_importance.append(float(np.mean(crit_raw[idx_start:idx_end])))
        log.info("Per-scale critic grad: %s",
                 {SCALE_LABELS[i]: f"{v:.6f}" for i, v in enumerate(scale_importance)})

        bar_importance = np.zeros(window_size)
        for bar in range(window_size):
            indices = []
            for s in range(n_scales):
                for f in range(features_per_scale):
                    indices.append(s * chunk + bar * features_per_scale + f)
            bar_importance[bar] = float(np.mean(crit_raw[indices]))
        log.info("Per-bar critic grad (last 5 bars): %s",
                 [f"{v:.6f}" for v in bar_importance[-5:]])

    else:
        # Standard per-dim aggregation
        df = pd.DataFrame({
            "feature": feature_names,
            "dim": list(range(D)),
            "critic_grad": jacobian_results["critic_jacobian_mean_abs"],
            "critic_grad_std": jacobian_results["critic_jacobian_std"],
            "actor_grad": jacobian_results["actor_jacobian_mean_abs"],
            "actor_grad_std": jacobian_results["actor_jacobian_std"],
            "loo_dq": loo_results["loo_delta_q_abs"],
            "loo_dq_rel": loo_results["loo_delta_q_rel"],
            "loo_dq_std": loo_results["loo_delta_q_std"],
        })

    # Ranks
    df["critic_rank"] = df["critic_grad"].rank(ascending=False).astype(int)
    df["loo_rank"] = df["loo_dq"].rank(ascending=False).astype(int)
    df["combined_rank"] = ((df["critic_rank"] + df["loo_rank"]) / 2).rank().astype(int)

    # Group labels
    def _group(name: str) -> str:
        if name.startswith("priv_") or name.startswith("priv "):
            return "private"
        parts = name.split("_")
        return parts[0] if parts[0] in ("s0", "s1", "s2") else "feature"

    df["group"] = df["feature"].apply(_group)

    return df.sort_values("combined_rank")


# ---------------------------------------------------------------------------
# Output: console
# ---------------------------------------------------------------------------

def print_results_table(df: pd.DataFrame) -> None:
    """Print ranked feature importance to console."""
    header = (
        f"{'Rank':>4}  {'Dim':>3}  {'Feature':<30}  "
        f"{'|dQ/ds|':>9}  {'LOO_dQ':>9}  {'|dmu/ds|':>9}  {'LOO_rel%':>8}"
    )
    log.info("")
    log.info("Feature Quality Diagnostic (FQ-1) — Ranked Feature Importance")
    log.info("=" * len(header))
    log.info(header)
    log.info("-" * len(header))
    for _, row in df.iterrows():
        log.info(
            "%4d  %3d  %-30s  %9.6f  %9.6f  %9.6f  %7.2f%%",
            row["combined_rank"], row["dim"], row["feature"],
            row["critic_grad"], row["loo_dq"], row["actor_grad"],
            row["loo_dq_rel"] * 100,
        )

    # Group summaries
    log.info("")
    log.info("Group Summary (mean |dQ/ds|):")
    group_means = df.groupby("group")["critic_grad"].mean().sort_values(ascending=False)
    for grp, val in group_means.items():
        log.info("  %-10s  %.6f", grp, val)


# ---------------------------------------------------------------------------
# Output: JSON
# ---------------------------------------------------------------------------

def save_results_json(
    df: pd.DataFrame,
    jacobian_results: dict,
    loo_results: dict,
    metadata: dict,
    output_path: str,
) -> None:
    """Save full results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output = {
        "experiment": EXPERIMENT_TAG,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "ranking": df[["feature", "dim", "combined_rank", "critic_grad",
                        "loo_dq", "actor_grad", "loo_dq_rel"]].to_dict("records"),
        "jacobian": {
            "critic_mean_abs": jacobian_results["critic_jacobian_mean_abs"].tolist(),
            "critic_std": jacobian_results["critic_jacobian_std"].tolist(),
            "actor_mean_abs": jacobian_results["actor_jacobian_mean_abs"].tolist(),
            "actor_std": jacobian_results["actor_jacobian_std"].tolist(),
        },
        "loo": {
            "delta_q_abs": loo_results["loo_delta_q_abs"].tolist(),
            "delta_q_rel": loo_results["loo_delta_q_rel"].tolist(),
        },
        "group_summary": df.groupby("group")["critic_grad"].mean().to_dict(),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Output: WandB
# ---------------------------------------------------------------------------

def log_to_wandb(
    df: pd.DataFrame,
    jacobian_results: dict,
    feature_names: list[str],
    metadata: dict,
) -> None:
    """Log all results to WandB."""
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed, skipping WandB logging")
        return

    wandb.init(
        project="FinRL-Pro-DS",
        entity="bigcan-chiwin-technology",
        name=f"fq1_{metadata.get('run_id', 'local')}",
        tags=[EXPERIMENT_TAG, "feature-quality", "jacobian", "loo", "diagnostic"],
        config=metadata,
        job_type="analysis",
    )

    # 1. Feature importance table
    wandb.log({"fq1/feature_importance": wandb.Table(dataframe=df.reset_index(drop=True))})

    # 2. Critic Jacobian bar chart
    df_sorted = df.sort_values("critic_grad", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = {"s0": "#2196F3", "s1": "#4CAF50", "s2": "#FF9800", "priv": "#E91E63"}
    bar_colors = [colors.get(g, "#999") for g in df_sorted["group"]]
    ax.barh(df_sorted["feature"], df_sorted["critic_grad"], color=bar_colors)
    ax.set_xlabel("Mean |dQ/ds|")
    ax.set_title("Critic Jacobian — Feature Sensitivity")
    fig.tight_layout()
    wandb.log({"fq1/critic_jacobian_bar": wandb.Image(fig)})
    plt.close(fig)

    # 3. Actor Jacobian bar chart
    df_sorted_a = df.sort_values("actor_grad", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    bar_colors_a = [colors.get(g, "#999") for g in df_sorted_a["group"]]
    ax.barh(df_sorted_a["feature"], df_sorted_a["actor_grad"], color=bar_colors_a)
    ax.set_xlabel("Mean |dmu/ds|")
    ax.set_title("Actor Jacobian — Policy Sensitivity")
    fig.tight_layout()
    wandb.log({"fq1/actor_jacobian_bar": wandb.Image(fig)})
    plt.close(fig)

    # 4. LOO bar chart
    df_sorted_l = df.sort_values("loo_dq", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    bar_colors_l = [colors.get(g, "#999") for g in df_sorted_l["group"]]
    ax.barh(df_sorted_l["feature"], df_sorted_l["loo_dq"], color=bar_colors_l)
    ax.set_xlabel("Mean |delta Q|")
    ax.set_title("Leave-One-Out — Q-Value Degradation")
    fig.tight_layout()
    wandb.log({"fq1/loo_delta_q_bar": wandb.Image(fig)})
    plt.close(fig)

    # 5. Gradient correlation heatmap
    cjr = jacobian_results["critic_jacobian_raw"]
    corr = np.corrcoef(cjr.T)  # (D, D)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=5)
    ax.set_title("Critic Jacobian — Feature Gradient Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wandb.log({"fq1/gradient_correlation_heatmap": wandb.Image(fig)})
    plt.close(fig)

    # 6. Actor vs Critic agreement scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    for grp in ["s0", "s1", "s2", "priv"]:
        mask = df["group"] == grp
        ax.scatter(df.loc[mask, "critic_grad"], df.loc[mask, "actor_grad"],
                   label=grp, color=colors.get(grp, "#999"), s=40, alpha=0.8)
    ax.set_xlabel("Critic |dQ/ds|")
    ax.set_ylabel("Actor |dmu/ds|")
    ax.set_title("Actor vs Critic Feature Sensitivity")
    ax.legend()
    # Diagonal reference line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="#ccc", linewidth=0.8)
    fig.tight_layout()
    wandb.log({"fq1/actor_critic_agreement": wandb.Image(fig)})
    plt.close(fig)

    # 7. Scale group comparison
    group_means = df.groupby("group")[["critic_grad", "actor_grad", "loo_dq"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(group_means))
    w = 0.25
    ax.bar(x - w, group_means["critic_grad"], w, label="Critic |dQ/ds|")
    ax.bar(x, group_means["actor_grad"], w, label="Actor |dmu/ds|")
    ax.bar(x + w, group_means["loo_dq"], w, label="LOO |dQ|")
    ax.set_xticks(x)
    ax.set_xticklabels(group_means.index)
    ax.set_ylabel("Mean Importance")
    ax.set_title("Feature Importance by Scale Group")
    ax.legend()
    fig.tight_layout()
    wandb.log({"fq1/scale_group_comparison": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()
    log.info("WandB logging complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature Quality Diagnostic (FQ-1): Jacobian + LOO analysis"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to SAC checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n_samples", type=int, default=5000, help="States to collect")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--wandb", action="store_true", help="Log to WandB")
    parser.add_argument("--loo_mode", default="zero", choices=["zero", "mean"],
                        help="LOO replacement: zero-mask or dataset mean")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_split", default="val", choices=["train", "val", "test"],
                        help="Data split for state collection")
    parser.add_argument("--run_id", default="", help="Override run ID for output naming")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Load ---
    agent, env, config, n_scales, obs_mode = load_agent_and_env(args)
    feat_cfg = config.get("features", {})
    window_size = feat_cfg.get("window_size", 30)
    features_per_scale = feat_cfg.get("features_per_scale", 8)

    feature_names = build_feature_names(
        n_scales=n_scales, obs_mode=obs_mode,
        window_size=window_size, features_per_scale=features_per_scale,
    )
    D = len(feature_names)
    log.info("Observation: %s mode, %d dims", obs_mode, D)

    # --- Collect states ---
    log.info("Collecting %d states from %s split...", args.n_samples, args.data_split)
    states = collect_states(
        agent, env, args.n_samples, n_scales, obs_mode,
        window_size, features_per_scale, device,
    )
    obs_np = states["obs_flat"]
    actions_np = states["actions"]
    log.info("Q-value stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
             states["q1"].mean(), states["q1"].std(),
             states["q1"].min(), states["q1"].max())

    # --- Jacobian analysis ---
    log.info("Computing Jacobians (%d samples, batch_size=%d)...", len(obs_np), args.batch_size)
    jac_results = compute_jacobians(
        agent, obs_np, actions_np, device, obs_mode,
        n_scales, window_size, features_per_scale, args.batch_size,
    )
    log.info("Critic Jacobian: top-3 dims by |dQ/ds| = %s",
             np.argsort(jac_results["critic_jacobian_mean_abs"])[-3:][::-1].tolist())

    # --- LOO analysis ---
    log.info("Computing Leave-One-Out (mode=%s)...", args.loo_mode)
    loo_results = compute_loo(
        agent, obs_np, actions_np, device, obs_mode,
        n_scales, window_size, features_per_scale,
        args.loo_mode, args.batch_size,
    )
    log.info("LOO: top-3 by |delta Q| = %s",
             np.argsort(loo_results["loo_delta_q_abs"])[-3:][::-1].tolist())

    # --- Aggregate ---
    df = aggregate_results(
        feature_names, jac_results, loo_results, obs_mode,
        n_scales, window_size, features_per_scale,
    )
    print_results_table(df)

    # --- Metadata ---
    run_id = args.run_id or os.path.basename(os.path.dirname(args.checkpoint)) or "local"
    metadata = {
        "experiment": EXPERIMENT_TAG,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "n_samples": args.n_samples,
        "device": str(device),
        "loo_mode": args.loo_mode,
        "data_split": args.data_split,
        "run_id": run_id,
        "agent_step": agent.step_count,
        "alpha": agent.log_alpha.exp().item(),
        "q_mean": float(states["q1"].mean()),
        "q_std": float(states["q1"].std()),
    }

    # --- Save JSON ---
    json_path = f"results/fq1_{run_id}.json"
    save_results_json(df, jac_results, loo_results, metadata, json_path)

    # --- WandB ---
    if args.wandb:
        log_to_wandb(df, jac_results, feature_names, metadata)

    elapsed = time.time() - t0
    log.info("FQ-1 complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
