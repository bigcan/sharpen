"""Planted-signal capability test for the real SAC stack (full-codebase audit 2026-09-23, section 3.7).

Everything that learns or scores is imported unchanged from the repo: MultiScaleOHLCVHandler,
ContinuousSwingEnv, create_vector_env, SACTrainer, SACAgent. Only the CONFIG is built here. The
defaults mirror ``configs/gmgp1_sac_gc_15min.yaml`` (production geometry and SAC hyperparameters),
except ``taker_fee = 0``: this is a capability test, so it is frictionless by design.

Splits (synthetic data starts 2018-01-01, 24/7 bars): train 2018-01-01..2018-10-01 (~26k 15-min bars,
comparable to the production training windows), test 2019-01-01..2020-12-31 (~70k bars). The trainer
does no validation or checkpoint selection, so the test window is untouched until evaluation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TRAIN_START, TRAIN_END = "2018-01-01", "2018-10-01"
TEST_START, TEST_END = "2019-01-01", "2020-12-31"
BARS_PER_YEAR = 365 * 96              # synthetic bars are 24/7 15-min bars
SIGMA = 0.0015                        # gen_synth marginal 15-min return sd (oracle scale)


def base_config(parquet: str, *, window: int = 30, channels=(32, 64, 64, 64), output_dim: int = 64,
                fusion: int = 256, deadband: float = 0.25, reward_mode: str = "dsr",
                episode_length: int = 500, total_timesteps: int = 3_000_000, num_envs: int = 20,
                batch_size: int = 512, buffer_size: int = 500_000, update_interval: int = 4,
                learning_starts: int = 5000, lr: float = 3e-4, initial_alpha: float = 0.2,
                actor_update_freq: int = 2, tau: float = 0.005, gamma: float = 0.99,
                use_amp: bool = True, torch_compile: bool = True, seed: int = 1,
                raw_channel_norm: str = "raw") -> dict:
    """Production-mirroring V7 SAC config on a synthetic parquet. ``raw_channel_norm="ema_z"``
    standardizes the log_return / parkinson channels (production feeds them raw, ~1e-3)."""
    return {
        "data": {
            "file_path": parquet, "ticker": "SYN", "handler_type": "multiscale",
            "train_start_date": TRAIN_START, "train_end_date": TRAIN_END,
            "val_start_date": TEST_START, "val_end_date": TEST_END,
            "test_start_date": TEST_START, "test_end_date": TEST_END,
        },
        "features": {"scales": [15, 60, 240], "window_size": window, "features_per_scale": 8,
                     "asset_class": "synthetic", "raw_channel_norm": raw_channel_norm},
        "env": {
            "mdp_version": "v7", "initial_balance": 100000.0, "window_size": window,
            "features_per_scale": 8, "taker_fee": 0.0, "deadband_threshold": deadband,
            "atr_cap_percentile": 90, "atr_cap_max_position": 0.5,
            "episode_length": episode_length, "random_start": True, "max_drawdown_pct": 0.30,
            "scales": [15, 60, 240],
            "reward": {"mode": reward_mode, "dsr_eta": 0.001, "dsr_scale": 1.0},
        },
        "network": {
            "scale_encoder": {"input_size": 8, "channels": list(channels), "kernel_size": 3,
                              "output_dim": output_dim},
            "private_dim": 5, "fusion_dim": fusion, "window_size": window,
        },
        "agents": {"sac": {
            "lr_actor": lr, "lr_critic": lr, "lr_alpha": lr, "gamma": gamma, "tau": tau,
            "batch_size": batch_size, "buffer_size": buffer_size, "initial_alpha": initial_alpha,
            "learning_starts": learning_starts, "update_interval": update_interval,
            "actor_update_freq": actor_update_freq, "gradient_clip": 10.0,
            "checkpoint_interval": 10 ** 12,
        }},
        "training": {
            "total_timesteps": total_timesteps, "log_interval": 10 ** 12, "num_envs": num_envs,
            "use_amp": use_amp, "amp_dtype": "bfloat16" if use_amp else "float32",
            "torch_compile": torch_compile, "use_shm": False,
        },
        "seed": seed,
    }


def make_eval_env(cfg: dict):
    """Single deterministic full-window test env: no random start, no truncation."""
    from sharpen.hpo.env_factory import make_env
    ec = copy.deepcopy(cfg)
    ec["env"]["episode_length"] = 0
    ec["env"]["random_start"] = False
    return make_env(ec, start_date=TEST_START, end_date=TEST_END,
                    norm_cutoff_date=TEST_START, seed=999)


def rollout(env, policy, max_steps: int = 10 ** 6) -> dict:
    """Roll ``policy(obs) -> float`` through a raw env. ``feat0`` is the last base-scale log return
    channel in the observation window as the policy saw it: the planted signal itself when features
    are raw, its EMA-Z → tanh transform when ``raw_channel_norm="ema_z"``."""
    obs, _ = env.reset()
    out = {k: [] for k in ("action", "position", "price_return", "reward", "equity", "feat0")}
    prev_close = env.current_close
    for _ in range(max_steps):
        f0 = float(obs["scale_0"][-1, 0])
        a = float(policy(obs))
        obs, rew, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        pr = (env.current_close - prev_close) / prev_close if prev_close > 0 else 0.0
        prev_close = env.current_close
        for k, v in (("action", a), ("position", info["position"]), ("price_return", pr),
                     ("reward", rew), ("equity", info["portfolio_value"]), ("feat0", f0)):
            out[k].append(v)
        if term or trunc:
            break
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def summarise(roll: dict) -> dict:
    """Frictionless per-bar P&L of the executed position; Sharpe at 24/7 15-min annualization.
    Correlations use ``roll["signal"]`` (the raw planted signal per decision) when present, else
    ``feat0``, so raw and standardized arms are scored against the same series."""
    pos, pr = roll["position"], roll["price_return"]
    f = roll.get("signal", roll["feat0"])
    pnl = pos * pr
    sd = pnl.std()
    sharpe = float(pnl.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-15 else 0.0

    def _corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 and y.std() > 1e-12 else 0.0

    return {
        "n_bars": int(len(pnl)), "sharpe_frictionless": sharpe,
        "sharpe_se": float(np.sqrt(BARS_PER_YEAR / max(len(pnl), 1))),
        "mean_pnl_bps": float(pnl.mean() * 1e4),
        "corr_action_signal": _corr(roll["action"], f), "corr_position_signal": _corr(pos, f),
        "mean_abs_position": float(np.abs(pos).mean()), "mean_position": float(pos.mean()),
        "action_std": float(roll["action"].std()),
        "turnover_per_bar": float(np.abs(np.diff(pos)).mean()) if len(pos) > 1 else 0.0,
        "total_return": float(roll["equity"][-1] / 100000.0 - 1.0),
        "obs_feat0_std": float(roll["feat0"].std()),   # scale of the signal channel the policy saw
    }


def oracle_policy(obs) -> float:
    """Linear oracle: position = clip(last base-scale return / sigma) (optimal sign for rho > 0)."""
    return float(np.clip(obs["scale_0"][-1, 0] / SIGMA, -1.0, 1.0))
