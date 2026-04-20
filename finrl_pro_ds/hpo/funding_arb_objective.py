"""HPO objective for Funding-Arb + Distributional SAC.

Wired into `make_objective` (`finrl_pro_ds/hpo/objective.py`) via env-type
dispatch so the distributed HPO stack (coordinator → worker → objective)
can run funding-arb DSAC under protocol v2.

Differences from the V7/cmgp1 HPO path:
  - Reads `config["environment"]` (funding-arb uses this key; V7 uses `env`).
  - Builds `FundingArbEnv` per trial from cached ccxt fetch (no parquet).
  - Samples DSAC-specific axes when `agents.sac.distributional: true`:
    cvar_alpha, n_quantiles, kappa — in addition to SAC HPs.
  - Objective = profit_factor (BUG-01) computed step-level from val portfolio.
  - Uses window 0 from the walk-forward schedule for HPO (protocol v2
    stage 1 = single HPO artifact; stage 3 handles walk-forward rollover).
"""
from __future__ import annotations

import copy
import gc
import logging
import threading

import numpy as np
import optuna
import torch
import wandb

from finrl_pro_ds.logging import trial_namespaced

logger = logging.getLogger("FinRL.HPO.FundingArb")

# Module-level data cache — prepare_data makes a ccxt fetch that is slow and
# identical across trials. Guard with a lock so concurrent workers in the same
# process don't race on the cache miss.
_DATA_CACHE: dict[tuple, dict] = {}
_DATA_LOCK = threading.Lock()


def _data_cache_key(config: dict) -> tuple:
    data = config.get("data", {})
    universe = config.get("universe", {})
    return (
        universe.get("data_exchange", "binance"),
        tuple(universe.get("assets", [])),
        data.get("frequency", "1h"),
        data.get("start_date"),
        data.get("end_date"),
    )


def _get_or_fetch_data(config: dict) -> dict:
    key = _data_cache_key(config)
    with _DATA_LOCK:
        if key in _DATA_CACHE:
            return _DATA_CACHE[key]
        # Lazy imports so HPO stack doesn't pay funding-arb import cost unless used.
        from scripts.funding_arb_runner import prepare_data
        logger.info("Funding-arb data cache MISS for %s — fetching", key)
        data = prepare_data(config)
        _DATA_CACHE[key] = data
        return data


def _compute_profit_factor(portfolio_values: list[float] | np.ndarray) -> float:
    """Step-level profit_factor for funding-arb.

    Funding-arb doesn't have discrete entry/exit trades like momentum strategies
    (it's continuous delta-neutral position sizing + basis/funding PnL). The
    natural profit_factor analog is over step returns:
        PF = sum(positive step returns) / |sum(negative step returns)|
    which matches the BUG-01 invariant interpretation for a continuous-position
    strategy.
    """
    pv = np.asarray(portfolio_values, dtype=np.float64)
    if pv.size < 2:
        return 0.0
    rets = (pv[1:] / np.maximum(pv[:-1], 1e-10)) - 1.0
    pos = float(rets[rets > 0].sum())
    neg = float(np.abs(rets[rets < 0].sum()))
    if neg > 1e-12:
        return pos / neg
    return 10.0 if pos > 1e-12 else 0.0


def _build_window_arrays(data: dict, assets: list[str], window: dict) -> tuple[dict, dict]:
    """Build (train_arrays, val_arrays) from cached data + window schedule."""
    from finrl_pro_ds.crypto.data.crypto_array_builder import (
        build_funding_arb_arrays,
    )
    train_arrays = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["train_start"], window["train_end"],
    )
    val_arrays = build_funding_arb_arrays(
        data["spot_ohlcv"], data["perp_ohlcv"],
        data["arb_features"], data["funding"],
        assets, window["val_start"], window["val_end"],
    )
    return train_arrays, val_arrays


def _sample_hps(trial: optuna.Trial, base_config: dict) -> dict:
    """Mirror of funding_arb_hpo_runner.define_funding_arb_search_space.

    Returns dict with keys: agent_params, env_overrides, dsac_params (if
    distributional), and a flat hpo_log dict for WandB.
    """
    is_dist = base_config.get("agents", {}).get("sac", {}).get("distributional", False)

    agent_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50_000, 500_000, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
    }
    env_overrides = {
        "reward_scaling": trial.suggest_float("reward_scaling", 1000.0, 50000.0, log=True),
        "lambda_delta": trial.suggest_float("lambda_delta", 10.0, 500.0, log=True),
        "deadband_threshold": trial.suggest_float("deadband_threshold", 0.005, 0.05),
    }
    out = {"agent_params": agent_params, "env_overrides": env_overrides, "dsac_params": {}}
    if is_dist:
        out["dsac_params"] = {
            "cvar_alpha": trial.suggest_float("cvar_alpha", 0.10, 0.50),
            "n_quantiles": trial.suggest_categorical("n_quantiles", [16, 32, 64]),
            "kappa": trial.suggest_float("kappa", 0.5, 2.0),
        }
    return out


def make_funding_arb_objective(
    base_config: dict,
    steps_per_trial: int,
    agent_type: str,
    device: str,
    trial_records: list[dict],
):
    """Factory returning an Optuna objective for funding-arb + (D)SAC.

    Args:
        base_config: Deep-copied experiment config (YAML). Must have
            `environment.type: "funding_arb"` and typically
            `agents.sac.distributional: true` for DSAC.
        steps_per_trial: Training timesteps per HPO trial.
        agent_type: "sac" (distributional flag lives in config).
        device: "cuda" or "cpu".
        trial_records: mutable list — one record appended per trial.
    """
    if agent_type != "sac":
        raise ValueError(
            f"funding-arb HPO supports agent_type='sac' only (config toggles "
            f"distributional). Got agent_type={agent_type!r}."
        )

    def objective(trial: optuna.Trial) -> float:
        trial_prefix = f"hpo/t{trial.number}"
        wandb.log({f"{trial_prefix}/started": True})

        # BUG-08 parity with V7 path: reset dynamo between trials.
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

        # Lazy imports — only pay the cost when we actually run funding-arb.
        from scripts.funding_arb_hpo_runner import (
            _DSACModelWrapper,
            _make_dsac_agent,
            _train_dsac_agent,
        )
        from scripts.funding_arb_runner import (
            _evaluate_agent_on_env,
            _make_sb3_agent,
            create_env,
        )

        config = copy.deepcopy(base_config)
        is_dist = config.get("agents", {}).get("sac", {}).get("distributional", False)

        try:
            search = _sample_hps(trial, config)
        except Exception as e:
            logger.error("Trial %d HP sampling failed: %s", trial.number, e)
            raise
        agent_params = search["agent_params"]
        env_overrides = search["env_overrides"]
        dsac_params = search["dsac_params"]

        wandb.log({
            f"{trial_prefix}/lr": agent_params["learning_rate"],
            f"{trial_prefix}/gamma": agent_params["gamma"],
            f"{trial_prefix}/tau": agent_params["tau"],
            f"{trial_prefix}/batch_size": agent_params["batch_size"],
            f"{trial_prefix}/buffer_size": agent_params["buffer_size"],
            f"{trial_prefix}/reward_scaling": env_overrides["reward_scaling"],
            f"{trial_prefix}/lambda_delta": env_overrides["lambda_delta"],
            f"{trial_prefix}/deadband_threshold": env_overrides["deadband_threshold"],
            **{f"{trial_prefix}/dsac_{k}": v for k, v in dsac_params.items()},
        })

        # Apply env_overrides to config so create_env picks them up
        cfg_for_env = copy.deepcopy(config)
        for k, v in env_overrides.items():
            cfg_for_env["environment"][k] = v

        try:
            data = _get_or_fetch_data(config)
        except Exception as e:
            logger.error("Trial %d data fetch failed: %s", trial.number, e)
            trial_records.append({
                "trial": trial.number, "mean_reward": float("nan"),
                "val_pf": 0.0, "trade_count": 0,
                "status": "error", "hps": dict(trial.params),
            })
            return 0.0

        wf = data["walk_forward"]
        if not wf["passed"] or not wf["window_schedule"]:
            logger.error("Trial %d: walk-forward schedule empty or failed", trial.number)
            return 0.0

        # Protocol v2 stage 1: single HPO run per workstream → one window.
        # Window selection: configurable, default 0 (earliest). Stage 3
        # (walk-forward) is where rollover happens.
        hpo_cfg = config.get("hpo", {})
        w_idx = int(hpo_cfg.get("hpo_window_index", 0))
        schedule = wf["window_schedule"]
        if w_idx >= len(schedule):
            logger.warning(
                "hpo_window_index=%d out of range (%d windows); using 0",
                w_idx, len(schedule),
            )
            w_idx = 0
        window = schedule[w_idx]
        assets = config["universe"]["assets"]

        env = None
        val_env = None
        try:
            train_arrays, val_arrays = _build_window_arrays(data, assets, window)
            env = create_env(train_arrays, cfg_for_env)

            if is_dist:
                agent = _make_dsac_agent(env, cfg_for_env, agent_params, dsac_params)
                _train_dsac_agent(
                    agent, env, steps_per_trial,
                    global_step_offset=trial.number * steps_per_trial,
                )
                model = _DSACModelWrapper(agent)
            else:
                # Scalar SAC fallback (matches funding_arb_hpo_runner SB3 path).
                from stable_baselines3.common.vec_env import DummyVecEnv

                def _mk():
                    return create_env(train_arrays, cfg_for_env)
                vec_env = DummyVecEnv([_mk])
                sac_cfg = {
                    **agent_params,
                    "network_arch": config["agents"]["sac"].get("network_arch", [256, 256]),
                }
                model = _make_sb3_agent("sac", vec_env, sac_cfg)
                model.learn(total_timesteps=steps_per_trial)
                vec_env.close()

            # Evaluate on val — protocol v2 stage 1 gate.
            val_env = create_env(val_arrays, cfg_for_env)
            val_metrics = _evaluate_agent_on_env(model, val_env)
            pv = val_metrics.get("_portfolio_values")
            profit_factor = _compute_profit_factor(pv) if pv is not None else 0.0
            total_return = float(val_metrics.get("total_return", 0.0))

            # Trade count proxy = active-pair sum across eval (funding-arb has
            # continuous positions; "active pairs" count is the cleanest proxy
            # for whether the agent is actually trading vs sitting flat).
            trade_count = int(val_metrics.get("avg_active_pairs", 0.0) * val_metrics.get("n_steps", 0))

            wandb.log({
                f"{trial_prefix}/profit_factor": profit_factor,
                f"{trial_prefix}/total_return": total_return,
                f"{trial_prefix}/sharpe": val_metrics.get("sharpe", 0.0),
                f"{trial_prefix}/funding_vs_costs": val_metrics.get("funding_vs_costs_ratio", 0.0),
                f"{trial_prefix}/max_dd": val_metrics.get("max_drawdown", 0.0),
                f"{trial_prefix}/trade_count": trade_count,
                f"{trial_prefix}/completed": True,
            })

            logger.info(
                "Trial %d: PF=%.4f return=%.4f sharpe=%.3f funding/costs=%.2f",
                trial.number, profit_factor, total_return,
                val_metrics.get("sharpe", 0.0),
                val_metrics.get("funding_vs_costs_ratio", 0.0),
            )

            trial_records.append({
                "trial": trial.number, "mean_reward": total_return,
                "val_pf": profit_factor, "trade_count": trade_count,
                "status": "completed", "hps": dict(trial.params),
            })
            return profit_factor

        except optuna.TrialPruned:
            wandb.log({f"{trial_prefix}/status": "pruned"})
            trial_records.append({
                "trial": trial.number, "mean_reward": float("nan"),
                "val_pf": 0.0, "trade_count": 0,
                "status": "pruned", "hps": dict(trial.params),
            })
            raise
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("Trial %d failed: %s\n%s", trial.number, e, tb)
            wandb.log({
                f"{trial_prefix}/error": str(e),
                f"{trial_prefix}/traceback": tb,
            })
            trial_records.append({
                "trial": trial.number, "mean_reward": float("nan"),
                "val_pf": 0.0, "trade_count": 0,
                "status": "error", "hps": dict(trial.params),
            })
            return 0.0
        finally:
            for e in (env, val_env):
                if e is not None:
                    try:
                        e.close()
                    except Exception:
                        pass
            gc.collect()

    # S488 round-2: trial_namespaced wraps per-trial namespace management.
    return trial_namespaced(objective)
