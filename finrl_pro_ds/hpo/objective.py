"""HPO objective factory — shared between serial and distributed HPO.

Extracted from scripts/run_full_pipeline.py (lines 681-1107).
The ``make_objective`` factory returns an Optuna objective closure that is
functionally identical to the inline objective previously defined inside
``run_hpo()``.
"""
import copy
import gc
import logging
import os

import numpy as np
import optuna
import torch
import wandb

from finrl_pro_ds.hpo.env_factory import create_vector_env
from finrl_pro_ds.hpo.evaluate import evaluate_for_hpo
from finrl_pro_ds.logging import trial_namespaced

logger = logging.getLogger("FinRL.HPO")


def _parse_frequency_to_minutes(freq: str) -> float:
    """Parse a frequency string (e.g. '1h', '15min', '4H', '30m') to minutes."""
    freq = freq.lower().strip()
    if freq.endswith("h"):
        return int(freq[:-1]) * 60
    if freq.endswith("min"):
        return int(freq[:-3])
    if freq.endswith("m"):
        return int(freq[:-1])
    if freq.endswith("d"):
        return int(freq[:-1]) * 1440
    return 1  # fallback


def make_objective(base_config, steps_per_trial, agent_type, device, trial_records):
    """Factory that returns an Optuna objective closure.

    The returned callable has the signature ``objective(trial) -> float`` and
    contains the **exact same logic** as the inline objective previously defined
    inside ``run_hpo()`` in ``scripts/run_full_pipeline.py``.

    Args:
        base_config: Deep-copied experiment config (YAML).
        steps_per_trial: Training timesteps per HPO trial.
        agent_type: One of "sac", "ppo", "iqn", "bdq".
        device: PyTorch device string ("cuda" or "cpu").
        trial_records: Mutable list — one record appended per trial.

    Returns:
        Optuna objective function: ``(optuna.Trial) -> float``
    """

    # Dispatch: funding-arb uses a different env schema (`config.environment`,
    # `FundingArbEnv`, DSAC CVaR axes) and doesn't fit the V7/cmgp1 objective
    # below. Route to the dedicated factory so the distributed HPO stack works
    # unchanged. Detection is env-type driven — same config key the runner uses.
    _env_type = (
        (base_config.get("environment") or {}).get("type")
        or (base_config.get("env") or {}).get("type")
    )
    if _env_type == "funding_arb":
        from finrl_pro_ds.hpo.funding_arb_objective import make_funding_arb_objective
        return make_funding_arb_objective(
            base_config, steps_per_trial, agent_type, device, trial_records,
        )

    def objective(trial):
        _mean_train_reward = float('nan')
        trial_prefix = f"hpo/t{trial.number}"
        # Trial-role tagging: when `hpo.enqueue_baseline: true`, Optuna seeds
        # trial 0 with the config's baseline HPs (see run_full_pipeline.run_hpo
        # and distributed_hpo_coordinator). Tag it distinctly so top-K filters
        # in WandB don't confuse the anchor with a representative search trial.
        _enqueue_baseline = bool(base_config.get("hpo", {}).get("enqueue_baseline", False))
        _is_baseline_anchor = _enqueue_baseline and trial.number == 0
        _trial_role = "baseline" if _is_baseline_anchor else "search"
        wandb.log({
            f"{trial_prefix}/started": True,
            f"{trial_prefix}/trial_role": _trial_role,
        })
        if _is_baseline_anchor and wandb.run is not None:
            # Append 'baseline-anchor' tag + set baseline_anchor_trial config
            # on the parent run (idempotent, best-effort).
            try:
                existing_tags = list(wandb.run.tags or [])
                if "baseline-anchor" not in existing_tags:
                    wandb.run.tags = tuple(existing_tags + ["baseline-anchor"])
                wandb.config.update(
                    {"baseline_anchor_trial": int(trial.number)},
                    allow_val_change=True,
                )
            except Exception as _e:  # pragma: no cover — tag update is best-effort
                logger.warning("Failed to append baseline-anchor tag: %s", _e)

        # FIX BUG-08: Reset torch.compile/Dynamo state between HPO trials.
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()

        config = copy.deepcopy(base_config)
        config["training"]["total_timesteps"] = steps_per_trial

        # -----------------------------------------------------------
        # Agent-specific HPO hyperparameter sampling
        # -----------------------------------------------------------
        if agent_type == "sac":
            narrow = base_config.get("hpo", {}).get("narrow_mode", False)
            if narrow:
                bs = base_config["agents"]["sac"]
                be = base_config["env"]
                be_r = be["reward"]
                nf = 0.2
                lr_actor = trial.suggest_float("lr_actor", bs["lr_actor"] * (1 - nf), bs["lr_actor"] * (1 + nf), log=True)
                lr_critic = trial.suggest_float("lr_critic", bs["lr_critic"] * (1 - nf), bs["lr_critic"] * (1 + nf), log=True)
                lr_alpha = trial.suggest_float("lr_alpha", bs["lr_alpha"] * (1 - nf), bs["lr_alpha"] * (1 + nf), log=True)
                tau = trial.suggest_float("tau", bs["tau"] * (1 - nf), bs["tau"] * (1 + nf), log=True)
                gamma = trial.suggest_float("gamma", max(0.90, bs["gamma"] - 0.02), min(0.999, bs["gamma"] + 0.02), log=True)
                initial_alpha = trial.suggest_float("initial_alpha", bs["initial_alpha"] * (1 - nf), bs["initial_alpha"] * (1 + nf), log=True)
                deadband = trial.suggest_categorical("deadband_threshold", [be["deadband_threshold"]])
                dsr_eta = trial.suggest_float("dsr_eta", be_r["dsr_eta"] * (1 - nf), be_r["dsr_eta"] * (1 + nf), log=True)
                gradient_clip = trial.suggest_float("gradient_clip", bs["gradient_clip"] * (1 - nf), bs["gradient_clip"] * (1 + nf), log=True)
            else:
                ss = base_config.get("hpo", {}).get("search_space", {})

                def _ss_f(name, lo, hi, **kw):
                    bounds = ss.get(name, {})
                    return trial.suggest_float(name, bounds.get("low", lo), bounds.get("high", hi), **kw)

                lr_actor = _ss_f("lr_actor", 2e-6, 1e-3, log=True)
                lr_critic = _ss_f("lr_critic", 2e-6, 1e-3, log=True)
                lr_alpha = _ss_f("lr_alpha", 2e-6, 1e-3, log=True)
                tau = _ss_f("tau", 2e-6, 0.01, log=True)
                gamma = _ss_f("gamma", 0.95, 0.999)
                initial_alpha = trial.suggest_float("initial_alpha", 0.05, 0.5, log=True)
                db_ss = ss.get("deadband_threshold", {})
                if db_ss.get("type") == "float":
                    deadband = trial.suggest_float("deadband_threshold", db_ss["low"], db_ss["high"])
                else:
                    deadband = trial.suggest_categorical("deadband_threshold", [0.15, 0.25, 0.35])
                dsr_eta = trial.suggest_float("dsr_eta", 0.0005, 0.01, log=True)
                gradient_clip = trial.suggest_float("gradient_clip", 1.0, 10.0, log=True)

            config["agents"]["sac"]["lr_actor"] = lr_actor
            config["agents"]["sac"]["lr_critic"] = lr_critic
            config["agents"]["sac"]["lr_alpha"] = lr_alpha
            config["agents"]["sac"]["tau"] = tau
            config["agents"]["sac"]["gamma"] = gamma
            config["agents"]["sac"]["initial_alpha"] = initial_alpha
            config["agents"]["sac"]["gradient_clip"] = gradient_clip
            config["env"]["deadband_threshold"] = deadband
            if "reward" not in config.get("env", {}):
                config["env"]["reward"] = {}
            config["env"]["reward"]["dsr_eta"] = dsr_eta

            # Leverage research axis (plan-a-new-research-lexical-sunrise.md).
            # Sampled only when hpo.search_space.max_leverage is declared so
            # legacy configs are unaffected. Bounds are validated upstream by
            # scripts/validate_config.py:check_max_leverage_bounds [0.5, 5.0].
            ss = base_config.get("hpo", {}).get("search_space", {})
            lev_ss = ss.get("max_leverage")
            if isinstance(lev_ss, dict):
                lev_lo = float(lev_ss.get("low", 0.5))
                lev_hi = float(lev_ss.get("high", 3.0))
                lev_log = bool(lev_ss.get("log", True))
                max_leverage = trial.suggest_float(
                    "max_leverage", lev_lo, lev_hi, log=lev_log,
                )
                config["env"]["max_leverage"] = max_leverage

            # Config-driven batch_size HPO (CMGP1+)
            ss = base_config.get("hpo", {}).get("search_space", {})
            bs_ss = ss.get("batch_size", {})
            if bs_ss.get("type") == "categorical" and bs_ss.get("choices"):
                batch_size = trial.suggest_categorical("batch_size", bs_ss["choices"])
                config["agents"]["sac"]["batch_size"] = batch_size

            # V8 MM: reward mode is an HPO categorical dimension
            if config.get("env", {}).get("mdp_version") == "v8":
                reward_mode = trial.suggest_categorical(
                    "reward_mode", ["dsr_pv", "pv_return", "dsr_simple"]
                )
                config["env"]["reward"]["mode"] = reward_mode

            # V9 MM: 1D skew-only — HPO searches base_spread_bps + max_skew_bps
            # Reward mode fixed to dsr_simple (best from v4 analysis)
            if config.get("env", {}).get("mdp_version") == "v9":
                base_spread = trial.suggest_float("base_spread_bps", 0.5, 5.0)
                max_skew = trial.suggest_float("max_skew_bps", 1.0, 5.0)
                config["env"]["base_spread_bps"] = base_spread
                config["env"]["max_skew_bps"] = max_skew
                config["env"]["reward"]["mode"] = "dsr_simple"

            # v6: Hard risk constraints (opt-in via config or search_space)
            sl_ss = ss.get("stop_loss_bps", {})
            if config.get("env", {}).get("stop_loss_hpo", False) or sl_ss:
                stop_loss_bps = trial.suggest_int("stop_loss_bps", sl_ss.get("low", 20), sl_ss.get("high", 200))
                config["env"]["stop_loss_bps"] = stop_loss_bps
            mh_ss = ss.get("max_holding_bars", {})
            if config.get("env", {}).get("max_holding_hpo", False) or mh_ss:
                max_holding_bars = trial.suggest_int("max_holding_bars", mh_ss.get("low", 5), mh_ss.get("high", 40))
                config["env"]["max_holding_bars"] = max_holding_bars

            hpo_log = {
                f"{trial_prefix}/lr_actor": lr_actor,
                f"{trial_prefix}/lr_critic": lr_critic,
                f"{trial_prefix}/lr_alpha": lr_alpha,
                f"{trial_prefix}/tau": tau,
                f"{trial_prefix}/gamma": gamma,
                f"{trial_prefix}/initial_alpha": initial_alpha,
                f"{trial_prefix}/gradient_clip": gradient_clip,
                f"{trial_prefix}/deadband_threshold": deadband,
                f"{trial_prefix}/dsr_eta": dsr_eta,
            }
            if config.get("env", {}).get("stop_loss_hpo", False) or ss.get("stop_loss_bps"):
                hpo_log[f"{trial_prefix}/stop_loss_bps"] = config["env"]["stop_loss_bps"]
            if config.get("env", {}).get("max_holding_hpo", False) or ss.get("max_holding_bars"):
                hpo_log[f"{trial_prefix}/max_holding_bars"] = config["env"]["max_holding_bars"]
            if ss.get("batch_size") and "batch_size" in config.get("agents", {}).get("sac", {}):
                hpo_log[f"{trial_prefix}/batch_size"] = config["agents"]["sac"]["batch_size"]
            if ss.get("max_leverage") and "max_leverage" in config.get("env", {}):
                hpo_log[f"{trial_prefix}/max_leverage"] = config["env"]["max_leverage"]
            if config.get("env", {}).get("mdp_version") == "v8":
                hpo_log[f"{trial_prefix}/reward_mode"] = config["env"]["reward"]["mode"]
            if config.get("env", {}).get("mdp_version") == "v9":
                hpo_log[f"{trial_prefix}/base_spread_bps"] = config["env"]["base_spread_bps"]
                hpo_log[f"{trial_prefix}/max_skew_bps"] = config["env"]["max_skew_bps"]
            wandb.log(hpo_log)

        elif agent_type == "ppo":
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
            ent_coef = trial.suggest_float("ent_coef", 0.005, 0.1, log=True)
            gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
            n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 8])
            target_kl = trial.suggest_float("target_kl", 0.01, 0.04)
            max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0])
            clip_eps = trial.suggest_categorical("clip_eps", [0.1, 0.2])

            config["agents"]["ppo"]["learning_rate"] = learning_rate
            config["agents"]["ppo"]["ent_coef"] = ent_coef
            config["agents"]["ppo"]["gae_lambda"] = gae_lambda
            config["agents"]["ppo"]["n_epochs"] = n_epochs
            config["agents"]["ppo"]["target_kl"] = target_kl
            config["agents"]["ppo"]["max_grad_norm"] = max_grad_norm
            config["agents"]["ppo"]["clip_eps"] = clip_eps

            wandb.log({
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/ent_coef": ent_coef,
                f"{trial_prefix}/gae_lambda": gae_lambda,
                f"{trial_prefix}/n_epochs": n_epochs,
                f"{trial_prefix}/target_kl": target_kl,
                f"{trial_prefix}/max_grad_norm": max_grad_norm,
                f"{trial_prefix}/clip_eps": clip_eps,
            })

        elif agent_type == "iqn":
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            num_quantiles = trial.suggest_categorical("num_quantiles", [8, 16, 32, 64])
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

            _iqn_exploration = config["agents"]["iqn"].get("exploration_mode", "noisy")
            if _iqn_exploration == "noisy":
                noisy_sigma0 = trial.suggest_float("noisy_sigma0", 0.3, 0.7)
            else:
                noisy_sigma0 = config["agents"]["iqn"].get("noisy_sigma0", 0.5)

            config["env"]["reward"]["sharpe_weight"] = 0.0
            config["env"]["reward"]["hindsight_weight"] = 0.0
            config["agents"]["iqn"]["learning_rate"] = learning_rate
            config["agents"]["iqn"]["num_quantiles"] = num_quantiles
            if _iqn_exploration == "noisy":
                config["agents"]["iqn"]["noisy_sigma0"] = noisy_sigma0
            config["agents"]["iqn"]["tau"] = tau

            is_multi_horizon = config["agents"]["iqn"].get("multi_horizon", False)
            if is_multi_horizon:
                gamma_short = trial.suggest_float("gamma_short", 0.90, 0.98)
                gamma_long = trial.suggest_float("gamma_long", 0.95, 0.999)
                config["agents"]["iqn"]["gamma_short"] = gamma_short
                config["agents"]["iqn"]["gamma_long"] = gamma_long
                config["agents"]["iqn"]["gamma"] = gamma_long
            else:
                gamma = trial.suggest_float("gamma", 0.93, 0.999)
                config["agents"]["iqn"]["gamma"] = gamma

            if config["env"].get("reward", {}).get("crra_gamma", 0.0) > 0:
                crra_gamma = trial.suggest_float("crra_gamma", 0.0, 1.5)
                config["env"]["reward"]["crra_gamma"] = crra_gamma

            if config["env"].get("reward", {}).get("mode") == "switch_centric":
                stay_reward_weight = trial.suggest_float("stay_reward_weight", 0.05, 0.3)
                config["env"]["reward"]["stay_reward_weight"] = stay_reward_weight

            if config["agents"]["iqn"].get("n_step_hpo", False):
                n_step = trial.suggest_categorical("n_step", [3, 5, 7, 10])
                config["agents"]["iqn"]["n_step"] = n_step

            if config["agents"]["iqn"].get("buffer_size_hpo", False):
                buffer_size = trial.suggest_int("buffer_size", 500000, 2000000, log=True)
                config["agents"]["iqn"]["buffer_size"] = buffer_size

            if config.get("network", {}).get("hidden_dim_hpo", False):
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
                config["network"]["micro_config"]["hidden_size"] = hidden_dim
                config["network"]["macro_config"]["hidden_sizes"] = [hidden_dim, hidden_dim // 2]

            hpo_log = {
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/num_quantiles": num_quantiles,
                f"{trial_prefix}/tau": tau,
            }
            if _iqn_exploration == "noisy":
                hpo_log[f"{trial_prefix}/noisy_sigma0"] = noisy_sigma0
            if config["env"].get("reward", {}).get("crra_gamma", 0.0) > 0:
                hpo_log[f"{trial_prefix}/crra_gamma"] = crra_gamma  # noqa: F821
            if is_multi_horizon:
                hpo_log[f"{trial_prefix}/gamma_short"] = gamma_short  # noqa: F821
                hpo_log[f"{trial_prefix}/gamma_long"] = gamma_long  # noqa: F821
            else:
                hpo_log[f"{trial_prefix}/gamma"] = gamma  # noqa: F821
            if config["env"].get("reward", {}).get("mode") == "switch_centric":
                hpo_log[f"{trial_prefix}/stay_reward_weight"] = stay_reward_weight  # noqa: F821
            if config["agents"]["iqn"].get("n_step_hpo", False):
                hpo_log[f"{trial_prefix}/n_step"] = n_step  # noqa: F821
            if config["agents"]["iqn"].get("buffer_size_hpo", False):
                hpo_log[f"{trial_prefix}/buffer_size"] = buffer_size  # noqa: F821
            if config.get("network", {}).get("hidden_dim_hpo", False):
                hpo_log[f"{trial_prefix}/hidden_dim"] = hidden_dim  # noqa: F821
            wandb.log(hpo_log)

        else:
            # BDQ hyperparams (4 dimensions)
            auxiliary_weight = trial.suggest_float("auxiliary_weight", 0.5, 1.5, log=True)
            learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
            epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.10)
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

            config["env"]["reward"]["sharpe_weight"] = 0.0
            config["env"]["reward"]["hindsight_weight"] = 0.0
            config["agents"]["bdq"]["auxiliary_weight"] = auxiliary_weight
            config["agents"]["bdq"]["learning_rate"] = learning_rate
            config["agents"]["bdq"]["epsilon_end"] = epsilon_end
            config["agents"]["bdq"]["tau"] = tau

            wandb.log({
                f"{trial_prefix}/learning_rate": learning_rate,
                f"{trial_prefix}/auxiliary_weight": auxiliary_weight,
                f"{trial_prefix}/epsilon_end": epsilon_end,
                f"{trial_prefix}/tau": tau,
            })

        # -----------------------------------------------------------
        # Create env and train (agent-agnostic)
        # -----------------------------------------------------------
        env = None
        eval_env = None
        try:
            hpo_num_envs = min(config.get("training", {}).get("num_envs",
                              config["env"].get("num_envs", 24)), 24)
            env = create_vector_env(config, num_envs=hpo_num_envs, gym_shm=False, use_sync=True)

            # S487 race-fix: trial-unique run_name so co-located replicas
            # don't clobber each other's checkpoints/sac_run/. Falls back to
            # a local-only prefix when DHPO_WORKER_ID is unset (serial HPO).
            _study_name = getattr(trial.study, "study_name", "hpo")
            _worker_id = os.environ.get("DHPO_WORKER_ID", "local")
            hpo_run_name = f"{_study_name}/worker_{_worker_id}/trial_{trial.number:04d}"
            if agent_type == "sac":
                from finrl_pro_ds.training.sac_trainer import SACTrainer
                trainer = SACTrainer(env, config, device=device, hpo_mode=True, run_name=hpo_run_name)
            elif agent_type == "ppo":
                from finrl_pro_ds.training.ppo_trainer import PPOTrainer
                trainer = PPOTrainer(env, config, device=device, hpo_mode=True, run_name=hpo_run_name)
            else:
                from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
                trainer = DeepScalperTrainer(env, config, device=device, hpo_mode=True, run_name=hpo_run_name)

            # V4.2: Evaluate on VALIDATION set (anti-overfitting)
            data_cfg = config.get("data", {})
            eval_config = copy.deepcopy(config)
            fee_schedule = eval_config.get("env", {}).get("fee_schedule")
            if fee_schedule:
                final_tier = fee_schedule[-1]
                mdp_ver_fee = eval_config.get("env", {}).get("mdp_version", "v5")
                if mdp_ver_fee == "v8":
                    final_fee = final_tier.get("ramp_to", final_tier.get("maker_fee", final_tier.get("taker_fee", 0.0)))
                    eval_config["env"]["maker_fee"] = final_fee
                else:
                    final_fee = final_tier.get("ramp_to", final_tier.get("taker_fee", 0.0))
                    eval_config["env"]["taker_fee"] = final_fee
                logger.info("[AUD-S129-01] HPO eval fee overridden from fee_schedule: %.6f", final_fee)
            eval_env = create_vector_env(
                eval_config, num_envs=1, gym_shm=False, use_sync=True,
                start_date=data_cfg.get("val_start_date"),
                end_date=data_cfg.get("val_end_date"),
                norm_cutoff_date=data_cfg.get("val_start_date"),
            )

            # Compute bar_minutes for correct Sharpe annualization
            mdp_ver = config.get("env", {}).get("mdp_version", "v5")
            hpo_scales = config.get("features", {}).get("scales", [])
            bar_duration_seconds = config.get("env", {}).get("bar_duration_seconds")
            if bar_duration_seconds is not None:
                hpo_bar_minutes = bar_duration_seconds / 60.0
            elif mdp_ver == "cmgp1":
                hpo_bar_minutes = _parse_frequency_to_minutes(config.get("data", {}).get("frequency", "1h"))
            elif mdp_ver in ("v7", "v8") and hpo_scales:
                hpo_bar_minutes = min(hpo_scales)
            elif "15min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 15
            elif "3min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 3
            elif "5min" in config.get("data", {}).get("file_path", ""):
                hpo_bar_minutes = 5
            else:
                hpo_bar_minutes = 1

            # Pruning callback (skipped when NopPruner is active — PERF-OPT S154)
            is_nop_pruner = isinstance(trial.study.pruner, optuna.pruners.NopPruner)
            if is_nop_pruner:
                pruning_callback = None
            else:
                def pruning_callback():
                    pf, tc = evaluate_for_hpo(eval_env, trainer.agent, max_steps=3000, bar_minutes=hpo_bar_minutes)
                    return pf

            trainer.train(optuna_trial=trial, pruning_callback=pruning_callback)

            if hasattr(trainer, 'episode_rewards') and len(trainer.episode_rewards) > 0:
                _mean_train_reward = float(np.mean(trainer.episode_rewards))

            # V4.2: Multi-seed eval for robust PF measurement (3 seeds, median).
            # B6 fix: also track per-seed eval completion fraction so post-hoc
            # leverage analysis can detect DD-termination selection bias
            # (high-L runs hit DD ~L^2 faster → smaller eval sample → biased PF).
            pf_values = []
            tc_values = []
            completion_pcts = []
            terminated_early_count = 0
            for eval_seed in [42, 123, 7]:
                eval_env.reset(seed=eval_seed)
                pf, tc, diag = evaluate_for_hpo(
                    eval_env, trainer.agent,
                    max_steps=50000, bar_minutes=hpo_bar_minutes,
                    return_diag=True,
                )
                pf_values.append(pf)
                tc_values.append(tc)
                completion_pcts.append(diag.get("_debug/eval_completion_pct", 1.0))
                terminated_early_count += int(diag.get("_debug/eval_terminated_early", 0))
            profit_factor = float(np.median(pf_values))
            trade_count = int(np.median(tc_values))
            mean_completion_pct = float(np.mean(completion_pcts))

            # Activity constraint — kill lazy holding agents
            min_trades = 30
            if trade_count < min_trades:
                wandb.log({f"{trial_prefix}/killed": "lazy_agent", f"{trial_prefix}/trades": trade_count})
                logger.info("Trial %d: KILLED (only %d trades, min=%d)", trial.number, trade_count, min_trades)
                trial_records.append({"trial": trial.number, "mean_reward": _mean_train_reward,
                                      "val_pf": -999.0, "trade_count": trade_count,
                                      "status": "killed_lazy", "hps": dict(trial.params)})
                return -999.0

            wandb.log({
                f"{trial_prefix}/profit_factor": profit_factor,
                f"{trial_prefix}/trade_count": trade_count,
                f"{trial_prefix}/eval_mean_completion_pct": mean_completion_pct,
                f"{trial_prefix}/eval_seeds_terminated_early": terminated_early_count,
                f"{trial_prefix}/completed": True,
            })
            logger.info(
                "Trial %d: PF=%.4f, Trades=%d, EvalCompletion=%.1f%% (early=%d/3)",
                trial.number, profit_factor, trade_count,
                mean_completion_pct * 100.0, terminated_early_count,
            )

            trial_records.append({"trial": trial.number, "mean_reward": _mean_train_reward,
                                  "val_pf": profit_factor, "trade_count": trade_count,
                                  "eval_completion_pct": mean_completion_pct,
                                  "eval_seeds_terminated_early": terminated_early_count,
                                  "status": "completed", "hps": dict(trial.params)})
            return profit_factor

        except optuna.TrialPruned:
            logger.info("Trial %d pruned.", trial.number)
            wandb.log({f"{trial_prefix}/status": "pruned"})
            trial_records.append({"trial": trial.number, "mean_reward": _mean_train_reward,
                                  "val_pf": 0.0, "trade_count": 0,
                                  "status": "pruned", "hps": dict(trial.params)})
            raise
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("Trial %d failed: %s\n%s", trial.number, e, tb)
            wandb.log({f"{trial_prefix}/error": str(e), f"{trial_prefix}/traceback": tb})
            trial_records.append({"trial": trial.number, "mean_reward": _mean_train_reward,
                                  "val_pf": 0.0, "trade_count": 0,
                                  "status": "error", "hps": dict(trial.params)})
            return 0.0
        finally:
            if env:
                try:
                    env.close()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
            if eval_env:
                try:
                    eval_env.close()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
            gc.collect()
            if hasattr(torch, '_dynamo'):
                torch._dynamo.reset()

    # S488 round-2: each trial's wandb.log calls auto-prefix `hpo/t<N>/*`
    # when the worker is attached to the coordinator's consolidated run.
    # Standalone (legacy) mode: no active namespace => pass-through, keys
    # already built with `trial_prefix` still work unchanged.
    return trial_namespaced(objective)
