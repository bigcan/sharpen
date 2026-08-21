"""
SAC Trainer — Training loop for continuous position control.

Handles:
  - Vectorized env data collection
  - SAC agent update cycle
  - Fee curriculum (linear ramp)
  - WandB logging
  - Checkpointing
"""
import logging
import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

import wandb
from finrl_pro_ds.agents.sac.sac_agent import SACAgent

logger = logging.getLogger(__name__)


class SACTrainer:
    """Training loop for SAC agent with ContinuousSwingEnv."""

    def __init__(
        self,
        env,
        config: dict,
        device: str = "cuda",
        run_name: Optional[str] = None,
        hpo_mode: bool = False,
    ):
        self.env = env
        self.config = config
        self.device = device
        self.hpo_mode = hpo_mode
        self.run_name = run_name or "sac_run"

        # Build agent
        net_cfg = dict(config.get("network", {}))
        sac_cfg = config["agents"]["sac"]

        # Forward feature config to network config
        features_cfg = config.get("features", {})
        if "window_size" not in net_cfg:
            net_cfg["window_size"] = features_cfg.get("window_size", 30)
        scale_enc = net_cfg.get("scale_encoder", {})
        if "input_size" not in scale_enc:
            scale_enc["input_size"] = features_cfg.get("features_per_scale", 7)
        net_cfg["scale_encoder"] = scale_enc

        # Derive n_scales from config
        scales = features_cfg.get("scales", config.get("env", {}).get("scales", [3, 15, 60]))
        net_cfg["n_scales"] = len(scales)
        self._n_scales = len(scales)

        # v6: Summary-stats observation mode
        self._obs_mode = net_cfg.get("obs_mode", "window")

        # HPO: cap buffer_size to avoid wasting memory on short trials
        buffer_size = sac_cfg.get("buffer_size", 1_000_000)
        if hpo_mode:
            buffer_size = min(buffer_size, 100_000)

        # Common agent kwargs shared by SACAgent and DistributionalSACAgent
        _agent_kwargs = dict(
            network_config=net_cfg,
            lr_actor=sac_cfg.get("lr_actor", 3e-4),
            lr_critic=sac_cfg.get("lr_critic", 3e-4),
            lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
            gamma=sac_cfg.get("gamma", 0.99),
            tau=sac_cfg.get("tau", 0.005),
            batch_size=sac_cfg.get("batch_size", 256),
            buffer_size=buffer_size,
            initial_alpha=sac_cfg.get("initial_alpha", 0.2),
            learning_starts=sac_cfg.get("learning_starts", 10_000),
            update_interval=sac_cfg.get("update_interval", 4),
            gradient_clip=sac_cfg.get("gradient_clip", 10.0),
            use_amp=config["training"].get("use_amp", False),
            amp_dtype=config["training"].get("amp_dtype", "float16"),
            torch_compile=config["training"].get("torch_compile", False),
            checkpoint_interval=sac_cfg.get("checkpoint_interval", 500_000),
            device=device,
            actor_update_freq=sac_cfg.get("actor_update_freq", 2),
            crossq=sac_cfg.get("crossq"),
        )

        if sac_cfg.get("distributional", False):
            from finrl_pro_ds.agents.sac.dsac_agent import DistributionalSACAgent
            self.agent = DistributionalSACAgent(
                **_agent_kwargs,
                n_quantiles=sac_cfg.get("n_quantiles", 32),
                cvar_alpha=sac_cfg.get("cvar_alpha", 0.25),
                quantile_embed_dim=sac_cfg.get("quantile_embed_dim", 64),
                kappa=sac_cfg.get("kappa", 1.0),
            )
        else:
            self.agent = SACAgent(**_agent_kwargs)

        # Training config
        self.total_timesteps = config["training"]["total_timesteps"]
        self.log_interval = config["training"].get("log_interval", 1000)
        self.checkpoint_interval = sac_cfg.get("checkpoint_interval", 500_000)
        self.learning_starts = sac_cfg.get("learning_starts", 10_000)
        # update_interval is gradient steps per env.step() call (total, not per-env).
        # FIX OPT-10: Do NOT scale by num_envs. In vectorized RL, all envs share one
        # model — 20 envs produce 20 transitions per env.step but the model is the same.
        # Scaling by num_envs caused UTD=40 (2*20), meaning 40 gradient steps per
        # env.step → replay ratio 1024x, SPS=8 (vs ~80 without scaling).
        # Standard SAC (Haarnoja, CleanRL, SB3) uses UTD=1-2 regardless of num_envs.
        getattr(env, 'num_envs', 1)
        self.update_interval = sac_cfg.get("update_interval", 4)

        # Auto-scale tau for high UTD. CrossQ has no target network, so tau and
        # the Polyak update do not exist for it — scaling them would be a no-op
        # that reads, in the logs, like a knob that is doing something.
        if getattr(self.agent, "crossq", False):
            logger.info(
                f"[CrossQ] target network removed (BatchRenorm critic). "
                f"tau/Polyak inactive; update_interval={self.update_interval} unchanged.",
            )
        elif self.update_interval > 1:
            raw_tau = self.agent.tau
            effective_tau = 1.0 - (1.0 - raw_tau) ** (1.0 / self.update_interval)
            self.agent.tau = effective_tau
            logger.info(
                f"[UTD] update_interval={self.update_interval} → tau auto-scaled: "
                f"{raw_tau:.6f} → {effective_tau:.6f}",
            )

        # Fee curriculum
        raw_schedule = config.get("env", {}).get("fee_schedule", [])
        self._fee_schedule = sorted(raw_schedule, key=lambda x: x["step"]) if raw_schedule else []
        self._fee_tier_applied = -1

        # RCRP: Regime-balanced replay (Path 1)
        rbrp_mode = config["training"].get("regime_balanced_replay", None)
        if rbrp_mode:
            self.agent.regime_balanced_replay = rbrp_mode
            logger.info(f"[RCRP] Regime-balanced replay enabled: mode={rbrp_mode}")

        # Checkpoint dir
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    @staticmethod
    def _assert_finite_metrics(metrics: dict, total_steps: int) -> None:
        """Halt the run the first time a training metric goes non-finite (NAN-01).

        Without this a NaN loss is simply logged and the loop continues. That is not
        hypothetical: run ``pqwttrqd`` (randd_log S553-cont-165) reached 75,000/75,000, exited
        ``state=finished``, and wrote a full gate verdict JSON — with ``actor_loss``/
        ``critic_loss``/``alpha`` NaN from the first logged point after ``learning_starts``.
        A crashed run is loud; that one was silent and indistinguishable from a real result.
        **``state=finished`` is not evidence a run trained.**

        Cheap by construction: ``train_step_mega`` already paid the ``.item()`` device syncs to
        build this dict, so these are plain Python floats — checking them costs nothing. The
        alternative, a ``torch.isnan`` probe inside the update loop, would force a sync on every
        gradient step and break the CPU/GPU overlap OPT-07/08 exists for.

        Raising (rather than warning) is right because NaN here is TERMINAL, not transient. AMP
        does have a legitimate transient path — ``GradScaler`` skipping a step when ``unscale_``
        finds inf/NaN — but that is NaN in the GRADIENTS and it self-corrects. A NaN in a metric
        means the FORWARD pass produced it, i.e. weights or inputs are already NaN, and neither
        un-NaNs itself. Failing at the first bad step turns a 27-minute silent waste into a
        loud one a few thousand steps in.

        Containment is ``SACAgent._guard_alpha_grad``'s job; this is detection. The two are
        deliberately separate — the guard must not suppress the signal this reads.
        """
        bad = {k: v for k, v in metrics.items()
               if isinstance(v, (int, float)) and not math.isfinite(float(v))}
        if bad:
            raise RuntimeError(
                f"NAN-01: non-finite training metric(s) {sorted(bad)} at total_steps="
                f"{total_steps} — halting instead of logging NaN for the rest of the run "
                f"(a 'finished' run with NaN losses looks exactly like a real result). "
                f"metrics={metrics}"
            )

    def train(self, optuna_trial=None, pruning_callback=None) -> str:
        """Main training loop. Returns path to final checkpoint."""
        num_envs = getattr(self.env, 'num_envs', 1)
        # FIX SEED-01: seed the initial reset. Without this the envs self-seed from
        # OS entropy and --seed never reaches them, so nominally identical runs
        # diverge (measured: 11.1pp on gmgp1_spx500_lo_wf_f1 at --seed 42). Prefer
        # the base seed recorded by create_vector_env; fall back to config.
        _seed = getattr(self.env, "finrl_base_seed", None)
        if _seed is None:
            _seed = self.config.get("seed")
        if _seed is None:
            logger.warning(
                "[SEED-01] No seed available — env episode starts are NOT reproducible. "
                "Pass --seed to run_full_pipeline for a reproducible run.",
            )
            obs, info = self.env.reset()
        else:
            logger.info("[SEED-01] Seeding env reset with base seed %d", int(_seed))
            obs, info = self.env.reset(seed=int(_seed))

        episode_reward = np.zeros(num_envs, dtype=np.float64)
        episode_length = np.zeros(num_envs, dtype=np.int64)
        total_steps = 0
        t_start = time.time()
        gradient_accumulator = 0.0
        metrics = None  # FIX R2-AUD-01: Initialize before learning_starts to prevent NameError
        # FIX R7-AUD-01: Track which envs just terminated/truncated.
        # Gymnasium 1.x auto-resets on the NEXT step after done, producing a
        # phantom step (reward=0, empty info, terminal→reset obs) that should
        # NOT be stored in the replay buffer.
        prev_any_done = np.zeros(num_envs, dtype=bool)

        # Limit PyTorch intra-op threads for SyncVectorEnv.
        # Default num_threads = all CPU cores → massive context-switch overhead
        # for small-batch ops (predict with batch=num_envs).
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 4))

        logger.info(
            f"SAC Training: {self.total_timesteps} steps, "
            f"{num_envs} envs, device={self.device}, "
            f"torch_threads={torch.get_num_threads()}",
        )

        # OPT-08: Pipeline overlap — GPU training runs in background thread while
        # CPU does env.step(). Deferred store ensures buffer writes don't race with sample().
        # Timeline per iteration:
        #   1. Wait for prev train (GPU done, buffer safe)
        #   2. store_batch (deferred from prev iteration)
        #   3. predict() (GPU, fast ~10ms)
        #   4. Submit train (GPU background — samples buffer, runs gradients)
        #   5. env.step() (CPU, ~200-300ms — overlaps with GPU training)
        train_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sac_train")
        train_future: Optional[Future] = None
        _deferred_store = None

        while total_steps < self.total_timesteps:
            # Diagnostic: one-time step-0 heartbeat to confirm loop entry
            if total_steps == 0:
                logger.info("[DIAG] Training loop entered, step 0")
                t_diag = time.time()

            # Apply fee curriculum
            self._apply_fee_schedule(total_steps)

            # OPT-08: Wait for previous training future — must complete before
            # store_batch (buffer write safety) and predict (GPU contention).
            if train_future is not None:
                m = train_future.result()
                if m is not None:
                    self._assert_finite_metrics(m, total_steps)
                    metrics = m
                train_future = None

            # Store PREVIOUS iteration's transitions (deferred from last loop).
            # Buffer write is safe here: training future is resolved above.
            if _deferred_store is not None:
                d_obs, d_act, d_rew, d_nobs, d_dones, d_vmask, d_rc = _deferred_store
                if np.all(d_vmask):
                    self.agent.store_batch(d_obs, d_act, d_rew, d_nobs, d_dones, regime_codes=d_rc)
                elif np.any(d_vmask):
                    v_obs = {k: v[d_vmask] for k, v in d_obs.items()}
                    v_next = {k: v[d_vmask] for k, v in d_nobs.items()}
                    v_rc = d_rc[d_vmask] if d_rc is not None else None
                    self.agent.store_batch(
                        v_obs, d_act[d_vmask], d_rew[d_vmask],
                        v_next, d_dones[d_vmask],
                        regime_codes=v_rc,
                    )
                _deferred_store = None

            # Get actions — prepare obs for agent.predict()
            if self._obs_mode == "summary_stats":
                # v6: Concat all scale summaries + private into flat (B, D) tensor
                parts = [obs[f"scale_{i}"] for i in range(self._n_scales)]
                parts.append(obs["private"])
                flat_np = np.concatenate(parts, axis=1)  # (B, D)
                scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).to(
                    self.device, non_blocking=True,
                )
                priv = None
            else:
                # Window mode: stack all scales into ONE (B,N,W,F) tensor, ONE H2D transfer
                # FIX OPT-01: Was creating N separate tensors + N .to(device) calls per step.
                # np.stack is cheap (contiguous source from SyncVectorEnv), single torch transfer.
                scale_np = np.stack(
                    [obs[f"scale_{i}"] for i in range(self._n_scales)], axis=1,
                )  # (B, N, W, F)
                scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).to(
                    self.device, non_blocking=True,
                )
                priv = torch.as_tensor(obs["private"], dtype=torch.float32).to(
                    self.device, non_blocking=True,
                )

            # Optional LOB microstructure features (Market Making V8)
            lob = None
            if "lob" in obs:
                lob = torch.as_tensor(obs["lob"], dtype=torch.float32).to(
                    self.device, non_blocking=True,
                )

            with torch.no_grad():
                actions = self.agent.predict(scale_stack, priv, deterministic=False, lob=lob)
                actions_np = actions.cpu().numpy()

            # Training updates — OPT-07+08: submit BEFORE env.step so GPU training
            # runs concurrently with CPU env work. Buffer was written above (safe).
            if total_steps >= self.learning_starts:
                gradient_accumulator += self.update_interval
                n_steps = int(gradient_accumulator)
                if n_steps >= 1:
                    train_future = train_executor.submit(self.agent.train_step_mega, n_steps)
                    gradient_accumulator -= n_steps

            # Step environment — CPU-bound, overlaps with GPU training (OPT-08)
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_np)

            # Defer transition storage to next iteration (after train_future resolves)
            # FIX R6-AUD-01: Only true termination sets done=1.0 (not truncation).
            dones_for_buffer = terms.astype(np.float32)
            # FIX R7-AUD-01: valid_mask filters phantom auto-reset transitions
            valid_mask = ~prev_any_done
            # RCRP: Extract regime codes from infos for replay balancing
            rc = infos.get("regime_code")
            regime_codes = rc.astype(np.int8) if rc is not None else None
            _deferred_store = (obs, actions_np, rewards, next_obs, dones_for_buffer, valid_mask, regime_codes)

            # Episode tracking uses actual episode boundaries (term OR trunc)
            any_done = np.logical_or(terms, truncs)

            # Diagnostic: first 100 steps timing
            if total_steps == 100 * num_envs:
                dt = time.time() - t_diag
                logger.info(f"[DIAG] First {total_steps} steps took {dt:.1f}s ({total_steps/dt:.0f} SPS)")

            # Track episodes
            episode_reward += rewards
            episode_length += 1

            for i in range(num_envs):
                if any_done[i]:
                    self.episode_rewards.append(episode_reward[i])
                    self.episode_lengths.append(episode_length[i])
                    episode_reward[i] = 0.0
                    episode_length[i] = 0

            prev_any_done = any_done.copy()
            obs = next_obs
            total_steps += num_envs
            self.agent.step_count = total_steps

            # HPO heartbeat: print progress every 10K steps so user can see it's alive
            if self.hpo_mode and total_steps % 10000 < num_envs:
                elapsed = time.time() - t_start
                sps = total_steps / max(elapsed, 1e-6)
                logger.info(
                    f"[HPO] step {total_steps}/{self.total_timesteps} "
                    f"({100*total_steps/self.total_timesteps:.0f}%) | "
                    f"SPS={sps:.0f} | buf={len(self.agent.replay_buffer)}",
                )
                # WandB heartbeat so fleet monitor doesn't flag as stalled
                try:
                    wandb.log({
                        "hpo/heartbeat_step": total_steps,
                        "hpo/heartbeat_sps": round(sps, 1),
                    }, commit=True)
                except Exception as e:
                    if not getattr(self, '_heartbeat_warn_logged', False):
                        logger.warning(f"WandB heartbeat failed (will not repeat): {e}")
                        self._heartbeat_warn_logged = True

            # Logging
            if total_steps % self.log_interval < num_envs and not self.hpo_mode:
                elapsed = time.time() - t_start
                sps = total_steps / max(elapsed, 1e-6)

                log_data = {
                    "train/total_steps": total_steps,
                    "train/sps": sps,
                    "train/buffer_size": len(self.agent.replay_buffer),
                }

                if metrics:
                    for k, v in metrics.items():
                        log_data[f"train/{k}"] = v

                if self.episode_rewards:
                    log_data["train/episode_reward_mean"] = np.mean(self.episode_rewards)
                    log_data["train/episode_length_mean"] = np.mean(self.episode_lengths)

                # Extract portfolio value if available
                pv = infos.get("portfolio_value")
                if pv is not None:
                    if hasattr(pv, "__len__"):
                        log_data["train/portfolio_value"] = float(pv[0])
                    else:
                        log_data["train/portfolio_value"] = float(pv)

                # Fee level
                log_data["train/taker_fee"] = self._get_current_fee(total_steps)

                wandb.log(log_data, step=total_steps)

            # Checkpointing — drain training future first to avoid saving mid-update
            if total_steps % self.checkpoint_interval < num_envs:
                if train_future is not None:
                    m = train_future.result()
                    if m is not None:
                        self._assert_finite_metrics(m, total_steps)
                        metrics = m
                    train_future = None
                ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_step_{total_steps}.pth")
                self.agent.save(ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

        # OPT-08: Drain last training future and flush deferred store
        if train_future is not None:
            train_future.result()
        if _deferred_store is not None:
            d_obs, d_act, d_rew, d_nobs, d_dones, d_vmask, d_rc = _deferred_store
            if np.all(d_vmask):
                self.agent.store_batch(d_obs, d_act, d_rew, d_nobs, d_dones, regime_codes=d_rc)
            elif np.any(d_vmask):
                v_obs = {k: v[d_vmask] for k, v in d_obs.items()}
                v_next = {k: v[d_vmask] for k, v in d_nobs.items()}
                v_rc = d_rc[d_vmask] if d_rc is not None else None
                self.agent.store_batch(
                    v_obs, d_act[d_vmask], d_rew[d_vmask],
                    v_next, d_dones[d_vmask],
                    regime_codes=v_rc,
                )
        train_executor.shutdown(wait=False)

        # Final checkpoint
        final_path = os.path.join(self.ckpt_dir, "checkpoint_final.pth")
        self.agent.save(final_path)
        logger.info(f"Training complete. Final checkpoint: {final_path}")

        return final_path

    def _apply_fee_schedule(self, step: int):
        """Apply fee curriculum based on current step.

        Finds the active tier and re-evaluates ramp interpolation every call
        so the fee updates continuously during linear ramps.
        """
        if not self._fee_schedule:
            return

        # Find the highest-index tier whose start step has been reached
        active_tier = None
        active_idx = -1
        for i, tier in enumerate(self._fee_schedule):
            if step >= tier["step"]:
                active_tier = tier
                active_idx = i

        if active_tier is None:
            return

        # Compute current fee (with linear ramp if configured)
        ramp_to = active_tier.get("ramp_to")
        ramp_end = active_tier.get("ramp_end_step")

        if ramp_to is not None and ramp_end is not None:
            if step >= ramp_end:
                current_fee = ramp_to
            else:
                progress = (step - active_tier["step"]) / max(ramp_end - active_tier["step"], 1)
                current_fee = active_tier["taker_fee"] + progress * (ramp_to - active_tier["taker_fee"])
        else:
            current_fee = active_tier["taker_fee"]

        # Apply to environment
        try:
            self.env.call("set_fees", current_fee)
        except (AttributeError, TypeError):
            if hasattr(self.env, 'set_fees'):
                self.env.set_fees(current_fee)

        # Log tier transitions
        if active_idx > self._fee_tier_applied:
            self._fee_tier_applied = active_idx
            if not self.hpo_mode:
                logger.info(f"Fee curriculum: step={step}, tier={active_idx}, taker_fee={current_fee:.6f}")

    def _get_current_fee(self, step: int) -> float:
        """Get current fee level for logging."""
        if not self._fee_schedule:
            return self.config.get("env", {}).get("taker_fee", 0.0)

        current_fee = self._fee_schedule[0].get("taker_fee", 0.0)
        for tier in self._fee_schedule:
            if step >= tier["step"]:
                ramp_to = tier.get("ramp_to")
                ramp_end = tier.get("ramp_end_step")
                if ramp_to is not None and ramp_end is not None:
                    if step >= ramp_end:
                        current_fee = ramp_to
                    else:
                        progress = (step - tier["step"]) / max(ramp_end - tier["step"], 1)
                        current_fee = tier["taker_fee"] + progress * (ramp_to - tier["taker_fee"])
                else:
                    current_fee = tier["taker_fee"]
        return current_fee
