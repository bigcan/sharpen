"""
SAC Trainer — Training loop for continuous position control.

Handles:
  - Vectorized env data collection
  - SAC agent update cycle
  - Fee curriculum (linear ramp)
  - WandB logging
  - Checkpointing
"""
import os
import time
import numpy as np
import torch
import wandb
import logging
from typing import Dict
from collections import deque

from finrl_pro_ds.agents.sac.sac_agent import SACAgent

logger = logging.getLogger(__name__)


class SACTrainer:
    """Training loop for SAC agent with ContinuousSwingEnv."""

    def __init__(
        self,
        env,
        config: Dict,
        device: str = "cuda",
        run_name: str = None,
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

        # HPO: cap buffer_size to avoid wasting memory on short trials
        buffer_size = sac_cfg.get("buffer_size", 1_000_000)
        if hpo_mode:
            buffer_size = min(buffer_size, 100_000)

        self.agent = SACAgent(
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
        )

        # Training config
        self.total_timesteps = config["training"]["total_timesteps"]
        self.log_interval = config["training"].get("log_interval", 1000)
        self.checkpoint_interval = sac_cfg.get("checkpoint_interval", 500_000)
        self.learning_starts = sac_cfg.get("learning_starts", 10_000)
        # update_interval is gradient steps per env step per env.
        # Scale by num_envs so UTD ≈ update_interval regardless of parallelism.
        # FIX R2-AUD-02: Read num_envs from actual env, not config (config["training"]
        # may differ from pipeline's actual env count read from config["env"]).
        num_envs = getattr(env, 'num_envs', 1)
        raw_ui = sac_cfg.get("update_interval", 4)
        # FIX AUD-S129-03: HPO must use the SAME UTD as full training so that
        # hyperparameters (especially tau, lr) are tuned in the correct regime.
        # Previous cap to UTD=1 caused HPO to select HPs for 12 gradient steps
        # that then ran at 96 — a fundamentally different optimization landscape.
        self.update_interval = raw_ui * num_envs

        # Auto-scale tau for high UTD
        if self.update_interval > 1:
            raw_tau = self.agent.tau
            effective_tau = 1.0 - (1.0 - raw_tau) ** (1.0 / self.update_interval)
            self.agent.tau = effective_tau
            logger.info(
                f"[UTD] update_interval={self.update_interval} → tau auto-scaled: "
                f"{raw_tau:.6f} → {effective_tau:.6f}"
            )

        # Fee curriculum
        raw_schedule = config.get("env", {}).get("fee_schedule", [])
        self._fee_schedule = sorted(raw_schedule, key=lambda x: x["step"]) if raw_schedule else []
        self._fee_tier_applied = -1

        # Checkpoint dir
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def train(self, optuna_trial=None, pruning_callback=None) -> str:
        """Main training loop. Returns path to final checkpoint."""
        num_envs = getattr(self.env, 'num_envs', 1)
        obs, info = self.env.reset()

        episode_reward = np.zeros(num_envs, dtype=np.float64)
        episode_length = np.zeros(num_envs, dtype=np.int64)
        total_steps = 0
        t_start = time.time()
        gradient_accumulator = 0.0
        metrics = None  # FIX R2-AUD-01: Initialize before learning_starts to prevent NameError

        # Limit PyTorch intra-op threads for SyncVectorEnv.
        # Default num_threads = all CPU cores → massive context-switch overhead
        # for small-batch ops (predict with batch=num_envs).
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 4))

        logger.info(
            f"SAC Training: {self.total_timesteps} steps, "
            f"{num_envs} envs, device={self.device}, "
            f"torch_threads={torch.get_num_threads()}"
        )

        while total_steps < self.total_timesteps:
            # Diagnostic: one-time step-0 heartbeat to confirm loop entry
            if total_steps == 0:
                logger.info("[DIAG] Training loop entered, step 0")
                t_diag = time.time()

            # Apply fee curriculum
            self._apply_fee_schedule(total_steps)

            # Get actions — extract scale tensors dynamically
            scale_tensors = [
                torch.tensor(obs[f"scale_{i}"], dtype=torch.float32).to(self.device, non_blocking=True)
                for i in range(self._n_scales)
            ]
            priv = torch.tensor(obs["private"], dtype=torch.float32).to(self.device, non_blocking=True)

            with torch.no_grad():
                actions = self.agent.predict(scale_tensors, priv, deterministic=False)
                actions_np = actions.cpu().numpy()

            # Step environment
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_np)

            # Store transitions
            dones = np.logical_or(terms, truncs).astype(np.float32)
            self.agent.store_batch(obs, actions_np, rewards, next_obs, dones)

            # Diagnostic: first 100 steps timing
            if total_steps == 100 * num_envs:
                dt = time.time() - t_diag
                logger.info(f"[DIAG] First {total_steps} steps took {dt:.1f}s ({total_steps/dt:.0f} SPS)")

            # Track episodes
            episode_reward += rewards
            episode_length += 1

            for i in range(num_envs):
                if dones[i]:
                    self.episode_rewards.append(episode_reward[i])
                    self.episode_lengths.append(episode_length[i])
                    episode_reward[i] = 0.0
                    episode_length[i] = 0

            obs = next_obs
            total_steps += num_envs
            self.agent.step_count = total_steps

            # Training updates
            if total_steps >= self.learning_starts:
                gradient_accumulator += self.update_interval
                while gradient_accumulator >= 1.0:
                    m = self.agent.train_step()
                    if m is not None:
                        metrics = m
                    gradient_accumulator -= 1.0

            # HPO heartbeat: print progress every 10K steps so user can see it's alive
            if self.hpo_mode and total_steps % 10000 < num_envs:
                elapsed = time.time() - t_start
                sps = total_steps / max(elapsed, 1e-6)
                logger.info(
                    f"[HPO] step {total_steps}/{self.total_timesteps} "
                    f"({100*total_steps/self.total_timesteps:.0f}%) | "
                    f"SPS={sps:.0f} | buf={len(self.agent.replay_buffer)}"
                )

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

            # Checkpointing
            if total_steps % self.checkpoint_interval < num_envs:
                ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_{total_steps}.pth")
                self.agent.save(ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

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
