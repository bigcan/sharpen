"""On-policy trainer for continuous PPO on the V7 ContinuousSwingEnv.

Plumbs observations from the vectorized V7 env EXACTLY like ``SACTrainer``:
the env emits ``{scale_0..scale_{N-1}: (B, W, F), private: (B, private_dim)}``;
scales are stacked into ``(B, N, W, F)`` for the encoder.

Loop:  for each rollout -> collect ``rollout_steps`` steps -> GAE -> PPO update.
"""
import logging
import os
import time
from collections import deque

import numpy as np
import torch

from sharpen.agents.ppo_continuous.ppo_continuous_agent import PPOContinuousAgent
from sharpen.agents.ppo_continuous.rollout_buffer import ContinuousRolloutBuffer

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class PPOContinuousTrainer:
    def __init__(self, env, config, device="cpu", run_name=None, hpo_mode=False):
        self.env = env
        self.config = config
        self.device = device
        self.hpo_mode = hpo_mode
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")

        features_cfg = config.get("features", {})
        scales = features_cfg.get("scales", config.get("env", {}).get("scales", [15, 60, 240]))
        self._n_scales = len(scales)

        net_cfg = dict(config["network"])
        # Derive encoder geometry (mirror SACTrainer's defaulting).
        scale_enc = dict(net_cfg.get("scale_encoder", {}))
        if "input_size" not in scale_enc:
            scale_enc["input_size"] = features_cfg.get("features_per_scale", 8)
        net_cfg["scale_encoder"] = scale_enc
        net_cfg["n_scales"] = self._n_scales

        self._window_size = config.get("env", {}).get("window_size", net_cfg.get("window_size", 30))
        self._features_per_scale = scale_enc.get("input_size", 8)
        self._private_dim = net_cfg.get("private_dim", 5)

        ppo_cfg = config.get("agents", {}).get("ppoc", config.get("agents", {}).get("ppo_continuous", {}))

        self.total_timesteps = config["training"]["total_timesteps"]
        self.log_interval = config["training"].get("log_interval", 1000)
        self.rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        self.checkpoint_interval = ppo_cfg.get("checkpoint_interval", 250000)
        self.num_envs = getattr(env, "num_envs", 1)

        self.agent = PPOContinuousAgent(
            network_config=net_cfg,
            lr=ppo_cfg.get("learning_rate", ppo_cfg.get("lr", 3e-4)),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_eps=ppo_cfg.get("clip_eps", 0.2),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            ent_coef=ppo_cfg.get("ent_coef", 0.0),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            rollout_steps=self.rollout_steps,
            batch_size=ppo_cfg.get("batch_size", 256),
            lr_schedule=ppo_cfg.get("lr_schedule", "linear"),
            total_timesteps=self.total_timesteps,
            use_amp=config["training"].get("use_amp", False),
            amp_dtype=config["training"].get("amp_dtype", "bfloat16"),
            clip_value_loss=ppo_cfg.get("clip_value_loss", True),
            target_kl=ppo_cfg.get("target_kl", None),
            torch_compile=config["training"].get("torch_compile", False),
            device=device,
        )

        self.buffer = ContinuousRolloutBuffer(
            rollout_steps=self.rollout_steps,
            num_envs=self.num_envs,
            n_scales=self._n_scales,
            window_size=self._window_size,
            features_per_scale=self._features_per_scale,
            private_dim=self._private_dim,
            action_dim=self.agent.action_dim,
        )

        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def _stack_obs(self, obs):
        """Vector-env obs dict -> (scale_stack tensor (B,N,W,F), private tensor (B,pd))."""
        scale_np = np.stack(
            [obs[f"scale_{i}"] for i in range(self._n_scales)], axis=1,
        )
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).to(self.device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).to(self.device, non_blocking=True)
        return scale_stack, priv, scale_np

    def train(self, optuna_trial=None, pruning_callback=None) -> str:
        B = self.num_envs
        logger.info(
            f"PPO-continuous training | device={self.device} | envs={B} | "
            f"rollout={self.rollout_steps} | total_steps={self.total_timesteps}",
        )

        obs, _info = self.env.reset()
        # Shape sanity
        assert obs["scale_0"].shape == (B, self._window_size, self._features_per_scale), (
            f"scale_0 shape {obs['scale_0'].shape} != "
            f"({B}, {self._window_size}, {self._features_per_scale})"
        )
        assert obs["private"].shape == (B, self._private_dim), (
            f"private shape {obs['private'].shape} != ({B}, {self._private_dim})"
        )

        global_step = 0
        start_time = time.time()
        curr_rewards = np.zeros(B)
        curr_lens = np.zeros(B)
        last_ckpt = 0

        while global_step < self.total_timesteps:
            self.buffer.reset()
            for _t in range(self.rollout_steps):
                scale_stack, priv, scale_np = self._stack_obs(obs)
                actions, log_probs, values = self.agent.act_rollout(scale_stack, priv)

                next_obs, rewards, term, trunc, infos = self.env.step(actions)
                if np.isnan(rewards).any():
                    rewards = np.nan_to_num(rewards, nan=0.0)

                self.buffer.store(
                    scale_stack=scale_np,
                    private=np.asarray(obs["private"], dtype=np.float32),
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values,
                    dones=term.astype(np.float32),
                )

                dones_for_reset = np.logical_or(term, trunc)
                for i in range(B):
                    curr_rewards[i] += rewards[i]
                    curr_lens[i] += 1
                    if dones_for_reset[i]:
                        self.episode_rewards.append(curr_rewards[i])
                        self.episode_lengths.append(curr_lens[i])
                        curr_rewards[i] = 0
                        curr_lens[i] = 0

                obs = next_obs
                global_step += B

            # GAE bootstrap with the value of the post-rollout obs.
            scale_stack, priv, _ = self._stack_obs(obs)
            last_values = self.agent.value_only(scale_stack, priv)
            self.buffer.compute_gae(
                gamma=self.agent.gamma,
                gae_lambda=self.agent.gae_lambda,
                last_values=last_values,
                last_dones=term.astype(np.float32),
            )

            self.agent.step_count = global_step
            metrics = self.agent.train_step(self.buffer)

            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1e-6)
            rmean = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            logger.info(
                f"step={global_step}/{self.total_timesteps} sps={sps:.0f} "
                f"reward={rmean:.4f} ploss={metrics['policy_loss']:.4f} "
                f"vloss={metrics['value_loss']:.4f} ent={metrics['entropy']:.3f} "
                f"kl={metrics['approx_kl']:.4f} clip={metrics['clip_fraction']:.3f}",
            )
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "step": global_step,
                    "train/sps": sps,
                    "train/reward_mean": rmean,
                    **{f"agent/{k}": v for k, v in metrics.items()},
                })

            if optuna_trial is not None and pruning_callback is not None:
                score = pruning_callback()
                optuna_trial.report(score, global_step)

            if global_step - last_ckpt >= self.checkpoint_interval:
                self.save_checkpoint(f"checkpoint_step_{global_step}.pth")
                last_ckpt = global_step

        self.save_checkpoint("checkpoint_final.pth")
        logger.info("PPO-continuous training complete.")
        return os.path.join(self.ckpt_dir, "checkpoint_final.pth")

    def save_checkpoint(self, filename):
        path = os.path.join(self.ckpt_dir, filename)
        self.agent.save(path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path, strict=True):
        self.agent.load(path)
