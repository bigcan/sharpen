"""Continuous PPO agent for the V7 ContinuousSwingEnv.

Interface:
  * ``predict(scale_input, private, deterministic, lob)`` -> action tensor
    (B, action_dim). SAC-COMPATIBLE so the existing backtest loop can drive it
    with zero changes (same obs path, same return shape as ``SACAgent.predict``).
  * ``act_rollout(scale_stack, private)`` -> (action, log_prob, value) numpy
    arrays for on-policy rollout collection.
  * ``train_step(buffer)`` -> dict of PPO update metrics (clipped surrogate).
  * ``save`` / ``load`` -> portable checkpoints (torch.compile prefix stripped).
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sharpen.agents.ppo_continuous.networks import PPOContinuousActorCritic
from sharpen.agents.ppo_continuous.rollout_buffer import ContinuousRolloutBuffer

logger = logging.getLogger(__name__)


class PPOContinuousAgent:
    def __init__(
        self,
        network_config: dict,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        rollout_steps: int = 2048,
        batch_size: int = 256,
        lr_schedule: str = "linear",
        total_timesteps: int = 1_000_000,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        clip_value_loss: bool = True,
        target_kl: Optional[float] = None,
        torch_compile: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.lr = lr
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self._use_bf16 = (self.amp_dtype == torch.bfloat16)
        self.clip_value_loss = clip_value_loss
        self.target_kl = target_kl
        self.step_count = 0

        # ---- Parse network_config IDENTICALLY to SACAgent.__init__ ----
        scale_cfg = dict(network_config.get("scale_encoder", {
            "input_size": 7, "channels": [32, 64, 64, 64],
            "kernel_size": 3, "output_dim": 64,
        }))
        if isinstance(scale_cfg.get("channels"), list):
            scale_cfg["channels"] = tuple(scale_cfg["channels"])
        private_dim = network_config.get("private_dim", 5)
        fusion_dim = network_config.get("fusion_dim", 256)
        self._obs_mode = network_config.get("obs_mode", "window")
        if self._obs_mode == "summary_stats":
            scale_cfg["summary_input_dim"] = network_config.get("summary_input_dim", 50)
        self._window_size = network_config.get("window_size", 30)
        self._features_per_scale = scale_cfg.get("input_size", 7)
        self._n_scales = network_config.get("n_scales", 3)
        self._private_dim = private_dim
        self.action_dim = network_config.get("action_dim", 1)

        lob_cfg = network_config.get("lob_encoder")
        if lob_cfg is not None and isinstance(lob_cfg.get("channels"), list):
            lob_cfg["channels"] = tuple(lob_cfg["channels"])

        self.network = PPOContinuousActorCritic(
            scale_cfg, private_dim, fusion_dim, self._n_scales,
            action_dim=self.action_dim, obs_mode=self._obs_mode,
            lob_encoder_config=lob_cfg,
        ).to(self.device)

        self._torch_compiled = False
        if torch_compile and self.device.type == "cuda":
            try:
                self.network = torch.compile(self.network, mode="default")
                self._torch_compiled = True
                logger.info("[torch.compile] PPO-continuous network compiled (mode=default)")
            except Exception as e:
                logger.warning(f"[torch.compile] failed, eager mode: {e}")

        _fused = self.device.type == "cuda"
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5, fused=_fused)
        # AMP scaler only needed for fp16; bf16 has fp32 dynamic range.
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=use_amp and not self._use_bf16 and self.device.type == "cuda",
        )

        self._total_timesteps_for_lr = max(total_timesteps, 1)
        self._base_lr = lr
        self._lr_schedule_type = lr_schedule

    # ------------------------------------------------------------------
    def _net(self):
        """Return the underlying module (unwrap torch.compile)."""
        return getattr(self.network, "_orig_mod", self.network)

    def _update_lr(self):
        if self._lr_schedule_type == "linear":
            frac = max(1.0 - self.step_count / self._total_timesteps_for_lr, 0.0)
            new_lr = self._base_lr * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

    # ------------------------------------------------------------------
    # SAC-compatible predict() — used by the shared backtest loop.
    # ------------------------------------------------------------------
    def predict(
        self,
        scale_input,
        private: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        lob: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return action tensor (B, action_dim) in [-1, 1] (matches SACAgent)."""
        if self._obs_mode == "summary_stats":
            scale_stack = scale_input
        elif isinstance(scale_input, list):
            scale_stack = torch.stack(scale_input, dim=1)
        else:
            scale_stack = scale_input
        net = self._net()
        if deterministic:
            net.actor.eval()
        with torch.no_grad():
            action, _ = net.actor.sample(
                scale_stack, private, deterministic=deterministic, lob=lob,
            )
        if deterministic:
            net.actor.train()
        return action

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def act_rollout(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stochastic action for rollout: (action, log_prob, value) as numpy."""
        net = self._net()
        with torch.no_grad():
            action, log_prob, value = net.act(
                scale_stack, private, deterministic=False, lob=lob,
            )
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def value_only(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        net = self._net()
        with torch.no_grad():
            v = net.value_only(scale_stack, private, lob=lob)
        return v.cpu().numpy()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def train_step(self, buffer: ContinuousRolloutBuffer) -> dict[str, float]:
        net = self._net()
        net.train()

        tot_pol = tot_val = tot_ent = tot_loss = tot_kl = tot_clip = 0.0
        n_updates = 0
        kl_stop = False

        for _epoch in range(self.n_epochs):
            for batch in buffer.iterate_minibatches(self.batch_size):
                scales = torch.as_tensor(batch["scales"], dtype=torch.float32, device=self.device)
                private = torch.as_tensor(batch["private"], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
                old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
                old_values = torch.as_tensor(batch["old_values"], dtype=torch.float32, device=self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    new_log_probs, new_values, entropy = net.evaluate_actions(
                        scales, private, actions,
                    )
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    ratio = torch.clamp(ratio, 1e-4, 100.0)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    if self.clip_value_loss:
                        v_clipped = old_values + torch.clamp(
                            new_values - old_values, -self.clip_eps, self.clip_eps,
                        )
                        vl_unclipped = (new_values - returns) ** 2
                        vl_clipped = (v_clipped - returns) ** 2
                        value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                    else:
                        value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                    entropy_loss = -entropy.mean()
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = (torch.abs(ratio - 1) > self.clip_eps).float().mean().item()

                tot_pol += policy_loss.item()
                tot_val += value_loss.item()
                tot_ent += entropy_loss.item()
                tot_loss += loss.item()
                tot_kl += approx_kl
                tot_clip += clip_fraction
                n_updates += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_stop = True
                    break
            if kl_stop:
                break

        current_lr = self.optimizer.param_groups[0]["lr"]
        self._update_lr()

        n = max(n_updates, 1)
        return {
            "loss_total": tot_loss / n,
            "policy_loss": tot_pol / n,
            "value_loss": tot_val / n,
            "entropy_loss": tot_ent / n,
            "entropy": -tot_ent / n,
            "approx_kl": tot_kl / n,
            "clip_fraction": tot_clip / n,
            "learning_rate": current_lr,
            "kl_early_stop": float(kl_stop),
        }

    # ------------------------------------------------------------------
    # Pipeline-compat no-ops (mirror SACAgent/PPOAgent interface)
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        pass

    def reset_hidden_state(self):
        pass

    def mask_hidden_state(self, dones):
        pass

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_compile_prefix(state_dict):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save(self, path: str):
        ckpt = {
            "network": self._strip_compile_prefix(self.network.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }
        if self.scaler.is_enabled():
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        net_sd = self._strip_compile_prefix(ckpt["network"])
        try:
            self._net().load_state_dict(net_sd)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                logger.warning(f"Checkpoint architecture mismatch: {e}. Fresh weights.")
                return
            raise
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.warning("Optimizer state mismatch — skipping optimizer restore.")
        self.step_count = ckpt.get("step_count", 0)
        self._update_lr()
        if "scaler" in ckpt and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
