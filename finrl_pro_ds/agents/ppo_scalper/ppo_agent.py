"""
PPO Agent for DeepScalper.

On-policy agent using Proximal Policy Optimization with clipped surrogate objective.
Implements the same interface contract as DeepScalperBDQ for pipeline compatibility.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict, Tuple, Optional

from finrl_pro_ds.agents.ppo_scalper.networks import PPOActorCritic
from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer
import logging

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO Agent with Multi-Discrete action space (Price × SignedQty).

    Key differences from BDQ:
    - On-policy: no replay buffer, uses rollout buffer
    - No epsilon-greedy: entropy bonus drives exploration
    - Clipped surrogate objective for stable policy updates
    - GAE for advantage estimation
    """

    def __init__(
        self,
        network_config: Dict,
        lr: float = 3e-4,
        gamma: float = 0.95,  # T1.2: LOB signal decay alignment
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.001,  # T1.3: Don't drown trading signal
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        rollout_steps: int = 2048,
        batch_size: int = 256,
        action_dims: int = 6, # Tier 2: Discrete(6)
        lr_schedule: str = "linear",
        total_timesteps: int = 1_000_000,
        use_amp: bool = False,
        clip_value_loss: bool = True,
        target_kl: Optional[float] = None,
        torch_compile: bool = False,    # PERF FIX-1: torch.compile for GPU kernel fusion
        device: str = "cpu",
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
        self.action_dims = action_dims
        self.lr_schedule = lr_schedule
        self.total_timesteps = total_timesteps
        self.use_amp = use_amp
        self.clip_value_loss = clip_value_loss  # AUDIT FIX FLAG-3: Toggleable value clipping
        self.target_kl = target_kl  # AUDIT FIX FLAG-4: KL early-stopping threshold
        self.lr = lr

        # Step counter for LR schedule
        self.step_count = 0

        # AUDIT FIX D1: Validate window_size consistency between env and network
        env_ws = network_config.get("micro_config", {}).get("window_size")
        if env_ws is not None:
            # This will be validated against env.window_size by the trainer;
            # here we just log it for debugging.
            logger.info(f"PPOAgent micro_config.window_size = {env_ws}")

        # Network
        net_cfg = dict(network_config)
        net_cfg["action_space_dims"] = action_dims
        self.network = PPOActorCritic(**net_cfg).to(self.device)

        # PERF FIX-1: torch.compile for GPU kernel fusion
        # R&D log: reduce-overhead crashes PPO — use mode="default"
        self._torch_compiled = False
        if torch_compile and self.device.type == "cuda":
            try:
                self.network = torch.compile(self.network, mode="default")
                self._torch_compiled = True
                print("[torch.compile] PPO network compiled (mode=default)")
            except Exception as e:
                print(f"[torch.compile] Failed, falling back to eager mode: {e}")

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # AMP
        self.scaler = torch.amp.GradScaler(device=str(self.device), enabled=self.use_amp)

        # LR scheduler (linear decay based on global environment steps)
        # FIX FIND-PPO-02: LambdaLR closure captured live self.step_count by reference,
        # causing non-monotonic LR decay. Replaced with explicit manual LR update
        # called from train_step() that reads step_count at the correct time.
        self._total_timesteps_for_lr = max(total_timesteps, 1)
        self._base_lr = lr
        self._lr_schedule_type = lr_schedule
        # No LambdaLR — LR is updated explicitly in _update_lr()
        self._lr_scheduler = None

        # V5: Stateless agent — no hidden state needed
        # (reset_hidden_state / mask_hidden_state kept as no-ops for trainer compat)

    def _update_lr(self):
        """FIX FIND-PPO-02: Explicit linear LR decay based on current step_count.
        Called once per rollout from train_step(), AFTER step_count is set."""
        if self._lr_schedule_type == "linear":
            frac = max(1.0 - self.step_count / self._total_timesteps_for_lr, 0.0)
            new_lr = self._base_lr * frac
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    # ------------------------------------------------------------------
    # Interface: predict()
    # ------------------------------------------------------------------
    def predict(
        self,
        micro: torch.Tensor,
        private_in: torch.Tensor,
        macro: torch.Tensor,
        deterministic: bool = False,
        qty_mask=None,
        context: Optional[Dict] = None,
        eval_epsilon: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions from the current policy.

        Args:
            micro: (B, W, 30) micro features
            private_in: (B, W, 5) private state
            macro: (B, M) macro features
            deterministic: If True, take argmax actions
            qty_mask: Optional (B, 6) mask

        Returns:
            actions: (B,) numpy int64
            log_probs: (B,) numpy float32
            values: (B,) numpy float32
        """
        # PERF FIX-4: non_blocking H2D transfers
        micro = micro.to(self.device, non_blocking=True)
        private_in = private_in.to(self.device, non_blocking=True)
        macro = macro.to(self.device, non_blocking=True)

        # Prepare qty mask tensor
        qty_mask_t = None
        if qty_mask is not None:
            qty_mask_t = torch.as_tensor(qty_mask, dtype=torch.float32, device=self.device)
            if qty_mask_t.dim() == 1:
                qty_mask_t = qty_mask_t.unsqueeze(0).expand(micro.shape[0], -1)

        self.network.eval()
        with torch.no_grad():
            actions, log_probs, values, _ = self.network(
                micro, private_in, macro,
                qty_mask=qty_mask_t,
                deterministic=deterministic,
            )

        self.network.train()

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Interface: train_step()
    # ------------------------------------------------------------------
    def train_step(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run K epochs of PPO minibatch updates on the rollout buffer.

        Args:
            rollout_buffer: Full rollout buffer with computed GAE advantages

        Returns:
            Dict of training metrics
        """
        self.network.train()

        # Accumulators for metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_acc = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            epoch_kl_exceeded = False
            for batch in rollout_buffer.iterate_minibatches(self.batch_size):
                # Convert to torch
                micro = torch.as_tensor(batch["micro"], dtype=torch.float32, device=self.device)
                private = torch.as_tensor(batch["private"], dtype=torch.float32, device=self.device)
                macro = torch.as_tensor(batch["macro"], dtype=torch.float32, device=self.device)
                old_actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).squeeze(-1)
                old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
                old_values = torch.as_tensor(batch["old_values"], dtype=torch.float32, device=self.device)
                # B4 fix: Pass stored qty_masks so evaluate_actions uses same masking as rollout
                qty_mask_t = torch.as_tensor(batch["qty_masks"], dtype=torch.float32, device=self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    # Evaluate current policy on old actions (with original action mask)
                    new_log_probs, new_values, entropy = self.network.evaluate_actions(
                        micro, private, macro, old_actions, qty_mask=qty_mask_t
                    )

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    # FIX N-3: Clamp ratio to prevent NaN/Inf from stale log-probs
                    ratio = torch.clamp(ratio, 1e-4, 100.0)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss — AUDIT FIX FLAG-3: Toggleable value clipping
                    if self.clip_value_loss:
                        values_clipped = old_values + torch.clamp(
                            new_values - old_values, -self.clip_eps, self.clip_eps
                        )
                        value_loss_unclipped = (new_values - returns) ** 2
                        value_loss_clipped = (values_clipped - returns) ** 2
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                    # Entropy loss (negative because we want to maximize entropy)
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Gradient step
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = (torch.abs(ratio - 1) > self.clip_eps).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss_acc += loss.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                n_updates += 1

                # AUDIT FIX FLAG-4: KL early-stopping — break epoch if policy diverges too far
                if self.target_kl is not None and approx_kl > self.target_kl:
                    epoch_kl_exceeded = True
                    break
            if epoch_kl_exceeded:
                break

        # Capture LR used for THIS update
        current_lr = self.optimizer.param_groups[0]["lr"]

        # FIX FIND-PPO-02: Explicit LR update based on step_count (set by trainer)
        self._update_lr()

        n = max(n_updates, 1)
        metrics = {
            "loss_total": total_loss_acc / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy_loss": total_entropy_loss / n,
            "entropy": -total_entropy_loss / n,  # Positive entropy
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_fraction / n,
            "learning_rate": current_lr,
        }

        return metrics

    # ------------------------------------------------------------------
    # Interface: compatibility methods
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """No-op. PPO doesn't use epsilon-greedy."""
        pass

    def reset_hidden_state(self):
        """No-op. V5 stateless agent has no hidden state."""
        pass

    def mask_hidden_state(self, dones: np.ndarray):
        """No-op. V5 stateless agent has no hidden state."""
        pass

    # ------------------------------------------------------------------
    # Interface: save / load
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_compile_prefix(state_dict):
        """Strip '_orig_mod.' prefix added by torch.compile for portable checkpoints."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save(self, path: str):
        """Save checkpoint."""
        # PERF FIX-1: Strip torch.compile prefix for portable checkpoints
        ckpt = {
            "network": self._strip_compile_prefix(self.network.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }
        # FIX FIND-V3-04c: Persist AMP GradScaler state
        if hasattr(self, 'scaler') and self.use_amp:
            ckpt['scaler'] = self.scaler.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        """Load checkpoint.

        AUDIT FIX C4: Handles architecture mismatch (e.g. V4 LSTM → V5 MLP)
        gracefully instead of crashing with opaque state_dict error.
        """
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            # PERF FIX-1: Strip torch.compile prefix from checkpoint keys
            network_sd = self._strip_compile_prefix(checkpoint["network"])
            self.network.load_state_dict(network_sd)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                logger.warning(
                    f"Checkpoint architecture mismatch (encoder changed?): {e}"
                )
                logger.warning("Starting with fresh network weights.")
                return  # Skip optimizer/scheduler restore — fresh start
            raise
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
        # FIX FIND-PPO-02: Restore LR from step_count (no scheduler state needed)
        self._update_lr()
        # FIX FIND-V3-04c: Restore AMP GradScaler state
        if 'scaler' in checkpoint and hasattr(self, 'scaler') and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
