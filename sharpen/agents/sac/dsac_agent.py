"""Distributional SAC (QR-SAC) Agent with CVaR Optimization.

Subclasses SACAgent to replace scalar Q-critics with quantile-conditional
critics. Optimizes CVaR (worst-alpha% of return distribution) instead of
mean return, producing regime-robust policies.

Algorithm:
  Critic: Quantile regression loss over N=32 quantile fractions
  Actor: CVaR-restricted policy gradient (tau sampled from [0, cvar_alpha])
  Target: Twin-critic min over quantile vectors with entropy penalty
"""
import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from sharpen.agents.sac.distributional_networks import DistributionalSACCriticNetwork
from sharpen.agents.sac.sac_agent import SACAgent

logger = logging.getLogger(__name__)


class DistributionalSACAgent(SACAgent):
    """QR-SAC agent with CVaR for tail-risk-aware trading.

    Subclasses SACAgent, overriding:
    - __init__: replaces scalar critics with DistributionalSACCriticNetwork
    - train_step_mega: distributional critic loss + CVaR actor loss
    - save/load: persists distributional config

    Everything else (predict, store_transition, store_batch, replay buffer,
    _obs_to_buffer, _unpack_buffer_to_stacks) is inherited unchanged.
    """

    def __init__(
        self,
        network_config: dict,
        n_quantiles: int = 32,
        cvar_alpha: float = 0.25,
        quantile_embed_dim: int = 64,
        kappa: float = 1.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        initial_alpha: float = 0.2,
        learning_starts: int = 10_000,
        update_interval: int = 4,
        gradient_clip: float = 10.0,
        use_amp: bool = False,
        amp_dtype: str = "float16",
        torch_compile: bool = False,
        checkpoint_interval: int = 500_000,
        device: str = "cpu",
        **kwargs,
    ):
        # CrossQ is a scalar-critic algorithm: its joint BatchRenorm pass is
        # implemented in SACAgent.train_step_mega, which DSAC overrides wholesale
        # with the quantile path. Enabling both would build BatchRenorm critics
        # that the DSAC update never uses, and silently train plain DSAC.
        if isinstance(kwargs.get("crossq"), dict):
            _crossq_on = bool(kwargs["crossq"].get("enabled", False))
        else:
            _crossq_on = bool(kwargs.get("crossq", False))
        if _crossq_on:
            raise NotImplementedError(
                "crossq is not supported by DistributionalSACAgent (agents.sac.distributional). "
                "CrossQ + quantile critics is an unimplemented combination, not a config error - "
                "set one or the other.",
            )

        # Store distributional params before super().__init__ builds scalar critics
        self._n_quantiles = n_quantiles
        self._cvar_alpha = cvar_alpha
        self._quantile_embed_dim = quantile_embed_dim
        self.kappa = kappa
        self._requested_torch_compile = torch_compile

        # Build everything via parent (actor, scalar critics, buffer, optimizers)
        # Pass torch_compile=False to prevent compiling scalar critics we'll discard
        super().__init__(
            network_config=network_config,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            initial_alpha=initial_alpha,
            learning_starts=learning_starts,
            update_interval=update_interval,
            gradient_clip=gradient_clip,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            torch_compile=False,  # defer — we'll compile new critics below
            checkpoint_interval=checkpoint_interval,
            device=device,
            **kwargs,
        )

        # Extract encoder config (same logic as parent __init__)
        scale_cfg = network_config.get("scale_encoder", {
            "input_size": 7, "channels": [32, 64, 64, 64],
            "kernel_size": 3, "output_dim": 64,
        })
        if isinstance(scale_cfg.get("channels"), list):
            scale_cfg["channels"] = tuple(scale_cfg["channels"])
        private_dim = network_config.get("private_dim", 5)
        fusion_dim = network_config.get("fusion_dim", 256)
        lob_cfg = network_config.get("lob_encoder")
        if lob_cfg is not None and isinstance(lob_cfg.get("channels"), list):
            lob_cfg["channels"] = tuple(lob_cfg["channels"])

        # Replace scalar critics with distributional critics
        self.critic1 = DistributionalSACCriticNetwork(
            scale_cfg, private_dim, fusion_dim,
            action_dim=self._action_dim,
            n_scales=self._n_scales,
            n_quantiles=n_quantiles,
            quantile_embed_dim=quantile_embed_dim,
            obs_mode=self._obs_mode,
            lob_encoder_config=lob_cfg,
        ).to(self.device, non_blocking=True)
        self.critic2 = DistributionalSACCriticNetwork(
            scale_cfg, private_dim, fusion_dim,
            action_dim=self._action_dim,
            n_scales=self._n_scales,
            n_quantiles=n_quantiles,
            quantile_embed_dim=quantile_embed_dim,
            obs_mode=self._obs_mode,
            lob_encoder_config=lob_cfg,
        ).to(self.device, non_blocking=True)

        # Replace targets
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device, non_blocking=True)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device, non_blocking=True)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Rebuild critic optimizer (parent's optimizer references discarded scalar critics)
        _fused = self.device.type == "cuda"
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic, fused=_fused,
        )

        # Apply torch.compile on new distributional targets + actor
        if self._requested_torch_compile and hasattr(torch, 'compile'):
            try:
                import torch._inductor.config as _inductor_cfg
                _inductor_cfg.max_autotune_gemm = False
                self.actor = torch.compile(self.actor, mode='default')
                self.target_critic1 = torch.compile(self.target_critic1, mode='default')
                self.target_critic2 = torch.compile(self.target_critic2, mode='default')
                logger.info(
                    "[torch.compile] DSAC: actor + 2 distributional targets compiled. "
                    "Training critics: eager (O-A encode/q_head split). "
                    "max_autotune_gemm=False (BUG-08 workaround)",
                )
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Cache for metrics on non-actor-update steps (BUG-DSAC-02 fix)
        self._last_q_cvar = None

        logger.info(
            f"[DSAC] Distributional SAC initialized: n_quantiles={n_quantiles}, "
            f"cvar_alpha={cvar_alpha}, kappa={kappa}, embed_dim={quantile_embed_dim}"
        )

    def _compute_quantile_loss(
        self,
        q_tau: torch.Tensor,
        T_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Quantile Huber loss between predicted and target quantiles.

        Adapted from IQNAgent._compute_quantile_loss (iqn_agent.py:336-359).

        For each pair (i, j) of predicted quantile q_tau_i and target T_tau_j:
          delta_ij = T_tau_j - q_tau_i
          huber_ij = |delta|<=kappa ? 0.5*delta^2/kappa : |delta|-0.5*kappa
          weight_ij = |tau_i - 1[delta<0]|
          loss = mean over i,j of (weight * huber)

        Args:
            q_tau: (B, N) predicted quantile values
            T_tau: (B, N') target quantile values
            tau: (B, N) quantile fractions for predictions

        Returns:
            (B,) per-sample quantile regression loss
        """
        # (B, N, 1) - (B, 1, N') -> (B, N, N')
        delta = T_tau.unsqueeze(1) - q_tau.unsqueeze(2)
        abs_delta = delta.abs()
        huber = torch.where(
            abs_delta <= self.kappa,
            0.5 * delta.pow(2) / self.kappa,
            abs_delta - 0.5 * self.kappa,
        )
        # Asymmetric weight: |tau - 1[delta < 0]|
        tau_expanded = tau.unsqueeze(2)  # (B, N, 1)
        weight = (tau_expanded - (delta < 0).float()).abs()
        return (weight * huber).mean(dim=1).mean(dim=1)  # (B,)

    def train_step_mega(self, n_steps: int = 1) -> Optional[dict[str, float]]:
        """Run n_steps QR-SAC gradient updates from a single mega-batch.

        Overrides SACAgent.train_step_mega with:
        - Distributional critic loss (quantile regression)
        - CVaR-aware actor loss (tau sampled from [0, cvar_alpha])
        """
        if len(self.replay_buffer) < self.learning_starts:
            return None

        mega_batch_size = n_steps * self.batch_size

        # --- ONE CPU phase: sample + transfer (identical to parent) ---
        if self.regime_balanced_replay:
            states, actions_np, rewards_np, next_states, dones_np, _ = \
                self.replay_buffer.sample_regime_balanced(mega_batch_size, mode=self.regime_balanced_replay)
        else:
            states, actions_np, rewards_np, next_states, dones_np, _ = \
                self.replay_buffer.sample(mega_batch_size)

        unpacked = self._unpack_buffer_to_stacks(states, next_states)
        all_scale_stack = unpacked.scale_stack
        all_next_scale_stack = unpacked.next_scale_stack
        all_lob = unpacked.lob
        all_next_lob = unpacked.next_lob
        if self._obs_mode == "summary_stats":
            all_priv = None
            all_npriv = None
        else:
            all_priv = self._to_device_pinned(states["private"])
            all_npriv = self._to_device_pinned(next_states["private"])
        all_actions = self._to_device_pinned(actions_np)
        all_rewards = self._to_device_pinned(rewards_np).unsqueeze(1)
        all_dones = self._to_device_pinned(dones_np).unsqueeze(1)

        amp_ctx = torch.amp.autocast(
            'cuda', dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == 'cuda',
        )

        metrics = None
        bs = self.batch_size
        N = self._n_quantiles

        # --- ONE GPU phase: n_steps gradient steps ---
        for step_i in range(n_steps):
            self._train_step_count += 1
            start = step_i * bs
            end = start + bs

            # Slice mini-batch (views, zero-copy)
            scale_stack = all_scale_stack[start:end]
            next_scale_stack = all_next_scale_stack[start:end]
            priv = all_priv[start:end] if all_priv is not None else None
            npriv = all_npriv[start:end] if all_npriv is not None else None
            lob_mb = all_lob[start:end] if all_lob is not None else None
            next_lob_mb = all_next_lob[start:end] if all_next_lob is not None else None
            actions = all_actions[start:end]
            rewards = all_rewards[start:end]  # (B, 1)
            dones = all_dones[start:end]      # (B, 1)

            alpha = self.log_alpha.exp().detach()

            # === CRITIC UPDATE (Distributional) ===
            with torch.no_grad():
                with amp_ctx:
                    next_action, next_log_prob = self.actor.sample(
                        next_scale_stack, npriv, lob=next_lob_mb,
                    )
                    # Sample target quantile fractions
                    target_tau = torch.rand(bs, N, device=self.device)

                    # Target quantile values from twin critics
                    tgt_feat1 = self.target_critic1.encode(next_scale_stack, npriv, lob=next_lob_mb)
                    tgt_feat2 = self.target_critic2.encode(next_scale_stack, npriv, lob=next_lob_mb)
                    tgt_q1, _ = self.target_critic1.q_head_forward(tgt_feat1, next_action, tau=target_tau)
                    tgt_q2, _ = self.target_critic2.q_head_forward(tgt_feat2, next_action, tau=target_tau)

                    # Min over twin critics per quantile, subtract entropy penalty
                    target_quantiles = torch.min(tgt_q1, tgt_q2) - alpha * next_log_prob  # (B, N)

                    # Bellman target: r + gamma * (1-d) * target_quantiles
                    # rewards/dones are (B,1) — broadcast to (B, N)
                    T_tau = rewards + (1.0 - dones) * self.gamma * target_quantiles  # (B, N)

            with amp_ctx:
                # Encode current state (O-A caching: reuse for actor update)
                c1_feat = self.critic1.encode(scale_stack, priv, lob=lob_mb)
                c2_feat = self.critic2.encode(scale_stack, priv, lob=lob_mb)

                # Sample independent tau for each critic
                tau1 = torch.rand(bs, N, device=self.device)
                tau2 = torch.rand(bs, N, device=self.device)

                q1_quantiles, _ = self.critic1.q_head_forward(c1_feat, actions, tau=tau1)  # (B, N)
                q2_quantiles, _ = self.critic2.q_head_forward(c2_feat, actions, tau=tau2)  # (B, N)

                # Quantile regression loss
                critic_loss = (
                    self._compute_quantile_loss(q1_quantiles, T_tau, tau1).mean()
                    + self._compute_quantile_loss(q2_quantiles, T_tau, tau2).mean()
                )

            self.critic_optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_optimizer)
            else:
                critic_loss.backward()
            critic_norm = nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.gradient_clip,
            )
            if self.scaler.is_enabled():
                self.scaler.step(self.critic_optimizer)
            else:
                self._step_if_finite(self.critic_optimizer, critic_norm)   # NAN-01

            # === ACTOR UPDATE (CVaR-aware) + ALPHA UPDATE ===
            if self._train_step_count % self.actor_update_freq == 0:
                with amp_ctx:
                    new_action, log_prob = self.actor.sample(scale_stack, priv, lob=lob_mb)

                    # CVaR trick: sample tau from [0, cvar_alpha] to focus on left tail
                    cvar_tau = torch.rand(bs, N, device=self.device) * self._cvar_alpha

                    # Evaluate both critics at CVaR-restricted quantiles
                    q1_cvar, _ = self.critic1.q_head_forward(c1_feat.detach(), new_action, tau=cvar_tau)
                    q2_cvar, _ = self.critic2.q_head_forward(c2_feat.detach(), new_action, tau=cvar_tau)
                    q_cvar = torch.min(q1_cvar, q2_cvar)  # (B, N)

                    # Actor loss: maximize CVaR (mean of worst-alpha quantiles)
                    actor_loss = (alpha * log_prob - q_cvar.mean(dim=1, keepdim=True)).mean()
                    self._last_q_cvar = q_cvar.detach()  # BUG-DSAC-06: detach to free graph

                self.actor_optimizer.zero_grad()
                if self.scaler.is_enabled():
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                else:
                    actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(self.actor_optimizer)
                else:
                    self._step_if_finite(self.actor_optimizer, actor_norm)   # NAN-01

                # Alpha update (float32, no AMP)
                alpha_loss = -(self.log_alpha * (log_prob.float() + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self._step_if_finite(self.alpha_optimizer, self.log_alpha.grad)   # NAN-01

                self._last_actor_loss = actor_loss
                self._last_alpha_loss = alpha_loss
                self._last_log_prob = log_prob

            # Scaler update
            self.scaler.update()

            # Target Polyak update (fused lerp_)
            with torch.no_grad():
                for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    tp.data.lerp_(p.data, self.tau)
                for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    tp.data.lerp_(p.data, self.tau)

            # Metrics (every 50 steps to avoid .item() sync overhead)
            if self._train_step_count % 50 == 0:
                metrics = {
                    "critic_loss": critic_loss.item(),
                    "grad_skips": float(self._nonfinite_grad_skips),
                    "actor_loss": self._last_actor_loss.item() if self._last_actor_loss is not None else 0.0,
                    "alpha": alpha.item(),
                    "alpha_loss": self._last_alpha_loss.item() if self._last_alpha_loss is not None else 0.0,
                    "entropy": -self._last_log_prob.mean().item() if self._last_log_prob is not None else 0.0,
                    "q1_mean": q1_quantiles.mean().item(),
                    "q1_std": q1_quantiles.std().item(),
                    "cvar_q_mean": self._last_q_cvar.mean().item() if self._last_q_cvar is not None else 0.0,
                    "quantile_spread": (q1_quantiles.max(dim=1).values - q1_quantiles.min(dim=1).values).mean().item(),
                }

        return metrics

    def save(self, path: str):
        """Save with distributional config metadata (single write)."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        actor_sd = self._unwrap_state_dict(self.actor)
        critic1_sd = self._unwrap_state_dict(self.critic1)
        critic2_sd = self._unwrap_state_dict(self.critic2)
        tc1_sd = self._unwrap_state_dict(self.target_critic1)
        tc2_sd = self._unwrap_state_dict(self.target_critic2)

        torch.save({
            "actor": actor_sd,
            "critic1": critic1_sd,
            "critic2": critic2_sd,
            "target_critic1": tc1_sd,
            "target_critic2": tc2_sd,
            "log_alpha": self.log_alpha.data,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "step_count": self.step_count,
            "dsac_config": {
                "n_quantiles": self._n_quantiles,
                "cvar_alpha": self._cvar_alpha,
                "kappa": self.kappa,
                "quantile_embed_dim": self._quantile_embed_dim,
            },
        }, path)

    def load(self, path: str):
        """Load with distributional config validation."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Validate distributional config matches
        dsac_cfg = checkpoint.get("dsac_config", {})
        if dsac_cfg:
            saved_nq = dsac_cfg.get("n_quantiles", self._n_quantiles)
            if saved_nq != self._n_quantiles:
                logger.warning(
                    f"[DSAC] Checkpoint n_quantiles={saved_nq} != current {self._n_quantiles}. "
                    "Load may fail if network dimensions differ."
                )

        self._load_state_dict(self.actor, checkpoint["actor"])
        self._load_state_dict(self.critic1, checkpoint["critic1"])
        self._load_state_dict(self.critic2, checkpoint["critic2"])
        self._load_state_dict(self.target_critic1, checkpoint["target_critic1"])
        self._load_state_dict(self.target_critic2, checkpoint["target_critic2"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
