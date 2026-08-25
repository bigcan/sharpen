"""
SAC (Soft Actor-Critic) Agent — Continuous Position Control

Entropy-regularized actor-critic for Box(-1,1) position fraction.
Twin Q-networks, automatic entropy coefficient tuning, Polyak-averaged targets.

Interface matches IQN/BDQ contract for pipeline compatibility:
  predict(), train_step(), save(), load(), reset_hidden_state(),
  mask_hidden_state(), decay_epsilon()
"""
import copy
import os
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from finrl_pro_ds.agents.common.flat_replay_buffer import FlatReplayBuffer
from finrl_pro_ds.agents.sac.networks import SACActorNetwork, SACCriticNetwork


class _UnpackedObs(NamedTuple):
    scale_stack: torch.Tensor
    next_scale_stack: torch.Tensor
    lob: Optional[torch.Tensor] = None
    next_lob: Optional[torch.Tensor] = None


class SACAgent:
    """Soft Actor-Critic agent for continuous position control.

    Drop-in compatible with the DeepScalper pipeline. Exposes identical
    interface to IQNAgent/DeepScalperBDQ.
    """

    def __init__(
        self,
        network_config: dict,
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
        # Pipeline compatibility kwargs
        **kwargs,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_interval = update_interval
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self._use_bf16 = (self.amp_dtype == torch.bfloat16)
        self.checkpoint_interval = checkpoint_interval
        self.step_count = 0
        self.actor_update_freq = kwargs.get("actor_update_freq", 2)
        self._train_step_count = 0
        # OB-AUD-01: Initialize metric stash so getattr fallback is never needed
        self._last_actor_loss = None
        self._last_alpha_loss = None
        self._nonfinite_grad_skips = 0   # NAN-01 — see _step_if_finite
        self._last_log_prob = None
        # pin_memory disabled: cudaHostAlloc overhead dominates for small batch
        # sizes typical in HPO/training. Sync .to(device) is faster in practice.
        self._use_pinned = False

        # Extract network config
        scale_cfg = network_config.get("scale_encoder", {
            "input_size": 7, "channels": [32, 64, 64, 64],
            "kernel_size": 3, "output_dim": 64,
        })
        # Convert list to tuple for nn.Module
        if isinstance(scale_cfg.get("channels"), list):
            scale_cfg["channels"] = tuple(scale_cfg["channels"])
        private_dim = network_config.get("private_dim", 5)
        fusion_dim = network_config.get("fusion_dim", 256)

        # v6: Summary-stats observation mode
        self._obs_mode = network_config.get("obs_mode", "window")
        if self._obs_mode == "summary_stats":
            # Pass summary_input_dim through scale_cfg for encoder construction
            scale_cfg["summary_input_dim"] = network_config.get("summary_input_dim", 50)

        # Window/feature config for replay buffer shapes
        window_size = network_config.get("window_size", 30)
        features_per_scale = scale_cfg.get("input_size", 7)
        self._window_size = window_size
        self._features_per_scale = features_per_scale
        self._n_scales = network_config.get("n_scales", 3)

        # Action dimension — configurable for multi-dim actions (e.g., MM 3D)
        self._action_dim = network_config.get("action_dim", 1)

        # Optional LOB encoder config (Market Making V8)
        lob_cfg = network_config.get("lob_encoder")
        if lob_cfg is not None:
            if isinstance(lob_cfg.get("channels"), list):
                lob_cfg["channels"] = tuple(lob_cfg["channels"])
        self._has_lob = lob_cfg is not None
        self._n_lob_features = network_config.get("n_lob_features", 0)
        if self._has_lob and self._obs_mode == "summary_stats":
            raise ValueError("LOB encoder is not compatible with summary_stats obs_mode")

        # Build networks
        self.actor = SACActorNetwork(scale_cfg, private_dim, fusion_dim, self._n_scales, action_dim=self._action_dim, obs_mode=self._obs_mode, lob_encoder_config=lob_cfg).to(self.device)
        self.critic1 = SACCriticNetwork(scale_cfg, private_dim, fusion_dim, action_dim=self._action_dim, n_scales=self._n_scales, obs_mode=self._obs_mode, lob_encoder_config=lob_cfg).to(self.device)
        self.critic2 = SACCriticNetwork(scale_cfg, private_dim, fusion_dim, action_dim=self._action_dim, n_scales=self._n_scales, obs_mode=self._obs_mode, lob_encoder_config=lob_cfg).to(self.device)

        # Target critics (Polyak-averaged)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Entropy coefficient (learnable)
        # FIX R2-AUD-08: Create Parameter directly on target device to preserve
        # nn.Parameter type (nn.Parameter.to() returns plain Tensor on device change).
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(initial_alpha, dtype=torch.float32, device=self.device)),
        )
        self.target_entropy = -float(self._action_dim)  # -dim(action_space)

        # Optimizers — OPT-09: fused=True uses single CUDA kernel for param update
        _fused = self.device.type == "cuda"
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, fused=_fused)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic, fused=_fused,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha, fused=_fused)

        # Replay buffer
        # v6: summary_stats mode stores flat (summary_dim,) instead of (W, F) windows
        if self._obs_mode == "summary_stats":
            summary_dim = network_config.get("summary_input_dim", 50)
            self._summary_dim = summary_dim
            self.replay_buffer = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(summary_dim,),
                macro_shape=(0,),
                private_shape=(0,),
                action_shape=(self._action_dim,),
                action_dtype=np.float32,
            )
        else:
            # Store scale_0 as "micro" (W,F), concat of remaining scales as "macro" (flat),
            # private as "private" (private_dim,)
            self._summary_dim = 0
            macro_flat_dim = window_size * features_per_scale * (self._n_scales - 1)
            if self._has_lob:
                macro_flat_dim += window_size * self._n_lob_features
            self.replay_buffer = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(window_size, features_per_scale),
                macro_shape=(macro_flat_dim,),
                private_shape=(private_dim,),
                action_shape=(self._action_dim,),
                action_dtype=np.float32,
            )

        # RCRP: Regime-balanced replay sampling (set from training config)
        self.regime_balanced_replay: str | None = None  # None/"balanced"/"inverse_freq"/"transition_boosted"

        # AMP scaler — disabled for BF16 (same dynamic range as FP32, no scaling needed)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=use_amp and not self._use_bf16 and self.device.type == 'cuda',
        )

        # torch.compile: full-model compile now works because networks accept
        # stacked (B,N,W,F) tensors instead of List[Tensor] (Session 127 fix).
        # OA-AUD-01: Training critics use encode()/q_head_forward() which bypass
        # full-model compile (OptimizedModule only intercepts __call__→forward).
        # Sub-module compile (critic.encoder/q_head individually) was tested but
        # breaks _load_state_dict due to nested _orig_mod prefix patterns
        # (OA-AUD-03). Training critics run eager — acceptable because O-A
        # encoder caching already eliminates 2/4 encoder passes per step.
        # Target critics + actor still use full forward() via __call__.
        if torch_compile and hasattr(torch, 'compile'):
            try:
                # FIX BUG-08: Disable max_autotune_gemm to prevent Inductor
                # decomposition/fallback conflict on aten.mm.default.
                # This removes the ambiguity between fallback ATen mm and
                # Triton-decomposed mm that caused 48/50 trial crashes in GMGP1-v4.
                import torch._inductor.config as _inductor_cfg
                _inductor_cfg.max_autotune_gemm = False
                self.actor = torch.compile(self.actor, mode='default')
                # Training critics: NOT compiled (encode/q_head_forward bypass).
                # O-A caching provides the main gain (2 fewer encoder passes).
                self.target_critic1 = torch.compile(self.target_critic1, mode='default')
                self.target_critic2 = torch.compile(self.target_critic2, mode='default')
                import logging
                logging.getLogger(__name__).info(
                    "[torch.compile] actor + 2 targets (full-model). "
                    "Training critics: eager (O-A encode/q_head split). "
                    "max_autotune_gemm=False (BUG-08 workaround)",
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"torch.compile failed: {e}")

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def predict(
        self,
        scale_input,
        private: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        lob: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict action given multi-scale observations.

        Args:
            scale_input: either a pre-stacked (B, N, W, F) tensor, a list of
                         N tensors each (B, W, F), or a flat (B, D) tensor in
                         summary_stats mode.
            private: (B, private_dim) or None in summary_stats mode
            deterministic: Use mean action (no sampling)
            lob: (B, W, n_lob_features) optional LOB microstructure tensor

        Returns:
            actions: (B, 1) continuous position fraction
        """
        if self._obs_mode == "summary_stats":
            # In summary mode, scale_input is already (B, flat_dim)
            scale_stack = scale_input
        elif isinstance(scale_input, list):
            scale_stack = torch.stack(scale_input, dim=1)
        else:
            scale_stack = scale_input
        # FIX R7-AUD-06: Disable dropout for deterministic inference (backtest/eval).
        # DilatedCNNEncoder has dropout=0.1 — without eval mode, deterministic
        # predictions have random noise, breaking backtest reproducibility.
        if deterministic:
            self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor.sample(
                scale_stack, private,
                deterministic=deterministic,
                lob=lob,
            )
        if deterministic:
            self.actor.train()
        return action

    def store_transition(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ):
        """Store a single transition in the replay buffer."""
        state = self._obs_to_buffer(obs)
        next_state = self._obs_to_buffer(next_obs)
        self.replay_buffer.push(
            state, action, reward, next_state, done, aux_target=0.0,
        )

    def store_batch(
        self,
        obs_batch: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_batch: dict[str, np.ndarray],
        dones: np.ndarray,
        regime_codes: np.ndarray | None = None,
    ):
        """Store N transitions at once using batch push."""
        states = self._obs_batch_to_buffer(obs_batch)
        next_states = self._obs_batch_to_buffer(next_obs_batch)
        aux = np.zeros(len(rewards), dtype=np.float32)
        self.replay_buffer.push_batch(
            states, actions, rewards, next_states, dones, aux,
            regime_codes=regime_codes,
        )

    def train_step(self) -> Optional[dict[str, float]]:
        """One SAC update step. Returns metrics dict or None if buffer too small.

        For better throughput, prefer train_step_mega(n_steps) which samples
        once and runs multiple gradient steps from GPU-resident data.
        """
        return self.train_step_mega(1)

    def _step_if_finite(self, optimizer: optim.Optimizer, total_norm: torch.Tensor) -> None:
        """AMP-off counterpart to ``GradScaler``'s skip-on-non-finite (NAN-01).

        When AMP is enabled with float16, ``scaler.step()`` skips the update whenever
        ``unscale_`` found inf/NaN in that optimizer's gradients. But the scaler is disabled
        for ``use_amp: false`` AND for ``amp_dtype: bfloat16`` (bf16 has fp32's exponent range,
        so loss scaling is pointless) — and on that branch the code fell through to a plain
        ``optimizer.step()`` with no check at all. ``clip_grad_norm_`` does not cover the gap:
        a non-finite gradient makes ``total_norm`` non-finite, so ``clip_coef`` is non-finite,
        so every gradient is multiplied to NaN and the step writes NaN into EVERY parameter of
        the network in one shot — strictly worse than the ``log_alpha`` case, which corrupts a
        single scalar.

        ``total_norm`` is a COMPLETE detector, which is why it is the thing checked: it is the
        square root of a sum over every gradient element, so it is non-finite if and only if at
        least one gradient element is. ``clip_grad_norm_`` has already poisoned the gradients
        by the time this runs, but skipping the step means they are never applied and the next
        ``zero_grad()`` clears them.

        Costs one device sync, which is deliberate and not a regression: the AMP path already
        pays exactly this — ``GradScaler._maybe_opt_step`` reads ``found_inf`` with ``.item()``
        before deciding. The point is to make the two branches behave identically, and
        skipping (rather than zeroing or clamping) is precisely GradScaler's own semantic.

        Skips are counted into ``grad_skips`` so a run that has silently stopped learning is
        visible in the metrics rather than looking like healthy-but-flat training.
        """
        if torch.isfinite(total_norm):
            optimizer.step()
        else:
            self._nonfinite_grad_skips += 1

    def _guard_alpha_grad(self) -> None:
        """Zero a non-finite ``log_alpha`` gradient in place, before the optimizer sees it (NAN-01).

        ``log_alpha`` is the ONE unguarded optimizer in this agent. The twin critics and the
        actor are both protected by ``GradScaler``, which skips an optimizer step whenever
        ``unscale_`` finds inf/NaN — but the alpha update is deliberately run outside AMP
        (its loss is already float32), so it never passes through the scaler and takes any
        gradient it is handed.

        That asymmetry is what turns a LOCAL fault into a GLOBAL, PERMANENT one. A single
        NaN row in ``log_prob`` (e.g. one poisoned observation in the batch — the crash in
        randd_log S553-cont-164/165) makes ``alpha_loss`` NaN via ``.mean()``, which writes a
        NaN gradient into ``log_alpha``. From that step on ``alpha = log_alpha.exp()`` is NaN,
        and since alpha multiplies the entropy term of the target for the ENTIRE batch
        (``target = min(q1,q2) − alpha·next_log_prob``), every subsequent target is NaN for
        every row — the run cannot recover even if no further bad row is ever sampled. Weights
        do not un-NaN themselves.

        Zeroing the gradient makes that step a no-op for ``log_alpha``, which is exactly the
        skip semantics ``GradScaler`` gives the other two optimizers.

        **Deliberately branch-free.** ``torch.isfinite(g).all()`` read in Python forces a
        device sync on every actor update, on the hot path OPT-07/08 restructured to overlap
        GPU training with CPU env work. ``nan_to_num_`` stays on-device and costs nothing
        measurable on a 1-element tensor.

        This CONTAINS the fault; it does not report it. Detection is
        ``SACTrainer``'s finite-metric assertion, which still sees the NaN ``actor_loss`` this
        guard does not (and must not) suppress, and halts the run.
        """
        g = self.log_alpha.grad
        if g is not None:
            torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)

    def train_step_mega(self, n_steps: int = 1) -> Optional[dict[str, float]]:
        """Run n_steps SAC gradient updates from a single mega-batch.

        OPT-07: Samples n_steps * batch_size transitions ONCE, transfers to
        GPU ONCE, then runs n_steps gradient steps from GPU-resident mini-batches.
        Eliminates CPU-GPU interleaving that caused 40 pipeline flushes per env.step().

        Args:
            n_steps: Number of gradient steps to run from the mega-batch.

        Returns:
            Latest metrics dict, or None if buffer too small.
        """
        if len(self.replay_buffer) < self.learning_starts:
            return None

        mega_batch_size = n_steps * self.batch_size

        # --- ONE CPU phase: sample + transfer ---
        if self.regime_balanced_replay:
            states, actions_np, rewards_np, next_states, dones_np, _ = \
                self.replay_buffer.sample_regime_balanced(mega_batch_size, mode=self.regime_balanced_replay)
        else:
            states, actions_np, rewards_np, next_states, dones_np, _ = \
                self.replay_buffer.sample(mega_batch_size)

        # Transfer ALL data to GPU at once
        unpacked = self._unpack_buffer_to_stacks(states, next_states)
        all_scale_stack = unpacked.scale_stack
        all_next_scale_stack = unpacked.next_scale_stack
        all_lob = unpacked.lob          # None if no LOB config
        all_next_lob = unpacked.next_lob
        # v6: In summary_stats mode, private is baked into the flat vector; pass None
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

        # --- ONE GPU phase: n_steps gradient steps, no CPU interruption ---
        for step_i in range(n_steps):
            self._train_step_count += 1
            start = step_i * bs
            end = start + bs

            # Slice mini-batch from GPU-resident mega-batch (views, zero-copy)
            scale_stack = all_scale_stack[start:end]
            next_scale_stack = all_next_scale_stack[start:end]
            priv = all_priv[start:end] if all_priv is not None else None
            npriv = all_npriv[start:end] if all_npriv is not None else None
            lob_mb = all_lob[start:end] if all_lob is not None else None
            next_lob_mb = all_next_lob[start:end] if all_next_lob is not None else None
            actions = all_actions[start:end]
            rewards = all_rewards[start:end]
            dones = all_dones[start:end]

            alpha = self.log_alpha.exp().detach()

            # --- Critic update ---
            # OPT-OA: Encode current state ONCE, reuse features for actor update
            with torch.no_grad():
                with amp_ctx:
                    next_action, next_log_prob = self.actor.sample(next_scale_stack, npriv, lob=next_lob_mb)
                    target_q1 = self.target_critic1(next_scale_stack, npriv, next_action, lob=next_lob_mb)
                    target_q2 = self.target_critic2(next_scale_stack, npriv, next_action, lob=next_lob_mb)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_value = rewards + (1.0 - dones) * self.gamma * target_q

            with amp_ctx:
                c1_feat = self.critic1.encode(scale_stack, priv, lob=lob_mb)
                c2_feat = self.critic2.encode(scale_stack, priv, lob=lob_mb)
                q1 = self.critic1.q_head_forward(c1_feat, actions)
                q2 = self.critic2.q_head_forward(c2_feat, actions)
                critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)

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

            # --- Actor + Alpha update (O-B: delayed, every actor_update_freq steps) ---
            if self._train_step_count % self.actor_update_freq == 0:
                with amp_ctx:
                    new_action, log_prob = self.actor.sample(scale_stack, priv, lob=lob_mb)
                    # OPT-OA: Reuse cached encoder features — .detach() prevents
                    # critic encoder gradients from flowing into actor update
                    q1_new = self.critic1.q_head_forward(c1_feat.detach(), new_action)
                    q2_new = self.critic2.q_head_forward(c2_feat.detach(), new_action)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (alpha * log_prob - q_new).mean()

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

                # --- Alpha update (float32 — no AMP) ---
                alpha_loss = -(self.log_alpha * (log_prob.float() + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self._guard_alpha_grad()
                self.alpha_optimizer.step()

                # Stash for metrics on non-actor steps
                self._last_actor_loss = actor_loss
                self._last_alpha_loss = alpha_loss
                self._last_log_prob = log_prob

            # Scaler update (no-op if disabled for BF16)
            self.scaler.update()

            # --- Target Polyak update (fused lerp_) ---
            with torch.no_grad():
                for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    tp.data.lerp_(p.data, self.tau)
                for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    tp.data.lerp_(p.data, self.tau)

            # Extract metrics periodically to avoid .item() CUDA sync overhead.
            if self._train_step_count % 50 == 0:
                metrics = {
                    "critic_loss": critic_loss.item(),
                    "grad_skips": float(self._nonfinite_grad_skips),
                    "actor_loss": self._last_actor_loss.item() if self._last_actor_loss is not None else 0.0,
                    "alpha": alpha.item(),
                    "alpha_loss": self._last_alpha_loss.item() if self._last_alpha_loss is not None else 0.0,
                    "entropy": -self._last_log_prob.mean().item() if self._last_log_prob is not None else 0.0,
                    "q1_mean": q1.mean().item(),
                    "q2_mean": q2.mean().item(),
                }

        return metrics

    def _to_device_pinned(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        """Transfer numpy array to GPU via pinned memory for true async DMA.

        Without pin_memory(), non_blocking=True is silently ignored because
        numpy arrays live in pageable memory. pin_memory() copies to page-locked
        RAM first, enabling actual async H2D transfer via DMA engine.
        """
        t = torch.as_tensor(arr, dtype=dtype)
        if self._use_pinned:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device, non_blocking=True)

    def _obs_to_buffer(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert multi-scale obs dict to replay buffer format."""
        if self._obs_mode == "summary_stats":
            # v6: Concat all scale summaries + private into one flat vector
            parts = [obs[f"scale_{i}"] for i in range(self._n_scales)]
            parts.append(obs["private"])
            flat = np.concatenate(parts).astype(np.float32)
            return {
                "micro": flat,
                "macro": np.array([], dtype=np.float32),
                "private": np.array([], dtype=np.float32),
            }
        macro_parts = [obs[f"scale_{i}"].flatten() for i in range(1, self._n_scales)]
        if self._has_lob and "lob" in obs:
            macro_parts.append(obs["lob"].flatten())
        return {
            "micro": obs["scale_0"],
            "macro": np.concatenate(macro_parts) if macro_parts else np.array([], dtype=np.float32),
            "private": obs["private"],
        }

    def _obs_batch_to_buffer(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert batch multi-scale obs dict to replay buffer format."""
        if self._obs_mode == "summary_stats":
            n = obs["scale_0"].shape[0]
            parts = [obs[f"scale_{i}"] for i in range(self._n_scales)]
            parts.append(obs["private"])
            flat = np.concatenate(parts, axis=1).astype(np.float32)
            return {
                "micro": flat,
                "macro": np.zeros((n, 0), dtype=np.float32),
                "private": np.zeros((n, 0), dtype=np.float32),
            }
        n = obs["scale_0"].shape[0]
        macro_parts = [obs[f"scale_{i}"].reshape(n, -1) for i in range(1, self._n_scales)]
        if self._has_lob and "lob" in obs:
            macro_parts.append(obs["lob"].reshape(n, -1))
        return {
            "micro": obs["scale_0"],
            "macro": np.concatenate(macro_parts, axis=1) if macro_parts else np.zeros((n, 0), dtype=np.float32),
            "private": obs["private"],
        }

    def _unpack_buffer_to_stacks(self, states, next_states) -> _UnpackedObs:
        """Unpack replay buffer micro/macro into stacked (B, N, W, F) tensors.

        v6 summary_stats mode: returns flat (B, D) tensors instead of stacked.
        LOB (if present) is extracted from the end of macro before scale unpacking.

        Returns _UnpackedObs namedtuple with scale_stack, next_scale_stack,
        and optional lob/next_lob tensors.
        """
        if self._obs_mode == "summary_stats":
            # v6: micro IS the flat summary+private vector — return directly
            s_flat = self._to_device_pinned(states["micro"])
            ns_flat = self._to_device_pinned(next_states["micro"])
            return _UnpackedObs(s_flat, ns_flat)

        chunk = self._window_size * self._features_per_scale

        # Extract LOB from end of macro before scale unpacking
        lob = None
        next_lob = None
        macro = states["macro"]
        nmacro = next_states["macro"]
        if self._has_lob and self._n_lob_features > 0:
            lob_flat_dim = self._window_size * self._n_lob_features
            lob_offset = macro.shape[1] - lob_flat_dim
            lob = self._to_device_pinned(
                macro[:, lob_offset:].reshape(-1, self._window_size, self._n_lob_features),
            )
            next_lob = self._to_device_pinned(
                nmacro[:, lob_offset:].reshape(-1, self._window_size, self._n_lob_features),
            )
            # Trim macro to only contain scale data
            macro = macro[:, :lob_offset]
            nmacro = nmacro[:, :lob_offset]

        # Scale 0 is stored as "micro"
        s0 = self._to_device_pinned(states["micro"])
        ns0 = self._to_device_pinned(next_states["micro"])
        scale_list = [s0]
        next_scale_list = [ns0]

        # Remaining scales are packed in "macro"
        if self._n_scales > 1:
            for i in range(1, self._n_scales):
                offset = (i - 1) * chunk
                flat = macro[:, offset:offset + chunk]
                nflat = nmacro[:, offset:offset + chunk]
                scale_list.append(
                    self._to_device_pinned(
                        flat.reshape(-1, self._window_size, self._features_per_scale),
                    ),
                )
                next_scale_list.append(
                    self._to_device_pinned(
                        nflat.reshape(-1, self._window_size, self._features_per_scale),
                    ),
                )

        # Stack into (B, N, W, F) for compile-friendly forward
        return _UnpackedObs(
            torch.stack(scale_list, dim=1),
            torch.stack(next_scale_list, dim=1),
            lob,
            next_lob,
        )

    def save(self, path: str):
        """Save all model state to checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Unwrap compiled modules if needed
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
        }, path)

    def load(self, path: str):
        """Load model state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

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

    @staticmethod
    def _unwrap_state_dict(module):
        """Handle torch.compile _orig_mod prefix."""
        sd = module.state_dict()
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    @staticmethod
    def _load_state_dict(module, state_dict):
        """Load state dict handling torch.compile prefix."""
        try:
            module.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Try adding _orig_mod prefix for compiled modules
            new_sd = {"_orig_mod." + k: v for k, v in state_dict.items()}
            module.load_state_dict(new_sd, strict=True)

    # --- Pipeline compatibility stubs ---

    def reset_hidden_state(self):
        """No-op: SAC uses stateless CNN, no hidden state."""
        pass

    def mask_hidden_state(self, dones):
        """No-op: SAC uses stateless CNN."""
        pass

    def decay_epsilon(self):
        """No-op: SAC has no epsilon. Increments step_count for compatibility."""
        self.step_count += 1
