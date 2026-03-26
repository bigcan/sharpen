"""
IQN (Implicit Quantile Network) Agent — Drop-in replacement for DeepScalperBDQ.

Addresses all 4 root causes of RL failure:
1. Q-mean can't resolve 0.3 bps signal → IQN learns full return distribution
2. Epsilon-greedy catastrophic for 95% Hold-optimal → NoisyNets (learned exploration)
3. 43K-step episodes destroy credit assignment → Daily episodes (env-side)
4. Hold domination in replay → Stratified sampling (trainer-side)

Multi-Horizon Mode (Session 67):
  Two sets of dueling heads (short + long discount), shared encoder backbone.
  Short-horizon captures immediate momentum, long-horizon captures strategic value.
  Action selection blends: Q = alpha * Q_short + (1-alpha) * Q_long.
  Ref: Fedus et al. (2019) "Hyperbolic Discounting and Learning over Multiple Horizons"

Interface is identical to DeepScalperBDQ for trainer/pipeline compatibility.
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer
from finrl_pro_ds.agents.deepscalper.iqn_network import IQNNetwork
from finrl_pro_ds.agents.deepscalper.per_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class IQNAgent:
    """Implicit Quantile Network agent with NoisyNet exploration.

    Drop-in replacement for DeepScalperBDQ. Exposes identical interface:
    predict(), train_step(), save(), load(), reset_hidden_state(),
    mask_hidden_state(), and all required properties.
    """

    def __init__(
        self,
        network_config: dict,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 500000,
        num_quantiles: int = 32,
        embedding_dim: int = 64,
        noisy_sigma0: float = 0.5,
        quantile_huber_kappa: float = 1.0,
        gradient_clip: float = 10.0,
        target_q_clip: float = 5000.0,
        auxiliary_weight: float = 0.1,
        use_amp: bool = False,
        amp_dtype: str = "float16",
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        torch_compile: bool = False,
        n_step: int = 1,
        multi_horizon: bool = False,
        gamma_short: float = 0.95,
        gamma_long: float = 0.99,
        horizon_alpha: float = 0.5,
        fee_threshold: float = 0.0,
        exploration_mode: str = "noisy",
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        action_dims=3,
        device: str = "cpu",
        # Absorb BDQ-specific kwargs for config compatibility
        **kwargs,
    ):
        self.device = torch.device(device)
        self.fee_threshold = fee_threshold
        self._exploration_mode = exploration_mode
        self.gamma = gamma
        self.n_step = n_step
        # Effective gamma for Bellman target: gamma^n for n-step returns
        self.gamma_n = gamma ** n_step

        # Multi-horizon: short-term momentum + long-term strategy
        self.multi_horizon = multi_horizon
        if multi_horizon:
            self.gamma_short = gamma_short
            self.gamma_long = gamma_long
            self.gamma_short_n = gamma_short ** n_step
            self.gamma_long_n = gamma_long ** n_step
            self.horizon_alpha = horizon_alpha
            # Sync base gamma with gamma_long so N-step buffer discounting
            # matches the long-horizon Bellman target exactly.
            self.gamma = gamma_long
            self.gamma_n = gamma_long ** n_step
        else:
            self.gamma_short = gamma
            self.gamma_long = gamma
            self.gamma_short_n = self.gamma_n
            self.gamma_long_n = self.gamma_n
            self.horizon_alpha = 1.0

        self.tau = tau
        self.batch_size = batch_size
        self.num_quantiles = num_quantiles
        self.kappa = quantile_huber_kappa
        self.gradient_clip = gradient_clip
        self.target_q_clip = target_q_clip
        self.auxiliary_weight = auxiliary_weight
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self._use_bf16 = (self.amp_dtype == torch.bfloat16)
        self.use_per = use_per

        # Exploration: NoisyNets (noisy) or epsilon-greedy (epsilon)
        if exploration_mode == "epsilon":
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self._epsilon_start = epsilon_start
            self._epsilon_decay_steps = 1  # Set by trainer
        else:
            self.epsilon = 0.0
            self.epsilon_end = 0.0
        self.step_count = 0

        # Stratified sampling config (set by trainer from config)
        self.stratified_sampling = False
        self.stratified_hold_action = 1  # Hold action index for Discrete(3)
        self.stratified_hold_ratio = 0.5

        # Normalize action_dims to single int (IQN uses Discrete, not MultiDiscrete)
        # FIX J-03: Fail fast if MultiDiscrete is passed — IQN only supports Discrete.
        if isinstance(action_dims, (list, tuple)):
            assert len(action_dims) == 1, (
                f"IQNAgent only supports Discrete action spaces (single dim), "
                f"got MultiDiscrete {action_dims}. Use BDQ for multi-branch actions."
            )
            self.n_actions = action_dims[0]
            self.action_dims = list(action_dims)
        else:
            self.n_actions = action_dims
            self.action_dims = [action_dims]

        # Build IQN network config
        net_cfg = dict(network_config)
        iqn_kwargs = {
            "micro_config": net_cfg.get("micro_config", {}),
            "macro_config": net_cfg.get("macro_config", {}),
            "fusion_dim": net_cfg.get("fusion_dim", 256),
            "n_actions": self.n_actions,
            "embedding_dim": embedding_dim,
            "noisy_sigma0": noisy_sigma0,
            "multi_horizon": multi_horizon,
            "exploration_mode": exploration_mode,
        }

        # Build networks
        self.policy_net = IQNNetwork(**iqn_kwargs).to(self.device)
        self.target_net = IQNNetwork(**iqn_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # torch.compile
        self._torch_compiled = False
        if torch_compile and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, mode="default")
                self.target_net = torch.compile(self.target_net, mode="default")
                self._torch_compiled = True
                logger.info("[torch.compile] IQN policy_net + target_net compiled (mode=default)")
            except Exception as e:
                logger.warning(f"[torch.compile] Failed, falling back to eager mode: {e}")

        # PERF-OPT S154 (O5): Fused Adam — single CUDA kernel per step
        _fused = self.device.type == "cuda"
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, fused=_fused)
        self._lr_scheduler = None  # Initialized by trainer

        # AMP GradScaler — disabled for BF16 (same dynamic range as FP32, no scaling needed)
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.use_amp and not self._use_bf16,
        )

        # Replay buffer
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
            )
        else:
            micro_cfg = network_config.get("micro_config", {})
            macro_cfg = network_config.get("macro_config", {})
            window_size = micro_cfg.get("window_size", 15)
            micro_input = micro_cfg.get("input_size", 30)
            private_input = micro_cfg.get("private_input_size", 5)
            macro_input = macro_cfg.get("input_size", 15)
            self.memory = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(window_size, micro_input),
                macro_shape=(macro_input,),
                private_shape=(window_size, private_input),
                action_shape=(len(self.action_dims),),
            )

        self._use_pinned = self.device.type == "cuda"
        self._hidden_state = None

    def reset_hidden_state(self):
        """Reset LSTM hidden state (episode start)."""
        self._hidden_state = None

    def mask_hidden_state(self, dones: np.ndarray):
        """Zero out hidden states for terminated environments."""
        if self._hidden_state is None:
            return
        h, c = self._hidden_state
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32).view(1, -1, 1)
        h = h * (1.0 - dones_t)
        c = c * (1.0 - dones_t)
        self._hidden_state = (h, c)

    def predict(
        self,
        micro: torch.Tensor,
        private_in: torch.Tensor,
        macro: torch.Tensor,
        deterministic: bool = False,
        qty_mask=None,
        context: Optional[dict] = None,
        eval_epsilon: float = 0.0,
    ) -> np.ndarray:
        """Select action by averaging over quantile Q-values.

        In multi-horizon mode, blends short and long horizon Q-means:
          Q = alpha * Q_short + (1-alpha) * Q_long

        NoisyNets provide exploration (no epsilon needed).
        In eval/deterministic mode, noise is suppressed by NoisyLinear.eval().

        Args:
            context: Optional dict with 'current_direction' (np.ndarray of shape (B,))
                for fee_threshold filtering. When fee_threshold > 0, switches are
                suppressed unless Q(switch) - Q(stay) > fee_threshold.
        """
        micro = micro.to(self.device, non_blocking=True)
        private_in = private_in.to(self.device, non_blocking=True)
        macro = macro.to(self.device, non_blocking=True)

        if deterministic:
            self.policy_net.eval()

        with torch.no_grad():
            batch_size = micro.shape[0]
            tau = torch.rand(batch_size, self.num_quantiles, device=self.device)

            if self.multi_horizon:
                q_short, q_long, _, new_hidden = self.policy_net.forward_dual(
                    micro, private_in, macro, tau, hidden=self._hidden_state,
                )
                self._hidden_state = new_hidden
                # Blend: alpha * short + (1-alpha) * long
                q_mean = (
                    self.horizon_alpha * q_short.mean(dim=1)
                    + (1.0 - self.horizon_alpha) * q_long.mean(dim=1)
                )
            else:
                q_tau, _, new_hidden = self.policy_net(
                    micro, private_in, macro, tau, hidden=self._hidden_state,
                )
                self._hidden_state = new_hidden
                q_mean = q_tau.mean(dim=1)

            # Apply action mask if provided
            if qty_mask is not None:
                qty_mask_t = torch.as_tensor(qty_mask, dtype=torch.float32, device=self.device)
                if qty_mask_t.dim() == 1:
                    qty_mask_t = qty_mask_t.unsqueeze(0).expand(batch_size, -1)
                q_mean = q_mean.masked_fill(qty_mask_t == 0, float("-inf"))

            actions = q_mean.argmax(dim=-1)  # (B,)

            # eval_epsilon: minimal exploration during deterministic eval
            if deterministic and eval_epsilon > 0:
                rand_mask = torch.rand(batch_size, device=self.device) < eval_epsilon
                if rand_mask.any():
                    random_actions = torch.randint(0, self.n_actions, (batch_size,), device=self.device)
                    actions = torch.where(rand_mask, random_actions, actions)

            # Epsilon-greedy exploration (OPT-C: replaces NoisyNets)
            if not deterministic and self._exploration_mode == "epsilon" and self.epsilon > 0:
                rand_mask = torch.rand(batch_size, device=self.device) < self.epsilon
                if rand_mask.any():
                    random_actions = torch.randint(0, self.n_actions, (batch_size,), device=self.device)
                    actions = torch.where(rand_mask, random_actions, actions)

            # Fee-threshold switching filter (inference-time only, P3)
            # Suppresses switches unless advantage > fee_threshold
            if self.fee_threshold > 0 and context is not None and self.n_actions == 2:
                current_dir = context.get("current_direction")
                if current_dir is not None:
                    current_dir = np.asarray(current_dir)
                    # Map direction (+1/-1) to action index (0=Long, 1=Short)
                    stay_idx = np.where(current_dir > 0, 0, 1).astype(np.int64)
                    switch_idx = 1 - stay_idx
                    q_np = q_mean.cpu().numpy()
                    acts_np = actions.cpu().numpy()
                    for i in range(batch_size):
                        if acts_np[i] == switch_idx[i]:
                            advantage = q_np[i, switch_idx[i]] - q_np[i, stay_idx[i]]
                            if advantage < self.fee_threshold:
                                acts_np[i] = stay_idx[i]  # Stay
                    if deterministic:
                        self.policy_net.train()
                    return acts_np

        if deterministic:
            self.policy_net.train()
        # PERF-OPT S154 (O4): Removed redundant _reset_noise() here.
        # With UTD=8, the last train_step() already resets noise before
        # the next predict() call.  Saves ~80-160 kernel dispatches/step.

        return actions.cpu().numpy()

    def _to_device_pinned(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        """Transfer numpy array to GPU via pinned memory for async DMA."""
        t = torch.as_tensor(arr, dtype=dtype)
        if self._use_pinned:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def _compute_quantile_loss(
        self, q_tau_a: torch.Tensor, T_tau: torch.Tensor, tau: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantile Huber loss between predicted and target quantiles.

        Args:
            q_tau_a: (B, N) predicted quantile values for taken action
            T_tau: (B, N') target quantile values
            tau: (B, N) quantile fractions

        Returns:
            quantile_loss: (B,) per-sample loss
        """
        delta = T_tau.unsqueeze(1) - q_tau_a.unsqueeze(2)  # (B, N, N')
        abs_delta = delta.abs()
        kappa = self.kappa
        huber = torch.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2) / kappa,
            abs_delta - 0.5 * kappa,
        )
        tau_expanded = tau.unsqueeze(2)
        weight = (tau_expanded - (delta < 0).float()).abs()
        return (weight * huber).mean(dim=1).mean(dim=1)  # (B,)

    def train_step(self) -> Optional[dict[str, float]]:
        """Single gradient step (backward compat). Prefer train_step_mega()."""
        return self.train_step_mega(1)

    def train_step_mega(self, n_steps: int = 1) -> Optional[dict[str, float]]:
        """Run n_steps IQN gradient updates from a single mega-batch.

        PERF-OPT S154 (O1): Samples n_steps * batch_size transitions ONCE,
        transfers to GPU ONCE, then runs n_steps gradient steps from
        GPU-resident mini-batches. Eliminates N separate CPU-GPU round-trips
        that caused pipeline stalls with UTD=8.

        For PER, falls back to single-step mode (PER priorities need per-step
        updates and the SumTree stores Python objects that can't be mega-batched).
        """
        self.policy_net.train()
        if len(self.memory) < self.batch_size:
            return None

        # PER fallback: can't mega-batch Python-object SumTree
        if self.use_per:
            metrics = None
            for _ in range(n_steps):
                metrics = self._train_step_single_per()
            return metrics

        # --- ONE CPU phase: sample n_steps * batch_size, transfer to GPU ---
        mega_batch_size = n_steps * self.batch_size
        if self.stratified_sampling and hasattr(self.memory, "sample_stratified"):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = (
                self.memory.sample_stratified(
                    mega_batch_size,
                    hold_action=self.stratified_hold_action,
                    hold_ratio=self.stratified_hold_ratio,
                )
            )
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = self.memory.sample(mega_batch_size)

        all_micro_s = self._to_device_pinned(state_batch["micro"])
        all_private_s = self._to_device_pinned(state_batch["private"])
        all_macro_s = self._to_device_pinned(state_batch["macro"])
        all_micro_ns = self._to_device_pinned(next_state_batch["micro"])
        all_private_ns = self._to_device_pinned(next_state_batch["private"])
        all_macro_ns = self._to_device_pinned(next_state_batch["macro"])
        all_actions = self._to_device_pinned(action_batch, dtype=torch.long)
        all_rewards = self._to_device_pinned(reward_batch)
        all_dones = self._to_device_pinned(done_batch)
        all_aux_targets = self._to_device_pinned(aux_target_batch).unsqueeze(1)

        # --- ONE GPU phase: n_steps gradient steps, no CPU interruption ---
        metrics = None
        bs = self.batch_size
        for step_i in range(n_steps):
            s = step_i * bs
            e = s + bs
            metrics = self._train_step_on_batch(
                all_micro_s[s:e], all_private_s[s:e], all_macro_s[s:e],
                all_micro_ns[s:e], all_private_ns[s:e], all_macro_ns[s:e],
                all_actions[s:e], all_rewards[s:e], all_dones[s:e],
                all_aux_targets[s:e], per_indices=None, is_weights_t=None,
            )
        return metrics

    def _train_step_single_per(self) -> Optional[dict[str, float]]:
        """Single gradient step with PER sampling (not mega-batchable)."""
        (state_batch, action_batch, reward_batch, next_state_batch,
         done_batch, aux_target_batch, per_indices, is_weights) = self.memory.sample(self.batch_size)
        is_weights_t = self._to_device_pinned(np.array(is_weights))

        def stack_dict_keys(batch_list, key):
            return self._to_device_pinned(np.array([s[key] for s in batch_list]))

        micro_s = stack_dict_keys(state_batch, "micro")
        private_s = stack_dict_keys(state_batch, "private")
        macro_s = stack_dict_keys(state_batch, "macro")
        micro_ns = stack_dict_keys(next_state_batch, "micro")
        private_ns = stack_dict_keys(next_state_batch, "private")
        macro_ns = stack_dict_keys(next_state_batch, "macro")
        actions = self._to_device_pinned(np.array(action_batch), dtype=torch.long)
        rewards = self._to_device_pinned(np.array(reward_batch))
        dones = self._to_device_pinned(np.array(done_batch))
        aux_targets = self._to_device_pinned(np.array(aux_target_batch)).unsqueeze(1)

        return self._train_step_on_batch(
            micro_s, private_s, macro_s, micro_ns, private_ns, macro_ns,
            actions, rewards, dones, aux_targets, per_indices, is_weights_t,
        )

    def _train_step_on_batch(
        self, micro_s, private_s, macro_s, micro_ns, private_ns, macro_ns,
        actions, rewards, dones, aux_targets, per_indices, is_weights_t,
    ) -> Optional[dict[str, float]]:
        """GPU-only gradient step on pre-transferred batch tensors."""

        B = micro_s.shape[0]
        N = self.num_quantiles
        N_prime = self.num_quantiles

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            tau = torch.rand(B, N, device=self.device)
            tau_prime = torch.rand(B, N_prime, device=self.device)
            actions_flat = actions.view(B, 1) if actions.dim() == 1 else actions[:, 0:1]
            r = rewards.view(B, 1)
            d = dones.view(B, 1)

            if self.multi_horizon:
                # --- Multi-Horizon Training ---
                # Policy net: both horizons
                q_short, q_long, pred_vol, _ = self.policy_net.forward_dual(
                    micro_s, private_s, macro_s, tau,
                )
                q_short_a = q_short.gather(2, actions_flat.unsqueeze(1).expand(-1, N, -1)).squeeze(2)
                q_long_a = q_long.gather(2, actions_flat.unsqueeze(1).expand(-1, N, -1)).squeeze(2)

                with torch.no_grad():
                    # Double DQN: use BLENDED Q from policy net to select action
                    q_next_short, q_next_long, _, _ = self.policy_net.forward_dual(
                        micro_ns, private_ns, macro_ns, tau_prime,
                    )
                    q_next_blended = (
                        self.horizon_alpha * q_next_short.mean(dim=1)
                        + (1.0 - self.horizon_alpha) * q_next_long.mean(dim=1)
                    )
                    a_star = q_next_blended.argmax(dim=-1, keepdim=True)  # (B, 1)

                    # Target net: both horizons
                    tgt_short, tgt_long, _, _ = self.target_net.forward_dual(
                        micro_ns, private_ns, macro_ns, tau_prime,
                    )
                    tgt_short_a = tgt_short.gather(2, a_star.unsqueeze(1).expand(-1, N_prime, -1)).squeeze(2)
                    tgt_long_a = tgt_long.gather(2, a_star.unsqueeze(1).expand(-1, N_prime, -1)).squeeze(2)

                    # Bellman targets with different gammas
                    T_short = r + self.gamma_short_n * (1.0 - d) * tgt_short_a
                    T_long = r + self.gamma_long_n * (1.0 - d) * tgt_long_a

                    if self.target_q_clip > 0:
                        T_short = T_short.clamp(-self.target_q_clip, self.target_q_clip)
                        T_long = T_long.clamp(-self.target_q_clip, self.target_q_clip)

                # Quantile Huber loss for each horizon
                loss_short = self._compute_quantile_loss(q_short_a, T_short, tau)
                loss_long = self._compute_quantile_loss(q_long_a, T_long, tau)
                quantile_loss = loss_short + loss_long  # (B,)

                # For metrics, use blended Q
                q_tau_a = self.horizon_alpha * q_short_a + (1.0 - self.horizon_alpha) * q_long_a
                T_tau = self.horizon_alpha * T_short + (1.0 - self.horizon_alpha) * T_long

            else:
                # --- Single-Horizon Training (original) ---
                q_tau_all, pred_vol, _ = self.policy_net(micro_s, private_s, macro_s, tau)
                q_tau_a = q_tau_all.gather(2, actions_flat.unsqueeze(1).expand(-1, N, -1)).squeeze(2)

                with torch.no_grad():
                    q_next_policy, _, _ = self.policy_net(micro_ns, private_ns, macro_ns, tau_prime)
                    a_star = q_next_policy.mean(dim=1).argmax(dim=-1, keepdim=True)

                    q_next_target, _, _ = self.target_net(micro_ns, private_ns, macro_ns, tau_prime)
                    q_target_a = q_next_target.gather(
                        2, a_star.unsqueeze(1).expand(-1, N_prime, -1),
                    ).squeeze(2)

                    T_tau = r + self.gamma_n * (1.0 - d) * q_target_a

                    if self.target_q_clip > 0:
                        T_tau = T_tau.clamp(-self.target_q_clip, self.target_q_clip)

                quantile_loss = self._compute_quantile_loss(q_tau_a, T_tau, tau)

            # PER weighting or uniform mean
            if is_weights_t is not None:
                main_loss = (quantile_loss * is_weights_t).mean()
            else:
                main_loss = quantile_loss.mean()

            # Auxiliary volatility loss
            if is_weights_t is not None:
                vol_loss = (nn.SmoothL1Loss(reduction="none")(pred_vol, aux_targets).squeeze(1) * is_weights_t).mean()
            else:
                vol_loss = nn.SmoothL1Loss()(pred_vol, aux_targets)

            total_loss = main_loss + self.auxiliary_weight * vol_loss

        if not torch.isfinite(total_loss):
            logger.warning(f"IQN Loss is {total_loss.item()} (NaN/Inf). Skipping update.")
            return None

        self.optimizer.zero_grad()

        if self.use_amp and not self._use_bf16:
            # FP16 path: GradScaler prevents underflow
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BF16 or FP32 path: no GradScaler needed
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.optimizer.step()

        # PER priority update
        if self.use_per and per_indices is not None:
            with torch.no_grad():
                td_errors = quantile_loss.detach().cpu().numpy()
            self.memory.update_priorities(per_indices, td_errors)

        # Polyak update + noise reset
        self._polyak_update()
        self._reset_noise()

        # PERF-OPT S154: Batch all GPU→CPU metric transfers into a single sync point.
        # Previously 10-14 individual .item() calls per train_step × 8 UTD = 80-112
        # GPU pipeline stalls per env-step batch. Now 1 sync point.
        q_mean = q_tau_a.mean()
        _metric_vals = [
            total_loss, main_loss, vol_loss, q_mean,
            q_mean, q_tau_a.std(), q_tau_a.max(), q_tau_a.min(),
            T_tau.mean(), T_tau.max(), T_tau.min(),
        ]
        if self.multi_horizon:
            _metric_vals.extend([
                q_short_a.mean(), q_long_a.mean(),
                loss_short.mean(), loss_long.mean(),
            ])
        _cpu = torch.stack([v.detach().float() for v in _metric_vals]).cpu().numpy()

        metrics = {
            "loss_total": float(_cpu[0]),
            "loss_price": 0.0,
            "loss_qty": float(_cpu[1]),
            "loss_aux": float(_cpu[2]),
            "q_qty_mean": float(_cpu[3]),
            "epsilon": self.epsilon,
            "q_value/mean": float(_cpu[4]),
            "q_value/std": float(_cpu[5]),
            "q_value/max": float(_cpu[6]),
            "q_value/min": float(_cpu[7]),
            "q_value/target_mean": float(_cpu[8]),
            "q_value/target_max": float(_cpu[9]),
            "q_value/target_min": float(_cpu[10]),
            "exploration_mode": 2.0,
        }
        if self.multi_horizon:
            metrics["q_value/short_mean"] = float(_cpu[11])
            metrics["q_value/long_mean"] = float(_cpu[12])
            metrics["loss_short"] = float(_cpu[13])
            metrics["loss_long"] = float(_cpu[14])
        if self.use_per:
            metrics["per_beta"] = self.memory.beta
            metrics["per_max_priority"] = self.memory._max_priority

        return metrics

    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy mode; no-op for NoisyNets."""
        self.step_count += 1
        if self._exploration_mode == "epsilon":
            # Linear decay from _epsilon_start to epsilon_end over _epsilon_decay_steps
            frac = min(self.step_count / max(self._epsilon_decay_steps, 1), 1.0)
            self.epsilon = self._epsilon_start + frac * (self.epsilon_end - self._epsilon_start)

    @torch.no_grad()
    def _polyak_update(self):
        """Vectorized Polyak averaging — identical to BDQ."""
        tau = self.tau
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.lerp_(p.data, tau)

    def _reset_noise(self):
        """Reset noise on policy network only.

        PERF-OPT S154 (O3): Target net is permanently in .eval() mode, so
        NoisyLinear.forward() uses only mu_weight/mu_bias and ignores noise
        buffers entirely.  Resetting target noise was pure waste (~80-160
        kernel dispatches per call).
        """
        net = self.policy_net._orig_mod if self._torch_compiled else self.policy_net
        net.reset_noise()

    @staticmethod
    def _strip_compile_prefix(state_dict):
        """Strip '_orig_mod.' prefix added by torch.compile."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save(self, path: str):
        policy_sd = self._strip_compile_prefix(self.policy_net.state_dict())
        target_sd = self._strip_compile_prefix(self.target_net.state_dict())
        ckpt = {
            "policy_net": policy_sd,
            "target_net": target_sd,
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "agent_type": "iqn",
        }
        if hasattr(self, "_lr_scheduler") and self._lr_scheduler is not None:
            ckpt["lr_scheduler"] = self._lr_scheduler.state_dict()
        if self.use_amp:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str, strict: bool = True):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        policy_sd = self._strip_compile_prefix(checkpoint["policy_net"])
        target_sd = self._strip_compile_prefix(checkpoint["target_net"])
        missing_p, unexpected_p = self.policy_net.load_state_dict(policy_sd, strict=strict)
        missing_t, unexpected_t = self.target_net.load_state_dict(target_sd, strict=strict)
        if missing_p or unexpected_p:
            logger.warning(f"[Warm-Start] Policy net — missing: {len(missing_p)}, unexpected: {len(unexpected_p)}")
        if missing_t or unexpected_t:
            logger.warning(f"[Warm-Start] Target net — missing: {len(missing_t)}, unexpected: {len(unexpected_t)}")
        if strict:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 0.0)
        if "step_count" in checkpoint:
            self.step_count = checkpoint["step_count"]
        if "lr_scheduler" in checkpoint and self._lr_scheduler is not None and strict:
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "scaler" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])
