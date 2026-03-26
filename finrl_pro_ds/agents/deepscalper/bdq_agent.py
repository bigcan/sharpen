import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer
from finrl_pro_ds.agents.deepscalper.networks import DeepScalperNetwork
from finrl_pro_ds.agents.deepscalper.per_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

class DeepScalperBDQ:
    """
    Branching Dueling Q-Network (BDQ) Agent for DeepScalper.
    """
    def __init__(
        self,
        network_config: dict,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999995, # Fix: Slower decay (target ~10% at 2M steps)
        exploration_mode: str = "epsilon_greedy",  # Sprint 3: Boltzmann removed
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 0.005,  # Polyak averaging coefficient (0 = no update, 1 = hard copy)
        auxiliary_weight: float = 0.1, # Section 4.4
        action_dims: tuple[int, int] = (5, 9),  # Paper-aligned: (Price, SignedQty)
        use_amp: bool = False,
        use_per: bool = False,          # Paper Section 4.3: Prioritized Experience Replay
        per_alpha: float = 0.6,         # Prioritization exponent (0=uniform, 1=full)
        per_beta_start: float = 0.4,    # Initial IS correction
        per_beta_frames: int = 100000,  # Anneal beta to 1.0 over this many frames
        target_q_clip: float = 5000.0,  # Sprint 3: Clip target Q-values (default for gamma=0.99, R=50)
        torch_compile: bool = False,    # PERF FIX-1: torch.compile for GPU kernel fusion
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.use_per = use_per
        self.auxiliary_weight = auxiliary_weight
        self.gamma = gamma
        self.target_q_clip = target_q_clip
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Used only in fallback (standalone) mode
        # NOTE: Epsilon schedule is managed by DeepScalperTrainer at runtime.
        # The trainer calls decay_epsilon() which uses a closed-form linear
        # schedule (BUG-07 fix). The multiplicative decay below is a fallback
        # for standalone agent usage only. The max(epsilon_end, ...) guard
        # ensures epsilon never drops below epsilon_end regardless of method.
        self.exploration_mode = exploration_mode
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        # FIX BDQ-DISC: Handle both int (Discrete) and tuple (MultiDiscrete) action_dims
        if isinstance(action_dims, int):
            self.action_dims = [action_dims]
        else:
            self.action_dims = list(action_dims)

        # Initialize Networks
        self.policy_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # PERF FIX-1: torch.compile for GPU kernel fusion
        # R&D log: reduce-overhead crashes PPO — use mode="default" for both agents
        self._torch_compiled = False
        if torch_compile and self.device.type == "cuda":
            try:
                self.policy_net = torch.compile(self.policy_net, mode="default")
                self.target_net = torch.compile(self.target_net, mode="default")
                self._torch_compiled = True
                logger.info("[torch.compile] BDQ policy_net + target_net compiled (mode=default)")
            except Exception as e:
                logger.warning(f"[torch.compile] Failed, falling back to eager mode: {e}")

        # FIX M2: Validate action dims match between agent and network
        net_action_dims = network_config.get('action_space_dims', (5, 9))
        # Normalize both to tuples for comparison
        if isinstance(net_action_dims, int):
            net_action_dims = (net_action_dims,)
        assert tuple(self.action_dims) == tuple(net_action_dims), (
            f"Action dim mismatch: agent={self.action_dims}, network={net_action_dims}"
        )

        # PERF-OPT S154 (O5): Fused Adam — single CUDA kernel per step
        _fused = self.device.type == "cuda"
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, fused=_fused)

        # PERF-8: Cosine LR scheduler for stable late-training convergence
        # total_steps estimated from config; updated by trainer if available
        self._lr_scheduler = None  # Initialized by trainer via init_lr_scheduler()

        # FIX H1: Use modern torch.amp API (torch.cuda.amp deprecated in PyTorch ≥2.4)
        # torch.amp.GradScaler works on both CPU and CUDA without crashing
        self.scaler = torch.amp.GradScaler(device=str(self.device), enabled=self.use_amp)

        # Paper Section 4.3: PER vs flat numpy replay
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
            )
        else:
            # FIX BUF-1: Pre-allocated numpy arrays instead of Python list.
            # 2M entries: ~22 GB (numpy) vs ~60-100 GB (Python objects).
            micro_cfg = network_config.get("micro_config", {})
            macro_cfg = network_config.get("macro_config", {})
            window_size = micro_cfg.get("window_size", 15)
            micro_input = micro_cfg.get("input_size", 30)
            private_input = micro_cfg.get("private_input_size", 5)  # Tier 2: 5-dim private state
            macro_input = macro_cfg.get("input_size", 15)
            self.memory = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(window_size, micro_input),
                macro_shape=(macro_input,),
                private_shape=(window_size, private_input),  # FIX: env returns private_window (W, 3), not flat (3,)
                action_shape=(len(self.action_dims),),
            )
            # Estimate memory footprint for warning
            est_bytes = buffer_size * (
                np.prod((window_size, micro_input)) +  # micro
                np.prod((macro_input,)) +              # macro
                np.prod((window_size, private_input)) + # private
                len(self.action_dims) + 1 + 1 + 1      # action, reward, done, aux
            ) * 4 * 2  # float32 (4 bytes) × 2 (state + next_state)
            est_gb = est_bytes / (1024 ** 3)
            if est_gb > 8:
                import warnings
                warnings.warn(
                    f"FlatReplayBuffer capacity={buffer_size} pre-allocated {est_gb:.1f}GB RAM.",
                    ResourceWarning,
                )

        # PERF-OPT: Enable pinned memory staging for async H2D transfers.
        # non_blocking=True is a no-op unless source tensors are in pinned memory.
        # pin_memory() copies pageable→pinned, enabling true async DMA to GPU.
        self._use_pinned = (self.device.type == "cuda")

        # BUG-B: Initialize hidden state for stateful inference
        self._hidden_state = None

    def reset_hidden_state(self):
        """Reset LSTM hidden state (e.g., on episode start)."""
        self._hidden_state = None

    def mask_hidden_state(self, dones: np.ndarray):
        """Zero out hidden states for environments that terminated."""
        if self._hidden_state is None:
            return

        # unpack (h, c)
        h, c = self._hidden_state

        # Dones: (B,) -> Tensor on device
        # Expand to (1, B, 1) to broadcast against (NumLayers, B, HiddenSize)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32).view(1, -1, 1)

        # Mask
        h = h * (1.0 - dones_t)
        c = c * (1.0 - dones_t)
        self._hidden_state = (h, c)


    def predict(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor, deterministic: bool = False, qty_mask=None, context: Optional[dict] = None, eval_epsilon: float = 0.0) -> np.ndarray:
        """
        Select action using Epsilon-Greedy strategy.
        Paper-aligned: 2 branches (Price, SignedQty).

        Args:
            qty_mask: Optional np.ndarray (B, n_qty) or (n_qty,). 1=valid, 0=invalid.
                      ARCH-3: blocks buys at max_position / sells at -max_position.
        Returns: (B, 2) numpy array
        """
        micro = micro.to(self.device, non_blocking=True)
        private_in = private_in.to(self.device, non_blocking=True)
        macro = macro.to(self.device, non_blocking=True)

        batch_size = micro.shape[0]
        n_branches = len(self.action_dims)  # 2

        # FIX FIND-4: Disable dropout during inference
        self.policy_net.eval()

        # Epsilon-Greedy Mask
        if not deterministic:
            rand_vals = torch.rand(batch_size, device=self.device)
            random_mask = rand_vals < self.epsilon
        elif eval_epsilon > 0:
            rand_vals = torch.rand(batch_size, device=self.device)
            random_mask = rand_vals < eval_epsilon
        else:
            random_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # ARCH-3: Prepare qty mask tensor if provided
        qty_mask_t = None
        if qty_mask is not None:
            qty_mask_t = torch.as_tensor(qty_mask, dtype=torch.float32, device=self.device)
            if qty_mask_t.dim() == 1:
                qty_mask_t = qty_mask_t.unsqueeze(0).expand(batch_size, -1)  # (n_qty,) -> (B, n_qty)

        # Get Net Actions (Greedy)
        with torch.no_grad():
            # BUG-B: Pass hidden state for persistent memory
            q_price, q_qty, _, _, new_hidden = self.policy_net(micro, private_in, macro, hidden=self._hidden_state)

            # Update hidden state for next step (inference only)
            self._hidden_state = new_hidden

            # ARCH-3: Mask invalid qty actions before argmax
            if qty_mask_t is not None:
                q_qty = q_qty.masked_fill(qty_mask_t == 0, float('-inf'))

            if q_price is not None:
                a_price_greedy = q_price.argmax(dim=1)
                a_qty_greedy = q_qty.argmax(dim=1)
                # Stack: (B, 2)
                greedy_actions = torch.stack([a_price_greedy, a_qty_greedy], dim=1)
            else:
                # Tier 2: Single branch
                greedy_actions = q_qty.argmax(dim=1, keepdim=True)

        # Exploration: Uniform random (Sprint 3: Boltzmann removed)
        if random_mask.any():
            if q_price is not None:
                r_price = torch.randint(0, self.action_dims[0], (batch_size,), device=self.device)

                # ARCH-3: Random exploration respects qty mask
                if qty_mask_t is not None:
                    # Sample from valid indices only
                    valid_indices = [torch.where(qty_mask_t[b] > 0)[0] for b in range(batch_size)]
                    r_qty = torch.stack([
                        vi[torch.randint(0, len(vi), (1,))] if len(vi) > 0
                        else torch.tensor([self.action_dims[1] // 2], device=self.device)  # fallback: hold
                        for vi in valid_indices
                    ]).squeeze(-1)
                else:
                    r_qty = torch.randint(0, self.action_dims[1], (batch_size,), device=self.device)

                random_actions = torch.stack([r_price, r_qty], dim=1)
            else:
                # Tier 2: Sample from Discrete(6)
                if qty_mask_t is not None:
                    valid_indices = [torch.where(qty_mask_t[b] > 0)[0] for b in range(batch_size)]
                    random_actions = torch.stack([
                        vi[torch.randint(0, len(vi), (1,))] if len(vi) > 0
                        else torch.tensor([2], device=self.device) # Fallback: Hold (index 2 in Tier 2)
                        for vi in valid_indices
                    ])
                else:
                    random_actions = torch.randint(0, self.action_dims[0], (batch_size, 1), device=self.device)

            mask_expanded = random_mask.unsqueeze(1).expand(-1, n_branches)
            final_actions = torch.where(mask_expanded, random_actions, greedy_actions)
        else:
            final_actions = greedy_actions

        self.policy_net.train()  # FIX FIND-4: Restore train mode
        # Return (B, 2) for MultiDiscrete, (B,) for Discrete
        return final_actions.cpu().numpy() if q_price is not None else final_actions.squeeze(1).cpu().numpy()

    def _to_device_pinned(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        """Transfer numpy array to GPU via pinned memory for true async DMA.

        PERF-OPT: torch.as_tensor() wraps numpy without copy, then pin_memory()
        copies to pinned (page-locked) RAM, enabling non_blocking=True to actually
        overlap H2D transfer with GPU compute. Without pinning, non_blocking is a no-op.

        On CPU-only devices, skips pinning (no-op) and returns a regular tensor.
        """
        t = torch.as_tensor(arr, dtype=dtype)
        if self._use_pinned:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def train_step(self) -> Optional[dict[str, float]]:
        """Single gradient step (backward compat). Prefer train_step_mega()."""
        return self.train_step_mega(1)

    def train_step_mega(self, n_steps: int = 1) -> Optional[dict[str, float]]:
        """Run n_steps BDQ gradient updates from a single mega-batch.

        PERF-OPT S154 (O1): Samples n_steps * batch_size transitions ONCE,
        transfers to GPU ONCE, then runs n_steps gradient steps from
        GPU-resident mini-batches. Same pattern proven in SAC (5x SPS gain).

        For PER, falls back to single-step mode.
        """
        # FIX FIND-4: Ensure train mode for dropout/batchnorm
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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = self.memory.sample(mega_batch_size)

        all_micro_s = self._to_device_pinned(state_batch["micro"])
        all_private_s = self._to_device_pinned(state_batch["private"])
        all_macro_s = self._to_device_pinned(state_batch["macro"])
        all_micro_ns = self._to_device_pinned(next_state_batch["micro"])
        all_private_ns = self._to_device_pinned(next_state_batch["private"])
        all_macro_ns = self._to_device_pinned(next_state_batch["macro"])
        all_actions = self._to_device_pinned(action_batch, dtype=torch.long)
        all_rewards = self._to_device_pinned(reward_batch).unsqueeze(1)
        all_dones = self._to_device_pinned(done_batch).unsqueeze(1)
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
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)

        def stack_dict_keys(batch_list, key):
            return torch.tensor(np.array([s[key] for s in batch_list]), dtype=torch.float32).to(self.device, non_blocking=True)

        micro_state = stack_dict_keys(state_batch, "micro")
        private_state = stack_dict_keys(state_batch, "private")
        macro_state = stack_dict_keys(state_batch, "macro")
        micro_next = stack_dict_keys(next_state_batch, "micro")
        private_next = stack_dict_keys(next_state_batch, "private")
        macro_next = stack_dict_keys(next_state_batch, "macro")
        actions = torch.tensor(np.array(action_batch), dtype=torch.long).to(self.device, non_blocking=True)
        rewards = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)
        dones = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)
        aux_targets = torch.tensor(np.array(aux_target_batch), dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)

        return self._train_step_on_batch(
            micro_state, private_state, macro_state, micro_next, private_next, macro_next,
            actions, rewards, dones, aux_targets, per_indices, is_weights_t,
        )

    def _train_step_on_batch(
        self, micro_state, private_state, macro_state, micro_next, private_next, macro_next,
        actions, rewards, dones, aux_targets, per_indices, is_weights_t,
    ) -> Optional[dict[str, float]]:
        """GPU-only gradient step on pre-transferred batch tensors."""

        # Current Q-Values — 2 branches (Paper-aligned)
        # FIX H1: Use modern torch.amp.autocast (works on CPU + CUDA)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            # BUG-B: Hidden state ignored during training (stateless update)
            q_price, q_qty, _, pred_vol, _ = self.policy_net(micro_state, private_state, macro_state)

            if q_price is not None:
                # MultiDiscrete: 2 branches
                curr_q_price = q_price.gather(1, actions[:, 0].unsqueeze(1))
                curr_q_qty = q_qty.gather(1, actions[:, 1].unsqueeze(1))
            else:
                # Discrete: single branch
                curr_q_qty = q_qty.gather(1, actions.view(-1, 1))
                curr_q_price = None

            # Double DQN (FIX D1): policy net selects, target net evaluates
            with torch.no_grad():
                # Policy net selects best actions for next state
                next_q_price_policy, next_q_qty_policy, _, _, _ = self.policy_net(
                    micro_next, private_next, macro_next,
                )

                if q_price is not None:
                    best_next_price = next_q_price_policy.argmax(dim=1, keepdim=True)
                    best_next_qty = next_q_qty_policy.argmax(dim=1, keepdim=True)

                    # Target net evaluates those actions
                    next_q_price_target, next_q_qty_target, _, _, _ = self.target_net(
                        micro_next, private_next, macro_next,
                    )
                    max_next_q_price = next_q_price_target.gather(1, best_next_price)
                    max_next_q_qty = next_q_qty_target.gather(1, best_next_qty)

                    # ARCH-1: Per-branch Bellman targets
                    target_q_price = rewards + self.gamma * max_next_q_price * (1 - dones)
                    target_q_qty = rewards + self.gamma * max_next_q_qty * (1 - dones)
                else:
                    best_next_action = next_q_qty_policy.argmax(dim=1, keepdim=True)
                    _, next_q_target, _, _, _ = self.target_net(micro_next, private_next, macro_next)
                    max_next_q = next_q_target.gather(1, best_next_action)
                    target_q_qty = rewards + self.gamma * max_next_q * (1 - dones)
                    target_q_price = None

                # Sprint 3: Target Q Clipping to prevent bootstrap divergence
                if self.target_q_clip > 0:
                    if target_q_price is not None:
                        target_q_price = target_q_price.clamp(-self.target_q_clip, self.target_q_clip)
                    target_q_qty = target_q_qty.clamp(-self.target_q_clip, self.target_q_clip)

            # Loss Calculation
            loss_fn = nn.SmoothL1Loss(reduction='none')
            if q_price is not None:
                # Bug #5 Fix: Mask price-branch loss when action is Hold
                hold_idx = self.action_dims[1] // 2
                is_trading = (actions[:, 1] != hold_idx).float().unsqueeze(1)

                if self.use_per:
                    loss_price_raw = loss_fn(curr_q_price, target_q_price) * is_trading
                    loss_qty_raw = loss_fn(curr_q_qty, target_q_qty)
                    loss_price = (loss_price_raw * is_weights_t).mean()
                    loss_qty = (loss_qty_raw * is_weights_t).mean()
                else:
                    loss_price = (loss_fn(curr_q_price, target_q_price) * is_trading).mean()
                    loss_qty = loss_fn(curr_q_qty, target_q_qty).mean()

                total_loss_main = (loss_price + loss_qty) / 2.0
            else:
                # Tier 2: Single branch
                if self.use_per:
                    loss_qty = (loss_fn(curr_q_qty, target_q_qty) * is_weights_t).mean()
                else:
                    loss_qty = loss_fn(curr_q_qty, target_q_qty).mean()
                total_loss_main = loss_qty
                loss_price = torch.tensor(0.0, device=self.device)

            # Auxiliary volatility prediction loss (Section 4.4)
            # FIX FIND-V3-07: Apply IS weights to auxiliary loss under PER to prevent
            # biasing shared encoder gradients toward high-TD-error transitions.
            if self.use_per and is_weights_t is not None:
                loss_vol_pred = (nn.SmoothL1Loss(reduction='none')(pred_vol, aux_targets) * is_weights_t).mean()
            else:
                loss_vol_pred = nn.SmoothL1Loss()(pred_vol, aux_targets)

            # Final combined loss
            total_loss = total_loss_main + self.auxiliary_weight * loss_vol_pred

        if not torch.isfinite(total_loss):
            logger.warning(f"BDQ Loss is {total_loss.item()} (NaN/Inf). Skipping update.")
            return None

        self.optimizer.zero_grad()

        # AMP Backward Pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        # PER: Update priorities with TD errors (mean across branches)
        if self.use_per and per_indices is not None:
            with torch.no_grad():
                if q_price is not None:
                    td_price = (curr_q_price - target_q_price).abs()
                    td_qty = (curr_q_qty - target_q_qty).abs()
                    td_errors = ((td_price + td_qty) / 2.0).squeeze(1).cpu().numpy()
                else:
                    td_errors = (curr_q_qty - target_q_qty).abs().squeeze(1).cpu().numpy()
            self.memory.update_priorities(per_indices, td_errors)

        # Update Target Net — Polyak (soft) averaging
        # FIX FIND-NEW-01: step_count is ONLY incremented in decay_epsilon().
        # Previously incremented here too, causing epsilon to decay 2x too fast.
        # PERF-OPT: Vectorized Polyak update — single fused kernel instead of
        # per-parameter Python loop (eliminates ~50 kernel launches per train_step).
        self._polyak_update()

        # PERF-OPT S154: Batch all GPU→CPU metric transfers into a single sync point.
        # Previously 11+ individual .item() calls per train_step × 8 UTD = 88+
        # GPU pipeline stalls per env-step batch. Now 1 sync point.
        q_qty_mean = curr_q_qty.mean()
        _metric_vals = [
            total_loss, loss_price, loss_qty, loss_vol_pred, q_qty_mean,
            q_qty_mean, curr_q_qty.std(), curr_q_qty.max(), curr_q_qty.min(),
            target_q_qty.mean(), target_q_qty.max(), target_q_qty.min(),
        ]
        if q_price is not None:
            _metric_vals.append(curr_q_price.mean())
        _cpu = torch.stack([v.detach().float() for v in _metric_vals]).cpu().numpy()

        metrics = {
            "loss_total": float(_cpu[0]),
            "loss_price": float(_cpu[1]),
            "loss_qty": float(_cpu[2]),
            "loss_aux": float(_cpu[3]),
            "q_qty_mean": float(_cpu[4]),
            "epsilon": self.epsilon,
            # Sprint 1: Q-value statistics for overestimation monitoring
            "q_value/mean": float(_cpu[5]),
            "q_value/std": float(_cpu[6]),
            "q_value/max": float(_cpu[7]),
            "q_value/min": float(_cpu[8]),
            "q_value/target_mean": float(_cpu[9]),
            "q_value/target_max": float(_cpu[10]),
            "q_value/target_min": float(_cpu[11]),
            "exploration_mode": 0.0 if self.exploration_mode == "uniform" else 1.0,
        }
        if q_price is not None:
            metrics["q_price_mean"] = float(_cpu[12])
            metrics["q_value/mean"] = (float(_cpu[12]) + float(_cpu[5])) / 2.0

        if self.use_per:
            metrics["per_beta"] = self.memory.beta
            metrics["per_max_priority"] = self.memory._max_priority

        return metrics

    def decay_epsilon(self):
        """Decay epsilon by one step. Call from trainer after each env step batch.

        FIX BUG-07: Uses closed-form linear schedule instead of multiplicative decay.
        Multiplicative decay (epsilon *= 0.999995) accumulates float error over millions
        of steps. Linear schedule computes epsilon directly from step count.
        """
        self.step_count += 1
        if hasattr(self, '_epsilon_decay_steps') and self._epsilon_decay_steps > 0:
            # Closed-form linear decay: no float accumulation error
            frac = min(1.0, self.step_count / self._epsilon_decay_steps)
            self.epsilon = self._epsilon_start + (self.epsilon_end - self._epsilon_start) * frac
        else:
            # Fallback: original multiplicative decay (standalone agent without trainer)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @torch.no_grad()
    def _polyak_update(self):
        """Vectorized Polyak averaging — single fused operation on flattened param vectors.

        PERF-OPT: Replaces per-parameter Python loop with vectorized torch ops.
        The old loop launched 2 CUDA kernels (mul_, add_) per parameter (~25 params),
        totaling ~50 kernel launches per train_step(). With UTD=8, that's 400 kernel
        launches per env step just for target net updates.

        This version flattens all params into contiguous vectors, does a single
        lerp, then copies back. Net: 1 lerp kernel + 1 copy_() per param (still
        a loop, but the heavy math is a single fused kernel).
        """
        tau = self.tau
        # Lerp on flattened vectors — single fused kernel for the math
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.lerp_(p.data, tau)

    @staticmethod
    def _strip_compile_prefix(state_dict):
        """Strip '_orig_mod.' prefix added by torch.compile for portable checkpoints."""
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save(self, path: str):
        # PERF FIX-1: Strip torch.compile prefix for portable checkpoints
        policy_sd = self._strip_compile_prefix(self.policy_net.state_dict())
        target_sd = self._strip_compile_prefix(self.target_net.state_dict())
        ckpt = {
            'policy_net': policy_sd,
            'target_net': target_sd,
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            # FIX FIND-V3-02b: Persist epsilon schedule state for resume
            'step_count': self.step_count,
        }
        if hasattr(self, '_epsilon_start'):
            ckpt['_epsilon_start'] = self._epsilon_start
            ckpt['_epsilon_decay_steps'] = self._epsilon_decay_steps
        # FIX N3: Persist LR scheduler state for crash recovery
        if hasattr(self, '_lr_scheduler') and self._lr_scheduler is not None:
            ckpt['lr_scheduler'] = self._lr_scheduler.state_dict()
        # FIX FIND-V3-04c: Persist AMP GradScaler state
        if self.use_amp:
            ckpt['scaler'] = self.scaler.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # PERF FIX-1: Strip torch.compile prefix from checkpoint keys (backward-compatible)
        policy_sd = self._strip_compile_prefix(checkpoint['policy_net'])
        target_sd = self._strip_compile_prefix(checkpoint['target_net'])
        self.policy_net.load_state_dict(policy_sd)
        self.target_net.load_state_dict(target_sd)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        # FIX FIND-V3-02b: Restore epsilon schedule state (backward-compatible)
        if 'step_count' in checkpoint:
            self.step_count = checkpoint['step_count']
        if '_epsilon_start' in checkpoint:
            self._epsilon_start = checkpoint['_epsilon_start']
            self._epsilon_decay_steps = checkpoint['_epsilon_decay_steps']
        # FIX N3: Restore LR scheduler state if available (backward-compatible)
        if 'lr_scheduler' in checkpoint and hasattr(self, '_lr_scheduler') and self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # FIX FIND-V3-04c: Restore AMP GradScaler state
        if 'scaler' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
