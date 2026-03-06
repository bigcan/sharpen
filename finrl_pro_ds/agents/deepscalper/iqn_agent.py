"""
IQN (Implicit Quantile Network) Agent — Drop-in replacement for DeepScalperBDQ.

Addresses all 4 root causes of RL failure:
1. Q-mean can't resolve 0.3 bps signal → IQN learns full return distribution
2. Epsilon-greedy catastrophic for 95% Hold-optimal → NoisyNets (learned exploration)
3. 43K-step episodes destroy credit assignment → Daily episodes (env-side)
4. Hold domination in replay → Stratified sampling (trainer-side)

Interface is identical to DeepScalperBDQ for trainer/pipeline compatibility.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

from finrl_pro_ds.agents.deepscalper.iqn_network import IQNNetwork
from finrl_pro_ds.agents.deepscalper.per_buffer import PrioritizedReplayBuffer
from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer


class IQNAgent:
    """Implicit Quantile Network agent with NoisyNet exploration.

    Drop-in replacement for DeepScalperBDQ. Exposes identical interface:
    predict(), train_step(), save(), load(), reset_hidden_state(),
    mask_hidden_state(), and all required properties.
    """

    def __init__(
        self,
        network_config: Dict,
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
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        torch_compile: bool = False,
        action_dims=3,
        device: str = "cpu",
        # Absorb BDQ-specific kwargs for config compatibility
        **kwargs,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_quantiles = num_quantiles
        self.kappa = quantile_huber_kappa
        self.gradient_clip = gradient_clip
        self.target_q_clip = target_q_clip
        self.auxiliary_weight = auxiliary_weight
        self.use_amp = use_amp
        self.use_per = use_per

        # IQN has no epsilon — NoisyNets handle exploration
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
                print("[torch.compile] IQN policy_net + target_net compiled (mode=default)")
            except Exception as e:
                print(f"[torch.compile] Failed, falling back to eager mode: {e}")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self._lr_scheduler = None  # Initialized by trainer

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler(device=str(self.device), enabled=self.use_amp)

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
    ) -> np.ndarray:
        """Select action by averaging over quantile Q-values.

        NoisyNets provide exploration (no epsilon needed).
        In eval/deterministic mode, noise is suppressed by NoisyLinear.eval().
        """
        micro = micro.to(self.device, non_blocking=True)
        private_in = private_in.to(self.device, non_blocking=True)
        macro = macro.to(self.device, non_blocking=True)

        if deterministic:
            self.policy_net.eval()

        with torch.no_grad():
            # Sample quantile fractions
            batch_size = micro.shape[0]
            tau = torch.rand(batch_size, self.num_quantiles, device=self.device)

            q_tau, _, new_hidden = self.policy_net(
                micro, private_in, macro, tau, hidden=self._hidden_state
            )
            self._hidden_state = new_hidden

            # Mean over quantiles -> (B, n_actions)
            q_mean = q_tau.mean(dim=1)

            # Apply action mask if provided
            if qty_mask is not None:
                qty_mask_t = torch.as_tensor(qty_mask, dtype=torch.float32, device=self.device)
                if qty_mask_t.dim() == 1:
                    qty_mask_t = qty_mask_t.unsqueeze(0).expand(batch_size, -1)
                q_mean = q_mean.masked_fill(qty_mask_t == 0, float("-inf"))

            actions = q_mean.argmax(dim=-1)  # (B,)

        if deterministic:
            self.policy_net.train()
        else:
            # Reset noise after each prediction step (for fresh exploration)
            self._reset_noise()

        return actions.cpu().numpy()

    def _to_device_pinned(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        """Transfer numpy array to GPU via pinned memory for async DMA."""
        t = torch.as_tensor(arr, dtype=dtype)
        if self._use_pinned:
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Double-DQN style IQN update with quantile Huber loss."""
        self.policy_net.train()
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        if self.use_per:
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
        else:
            # Stratified sampling: over-represent non-Hold transitions
            if self.stratified_sampling and hasattr(self.memory, "sample_stratified"):
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = (
                    self.memory.sample_stratified(
                        self.batch_size,
                        hold_action=self.stratified_hold_action,
                        hold_ratio=self.stratified_hold_ratio,
                    )
                )
            else:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = self.memory.sample(self.batch_size)
            per_indices = None
            is_weights_t = None

            micro_s = self._to_device_pinned(state_batch["micro"])
            private_s = self._to_device_pinned(state_batch["private"])
            macro_s = self._to_device_pinned(state_batch["macro"])
            micro_ns = self._to_device_pinned(next_state_batch["micro"])
            private_ns = self._to_device_pinned(next_state_batch["private"])
            macro_ns = self._to_device_pinned(next_state_batch["macro"])
            actions = self._to_device_pinned(action_batch, dtype=torch.long)
            rewards = self._to_device_pinned(reward_batch)
            dones = self._to_device_pinned(done_batch)
            aux_targets = self._to_device_pinned(aux_target_batch).unsqueeze(1)

        B = micro_s.shape[0]
        N = self.num_quantiles
        N_prime = self.num_quantiles

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            # Sample quantile fractions for policy and target
            tau = torch.rand(B, N, device=self.device)           # policy quantiles
            tau_prime = torch.rand(B, N_prime, device=self.device)  # target quantiles

            # Policy net: Q_tau(s, tau) -> (B, N, n_actions)
            q_tau_all, pred_vol, _ = self.policy_net(micro_s, private_s, macro_s, tau)

            # Select taken action: (B, N, n_actions) -> (B, N)
            actions_flat = actions.view(B, 1) if actions.dim() == 1 else actions[:, 0:1]
            q_tau_a = q_tau_all.gather(2, actions_flat.unsqueeze(1).expand(-1, N, -1)).squeeze(2)  # (B, N)

            with torch.no_grad():
                # Double DQN: policy net selects best action for next state
                q_next_policy, _, _ = self.policy_net(micro_ns, private_ns, macro_ns, tau_prime)
                a_star = q_next_policy.mean(dim=1).argmax(dim=-1, keepdim=True)  # (B, 1)

                # Target net evaluates those actions
                q_next_target, _, _ = self.target_net(micro_ns, private_ns, macro_ns, tau_prime)
                q_target_a = q_next_target.gather(
                    2, a_star.unsqueeze(1).expand(-1, N_prime, -1)
                ).squeeze(2)  # (B, N')

                # Bellman target: r + gamma * (1-done) * Q_target
                r = rewards.view(B, 1)
                d = dones.view(B, 1)
                T_tau = r + self.gamma * (1.0 - d) * q_target_a  # (B, N')

                # Clip target Q-values to prevent divergence (matches BDQ's target_q_clip)
                if self.target_q_clip > 0:
                    T_tau = T_tau.clamp(-self.target_q_clip, self.target_q_clip)

            # Quantile Huber loss
            # delta: (B, N, N') = T_tau[:, None, :] - Q_tau_a[:, :, None]
            delta = T_tau.unsqueeze(1) - q_tau_a.unsqueeze(2)  # (B, N, N')

            # Huber loss element-wise
            abs_delta = delta.abs()
            kappa = self.kappa
            huber = torch.where(
                abs_delta <= kappa,
                0.5 * delta.pow(2) / kappa,
                abs_delta - 0.5 * kappa,
            )

            # Asymmetric weighting: |tau - I(delta < 0)|
            tau_expanded = tau.unsqueeze(2)  # (B, N, 1)
            weight = (tau_expanded - (delta < 0).float()).abs()  # (B, N, N')

            # Loss per sample: mean over N (policy quantiles), mean over N' (target quantiles)
            # FIX J-01: was .sum(dim=1) over N — loss scaled linearly with num_quantiles,
            # breaking HPO across different N values and causing gradient explosions at large N.
            quantile_loss = (weight * huber).mean(dim=1).mean(dim=1)  # (B,)

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
            print(f"WARNING: IQN Loss is {total_loss.item()} (NaN/Inf). Skipping update.", flush=True)
            return None

        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
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

        # Metrics
        metrics = {
            "loss_total": total_loss.item(),
            "loss_price": 0.0,  # No price branch in IQN
            "loss_qty": main_loss.item(),
            "loss_aux": vol_loss.item(),
            "q_qty_mean": q_tau_a.mean().item(),
            "epsilon": 0.0,
            "q_value/mean": q_tau_a.mean().item(),
            "q_value/std": q_tau_a.std().item(),
            "q_value/max": q_tau_a.max().item(),
            "q_value/min": q_tau_a.min().item(),
            "q_value/target_mean": T_tau.mean().item(),
            "q_value/target_max": T_tau.max().item(),
            "q_value/target_min": T_tau.min().item(),
            "exploration_mode": 2.0,  # Sentinel for NoisyNet
        }
        if self.use_per:
            metrics["per_beta"] = self.memory.beta
            metrics["per_max_priority"] = self.memory._max_priority

        return metrics

    def decay_epsilon(self):
        """No-op for IQN — NoisyNets handle exploration. Increments step_count for compatibility."""
        self.step_count += 1

    @torch.no_grad()
    def _polyak_update(self):
        """Vectorized Polyak averaging — identical to BDQ."""
        tau = self.tau
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.lerp_(p.data, tau)

    def _reset_noise(self):
        """Reset noise on both policy and target networks."""
        net = self.policy_net._orig_mod if self._torch_compiled else self.policy_net
        tgt = self.target_net._orig_mod if self._torch_compiled else self.target_net
        net.reset_noise()
        tgt.reset_noise()

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

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        policy_sd = self._strip_compile_prefix(checkpoint["policy_net"])
        target_sd = self._strip_compile_prefix(checkpoint["target_net"])
        self.policy_net.load_state_dict(policy_sd)
        self.target_net.load_state_dict(target_sd)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 0.0)
        if "step_count" in checkpoint:
            self.step_count = checkpoint["step_count"]
        if "lr_scheduler" in checkpoint and self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "scaler" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])
