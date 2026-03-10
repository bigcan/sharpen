"""
SAC (Soft Actor-Critic) Agent — Continuous Position Control

Entropy-regularized actor-critic for Box(-1,1) position fraction.
Twin Q-networks, automatic entropy coefficient tuning, Polyak-averaged targets.

Interface matches IQN/BDQ contract for pipeline compatibility:
  predict(), train_step(), save(), load(), reset_hidden_state(),
  mask_hidden_state(), decay_epsilon()
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

from finrl_pro_ds.agents.sac.networks import SACActorNetwork, SACCriticNetwork
from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer


class SACAgent:
    """Soft Actor-Critic agent for continuous position control.

    Drop-in compatible with the DeepScalper pipeline. Exposes identical
    interface to IQNAgent/DeepScalperBDQ.
    """

    def __init__(
        self,
        network_config: Dict,
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

        # Window/feature config for replay buffer shapes
        window_size = network_config.get("window_size", 30)
        features_per_scale = scale_cfg.get("input_size", 7)
        self._window_size = window_size
        self._features_per_scale = features_per_scale
        self._n_scales = network_config.get("n_scales", 3)

        # Build networks
        self.actor = SACActorNetwork(scale_cfg, private_dim, fusion_dim, self._n_scales).to(self.device)
        self.critic1 = SACCriticNetwork(scale_cfg, private_dim, fusion_dim, n_scales=self._n_scales).to(self.device)
        self.critic2 = SACCriticNetwork(scale_cfg, private_dim, fusion_dim, n_scales=self._n_scales).to(self.device)

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
            torch.log(torch.tensor(initial_alpha, dtype=torch.float32, device=self.device))
        )
        self.target_entropy = -1.0  # -dim(action_space)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay buffer
        # Store scale_0 as "micro" (W,F), concat of remaining scales as "macro" (flat),
        # private as "private" (private_dim,)
        macro_flat_dim = window_size * features_per_scale * (self._n_scales - 1)
        self.replay_buffer = FlatReplayBuffer(
            capacity=buffer_size,
            micro_shape=(window_size, features_per_scale),
            macro_shape=(macro_flat_dim,),
            private_shape=(private_dim,),
            action_shape=(1,),
            action_dtype=np.float32,
        )

        # AMP scaler — disabled for BF16 (same dynamic range as FP32, no scaling needed)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=use_amp and not self._use_bf16 and self.device.type == 'cuda',
        )

        # Optional torch.compile
        # FIX R4-AUD-04: Also compile target critics for consistent JIT perf
        # during target value computation. Polyak lerp_ operates on raw .data,
        # unaffected by compile wrappers.
        if torch_compile and hasattr(torch, 'compile'):
            try:
                self.actor = torch.compile(self.actor, mode='default')
                self.critic1 = torch.compile(self.critic1, mode='default')
                self.critic2 = torch.compile(self.critic2, mode='default')
                self.target_critic1 = torch.compile(self.target_critic1, mode='default')
                self.target_critic2 = torch.compile(self.target_critic2, mode='default')
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"torch.compile failed: {e}")

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def predict(
        self,
        scale_tensors: list,
        private: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Predict action given multi-scale observations.

        Args:
            scale_tensors: list of N tensors, each (B, W, F)
            private: (B, private_dim)
            deterministic: Use mean action (no sampling)

        Returns:
            actions: (B, 1) continuous position fraction
        """
        with torch.no_grad():
            action, _ = self.actor.sample(
                scale_tensors, private,
                deterministic=deterministic,
            )
        return action

    def store_transition(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ):
        """Store a single transition in the replay buffer."""
        state = self._obs_to_buffer(obs)
        next_state = self._obs_to_buffer(next_obs)
        self.replay_buffer.push(
            state, action, reward, next_state, done, aux_target=0.0
        )

    def store_batch(
        self,
        obs_batch: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_batch: Dict[str, np.ndarray],
        dones: np.ndarray,
    ):
        """Store N transitions at once using batch push."""
        states = self._obs_batch_to_buffer(obs_batch)
        next_states = self._obs_batch_to_buffer(next_obs_batch)
        aux = np.zeros(len(rewards), dtype=np.float32)
        self.replay_buffer.push_batch(
            states, actions, rewards, next_states, dones, aux
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """One SAC update step. Returns metrics dict or None if buffer too small."""
        if len(self.replay_buffer) < self.learning_starts:
            return None

        # Sample batch
        states, actions_np, rewards_np, next_states, dones_np, _ = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors — dynamic scale unpacking
        scale_tensors, next_scale_tensors = self._unpack_buffer_to_scales(states, next_states)
        priv = torch.tensor(states["private"], dtype=torch.float32).to(self.device, non_blocking=True)

        actions = torch.tensor(actions_np, dtype=torch.float32).to(self.device, non_blocking=True)
        rewards = torch.tensor(rewards_np, dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)
        dones = torch.tensor(dones_np, dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)

        npriv = torch.tensor(next_states["private"], dtype=torch.float32).to(self.device, non_blocking=True)

        alpha = self.log_alpha.exp().detach()
        amp_ctx = torch.amp.autocast(
            'cuda', dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == 'cuda',
        )

        # --- Critic update ---
        with torch.no_grad():
            with amp_ctx:
                next_action, next_log_prob = self.actor.sample(next_scale_tensors, npriv)
                target_q1 = self.target_critic1(next_scale_tensors, npriv, next_action)
                target_q2 = self.target_critic2(next_scale_tensors, npriv, next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                target_value = rewards + (1.0 - dones) * self.gamma * target_q

        with amp_ctx:
            q1 = self.critic1(scale_tensors, priv, actions)
            q2 = self.critic2(scale_tensors, priv, actions)
            critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        if self.scaler.is_enabled():
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
        else:
            critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.gradient_clip,
        )
        if self.scaler.is_enabled():
            self.scaler.step(self.critic_optimizer)
        else:
            self.critic_optimizer.step()

        # --- Actor update ---
        with amp_ctx:
            new_action, log_prob = self.actor.sample(scale_tensors, priv)
            q1_new = self.critic1(scale_tensors, priv, new_action)
            q2_new = self.critic2(scale_tensors, priv, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        if self.scaler.is_enabled():
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
        else:
            actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        if self.scaler.is_enabled():
            self.scaler.step(self.actor_optimizer)
        else:
            self.actor_optimizer.step()

        # --- Alpha update (float32 — no AMP) ---
        alpha_loss = -(self.log_alpha * (log_prob.float() + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Scaler update (no-op if disabled for BF16)
        self.scaler.update()

        # --- Target Polyak update (fused lerp_) ---
        with torch.no_grad():
            for tp, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                tp.data.lerp_(p.data, self.tau)
            for tp, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": -log_prob.mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    def _obs_to_buffer(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert multi-scale obs dict to replay buffer format."""
        macro_parts = [obs[f"scale_{i}"].flatten() for i in range(1, self._n_scales)]
        return {
            "micro": obs["scale_0"],
            "macro": np.concatenate(macro_parts) if macro_parts else np.array([], dtype=np.float32),
            "private": obs["private"],
        }

    def _obs_batch_to_buffer(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert batch multi-scale obs dict to replay buffer format."""
        n = obs["scale_0"].shape[0]
        macro_parts = [obs[f"scale_{i}"].reshape(n, -1) for i in range(1, self._n_scales)]
        return {
            "micro": obs["scale_0"],
            "macro": np.concatenate(macro_parts, axis=1) if macro_parts else np.zeros((n, 0), dtype=np.float32),
            "private": obs["private"],
        }

    def _unpack_buffer_to_scales(self, states, next_states):
        """Unpack replay buffer micro/macro into list of scale tensors."""
        chunk = self._window_size * self._features_per_scale

        # Scale 0 is stored as "micro"
        s0 = torch.tensor(states["micro"], dtype=torch.float32).to(self.device, non_blocking=True)
        ns0 = torch.tensor(next_states["micro"], dtype=torch.float32).to(self.device, non_blocking=True)
        scale_tensors = [s0]
        next_scale_tensors = [ns0]

        # Remaining scales are packed in "macro"
        if self._n_scales > 1:
            macro = states["macro"]
            nmacro = next_states["macro"]
            for i in range(1, self._n_scales):
                offset = (i - 1) * chunk
                flat = macro[:, offset:offset + chunk]
                nflat = nmacro[:, offset:offset + chunk]
                scale_tensors.append(
                    torch.tensor(
                        flat.reshape(-1, self._window_size, self._features_per_scale),
                        dtype=torch.float32,
                    ).to(self.device, non_blocking=True)
                )
                next_scale_tensors.append(
                    torch.tensor(
                        nflat.reshape(-1, self._window_size, self._features_per_scale),
                        dtype=torch.float32,
                    ).to(self.device, non_blocking=True)
                )

        return scale_tensors, next_scale_tensors

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
