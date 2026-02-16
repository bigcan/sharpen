import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, Any, List, Optional
import os
import copy

from finrl_pro_ds.agents.deepscalper.networks import DeepScalperNetwork
from finrl_pro_ds.agents.deepscalper.per_buffer import PrioritizedReplayBuffer
from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, aux_target=0.0):
        """
        state, next_state: Dict of tensors or np arrays
        action: [price, qty]
        reward: float
        done: bool
        aux_target: float (Volatility Target)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done, aux_target)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        # List sampling is much faster than deque sampling
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, aux_target = zip(*batch)
        return state, action, reward, next_state, done, aux_target
    
    def __len__(self):
        return len(self.buffer)

class DeepScalperBDQ:
    """
    Branching Dueling Q-Network (BDQ) Agent for DeepScalper.
    """
    def __init__(
        self,
        network_config: Dict,
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
        action_dims: Tuple[int, int] = (5, 9),  # Paper-aligned: (Price, SignedQty)
        use_amp: bool = False,
        use_per: bool = False,          # Paper Section 4.3: Prioritized Experience Replay
        per_alpha: float = 0.6,         # Prioritization exponent (0=uniform, 1=full)
        per_beta_start: float = 0.4,    # Initial IS correction
        per_beta_frames: int = 100000,  # Anneal beta to 1.0 over this many frames
        target_q_clip: float = 5000.0,  # Sprint 3: Clip target Q-values (default for gamma=0.99, R=50)
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.use_per = use_per
        self.auxiliary_weight = auxiliary_weight
        self.gamma = gamma
        self.target_q_clip = target_q_clip
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.exploration_mode = exploration_mode
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.action_dims = list(action_dims)  # FIX: Use constructor param
        
        # Initialize Networks
        self.policy_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # FIX M2: Validate action dims match between agent and network
        net_action_dims = network_config.get('action_space_dims', (5, 9))
        assert tuple(self.action_dims) == tuple(net_action_dims), (
            f"Action dim mismatch: agent={self.action_dims}, network={net_action_dims}"
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
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
                beta_frames=per_beta_frames
            )
        else:
            # FIX BUF-1: Pre-allocated numpy arrays instead of Python list.
            # 2M entries: ~22 GB (numpy) vs ~60-100 GB (Python objects).
            micro_cfg = network_config.get("micro_config", {})
            macro_cfg = network_config.get("macro_config", {})
            window_size = micro_cfg.get("window_size", 15)
            micro_input = micro_cfg.get("input_size", 30)
            private_input = micro_cfg.get("private_input_size", 3)
            macro_input = macro_cfg.get("input_size", 15)
            self.memory = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(window_size, micro_input),
                macro_shape=(macro_input,),
                private_shape=(window_size, private_input),  # FIX: env returns private_window (W, 3), not flat (3,)
                action_shape=(len(action_dims),),
            )
            # Estimate memory footprint for warning
            est_bytes = buffer_size * (
                np.prod((window_size, micro_input)) +  # micro
                np.prod((macro_input,)) +              # macro
                np.prod((window_size, private_input)) + # private
                len(action_dims) + 1 + 1 + 1           # action, reward, done, aux
            ) * 4 * 2  # float32 (4 bytes) × 2 (state + next_state)
            est_gb = est_bytes / (1024 ** 3)
            if est_gb > 8:
                import warnings
                warnings.warn(
                    f"FlatReplayBuffer capacity={buffer_size} pre-allocated {est_gb:.1f}GB RAM.",
                    ResourceWarning
                )
        
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


    def predict(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor, deterministic: bool = False, qty_mask=None) -> np.ndarray:
        """
        Select action using Epsilon-Greedy strategy.
        Paper-aligned: 2 branches (Price, SignedQty).
        
        Args:
            qty_mask: Optional np.ndarray (B, n_qty) or (n_qty,). 1=valid, 0=invalid.
                      ARCH-3: blocks buys at max_position / sells at -max_position.
        Returns: (B, 2) numpy array
        """
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        batch_size = micro.shape[0]
        n_branches = len(self.action_dims)  # 2
        
        # FIX FIND-4: Disable dropout during inference
        self.policy_net.eval()
        
        # Epsilon-Greedy Mask
        if not deterministic:
            rand_vals = torch.rand(batch_size, device=self.device)
            random_mask = rand_vals < self.epsilon
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
            
            a_price_greedy = q_price.argmax(dim=1)
            a_qty_greedy = q_qty.argmax(dim=1)
            
            # Stack: (B, 2)
            greedy_actions = torch.stack([a_price_greedy, a_qty_greedy], dim=1)
            
        # Exploration: Uniform random (Sprint 3: Boltzmann removed)
        if random_mask.any():
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
            
            mask_expanded = random_mask.unsqueeze(1).expand(-1, n_branches)
            final_actions = torch.where(mask_expanded, random_actions, greedy_actions)
        else:
            final_actions = greedy_actions
        
        self.policy_net.train()  # FIX FIND-4: Restore train mode
        return final_actions.cpu().numpy()
    
    def train_step(self) -> Optional[Dict[str, float]]:
        # FIX FIND-4: Ensure train mode for dropout/batchnorm
        self.policy_net.train()
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample Batch — PER returns (indices, is_weights) alongside transitions
        if self.use_per:
            (state_batch, action_batch, reward_batch, next_state_batch,
             done_batch, aux_target_batch, per_indices, is_weights) = self.memory.sample(self.batch_size)
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(self.device)  # (B, 1)
            # PER buffer returns tuples of dicts — need list comprehension to stack
            def stack_dict_keys(batch_list, key):
                return torch.tensor(np.array([s[key] for s in batch_list]), dtype=torch.float32).to(self.device)
            micro_state = stack_dict_keys(state_batch, "micro")
            private_state = stack_dict_keys(state_batch, "private")
            macro_state = stack_dict_keys(state_batch, "macro")
            micro_next = stack_dict_keys(next_state_batch, "micro")
            private_next = stack_dict_keys(next_state_batch, "private")
            macro_next = stack_dict_keys(next_state_batch, "macro")
            actions = torch.tensor(np.array(action_batch), dtype=torch.long).to(self.device)
            rewards = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(self.device)
            dones = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(self.device)
            aux_targets = torch.tensor(np.array(aux_target_batch), dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            # FIX BUF-1: FlatReplayBuffer returns pre-stacked numpy arrays (dicts with batch dim)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, aux_target_batch = self.memory.sample(self.batch_size)
            per_indices = None
            is_weights_t = None
            # Zero-copy path: numpy → torch tensor (no list comprehension)
            micro_state = torch.as_tensor(state_batch["micro"], dtype=torch.float32).to(self.device)
            private_state = torch.as_tensor(state_batch["private"], dtype=torch.float32).to(self.device)
            macro_state = torch.as_tensor(state_batch["macro"], dtype=torch.float32).to(self.device)
            micro_next = torch.as_tensor(next_state_batch["micro"], dtype=torch.float32).to(self.device)
            private_next = torch.as_tensor(next_state_batch["private"], dtype=torch.float32).to(self.device)
            macro_next = torch.as_tensor(next_state_batch["macro"], dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(action_batch, dtype=torch.long).to(self.device)
            rewards = torch.as_tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
            dones = torch.as_tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
            aux_targets = torch.as_tensor(aux_target_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q-Values — 2 branches (Paper-aligned)
        # FIX H1: Use modern torch.amp.autocast (works on CPU + CUDA)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            # BUG-B: Hidden state ignored during training (stateless update)
            q_price, q_qty, _, pred_vol, _ = self.policy_net(micro_state, private_state, macro_state)
        
            # Gather Q-values for taken actions: (B, 1)
            curr_q_price = q_price.gather(1, actions[:, 0].unsqueeze(1))
            curr_q_qty = q_qty.gather(1, actions[:, 1].unsqueeze(1))
            
            # Double DQN (FIX D1): policy net selects, target net evaluates
            with torch.no_grad():
                # Policy net selects best actions for next state
                next_q_price_policy, next_q_qty_policy, _, _, _ = self.policy_net(
                    micro_next, private_next, macro_next
                )
                best_next_price = next_q_price_policy.argmax(dim=1, keepdim=True)
                best_next_qty = next_q_qty_policy.argmax(dim=1, keepdim=True)
                
                # Target net evaluates those actions
                next_q_price_target, next_q_qty_target, _, _, _ = self.target_net(
                    micro_next, private_next, macro_next
                )
                max_next_q_price = next_q_price_target.gather(1, best_next_price)
                max_next_q_qty = next_q_qty_target.gather(1, best_next_qty)
                
                # ARCH-1: Per-branch Bellman targets (Independent Q-learning within Dueling BDQ)
                # Decouples price/qty streams to prevent noise propagation
                target_q_price = rewards + self.gamma * max_next_q_price * (1 - dones)
                target_q_qty = rewards + self.gamma * max_next_q_qty * (1 - dones)
                
                # Sprint 3: Target Q Clipping to prevent bootstrap divergence
                if self.target_q_clip > 0:
                    target_q_price = target_q_price.clamp(-self.target_q_clip, self.target_q_clip)
                    target_q_qty = target_q_qty.clamp(-self.target_q_clip, self.target_q_clip)
                
            # Bug #5 Fix: Mask price-branch loss when action is Hold (qty = middle index)
            # During Hold, the price action is irrelevant — masking prevents noisy gradients
            hold_idx = self.action_dims[1] // 2
            is_trading = (actions[:, 1] != hold_idx).float().unsqueeze(1)

            # Loss — 2 branches
            if self.use_per:
                loss_fn = nn.SmoothL1Loss(reduction='none')
                loss_price_raw = loss_fn(curr_q_price, target_q_price) * is_trading
                loss_qty_raw = loss_fn(curr_q_qty, target_q_qty)
                
                loss_price = (loss_price_raw * is_weights_t).mean()
                loss_qty = (loss_qty_raw * is_weights_t).mean()
            else:
                loss_fn = nn.SmoothL1Loss(reduction='none')
                loss_price = (loss_fn(curr_q_price, target_q_price) * is_trading).mean()
                loss_qty = loss_fn(curr_q_qty, target_q_qty).mean()
            
            # Auxiliary volatility prediction loss (Section 4.4)
            loss_vol_pred = nn.SmoothL1Loss()(pred_vol, aux_targets)
            
            # Average branch losses (CRIT-1: balanced V(s) gradient)
            total_loss = (loss_price + loss_qty) / 2.0 + self.auxiliary_weight * loss_vol_pred
        
        if not torch.isfinite(total_loss):
            print(f"WARNING: BDQ Loss is {total_loss.item()} (NaN/Inf). Skipping update.", flush=True)
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
                td_price = (curr_q_price - target_q_price).abs()
                td_qty = (curr_q_qty - target_q_qty).abs()
                # Mean TD error across 2 action branches per sample
                td_errors = ((td_price + td_qty) / 2.0).squeeze(1).cpu().numpy()
            self.memory.update_priorities(per_indices, td_errors)
        
        # Update Target Net — Polyak (soft) averaging
        self.step_count += 1
        with torch.no_grad():
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            
        # Logging Metrics — 2 branches
        metrics = {
            "loss_total": total_loss.item(),
            "loss_price": loss_price.item(),
            "loss_qty": loss_qty.item(),
            "loss_aux": loss_vol_pred.item(),
            "q_price_mean": curr_q_price.mean().item(),
            "q_qty_mean": curr_q_qty.mean().item(),
            "epsilon": self.epsilon,
            # Sprint 1: Q-value statistics for overestimation monitoring
            "q_value/mean": (curr_q_price.mean().item() + curr_q_qty.mean().item()) / 2.0,
            "q_value/std": (curr_q_price.std().item() + curr_q_qty.std().item()) / 2.0,
            "q_value/max": max(curr_q_price.max().item(), curr_q_qty.max().item()),
            "q_value/min": min(curr_q_price.min().item(), curr_q_qty.min().item()),
            "q_value/target_mean": (target_q_price.mean().item() + target_q_qty.mean().item()) / 2.0,
            "q_value/target_max": max(target_q_price.max().item(), target_q_qty.max().item()),
            "q_value/target_min": min(target_q_price.min().item(), target_q_qty.min().item()),
            "exploration_mode": 0.0 if self.exploration_mode == "uniform" else 1.0,
        }
        
        if self.use_per:
            metrics["per_beta"] = self.memory.beta
            metrics["per_max_priority"] = self.memory._max_priority
            
        return metrics

    def decay_epsilon(self):
        """Decay epsilon by one step. Call from trainer after each env step batch."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        ckpt = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        # FIX N3: Persist LR scheduler state for crash recovery
        if hasattr(self, '_lr_scheduler') and self._lr_scheduler is not None:
            ckpt['lr_scheduler'] = self._lr_scheduler.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        # FIX: weights_only=False required for PyTorch ≥2.6 (default changed to True).
        # Our checkpoints contain numpy scalars (epsilon, LR scheduler state) which
        # are rejected by the safe unpickler. These are our own trusted checkpoints.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        # FIX N3: Restore LR scheduler state if available (backward-compatible)
        if 'lr_scheduler' in checkpoint and hasattr(self, '_lr_scheduler') and self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
