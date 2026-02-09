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
        action: [dir, price, vol]
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
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        auxiliary_weight: float = 0.1, # Section 4.4
        action_dims: Tuple[int, int, int] = (3, 5, 5),  # FIX: Configurable action dims
        use_amp: bool = False,
        use_per: bool = False,          # Paper Section 4.3: Prioritized Experience Replay
        per_alpha: float = 0.6,         # Prioritization exponent (0=uniform, 1=full)
        per_beta_start: float = 0.4,    # Initial IS correction
        per_beta_frames: int = 100000,  # Anneal beta to 1.0 over this many frames
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.use_per = use_per
        self.auxiliary_weight = auxiliary_weight
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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
        net_action_dims = network_config.get('action_space_dims', (3, 5, 5))
        assert tuple(self.action_dims) == tuple(net_action_dims), (
            f"Action dim mismatch: agent={self.action_dims}, network={net_action_dims}"
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
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
            window_size = micro_cfg.get("window_size", 50)
            micro_input = micro_cfg.get("input_size", 27)
            private_input = micro_cfg.get("private_input_size", 2)
            macro_input = macro_cfg.get("input_size", 11)
            self.memory = FlatReplayBuffer(
                capacity=buffer_size,
                micro_shape=(window_size, micro_input),
                macro_shape=(macro_input,),
                private_shape=(window_size, private_input),  # FIX: env returns private_window (W, 2), not flat (2,)
                action_shape=(len(action_dims),),
            )
            est_gb = self.memory.nbytes() / (1024**3)
            if est_gb > 8:
                import warnings
                warnings.warn(
                    f"FlatReplayBuffer capacity={buffer_size} pre-allocated {est_gb:.1f}GB RAM.",
                    ResourceWarning
                )

    def get_probs(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return action probabilities via temperature-scaled softmax of Q-values.
        Includes numerical stability fix (subtract max).
        """
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        # FIX FIND-4: Disable dropout during inference
        self.policy_net.eval()
        with torch.no_grad():
            q_dir, q_price, q_vol, _, _ = self.policy_net(micro, private_in, macro)
            
            def safe_softmax(q, t):
                # Subtract max for numerical stability to prevent overflow
                q_scaled = (q - q.max(dim=1, keepdim=True)[0]) / t
                return torch.softmax(q_scaled, dim=1)
            
            p_dir = safe_softmax(q_dir, temp)
            p_price = safe_softmax(q_price, temp)
            p_vol = safe_softmax(q_vol, temp)
            
        self.policy_net.train()  # Restore train mode
        return p_dir, p_price, p_vol

    def predict(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Select action using Epsilon-Greedy strategy.
        Supports Batch Input.
        Input shapes:
        micro: (B, Window, Features)
        macro: (B, Features)
        Returns: (B, 3) numpy array
        """
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        batch_size = micro.shape[0]
        
        # FIX FIND-4: Disable dropout during inference
        self.policy_net.eval()
        
        # Epsilon-Greedy Mask
        # We need independent random choices for each item in batch if we were doing true vector env exploration
        # But commonly we just use the same epsilon check or per-env check.
        # For efficiency, we can generate a mask.
        
        if not deterministic:
            rand_vals = torch.rand(batch_size, device=self.device)
            random_mask = rand_vals < self.epsilon
        else:
            random_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            
        # Get Net Actions (Greedy)
        with torch.no_grad():
            q_dir, q_price, q_vol, _, _ = self.policy_net(micro, private_in, macro)
            
            # (B,) indices 
            a_dir_greedy = q_dir.argmax(dim=1)
            a_price_greedy = q_price.argmax(dim=1)
            a_vol_greedy = q_vol.argmax(dim=1)
            
            # Stack: (B, 3)
            greedy_actions = torch.stack([a_dir_greedy, a_price_greedy, a_vol_greedy], dim=1)
            
        # If any random, generate random actions for ALL (simplest) then mask, OR just fills
        if random_mask.any():
            # Generate random actions for the whole batch (wasteful but vector-friendly)
            # or just for masked ones.
            # Torch doesn't have randint for different ranges per column easily in one go if dims differ.
            # But dims are fixed: 3, 5, 5.
            
            # Random Actions: (B, 3)
            r_dir = torch.randint(0, self.action_dims[0], (batch_size,), device=self.device)
            r_price = torch.randint(0, self.action_dims[1], (batch_size,), device=self.device)
            r_vol = torch.randint(0, self.action_dims[2], (batch_size,), device=self.device)
            
            random_actions = torch.stack([r_dir, r_price, r_vol], dim=1)
            
            # Combine
            # Where mask is true, use random. Else greedy.
            # mask is (B,) -> unsqueeze to (B,1)
            mask_expanded = random_mask.unsqueeze(1).expand(-1, 3)
            final_actions = torch.where(mask_expanded, random_actions, greedy_actions)
        else:
            final_actions = greedy_actions
        
        self.policy_net.train()  # FIX FIND-4: Restore train mode (parity with get_probs)
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
        
        # Current Q-Values, Loss Computation under autocast
        # FIX H1: Use modern torch.amp.autocast (works on CPU + CUDA)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            q_dir, q_price, q_vol, _, pred_vol = self.policy_net(micro_state, private_state, macro_state)
        
            # Gather Q-values for taken actions
            curr_q_dir = q_dir.gather(1, actions[:, 0].unsqueeze(1))
            curr_q_price = q_price.gather(1, actions[:, 1].unsqueeze(1))
            curr_q_vol = q_vol.gather(1, actions[:, 2].unsqueeze(1))
            
            # FIX D1: Double DQN — use policy net to SELECT best next action,
            # then EVALUATE that action with target net.
            # This reduces Q-value overestimation bias.
            with torch.no_grad():
                # Policy net selects best actions for next state
                next_q_dir_policy, next_q_price_policy, next_q_vol_policy, _, _ = self.policy_net(
                    micro_next, private_next, macro_next
                )
                best_next_dir = next_q_dir_policy.argmax(dim=1, keepdim=True)
                best_next_price = next_q_price_policy.argmax(dim=1, keepdim=True)
                best_next_vol = next_q_vol_policy.argmax(dim=1, keepdim=True)
                
                # Target net evaluates those actions
                next_q_dir_target, next_q_price_target, next_q_vol_target, _, _ = self.target_net(
                    micro_next, private_next, macro_next
                )
                max_next_q_dir = next_q_dir_target.gather(1, best_next_dir)
                max_next_q_price = next_q_price_target.gather(1, best_next_price)
                max_next_q_vol = next_q_vol_target.gather(1, best_next_vol)
                
                # Target = r + gamma * Q_target(s', argmax_a Q_policy(s', a)) * (1 - done)
                target_q_dir = rewards + self.gamma * max_next_q_dir * (1 - dones)
                target_q_price = rewards + self.gamma * max_next_q_price * (1 - dones)
                target_q_vol = rewards + self.gamma * max_next_q_vol * (1 - dones)
                
            # Loss — PER: per-sample Huber, weighted by IS; Uniform: standard mean
            if self.use_per:
                loss_fn = nn.SmoothL1Loss(reduction='none')  # Per-sample losses
                loss_dir_raw = loss_fn(curr_q_dir, target_q_dir)     # (B, 1)
                loss_price_raw = loss_fn(curr_q_price, target_q_price)
                loss_vol_raw = loss_fn(curr_q_vol, target_q_vol)
                
                # IS-weighted mean (Schaul et al. 2016, Eq. 3)
                loss_dir = (loss_dir_raw * is_weights_t).mean()
                loss_price = (loss_price_raw * is_weights_t).mean()
                loss_vol = (loss_vol_raw * is_weights_t).mean()
            else:
                loss_fn = nn.SmoothL1Loss()
                loss_dir = loss_fn(curr_q_dir, target_q_dir)
                loss_price = loss_fn(curr_q_price, target_q_price)
                loss_vol = loss_fn(curr_q_vol, target_q_vol)
            
            loss_vol_pred = nn.MSELoss()(pred_vol, aux_targets)
            
            total_loss = loss_dir + loss_price + loss_vol + self.auxiliary_weight * loss_vol_pred
        
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
                td_dir = (curr_q_dir - target_q_dir).abs()
                td_price = (curr_q_price - target_q_price).abs()
                td_vol = (curr_q_vol - target_q_vol).abs()
                # Mean TD error across 3 action branches per sample
                td_errors = ((td_dir + td_price + td_vol) / 3.0).squeeze(1).cpu().numpy()
            
            # FIX FIND-1: Note on Stale Q-Values
            # We use Q-values from *before* the optimization step to compute TD errors.
            # Ideally, we would re-run the forward pass with the updated network to get
            # fresh Q-values (Schaul et al., 2016). However, that requires a second
            # batch forward pass, doubling computational cost. The approximation of
            # using pre-update Q-values is standard practice (e.g., Stable Baselines3).
            self.memory.update_priorities(per_indices, td_errors)
        
        # NOTE: Epsilon decay moved to dedicated method for decoupling
        # Call decay_epsilon() from trainer after each env step batch
        
        # Update Target Net
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            state_dict = self.policy_net.state_dict()
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.target_net.load_state_dict(clean_state_dict)
            
        # Logging Metrics
        metrics = {
            "loss_total": total_loss.item(),
            "loss_dir": loss_dir.item(),
            "loss_price": loss_price.item(),
            "loss_vol": loss_vol.item(),
            "loss_aux": loss_vol_pred.item(),
            "q_dir_mean": curr_q_dir.mean().item(),
            "q_price_mean": curr_q_price.mean().item(),
            "q_vol_mean": curr_q_vol.mean().item(),
            "epsilon": self.epsilon
        }
        
        if self.use_per:
            metrics["per_beta"] = self.memory.beta
            metrics["per_max_priority"] = self.memory._max_priority
            
        return metrics

    def decay_epsilon(self):
        """Decay epsilon by one step. Call from trainer after each env step batch."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
