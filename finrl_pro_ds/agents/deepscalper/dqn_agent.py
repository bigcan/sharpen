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

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        state, next_state: Dict of tensors or np arrays
        action: [dir, price, vol]
        reward: float
        done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DeepScalperDQN:
    """
    Branching Dueling DQN Agent for DeepScalper.
    """
    def __init__(
        self,
        network_config: Dict,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Initialize Networks
        self.policy_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net = DeepScalperNetwork(**network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Action dimensions (Dir, Price, Vol)
        self.action_dims = [3, 5, 5] 

    def get_probs(self, micro: torch.Tensor, macro: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return action probabilities via temperature-scaled softmax of Q-values.
        Includes numerical stability fix (subtract max).
        """
        micro = micro.to(self.device)
        macro = macro.to(self.device)
        
        with torch.no_grad():
            q_dir, q_price, q_vol, _ = self.policy_net(micro, macro)
            
            def safe_softmax(q, t):
                # Subtract max for numerical stability to prevent overflow
                q_scaled = (q - q.max(dim=1, keepdim=True)[0]) / t
                return torch.softmax(q_scaled, dim=1)
            
            p_dir = safe_softmax(q_dir, temp)
            p_price = safe_softmax(q_price, temp)
            p_vol = safe_softmax(q_vol, temp)
            
        return p_dir, p_price, p_vol

    def predict(self, micro: torch.Tensor, macro: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Select action using Epsilon-Greedy strategy.
        Input shapes:
        micro: (1, Window, Features)
        macro: (1, Features)
        """
        micro = micro.to(self.device)
        macro = macro.to(self.device)
        
        if not deterministic and random.random() < self.epsilon:
            # Random action for each branch
            a_dir = random.randint(0, self.action_dims[0] - 1)
            a_price = random.randint(0, self.action_dims[1] - 1)
            a_vol = random.randint(0, self.action_dims[2] - 1)
            return np.array([a_dir, a_price, a_vol])
        
        with torch.no_grad():
            q_dir, q_price, q_vol, _ = self.policy_net(micro, macro)
            
            a_dir = q_dir.argmax(dim=1).item()
            a_price = q_price.argmax(dim=1).item()
            a_vol = q_vol.argmax(dim=1).item()
            
            return np.array([a_dir, a_price, a_vol])
    
    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample Batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        # Prepare Tensors
        # Assuming state is Dict[str, np.ndarray] or similar, need to collate
        # We need to stack micro and macro components
        # Note: Replay buffer stores list of tuples.
        
        def stack_dict_keys(batch_list, key):
            return torch.tensor(np.array([s[key] for s in batch_list]), dtype=torch.float32).to(self.device)
        
        micro_state = stack_dict_keys(state_batch, "micro")
        macro_state = stack_dict_keys(state_batch, "macro")
        
        micro_next = stack_dict_keys(next_state_batch, "micro")
        macro_next = stack_dict_keys(next_state_batch, "macro")
        
        actions = torch.tensor(np.array(action_batch), dtype=torch.long).to(self.device) # (B, 3)
        rewards = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(self.device) # (B, 1)
        dones = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(self.device) # (B, 1)
        
        # Current Q-Values
        q_dir, q_price, q_vol, _ = self.policy_net(micro_state, macro_state)
        
        # Gather Q-values for taken actions
        # actions[:, 0] is direction indices
        curr_q_dir = q_dir.gather(1, actions[:, 0].unsqueeze(1))
        curr_q_price = q_price.gather(1, actions[:, 1].unsqueeze(1))
        curr_q_vol = q_vol.gather(1, actions[:, 2].unsqueeze(1))
        
        # Target Q-Values (Double DQN Logic could be added here, sticking to standard DQN for now)
        with torch.no_grad():
            next_q_dir, next_q_price, next_q_vol, _ = self.target_net(micro_next, macro_next)
            
            # Max next Q
            max_next_q_dir = next_q_dir.max(1)[0].unsqueeze(1)
            max_next_q_price = next_q_price.max(1)[0].unsqueeze(1)
            max_next_q_vol = next_q_vol.max(1)[0].unsqueeze(1)
            
            # Target = r + gamma * max_next_Q * (1 - done)
            # Branching DQN uses the SAME reward for all branches (common reward)
            target_q_dir = rewards + self.gamma * max_next_q_dir * (1 - dones)
            target_q_price = rewards + self.gamma * max_next_q_price * (1 - dones)
            target_q_vol = rewards + self.gamma * max_next_q_vol * (1 - dones)
            
        # Loss (Huber or MSE)
        loss_fn = nn.MSELoss()
        loss_dir = loss_fn(curr_q_dir, target_q_dir)
        loss_price = loss_fn(curr_q_price, target_q_price)
        loss_vol = loss_fn(curr_q_vol, target_q_vol)
        
        total_loss = loss_dir + loss_price + loss_vol
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update Target Net
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return total_loss.item()

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
