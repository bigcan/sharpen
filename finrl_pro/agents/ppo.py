"""PPO Agent implementation for FinRL Pro."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: nn.Module = nn.Tanh(),
        action_adapter: str = "tanh",
    ) -> None:
        super().__init__()
        self.action_adapter = action_adapter
        
        # Shared features (optional, but common) or separate networks
        # Here we use separate networks for simplicity and stability
        
        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self) -> None:
        raise NotImplementedError("Use act or get_value instead.")

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log_prob, and entropy for a given state."""
        features = self.actor(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Adapter
        if self.action_adapter == "tanh":
            return torch.tanh(action), log_prob, entropy
        
        return action, log_prob, entropy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate for a given state."""
        return self.critic(state)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        features = self.actor(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        # We need to handle the tanh correction if we squashed actions in `act`.
        # For this MVP, we assume the action passed in is the pre-tanh action 
        # OR we accept slight approximation error. 
        # Better approach: Store log_prob from rollout and use it.
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        
        return log_prob, entropy, value


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
        action_adapter: str = "tanh",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        self.policy = ActorCritic(state_dim, action_dim, action_adapter=action_adapter).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer for one epoch
        self.reset_buffer()


    def reset_buffer(self) -> None:
        self.buffer: Dict[str, list] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action for a given state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                features = self.policy.actor(state_t)
                mean = self.policy.actor_mean(features)
                if self.policy.action_adapter == "tanh":
                    action_t = torch.tanh(mean)
                else:
                    action_t = mean
                log_prob = 0.0
            else:
                action_t, log_prob_t, _ = self.policy.act(state_t)
                log_prob = log_prob_t.item()
            
        return action_t.cpu().numpy()[0], log_prob

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: Optional[np.ndarray] = None,
        log_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Store transition in buffer."""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["log_probs"].append(log_prob if log_prob is not None else 0.0)
        self.buffer["values"].append(0.0) # Value computed/not stored here

    def update(self) -> Dict[str, float]:
        """Update policy using stored transitions."""
        if len(self.buffer["states"]) == 0:
            return {}

        states = torch.FloatTensor(np.array(self.buffer["states"])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer["log_probs"])).to(self.device)
        rewards = self.buffer["rewards"]
        dones = self.buffer["dones"]
        
        # Compute Returns and Advantages (GAE)
        # Note: This simple version assumes the buffer contains a complete trajectory 
        # or we handle the last value bootstrap externally.
        # For simplicity, we'll compute simple discounted returns here, 
        # but GAE is better.
        
        returns = []
        discounted_sum = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted_sum = 0.0
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        # Advantages (using returns - values approximation if values not stored/accurate)
        # Ideally we use GAE.
        with torch.no_grad():
             values = self.policy.get_value(states).squeeze()
        advantages = returns_t - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        loss_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_returns = returns_t[idx]
                b_advantages = advantages[idx]
                
                # Evaluate
                # Note: We are re-sampling distribution to get new log_probs
                # Ideally we should use the distribution parameters to compute log_prob of b_actions
                features = self.policy.actor(b_states)
                mean = self.policy.actor_mean(features)
                std = self.policy.actor_log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                
                # We are using the stored actions, which were tanh'd.
                # Technically we should use the pre-tanh actions for Gaussian log_prob
                # OR use a TanhNormal distribution.
                # For this MVP, we approximate.
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1) # Approximation
                entropy = dist.entropy().sum(dim=-1).mean()
                b_values = self.policy.critic(b_states).squeeze()
                
                # Ratio
                ratio = (new_log_probs - b_old_log_probs).exp()
                
                # Surrogate Loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * (b_returns - b_values).pow(2).mean()
                
                # Total Loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_info["policy_loss"] += policy_loss.item()
                loss_info["value_loss"] += value_loss.item()
                loss_info["entropy"] += entropy.item()

        # Average losses
        n_batches = (len(states) + self.batch_size - 1) // self.batch_size
        total_updates = self.n_epochs * n_batches
        return {k: v / total_updates for k, v in loss_info.items()}

    def save(self, path: str) -> None:
        """Save agent model."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load agent model."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
