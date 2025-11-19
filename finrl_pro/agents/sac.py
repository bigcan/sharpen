"""SAC Agent implementation for FinRL Pro."""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


from finrl_pro.agents.buffer import ReplayBuffer


class Actor(nn.Module):
    """Gaussian Policy for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class Critic(nn.Module):
    """Twin Q-Networks."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class SACAgent:
    """Soft Actor-Critic (SAC) Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        batch_size: int = 256,
        buffer_size: int = 100000,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        if self.auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                _, _, action_t = self.actor.sample(state_t)
            else:
                action_t, _, _ = self.actor.sample(state_t)
        return action_t.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self) -> Dict[str, float]:
        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(batch["state"]).to(self.device)
        action = torch.FloatTensor(batch["action"]).to(self.device)
        next_state = torch.FloatTensor(batch["next_state"]).to(self.device)
        reward = torch.FloatTensor(batch["reward"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_done"]).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.gamma * target_Q

        # Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        new_action, log_prob, _ = self.actor.sample(state)
        Q1_new, Q2_new = self.critic(state, new_action)
        Q_new = torch.min(Q1_new, Q2_new)
        
        actor_loss = (self.alpha * log_prob - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha Update
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()

        # Soft Update Target Network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha": self.alpha,
        }

    def save(self, path: str) -> None:
        """Save agent model."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha if self.auto_alpha else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.auto_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
