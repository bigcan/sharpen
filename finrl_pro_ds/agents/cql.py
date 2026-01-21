"""CQL Agent implementation for FinRL Pro."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from finrl_pro_ds.agents.sac import SACAgent


class CQLAgent(SACAgent):
    """Conservative Q-Learning (CQL) Agent.
    
    Inherits from SACAgent and adds the conservative loss term.
    Reference: https://arxiv.org/abs/2006.04779
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cql_weight: float = 1.0,
        temp: float = 1.0,
        min_q_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(state_dim, action_dim, **kwargs)
        self.cql_weight = cql_weight
        self.temp = temp
        self.min_q_weight = min_q_weight

    def update(self) -> Dict[str, float]:
        if self.replay_buffer.size < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(batch["state"]).to(self.device)
        action = torch.FloatTensor(batch["action"]).to(self.device)
        next_state = torch.FloatTensor(batch["next_state"]).to(self.device)
        reward = torch.FloatTensor(batch["reward"]).to(self.device)
        not_done = torch.FloatTensor(batch["not_done"]).to(self.device)

        # --- Critic Update (with CQL) ---
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        bellman_error = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # CQL Loss Calculation
        # We need to estimate log(sum(exp(Q(s, a')))) for random actions a'
        # Ideally we sample multiple actions per state.
        # For simplicity in this MVP, we sample 10 random actions + current policy actions.
        
        num_random = 10
        random_actions = torch.FloatTensor(self.batch_size, num_random, self.action_dim).uniform_(-1, 1).to(self.device)
        curr_actions, curr_log_pis, _ = self.actor.sample(state) # (batch, action_dim)
        curr_actions = curr_actions.unsqueeze(1) # (batch, 1, action_dim)
        next_actions, next_log_pis, _ = self.actor.sample(next_state)
        next_actions = next_actions.unsqueeze(1)
        
        # Concatenate all candidate actions: random, current policy, next policy
        # Shape: (batch, num_candidates, action_dim)
        # Note: Full CQL implementation is more complex. 
        # Here we implement a simplified version: penalize random actions vs data actions.
        
        # Q values for random actions
        # Reshape state for broadcasting: (batch, 1, state_dim) -> (batch, num_random, state_dim)
        state_rep = state.unsqueeze(1).repeat(1, num_random, 1)
        # Flatten for critic: (batch * num_random, ...)
        flat_state = state_rep.view(-1, self.state_dim)
        flat_actions = random_actions.view(-1, self.action_dim)
        
        q1_rand, q2_rand = self.critic(flat_state, flat_actions)
        q1_rand = q1_rand.view(self.batch_size, num_random, 1)
        q2_rand = q2_rand.view(self.batch_size, num_random, 1)
        
        # LogSumExp of Q-values for random actions (push these down)
        cql_loss1 = torch.logsumexp(q1_rand / self.temp, dim=1).mean() * self.min_q_weight
        cql_loss2 = torch.logsumexp(q2_rand / self.temp, dim=1).mean() * self.min_q_weight
        
        # Q-values for data actions (push these up)
        cql_loss1 -= current_Q1.mean() * self.min_q_weight
        cql_loss2 -= current_Q2.mean() * self.min_q_weight
        
        cql_loss = (cql_loss1 + cql_loss2) * self.cql_weight

        total_critic_loss = bellman_error + cql_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update (Standard SAC) ---
        # Note: Some CQL variants modify actor update too, but standard SAC actor is fine.
        
        new_action, log_prob, _ = self.actor.sample(state)
        Q1_new, Q2_new = self.critic(state, new_action)
        Q_new = torch.min(Q1_new, Q2_new)
        
        actor_loss = (self.alpha * log_prob - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha Update ---
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft Updates ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": bellman_error.item(),
            "cql_loss": cql_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }
