# Portions of this file are derived from ElegantRL
# (https://github.com/AI4Finance-Foundation/ElegantRL), Copyright 2024
# AI4Finance Foundation Inc., licensed under the Apache License, Version 2.0.
# Modified by Keng Lee, 2026. See the NOTICE file for details.

"""DQN Agent classes for AlphaSeek production training.

Ported from contest/reference/erl_agent.py with production fixes:
- Imports from production .nets and .replay_buffer (no sys.path hacks)
- logging instead of print()
- non_blocking=True on .to(device) calls
- Lightweight AlphaSeekAgentConfig replaces contest Config

Three agent types with different Q-network architectures:
- AgentDoubleDQN: QNetTwin (twin Q-heads, no dueling)
- AgentD3QN: QNetTwinDuel (twin Q-heads + dueling advantage/value)
- AgentTwinD3QN: QNetTwin (identical to DoubleDQN — historical naming)
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .nets import QNetTwin, QNetTwinDuel
from .replay_buffer import AlphaSeekReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class AlphaSeekAgentConfig:
    """Lightweight config for agent construction — replaces contest Config."""

    gamma: float = 0.995
    num_envs: int = 4096
    batch_size: int = 1024
    repeat_times: float = 2.0
    reward_scale: float = 4096.0
    learning_rate: float = 1e-4
    if_off_policy: bool = True
    clip_grad_norm: float = 3.0
    soft_update_tau: float = 5e-4
    state_value_tau: float = 0.01
    explore_rate: float = 0.02
    net_dims: tuple = (256, 256)

    # Env dimensions (set from simulator)
    state_dim: int = 10
    action_dim: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> AlphaSeekAgentConfig:
        """Build config from a flat dict (e.g., merged YAML + HPO params)."""
        net_dims = d.get("net_dims", (256, 256))
        if isinstance(net_dims, str):
            net_dims = tuple(int(x.strip()) for x in net_dims.split(","))
        elif isinstance(net_dims, list):
            net_dims = tuple(net_dims)
        return cls(
            gamma=d.get("gamma", cls.gamma),
            num_envs=d.get("num_envs", cls.num_envs),
            batch_size=d.get("batch_size", cls.batch_size),
            repeat_times=d.get("repeat_times", cls.repeat_times),
            reward_scale=d.get("reward_scale", cls.reward_scale),
            learning_rate=d.get("learning_rate", cls.learning_rate),
            if_off_policy=d.get("if_off_policy", cls.if_off_policy),
            clip_grad_norm=d.get("clip_grad_norm", cls.clip_grad_norm),
            soft_update_tau=d.get("soft_update_tau", cls.soft_update_tau),
            state_value_tau=d.get("state_value_tau", cls.state_value_tau),
            explore_rate=d.get("explore_rate", cls.explore_rate),
            net_dims=net_dims,
            state_dim=d.get("state_dim", cls.state_dim),
            action_dim=d.get("action_dim", cls.action_dim),
        )


class AgentDoubleDQN:
    """Double DQN agent with twin Q-heads and epsilon-greedy exploration.

    Uses QNetTwin architecture: shared encoder → two independent Q-value heads.
    Target network updated via Polyak averaging (soft_update_tau).
    """

    act_class = QNetTwin
    cri_class = None  # means self.cri = self.act

    def __init__(
        self,
        net_dims: tuple[int, ...],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: AlphaSeekAgentConfig | None = None,
    ):
        if args is None:
            args = AlphaSeekAgentConfig()

        self.gamma = args.gamma
        self.num_envs = args.num_envs
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.clip_grad_norm = args.clip_grad_norm
        self.soft_update_tau = args.soft_update_tau
        self.state_value_tau = args.state_value_tau

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu",
        )

        # Build networks
        act_class = getattr(self, "act_class", QNetTwin)
        cri_class = getattr(self, "cri_class", None)

        self.act = act_class(list(net_dims), state_dim, action_dim).to(
            self.device, non_blocking=True,
        )
        self.cri = (
            cri_class(list(net_dims), state_dim, action_dim).to(
                self.device, non_blocking=True,
            )
            if cri_class
            else self.act
        )

        # Target networks (deep copy)
        self.act_target = self.cri_target = deepcopy(self.act)
        self.act.explore_rate = args.explore_rate

        # torch.compile: simple MLP with static shapes — compiles cleanly
        if self.device.type == "cuda":
            self.act = torch.compile(self.act)
            self.act_target = self.cri_target = torch.compile(self.act_target)

        # Optimizers (fused=True for CUDA tensors)
        use_fused = self.device.type == "cuda"
        self.act_optimizer = torch.optim.AdamW(
            self.act.parameters(), self.learning_rate, fused=use_fused,
        )
        self.cri_optimizer = (
            torch.optim.AdamW(
                self.cri.parameters(), self.learning_rate, fused=use_fused,
            )
            if cri_class
            else self.act_optimizer
        )

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        # Checkpoint attrs
        self.save_attr_names = {
            "act", "act_target", "act_optimizer",
            "cri", "cri_target", "cri_optimizer",
        }

    def explore_env(
        self, env, horizon_len: int, if_random: bool = False,
    ) -> tuple[Tensor, ...]:
        """Collect trajectories from vectorized env.

        Returns
        -------
        tuple of (states, actions, rewards, undones)
            states:  (horizon_len, num_envs, state_dim)
            actions: (horizon_len, num_envs, 1)
            rewards: (horizon_len, num_envs)
            undones: (horizon_len, num_envs)
        """
        states = torch.zeros(
            (horizon_len, self.num_envs, self.state_dim),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.zeros(
            (horizon_len, self.num_envs, 1),
            dtype=torch.int32, device=self.device,
        )
        rewards = torch.zeros(
            (horizon_len, self.num_envs),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.zeros(
            (horizon_len, self.num_envs),
            dtype=torch.bool, device=self.device,
        )

        state = self.last_state
        get_action = self.act_target.get_action

        for t in range(horizon_len):
            if if_random:
                action = torch.randint(
                    self.action_dim, size=(self.num_envs, 1), device=self.device,
                )
            else:
                action = get_action(state).detach()

            states[t] = state
            state, reward, done, _ = env.step(action)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.float()
        return states, actions, rewards, undones

    def get_obj_critic(
        self, buffer: AlphaSeekReplayBuffer, batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute Double DQN critic loss.

        Returns (loss, q_values) tuple.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            next_qs = (
                torch.min(*self.cri_target.get_q1_q2(next_ss))
                .max(dim=1, keepdim=True)[0]
                .squeeze(1)
            )
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [
            qs.gather(1, actions.long()).squeeze(1)
            for qs in self.act.get_q1_q2(states)
        ]
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def update_net(self, buffer: AlphaSeekReplayBuffer) -> tuple[float, ...]:
        """Run SGD updates on the Q-network using replay buffer data.

        Returns (avg_critic_loss, avg_q_value) tuple.
        """
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self._update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self._get_cumulative_rewards(
                    rewards=rewards, undones=undones,
                ).reshape((-1,)),
            )

        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self._optimizer_update(self.cri_optimizer, obj_critic)
            self._soft_update(self.cri_target, self.cri, self.soft_update_tau)

        return obj_critics / update_times, obj_actors / update_times

    def save_agent(self, cwd: str) -> None:
        """Save agent state dicts to disk."""
        os.makedirs(cwd, exist_ok=True)
        for attr_name in self.save_attr_names:
            file_path = os.path.join(cwd, f"{attr_name}.pth")
            obj = getattr(self, attr_name)
            # Unwrap torch.compile wrapper for serialization
            if hasattr(obj, "_orig_mod"):
                obj = obj._orig_mod
            torch.save(obj, file_path)
        logger.info(f"Agent saved to {cwd}")

    def load_agent(self, cwd: str) -> None:
        """Load agent state dicts from disk."""
        for attr_name in self.save_attr_names:
            file_path = os.path.join(cwd, f"{attr_name}.pth")
            if os.path.isfile(file_path):
                obj = torch.load(file_path, map_location=self.device, weights_only=False)
                # Re-apply torch.compile on CUDA
                if self.device.type == "cuda" and isinstance(obj, torch.nn.Module):
                    obj = torch.compile(obj)
                setattr(self, attr_name, obj)
        logger.info(f"Agent loaded from {cwd}")

    # --- Private helpers ---

    @staticmethod
    def _soft_update(
        target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float,
    ) -> None:
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.lerp_(cur.data, tau)

    def _optimizer_update(self, optimizer: torch.optim.Optimizer, objective: Tensor) -> None:
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(
            parameters=optimizer.param_groups[0]["params"],
            max_norm=self.clip_grad_norm,
        )
        optimizer.step()

    def _get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target(last_state).argmax(dim=1).detach().float()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def _update_avg_std_for_normalization(
        self, states: Tensor, returns: Tensor,
    ) -> None:
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentD3QN(AgentDoubleDQN):
    """Dueling Double DQN — uses QNetTwinDuel (advantage + value decomposition)."""

    act_class = QNetTwinDuel
    cri_class = None


class AgentTwinD3QN(AgentDoubleDQN):
    """Twin DQN — uses QNetTwin (same as DoubleDQN, historical naming)."""

    act_class = QNetTwin
    cri_class = None


# Registry mapping string names to agent classes
AGENT_MAP: dict[str, type[AgentDoubleDQN]] = {
    "D3QN": AgentD3QN,
    "AgentD3QN": AgentD3QN,
    "DoubleDQN": AgentDoubleDQN,
    "AgentDoubleDQN": AgentDoubleDQN,
    "TwinD3QN": AgentTwinD3QN,
    "AgentTwinD3QN": AgentTwinD3QN,
}
