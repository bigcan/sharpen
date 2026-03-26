"""Q-network architectures for AlphaSeek DQN ensemble.

Copied from contest/reference/erl_net.py to avoid sys.path manipulation.
These classes must remain identical to the contest code — trained checkpoints
contain baked-in state_avg/state_std/value_avg/value_std parameters that
depend on the exact network structure.

DO NOT MODIFY these classes unless you retrain all agents from scratch.
"""

import torch
import torch.nn as nn

TEN = torch.Tensor


class QNetBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg


class QNetTwin(QNetBase):
    """Double DQN — used by AgentDoubleDQN and AgentTwinD3QN."""

    def __init__(self, dims: list[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        return q_val

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val1 = self.value_re_norm(self.net_val1(s_enc))
        q_val2 = self.value_re_norm(self.net_val2(s_enc))
        return q_val1, q_val2

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwinDuel(QNetBase):
    """D3QN: Dueling Double DQN — used by AgentD3QN."""

    def __init__(self, dims: list[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        q_adv = self.net_adv1(s_enc)
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv
        value = self.value_re_norm(value)
        return value

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)

        q_val1 = self.net_val1(s_enc)
        q_adv1 = self.net_adv1(s_enc)
        q_duel1 = self.value_re_norm(
            q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1,
        )

        q_val2 = self.net_val2(s_enc)
        q_adv2 = self.net_adv2(s_enc)
        q_duel2 = self.value_re_norm(
            q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2,
        )
        return q_duel1, q_duel2

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


def build_mlp(
    dims: list[int], activation: type[nn.Module] | None = None, if_raw_out: bool = True,
) -> nn.Sequential:
    if activation is None:
        activation = nn.ReLU
    net_list: list[nn.Module] = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer: nn.Linear, std: float = 1.0, bias_const: float = 1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


# Network class registry — maps agent class names to their Q-network class.
# Used by AlphaSeekAgent.load() to auto-select the right architecture.
AGENT_NET_MAP: dict[str, type[QNetBase]] = {
    "AgentD3QN": QNetTwinDuel,
    "D3QN": QNetTwinDuel,
    "AgentDoubleDQN": QNetTwin,
    "DoubleDQN": QNetTwin,
    "AgentTwinD3QN": QNetTwin,
    "TwinD3QN": QNetTwin,
}
