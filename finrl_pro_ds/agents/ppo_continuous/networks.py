"""Networks for continuous PPO on the V7 ContinuousSwingEnv.

The actor is the *same* ``SACActorNetwork`` SAC uses (tanh-squashed diagonal
Gaussian over Box(-1, 1)), so the policy architecture is identical to SAC. The
only added piece is a state-value critic V(s) on a separate encoder.

Numerical contract (verified by the Math skill):
  * Action sampling / log-prob with tanh correction lives in
    ``SACActorNetwork.sample`` (re-used verbatim).
  * ``evaluate_actions`` re-derives log-prob for a STORED squashed action ``a``
    by inverting the squash, ``z = atanh(a)``, then applying the same Gaussian
    log-prob + tanh Jacobian correction. This MUST match ``sample`` exactly up
    to floating-point error, otherwise the PPO importance ratio is biased.
"""
from typing import Optional

import torch
import torch.nn as nn

from finrl_pro_ds.agents.sac.networks import (
    MultiScaleEncoder,
    SACActorNetwork,
    SummaryStatsEncoder,
)

# atanh numerical guard: clamp squashed actions away from +-1 before inverting
# the tanh, otherwise atanh -> +-inf.
_ATANH_EPS = 1e-6


class ValueNetwork(nn.Module):
    """State-value V(s) critic for PPO.

    Mirrors ``SACCriticNetwork``'s encoder stack but takes NO action input and
    emits a scalar value instead of a Q-value. A dedicated encoder (not shared
    with the actor) is standard PPO practice and avoids actor/critic gradient
    interference (the shared-network pathology PPG addresses).
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
        obs_mode: str = "window",
        lob_encoder_config: Optional[dict] = None,
    ):
        super().__init__()
        self._obs_mode = obs_mode

        if obs_mode == "summary_stats":
            summary_dim = scale_encoder_config.get("summary_input_dim", 50)
            self.encoder = SummaryStatsEncoder(summary_dim, fusion_dim)
        else:
            self.encoder = MultiScaleEncoder(
                scale_encoder_config, private_dim, fusion_dim, n_scales,
                lob_encoder_config=lob_encoder_config,
            )

        self.v_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init (PPO best practice); value output gain=1.0.
        for layer in self.v_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2.0 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.v_head[-1].weight, gain=1.0)

    def forward(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return V(s): shape (B,)."""
        if self._obs_mode == "summary_stats":
            features = self.encoder(scale_stack)
        else:
            features = self.encoder(scale_stack, private, lob=lob)
        return self.v_head(features).squeeze(-1)


class PPOContinuousActorCritic(nn.Module):
    """Continuous PPO actor-critic for V7 (Box(-1, 1) position fraction).

    actor : SACActorNetwork (tanh-squashed Gaussian) — identical to SAC.
    critic: ValueNetwork V(s) on a separate encoder.
    """

    def __init__(
        self,
        scale_encoder_config: dict,
        private_dim: int = 5,
        fusion_dim: int = 256,
        n_scales: int = 3,
        action_dim: int = 1,
        obs_mode: str = "window",
        lob_encoder_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self._obs_mode = obs_mode

        self.actor = SACActorNetwork(
            scale_encoder_config, private_dim, fusion_dim, n_scales,
            action_dim=action_dim, obs_mode=obs_mode,
            lob_encoder_config=lob_encoder_config,
        )
        self.critic = ValueNetwork(
            scale_encoder_config, private_dim, fusion_dim, n_scales,
            obs_mode=obs_mode, lob_encoder_config=lob_encoder_config,
        )

    # ------------------------------------------------------------------
    # Rollout-time: sample an action + its log-prob + value
    # ------------------------------------------------------------------
    def act(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        lob: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for rollout collection.

        Returns:
            action:   (B, action_dim) in [-1, 1]
            log_prob: (B,) joint log-prob of the sampled action
            value:    (B,) V(s)
        """
        action, log_prob = self.actor.sample(
            scale_stack, private, deterministic=deterministic, lob=lob,
        )
        value = self.critic(scale_stack, private, lob=lob)
        # SACActorNetwork.sample returns log_prob shaped (B, 1) for the summed
        # multi-dim case and (B, action_dim) for the deterministic branch;
        # collapse to (B,). For action_dim == 1 both are (B, 1).
        log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)
        return action, log_prob, value

    def value_only(
        self,
        scale_stack: torch.Tensor,
        private: Optional[torch.Tensor] = None,
        lob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """V(s) only — used for the GAE bootstrap value at rollout end."""
        return self.critic(scale_stack, private, lob=lob)

    # ------------------------------------------------------------------
    # Update-time: re-evaluate STORED actions under the current policy
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        scale_stack: torch.Tensor,
        private: torch.Tensor,
        actions: torch.Tensor,
        lob: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-derive (log_prob, value, entropy) for stored squashed actions.

        Args:
            actions: (B, action_dim) tanh-squashed actions in [-1, 1] as stored
                     during rollout.

        Returns:
            log_prob: (B,) joint log-prob under the current policy
            value:    (B,) V(s)
            entropy:  (B,) pre-squash diagonal-Gaussian differential entropy,
                      summed over action dims (the standard PPO entropy bonus
                      for a Gaussian policy; the tanh Jacobian has no closed-form
                      entropy, so the base-Gaussian entropy is used — identical
                      convention to common continuous-PPO implementations).
        """
        mu, log_sigma = self.actor.forward(scale_stack, private, lob=lob)
        sigma = log_sigma.exp()

        # Invert the tanh squash to recover the pre-squash Gaussian sample.
        a = torch.clamp(actions, -1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
        # atanh(a) = 0.5 * (log(1 + a) - log(1 - a)), computed stably via log1p.
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        dist = torch.distributions.Normal(mu, sigma)
        # Same tanh Jacobian correction as SACActorNetwork.sample (uses the
        # clamped action `a`, matching sample()'s clamp-based correction).
        log_prob = dist.log_prob(z) - torch.log(torch.clamp(1.0 - a.pow(2), min=1e-6))
        log_prob = log_prob.sum(dim=-1)  # (B,)

        entropy = dist.entropy().sum(dim=-1)  # (B,)
        value = self.critic(scale_stack, private, lob=lob)
        return log_prob, value, entropy
