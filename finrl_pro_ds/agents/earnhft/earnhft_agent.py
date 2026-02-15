"""
Unified EarnHFT Agent — 3-level hierarchy wrapper.

Wraps the Q-Teacher, low-level agent pool, and high-level router into
a single `predict()` interface for backtesting and deployment.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_TORCH = False


class EarnHFTAgent:
    """Unified EarnHFT agent wrapping the 3-level hierarchy.

    Usage:
        1. Load trained pool agents and router from checkpoints
        2. Call `predict()` at each tick — the router selects the agent,
           then that agent produces the trading action

    Args:
        pool_agents: List of DiscretePPOAgent instances.
        router: RouterDQN instance.
        num_actions: Number of discrete position levels.
        max_holding: Maximum position size.
        minute_interval: Ticks between router decisions.
        device: Torch device.
    """

    def __init__(
        self,
        pool_agents: List,
        router,
        num_actions: int = 5,
        max_holding: float = 1.0,
        minute_interval: int = 60,
        device: str = "cpu",
    ):
        self.pool_agents = pool_agents
        self.router = router
        self.num_actions = num_actions
        self.max_holding = max_holding
        self.minute_interval = minute_interval
        self.pool_size = len(pool_agents)
        self.device = device

        # State
        self._tick_count = 0
        self._current_agent_idx = 0
        self._minute_features_accumulator = []

    def reset(self):
        """Reset agent state for a new episode."""
        self._tick_count = 0
        self._current_agent_idx = 0
        self._minute_features_accumulator = []
        for agent in self.pool_agents:
            agent.reset_hidden_state()

    def predict(
        self,
        micro: "torch.Tensor",
        private: "torch.Tensor",
        macro: Optional["torch.Tensor"] = None,
        minute_features: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Predict trading action through the hierarchy.

        At each tick:
          - If at a router decision point (every minute_interval ticks),
            the router selects which pool agent to use
          - The selected pool agent produces the trading action

        Args:
            micro: (1, W, micro_dim) LOB features.
            private: (1, W, private_dim) private state.
            macro: (1, macro_dim) macro features, or None.
            minute_features: Aggregated minute features for router, shape (feature_dim,).
            deterministic: Use greedy actions.

        Returns:
            Dict with 'action', 'agent_idx', 'log_prob', 'value'.
        """
        # Router decision at minute boundary
        if self._tick_count % self.minute_interval == 0 and minute_features is not None:
            self._current_agent_idx = self.router.select_action(
                minute_features, eval_mode=deterministic
            )

        # Get action from selected pool agent
        agent = self.pool_agents[self._current_agent_idx]
        actions, log_probs, values = agent.predict(
            micro, private, macro, deterministic=deterministic
        )

        self._tick_count += 1

        return {
            "action": int(actions[0]),
            "agent_idx": self._current_agent_idx,
            "log_prob": float(log_probs[0]),
            "value": float(values[0]),
        }

    @classmethod
    def from_checkpoints(
        cls,
        pool_checkpoint_paths: List[str],
        router_checkpoint_path: str,
        network_config: Dict,
        router_obs_dim: int,
        num_actions: int = 5,
        max_holding: float = 1.0,
        minute_interval: int = 60,
        device: str = "cpu",
    ) -> "EarnHFTAgent":
        """Load EarnHFT agent from saved checkpoints.

        Args:
            pool_checkpoint_paths: Paths to DiscretePPO checkpoints.
            router_checkpoint_path: Path to RouterDQN checkpoint.
            network_config: Config for DiscretePPOActorCritic.
            router_obs_dim: Feature dimension for router.
            num_actions: Number of discrete actions.
            max_holding: Maximum position.
            minute_interval: Router decision interval.
            device: Torch device.

        Returns:
            Initialized EarnHFTAgent.
        """
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN

        pool_size = len(pool_checkpoint_paths)

        # Load pool agents
        pool_agents = []
        for i, path in enumerate(pool_checkpoint_paths):
            agent = DiscretePPOAgent(network_config=network_config, device=device)
            agent.load(path)
            pool_agents.append(agent)
            logger.info(f"Loaded pool agent {i} from {path}")

        # Load router
        router = RouterDQN(
            obs_dim=router_obs_dim,
            pool_size=pool_size,
            device=device,
        )
        router.load(router_checkpoint_path)
        logger.info(f"Loaded router from {router_checkpoint_path}")

        return cls(
            pool_agents=pool_agents,
            router=router,
            num_actions=num_actions,
            max_holding=max_holding,
            minute_interval=minute_interval,
            device=device,
        )
