"""Model registry for AlphaSeek agent checkpoints.

Tracks model versions, metadata, and provides a factory for building
ensembles from YAML config.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .agent_wrapper import AlphaSeekAgent
from .ensemble import AlphaSeekEnsemble, EnsembleStrategy

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model checkpoint."""
    name: str
    agent_type: str
    net_dims: tuple[int, ...]
    checkpoint_dir: str
    version: int = 1
    wandb_run_id: Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""


class ModelRegistry:
    """Registry of AlphaSeek model checkpoints.

    Stores metadata in a models.json file alongside checkpoints.
    Provides a factory method to build ensembles from config dicts.

    Usage:
        registry = ModelRegistry("/path/to/models")
        registry.register("d3qn_tuned", "D3QN", (256, 256), "/path/to/ckpt")
        ensemble = registry.load_ensemble(config_dict)
    """

    METADATA_FILE = "models.json"

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._models: dict[str, ModelInfo] = {}
        self._metadata_path = os.path.join(base_dir, self.METADATA_FILE)
        self._load_metadata()

    def _load_metadata(self):
        """Load existing model metadata from disk."""
        if os.path.isfile(self._metadata_path):
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, info in data.items():
                info["net_dims"] = tuple(info["net_dims"])
                self._models[name] = ModelInfo(**info)
            logger.info("Loaded %d models from registry at %s", len(self._models), self._metadata_path)

    def _save_metadata(self):
        """Persist model metadata to disk."""
        os.makedirs(self.base_dir, exist_ok=True)
        data = {}
        for name, info in self._models.items():
            d = asdict(info)
            d["net_dims"] = list(d["net_dims"])  # JSON doesn't support tuples
            data[name] = d
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        agent_type: str,
        net_dims: tuple[int, ...],
        checkpoint_dir: str,
        wandb_run_id: Optional[str] = None,
        notes: str = "",
    ) -> ModelInfo:
        """Register a model checkpoint.

        If a model with the same name exists, increments the version.
        """
        # Validate checkpoint exists
        act_path = os.path.join(checkpoint_dir, "act.pth")
        act_target_path = os.path.join(checkpoint_dir, "act_target.pth")
        if not (os.path.isfile(act_path) or os.path.isfile(act_target_path)):
            raise FileNotFoundError(
                f"No act.pth or act_target.pth found in {checkpoint_dir}"
            )

        version = 1
        if name in self._models:
            version = self._models[name].version + 1
            logger.info("Updating %s from v%d to v%d", name, version - 1, version)

        info = ModelInfo(
            name=name,
            agent_type=agent_type,
            net_dims=net_dims,
            checkpoint_dir=checkpoint_dir,
            version=version,
            wandb_run_id=wandb_run_id,
            notes=notes,
        )
        self._models[name] = info
        self._save_metadata()
        logger.info("Registered model: %s v%d (%s)", name, version, agent_type)
        return info

    def get(self, name: str) -> ModelInfo:
        """Get model info by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not in registry. Available: {list(self._models.keys())}")
        return self._models[name]

    def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def load_agent(self, name: str, device: str = "cpu") -> AlphaSeekAgent:
        """Load a single agent from the registry."""
        info = self.get(name)
        agent = AlphaSeekAgent(
            agent_type=info.agent_type,
            net_dims=info.net_dims,
            device=device,
        )
        agent.load(info.checkpoint_dir)
        return agent

    def load_ensemble(self, config: dict, device: str = "cpu") -> AlphaSeekEnsemble:
        """Build an ensemble from a config dict.

        Config format:
            alphaseek:
              agents:
                - name: d3qn_tuned       # registry name OR inline spec
                  type: D3QN              # agent type (if inline)
                  net_dims: [256, 256]    # (if inline)
                  checkpoint_dir: /path   # (if inline)
              ensemble_strategy: q_average
              confidence_threshold: 0.001
              agent_weights: [0.4, 0.3, 0.3]  # optional
        """
        alphaseek_cfg = config.get("alphaseek", config)
        agent_configs = alphaseek_cfg.get("agents", [])
        strategy_str = alphaseek_cfg.get("ensemble_strategy", "majority_vote")
        confidence_threshold = alphaseek_cfg.get("confidence_threshold", 0.001)
        agent_weights = alphaseek_cfg.get("agent_weights")

        strategy = EnsembleStrategy(strategy_str)

        agents: list[AlphaSeekAgent] = []
        for acfg in agent_configs:
            if isinstance(acfg, str):
                # Registry name shorthand
                agents.append(self.load_agent(acfg, device=device))
            elif "name" in acfg and acfg["name"] in self._models:
                # Load from registry
                agents.append(self.load_agent(acfg["name"], device=device))
            else:
                # Inline spec
                agent = AlphaSeekAgent(
                    agent_type=acfg["type"],
                    net_dims=tuple(acfg.get("net_dims", (128, 128, 128))),
                    device=device,
                )
                agent.load(acfg["checkpoint_dir"])
                agents.append(agent)

        ensemble = AlphaSeekEnsemble(
            agents=agents,
            strategy=strategy,
            agent_weights=agent_weights,
            confidence_threshold=confidence_threshold,
        )
        logger.info(
            "Built ensemble from config: %d agents, strategy=%s",
            len(agents), strategy.value,
        )
        return ensemble

    def __repr__(self) -> str:
        return f"ModelRegistry(base_dir={self.base_dir}, models={len(self._models)})"
