"""Module registry helpers for FinRL Pro extension boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class FinRLProModule:
    """Represents metadata for a FinRL Pro module."""

    dotted_path: str
    owner_team: str | None = None
    change_control_id: str | None = None

    def __post_init__(self) -> None:
        if not self.dotted_path.startswith("finrl_pro."):
            raise ValueError(
                f"Module '{self.dotted_path}' must reside within the finrl_pro namespace."
            )


class ModuleRegistry:
    """In-memory registry backed by the modules manifest."""

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = manifest_path
        self._modules: Dict[str, FinRLProModule] = {}

    @property
    def manifest_path(self) -> Path:
        """Return the manifest path used for persistence."""
        return self._manifest_path

    def register(self, module: FinRLProModule) -> None:
        """Register a module with the registry."""
        self._modules[module.dotted_path] = module

    def find(self, dotted_path: str) -> FinRLProModule | None:
        """Return metadata for the requested module path."""
        return self._modules.get(dotted_path)

    def all_modules(self) -> list[FinRLProModule]:
        """Return all registered modules."""
        return list(self._modules.values())

    def load(self) -> None:
        """Load registry data from the modules manifest."""
        raise NotImplementedError("Manifest loading will be implemented in a later phase.")

    def save(self) -> None:
        """Persist registry data to the modules manifest."""
        raise NotImplementedError("Manifest persistence will be implemented in a later phase.")
