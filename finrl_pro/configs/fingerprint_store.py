"""Fingerprint manifest store interfaces for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class FingerprintRecord:
    """Represents a single experiment fingerprint entry."""

    fingerprint_id: str
    config_path: str
    dataset_hash: str
    seed: int
    mlflow_run_id: str


@dataclass(slots=True)
class FingerprintStore:
    """Manage storage of experiment fingerprints for FinRL Pro."""

    manifest_path: Path
    records: dict[str, FingerprintRecord] = field(default_factory=dict)

    def load(self) -> None:
        """Load fingerprints from the backing manifest."""
        raise NotImplementedError("Manifest loading will be implemented in US2.")

    def save(self) -> None:
        """Persist fingerprints to the backing manifest."""
        raise NotImplementedError("Manifest persistence will be implemented in US2.")

    def register(self, record: FingerprintRecord) -> None:
        """Register or update an experiment fingerprint."""
        raise NotImplementedError("Fingerprint registration will be implemented in US2.")

    def list_records(self) -> Iterable[FingerprintRecord]:
        """Return the fingerprints recorded in the store."""
        return self.records.values()
