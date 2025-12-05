"""Fingerprint manifest store for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from finrl_pro.mlops.fingerprint import ExperimentFingerprint, serialize_fingerprints

@dataclass(slots=True)
class FingerprintStore:
    """Manage storage of experiment fingerprints for FinRL Pro."""

    manifest_path: Path
    records: dict[str, ExperimentFingerprint] = field(default_factory=dict)

    def load(self) -> None:
        """Load fingerprints from the backing manifest."""
        if not self.manifest_path.exists():
            self.records.clear()
            return

        payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        entries = payload.get("fingerprints", []) or []
        self.records.clear()
        for entry in entries:
            fingerprint = ExperimentFingerprint.from_dict(entry)
            fingerprint.validate()
            self.records[fingerprint.fingerprint_id] = fingerprint

    def save(self) -> None:
        """Persist fingerprints to the backing manifest."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprints": serialize_fingerprints(self.records.values())}
        self.manifest_path.write_text(
            yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
        )

    def register(self, record: ExperimentFingerprint) -> None:
        """Register or update an experiment fingerprint."""
        record.validate()
        self.records[record.fingerprint_id] = record

    def get(self, fingerprint_id: str) -> ExperimentFingerprint | None:
        """Return a fingerprint by identifier if present."""
        return self.records.get(fingerprint_id)

    def list_records(self) -> Iterable[ExperimentFingerprint]:
        """Return the fingerprints recorded in the store."""
        return self.records.values()
