"""Benchmark catalog entities for FinRL Pro evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping

import yaml


@dataclass(slots=True)
class BenchmarkCatalogEntry:
    """Represents a frozen benchmark dataset and baseline reference."""

    benchmark_id: str
    label: str
    dataset_hash: str
    baseline_agent_checkpoint: str
    timeframe: str
    metrics_baseline: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure the entry contains required metadata."""
        if not self.benchmark_id:
            raise ValueError("benchmark_id is required.")
        if not self.label:
            raise ValueError("label is required.")
        if not self.dataset_hash:
            raise ValueError("dataset_hash is required.")
        if not self.baseline_agent_checkpoint:
            raise ValueError("baseline_agent_checkpoint is required.")
        if not self.timeframe:
            raise ValueError("timeframe is required.")
        if not self.metrics_baseline:
            raise ValueError("metrics_baseline cannot be empty.")


class BenchmarkCatalog:
    """Load and query benchmark catalog entries."""

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = manifest_path
        self._entries: Dict[str, BenchmarkCatalogEntry] = {}

    def load(self) -> None:
        """Load benchmark entries from YAML manifest."""
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Benchmark catalog manifest not found: {self._manifest_path}"
            )

        payload = yaml.safe_load(self._manifest_path.read_text(encoding="utf-8")) or {}
        entries = payload.get("benchmarks", []) or []
        self._entries.clear()
        for entry in entries:
            catalog_entry = BenchmarkCatalogEntry(
                benchmark_id=str(entry["benchmark_id"]),
                label=str(entry["label"]),
                dataset_hash=str(entry["dataset_hash"]),
                baseline_agent_checkpoint=str(entry["baseline_agent_checkpoint"]),
                timeframe=str(entry["timeframe"]),
                metrics_baseline={
                    key: float(value) for key, value in entry.get("metrics_baseline", {}).items()
                },
            )
            catalog_entry.validate()
            self._entries[catalog_entry.benchmark_id] = catalog_entry

    def get(self, benchmark_id: str) -> BenchmarkCatalogEntry:
        """Lookup a benchmark entry by identifier."""
        try:
            return self._entries[benchmark_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Benchmark '{benchmark_id}' not found.") from exc

    def list_entries(self) -> Iterable[BenchmarkCatalogEntry]:
        """Return all benchmark entries."""
        return list(self._entries.values())

    def to_mapping(self) -> Mapping[str, BenchmarkCatalogEntry]:
        """Expose the catalog entries as a mapping."""
        return dict(self._entries)
