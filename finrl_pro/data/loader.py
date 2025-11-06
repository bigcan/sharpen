"""Module: loader
Purpose: Define data ingestion and preprocessing scaffolds for FinRL Pro.

Adds a resolver for dataset references. If a reference uses the
`snapshot://<snapshot_id>` scheme, it is intended to be resolved via
the database-backed snapshots per specs/001-db-snapshots.
"""

from __future__ import annotations


SNAPSHOT_SCHEME = "snapshot://"


class DataLoader:
    """Placeholder for future data loading pipelines."""

    def load(self) -> None:
        """Execute the data loading routine once implemented."""
        raise NotImplementedError("Data loading pipeline not yet implemented.")

    @staticmethod
    def resolve_dataset(dataset_hash: str):
        """Resolve a dataset reference to a concrete source.

        Supported:
        - snapshot://<snapshot_id> → load from DB (not yet implemented)
        - dvc://... or file paths → to be implemented as needed
        """
        if dataset_hash.startswith(SNAPSHOT_SCHEME):
            snapshot_id = dataset_hash[len(SNAPSHOT_SCHEME) :]
            raise NotImplementedError(
                f"Snapshot resolution not implemented yet for id '{snapshot_id}'."
            )
        raise NotImplementedError(f"Unknown dataset reference: {dataset_hash}")
