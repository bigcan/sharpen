"""CLI command to replay experiments using stored fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.training.trainer import Trainer

DEFAULT_MANIFEST = Path("finrl_pro/configs/fingerprints.yaml")


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for the reproducibility replay command."""
    parser = argparse.ArgumentParser(
        prog="finrl_pro.trainer.reproduce",
        description="Reproduce an experiment using a stored fingerprint.",
    )
    parser.add_argument(
        "fingerprint_id",
        help="Identifier of the fingerprint to replay.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=(
            "Path to the fingerprint manifest (defaults to "
            "finrl_pro/configs/fingerprints.yaml)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = Path(args.manifest)
    store = FingerprintStore(manifest_path=manifest_path)
    store.load()
    trainer = Trainer(fingerprint_store=store)

    fingerprint = trainer.reproduce(args.fingerprint_id)
    print(json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
