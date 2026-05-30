#!/usr/bin/env python3
"""Backfill the v2.6 `deployed_config` field into a legacy v2.5 verdict.json.

Single-purpose sidecar for the Stage 2.5-R Sensitivity Audit (v2.7-A,
S553). Reads the L1 multiseed config's `env.deadband_threshold` +
`env.max_leverage`, writes them as the verdict's `deployed_config` block.
Idempotent. Does NOT bump `schema_version`.

Usage:
    python scripts/backfill_deployed_config_field.py \\
        --workstream <name> \\
        --config configs/<l1_multiseed_yaml>

Architecture: ``.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md``
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill-deployed-config")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstream", required=True, type=str)
    parser.add_argument("--config", required=True, type=Path,
                        help="L1 multiseed YAML containing env.deadband_threshold + env.max_leverage")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    verdict_path = args.results_root / f"{args.workstream}_ensemble" / "verdict.json"
    if not verdict_path.exists():
        log.error(f"verdict missing: {verdict_path}")
        return 1

    if not args.config.exists():
        log.error(f"config missing: {args.config}")
        return 1

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    env = cfg.get("env") or {}
    if "deadband_threshold" not in env or "max_leverage" not in env:
        log.error(
            f"config {args.config} lacks env.deadband_threshold or env.max_leverage"
        )
        return 1

    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    deployed = {
        "deadband_threshold": float(env["deadband_threshold"]),
        "max_leverage": float(env["max_leverage"]),
    }

    if verdict.get("deployed_config") == deployed:
        log.info(f"deployed_config already up-to-date in {verdict_path} — no-op")
        return 0

    verdict["deployed_config"] = deployed
    # Atomic write
    tmp = verdict_path.with_suffix(verdict_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(verdict, indent=2, sort_keys=False), encoding="utf-8",
    )
    tmp.replace(verdict_path)
    log.info(
        f"deployed_config backfilled: deadband={deployed['deadband_threshold']} "
        f"max_leverage={deployed['max_leverage']} → {verdict_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
