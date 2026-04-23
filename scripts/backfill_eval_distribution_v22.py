#!/usr/bin/env python3
"""Protocol v2.2 §2 eval_distribution backfill for paper-deployed workstreams.

Paper-deployed strategies with manifests predating the v2.2 amendment
(SG-1 XAUUSD per S489, GMGP1 XAUUSD CME per S493) need `eval_distribution`
retro-populated so live §8.2 action-drift exits LOG_ONLY mode.

This script reads existing trajectory parquet files — written by
`scripts/sg1_xauusd_ensemble_eval.py` (and siblings) during the original
Stage 2 / 2.5 run — and computes the v2.2 eval_distribution block from
the already-persisted action columns. No GPU / no checkpoint load needed.

Inputs:
    --dir  <path>         directory containing `solo_<seed>_trajectory.parquet`
                          files and one or more `ens_<rule>_trajectory.parquet`
                          files (the output dir of the eval script)
    --chosen-rule <name>  ensemble rule for the Stage 2.5 ensemble_report
                          (defaults to `ens_agreement` when present)
    --workstream <label>  label written into the report payload
    --window <start>:<end>  ISO-date pair naming the eval window

Outputs written into <dir>:
    seed_report.json       v2.2 Stage 2 per-seed eval_distributions
    ensemble_report.json   v2.2 Stage 2.5 ensemble_eval_distribution +
                           per-seed block (same as seed_report)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finrl_pro_ds.reporting import compute_eval_distribution  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill-v22")


def _find_solo_seeds(directory: Path) -> dict[int, Path]:
    """Return {seed: trajectory_path} for every `solo_<seed>_trajectory.parquet`."""
    pat = re.compile(r"solo_(\d+)_trajectory\.parquet$")
    out: dict[int, Path] = {}
    for f in sorted(directory.glob("solo_*_trajectory.parquet")):
        m = pat.search(f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def _find_ensemble_trajectory(directory: Path, rule: str) -> Optional[Path]:
    p = directory / f"{rule}_trajectory.parquet"
    return p if p.exists() else None


def backfill_directory(
    directory: Path,
    *,
    chosen_rule: str,
    workstream: str,
    window: Optional[str],
    deadband: float,
) -> dict:
    """Write v2.2 eval_distribution artifacts into `directory`.

    Returns a summary dict of what was written.
    """
    solos = _find_solo_seeds(directory)
    if not solos:
        raise FileNotFoundError(
            f"no solo_<seed>_trajectory.parquet in {directory}"
        )
    log.info(f"[{directory.name}] solo seeds found: {sorted(solos)}")

    seed_distributions: dict[str, dict] = {}
    for seed, traj_path in solos.items():
        df = pd.read_parquet(traj_path)
        if "action_agg" not in df.columns:
            raise KeyError(f"{traj_path} lacks action_agg column")
        seed_distributions[str(seed)] = compute_eval_distribution(
            df["action_agg"].to_numpy(), deadband=deadband,
        )
    seed_report = {
        "protocol": "v2.2_stage_2_seed_report",
        "source": "backfill_from_existing_trajectories",
        "workstream": workstream,
        "window": window,
        "seeds": sorted(int(s) for s in seed_distributions),
        "eval_distribution_by_seed": seed_distributions,
    }
    (directory / "seed_report.json").write_text(
        json.dumps(seed_report, indent=2, default=str),
    )
    log.info(f"[{directory.name}] wrote seed_report.json ({len(solos)} seeds)")

    ens_path = _find_ensemble_trajectory(directory, chosen_rule)
    ensemble_report: dict = {
        "protocol": "v2.2_stage_2_5_ensemble_report",
        "source": "backfill_from_existing_trajectories",
        "workstream": workstream,
        "window": window,
        "chosen_rule": chosen_rule,
        "per_seed_eval_distribution": seed_distributions,
    }
    if ens_path is None:
        log.warning(
            f"[{directory.name}] no {chosen_rule}_trajectory.parquet — "
            f"ensemble_eval_distribution omitted. Live drift baseline "
            f"will fall back to per-seed mode."
        )
        ensemble_report["ensemble_eval_distribution"] = None
    else:
        ens_df = pd.read_parquet(ens_path)
        if "action_agg" not in ens_df.columns:
            raise KeyError(f"{ens_path} lacks action_agg column")
        ensemble_report["ensemble_eval_distribution"] = compute_eval_distribution(
            ens_df["action_agg"].to_numpy(),
            deadband=deadband,
            composition_rule=chosen_rule,
        )
        log.info(
            f"[{directory.name}] wrote ensemble_report.json (rule={chosen_rule}, "
            f"n={len(ens_df)})",
        )
    (directory / "ensemble_report.json").write_text(
        json.dumps(ensemble_report, indent=2, default=str),
    )
    return {
        "directory": str(directory),
        "seeds": sorted(solos),
        "chosen_rule": chosen_rule,
        "ensemble_baseline_resolved": ens_path is not None,
        "baseline_path": str(directory / "ensemble_report.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", required=True, type=Path, action="append",
        help="directory with solo_<seed>_trajectory.parquet files (repeatable)",
    )
    parser.add_argument(
        "--chosen-rule", default="ens_agreement",
        help="ensemble rule used for the deployed strategy (default: ens_agreement)",
    )
    parser.add_argument(
        "--workstream", default="",
        help="workstream label written into reports",
    )
    parser.add_argument(
        "--window", default=None,
        help="ISO-date eval window written into reports (e.g. 2025-09-01:2025-10-31)",
    )
    parser.add_argument(
        "--deadband", type=float, default=0.25,
        help="scalar deadband threshold for deadband_frac computation "
             "(v2.2 §8.2 default: 0.25; match the env's deadband_threshold)",
    )
    args = parser.parse_args()

    summaries = []
    for d in args.dir:
        if not d.is_dir():
            log.error(f"not a directory: {d}")
            return 2
        summaries.append(
            backfill_directory(
                d,
                chosen_rule=args.chosen_rule,
                workstream=args.workstream,
                window=args.window,
                deadband=args.deadband,
            )
        )

    print("\n=== BACKFILL SUMMARY ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
