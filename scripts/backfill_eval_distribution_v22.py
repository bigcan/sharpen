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

Reproduction from a fresh clone:
    Trajectory parquets live under `results/` which is gitignored. Regenerate
    them by running the upstream ensemble_eval first (requires committed
    checkpoints + data):
        python scripts/sg1_xauusd_ensemble_eval.py --config <cfg>
    Then run this backfill. The output JSONs should be copied to
    `baselines/<workstream>/` so they are committed + baked into the live
    engine Docker image (`Dockerfile.live-engine` COPY baselines/).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sharpen.reporting import (  # noqa: E402
    DEFAULT_PHASE_SPECS,
    compute_challenge_target_hit_rates,
    compute_eval_distribution,
    parse_phase_spec_arg,
)

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


def _rolling_realized_vol(pv: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling realized vol proxy: std of log-return of portfolio_value.

    Portfolio-value returns depend on the strategy's trades, so this is a
    proxy — not the market-vol baseline we ultimately want. For the backfill
    path it's the only vol signal available in the persisted trajectory.
    Upstream (T3) should extend run_rule to capture bar `close` so market
    vol can be computed directly; until then the buckets approximate.
    """
    pv = np.asarray(pv, dtype=np.float64)
    if pv.size < 2:
        return np.zeros(pv.size)
    log_ret = np.zeros(pv.size)
    np.log(pv[1:] / np.clip(pv[:-1], 1e-9, None), out=log_ret[1:])
    vol = np.zeros(pv.size)
    for i in range(pv.size):
        lo = max(0, i - window + 1)
        vol[i] = float(np.std(log_ret[lo:i + 1], ddof=0)) if i >= 1 else 0.0
    return vol


def backfill_directory(
    directory: Path,
    *,
    chosen_rule: str,
    workstream: str,
    window: Optional[str],
    deadband: float,
    phase_specs: Optional[list[dict]] = None,
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

    # Equal-weight quartile shares as regime_quartiles marker (the backfill
    # can't know the training-time vol distribution, so it declares the 4
    # quartiles exist so the helper computes per-sample cutpoints and the
    # live tracker picks them up via baseline fallback).
    equal_quartiles = {
        "vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25,
    }

    specs = phase_specs if phase_specs is not None else list(DEFAULT_PHASE_SPECS)

    seed_distributions: dict[str, dict] = {}
    seed_hit_rates: dict[str, dict] = {}
    for seed, traj_path in solos.items():
        df = pd.read_parquet(traj_path)
        if "action_agg" not in df.columns:
            raise KeyError(f"{traj_path} lacks action_agg column")
        bar_vol = None
        pv_arr = None
        if "portfolio_value" in df.columns:
            pv_arr = df["portfolio_value"].to_numpy()
            bar_vol = _rolling_realized_vol(pv_arr)
        seed_distributions[str(seed)] = compute_eval_distribution(
            df["action_agg"].to_numpy(),
            bar_vol=bar_vol,
            regime_quartiles=equal_quartiles if bar_vol is not None else None,
            deadband=deadband,
        )
        if pv_arr is not None:
            seed_hit_rates[str(seed)] = compute_challenge_target_hit_rates(
                pv_arr, phase_specs=specs,
            )
        else:
            log.warning(
                f"[{traj_path.name}] no portfolio_value column — "
                f"challenge_target_hit_rate omitted",
            )
    seed_report = {
        "protocol": "v2.2_stage_2_seed_report",
        "source": "backfill_from_existing_trajectories",
        "workstream": workstream,
        "window": window,
        "seeds": sorted(int(s) for s in seed_distributions),
        "eval_distribution_by_seed": seed_distributions,
        "challenge_target_hit_rate_by_seed": seed_hit_rates,
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
        "challenge_target_hit_rate_by_seed": seed_hit_rates,
    }
    if ens_path is None:
        log.warning(
            f"[{directory.name}] no {chosen_rule}_trajectory.parquet — "
            f"ensemble_eval_distribution + ensemble_challenge_target_hit_rate "
            f"omitted. Live drift baseline will fall back to per-seed mode."
        )
        ensemble_report["ensemble_eval_distribution"] = None
        ensemble_report["ensemble_challenge_target_hit_rate"] = None
    else:
        ens_df = pd.read_parquet(ens_path)
        if "action_agg" not in ens_df.columns:
            raise KeyError(f"{ens_path} lacks action_agg column")
        bar_vol = None
        ens_pv = None
        if "portfolio_value" in ens_df.columns:
            ens_pv = ens_df["portfolio_value"].to_numpy()
            bar_vol = _rolling_realized_vol(ens_pv)
        ensemble_report["ensemble_eval_distribution"] = compute_eval_distribution(
            ens_df["action_agg"].to_numpy(),
            bar_vol=bar_vol,
            regime_quartiles=equal_quartiles if bar_vol is not None else None,
            deadband=deadband,
            composition_rule=chosen_rule,
        )
        if ens_pv is not None:
            ensemble_report["ensemble_challenge_target_hit_rate"] = (
                compute_challenge_target_hit_rates(ens_pv, phase_specs=specs)
            )
        else:
            ensemble_report["ensemble_challenge_target_hit_rate"] = None
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
    parser.add_argument(
        "--phase-spec", default=None,
        help="comma-separated 'label:target_pct[:window_bars[:stride_bars]]' phases "
             "for challenge_target_hit_rate_per_window (default: step1:0.10,step2:0.05). "
             "Set window_bars/stride_bars to subdivide a long L1 eval into multiple "
             "non-overlapping windows for a more stable hit-rate estimate.",
    )
    args = parser.parse_args()
    phase_specs = parse_phase_spec_arg(args.phase_spec) if args.phase_spec else None

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
                phase_specs=phase_specs,
            )
        )

    print("\n=== BACKFILL SUMMARY ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
