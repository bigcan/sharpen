"""Live-driven recalibration of an ensemble drift baseline.

Generalizes the manual sg1-xauusd S535-R1 workflow (commit 8767aecc) so the
next live-recal cycle is one CLI invocation, not a hand-edited JSON.

The helper:
  1. Pulls `drift/*` history from a live WandB run over a `_step` range.
  2. Filters by status (default: keep OK only — discards the false-CRIT
     emissions caused by the stale baseline being recalibrated against).
  3. Computes global + per-quartile live aggregates from `drift/*_frac_live`.
  4. Patches `ensemble_eval_distribution.{deadband_frac,saturation_frac}` and
     `by_vol_quartile.q{1..4}.{deadband_frac,saturation_frac}` to the live
     observations.
  5. Appends a `live_recal_metadata` block (preserves the original audit
     values inside `what_changed`) and bumps the `protocol` marker.

Use when a strategy is profitably running but the static baseline has gone
stale (regime shift, feature drift, etc.) and is producing false WARN/CRIT
on a healthy live policy.

Usage:
    python scripts/recal_drift_baseline.py \
        --baseline baselines/sg1_xauusd_fold_07/ensemble_report.json \
        --source-run bigcan-chiwin-technology/FinRL-Pro-DS/<run_id> \
        --step-range 2100,4506 \
        --reason "XAU regime shifted Feb -> May; live policy steady at deadband_frac~0.80"

Add --dry-run to inspect the diff before writing.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

QUARTILES = ("q1", "q2", "q3", "q4")
DRIFT_KEYS = (
    "drift/status",
    "drift/bucket",
    "drift/n_bars",
    "drift/deadband_frac_live",
    "drift/saturation_frac_live",
)


def _parse_step_range(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--step-range must be 'start,end'; got {s!r}"
        )
    return int(parts[0]), int(parts[1])


def _fetch_drift_history(run_path: str, step_range: tuple[int, int]) -> pd.DataFrame:
    """Pull drift/* keys from a WandB run; returns rows in [start, end] _step."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    df = run.history(
        keys=list(DRIFT_KEYS),
        pandas=True,
        x_axis="_step",
    )
    if df.empty:
        raise RuntimeError(
            f"Run {run_path} has no rows for keys={list(DRIFT_KEYS)}. "
            "Confirm drift_tracker is enabled and the run logged drift/*."
        )
    lo, hi = step_range
    mask = (df["_step"] >= lo) & (df["_step"] <= hi)
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError(
            f"No drift emissions in _step range [{lo}, {hi}]. "
            f"Run total step range was [{int(df['_step'].min())}, {int(df['_step'].max())}]."
        )
    return out


def _agg_stats(series: pd.Series) -> dict[str, float | int]:
    """Mean/median/std/p05/p95/min/max/n for a numeric series; NaN-safe."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": math.nan, "median": math.nan, "std": math.nan,
                "p05": math.nan, "p95": math.nan, "min": math.nan,
                "max": math.nan, "n": 0}
    return {
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "std": round(float(s.std(ddof=0)), 6),
        "p05": round(float(s.quantile(0.05)), 6),
        "p95": round(float(s.quantile(0.95)), 6),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "n": int(s.size),
    }


def _summarise(history: pd.DataFrame, require_status: str) -> dict[str, Any]:
    """Compute global + per-bucket aggregates from a drift history slice."""
    df = history.copy()
    if require_status and require_status != "all":
        df = df.loc[df["drift/status"].astype(str) == require_status]
    if df.empty:
        raise RuntimeError(
            f"No rows survive status filter ={require_status!r}. "
            f"Available: {sorted(history['drift/status'].dropna().unique().tolist())}. "
            "Pass --require-status all to disable filtering."
        )

    global_stats = {
        "deadband": _agg_stats(df["drift/deadband_frac_live"]),
        "saturation": _agg_stats(df["drift/saturation_frac_live"]),
    }

    by_bucket: dict[str, dict[str, Any]] = {}
    if "drift/bucket" in df.columns:
        for bucket, sub in df.groupby("drift/bucket"):
            if not isinstance(bucket, str) or bucket not in QUARTILES:
                continue
            by_bucket[bucket] = {
                "deadband": _agg_stats(sub["drift/deadband_frac_live"]),
                "saturation": _agg_stats(sub["drift/saturation_frac_live"]),
            }

    return {
        "n_emissions": int(len(df)),
        "step_range_observed": [int(df["_step"].min()), int(df["_step"].max())],
        "global": global_stats,
        "by_vol_quartile": by_bucket,
    }


def _patch_baseline(
    baseline: dict,
    summary: dict,
    *,
    bucket_strategy: str,
) -> tuple[dict, dict]:
    """Returns (patched_baseline, what_changed). bucket_strategy = 'observed'
    (use per-bucket live mean if live data present for that bucket; else fall
    back to global) or 'uniform' (use global mean for all buckets — matches
    the S535-R1 manual recal where live std was tiny across regimes).
    """
    cfg = copy.deepcopy(baseline)
    eed = cfg.get("ensemble_eval_distribution")
    if not isinstance(eed, dict):
        raise KeyError(
            "Baseline missing top-level 'ensemble_eval_distribution' block; "
            "this helper only handles ensemble baselines (Stage 2.5 schema)."
        )

    what_changed: dict[str, list[float | None]] = {}

    def _set(path: list[str], new: float, container: dict) -> None:
        node: Any = container
        for k in path[:-1]:
            node = node[k]
        last = path[-1]
        old = node.get(last) if isinstance(node, dict) else node[last]
        if old is None:
            old = math.nan
        node[last] = round(float(new), 6)
        what_changed[".".join(path)] = [
            round(float(old), 6) if old == old else None,
            round(float(new), 6),
        ]

    g_dead = summary["global"]["deadband"]["mean"]
    g_sat = summary["global"]["saturation"]["mean"]
    _set(["ensemble_eval_distribution", "deadband_frac"], g_dead, cfg)
    _set(["ensemble_eval_distribution", "saturation_frac"], g_sat, cfg)

    bvq = eed.get("by_vol_quartile") or {}
    for q in QUARTILES:
        if q not in bvq:
            continue
        bucket_obs = summary["by_vol_quartile"].get(q) if bucket_strategy == "observed" else None
        dead = (bucket_obs or {}).get("deadband", {}).get("mean")
        sat = (bucket_obs or {}).get("saturation", {}).get("mean")
        if dead is None or (isinstance(dead, float) and math.isnan(dead)):
            dead = g_dead
        if sat is None or (isinstance(sat, float) and math.isnan(sat)):
            sat = g_sat
        _set(["ensemble_eval_distribution", "by_vol_quartile", q, "deadband_frac"], dead, cfg)
        _set(["ensemble_eval_distribution", "by_vol_quartile", q, "saturation_frac"], sat, cfg)

    return cfg, what_changed


def _attach_metadata(
    cfg: dict,
    *,
    source_run: str,
    step_range: tuple[int, int],
    summary: dict,
    what_changed: dict,
    reason: str,
    require_status: str,
    bucket_strategy: str,
    tag: str,
) -> None:
    cfg["live_recal_metadata"] = {
        "recalibrated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "recalibrated_by": "scripts/recal_drift_baseline.py",
        "reason": reason,
        "source_run": source_run,
        "source_step_range": list(step_range),
        "step_range_observed": summary["step_range_observed"],
        "n_drift_emissions": summary["n_emissions"],
        "require_status": require_status,
        "bucket_strategy": bucket_strategy,
        "live_global_stats": summary["global"],
        "live_by_vol_quartile_stats": summary["by_vol_quartile"],
        "what_changed": what_changed,
        "what_preserved": [
            "per_seed_eval_distribution.* (audit trail of original baseline)",
            "ensemble_eval_distribution.histogram_bins / counts / mean / std / entropy (informational)",
            "regime_cutpoints (vol bucketing thresholds unchanged)",
            "challenge_target_hit_rate_by_seed",
            "ensemble_challenge_target_hit_rate",
        ],
        "monthly_recal_runbook_followup": (
            "Re-run with --source-run pointing at the most recent paper run "
            "and --step-range covering the last ~1500 OK drift emissions. "
            "Auto-trigger watcher (S535-R3 part b) is the next deliverable."
        ),
    }
    proto = cfg.get("protocol", "")
    if "__live_recal" not in proto:
        cfg["protocol"] = f"{proto}__live_recal_{tag}" if proto else f"live_recal_{tag}"


def _print_diff(what_changed: dict, summary: dict) -> None:
    print("\n=== Live aggregates ===")
    g = summary["global"]
    print(f"  n_emissions: {summary['n_emissions']}  "
          f"step_range: {summary['step_range_observed']}")
    print(f"  deadband_frac: mean={g['deadband']['mean']:.4f} "
          f"std={g['deadband']['std']:.4f} "
          f"p05={g['deadband']['p05']:.4f} p95={g['deadband']['p95']:.4f} "
          f"n={g['deadband']['n']}")
    print(f"  saturation_frac: mean={g['saturation']['mean']:.4f} "
          f"max={g['saturation']['max']:.4f} n={g['saturation']['n']}")
    if summary["by_vol_quartile"]:
        print("  per-bucket:")
        for q in QUARTILES:
            b = summary["by_vol_quartile"].get(q)
            if not b:
                continue
            print(f"    {q}: deadband_mean={b['deadband']['mean']:.4f} "
                  f"sat_mean={b['saturation']['mean']:.4f} "
                  f"n={b['deadband']['n']}")
    print("\n=== Patches ===")
    for k, (old, new) in what_changed.items():
        old_s = f"{old:.4f}" if isinstance(old, (int, float)) and old is not None else str(old)
        new_s = f"{new:.4f}" if isinstance(new, (int, float)) and new is not None else str(new)
        print(f"  {k}: {old_s} -> {new_s}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True,
                   help="ensemble_report.json to patch (modified in place unless --output)")
    p.add_argument("--source-run", required=True,
                   help="Full WandB run path, e.g. 'bigcan-chiwin-technology/FinRL-Pro-DS/<id>'")
    p.add_argument("--step-range", type=_parse_step_range, required=True,
                   help="Comma-separated _step bounds: 'start,end' (inclusive)")
    p.add_argument("--reason", required=True,
                   help="One-line justification recorded in live_recal_metadata.reason")
    p.add_argument("--require-status", default="OK",
                   choices=("OK", "WARN", "CRIT", "WARMUP", "all"),
                   help="Filter drift emissions by status before aggregating (default: OK)")
    p.add_argument("--bucket-strategy", default="observed",
                   choices=("observed", "uniform"),
                   help="'observed': per-bucket live mean (fallback to global if missing). "
                        "'uniform': use global mean for all buckets (matches S535-R1 sg1-xauusd).")
    p.add_argument("--tag", default=None,
                   help="Short marker appended to protocol field (default: today UTC date)")
    p.add_argument("--output", type=Path, default=None,
                   help="Override target path; default = --baseline (in-place)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print diff but do not write")
    args = p.parse_args()

    if not args.baseline.is_file():
        print(f"ERROR: baseline not found: {args.baseline}", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    print(f"Fetching drift/* history from {args.source_run} "
          f"_step in {list(args.step_range)} ...")
    history = _fetch_drift_history(args.source_run, args.step_range)
    print(f"  {len(history)} rows retrieved.")

    summary = _summarise(history, require_status=args.require_status)
    patched, what_changed = _patch_baseline(
        baseline, summary, bucket_strategy=args.bucket_strategy,
    )

    tag = args.tag or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    _attach_metadata(
        patched,
        source_run=args.source_run,
        step_range=args.step_range,
        summary=summary,
        what_changed=what_changed,
        reason=args.reason,
        require_status=args.require_status,
        bucket_strategy=args.bucket_strategy,
        tag=tag,
    )

    _print_diff(what_changed, summary)

    if args.dry_run:
        print("\n[dry-run] no file written")
        return 0

    out = args.output or args.baseline
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patched, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
