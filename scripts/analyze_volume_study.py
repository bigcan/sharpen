#!/usr/bin/env python
"""GMGP1 Training-Volume Study — analyzer.

Reads the manifest written by `scripts/run_volume_study.py`, pulls the per-budget
WandB runs, extracts per-seed test/val PF + DD + learning curves, and evaluates
the pre-registered hypotheses (H1-H4) declared in
`configs/gmgp1_volume_study.gates.yaml`.

Outputs:
    - results/volume_study_<ts>/verdict.json       — machine-readable
    - results/volume_study_<ts>/report.md          — markdown summary
    - results/volume_study_<ts>/learning_curves/   — per-budget PNGs (optional)

WandB layout assumed (per S488 consolidation):
    one parent run per budget (tag "budget-<label>"), with child seeds
    namespaced as `seed<N>/*`. Metric keys per CLAUDE.md backtest keys:
    Profit_Factor_Daily, Sharpe_Ratio, Sortino_Ratio, Max_Drawdown.

Usage:
    python scripts/analyze_volume_study.py \
        --manifest results/volume_study_20260426_222604/manifest.json \
        [--no_plots] [--write_randd]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_volume_study.gates.yaml"

# Backtest metric keys actually emitted by the V7 SAC pipeline. CLAUDE.md
# documents the legacy capitalized names (Profit_Factor_Daily etc), but
# `run_full_pipeline.py` writes lowercase under `seed{N}/backtest_{val,test}/`.
# Verified empirically against the volume study smoke run (`ou69yn1r`).
# `sharpe_daily` is preferred over `sharpe` — the raw value is minute-level
# and inflated by autocorrelation; the daily resample is the trustworthy view.
SUMMARY_KEYS = (
    "profit_factor",
    "sharpe_daily",
    "sortino",
    "max_drawdown",
    "total_return",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] analyze - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("analyze_volume_study")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gates() -> dict:
    with GATES_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_runs_for_budget(api, project: str, entity: str, label: str,
                          study_id: str) -> list:
    """Find WandB runs tagged for a specific budget label of this study run.

    Filters by three tags: `volume-study` (workstream), `budget-<label>`
    (sweep level), and `study-id-<ts>` (disambiguates re-launches of the
    same study). All three are written by run_volume_study.py into the
    derived configs, so the parent launcher run inherits them.
    """
    filters = {
        "$and": [
            {"tags": {"$in": ["volume-study"]}},
            {"tags": {"$in": [f"budget-{label}"]}},
            {"tags": {"$in": [f"study-id-{study_id}"]}},
        ]
    }
    return list(api.runs(f"{entity}/{project}", filters=filters))


_SEED_FROM_NAME_RE = re.compile(r"-seed(\d+)_")


def _extract_from_one_run_consolidated(run, seeds: list[int]) -> dict[int, dict]:
    """Consolidated parent run: per-seed namespaced keys (seed<N>/...)."""
    out: dict[int, dict] = {}
    summary = dict(run.summary_metrics)
    for s in seeds:
        ns = f"seed{s}/"
        row = {"seed": s, "run_id": run.id, "run_name": run.name}
        any_found = False
        for key in SUMMARY_KEYS:
            ns_key = f"{ns}{key}"
            if ns_key in summary:
                row[key] = summary[ns_key]
                any_found = True
            elif key in summary and len(seeds) == 1:
                row[key] = summary[key]
                any_found = True
            else:
                row[key] = None
        for split in ("val", "test"):
            for key in SUMMARY_KEYS:
                ns_key = f"{ns}backtest_{split}/{key}"
                if ns_key in summary:
                    row[f"{split}_{key}"] = summary[ns_key]
                    any_found = True
        if any_found:
            out[s] = row
    return out


def _extract_from_one_run_separate(run, seed: int) -> dict | None:
    """One seed run from --separate_runs mode: flat keys, no namespace."""
    summary = dict(run.summary_metrics)
    row = {"seed": seed, "run_id": run.id, "run_name": run.name}
    any_found = False
    for split in ("val", "test"):
        for key in SUMMARY_KEYS:
            flat_key = f"backtest_{split}/{key}"
            if flat_key in summary:
                row[f"{split}_{key}"] = summary[flat_key]
                any_found = True
    return row if any_found else None


def extract_seed_summaries(runs, seeds: list[int]) -> dict[int, dict]:
    """Pull per-seed summary metrics from EITHER:
      * one consolidated parent run (seed<N>/<metric> namespacing), OR
      * many separate-runs (one wandb run per seed, flat keys).

    Detected by run name pattern: `-seed<N>_` in the name → separate_runs entry.
    Backwards-compatible with single-run callers (older code paths pass `run`,
    not a list).
    """
    if not isinstance(runs, list):
        runs = [runs]
    if not runs:
        return {}

    # Classify each run by name pattern.
    separate_runs: dict[int, object] = {}
    consolidated_runs: list = []
    for r in runs:
        m = _SEED_FROM_NAME_RE.search(r.name)
        if m and int(m.group(1)) in seeds:
            separate_runs[int(m.group(1))] = r
        else:
            consolidated_runs.append(r)

    out: dict[int, dict] = {}
    # Separate-runs mode: one run per seed, flat keys
    for seed, run in separate_runs.items():
        row = _extract_from_one_run_separate(run, seed)
        if row is not None:
            out[seed] = row
    # Consolidated mode: namespace-extract from any non-seed-named run
    # (typically only one such run per budget — the parent).
    for run in consolidated_runs:
        merged = _extract_from_one_run_consolidated(run, seeds)
        for s, row in merged.items():
            # Don't overwrite separate-runs results (those are higher-fidelity)
            out.setdefault(s, row)
    return out


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _cv(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    if m == 0:
        return float("nan")
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (var ** 0.5) / abs(m)


def evaluate_h1_monotonicity(per_budget_test_pf: dict[int, list[float]],
                             gates: dict) -> dict:
    """Kendall-tau on (budget, median_test_pf) pairs."""
    from scipy.stats import kendalltau

    h1 = gates["hypotheses"]["h1_monotonic_improvement"]
    budgets_sorted = sorted(per_budget_test_pf.keys())
    medians = [_median(per_budget_test_pf[b]) for b in budgets_sorted]

    if any(m != m for m in medians):  # NaN check
        return {"verdict": "AMBIGUOUS", "reason": "missing test PF for some budget",
                "medians": dict(zip(budgets_sorted, medians))}

    tau, p = kendalltau(budgets_sorted, medians)
    passed = (tau >= h1["tau_min"]) and (p <= h1["p_max"])
    return {
        "verdict": "PASS" if passed else "FAIL",
        "tau": float(tau), "p": float(p),
        "tau_min": h1["tau_min"], "p_max": h1["p_max"],
        "medians": dict(zip(budgets_sorted, medians)),
    }


def evaluate_h2_variance(per_budget_test_pf: dict[int, list[float]]) -> dict:
    """CV(PF) at largest budget <= CV(PF) at smallest."""
    budgets_sorted = sorted(per_budget_test_pf.keys())
    if len(budgets_sorted) < 2:
        return {"verdict": "AMBIGUOUS", "reason": "need >=2 budgets"}
    cv_min = _cv(per_budget_test_pf[budgets_sorted[0]])
    cv_max = _cv(per_budget_test_pf[budgets_sorted[-1]])
    return {
        "verdict": "PASS" if cv_max <= cv_min else "FAIL",
        "cv_smallest_budget": cv_min, "cv_largest_budget": cv_max,
        "smallest_budget": budgets_sorted[0], "largest_budget": budgets_sorted[-1],
    }


def evaluate_h3_overfit(per_budget_val_pf: dict[int, list[float]],
                        per_budget_test_pf: dict[int, list[float]],
                        gates: dict) -> dict:
    """val/test gap at largest budget > ratio_threshold * gap at smallest."""
    h3 = gates["hypotheses"]["h3_overfitting_onset"]
    budgets_sorted = sorted(per_budget_test_pf.keys())
    if not per_budget_val_pf or any(b not in per_budget_val_pf for b in budgets_sorted):
        return {"verdict": "AMBIGUOUS", "reason": "val PF unavailable"}
    smallest, largest = budgets_sorted[0], budgets_sorted[-1]
    gap_small = _median(per_budget_val_pf[smallest]) - _median(per_budget_test_pf[smallest])
    gap_large = _median(per_budget_val_pf[largest]) - _median(per_budget_test_pf[largest])
    if gap_small <= 0:
        return {"verdict": "AMBIGUOUS",
                "reason": f"val<=test at smallest budget ({smallest}); gap={gap_small:.3f}",
                "gap_smallest": gap_small, "gap_largest": gap_large}
    ratio = gap_large / gap_small
    overfit = ratio > h3["ratio_threshold"]
    return {
        "verdict": "OVERFIT_FLAG" if overfit else "OK",
        "gap_smallest": gap_small, "gap_largest": gap_large,
        "ratio": ratio, "ratio_threshold": h3["ratio_threshold"],
    }


def evaluate_h4_diminishing(per_budget_test_pf: dict[int, list[float]]) -> dict:
    """Compare median(5M)-median(3M) vs median(2M)-median(1M)."""
    budgets = sorted(per_budget_test_pf.keys())
    needed = (1_000_000, 2_000_000, 3_000_000, 5_000_000)
    if not all(b in per_budget_test_pf for b in needed):
        return {"verdict": "AMBIGUOUS",
                "reason": f"H4 needs {needed}; got {budgets}"}
    early = _median(per_budget_test_pf[2_000_000]) - _median(per_budget_test_pf[1_000_000])
    late = _median(per_budget_test_pf[5_000_000]) - _median(per_budget_test_pf[3_000_000])
    return {
        "verdict": "DIMINISHING_RETURNS" if late < early else "STILL_GROWING",
        "early_gain_1m_to_2m": early,
        "late_gain_3m_to_5m": late,
    }


# ---------------------------------------------------------------------------
# Anomaly checks
# ---------------------------------------------------------------------------

def detect_buffer_saturation(run, seeds: list[int], gates: dict) -> list[dict]:
    """Per-seed critic_loss blowup detector for the 5M budget."""
    flags = []
    anomaly = gates["anomalies"]["buffer_saturation"]
    blowup_step = anomaly["blowup_check_step"]
    blowup_ratio = anomaly["critic_loss_blowup_ratio"]

    for s in seeds:
        keys = [f"seed{s}/train/total_steps", f"seed{s}/train/critic_loss"]
        try:
            hist = list(run.scan_history(keys=keys, page_size=2000))
        except Exception as e:
            log.warning("seed %d: history scan failed (%s)", s, e)
            continue
        if not hist:
            continue
        # Find baseline at step ~blowup_step and compare to final
        before = [h for h in hist if h.get(keys[0], 0) <= blowup_step
                  and h.get(keys[1]) is not None]
        after = [h for h in hist if h.get(keys[0], 0) > blowup_step
                 and h.get(keys[1]) is not None]
        if not before or not after:
            continue
        baseline = before[-1][keys[1]]
        peak_after = max(h[keys[1]] for h in after)
        if baseline > 0 and peak_after > blowup_ratio * baseline:
            flags.append({
                "seed": s,
                "baseline_critic_loss": baseline,
                "peak_after_blowup_step": peak_after,
                "ratio": peak_after / baseline,
            })
    return flags


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(out_dir: Path, manifest: dict, gates: dict,
                 per_seed: list[dict], verdict: dict) -> Path:
    """Markdown summary appendable to randd_log.md."""
    md = []
    md.append(f"# GMGP1 Volume Study — {manifest['timestamp']}\n")
    md.append(f"- Test bed: `{gates['test_bed']}` ({gates['asset']})")
    md.append(f"- Budgets: {gates['step_budgets']}")
    md.append(f"- Seeds: {gates['seeds']} (N={gates['n_seeds']})")
    md.append(f"- Fee model: `{gates['fee_model']}`")
    md.append(f"- Instance: `{manifest['instance']}` (gpu={manifest['gpu']}, "
              f"concurrent={manifest['concurrent']})\n")

    md.append("## Per-seed summary\n")
    md.append("| Budget | Seed | Test PF | Test Sharpe (daily) | Test MDD | Val PF |")
    md.append("|---|---|---|---|---|---|")
    for r in sorted(per_seed, key=lambda x: (x["budget"], x["seed"])):
        md.append(f"| {r['budget']:>7} | {r['seed']} | "
                  f"{r.get('test_profit_factor','—'):.3f} | "
                  f"{r.get('test_sharpe_daily','—')} | "
                  f"{r.get('test_max_drawdown','—')} | "
                  f"{r.get('val_profit_factor','—')} |"
                  if isinstance(r.get('test_profit_factor'), (int, float))
                  else f"| {r['budget']:>7} | {r['seed']} | n/a | n/a | n/a | n/a |")

    md.append("\n## Hypothesis verdicts\n")
    for hid, vres in verdict["hypotheses"].items():
        md.append(f"### {hid}: **{vres['verdict']}**")
        for k, v in vres.items():
            if k == "verdict":
                continue
            md.append(f"- {k}: {v}")
        md.append("")

    if verdict.get("anomalies"):
        md.append("## Anomalies flagged\n")
        for a in verdict["anomalies"]:
            md.append(f"- `{a}`")

    md.append(f"\n## Provenance\n")
    md.append(f"- Plan: `{gates['provenance']['plan_file']}`")
    md.append(f"- Upstream HP run: `{gates['provenance']['upstream_hp_run']}`")
    md.append(f"- Manifest: `results/volume_study_{manifest['timestamp']}/manifest.json`")

    out = out_dir / "report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--entity", default="bigcan-chiwin-technology")
    p.add_argument("--project", default="FinRL-Pro-DS")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip learning-curve plots (faster)")
    p.add_argument("--write_randd", action="store_true",
                   help="Append the report to randd_log.md")
    args = p.parse_args()

    if not args.manifest.exists():
        log.error("manifest not found: %s", args.manifest)
        return 2

    manifest = load_manifest(args.manifest)
    gates = load_gates()
    out_dir = args.manifest.parent

    if manifest.get("dry_run"):
        log.error("manifest is from a --dry_run dispatch; no data to analyze")
        return 2

    # Lazy import — wandb is heavy and we want validate-mode to work without it.
    import wandb
    api = wandb.Api(timeout=60)

    seeds = list(manifest["seeds"])
    per_seed_rows: list[dict] = []
    per_budget_test_pf: dict[int, list[float]] = {}
    per_budget_val_pf: dict[int, list[float]] = {}
    anomalies: list[str] = []

    for run_meta in manifest["runs"]:
        if run_meta["exit_code"] != 0:
            log.warning("budget=%s skipped (dispatcher exit_code=%d)",
                        run_meta["label"], run_meta["exit_code"])
            continue
        budget = run_meta["budget"]
        label = run_meta["label"]
        runs = fetch_runs_for_budget(api, args.project, args.entity, label,
                                     study_id=manifest["timestamp"])
        if not runs:
            log.warning("budget=%s: no WandB runs found", label)
            continue

        log.info("budget=%s: %d run(s) found", label, len(runs))
        for r in runs:
            log.info("  - %s (%s)", r.name, r.id)
        seed_rows = extract_seed_summaries(runs, seeds)

        for s, row in seed_rows.items():
            row["budget"] = budget
            row["budget_label"] = label
            per_seed_rows.append(row)
            if row.get("test_profit_factor") is not None:
                per_budget_test_pf.setdefault(budget, []).append(
                    row["test_profit_factor"])
            if row.get("val_profit_factor") is not None:
                per_budget_val_pf.setdefault(budget, []).append(
                    row["val_profit_factor"])

        # Buffer saturation only meaningful at the highest budget(s).
        # Use the first available run for history scan; in separate_runs mode
        # we'd want per-seed scans but the anomaly detector currently expects
        # consolidated namespacing — skip cleanly if no consolidated run found.
        if budget >= 2_000_000:
            consolidated = next((r for r in runs if not _SEED_FROM_NAME_RE.search(r.name)), None)
            buf_flags = detect_buffer_saturation(consolidated, seeds, gates) if consolidated else []
            for f in buf_flags:
                anomalies.append(
                    f"buffer_saturation: budget={label} seed={f['seed']} "
                    f"critic_loss {f['baseline_critic_loss']:.3f} -> "
                    f"{f['peak_after_blowup_step']:.3f} ({f['ratio']:.1f}x)"
                )

    # Hypothesis evaluation
    verdict: dict[str, Any] = {
        "timestamp": manifest["timestamp"],
        "hypotheses": {
            "h1_monotonic_improvement": evaluate_h1_monotonicity(
                per_budget_test_pf, gates),
            "h2_variance_tightens": evaluate_h2_variance(per_budget_test_pf),
            "h3_overfitting_onset": evaluate_h3_overfit(
                per_budget_val_pf, per_budget_test_pf, gates),
            "h4_diminishing_returns": evaluate_h4_diminishing(per_budget_test_pf),
        },
        "anomalies": anomalies,
        "n_seed_rows": len(per_seed_rows),
    }

    # Persist
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    (out_dir / "per_seed.json").write_text(
        json.dumps(per_seed_rows, indent=2, default=str), encoding="utf-8")
    report_path = write_report(out_dir, manifest, gates, per_seed_rows, verdict)

    log.info("verdict: %s", json.dumps(
        {k: v.get("verdict") for k, v in verdict["hypotheses"].items()}))
    log.info("report:  %s", report_path.relative_to(PROJECT_ROOT))
    log.info("verdict json: %s", (out_dir / "verdict.json").relative_to(PROJECT_ROOT))

    if args.write_randd:
        randd = PROJECT_ROOT / "randd_log.md"
        with randd.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(report_path.read_text(encoding="utf-8"))
        log.info("appended report to %s", randd.relative_to(PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
