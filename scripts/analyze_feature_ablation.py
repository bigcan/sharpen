#!/usr/bin/env python
"""GMGP1-BTC Feature-Ablation — analysis / verdict.

Reads the dispatcher manifest (results/feature_ablation_<ts>/manifest.json) and
the pre-registered gates (configs/gmgp1_btc_feature_ablation.gates.yaml), fetches
each rung's per-seed test metrics from WandB (tag-filtered), and evaluates:

  baseline_reference : r0 median test PF = THE feature-engineering baseline.
  H1 feature_monotonic : Kendall-tau on median test PF across r0..r4.
  H2 multiscale_value  : MWU(r5 vs r4) two-sided — do coarse scales earn keep?
  H3 per_feature_sig   : MWU(rung_k vs rung_{k-1}) — is each feature justified?
  marginal table       : median PF delta of each rung vs r0.
  anchor               : r5 (= canary feature set) median PF in expected band.

WandB extraction mirrors scripts/analyze_volume_study.py (separate_runs flat
keys). Writes <results_dir>/feature_ablation_report.md + verdict.json.

Usage:
    python scripts/analyze_feature_ablation.py \
        --results_dir results/feature_ablation_20260616_180000 \
        --study_id 20260616_180000
    python scripts/analyze_feature_ablation.py --selftest   # stats sanity, no WandB
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATES = PROJECT_ROOT / "configs" / "gmgp1_btc_feature_ablation.gates.yaml"
WORKSTREAM_TAG = "feature-ablation"

# Flat WandB summary keys (backtest_<split>/<key>) — identical to analyze_volume_study.
SUMMARY_KEYS = ("profit_factor", "sharpe_daily", "sortino", "max_drawdown", "total_return")
PF = "profit_factor"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] analyze_featabl - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("analyze_featabl")

_SEED_FROM_NAME_RE = re.compile(r"-seed(\d+)_")


# ---------------------------------------------------------------------------
# WandB fetch + extract (mirrors analyze_volume_study.py, separate_runs path)
# ---------------------------------------------------------------------------
def fetch_runs_for_rung(api, project: str, entity: str, rung_id: str,
                        study_id: str) -> list:
    filters = {"$and": [
        {"tags": {"$in": [WORKSTREAM_TAG]}},
        {"tags": {"$in": [f"rung-{rung_id}"]}},
        {"tags": {"$in": [f"study-id-{study_id}"]}},
    ]}
    return list(api.runs(f"{entity}/{project}", filters=filters))


def _extract_separate(run, seed: int) -> dict | None:
    summary = dict(run.summary_metrics)
    row: dict = {"seed": seed, "run_id": run.id, "run_name": run.name}
    found = False
    for split in ("val", "test"):
        for key in SUMMARY_KEYS:
            flat = f"backtest_{split}/{key}"
            if flat in summary:
                row[f"{split}_{key}"] = summary[flat]
                found = True
    return row if found else None


def extract_seed_summaries(runs, seeds: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in runs:
        m = _SEED_FROM_NAME_RE.search(r.name)
        if m and int(m.group(1)) in seeds:
            row = _extract_separate(r, int(m.group(1)))
            if row is not None:
                out[int(m.group(1))] = row
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None and x == x)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _cv(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    if m == 0:
        return float("nan")
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (var ** 0.5) / abs(m)


def _mwu(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U two-sided. Returns (U, p). NaN if degenerate."""
    a = [x for x in a if x is not None and x == x]
    b = [x for x in b if x is not None and x == x]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    from scipy.stats import mannwhitneyu
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return float("nan"), float("nan")


def evaluate_h1(rung_pfs: dict[str, list[float]], rung_order: list[str],
                gates: dict) -> dict:
    """Kendall-tau on (rung index, median test PF) for the single-scale ladder."""
    from scipy.stats import kendalltau
    idx, medians = [], []
    for i, rid in enumerate(rung_order):
        med = _median(rung_pfs.get(rid, []))
        idx.append(i)
        medians.append(med)
    h1 = next((h for h in [gates["hypotheses"]["h1_feature_monotonic"]]), {})
    if any(m != m for m in medians):  # NaN present
        return {"status": "INCOMPLETE", "medians": dict(zip(rung_order, medians))}
    tau, p = kendalltau(idx, medians)
    tau_min = float(h1.get("tau_min", 0.6))
    p_max = float(h1.get("p_max", 0.10))
    status = "PASS" if (tau >= tau_min and p <= p_max) else (
        "AMBIGUOUS" if tau >= tau_min else "FAIL")
    return {"status": status, "tau": round(tau, 4), "p": round(p, 4),
            "tau_min": tau_min, "p_max": p_max,
            "medians": {r: round(m, 4) for r, m in zip(rung_order, medians)}}


def evaluate_h2(rung_pfs: dict[str, list[float]], gates: dict) -> dict:
    h2 = gates["hypotheses"]["h2_multiscale_value"]
    hi, lo = h2["compare"]  # ["r5","r4"]
    a, b = rung_pfs.get(hi, []), rung_pfs.get(lo, [])
    med_a, med_b = _median(a), _median(b)
    u, p = _mwu(a, b)
    p_max = float(h2.get("p_max", 0.10))
    higher = med_a > med_b
    if med_a != med_a or med_b != med_b:
        status = "INCOMPLETE"
    elif higher and p <= p_max:
        status = "PASS"
    elif higher:
        status = "AMBIGUOUS"
    else:
        status = "FAIL"
    return {"status": status, "compare": f"{hi} vs {lo}",
            f"median_{hi}": round(med_a, 4), f"median_{lo}": round(med_b, 4),
            "delta": round(med_a - med_b, 4), "U": u, "p": round(p, 4) if p == p else p,
            "p_max": p_max}


def evaluate_h3(rung_pfs: dict[str, list[float]], rung_order: list[str],
                gates: dict) -> list[dict]:
    """MWU each rung_k vs rung_{k-1} along the full ladder (r0->r5)."""
    p_max = float(gates["significance"].get("p_max", 0.10))
    rows = []
    for i in range(1, len(rung_order)):
        cur, prev = rung_order[i], rung_order[i - 1]
        a, b = rung_pfs.get(cur, []), rung_pfs.get(prev, [])
        med_d = _median(a) - _median(b)
        u, p = _mwu(a, b)
        claim = "adds" if (med_d > 0 and p == p and p <= p_max) else (
            "neutral/decorative" if (p != p or p > p_max) else "hurts")
        rows.append({"step": f"{prev}->{cur}", "median_delta": round(med_d, 4),
                     "p": round(p, 4) if p == p else None, "verdict": claim})
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_verdict(gates: dict, rungs: list[dict], per_rung: dict) -> dict:
    rung_order = [r["id"] for r in rungs]
    rung_pfs = {rid: [s.get(f"test_{PF}") for s in per_rung.get(rid, {}).values()]
                for rid in rung_order}
    single_scale = [r["id"] for r in rungs if len(r["scales"]) == 1]  # r0..r4

    baseline_pf = _median(rung_pfs.get("r0", []))
    table = []
    for r in rungs:
        rid = r["id"]
        pfs = rung_pfs.get(rid, [])
        med = _median(pfs)
        table.append({
            "rung": rid, "label": r["label"], "n_feat": len(r["feature_indices"]),
            "n_scale": len(r["scales"]), "n_seeds": len(per_rung.get(rid, {})),
            "median_test_pf": round(med, 4) if med == med else None,
            "delta_vs_r0": round(med - baseline_pf, 4) if (med == med and baseline_pf == baseline_pf) else None,
            "cv_pf": round(_cv(pfs), 4) if _cv(pfs) == _cv(pfs) else None,
        })

    anchor_band = [0.95, 1.05]
    r5_med = _median(rung_pfs.get("r5", []))
    anchor = {"r5_median_pf": round(r5_med, 4) if r5_med == r5_med else None,
              "expected_band": anchor_band,
              "in_band": (anchor_band[0] <= r5_med <= anchor_band[1]) if r5_med == r5_med else None}

    return {
        "workstream": gates.get("workstream"),
        "baseline_reference_r0_median_pf": round(baseline_pf, 4) if baseline_pf == baseline_pf else None,
        "marginal_table": table,
        "h1_feature_monotonic": evaluate_h1(rung_pfs, single_scale, gates),
        "h2_multiscale_value": evaluate_h2(rung_pfs, gates),
        "h3_per_feature": evaluate_h3(rung_pfs, rung_order, gates),
        "anchor_r5_vs_canary": anchor,
    }


def write_report(out_dir: Path, verdict: dict, per_rung: dict, rungs: list[dict]) -> None:
    md = ["# GMGP1-BTC Feature-Ablation — Verdict\n",
          f"Baseline (r0 median test PF) = **{verdict['baseline_reference_r0_median_pf']}**",
          "(expected near-breakeven on clean BTC — the reference, not a target).\n",
          "## Marginal contribution table\n",
          "| Rung | Features | Scales | Seeds | Median test PF | Δ vs r0 | CV(PF) |",
          "|------|----------|--------|-------|----------------|---------|--------|"]
    for row in verdict["marginal_table"]:
        md.append(f"| {row['rung']} {row['label']} | {row['n_feat']} | {row['n_scale']} "
                  f"| {row['n_seeds']} | {row['median_test_pf']} | {row['delta_vs_r0']} "
                  f"| {row['cv_pf']} |")
    h1 = verdict["h1_feature_monotonic"]
    h2 = verdict["h2_multiscale_value"]
    md += [f"\n## H1 feature_monotonic (r0->r4): **{h1.get('status')}**",
           f"- Kendall tau={h1.get('tau')} (min {h1.get('tau_min')}), p={h1.get('p')} (max {h1.get('p_max')})",
           f"- medians: {h1.get('medians')}",
           f"\n## H2 multiscale_value ({h2.get('compare')}): **{h2.get('status')}**",
           f"- delta={h2.get('delta')}  p={h2.get('p')} (max {h2.get('p_max')})",
           "\n## H3 per-feature significance"]
    md.append("| Step | Median ΔPF | p | Verdict |")
    md.append("|------|-----------|---|---------|")
    for r in verdict["h3_per_feature"]:
        md.append(f"| {r['step']} | {r['median_delta']} | {r['p']} | {r['verdict']} |")
    anc = verdict["anchor_r5_vs_canary"]
    md.append(f"\n## Anchor: r5 (canary feature set) median PF = {anc['r5_median_pf']} "
              f"— in band {anc['expected_band']}? **{anc['in_band']}**")
    (out_dir / "feature_ablation_report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
def _selftest() -> int:
    """Stats sanity with synthetic data (no WandB)."""
    rungs = [{"id": f"r{i}", "label": f"l{i}",
              "feature_indices": list(range(i + 1)),
              "scales": [15] if i < 5 else [15, 60, 240]} for i in range(6)]
    # monotone-increasing PF r0..r4, r5 > r4
    base = {"r0": [0.98, 0.99, 1.00, 0.97, 1.01], "r1": [1.02, 1.05, 1.03, 1.04, 1.06],
            "r2": [1.08, 1.10, 1.07, 1.09, 1.11], "r3": [1.12, 1.14, 1.13, 1.15, 1.11],
            "r4": [1.16, 1.18, 1.15, 1.17, 1.19], "r5": [1.28, 1.30, 1.27, 1.31, 1.29]}
    per_rung = {rid: {seed: {f"test_{PF}": v for k, v in [("", pv)]}
                      for seed, pv in zip([123, 456, 789, 1024, 2026], pfs)}
                for rid, pfs in base.items()}
    gates = yaml.safe_load(DEFAULT_GATES.read_text(encoding="utf-8"))
    v = build_verdict(gates, rungs, per_rung)
    assert v["h1_feature_monotonic"]["status"] == "PASS", v["h1_feature_monotonic"]
    assert v["h2_multiscale_value"]["status"] in ("PASS", "AMBIGUOUS"), v["h2_multiscale_value"]
    assert v["baseline_reference_r0_median_pf"] == 0.99
    log.info("selftest PASS: H1=%s H2=%s baseline=%s",
             v["h1_feature_monotonic"]["status"], v["h2_multiscale_value"]["status"],
             v["baseline_reference_r0_median_pf"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path, help="results/feature_ablation_<ts>/")
    p.add_argument("--study_id", help="study-id tag (default: read from manifest)")
    p.add_argument("--gates_file", type=Path, default=DEFAULT_GATES)
    p.add_argument("--project", default="FinRL-Pro-DS")
    p.add_argument("--entity", default="bigcan-chiwin-technology")
    p.add_argument("--selftest", action="store_true", help="Run stats self-test, no WandB.")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if not args.results_dir:
        log.error("--results_dir required (or use --selftest)")
        return 2

    gates = yaml.safe_load(args.gates_file.read_text(encoding="utf-8"))
    rungs = list(gates["rungs"])
    seeds = [int(s) for s in gates["seeds"]]
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    study_id = args.study_id or manifest.get("study_id")
    if not study_id:
        log.error("no study_id (pass --study_id or ensure manifest has it)")
        return 2
    log.info("analyzing study_id=%s, %d rungs", study_id, len(rungs))

    import wandb
    api = wandb.Api(timeout=60)
    per_rung: dict[str, dict] = {}
    for r in rungs:
        runs = fetch_runs_for_rung(api, args.project, args.entity, r["id"], study_id)
        per_rung[r["id"]] = extract_seed_summaries(runs, seeds)
        log.info("rung=%s: %d/%d seeds with metrics", r["id"],
                 len(per_rung[r["id"]]), len(seeds))

    verdict = build_verdict(gates, rungs, per_rung)
    write_report(args.results_dir, verdict, per_rung, rungs)
    log.info("baseline r0 PF=%s | H1=%s | H2=%s | report=%s",
             verdict["baseline_reference_r0_median_pf"],
             verdict["h1_feature_monotonic"]["status"],
             verdict["h2_multiscale_value"]["status"],
             args.results_dir / "feature_ablation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
