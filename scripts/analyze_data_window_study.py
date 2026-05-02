#!/usr/bin/env python
"""GMGP1 Training-Data-Window Study (Axis A) — analyzer.

Reads the manifest written by `scripts/run_data_window_study.py`, pulls per-cell
WandB runs (cell = (window, budget)), extracts per-seed test/val PF + DD, and
evaluates the pre-registered hypotheses (G1-G6) declared in
`configs/gmgp1_data_window_study.gates.yaml`. Maps verdicts to the decision
tree (STRONG_GO / GO / PARTIAL_GO / DEFER / NO_GO).

Outputs:
    - results/data_window_study_<ts>/verdict.json            — machine-readable
    - results/data_window_study_<ts>/per_seed.json           — flat per-seed rows
    - results/data_window_study_<ts>/pf_surface.json         — 2D window x budget
    - results/data_window_study_<ts>/report.md               — markdown summary

WandB layout assumed (per S488 consolidation + run_data_window_study.py
--separate_runs default): one wandb run per seed, tagged with
  `data-window-study`, `window-<slug>`, `budget-<label>`, `study-id-<ts>`.

Usage:
    python scripts/analyze_data_window_study.py \
        --manifest results/data_window_study_20260430_120000/manifest.json \
        [--write_randd]
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
GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_data_window_study.gates.yaml"

# Backtest metric keys actually emitted by V7 SAC pipeline (lowercase under
# `seed{N}/backtest_{val,test}/`). Same as Volume Study analyzer; verified
# against volume study smoke run `ou69yn1r`.
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
log = logging.getLogger("analyze_data_window_study")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gates() -> dict:
    with GATES_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_runs_for_cell(api, project: str, entity: str,
                        window_slug: str, budget_label: str,
                        study_id: str) -> list:
    """Find WandB runs tagged for a specific (window, budget) cell of this study."""
    filters = {
        "$and": [
            {"tags": {"$in": ["data-window-study"]}},
            {"tags": {"$in": [f"window-{window_slug}"]}},
            {"tags": {"$in": [f"budget-{budget_label}"]}},
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
    """Pull per-seed summary metrics from EITHER consolidated parent runs OR
    --separate_runs (one run per seed). Detects via run-name pattern."""
    if not isinstance(runs, list):
        runs = [runs]
    if not runs:
        return {}

    separate_runs: dict[int, object] = {}
    consolidated_runs: list = []
    for r in runs:
        m = _SEED_FROM_NAME_RE.search(r.name)
        if m and int(m.group(1)) in seeds:
            separate_runs[int(m.group(1))] = r
        else:
            consolidated_runs.append(r)

    out: dict[int, dict] = {}
    for seed, run in separate_runs.items():
        row = _extract_from_one_run_separate(run, seed)
        if row is not None:
            out[seed] = row
    for run in consolidated_runs:
        merged = _extract_from_one_run_consolidated(run, seeds)
        for s, row in merged.items():
            out.setdefault(s, row)
    return out


# ---------------------------------------------------------------------------
# Stats utilities
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


def _isnan(x: float) -> bool:
    return x != x


# ---------------------------------------------------------------------------
# Hypothesis tests (G1-G6)
# ---------------------------------------------------------------------------

def evaluate_g1_direction(window_order: list[str],
                          window_months: dict[str, int],
                          per_window_test_pf_at_budget: dict[str, list[float]],
                          gates: dict) -> dict:
    """G1: Kendall-tau on (train_window_months, median_test_pf) for ONE budget.

    Sign UNKNOWN a priori — accept either +tau or -tau if |tau| >= 0.40 and
    p <= 0.10. The DIRECTION is reported in the verdict so downstream
    decision logic can route correctly.
    """
    from scipy.stats import kendalltau

    g1 = gates["hypotheses"]["g1_monotonic_direction"]
    months_sorted = [(window_months[w], w) for w in window_order
                     if w in per_window_test_pf_at_budget
                     and per_window_test_pf_at_budget[w]]
    months_sorted.sort()
    if len(months_sorted) < 3:
        return {"verdict": "AMBIGUOUS",
                "reason": f"need >=3 windows with PF data; got {len(months_sorted)}",
                "windows_present": [w for _, w in months_sorted]}

    months = [m for m, _ in months_sorted]
    medians = [_median(per_window_test_pf_at_budget[w]) for _, w in months_sorted]
    if any(_isnan(m) for m in medians):
        return {"verdict": "AMBIGUOUS",
                "reason": "missing test PF for some window",
                "medians_by_window": dict(zip([w for _, w in months_sorted], medians))}

    tau, p = kendalltau(months, medians)
    passed = (abs(tau) >= g1["tau_min"]) and (p <= g1["p_max"])
    direction = "POSITIVE" if tau > 0 else ("NEGATIVE" if tau < 0 else "FLAT")
    return {
        "verdict": "PASS" if passed else "FAIL",
        "direction": direction,
        "tau": float(tau), "p": float(p),
        "tau_min": g1["tau_min"], "p_max": g1["p_max"],
        "medians_by_window": dict(zip([w for _, w in months_sorted], medians)),
        "months_axis": months,
    }


def evaluate_g2_stability(window_order: list[str],
                          per_window_test_pf_at_budget: dict[str, list[float]]
                          ) -> dict:
    """G2: CV(test PF) at A_long <= CV at A_short."""
    short = window_order[0]   # by gates ordering: A_short, A_base, A_medium, A_long
    long_ = window_order[-1]
    if short not in per_window_test_pf_at_budget or long_ not in per_window_test_pf_at_budget:
        return {"verdict": "AMBIGUOUS",
                "reason": f"need both {short} and {long_} present",
                "present": list(per_window_test_pf_at_budget)}
    cv_short = _cv(per_window_test_pf_at_budget[short])
    cv_long = _cv(per_window_test_pf_at_budget[long_])
    if _isnan(cv_short) or _isnan(cv_long):
        return {"verdict": "AMBIGUOUS",
                "reason": "CV undefined (need >=2 seeds + non-zero mean)",
                f"cv_{short}": cv_short, f"cv_{long_}": cv_long}
    return {
        "verdict": "PASS" if cv_long <= cv_short else "FAIL",
        f"cv_{short}": cv_short,
        f"cv_{long_}": cv_long,
        "delta_cv": cv_long - cv_short,
    }


def evaluate_g3_regime_decay(per_window_test_pf_at_budget: dict[str, list[float]],
                             gates: dict) -> dict:
    """G3: A_long must NOT drop more than 10% (relative) below A_base."""
    g3 = gates["hypotheses"]["g3_regime_decay_flag"]
    rule_str = str(g3["rule"])  # "(median(A_long) - median(A_base)) / median(A_base) >= -0.10"
    base_pfs = per_window_test_pf_at_budget.get("A_base", [])
    long_pfs = per_window_test_pf_at_budget.get("A_long", [])
    if not base_pfs or not long_pfs:
        return {"verdict": "AMBIGUOUS",
                "reason": "need both A_base and A_long present"}
    m_base = _median(base_pfs)
    m_long = _median(long_pfs)
    if _isnan(m_base) or _isnan(m_long) or m_base == 0:
        return {"verdict": "AMBIGUOUS",
                "reason": "median undefined or A_base median is zero"}
    rel_change = (m_long - m_base) / m_base
    threshold = -0.10  # mirrors the rule embedded in gates.yaml
    triggered = rel_change < threshold
    return {
        "verdict": "FLAG_TRIGGERED" if triggered else "OK",
        "median_A_base": m_base,
        "median_A_long": m_long,
        "relative_change": rel_change,
        "threshold_relative_drop": threshold,
        "rule": rule_str,
    }


def evaluate_g4_significance(per_window_test_pf_at_budget: dict[str, list[float]],
                             gates: dict) -> dict:
    """G4: Mann-Whitney U two-sided on A_base vs A_long PF distributions."""
    from scipy.stats import mannwhitneyu

    g4 = gates["hypotheses"]["g4_significance"]
    base_pfs = [x for x in per_window_test_pf_at_budget.get("A_base", []) if x is not None]
    long_pfs = [x for x in per_window_test_pf_at_budget.get("A_long", []) if x is not None]
    if len(base_pfs) < 3 or len(long_pfs) < 3:
        return {"verdict": "AMBIGUOUS",
                "reason": f"need >=3 seeds each; got base={len(base_pfs)}, long={len(long_pfs)}"}
    try:
        u, p = mannwhitneyu(base_pfs, long_pfs, alternative=g4["alternative"])
    except ValueError as e:
        return {"verdict": "AMBIGUOUS", "reason": f"mannwhitneyu raised: {e}"}
    return {
        "verdict": "PASS" if p <= g4["p_max"] else "FAIL",
        "u_statistic": float(u),
        "p_value": float(p),
        "p_max": g4["p_max"],
        "n_base": len(base_pfs),
        "n_long": len(long_pfs),
    }


def evaluate_g5_sim_to_live(per_window_test_pf_at_budget: dict[str, list[float]],
                            per_window_val_pf_at_budget: dict[str, list[float]]
                            ) -> dict:
    """G5: median(val_pf)/median(test_pf) at A_long <= ratio at A_base.

    Ratio > 1 means agent generalizes from val to test poorly — overfit
    flag. Smaller ratio = better OOS retention.
    """
    pairs = {}
    for w in ("A_base", "A_long"):
        v = per_window_val_pf_at_budget.get(w, [])
        t = per_window_test_pf_at_budget.get(w, [])
        if not v or not t:
            pairs[w] = None
            continue
        m_v = _median(v)
        m_t = _median(t)
        if _isnan(m_v) or _isnan(m_t) or m_t == 0:
            pairs[w] = None
        else:
            pairs[w] = m_v / m_t
    if pairs["A_base"] is None or pairs["A_long"] is None:
        return {"verdict": "AMBIGUOUS",
                "reason": "missing val or test PF for A_base or A_long",
                "ratios": pairs}
    return {
        "verdict": "PASS" if pairs["A_long"] <= pairs["A_base"] else "FAIL",
        "ratio_A_base": pairs["A_base"],
        "ratio_A_long": pairs["A_long"],
        "delta": pairs["A_long"] - pairs["A_base"],
    }


def evaluate_g6_overfitting_rescue(pf_surface: dict[str, dict[int, list[float]]],
                                   long_window: str = "A_long") -> dict:
    """G6: median(test PF at A_long, 2M) >= median(test PF at A_long, 500K).

    Decisive for STRONG_GO upgrade: vindicates buffer-multiplicity mechanism
    if longer history rescues the budget-axis overfitting that the Volume
    Study found at 10mo.
    """
    cells = pf_surface.get(long_window, {})
    if 500_000 not in cells or 2_000_000 not in cells:
        return {"verdict": "AMBIGUOUS",
                "reason": f"need both 500K and 2M cells at {long_window}",
                "present_budgets_at_long": list(cells)}
    m_500k = _median(cells[500_000])
    m_2m = _median(cells[2_000_000])
    if _isnan(m_500k) or _isnan(m_2m):
        return {"verdict": "AMBIGUOUS",
                "reason": "median undefined for one of the cells"}
    return {
        "verdict": "PASS" if m_2m >= m_500k else "FAIL",
        f"median_test_pf_{long_window}_500k": m_500k,
        f"median_test_pf_{long_window}_2m": m_2m,
        "delta": m_2m - m_500k,
    }


# ---------------------------------------------------------------------------
# Decision tree (per gates.yaml decision_tree:)
# ---------------------------------------------------------------------------

def map_to_decision(verdict_per_budget: dict[int, dict[str, dict]],
                    g6: dict, gates: dict) -> dict:
    """Map G1-G6 outcomes to STRONG_GO / GO / PARTIAL_GO / DEFER / NO_GO.

    Uses the decision_tree block from gates.yaml as the source of truth;
    this function encodes the logical mapping in code.
    """
    # Helpers — collapse per-budget verdicts into a logical bool per gate.
    def all_pass(gate: str) -> bool:
        return all(v[gate].get("verdict") == "PASS"
                   for v in verdict_per_budget.values())
    def any_pass(gate: str) -> bool:
        return any(v[gate].get("verdict") == "PASS"
                   for v in verdict_per_budget.values())
    def all_fail(gate: str) -> bool:
        return all(v[gate].get("verdict") == "FAIL"
                   for v in verdict_per_budget.values())
    def g3_triggered() -> bool:
        return any(v["g3_regime_decay_flag"].get("verdict") == "FLAG_TRIGGERED"
                   for v in verdict_per_budget.values())
    def g2_fail() -> bool:
        return any(v["g2_variance_stability"].get("verdict") == "FAIL"
                   for v in verdict_per_budget.values())

    # Order matters: STRONG_GO is most specific, NO_GO is the catch-all.
    if g6.get("verdict") == "PASS":
        return {
            "decision": "STRONG_GO",
            "rationale": "G6 PASS — longer history rescues 2M-step overfitting; "
                         "buffer-multiplicity mechanism is real and exploitable.",
            "action": gates["decision_tree"]["STRONG_GO"]["action"],
        }

    if all_pass("g1_monotonic_direction") and not g3_triggered() and not g2_fail():
        return {
            "decision": "GO",
            "rationale": "G1 PASS at both budgets, G2 not violated, G3 not triggered.",
            "action": gates["decision_tree"]["GO"]["action"],
        }

    if any_pass("g1_monotonic_direction") and not all_pass("g1_monotonic_direction"):
        return {
            "decision": "PARTIAL_GO",
            "rationale": "G1 PASS at one budget but FAIL at the other — "
                         "optimum-budget shifts with train-window size.",
            "action": gates["decision_tree"]["PARTIAL_GO"]["action"],
        }

    if (any_pass("g1_monotonic_direction") and g3_triggered()) or g2_fail():
        return {
            "decision": "DEFER",
            "rationale": "G3 regime-decay flag triggered (older data poisons buffer) "
                         "OR G2 variance grew with history — needs structural fix.",
            "action": gates["decision_tree"]["DEFER"]["action"],
        }

    if all_fail("g1_monotonic_direction"):
        return {
            "decision": "NO_GO",
            "rationale": "G1 FAIL at both budgets — no monotone trend; "
                         "data-axis is not a productive lever in this setting.",
            "action": gates["decision_tree"]["NO_GO"]["action"],
        }

    return {
        "decision": "AMBIGUOUS",
        "rationale": "verdict pattern does not match any decision-tree branch; "
                     "manual review needed.",
        "action": "Open issue and review per-budget verdicts before promoting.",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_pf_surface(window_order: list[str],
                      window_months: dict[str, int],
                      budgets: list[int],
                      pf_surface: dict[str, dict[int, list[float]]]) -> list[str]:
    """Render the 2D PF surface as a markdown table."""
    md = []
    md.append("| Window (months) \\ Budget | "
              + " | ".join(f"{b//1000}k" if b < 1_000_000 else f"{b//1_000_000}M"
                           for b in budgets) + " |")
    md.append("|" + "---|" * (1 + len(budgets)))
    for w in window_order:
        m_label = f"{w} ({window_months.get(w, '?')}mo)"
        cells = []
        for b in budgets:
            pfs = pf_surface.get(w, {}).get(b, [])
            if not pfs:
                cells.append("—")
            else:
                med = _median(pfs)
                cv = _cv(pfs)
                cells.append(f"{med:.3f} (CV={cv*100:.1f}%, n={len(pfs)})"
                             if not _isnan(cv) else f"{med:.3f} (n={len(pfs)})")
        md.append(f"| {m_label} | " + " | ".join(cells) + " |")
    return md


def write_report(out_dir: Path, manifest: dict, gates: dict,
                 per_seed: list[dict], verdict: dict,
                 pf_surface: dict[str, dict[int, list[float]]]) -> Path:
    """Markdown summary appendable to randd_log.md."""
    window_order = list(gates["windows"].keys())   # A_short, A_base, A_medium, A_long
    window_months = {w: gates["windows"][w]["train_window_months"]
                     for w in window_order}
    budgets = list(gates["step_budgets"])

    md = []
    md.append(f"# GMGP1 Data-Window Study (Axis A) — {manifest['timestamp']}\n")
    md.append(f"- Test bed: `{gates['test_bed']}` ({gates['asset']} on {gates['exchange']})")
    md.append(f"- Windows: {list(gates['windows'].keys())}")
    md.append(f"- Budgets: {gates['step_budgets']}")
    md.append(f"- Seeds: {gates['seeds']} (N={gates['n_seeds']})")
    md.append(f"- Fee model: `{gates['fee_model']}`")
    md.append(f"- Total cells: {len(window_order) * len(budgets)}, "
              f"total runs: {gates['n_runs']}")
    md.append("")

    md.append("## PF surface (test PF, by window x budget)\n")
    md.extend(render_pf_surface(window_order, window_months, budgets, pf_surface))
    md.append("")

    md.append("## Hypothesis verdicts (per budget)\n")
    for budget, gates_eval in verdict["per_budget"].items():
        md.append(f"### Budget = {budget}\n")
        for gid, vres in gates_eval.items():
            md.append(f"#### {gid}: **{vres.get('verdict', '?')}**")
            for k, v in vres.items():
                if k == "verdict":
                    continue
                md.append(f"- {k}: {v}")
            md.append("")
    md.append("")

    md.append("## G6: Overfitting rescue (cross-budget at A_long)\n")
    g6 = verdict["g6_overfitting_rescue"]
    md.append(f"- **Verdict:** {g6.get('verdict', '?')}")
    for k, v in g6.items():
        if k == "verdict":
            continue
        md.append(f"- {k}: {v}")
    md.append("")

    md.append("## Decision\n")
    d = verdict["decision"]
    md.append(f"- **{d['decision']}**: {d['rationale']}")
    md.append(f"- Action: {d['action']}")
    md.append("")

    if verdict.get("anomalies"):
        md.append("## Anomalies flagged\n")
        for a in verdict["anomalies"]:
            md.append(f"- `{a}`")
        md.append("")

    md.append("## Provenance\n")
    md.append(f"- Research artifact: `{gates['provenance']['research_artifact']}`")
    md.append(f"- Upstream HP run: `{gates['provenance']['upstream_hp_run']}`")
    md.append(f"- Twin study: `{gates['provenance']['twin_study']}`")
    md.append(f"- Manifest: `results/data_window_study_{manifest['timestamp']}/manifest.json`")

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

    import wandb
    api = wandb.Api(timeout=60)

    seeds = list(manifest["seeds"])
    window_order = list(gates["windows"].keys())
    window_months = {w: gates["windows"][w]["train_window_months"]
                     for w in window_order}
    budgets = list(gates["step_budgets"])

    # PF surface[window][budget] = list[seed_test_pf]
    pf_surface_test: dict[str, dict[int, list[float]]] = {}
    pf_surface_val: dict[str, dict[int, list[float]]] = {}
    per_seed_rows: list[dict] = []
    anomalies: list[str] = []

    for run_meta in manifest["runs"]:
        if run_meta["exit_code"] != 0:
            log.warning("cell window=%s budget=%s skipped (dispatcher exit_code=%d)",
                        run_meta["window"], run_meta["budget_label"],
                        run_meta["exit_code"])
            continue
        window = run_meta["window"]
        window_slug = run_meta["window_slug"]
        budget = run_meta["budget"]
        budget_label = run_meta["budget_label"]
        cell_study_id = run_meta.get("study_id", manifest["timestamp"])
        runs = fetch_runs_for_cell(api, args.project, args.entity,
                                   window_slug, budget_label,
                                   study_id=cell_study_id)
        if not runs:
            log.warning("cell window=%s budget=%s: no WandB runs found",
                        window, budget_label)
            continue
        log.info("cell window=%s budget=%s: %d run(s) found",
                 window, budget_label, len(runs))
        for r in runs:
            log.info("  - %s (%s)", r.name, r.id)

        seed_rows = extract_seed_summaries(runs, seeds)
        for s, row in seed_rows.items():
            row["window"] = window
            row["window_slug"] = window_slug
            row["train_window_months"] = run_meta.get("train_window_months")
            row["train_start_date"] = run_meta.get("train_start_date")
            row["budget"] = budget
            row["budget_label"] = budget_label
            per_seed_rows.append(row)
            if row.get("test_profit_factor") is not None:
                pf_surface_test.setdefault(window, {}).setdefault(budget, []).append(
                    row["test_profit_factor"])
            if row.get("val_profit_factor") is not None:
                pf_surface_val.setdefault(window, {}).setdefault(budget, []).append(
                    row["val_profit_factor"])

    # Per-budget G1-G5 evaluation
    verdict_per_budget: dict[int, dict[str, dict]] = {}
    for b in budgets:
        per_window_test_at_b = {
            w: pf_surface_test.get(w, {}).get(b, []) for w in window_order
        }
        per_window_val_at_b = {
            w: pf_surface_val.get(w, {}).get(b, []) for w in window_order
        }
        verdict_per_budget[b] = {
            "g1_monotonic_direction": evaluate_g1_direction(
                window_order, window_months, per_window_test_at_b, gates),
            "g2_variance_stability": evaluate_g2_stability(
                window_order, per_window_test_at_b),
            "g3_regime_decay_flag": evaluate_g3_regime_decay(
                per_window_test_at_b, gates),
            "g4_significance": evaluate_g4_significance(
                per_window_test_at_b, gates),
            "g5_sim_to_live_indirect": evaluate_g5_sim_to_live(
                per_window_test_at_b, per_window_val_at_b),
        }

    # Cross-budget G6 evaluation
    g6_verdict = evaluate_g6_overfitting_rescue(pf_surface_test)

    # Decision-tree mapping
    decision = map_to_decision(verdict_per_budget, g6_verdict, gates)

    verdict: dict[str, Any] = {
        "timestamp": manifest["timestamp"],
        "per_budget": verdict_per_budget,
        "g6_overfitting_rescue": g6_verdict,
        "decision": decision,
        "anomalies": anomalies,
        "n_seed_rows": len(per_seed_rows),
    }

    # Persist
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    (out_dir / "per_seed.json").write_text(
        json.dumps(per_seed_rows, indent=2, default=str), encoding="utf-8")
    (out_dir / "pf_surface.json").write_text(
        json.dumps({"test": pf_surface_test, "val": pf_surface_val},
                   indent=2, default=str), encoding="utf-8")
    report_path = write_report(out_dir, manifest, gates,
                               per_seed_rows, verdict, pf_surface_test)

    log.info("decision: %s", decision["decision"])
    log.info("per-budget summary:")
    for b, v in verdict_per_budget.items():
        log.info("  budget=%s  G1=%s  G2=%s  G3=%s  G4=%s  G5=%s",
                 b,
                 v["g1_monotonic_direction"].get("verdict"),
                 v["g2_variance_stability"].get("verdict"),
                 v["g3_regime_decay_flag"].get("verdict"),
                 v["g4_significance"].get("verdict"),
                 v["g5_sim_to_live_indirect"].get("verdict"))
    log.info("  G6 (cross-budget at A_long): %s", g6_verdict.get("verdict"))
    log.info("report:       %s", report_path.relative_to(PROJECT_ROOT))
    log.info("verdict json: %s", (out_dir / "verdict.json").relative_to(PROJECT_ROOT))
    log.info("pf_surface:   %s", (out_dir / "pf_surface.json").relative_to(PROJECT_ROOT))

    if args.write_randd:
        randd = PROJECT_ROOT / "randd_log.md"
        with randd.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(report_path.read_text(encoding="utf-8"))
        log.info("appended report to %s", randd.relative_to(PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
