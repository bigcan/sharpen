"""Q2 train_parity gate computation — pull WandB metrics + apply ADR-6 thresholds.

Pairs with `scripts/prop_firm_ab_compare.py` Q2_THRESHOLDS. The harness's
``run_training_parity`` stub leaves gate evaluation to a follow-up step;
this script is that step.

Usage:
    python scripts/q2_compute_gates.py \\
        --v7-run-id vw1kbeoj \\
        --rs-run-id j3kr9jb1 \\
        --out-dir results/ab_prop_firm_decoupling/<ts>/train_parity

Reads:
    - WandB scalar history for both runs (train/critic_loss only — terminal_Q,
      actor_loss, return_KL dropped per S498 protocol revision because S496
      disambiguation showed them noise-bound at 100K SAC steps)

Writes:
    - decision.json  — critic_loss ratio + verdict
    - report.md      — human-readable summary

Verdict rule (ADR-6 Q2, revised S498):
    - ratio in [0.95, 1.05]                → AMBIGUOUS (escalate seeds)
    - ratio in [0.80, 0.95) ∪ (1.05, 1.25] → PASS (above-noise parity)
    - ratio outside [0.80, 1.25]           → FAIL (clear divergence)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb

# Q2 thresholds (mirror scripts/prop_firm_ab_compare.Q2_THRESHOLDS).
# See header docstring for verdict semantics (S498 protocol revision).
CRITIC_LOSS_BAND = (0.80, 1.25)
CRITIC_LOSS_AMBIGUOUS = (0.95, 1.05)

# Tail window for loss medians (last N log points)
TAIL_WINDOW = 20


def _fetch_history(run_id: str, entity: str, project: str) -> dict[str, list[float]]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = ["train/critic_loss", "_step"]
    hist = run.scan_history(keys=keys)
    out: dict[str, list[float]] = {k: [] for k in keys}
    for row in hist:
        for k in keys:
            v = row.get(k)
            if v is not None:
                out[k].append(float(v))
    return out


def _tail_median(xs: list[float], n: int = TAIL_WINDOW) -> float:
    if not xs:
        return float("nan")
    return float(np.median(xs[-n:]))


def _classify_critic_loss_ratio(
    cl_v7: float, cl_rs: float,
    band: tuple[float, float] = CRITIC_LOSS_BAND,
    ambiguous: tuple[float, float] = CRITIC_LOSS_AMBIGUOUS,
) -> dict[str, Any]:
    """Apply S498 3-tier verdict to a critic_loss ratio (B/A).

    S496 disambiguation showed V7-vs-V7 ratios in [1.02, 1.17] are noise;
    a ratio inside the inner ``ambiguous`` band cannot be distinguished
    from same-wrapper noise so we ask for more seeds rather than declare
    parity.
    """
    if abs(cl_v7) < 1e-12:
        # Pathological: V7 critic_loss collapsed. RS must also collapse to
        # match. Tag as FAIL otherwise so it's surfaced loudly.
        passed = abs(cl_rs) < 1e-12
        return {
            "verdict": "PASS" if passed else "FAIL",
            "A": cl_v7, "B": cl_rs, "ratio": None,
            "band": band, "ambiguous": ambiguous,
            "note": "A near zero — comparing magnitudes",
        }
    ratio = cl_rs / cl_v7
    lo, hi = band
    amb_lo, amb_hi = ambiguous
    if not (lo <= ratio <= hi):
        verdict = "FAIL"
    elif amb_lo <= ratio <= amb_hi:
        verdict = "AMBIGUOUS"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "A": cl_v7, "B": cl_rs, "ratio": ratio,
        "band": band, "ambiguous": ambiguous,
    }


def compute_q2_decision(v7: dict[str, list[float]], rs: dict[str, list[float]]) -> dict[str, Any]:
    cl_v7 = _tail_median(v7["train/critic_loss"])
    cl_rs = _tail_median(rs["train/critic_loss"])
    critic = _classify_critic_loss_ratio(cl_v7, cl_rs)

    return {
        "verdict": critic["verdict"],
        "critic_loss_ratio": critic,
        "tail_window_log_points": TAIL_WINDOW,
        "samples": {
            "v7_critic_loss_n": len(v7["train/critic_loss"]),
            "rs_critic_loss_n": len(rs["train/critic_loss"]),
        },
    }


def write_report(out_dir: Path, decision: dict[str, Any], v7_id: str, rs_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str))

    cl = decision["critic_loss_ratio"]
    ratio_str = f"{cl['ratio']:.4f}" if cl.get("ratio") is not None else "n/a"
    lines = [
        "# Q2 train_parity decision",
        "",
        f"**Verdict:** {decision['verdict']}",
        f"**V7 run:** `{v7_id}`",
        f"**RS run:** `{rs_id}`",
        f"**Tail window:** last {decision['tail_window_log_points']} log points",
        "",
        "## Critic-loss parity (S498 protocol)",
        f"- A (V7) = {cl['A']:.4f}",
        f"- B (RS) = {cl['B']:.4f}",
        f"- B/A = {ratio_str}",
        f"- Acceptance band = {cl['band']}",
        f"- Ambiguous band = {cl['ambiguous']} (inside → AMBIGUOUS, escalate seeds)",
    ]
    if cl.get("note"):
        lines.append(f"- Note: {cl['note']}")
    lines.extend([
        "",
        "## Provenance",
        f"- V7 run: https://wandb.ai/bigcan-chiwin-technology/FinRL-Pro-DS/runs/{v7_id}",
        f"- RS run: https://wandb.ai/bigcan-chiwin-technology/FinRL-Pro-DS/runs/{rs_id}",
        f"- Sample counts: {decision['samples']}",
        "",
        "---",
        "Generated by `scripts/q2_compute_gates.py` (Q2 of ADR-6 prop-firm decoupling, S498 protocol revision)",
    ])
    p = out_dir / "report.md"
    p.write_text("\n".join(lines))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v7-run-id", required=True, help="WandB run id for control arm (PropFirmWrapperV7)")
    ap.add_argument("--rs-run-id", required=True, help="WandB run id for treatment arm (RiskShapingWrapper)")
    ap.add_argument("--entity", default="bigcan-chiwin-technology")
    ap.add_argument("--project", default="FinRL-Pro-DS")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    print(f"Fetching V7 history: {args.v7_run_id}")
    v7 = _fetch_history(args.v7_run_id, args.entity, args.project)
    print(f"  critic_loss N={len(v7['train/critic_loss'])}")

    print(f"Fetching RS history: {args.rs_run_id}")
    rs = _fetch_history(args.rs_run_id, args.entity, args.project)
    print(f"  critic_loss N={len(rs['train/critic_loss'])}")

    decision = compute_q2_decision(v7, rs)
    decision["v7_run_id"] = args.v7_run_id
    decision["rs_run_id"] = args.rs_run_id

    report_path = write_report(args.out_dir, decision, args.v7_run_id, args.rs_run_id)
    print(f"\nVerdict: {decision['verdict']}")
    cl = decision["critic_loss_ratio"]
    ratio_str = f"{cl['ratio']:.4f}" if cl.get("ratio") is not None else "n/a"
    print(f"  critic_loss B/A = {ratio_str}  band={cl['band']}  ambiguous={cl['ambiguous']}")
    print(f"Report: {report_path}")
    # Exit codes: PASS=0, AMBIGUOUS=2 (escalate), FAIL=1.
    return {"PASS": 0, "AMBIGUOUS": 2, "FAIL": 1}.get(decision["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
