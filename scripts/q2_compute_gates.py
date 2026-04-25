"""Q2 train_parity gate computation — pull WandB metrics + apply ADR-6 thresholds.

Pairs with `scripts/prop_firm_ab_compare.py` Q2_THRESHOLDS (terminal Q ratio
band, critic/actor loss ratio bands, return-KL). The harness's
``run_training_parity`` stub leaves gate evaluation to a follow-up step;
this script is that step.

Usage:
    python scripts/q2_compute_gates.py \\
        --v7-run-id vw1kbeoj \\
        --rs-run-id j3kr9jb1 \\
        --out-dir results/ab_prop_firm_decoupling/<ts>/train_parity

Reads:
    - WandB scalar history for both runs (train/critic_loss, train/actor_loss,
      train/q1_mean, train/q2_mean, train/episode_reward_mean)

Writes:
    - decision.json  — full metric table + per-gate PASS/FAIL + verdict
    - report.md      — human-readable summary

Verdict rule (ADR-6 Q2):
    - PASS = all 4 gates pass
    - FAIL = any gate fails
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb

# Q2 thresholds (mirror scripts/prop_firm_ab_compare.Q2_THRESHOLDS)
TERMINAL_Q_BAND = (0.90, 1.10)
CRITIC_LOSS_BAND = (0.80, 1.25)
ACTOR_LOSS_BAND = (0.80, 1.25)
RETURN_KL_MAX = 0.05

# Tail window for terminal Q / loss medians (last N log points)
TAIL_WINDOW = 20


def _fetch_history(run_id: str, entity: str, project: str) -> dict[str, list[float]]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = [
        "train/critic_loss",
        "train/actor_loss",
        "train/q1_mean",
        "train/q2_mean",
        "train/episode_reward_mean",
        "_step",
    ]
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


def _gaussian_kl(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float) -> float:
    """KL(N(mu_a,sig_a) || N(mu_b,sig_b)) — assume Gaussian return distributions.

    Per ADR-6 Q2: return-KL is the divergence between V7 and RS episodic-return
    distributions. We approximate them as Gaussians from the training-time
    episode_reward_mean stream (last TAIL_WINDOW points).
    """
    if sigma_a < 1e-9 or sigma_b < 1e-9:
        return float("inf")
    return float(np.log(sigma_b / sigma_a) + (sigma_a**2 + (mu_a - mu_b) ** 2) / (2 * sigma_b**2) - 0.5)


def _gate_ratio(label: str, a: float, b: float, band: tuple[float, float]) -> dict[str, Any]:
    if abs(a) < 1e-12:
        passed = abs(b) < 1e-12
        return {"pass": bool(passed), "A": a, "B": b, "ratio": None,
                "band": band, "note": "A near zero — comparing magnitudes"}
    ratio = b / a
    lo, hi = band
    return {"pass": lo <= ratio <= hi, "A": a, "B": b, "ratio": ratio, "band": band}


def compute_q2_decision(v7: dict[str, list[float]], rs: dict[str, list[float]]) -> dict[str, Any]:
    # Terminal Q = mean(q1, q2) at last TAIL_WINDOW
    q_v7 = _tail_median([(a + b) / 2 for a, b in zip(v7["train/q1_mean"], v7["train/q2_mean"])])
    q_rs = _tail_median([(a + b) / 2 for a, b in zip(rs["train/q1_mean"], rs["train/q2_mean"])])

    cl_v7 = _tail_median(v7["train/critic_loss"])
    cl_rs = _tail_median(rs["train/critic_loss"])

    al_v7 = _tail_median(v7["train/actor_loss"])
    al_rs = _tail_median(rs["train/actor_loss"])

    # Return distributions (last TAIL_WINDOW episode_reward_mean log points)
    er_v7 = v7["train/episode_reward_mean"][-TAIL_WINDOW:] if v7["train/episode_reward_mean"] else []
    er_rs = rs["train/episode_reward_mean"][-TAIL_WINDOW:] if rs["train/episode_reward_mean"] else []
    if er_v7 and er_rs:
        mu_v7, sig_v7 = float(np.mean(er_v7)), float(np.std(er_v7) + 1e-8)
        mu_rs, sig_rs = float(np.mean(er_rs)), float(np.std(er_rs) + 1e-8)
        kl = _gaussian_kl(mu_rs, sig_rs, mu_v7, sig_v7)  # KL(B||A)
    else:
        mu_v7 = mu_rs = sig_v7 = sig_rs = float("nan")
        kl = float("nan")

    gates = {
        "terminal_q_ratio": _gate_ratio("terminal_Q", q_v7, q_rs, TERMINAL_Q_BAND),
        "critic_loss_ratio": _gate_ratio("critic_loss", cl_v7, cl_rs, CRITIC_LOSS_BAND),
        "actor_loss_ratio": _gate_ratio("actor_loss", al_v7, al_rs, ACTOR_LOSS_BAND),
        "return_kl": {
            "pass": kl <= RETURN_KL_MAX if not np.isnan(kl) else False,
            "kl": kl,
            "max": RETURN_KL_MAX,
            "v7": {"mu": mu_v7, "sigma": sig_v7, "n": len(er_v7)},
            "rs": {"mu": mu_rs, "sigma": sig_rs, "n": len(er_rs)},
        },
    }

    fails = [k for k, g in gates.items() if not g["pass"]]
    verdict = "PASS" if not fails else "FAIL"

    return {
        "verdict": verdict,
        "failed_gates": fails,
        "gates": gates,
        "tail_window_log_points": TAIL_WINDOW,
        "samples": {
            "v7_q_mean_n": len(v7["train/q1_mean"]),
            "rs_q_mean_n": len(rs["train/q1_mean"]),
            "v7_critic_loss_n": len(v7["train/critic_loss"]),
            "rs_critic_loss_n": len(rs["train/critic_loss"]),
        },
    }


def write_report(out_dir: Path, decision: dict[str, Any], v7_id: str, rs_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str))

    lines = [
        "# Q2 train_parity decision",
        "",
        f"**Verdict:** {decision['verdict']}",
        f"**V7 run:** `{v7_id}`",
        f"**RS run:** `{rs_id}`",
        f"**Tail window:** last {decision['tail_window_log_points']} log points",
        "",
    ]
    if decision["failed_gates"]:
        lines.append(f"**Failed gates:** {', '.join(decision['failed_gates'])}")
        lines.append("")
    lines.append("## Gate results")
    for name, g in decision["gates"].items():
        status = "PASS" if g["pass"] else "FAIL"
        if "ratio" in g:
            band = g.get("band", "?")
            ratio_str = f"{g['ratio']:.4f}" if g.get('ratio') is not None else "n/a"
            lines.append(f"- **{name}**: {status} — A={g['A']:.4f}, B={g['B']:.4f}, B/A={ratio_str}, band={band}")
        else:
            kl_val = g['kl']
            kl_str = f"{kl_val:.4f}" if not np.isnan(kl_val) else "n/a"
            lines.append(f"- **{name}**: {status} — KL={kl_str} (max={g['max']}), V7 N(mu={g['v7']['mu']:.4f}, sig={g['v7']['sigma']:.4f}), RS N(mu={g['rs']['mu']:.4f}, sig={g['rs']['sigma']:.4f})")
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"- V7 run: https://wandb.ai/bigcan-chiwin-technology/FinRL-Pro-DS/runs/{v7_id}")
    lines.append(f"- RS run: https://wandb.ai/bigcan-chiwin-technology/FinRL-Pro-DS/runs/{rs_id}")
    lines.append(f"- Sample counts: {decision['samples']}")
    lines.append("")
    lines.append("---")
    lines.append("Generated by `scripts/q2_compute_gates.py` (Q2 of ADR-6 prop-firm decoupling)")
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
    print(f"  q1_mean N={len(v7['train/q1_mean'])}, critic_loss N={len(v7['train/critic_loss'])}")

    print(f"Fetching RS history: {args.rs_run_id}")
    rs = _fetch_history(args.rs_run_id, args.entity, args.project)
    print(f"  q1_mean N={len(rs['train/q1_mean'])}, critic_loss N={len(rs['train/critic_loss'])}")

    decision = compute_q2_decision(v7, rs)
    decision["v7_run_id"] = args.v7_run_id
    decision["rs_run_id"] = args.rs_run_id

    report_path = write_report(args.out_dir, decision, args.v7_run_id, args.rs_run_id)
    print(f"\nVerdict: {decision['verdict']}")
    if decision["failed_gates"]:
        print(f"Failed: {decision['failed_gates']}")
    print(f"Report: {report_path}")
    return 0 if decision["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
