"""Read the substrate-power stamp off an MDE surface through the GUARD'S OWN accessor.

WHY THIS EXISTS (S553-cont-166, POWER-LORD-01). The `us_equity` power ceiling
(`configs/us_equity_power.gates.yaml`, `plausible_delta_sr_max: 1.457`) justifies itself with two
readings, and both are SHIPPED-GRID readings:

    19.61y panel (holdout 6.85y): MDE 1.312 < 1.457  -> PASS, ~10% margin
    12.07y panel (holdout 4.22y): MDE 1.782 > 1.457  -> REFUSE

That second line is the file's own proof that the gate bites rather than rubber-stamps. Refining the
beta grid lowers EVERY measured MDE (the crossing is found earlier), so both numbers move DOWN — and
if the 12.07y reading drops under the ceiling, the ceiling loses its demonstration AND a panel that
was refused becomes admissible. That is a verdict change, and it has to be looked at deliberately
rather than discovered later.

This tool reads any sweep artifact at any set of panel depths through `stamp_substrate_power` — the
same call the orchestrator makes — so the comparison is the gate's own arithmetic, not a re-derivation
that merely looks like it.

Usage:
    python scripts/research/crucible_power_stamp_readout.py \
        --sweep results/crucible_calibration_union/calibration_xsec_mde_sweep_corrected.json \
        --compare results/_calib_backup_pre_refined_grid_20260825/crucible_calibration_union/calibration_xsec_mde_sweep_corrected.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible.orchestrator.substrate import (  # noqa: E402
    _power_holdout_bars,
    stamp_substrate_power,
)

# The two panels the us_equity ceiling cites, plus the depth the live tick actually stamps.
# (label, panel_T bars, holdout_frac) — 19.61y is the mined panel; 12.07y is the shorter cache the
# ceiling points at as the one it REFUSES.
PANELS = [
    ("us_equity 19.61y (mined)", 4930, 0.35),
    ("us_equity 12.07y (shorter cache)", 3042, 0.35),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def read_stamps(path: Path, ceiling: float) -> dict:
    sweep = json.loads(path.read_text(encoding="utf-8"))
    ct = sweep["mde_sweep"].get("candidate_type", "cross_sectional")
    betas = sweep["mde_sweep"].get("beta_grid", [])
    out = {"path": str(path), "sha": _sha(path), "n_betas": len(betas), "candidate_type": ct,
           "panels": {}}
    for label, t, hf in PANELS:
        sp = stamp_substrate_power(t, hf, {ct: sweep}, out["sha"], candidate_types=(ct,))
        out["panels"][label] = {
            "panel_T": t, "holdout_bars": _power_holdout_bars(t, hf),
            "implied_mde_delta_sr": sp.implied_mde_delta_sr,
            "interp_mode": sp.interp_mode,
            "verdict": ("PASS" if sp.implied_mde_delta_sr <= ceiling else "REFUSE"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--compare", default="", help="prior surface to diff against (optional)")
    ap.add_argument("--ceiling", type=float, default=1.457,
                    help="plausible_delta_sr_max from the power gates file")
    args = ap.parse_args()

    new = read_stamps(ROOT / args.sweep, args.ceiling)
    old = read_stamps(ROOT / args.compare, args.ceiling) if args.compare else None

    print(f"ceiling plausible_delta_sr_max = {args.ceiling}")
    print(f"NEW  {new['sha']}  {new['n_betas']} betas  ({new['candidate_type']})")
    if old:
        print(f"PRIOR {old['sha']}  {old['n_betas']} betas")
    print()
    hdr = f"{'panel':<34} {'holdout':>8} {'MDE':>9} {'verdict':>8}"
    if old:
        hdr += f" {'prior MDE':>10} {'prior':>7} {'change':>18}"
    print(hdr)
    print("-" * len(hdr))
    flipped = []
    for label in new["panels"]:
        n = new["panels"][label]
        row = f"{label:<34} {n['holdout_bars']:>8} {n['implied_mde_delta_sr']:>9.4f} {n['verdict']:>8}"
        if old:
            o = old["panels"][label]
            delta = n["implied_mde_delta_sr"] - o["implied_mde_delta_sr"]
            note = "same" if n["verdict"] == o["verdict"] else f"** {o['verdict']} -> {n['verdict']} **"
            if n["verdict"] != o["verdict"]:
                flipped.append((label, o["verdict"], n["verdict"]))
            row += f" {o['implied_mde_delta_sr']:>10.4f} {o['verdict']:>7} {delta:>+9.4f} {note:>8}"
        print(row)
    print()
    if flipped:
        print("⚠ VERDICT CHANGE — the refinement moved a gate decision, not just a number:")
        for label, a, b in flipped:
            print(f"    {label}: {a} -> {b}")
        print("  This is an operator-visible change; do not ship it silently.")
    else:
        print("No verdict changed: the refinement moved the numbers, not the decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
