"""Merge MDE sweep JSONs into one union curve for the substrate-power guard (S553-cont-151).

WHY. The deep grid (2026-08-01/02, ``results/crucible_calibration_deep/``) was written to a SEPARATE
directory from the original sweep, so the guard — which reads exactly one file per (contract,
candidate_type) — can see one or the other, never both. Neither alone is right:

  * the ORIGINAL curve tops out at holdout 8,064, so any deeper substrate reads ``unmeasured_high``
    and is refused out of IGNORANCE rather than measurement;
  * the DEEP curve starts at holdout 2,016, so every daily substrate (holdout ~1,011) drops off the
    BOTTOM into ``extrapolated_low``, replacing four measured anchors with an extrapolation and
    changing recorded provenance numbers for substrates whose verdicts are settled.

The union has both. It is safe to take because the two runs AGREE where they overlap: same harness,
same contract, same thresholds, same seeds and target power, so the shared depths (holdout 2,016 /
4,032 / 8,064) come back bit-identical. That agreement is not assumed — it is CHECKED here, and a
disagreement beyond ``--tol`` aborts rather than silently picking a winner.

The merge is over ROWS only. Every field the guard's accessors read (``mde_sweep.rows[*]`` ->
``holdout_bars`` / ``mde_realized_delta_sr``, and ``mde_sweep.candidate_type``) is preserved, and the
inputs' identifying metadata is required to match, so the union is the same KIND of object as its
parts rather than a new one.

Usage:
    python scripts/research/crucible_merge_mde_sweeps.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Fields that must be identical across inputs — they identify what the curve is a curve FOR.
#: ``t_grid``/``n_seeds``/``beta_grid`` are deliberately NOT here: extending the grid is the whole
#: point, and a deeper run may need a different beta ladder to bracket a smaller effect.
_MUST_MATCH = ("harness_version", "contract")
_SWEEP_MUST_MATCH = ("contract", "power_target", "holdout_frac")

MERGES = [
    ("calibration_mde_sweep_corrected.json", "overlay"),
    ("calibration_xsec_mde_sweep_corrected.json", "cross_sectional"),
]


def _row_key(r: dict) -> tuple:
    """Identity of a measured point: depth, plus the breadth axis when the sweep has one."""
    return (int(r["holdout_bars"]), r.get("n"))


def merge(base_path: Path, deep_path: Path, out_path: Path, *, tol: float) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    deep = json.loads(deep_path.read_text(encoding="utf-8"))

    for f in _MUST_MATCH:
        if base.get(f) != deep.get(f):
            raise SystemExit(f"REFUSING to merge: {f!r} differs ({base.get(f)!r} vs {deep.get(f)!r})")
    bs, ds = base["mde_sweep"], deep["mde_sweep"]
    for f in _SWEEP_MUST_MATCH:
        if bs.get(f) != ds.get(f):
            raise SystemExit(
                f"REFUSING to merge: mde_sweep.{f} differs ({bs.get(f)!r} vs {ds.get(f)!r})")
    if bs.get("candidate_type") != ds.get("candidate_type"):
        raise SystemExit("REFUSING to merge: candidate_type differs — these curves characterize "
                         "different scoring paths and pooling them would hide the worse one")

    rows: dict[tuple, dict] = {_row_key(r): r for r in bs["rows"]}
    n_new = n_shared = 0
    for r in ds["rows"]:
        k = _row_key(r)
        if k in rows:
            a = rows[k].get("mde_realized_delta_sr")
            b = r.get("mde_realized_delta_sr")
            if a is not None and b is not None and abs(float(a) - float(b)) > tol:
                raise SystemExit(
                    f"REFUSING to merge: overlapping point {k} disagrees ({a} vs {b} > tol {tol}). "
                    "The runs are not the same measurement; re-measure rather than pick one.")
            n_shared += 1
        else:
            rows[k] = r
            n_new += 1

    merged = dict(base)
    merged["mde_sweep"] = {**bs, "rows": [rows[k] for k in sorted(rows, key=lambda z: (z[1] is not None, z[1], z[0]) if z[1] is not None else (False, 0, z[0]))],
                           "t_grid": sorted(set(map(int, bs.get("t_grid", []))) | set(map(int, ds.get("t_grid", [])))),
                           "merged_from": [str(base_path.relative_to(ROOT)), str(deep_path.relative_to(ROOT))],
                           "merged_note": ("union of two runs of the SAME harness/contract; overlapping "
                                           f"points verified equal within {tol}")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    depths = sorted({int(r["holdout_bars"]) for r in merged["mde_sweep"]["rows"]})
    print(f"  {out_path.name}: {len(rows)} rows ({n_shared} shared+verified, {n_new} new)")
    print(f"    holdout depths: {depths}")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="results/crucible_calibration")
    ap.add_argument("--deep-dir", default="results/crucible_calibration_deep")
    ap.add_argument("--out-dir", default="results/crucible_calibration_union")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    for fname, ctype in MERGES:
        b, d = ROOT / args.base_dir / fname, ROOT / args.deep_dir / fname
        if not b.exists() or not d.exists():
            print(f"  SKIP {fname} ({'base' if not b.exists() else 'deep'} missing)")
            continue
        print(f"[{ctype}]")
        merge(b, d, ROOT / args.out_dir / fname, tol=args.tol)
    print("\nPoint configs/crucible_power.gates.yaml at the union files to use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
