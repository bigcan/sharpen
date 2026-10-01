"""SharpOps promotion check (Tier-2 N3): can this workstream be promoted on today's tree?

    python scripts/sharpops_promotion_check.py --workstream tailwind-v1

Loads the workstream's newest enforcing artifact (``configs/sharpops_ladder.gates.yaml``
``ladder.workstreams``) and exits

  0  PASS   — the artifact's verdict is PASS and nothing is stale;
  3  REVIEW — the artifact says REVIEW;
  1  BLOCK / FAIL / STALE — the verdict blocks, the artifact is missing, a pinned data file or
     a config it hashed has changed since, or the gate registry is not clean.

A PASS here authorises nothing on its own: a Tier-2 audit on the final tree and the operator's
gate-record decision still come first (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sharpen.sharpops import gate_registry as gr  # noqa: E402

LADDER = ROOT / "configs" / "sharpops_ladder.gates.yaml"
EXIT = {"PASS": 0, "REVIEW": 3, "BLOCK": 1, "FAIL": 1, "STALE": 1}
log = logging.getLogger("sharpops_promotion_check")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(workstream: str, *, root: Path = ROOT, ladder: Path = LADDER) -> dict:
    spec = yaml.safe_load(ladder.read_text(encoding="utf-8"))["ladder"]["workstreams"].get(workstream)
    if spec is None:
        return {"decision": "FAIL", "reasons": [f"unknown workstream {workstream!r}"]}
    reasons: list[str] = []
    reg = gr.check(root, root / "configs" / "gates_registry.json")
    if any(reg.values()):
        reasons.append(f"gate registry not clean: {reg}")
    arts = sorted(root.glob(spec["enforcing_artifact_glob"]))
    if not arts:
        return {"decision": "FAIL", "reasons": reasons + ["no enforcing artifact"], "artifact": None}
    art_path = arts[-1]
    art = json.loads(art_path.read_text(encoding="utf-8"))
    for rel, h in ((art.get("pins") or {}).get("files_sha256") or {}).items():
        p = root / rel
        if not p.exists() or _sha(p) != h:
            reasons.append(f"STALE: pinned data {rel} changed or missing since the artifact")
    for rel, h in ((art.get("lineage") or {}).get("config_sha256") or {}).items():
        p = root / rel
        if not p.exists() or _sha(p) != h:
            reasons.append(f"STALE: config {rel} changed since the artifact")
    verdict = (art.get("verdict") or {}).get("decision", "FAIL")
    if any(r.startswith("STALE") for r in reasons):
        decision = "STALE"
    elif reasons:
        decision = "FAIL"
    else:
        decision = verdict if verdict in EXIT else "FAIL"
    return {"workstream": workstream, "target_rung": spec.get("target_rung"),
            "artifact": str(art_path.relative_to(root)).replace("\\", "/"),
            "artifact_verdict": verdict, "decision": decision, "reasons": reasons}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workstream", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = evaluate(args.workstream)
    log.info(json.dumps(res, indent=1))
    return EXIT[res["decision"]]


if __name__ == "__main__":
    sys.exit(main())
