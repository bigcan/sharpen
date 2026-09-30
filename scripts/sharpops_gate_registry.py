"""Check or (re-)register the hashed gate files (configs/gates_registry.json).

    python scripts/sharpops_gate_registry.py --check
    python scripts/sharpops_gate_registry.py --register configs/x.gates.yaml --reason "..."

``--check`` exits 1 if any gates file changed, is unregistered, or disappeared. A registration
requires ``--reason``: the decision memo or waiver it rests on (SharpOps promotion standard §5).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharpen.sharpops import gate_registry as gr  # noqa: E402

log = logging.getLogger("sharpops_gate_registry")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--register", nargs="+", metavar="PATH")
    ap.add_argument("--reason", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.check:
        res = gr.check()
        log.info(json.dumps(res, indent=1))
        return 1 if any(res.values()) else 0
    try:
        gr.register([p.replace("\\", "/") for p in args.register], args.reason or "")
    except ValueError as e:
        log.error("refused: %s", e)
        return 1
    log.info("registered %d file(s)", len(args.register))
    return 0


if __name__ == "__main__":
    sys.exit(main())
