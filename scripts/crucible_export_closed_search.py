"""Share a Crucible search that found nothing: export a trial ledger to the community ledger.

    python scripts/crucible_export_closed_search.py \
        --ledger results/crucible_orchestrator/real/trial_ledger.db --out community/ledger/

Writes one JSON file per substrate to ``<out>/<substrate>/<date>-<id>.json``: how many hypotheses
were registered, scored and tested out of sample, how strong that test was next to the smallest edge
the contract accepts, the dedup hashes of what was scored, and a verdict —

    CLOSED_DECISIVE     the test could have seen such an edge and did not
    OPEN_UNDERPOWERED   the test could not tell either way (NOT "nothing there")

No formula, score or per-hypothesis verdict is written, and the ledger is opened read-only.

A substrate that holds a PROMISING result is left out unless you pass ``--include-promising``, so
sharing a dead end never reveals a hit. Use ``--list`` to see what a ledger contains, and
``--substrate`` / ``--run-id`` to choose what to share.

Describe the data so the next person can tell whether theirs is the same search:

    --universe "US common stocks, top 500 by market cap" \
    --data-source "Yahoo Finance daily bars" --data-start 2008-01-01 --data-end 2026-08-08
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible import community  # noqa: E402

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _date(value: str | None) -> str | None:
    if value is not None and not _DATE.match(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date")
    return value


def _description(args, d: dict | None = None) -> community.SubstrateDescription:
    """One substrate's description: its ``--describe`` entry first, the flags fill what it left out."""
    d = d or {}
    return community.SubstrateDescription(
        universe=d.get("universe") or args.universe,
        data_sources=tuple(d.get("data_sources") or args.data_source or ()),
        data_start=_date(d.get("data_start") or args.data_start),
        data_end=_date(d.get("data_end") or args.data_end),
        search_space=d.get("search_space") or args.search_space)


def _descriptions(args) -> dict[str, community.SubstrateDescription]:
    if not args.describe:
        return {}
    raw = json.loads(Path(args.describe).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--describe must be a JSON object keyed by substrate id")
    return {str(sub): _description(args, d) for sub, d in raw.items()}


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _summary(payload: dict) -> str:
    t = payload["trials"]
    tested = ("count not recorded" if t["holdout_tested"] is None else
              f"{'' if t['holdout_count_complete'] else 'at least '}{t['holdout_tested']}")
    return (f"{payload['verdict']:<18} {payload['hypotheses']['count']} hypotheses scored; "
            f"tested out of sample: {tested}; {community.power_phrase(payload)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True, help="path to a trial_ledger.db")
    ap.add_argument("--out", default="community/ledger", help="community ledger directory")
    ap.add_argument("--orchestrator", default=None,
                    help="orchestrator.db (default: beside the ledger)")
    ap.add_argument("--manifests", default=None,
                    help="directory holding run_manifest.json files (default: the ledger's directory)")
    ap.add_argument("--substrate", action="append", default=None,
                    help="export only this substrate (repeatable)")
    ap.add_argument("--run-id", action="append", default=None,
                    help="export only these runs (repeatable; see --list)")
    ap.add_argument("--include-promising", action="store_true",
                    help="also export substrates that hold a PROMISING result (off by default)")
    ap.add_argument("--describe", default=None,
                    help="JSON file: {substrate: {universe, data_sources, data_start, data_end, "
                         "search_space}}")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--data-source", action="append", default=None, help="repeatable")
    ap.add_argument("--data-start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--data-end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--search-space", default=None,
                    help='what was searched, e.g. "WorldQuant-101 seeds plus genetic offspring"')
    ap.add_argument("--config", default=str(community.DEFAULT_FUNNEL_GATES),
                    help="funnel gates file (the economic floor of the shipped contract)")
    ap.add_argument("--corrected-config", default=str(community.DEFAULT_CORRECTED_GATES),
                    help="corrected-contract gates file (guards.uplift_min, the economic floor)")
    ap.add_argument("--search-memory-config", default=str(community.DEFAULT_SEARCH_MEMORY_GATES))
    ap.add_argument("--list", action="store_true",
                    help="show what the ledger contains and what would be exported; write nothing")
    ap.add_argument("--dry-run", action="store_true", help="build the files but do not write them")
    args = ap.parse_args(argv)

    ledger = Path(args.ledger)
    if not ledger.is_file():
        ap.error(f"no such ledger: {args.ledger}")

    try:
        result = community.export_closed_searches(
            ledger, orchestrator_db=args.orchestrator, manifests_dir=args.manifests,
            substrates=args.substrate, run_ids=args.run_id,
            include_promising=args.include_promising, descriptions=_descriptions(args),
            default_description=_description(args), corrected_gates=args.corrected_config,
            funnel_gates=args.config, search_memory_gates=args.search_memory_config)
    except (ValueError, FileNotFoundError, argparse.ArgumentTypeError) as e:
        ap.error(str(e))

    out = Path(args.out)
    if args.list:
        for sub in community.ledger_overview(ledger, orchestrator_db=args.orchestrator):
            note = "  [holds a PROMISING result - not exported by default]" if sub["holds_promising"] else ""
            print(f"{sub['substrate']}{note}")
            for run in sub["runs"]:
                print(f"    {run['run_id']}  ({run['trials']} trials, {run['scored']} scored)")
        print()
    for payload in result.payloads:
        path = out / community.relative_path(payload)
        print(f"{'would write' if args.list or args.dry_run else 'wrote':<11} {_shown(path)}")
        print(f"            {_summary(payload)}")
        if not payload["substrate"]["data_sources"]:
            print("            (no data description - add --data-source/--data-start/--data-end so "
                  "others can tell whether theirs is the same search)")
    for sub, reason in result.skipped:              # local information; never written to a file
        print(f"held back   {sub}: {reason}", file=sys.stderr)
    if not result.payloads:
        print("nothing to export.")
        return 0
    if not (args.list or args.dry_run):
        community.write_exports(result.payloads, out)
        print(f"\nNext: open a pull request adding the file(s) under {_shown(out)}/ "
              "(see CONTRIBUTING.md, 'Share a search that found nothing').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
