"""Read and check the community ledger of closed searches (``community/ledger/``).

    python scripts/crucible_community_ledger.py check --substrate us_equity
    python scripts/crucible_community_ledger.py validate

``check`` answers "has anyone already searched this?" before you spend compute. It prints advice for
a person; nothing in the mining funnel reads the community ledger.

``validate`` checks every file against ``docs/schemas/community_closed_search.schema.json`` and for
internal consistency (the id matches the content, the counts add up, the verdict follows from them,
no local path or address). It exits 1 on any problem, and CI runs it on every pull request.

To add your own search, see ``scripts/crucible_export_closed_search.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible import community  # noqa: E402


def _validate(ledger: Path) -> int:
    report = community.validate_tree(ledger)
    bad = {rel: errors for rel, errors in report.items() if errors}
    for rel, errors in bad.items():
        print(f"INVALID {rel}")
        for e in errors:
            print(f"    {e}")
    print(f"{len(report) - len(bad)} valid, {len(bad)} invalid ({ledger.as_posix()})")
    return 1 if bad else 0


def _check(ledger: Path, substrates: list[str]) -> int:
    entries = community.load_community_ledger(ledger)
    if not substrates:
        substrates = sorted({e["substrate"]["id"] for e in entries})
        if not substrates:
            print("the community ledger is empty.")
    for sub in substrates:
        print(community.describe_substrate(entries, sub))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("validate", "check every file; exit 1 on any problem"),
                       ("check", "what the ledger says about a substrate")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--ledger", default=str(community.DEFAULT_LEDGER_DIR),
                       help="community ledger directory (default: community/ledger)")
        if name == "check":
            p.add_argument("--substrate", action="append", default=None,
                           help="substrate id (repeatable; default: every substrate on record)")
    args = ap.parse_args(argv)
    ledger = Path(args.ledger)
    if not ledger.is_dir():
        ap.error(f"no such directory: {args.ledger}")
    if args.command == "validate":
        return _validate(ledger)
    return _check(ledger, args.substrate or [])


if __name__ == "__main__":
    raise SystemExit(main())
