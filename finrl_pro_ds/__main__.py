"""CLI entry point for FinRL Pro scaffolding commands."""

from __future__ import annotations

import argparse
from typing import Iterable

from finrl_pro_ds.utils.module_scaffolder import ModuleScaffolder


def main(argv: Iterable[str] | None = None) -> None:
    """Dispatch FinRL Pro CLI commands."""
    parser = argparse.ArgumentParser(prog="finrl-pro")
    subparsers = parser.add_subparsers(dest="command")

    scaffold_parser = subparsers.add_parser(
        "scaffold", help="Scaffold a module inside the finrl_pro_ds namespace."
    )
    scaffold_parser.add_argument(
        "module",
        help="Dotted path for the new module (must start with finrl_pro_ds.).",
    )
    scaffold_parser.add_argument(
        "--doc",
        dest="doc",
        default=None,
        help="Optional docstring to write into the generated module.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "scaffold":
        scaffolder = ModuleScaffolder()
        target = scaffolder.scaffold(args.module, docstring=args.doc)
        print(f"Created module at {target}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
