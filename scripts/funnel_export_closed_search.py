"""Share a signal-funnel search that found nothing: export a scorecard to the community ledger.

    python scripts/funnel_export_closed_search.py \
        --scorecard results/<batch>/scorecard.json \
        --describe community/ledger/_describe/<batch>.json --out community/ledger/

For a batch scored directly by ``sharpen.signals`` (``evaluate_batch`` then ``write_scorecard``)
against a pre-registration, without the Crucible orchestrator. Writes one
``funnel_closed_search`` file to ``<out>/<substrate>/<preregistered_on>-<id>.json``: how many specs
were pre-registered, scored and passed, the declared multiplicity they were deflated against, the
author's measured power, the specs' hashes, and a verdict —

    CLOSED_DECISIVE     a planted control was recovered and the detection bar is at or below the
                        economic floor (smallest IC-IR worth having), and nothing passed
    OPEN_UNDERPOWERED   the test could not tell either way (NOT "nothing there")

No signal name, score or per-signal verdict is written. A scorecard holding a PROMISING card is
refused (exit 1, nothing written) unless you pass ``--include-promising``.

The describe file says what the scorecard cannot:

    {"substrate": {"id", "universe", "data_sources", "data_start", "data_end", "search_space"},
     "campaign": {"preregistered_on", "gates_sha256_12", "primary_horizon_days" (optional)},
     "power": {"detection_bar_ic_ir", "economic_floor_ic_ir", "planted_control_passed"},
     "sharpen_version": "<version that scored the batch>"}

Keep describe files under ``community/ledger/_describe/``; the validator skips that directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible import community  # noqa: E402

_KEYS = {"substrate": {"id", "universe", "data_sources", "data_start", "data_end", "search_space"},
         "campaign": {"preregistered_on", "gates_sha256_12", "primary_horizon_days"},
         "power": {"detection_bar_ic_ir", "economic_floor_ic_ir", "planted_control_passed"},
         "sharpen_version": None}


def _read_describe(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--describe must be a JSON object")
    unknown = set(raw) - set(_KEYS)
    missing = set(_KEYS) - set(raw)
    if unknown or missing:
        raise ValueError(f"--describe: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for block, keys in _KEYS.items():
        if keys is None:
            continue
        if not isinstance(raw[block], dict):
            raise ValueError(f"--describe: {block} must be an object")
        extra = set(raw[block]) - keys
        if extra:
            raise ValueError(f"--describe: unknown keys in {block}: {sorted(extra)}")
    return raw


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build(scorecard: Path, describe: Path, *, include_promising: bool = False) -> dict:
    d = _read_describe(describe)
    sub, camp, pw = d["substrate"], d["campaign"], d["power"]
    if "id" not in sub or "preregistered_on" not in camp or "gates_sha256_12" not in camp:
        raise ValueError("--describe needs substrate.id, campaign.preregistered_on and "
                         "campaign.gates_sha256_12")
    card = json.loads(scorecard.read_text(encoding="utf-8"))
    return community.export_funnel_search(
        card,
        substrate_id=str(sub["id"]),
        description=community.SubstrateDescription(
            universe=sub.get("universe"), data_sources=tuple(sub.get("data_sources") or ()),
            data_start=sub.get("data_start"), data_end=sub.get("data_end"),
            search_space=sub.get("search_space")),
        campaign=community.FunnelCampaign(
            preregistered_on=str(camp["preregistered_on"]),
            gates_sha256_12=str(camp["gates_sha256_12"]),
            primary_horizon_days=camp.get("primary_horizon_days")),
        power=community.FunnelPower(
            detection_bar_ic_ir=pw.get("detection_bar_ic_ir"),
            economic_floor_ic_ir=pw.get("economic_floor_ic_ir"),
            planted_control_passed=pw.get("planted_control_passed")),
        sharpen_version=str(d["sharpen_version"]),
        include_promising=include_promising)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", required=True, help="scorecard.json written by write_scorecard")
    ap.add_argument("--describe", required=True,
                    help="JSON file: substrate, campaign, power and sharpen_version (see above)")
    ap.add_argument("--out", default="community/ledger", help="community ledger directory")
    ap.add_argument("--include-promising", action="store_true",
                    help="export a scorecard holding a PROMISING card as PROMISING_FOUND "
                         "(off by default)")
    ap.add_argument("--dry-run", action="store_true", help="build the file but do not write it")
    args = ap.parse_args(argv)

    scorecard, describe = Path(args.scorecard), Path(args.describe)
    for path in (scorecard, describe):
        if not path.is_file():
            ap.error(f"no such file: {path.as_posix()}")
    try:
        payload = build(scorecard, describe, include_promising=args.include_promising)
    except community.PromisingHeldBack as e:            # local information; nothing is written
        print(f"held back: {e}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as e:
        ap.error(str(e))

    out = Path(args.out)
    path = out / community.relative_path(payload)
    t, c = payload["trials"], payload["campaign"]
    print(f"{'would write' if args.dry_run else 'wrote':<11} {_shown(path)}")
    print(f"            {payload['verdict']} {payload['verdict_reasons']}: {t['scored']} of "
          f"{t['preregistered']} specs scored, deflated for {c['declared_multiplicity']}; "
          f"{community.power_phrase(payload)}")
    if not args.dry_run:
        community.write_exports([payload], out)
        print(f"\nNext: open a pull request adding the file under {_shown(out)}/ "
              "(see CONTRIBUTING.md, 'Share a search that found nothing').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
