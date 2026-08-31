"""Crucible P5 — governance scan: hand off CLEARED survivors to the human Tier-2 gate (spec §3).

Opens a lockbox DB, finds survivors that have CLEARED forward incubation (CR-8) and have not yet been
handed off, and for each: builds a Tier-2 handoff packet (verbatim verdict + forward evidence + the
EXACT human-run `deep_strategy_audit` command), notifies the operator, and records the handoff so it
happens exactly once. It NEVER runs the audit — promotion to capital is human-initiated (CLAUDE.md).

Run it standalone or after ``crucible_orchestrator.py`` (point ``--cards`` at the orchestrator's
discovery-card output so the packet carries the in-sample verdict).

Usage:
  python scripts/research/crucible_governance.py --lockbox results/.../lockbox.db \
      --gov results/.../governance.db --out results/.../handoffs --cards results/.../synthetic \
      --workstream crucible --scope overlay --now-ts 2026-07-02T00:00:00
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crucible.governance import (  # noqa: E402
    FileNotifier,
    GovernanceStore,
    card_from_dir,
    scan_and_handoff,
)
from sharpen.crucible.lockbox.lockbox import Lockbox  # noqa: E402

log = logging.getLogger("crucible_governance")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible governance — hand off CLEARED survivors")
    ap.add_argument("--lockbox", required=True, help="lockbox SQLite DB")
    ap.add_argument("--gov", default=None, help="governance SQLite DB (default: alongside --lockbox)")
    ap.add_argument("--out", required=True, help="dir to write handoff packets + notifications")
    ap.add_argument("--cards", default=None, help="dir of discovery cards for the verbatim verdict")
    ap.add_argument("--substrate", default=None, help="restrict to one substrate_id")
    ap.add_argument("--workstream", default="crucible")
    ap.add_argument("--scope", default="crucible-survivor")
    ap.add_argument("--now-ts", default=None, help="ISO handoff timestamp (default: now UTC)")
    args = ap.parse_args()

    now_ts = args.now_ts or datetime.now(timezone.utc).isoformat()
    gov_path = args.gov or str(Path(args.lockbox).with_name("governance.db"))
    out_dir = Path(args.out)

    lockbox = Lockbox(args.lockbox)
    store = GovernanceStore(gov_path)
    notifier = FileNotifier(out_dir / "notifications")
    lookup = card_from_dir(args.cards) if args.cards else None

    created = scan_and_handoff(
        lockbox, store, notifier, handoff_ts=now_ts, workstream=args.workstream, scope=args.scope,
        substrate_id=args.substrate, card_lookup=lookup, out_dir=out_dir)

    if not created:
        log.info("no newly-CLEARED survivors to hand off (eligible-but-already-handed-off are skipped)")
    for h in created:
        log.info("HANDOFF %s -> human Tier-2: %s", h.candidate_hash, h.audit_command)
    log.info("done: %d handoff(s) written to %s", len(created), out_dir)
    lockbox.close()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
