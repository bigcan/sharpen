"""crucible-v15.0 operator tool — find (and, only on request, reopen) pre-registrations that were CHARGED
but NEVER TESTED before v15.0, and quantify the online-FDR wealth they consumed.

Before crucible-v15.0 two defects combined (deep audit 2026-09-30):

  * a pre-registered seed over the GP search bounds (> max_ast_nodes, or turnover > 2x the soft cap) was
    culled before fitness and never reached the holdout — 40 of the 100 published WQ101 formulas exceed
    24 nodes — yet it was ledgered SCORED_NOT_SELECTED ("scored and lost");
  * the orchestrator charged a LORD++ test for every fresh spec, tested or not (the 2026-08-10 us_equity
    extended-bank tick: 107 charged, 16 adjudicated), and every phantom charge also decayed the level of
    the next REAL test.

v15.0 fixes both going forward. This tool deals with the RECORD. It is DRY-RUN by default and prints:

  1. per substrate, from the tick log: pre-registered vs holdout-adjudicated per mined tick, i.e. how many
     LORD++ charges bought no test (phantoms), and what the next level is now vs after compaction;
  2. per ledger row: pre-registrations with no rejection class whose formula is over the size bound —
     DEFINITELY never tested (turnover culls cannot be identified without rebuilding the panel, so they are
     reported as a count gap, not guessed).

Two opt-in actions, each on an explicit flag (operator decisions — they rewrite a store's state):

  --apply-reopen      set verdict = NULL on the definitely-untested rows. The ledger view (and the Author's
                      dedup) treat a NULL verdict as "registered, not yet scored", exactly the state the
                      2026-08-09 livelock fix made re-proposable: the next tick may pre-register them again —
                      same candidate_hash, ORIGINAL proposal_ts and spec_json kept (the upsert's
                      first-value-wins rule), so it is the same hypothesis finally tested, not a new one.
  --apply-fdr-refund  compact the substrate's LORD++ account by its phantom count. Valid because a test
                      that never ran cannot produce a false discovery, and every real test's historical
                      level was <= what the compacted stream would have given it (compaction only moves
                      real tests EARLIER in the index). Refused when the account holds a discovery (the
                      discovery indices would need re-mapping — do that by hand).

Never run it against a store another process is writing. Back the store up first.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.orchestrator.fdr import OnlineFDR  # noqa: E402
from sharpen.signals.generation.grammar import node_count, parse  # noqa: E402

log = logging.getLogger("crucible_v15_reopen")


def _phantoms(orch_db: Path) -> dict[str, dict]:
    """Per substrate: tick-log phantom charges (pre-registered minus holdout-adjudicated, summed over
    mined ticks whose n_holdout_tested is known) + ticks whose denominator is unknown (pre-v12 rows)."""
    con = sqlite3.connect(f"file:{orch_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out: dict[str, dict] = {}
    for r in con.execute("SELECT substrate_id, n_preregistered, n_holdout_tested FROM ticks "
                         "WHERE mined = 1 ORDER BY id"):
        d = out.setdefault(r["substrate_id"], {"phantom": 0, "tested": 0, "unknown_ticks": 0,
                                               "unknown_preregs": 0})
        if r["n_holdout_tested"] is None:
            d["unknown_ticks"] += 1
            d["unknown_preregs"] += int(r["n_preregistered"] or 0)
            continue
        d["tested"] += int(r["n_holdout_tested"])
        d["phantom"] += max(0, int(r["n_preregistered"] or 0) - int(r["n_holdout_tested"]))
    for sub, d in out.items():
        row = con.execute("SELECT state_json FROM fdr_state WHERE substrate_id = ?", (sub,)).fetchone()
        d["fdr"] = json.loads(row["state_json"]) if row else None
    con.close()
    return out


def _untested_rows(ledger_db: Path, max_ast_nodes: int) -> list[dict]:
    """Pre-registrations with no rejection class, recorded as lost, whose formula exceeds the size bound
    (culled before fitness pre-v15 → definitely never adjudicated)."""
    con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = {r["name"] for r in con.execute("PRAGMA table_info(trial_ledger)")}
    rc = "rejection_class IS NULL AND " if "rejection_class" in cols else ""
    rows = con.execute(
        "SELECT candidate_hash, formula, verdict, first_seen_run, fdr_wealth_charged FROM trial_ledger "
        f"WHERE {rc}spec_json IS NOT NULL AND formula IS NOT NULL "
        "AND verdict IN ('SCORED_NOT_SELECTED', 'LOGGED')").fetchall()
    con.close()
    out = []
    for r in rows:
        try:
            n = node_count(parse(r["formula"]))
        except Exception:                                     # noqa: BLE001 — unparseable legacy row
            continue
        if n > max_ast_nodes:
            out.append({**dict(r), "nodes": n})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", required=True,
                    help="directory holding trial_ledger.db + orchestrator.db (e.g. results/.../real)")
    ap.add_argument("--max-ast-nodes", type=int, default=24,
                    help="the search bound the store's runs used (every shipped gates file: 24)")
    ap.add_argument("--apply-reopen", action="store_true")
    ap.add_argument("--apply-fdr-refund", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = Path(args.store)
    ledger_db, orch_db = store / "trial_ledger.db", store / "orchestrator.db"

    phantoms = _phantoms(orch_db) if orch_db.exists() else {}
    for sub, d in phantoms.items():
        fdr = d["fdr"]
        line = (f"[{sub}] tested {d['tested']}  phantom charges {d['phantom']}  "
                f"ticks with unknown denominator {d['unknown_ticks']} ({d['unknown_preregs']} preregs)")
        if fdr:
            now = OnlineFDR.from_json(fdr)
            compact = OnlineFDR.from_json({**fdr, "num_tests": max(0, now.num_tests - d["phantom"])})
            line += (f"  | account num_tests {now.num_tests} next level {now.next_level():.3g} -> "
                     f"after compaction {compact.num_tests} next level {compact.next_level():.3g}"
                     f"{'  (HAS DISCOVERIES — refund refused)' if now.discoveries else ''}")
        log.info(line)

    untested = _untested_rows(ledger_db, args.max_ast_nodes) if ledger_db.exists() else []
    log.info("definitely-untested pre-registrations (> %d nodes, no rejection class): %d",
             args.max_ast_nodes, len(untested))
    for r in untested[:20]:
        log.info("  %s  %-19s nodes=%d  run=%s", r["candidate_hash"], r["verdict"], r["nodes"],
                 r["first_seen_run"])

    if not (args.apply_reopen or args.apply_fdr_refund):
        log.info("DRY RUN — nothing written. --apply-reopen / --apply-fdr-refund are operator decisions.")
        return 0
    for db in (ledger_db, orch_db):
        if db.exists():
            shutil.copy2(db, db.with_suffix(db.suffix + ".pre_v15_reopen.bak"))
    if args.apply_reopen and untested:
        con = sqlite3.connect(ledger_db)
        con.executemany("UPDATE trial_ledger SET verdict = NULL WHERE candidate_hash = ?",
                        [(r["candidate_hash"],) for r in untested])
        con.commit()
        con.close()
        log.info("reopened %d rows (verdict -> NULL); backup written beside the store", len(untested))
    if args.apply_fdr_refund:
        con = sqlite3.connect(orch_db)
        for sub, d in phantoms.items():
            if not d["fdr"] or not d["phantom"]:
                continue
            if d["fdr"].get("discoveries"):
                log.warning("[%s] account holds discoveries — refund REFUSED (re-map indices by hand)", sub)
                continue
            new = {**d["fdr"], "num_tests": max(0, int(d["fdr"]["num_tests"]) - d["phantom"])}
            con.execute("UPDATE fdr_state SET state_json = ? WHERE substrate_id = ?",
                        (json.dumps(new), sub))
            log.info("[%s] LORD++ account compacted by %d phantom tests", sub, d["phantom"])
        con.commit()
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
