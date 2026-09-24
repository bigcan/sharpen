"""Build the ATL x Jev filing corpus (Phase 2 step 3; ``docs/research/atl_jev_phase2_architecture.md`` §2).

    python scripts/research/jev_build_corpus.py --mode screening
    python scripts/research/jev_build_corpus.py --mode screening --tickers AAPL,TXN,SO \
        --since 2019-01-01 --until 2020-01-01 --out data/atl_jev/corpus_smoke

The universe is every S&P 500 member during the mode's evaluation window. Tickers map to CIKs by exact key
only; the map and its coverage are written beside the corpus (full runs also write
``results/atl_jev/phase2/cik_map_<mode>.json``). Resumable: a CIK whose shard carries the current stamp is
skipped, and a failed CIK keeps no shard, so rerunning the same command finishes the job. Clean mode is
P4-only (``--p4-authorization``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sharpen.crucible.data.edgar_filings as edgar_filings_module  # noqa: E402
from sharpen.crucible.data.edgar_filings import EdgarFilingsClient  # noqa: E402
from sharpen.data.fundamentals import load_ticker_cik_map  # noqa: E402
from sharpen.jev.corpus import (  # noqa: E402
    Filer,
    acceptance_spells,
    build_cik_map,
    build_corpus,
    build_stamp,
    corpus_window,
    coverage_by_year,
    load_constituents,
    load_pit_rows,
    load_sec_tickers,
    max_hold_days,
    membership_spells,
    universe_tickers,
)
from sharpen.jev.stamps import gates_sha, git_commit, source_sha  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
EQ = ROOT / "data" / "raw" / "equity_panel"
SEC_CACHE = ROOT / "data" / "raw" / "fundamentals"
EDGAR_CACHE = ROOT / "data" / "edgar_filings"
DEFAULT_OUT = ROOT / "data" / "atl_jev" / "corpus"
RESULTS = ROOT / "results" / "atl_jev" / "phase2"

log = logging.getLogger("jev_build_corpus")


def _day(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("screening", "clean"))
    ap.add_argument("--p4-authorization", type=Path, help="the P4 sentinel; required by --mode clean")
    ap.add_argument("--tickers", help="comma-separated subset of the universe (a smoke test)")
    ap.add_argument("--since", help="narrow the acceptance window (YYYY-MM-DD, inside the mode's window)")
    ap.add_argument("--until", help="exclusive end of the narrowed window (YYYY-MM-DD)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=1, help="concurrent document fetches")
    ap.add_argument("--per-second", type=float, default=2.0,
                    help="EDGAR request rate. SEC's stated ceiling is 10/s, but 4/s over 4 connections drew HTTP 429")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    phase1 = yaml.safe_load(GATES.read_bytes())["phase1"]
    narrow = None
    if args.since or args.until:
        if not (args.since and args.until):
            ap.error("--since and --until go together")
        narrow = (_day(args.since), _day(args.until))
    window = corpus_window(phase1, mode=args.mode, p4_authorization=args.p4_authorization, narrow=narrow)
    smoke = bool(args.tickers or narrow)

    dates, members = load_pit_rows(EQ / "sp500_pit_members.csv")
    universe = universe_tickers(dates, members, window.rows_start, window.rows_end)
    load_ticker_cik_map(cache_dir=SEC_CACHE)                      # fetches and caches company_tickers.json once
    cons, cons_names = load_constituents(EQ / "sp500_constituents.csv")
    sec, sec_titles = load_sec_tickers(SEC_CACHE / "company_tickers.json")
    cmap = build_cik_map(universe, constituents=cons, sec_tickers=sec)
    coverage = coverage_by_year(cmap, dates, members, window.rows_start, window.rows_end)

    hold = max_hold_days(phase1)
    filers = [Filer(cik, ts, tuple(n for n in (cons_names.get(cik), *sec_titles.get(cik, ())) if n),
                    acceptance_spells(membership_spells(dates, members, ts, window.rows_start, window.rows_end),
                                      window, hold))
              for cik, ts in cmap.filers().items()]
    filers = [f for f in filers if f.spells]          # a narrowed smoke window can miss a name's membership
    if args.tickers:
        want = {t.strip().upper().replace(".", "-") for t in args.tickers.split(",") if t.strip()}
        missing = sorted(want - set(cmap.mapped))
        if missing:
            raise SystemExit(f"not mapped members of the {args.mode} universe: {missing}")
        filers = [f for f in filers if want & set(f.tickers)]

    out_dir = args.out / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    edgar = EdgarFilingsClient(cache_dir=EDGAR_CACHE, per_second=args.per_second)
    t0 = time.monotonic()

    def progress(i: int, n: int, f: Filer, local) -> None:
        if i % 10 == 0 or i == n:
            log.info("[%d/%d] %.0f min | last %s: %d in scope of %d listed", i, n, (time.monotonic() - t0) / 60,
                     ",".join(f.tickers), local.in_scope, local.listed)

    stats = build_corpus(filers, window, edgar, out_dir, max_chars=int(phase1["filings"]["max_chars"]),
                         workers=args.workers, on_progress=progress)

    cik_record = {"universe": phase1["universe"], "window": window.to_json(), "n_universe": len(universe),
                  "coverage_by_year": coverage, **cmap.to_json()}
    (out_dir / "cik_map.json").write_text(json.dumps(cik_record, indent=1), encoding="utf-8")
    if not smoke:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"cik_map_{args.mode}.json").write_text(json.dumps(cik_record, indent=1), encoding="utf-8")
    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "smoke": smoke,
                "stamp": build_stamp(window, int(phase1["filings"]["max_chars"])),
                "gates_sha": gates_sha(GATES), "edgar_filings_sha": source_sha(edgar_filings_module.__file__),
                **git_commit(ROOT), "n_filers": len(filers), "stats": stats.to_json(),
                "elapsed_min": round((time.monotonic() - t0) / 60, 1)}
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log.info("corpus %s: %s", out_dir, json.dumps(stats.to_json()))
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
