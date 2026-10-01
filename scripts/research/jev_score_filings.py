"""Score the ATL x Jev corpus with Jev (Phase 2 step 4; ``docs/research/atl_jev_phase2_architecture.md`` §3).

    python scripts/research/jev_score_filings.py --mode screening --max-usd 10
    python scripts/research/jev_score_filings.py --mode screening --smoke --max-usd 0.05 \
        --corpus data/atl_jev/corpus_smoke/screening --out data/atl_jev/scores_smoke/screening

Refuses unless the gates file's questionnaire is this code's and the corpus was built by the current text
pipeline; outside ``--smoke`` the corpus must also span the mode's full window with no failed filer. ``--max-usd``
is required: the run stops before any call if an upper-bound estimate exceeds it, and between chunks if the spend
does. Answers land in the shared per-question cache, so rerunning an interrupted run costs nothing twice.
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

from sharpen.jev import JevClient, JevError  # noqa: E402
from sharpen.jev.corpus import build_stamp, corpus_window, read_corpus  # noqa: E402
from sharpen.jev.scoring import (  # noqa: E402
    ServedVersionChanged,
    check_corpus_stamp,
    check_questionnaire,
    score_corpus,
    write_scores,
)
from sharpen.jev.stamps import gates_sha, git_commit  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
JEV_CACHE = ROOT / "results" / "atl_jev" / "jev_cache.sqlite"
# Pre-run cost estimate: Jev bills input tokens. The 2019 smoke corpus billed 3.08 text characters per input token
# (questions included), so text_chars / 2.5 over-counts; cache hits are not subtracted, which over-counts again.
# The hard stop is the per-chunk check of the actual bill against --max-usd.
CHARS_PER_TOKEN_LOWER_BOUND = 2.5
# Waits after a JevError that outlasted the client's own retries (e.g. HTTP 529 "system_overloaded"), then a cap.
OVERLOAD_WAITS_S = (60, 120, 240, 480, 900)
MAX_OVERLOAD_WAIT_S = 2 * 3600

log = logging.getLogger("jev_score_filings")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("screening", "clean"))
    ap.add_argument("--p4-authorization", type=Path, help="the P4 sentinel; required by --mode clean")
    ap.add_argument("--corpus", type=Path, help="corpus shard directory (default data/atl_jev/corpus/<mode>)")
    ap.add_argument("--out", type=Path, help="output directory (default data/atl_jev/scores/<mode>)")
    ap.add_argument("--smoke", action="store_true", help="allow a narrowed or partial corpus (never a verdict)")
    ap.add_argument("--limit", type=int, help="score only the first N filings (implies --smoke)")
    ap.add_argument("--max-usd", type=float, required=True, help="spend ceiling for this run")
    ap.add_argument("--chunk", type=int, default=256, help="filings per progress/budget/version check")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    smoke = args.smoke or args.limit is not None

    gates = yaml.safe_load(GATES.read_bytes())
    phase1, jcfg = gates["phase1"], gates["jev"]
    check_questionnaire(phase1)
    window = corpus_window(phase1, mode=args.mode, p4_authorization=args.p4_authorization)
    corpus_dir = args.corpus or ROOT / "data" / "atl_jev" / "corpus" / args.mode
    out_dir = args.out or ROOT / "data" / "atl_jev" / "scores" / args.mode

    man_path = corpus_dir / "_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"{corpus_dir} has no _manifest.json: the corpus build has not finished")
    corpus_manifest = json.loads(man_path.read_text(encoding="utf-8"))
    stamp = corpus_manifest["stamp"]
    check_corpus_stamp(stamp, mode=args.mode)
    if not smoke:
        full = build_stamp(window, int(phase1["filings"]["max_chars"]))
        if (stamp["since"], stamp["until"]) != (full["since"], full["until"]):
            raise SystemExit(f"corpus window {stamp['since']}..{stamp['until']} is not the full {args.mode} window")
        if corpus_manifest["stats"]["failed"]:
            raise SystemExit(f"corpus has failed filers {sorted(corpus_manifest['stats']['failed'])}: rerun the build")
    corpus = read_corpus(corpus_dir, expect_stamp=stamp)
    if args.limit is not None:
        corpus = corpus.head(args.limit)

    price = float(jcfg["price_usd_per_m_input"])
    upper = corpus["text_len"].sum() / CHARS_PER_TOKEN_LOWER_BOUND * price / 1e6
    log.info("%d filings, %d with text; cost upper bound $%.2f (ceiling $%.2f)", len(corpus),
             int((corpus["text_len"] > 0).sum()), upper, args.max_usd)
    if upper > args.max_usd:
        raise SystemExit(f"estimated cost up to ${upper:.2f} exceeds --max-usd {args.max_usd}")

    client = JevClient(model=jcfg["model"], cache_path=JEV_CACHE,
                       max_questions_per_request=int(jcfg["max_questions_per_request"]),
                       max_retries=int(jcfg["max_retries"]))
    t0 = time.monotonic()

    def on_chunk(done: int, total: int, stats) -> None:
        spent = client.usage.cost_usd(price)
        log.info("%d/%d filings | %.1f min | $%.4f | served %s", done, total, (time.monotonic() - t0) / 60, spent,
                 dict(stats.served_models))
        if spent > args.max_usd:
            raise SystemExit(f"spend ${spent:.4f} passed --max-usd {args.max_usd}")

    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": args.mode,
                "smoke": smoke, "corpus": str(corpus_dir), "corpus_stamp": stamp, "gates_sha": gates_sha(GATES),
                "questionnaire": phase1["questionnaire"], "model_requested": jcfg["model"], **git_commit(ROOT)}
    waited, attempt = 0, 0
    try:
        while True:
            try:
                df, stats = score_corpus(corpus, client, concurrency=int(jcfg["concurrency"]), chunk=args.chunk,
                                         on_chunk=on_chunk)
                break
            except JevError as exc:
                # A sustained overload outlasts the client's per-call retries. Answers already received are
                # cached, so waiting and rerunning costs nothing twice; the same client keeps the spend total.
                wait = OVERLOAD_WAITS_S[min(attempt, len(OVERLOAD_WAITS_S) - 1)]
                if waited + wait > MAX_OVERLOAD_WAIT_S:
                    raise
                log.warning("Jev unavailable (%s); resuming from the cache in %d min", str(exc)[:160], wait // 60)
                time.sleep(wait)
                waited, attempt = waited + wait, attempt + 1
    except ServedVersionChanged as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_manifest.json").write_text(json.dumps({**manifest, "status": "STOPPED", "reason": str(exc),
                                                            "jev_usage": client.usage.to_json(price)}, indent=1),
                                                encoding="utf-8")
        log.error("%s", exc)
        return 3
    manifest.update({"status": "COMPLETE", "stats": stats.to_json(), "jev_usage": client.usage.to_json(price),
                     "elapsed_min": round((time.monotonic() - t0) / 60, 1)})
    path = write_scores(df, out_dir, manifest)
    log.info("scores %s: %s | usage %s", path, json.dumps(stats.to_json()), json.dumps(manifest["jev_usage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
