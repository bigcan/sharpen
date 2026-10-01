"""Phase-1 instrument check — does each draft question MEASURE something? (ATL x Jev pre-registration)

Runs the draft questionnaire (``sharpen.jev.questionnaire``) on screening-era filings BEFORE it is frozen and
reads NO returns: the only question is whether the instrument is valid, never whether it predicts. Rules
(``configs/atl_jev.gates.yaml`` phase1.instrument_check, sha256 stamped):

* non-degenerate: a question's mapped answers must vary (std >= ``min_answer_std``) once it has
  ``min_n_for_std`` answers — a question Jev answers identically everywhere carries no information;
* internally consistent: an outlook cannot be both raised (E1) and lowered (E2) on most releases.

Sample: Item 2.02 releases (EX-99.1) and adverse-item 8-Ks without a 2.02, from S&P 500 constituents in the
instrument window (screening era only — the clean window stays untouched), drawn per company from a seeded
stream. Masking as in the strategy. Answers are cached, so Phase 3 reuses them for free.

    python scripts/research/jev_instrument_check.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.data.edgar_filings import EdgarFilingsClient  # noqa: E402
from sharpen.jev import JevClient  # noqa: E402
from sharpen.jev.anonymize import anonymize  # noqa: E402
from sharpen.jev.questionnaire import (  # noqa: E402
    ADVERSE_ITEMS,
    EARNINGS_ITEM,
    QUESTIONS_V1,
    map_answer,
    questionnaire_hash,
    questions_for,
)

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
OUT_DIR = ROOT / "results" / "atl_jev" / "phase1"
JEV_CACHE = ROOT / "results" / "atl_jev" / "jev_cache.sqlite"
EDGAR_CACHE = ROOT / "data" / "edgar_filings"
CONSTITUENTS = ROOT / "data" / "raw" / "equity_panel" / "sp500_constituents.csv"
KEYWORDS = re.compile(r"(outlook|guidance|raise[sd]?|lower(ed|s)?|withdr[ae]w|impairment|restructuring|"
                      r"restatement|resign|depart|successor)", re.I)

log = logging.getLogger("jev_instrument_check")


def document_text(docs: dict[str, str], earnings: bool, name: str, tickers: list[str], aliases: list[str],
                  max_chars: int) -> str:
    """Masked text in the strategy's form: an earnings release is its EX-99.1 (8-K body if absent); an event
    8-K is its body (cover page stripped) followed by any EX-99.1."""
    parts = []
    if earnings and docs.get("EX-99.1"):
        parts.append(anonymize(docs["EX-99.1"], legal_name=name, tickers=tickers, aliases=aliases,
                               drop_cover_page=False))
    else:
        if docs.get("8-K"):
            parts.append(anonymize(docs["8-K"], legal_name=name, tickers=tickers, aliases=aliases))
        if docs.get("EX-99.1"):
            parts.append(anonymize(docs["EX-99.1"], legal_name=name, tickers=tickers, aliases=aliases,
                                   drop_cover_page=False))
    return "\n\n".join(parts)[:max_chars]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    graw = GATES.read_bytes()
    gates = yaml.safe_load(graw)
    g, p1, jcfg = gates["phase1"]["instrument_check"], gates["phase1"], gates["jev"]
    since = datetime.fromisoformat(g["window"][0]).replace(tzinfo=timezone.utc)
    until = datetime.fromisoformat(g["window"][1]).replace(hour=23, minute=59, tzinfo=timezone.utc)

    cons, seen = [], set()
    with open(CONSTITUENTS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["CIK"]) not in seen:
                seen.add(int(r["CIK"]))
                cons.append({"ticker": r["Symbol"].replace(".", "-"), "name": r["Security"], "cik": int(r["CIK"])})
    order = [cons[i] for i in np.random.default_rng(int(g["sample_seed"])).permutation(len(cons))]

    edgar = EdgarFilingsClient(cache_dir=EDGAR_CACHE)
    want = {"earnings": int(g["n_earnings"]), "event": int(g["n_events"])}
    picked: list[dict] = []
    for c in order:
        if all(sum(p["kind"] == k for p in picked) >= n for k, n in want.items()):
            break
        crng = np.random.default_rng([int(g["sample_seed"]), c["cik"]])
        try:
            refs = edgar.filings(c["cik"], forms=("8-K",), since=since, until=until)
            sub = edgar.submissions(c["cik"])
        except Exception as exc:                          # noqa: BLE001 - one filer must not stop the check
            log.warning("skip %s: %r", c["ticker"], exc)
            continue
        aliases = [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", [])]
        tickers = sorted({c["ticker"], *sub.get("tickers", [])})
        pools = {"earnings": [r for r in refs if EARNINGS_ITEM in r.items],
                 "event": [r for r in refs if EARNINGS_ITEM not in r.items and set(r.items) & ADVERSE_ITEMS]}
        for kind, pool in pools.items():
            if not pool or sum(p["kind"] == kind for p in picked) >= want[kind]:
                continue
            ref = pool[int(crng.integers(len(pool)))]
            try:
                docs = edgar.documents(ref, types=("8-K", "EX-99.1"))
            except Exception as exc:                      # noqa: BLE001
                log.warning("skip %s %s: %r", c["ticker"], ref.accession, exc)
                continue
            text = document_text(docs, kind == "earnings", c["name"], tickers, aliases,
                                 int(p1["filings"]["max_chars"]))
            if len(text) < 300:
                continue
            raw = (docs.get("EX-99.1") or "") + " " + (docs.get("8-K") or "")
            picked.append({"kind": kind, "ticker": c["ticker"], "accession": ref.accession,
                           "items": list(ref.items), "text": text, "raw": raw,
                           "questions": questions_for(ref.items)})

    client = JevClient(model=jcfg["model"], cache_path=JEV_CACHE,
                       max_questions_per_request=int(jcfg["max_questions_per_request"]),
                       max_retries=int(jcfg["max_retries"]))
    results = client.ask_many([(p["text"], {q.qid: q.request() for q in p["questions"]}) for p in picked],
                              concurrency=int(jcfg["concurrency"]))

    by_q: dict[str, list[float]] = {q.qid: [] for q in QUESTIONS_V1}
    conflict = n_rel = 0
    rows = []
    for p, res in zip(picked, results):
        mapped = {q.qid: map_answer(q, res[q.qid]) for q in p["questions"]}
        for k, v in mapped.items():
            by_q[k].append(v)
        if p["kind"] == "earnings":
            n_rel += 1
            conflict += int(res["E1"].value > 0.5 and res["E2"].value > 0.5)
        rows.append({"kind": p["kind"], "ticker": p["ticker"], "accession": p["accession"],
                     "items": ",".join(p["items"]), **{k: round(v, 3) for k, v in mapped.items()}})

    stats, failures = {}, []
    for q in QUESTIONS_V1:
        v = np.array(by_q[q.qid])
        st = {"n": int(v.size), "mean": round(float(v.mean()), 3) if v.size else None,
              "std": round(float(v.std()), 3) if v.size else None,
              "decisive_frac": round(float((np.abs(v) > 0.5).mean()), 3) if v.size else None}
        if v.size >= int(g["min_n_for_std"]):
            st["rule"] = "applied"
            if v.std() < float(g["min_answer_std"]):
                failures.append(f"{q.qid}: std {v.std():.3f} < {g['min_answer_std']}")
        else:
            st["rule"] = f"reported only (n < {g['min_n_for_std']})"
        stats[q.qid] = st
    conflict_frac = conflict / max(n_rel, 1)
    if conflict_frac > float(g["max_guidance_conflict_frac"]):
        failures.append(f"E1&E2 conflict {conflict_frac:.3f} > {g['max_guidance_conflict_frac']}")

    # Spot-check material for a human read: the most decisive E1/E2/M1/M2 answers with keyword context.
    spot = {}
    for qid in ("E1", "E2", "M1", "M2"):
        scored = [(r[qid], i) for i, r in enumerate(rows) if qid in r]
        for label, (val, i) in (("max", max(scored, default=(None, None))), ("min", min(scored, default=(None, None)))):
            if i is None:
                continue
            ctx = [m.group(0) for m in KEYWORDS.finditer(picked[i]["raw"])][:8]
            snippets = [picked[i]["raw"][max(0, m.start() - 120): m.end() + 160].replace("\n", " ")
                        for m in list(KEYWORDS.finditer(picked[i]["raw"]))[:3]]
            spot[f"{qid}_{label}"] = {"value": val, "ticker": rows[i]["ticker"],
                                      "accession": rows[i]["accession"], "keywords": ctx, "snippets": snippets}

    record = {"check": "phase-1 instrument check (no returns read)", "gates_sha256": hashlib.sha256(graw).hexdigest(),
              "questionnaire_hash": questionnaire_hash(), "rules": g,
              "n_filings": {k: sum(p["kind"] == k for p in picked) for k in want},
              "per_question": stats, "guidance_conflict_frac": round(conflict_frac, 3),
              "failures": failures, "passed": not failures,
              "jev_usage": client.usage.to_json(float(jcfg["price_usd_per_m_input"])), "spot_check": spot}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "instrument_check.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    with open(OUT_DIR / "instrument_check_answers.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["kind", "ticker", "accession", "items"] + [q.qid for q in QUESTIONS_V1]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    log.info("instrument check: %s | %s | failures=%s | %s", record["n_filings"], json.dumps(stats),
             failures, json.dumps(record["jev_usage"]))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
