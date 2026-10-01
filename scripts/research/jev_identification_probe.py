"""Phase-0 C3 — can Jev name the filer of an ANONYMIZED earnings release? (ATL x Jev plan §3.2)

WHY. Masking names, tickers, places and dates (plan C2) only protects a backtest if the masked text
no longer tells Jev WHICH company — and therefore which known history — it is reading. If Jev can
still pick the filer, its answers about a pre-cutoff filing can lean on memory, and only the clean
window (post-T_c) and the forward lockbox can certify the strategy.

HOW. One Item 2.02 8-K per company (random, seeded) from ``period_start``..``period_end``: all ATL
DJIA-30 names plus random other S&P 500 names up to ``n_filings``. The EX-99.1 press release (the 8-K
body when absent) is anonymized with :mod:`sharpen.jev.anonymize` and truncated to ``max_chars``. Jev
then answers two ``choice`` questions: which of ``n_candidates`` companies (true filer + same-GICS-sector
decoys, shuffled) published it, and which calendar year. The same questions on the UNMASKED text are a
control that shows the question itself works. Identifiable = masked top-1 accuracy above
``max_top1_accuracy`` (configs/atl_jev.gates.yaml, sha256 stamped).

    python scripts/research/jev_identification_probe.py [--n N] [--skip-control]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.data.edgar_filings import EdgarFilingsClient  # noqa: E402
from sharpen.jev import JevClient  # noqa: E402
from sharpen.jev.anonymize import anonymize, residual_identifiers  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
OUT_DIR = ROOT / "results" / "atl_jev" / "phase0"
JEV_CACHE = ROOT / "results" / "atl_jev" / "jev_cache.sqlite"
EDGAR_CACHE = ROOT / "data" / "edgar_filings"
CONSTITUENTS = ROOT / "data" / "raw" / "equity_panel" / "sp500_constituents.csv"
DJIA_30 = ("AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD",
           "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW",
           "TRV", "UNH", "V", "WMT")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

log = logging.getLogger("jev_identification_probe")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def load_constituents() -> list[dict]:
    rows, seen = [], set()
    with open(CONSTITUENTS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cik = int(r["CIK"])
            if cik in seen:                               # GOOG/GOOGL, FOX/FOXA, NWS/NWSA: one per filer
                continue
            seen.add(cik)
            rows.append({"ticker": r["Symbol"].replace(".", "-"), "name": r["Security"],
                         "sector": r["GICS Sector"], "cik": cik})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=None, help="override n_filings (debug)")
    ap.add_argument("--skip-control", action="store_true", help="masked condition only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    graw = GATES.read_bytes()
    gates = yaml.safe_load(graw)
    g, jcfg = gates["phase0"]["identification_probe"], gates["jev"]
    n_target = int(args.n or g["n_filings"])
    rng = np.random.default_rng(int(g["sample_seed"]))
    since = datetime.fromisoformat(g["period_start"]).replace(tzinfo=timezone.utc)
    until = datetime.fromisoformat(g["period_end"]).replace(hour=23, minute=59, tzinfo=timezone.utc)
    years = [str(y) for y in range(since.year, until.year + 1)]

    cons = load_constituents()
    by_sector: dict[str, list[dict]] = {}
    for c in cons:
        by_sector.setdefault(c["sector"], []).append(c)
    djia = [c for c in cons if c["ticker"] in DJIA_30]
    others = [c for c in cons if c["ticker"] not in DJIA_30]
    order = djia + [others[i] for i in rng.permutation(len(others))]

    edgar = EdgarFilingsClient(cache_dir=EDGAR_CACHE)
    samples = []
    for c in order:
        if len(samples) >= n_target:
            break
        # Per-company stream: a transient failure on one filer must not reshuffle every later draw
        # (the shared stream made the sample path-dependent — audit 2026-09-23).
        crng = np.random.default_rng([int(g["sample_seed"]), int(c["cik"])])
        try:
            refs = [r for r in edgar.filings(c["cik"], forms=("8-K",), since=since, until=until)
                    if "2.02" in r.items]
            if not refs:
                log.info("skip %s: no Item 2.02 8-K in window", c["ticker"])
                continue
            ref = refs[int(crng.integers(len(refs)))]
            docs = edgar.documents(ref, types=("8-K", "EX-99.1"))
            doc_type = "EX-99.1" if docs.get("EX-99.1") else "8-K"
            raw = docs.get(doc_type) or ""
            if len(raw) < 500:
                log.info("skip %s: document too short (%d chars)", c["ticker"], len(raw))
                continue
            sub = edgar.submissions(c["cik"])
            aliases = [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", [])]
            tickers = sorted({c["ticker"], *sub.get("tickers", [])})
        except Exception as exc:                          # noqa: BLE001 - one bad filer must not stop the probe
            log.warning("skip %s: %r", c["ticker"], exc)
            continue
        # The cover page exists only on the 8-K itself; stripping an exhibit at an incidental
        # "Item N.NN" mention would silently drop its text (audit finding, measured 0/182 on 2026-09-23).
        masked = anonymize(raw, legal_name=c["name"], tickers=tickers, aliases=aliases,
                           drop_cover_page=(doc_type == "8-K"))[: int(g["max_chars"])]
        decoys = [d for d in by_sector[c["sector"]] if d["cik"] != c["cik"]]
        pick = [decoys[i] for i in crng.choice(len(decoys), size=int(g["n_candidates"]) - 1, replace=False)]
        cands = pick + [c]
        cands = [cands[i] for i in crng.permutation(len(cands))]
        samples.append({
            "ticker": c["ticker"], "name": c["name"], "sector": c["sector"], "djia": c["ticker"] in DJIA_30,
            "accession": ref.accession, "accepted_utc": ref.accepted_utc.isoformat(), "doc_type": doc_type,
            "year": str(ref.accepted_utc.year), "masked": masked, "raw": raw[: int(g["max_chars"])],
            "residual": residual_identifiers(masked, legal_name=c["name"], tickers=tickers, aliases=aliases),
            "options": {LETTERS[i]: d["name"] for i, d in enumerate(cands)},
            "answer": LETTERS[[d["cik"] for d in cands].index(c["cik"])]})
        log.info("[%d/%d] %s %s %s", len(samples), n_target, c["ticker"], ref.accession, ref.accepted_utc.date())

    def questions(s: dict) -> dict:
        return {"company": {"type": "choice", "criteria": s["options"], "instructions":
                            "Company names, tickers, places and dates may have been removed from this "
                            "document. Which company most likely published it?"},
                "year": {"type": "choice", "criteria": {y: y for y in years}, "instructions":
                         "In which calendar year was this document most likely published?"}}

    client = JevClient(model=jcfg["model"], cache_path=JEV_CACHE,
                       max_questions_per_request=int(jcfg["max_questions_per_request"]),
                       max_retries=int(jcfg["max_retries"]))
    conditions = ["masked"] if args.skip_control else ["masked", "raw"]
    t0 = time.monotonic()
    res = {}
    for cond in conditions:
        res[cond] = client.ask_many([(s[cond], questions(s)) for s in samples],
                                    concurrency=int(jcfg["concurrency"]))
    wall = time.monotonic() - t0

    summary = {}
    for cond in conditions:
        hit = np.array([r["company"].value == s["answer"] for r, s in zip(res[cond], samples)])
        p_true = np.array([(r["company"].probabilities or {}).get(s["answer"], float("nan"))
                           for r, s in zip(res[cond], samples)])
        yr = np.array([r["year"].value == s["year"] for r, s in zip(res[cond], samples)])
        dj = np.array([s["djia"] for s in samples])
        summary[cond] = {
            "company_top1": round(float(hit.mean()), 4), "company_top1_wilson95": wilson(int(hit.sum()), hit.size),
            "company_top1_djia": round(float(hit[dj].mean()), 4) if dj.any() else None,
            "company_top1_other": round(float(hit[~dj].mean()), 4) if (~dj).any() else None,
            "mean_p_true_company": round(float(np.nanmean(p_true)), 4),
            "year_top1": round(float(yr.mean()), 4), "year_top1_wilson95": wilson(int(yr.sum()), yr.size)}
    identifiable = bool(summary["masked"]["company_top1"] > float(g["max_top1_accuracy"]))

    record = {
        "probe": "C3 identification", "gates_file": str(GATES.relative_to(ROOT)),
        "gates_sha256": hashlib.sha256(graw).hexdigest(), "gates": g, "n_filings": len(samples),
        "n_djia": int(sum(s["djia"] for s in samples)),
        "chance": {"company": 1 / int(g["n_candidates"]), "year": 1 / len(years)},
        "masked_with_residual_identifiers": int(sum(bool(s["residual"]) for s in samples)),
        "summary": summary, "decision": {"identifiable": identifiable,
                                         "rule": f"masked company_top1 > {g['max_top1_accuracy']}"},
        "wall_s": round(wall, 1), "jev_usage": client.usage.to_json(float(jcfg["price_usd_per_m_input"])),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "identification_probe.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    rates = OUT_DIR / "jev_rates_identification.json"
    prev = json.loads(rates.read_text(encoding="utf-8"))["requests"] if rates.exists() else 0
    if client.usage.requests > prev:     # keep the LARGEST live measurement; cached re-runs never overwrite
        rates.write_text(json.dumps(
            {"source": "live run", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record["jev_usage"]},
            indent=1), encoding="utf-8")
    with open(OUT_DIR / "identification_probe_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "sector", "djia", "accession", "accepted_utc", "doc_type", "answer", "residual",
                    *[f"{c}_{k}" for c in conditions for k in ("company", "p_true", "year")]])
        for i, s in enumerate(samples):
            row = [s["ticker"], s["sector"], int(s["djia"]), s["accession"], s["accepted_utc"], s["doc_type"],
                   s["answer"], "|".join(s["residual"])]
            for c in conditions:
                r = res[c][i]
                row += [r["company"].value, (r["company"].probabilities or {}).get(s["answer"]), r["year"].value]
            w.writerow(row)
    log.info("n=%d | %s | identifiable=%s | %s", len(samples), json.dumps(summary), identifiable,
             json.dumps(record["jev_usage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
