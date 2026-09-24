"""Run the ATL x Jev screening legs (Phase 2 step 6; ``docs/research/atl_jev_phase2_architecture.md`` §6).

    python scripts/research/atl_jev_evaluate.py --leg p1
    python scripts/research/atl_jev_evaluate.py --leg p2
    python scripts/research/atl_jev_evaluate.py --leg p3
    python scripts/research/atl_jev_evaluate.py --leg screening      # the K3 decision

Inputs: the screening panel (``atl_jev_build_panel.py``), the complete screening scores (``jev_score_filings.py``),
and for P3 the corpus plus the pinned Loughran-McDonald file (``configs/atl_jev_baselines.yaml``). Each leg writes
``results/atl_jev/phase3/<leg>.json`` stamped with its inputs. P2, P3 and the decision refuse to run unless P1 is
complete on the same inputs. P3 reports BLOCKED, never a pass, while the lexicon is unpinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.version import gates_hash  # noqa: E402
from sharpen.jev.baselines import baseline_scores, load_lm_lexicon  # noqa: E402
from sharpen.jev.corpus import read_corpus  # noqa: E402
from sharpen.jev.evaluation import (  # noqa: E402
    FROZEN_FUNNEL_GATES_HASH,
    k3_decision,
    promising,
    registered,
    run_p1,
    run_p2,
    run_p3,
)
from sharpen.jev.scoring import check_questionnaire, to_filing_scores  # noqa: E402
from sharpen.jev.stamps import gates_sha, git_commit  # noqa: E402
from sharpen.signals.gates import Gates  # noqa: E402
from sharpen.signals.scorecard import to_json  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
FUNNEL = ROOT / "configs" / "signal_eval.gates.yaml"
BASELINES_CFG = ROOT / "configs" / "atl_jev_baselines.yaml"
PANEL = ROOT / "data" / "atl_jev" / "panels" / "screening.pkl"
SCORES = ROOT / "data" / "atl_jev" / "scores" / "screening"
CORPUS = ROOT / "data" / "atl_jev" / "corpus" / "screening"
OUT = ROOT / "results" / "atl_jev" / "phase3"

log = logging.getLogger("atl_jev_evaluate")


def _load_inputs(phase1: dict) -> tuple:
    if gates_hash(FUNNEL) != FROZEN_FUNNEL_GATES_HASH:
        raise SystemExit(f"{FUNNEL.name} hashes to {gates_hash(FUNNEL)}, not the frozen {FROZEN_FUNNEL_GATES_HASH}")
    for need, how in ((PANEL, "atl_jev_build_panel.py"), (SCORES / "_manifest.json", "jev_score_filings.py")):
        if not need.is_file():
            raise SystemExit(f"missing {need}: run scripts/research/{how} first")
    blob = PANEL.read_bytes()
    pman = json.loads(PANEL.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(blob).hexdigest() != pman["panel_sha256"]:
        raise SystemExit("screening panel does not match its manifest: rebuild it")
    saved = pickle.loads(blob)                        # noqa: S301 - our own artifact, hash-checked above
    sman = json.loads((SCORES / "_manifest.json").read_text(encoding="utf-8"))
    if sman.get("status") != "COMPLETE" or sman.get("smoke"):
        raise SystemExit(f"scores at {SCORES} are not a complete screening run (status {sman.get('status')}, "
                         f"smoke {sman.get('smoke')})")
    if sman["questionnaire"] != phase1["questionnaire"]:
        raise SystemExit("scores were produced under another questionnaire")
    scores = pd.read_parquet(SCORES / "filing_scores.parquet")
    stamps = {"gates_sha": gates_sha(GATES), "funnel_gates_hash": gates_hash(FUNNEL),
              "questionnaire": phase1["questionnaire"], "panel_sha256": pman["panel_sha256"],
              "scores_built_utc": sman["built_utc"], "served_models": sman["stats"]["served_models"],
              "corpus_stamp": sman["corpus_stamp"]}
    return saved["panel"], saved["calendar"], scores, stamps


def _write(leg: str, record: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{leg}.json"
    p.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return p


def _require_p1(stamps: dict) -> dict:
    p = OUT / "p1.json"
    if not p.is_file():
        raise SystemExit("run --leg p1 first")
    p1 = json.loads(p.read_text(encoding="utf-8"))
    if p1.get("status") != "COMPLETE" or p1.get("stamps") != stamps:
        raise SystemExit("p1.json is incomplete or was computed on other inputs: rerun --leg p1")
    return p1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", required=True, choices=("p1", "p2", "p3", "screening"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    phase1 = yaml.safe_load(GATES.read_bytes())["phase1"]
    check_questionnaire(phase1)
    gates = Gates.from_yaml(FUNNEL)
    panel, calendar, scores, stamps = _load_inputs(phase1)
    base = {"leg": args.leg, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stamps": stamps, **git_commit(ROOT)}
    t0 = time.monotonic()

    if args.leg == "p1":
        rs = run_p1(registered(phase1, scores, calendar), panel, gates, phase1)
        # a ticker spelled differently in the corpus and the panel would drop its filings without an error
        fs = to_filing_scores(scores)
        in_panel = np.isin(fs.ticker, np.asarray(panel.tickers, dtype=str))
        coverage = {"filing_ticker_rows": int(fs.ticker.size), "in_panel": int(in_panel.sum()),
                    "tickers_not_in_panel": sorted(set(fs.ticker[~in_panel].tolist()))}
        rec = {**base, "status": "COMPLETE", "promising": promising(rs), "filing_coverage": coverage,
               "scorecard": to_json(rs)}
    elif args.leg in ("p2", "p3"):
        p1 = _require_p1(stamps)
        names = p1["promising"]
        results: dict = {}
        if args.leg == "p3":
            cfg = yaml.safe_load(BASELINES_CFG.read_text(encoding="utf-8"))["lm_lexicon"]
            try:
                lex = load_lm_lexicon(ROOT / cfg["path"], sha256=cfg["sha256"])
            except (ValueError, FileNotFoundError) as exc:
                rec = {**base, "status": "BLOCKED", "reason": str(exc), "signals": {}}
                log.error("P3 blocked: %s", exc)
                _write("p3", rec)
                return 2
            corpus_man = json.loads((CORPUS / "_manifest.json").read_text(encoding="utf-8"))
            if corpus_man["stamp"] != stamps["corpus_stamp"]:
                raise SystemExit("the corpus on disk is not the one the scores were built from")
            baselines = baseline_scores(read_corpus(CORPUS, expect_stamp=corpus_man["stamp"]), lex)
            base["lm_lexicon"] = {"release": cfg["release"], "sha256": lex.sha256}
        for n in names:
            log.info("%s: %s", args.leg, n)
            results[n] = (run_p2(n, phase1, scores, calendar, panel, gates) if args.leg == "p2"
                          else run_p3(n, phase1, scores, baselines, calendar, panel, gates))
            if args.leg == "p2":
                card = next(c for c in p1["scorecard"]["cards"] if c["name"] == n)
                p1_ic = card["gross"]["by_horizon"][str(gates.primary_horizon)]["ic_mean"]
                if p1_ic is None or abs(results[n]["real_ic"] - p1_ic) > 1e-9:
                    raise SystemExit(f"{n}: P2's IC {results[n]['real_ic']} does not reproduce P1's {p1_ic}")
        rec = {**base, "status": "COMPLETE", "signals": results,
               "note": "" if names else "no PROMISING signal in P1: K3 fires regardless"}
    else:
        p1 = _require_p1(stamps)
        legs = {}
        for leg in ("p2", "p3"):
            path = OUT / f"{leg}.json"
            rec_leg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if rec_leg.get("status") != "COMPLETE" or rec_leg.get("stamps") != stamps:
                raise SystemExit(f"{leg}.json is missing, not complete, or from other inputs: run --leg {leg}")
            legs[leg] = rec_leg["signals"]
        rec = {**base, "status": "COMPLETE", **k3_decision(p1["promising"], legs["p2"], legs["p3"])}
    rec["elapsed_min"] = round((time.monotonic() - t0) / 60, 1)
    path = _write(args.leg, rec)
    log.info("%s -> %s", args.leg, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
