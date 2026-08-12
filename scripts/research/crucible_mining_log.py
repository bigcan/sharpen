"""Independent mining log for Crucible — the fact extractor (S553-cont-158).

The Crucible already records everything a *single* run needs to be reproducible (tick row, trial
ledger, run manifest, gates hash). What it has never had is a record of the CAMPAIGN: which
substrates have been mined, at which settings, how much FDR wealth each one cost, and — the part
that keeps being misread — whether a ``promising=0`` was a RESULT or a VACUITY.

Three concrete reasons this cannot be answered by looking at any one store:

1. **The record is scattered.** Ticks live in at least nine ``results/crucible_orchestrator*/``
   directories, plus ad-hoc stores under scratch dirs, plus other git worktrees. There is no
   single store that contains the campaign.
2. **The decisive runs are the ones most likely to be lost.** The two H=21 us_equity runs that
   closed that substrate (``n_holdout_tested`` 98 and 97) were written to a scratch store on
   purpose, so they would not spend the production LORD++ account. Scratch is wiped; the finding
   is not reproducible from the repo alone once it is.
3. **``crucible_hypothesis_loop.py`` writes no tick at all.** A manual cycle produces a
   ``trial_ledger.db`` + ``loop_summary.json`` and nothing else, so mining done that way is
   invisible to ``orchestrator.db``.

This script reads every store it can find and emits one canonical, machine-checkable view. It
computes NO verdict and reads NO alpha — it only counts what the funnel already recorded, so it is
outside the CRU-2 anti-oracle boundary by construction.

The one derived quantity that matters is ``outcome_class``, which encodes the lesson that cost this
project five weeks twice over: a tick that mined but never reached the holdout gate proves nothing.

    TESTED             mined, and n_holdout_tested > 0        -> the zero is a RESULT
    SCREENED_ONLY      mined, n_holdout_tested == 0           -> the zero is VACUOUS
    SCREENED_UNKNOWN   mined, n_holdout_tested IS NULL        -> pre-v12 row; UNKNOWABLE, never read as 0
    NO_SCORE           mined, but nothing was scored          -> dedup livelock shape
    COHORT_ONLY        did not mine, but SPENT FDR wealth     -> a cohort re-adjudication of the pool
    SKIP_UNDERPOWERED  power guard refused before mining
    IDLE               not dirty / no fresh hypotheses
    ERROR              tick raised

Usage:
    python scripts/research/crucible_mining_log.py
    python scripts/research/crucible_mining_log.py --root C:/tmp --out docs/research/crucible_mining_log_facts.md
    python scripts/research/crucible_mining_log.py --jsonl results/crucible_mining_log/ticks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

TICK_COLS = ("tick_ts", "substrate_id", "dirty", "mined", "reason", "n_preregistered", "n_scored",
             "n_promising", "fdr_charged_total", "status", "panel_T", "holdout_bars",
             "implied_mde_delta_sr", "power_interp_mode", "n_holdout_tested", "manifest_hash")

# The power ceiling the gates file enforces. Quoted here for READING the log only; the authoritative
# value is `configs/crucible_power.gates.yaml` and this script never gates anything on it.
POWER_CEILING_REFERENCE = 0.50


def repo_root() -> Path:
    """Main checkout, even when this script runs from a worktree (`.git` is a file there)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        dot_git = parent / ".git"
        if dot_git.is_dir():
            return parent
        if dot_git.is_file():
            line = dot_git.read_text(encoding="utf-8").strip()
            if line.startswith("gitdir:"):
                common = Path(line.split(":", 1)[1].strip())
                # .../<main>/.git/worktrees/<name>  ->  <main>
                for anc in common.parents:
                    if anc.name == ".git":
                        return anc.parent
    return here.parents[2]


def default_roots() -> list[Path]:
    root = repo_root()
    roots = [root]
    wt = root / ".claude" / "worktrees"
    if wt.is_dir():
        roots.extend(sorted(p for p in wt.iterdir() if p.is_dir()))
    return roots


def _rows(db: Path, sql: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql)]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _has_table(db: Path, table: str) -> bool:
    return bool(_rows(db, f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'"))


def read_ticks(db: Path) -> list[dict]:
    if not _has_table(db, "ticks"):
        return []
    present = {r["name"] for r in _rows(db, "PRAGMA table_info(ticks)")}
    cols = [c for c in TICK_COLS if c in present]
    out = []
    for r in _rows(db, f"SELECT {', '.join(cols)} FROM ticks ORDER BY id"):
        out.append({c: r.get(c) for c in TICK_COLS})
    return out


def read_ledger(db: Path) -> dict:
    """Per-store trial-ledger rollup. Counts only — no formula, no DSR, no verdict is interpreted."""
    if not db.exists() or not _has_table(db, "trial_ledger"):
        return {}
    present = {r["name"] for r in _rows(db, "PRAGMA table_info(trial_ledger)")}
    agg: dict = {"n_rows": _rows(db, "SELECT COUNT(*) n FROM trial_ledger")[0]["n"]}
    agg["by_verdict"] = {(r["verdict"] or "(none)"): r["n"] for r in
                         _rows(db, "SELECT verdict, COUNT(*) n FROM trial_ledger GROUP BY 1")}
    if "candidate_type" in present:
        agg["by_type"] = {(r["candidate_type"] or "(none)"): r["n"] for r in
                          _rows(db, "SELECT candidate_type, COUNT(*) n FROM trial_ledger GROUP BY 1")}
    agg["versions"] = sorted(
        {r["crucible_version"] for r in _rows(db, "SELECT DISTINCT crucible_version FROM trial_ledger")
         if r["crucible_version"]})
    span = _rows(db, "SELECT MIN(proposal_ts) lo, MAX(proposal_ts) hi, "
                     "COUNT(DISTINCT first_seen_run) runs FROM trial_ledger")
    if span:
        agg.update(first_ts=span[0]["lo"], last_ts=span[0]["hi"], n_runs=span[0]["runs"])
    if "rejection_class" in present:
        agg["classified_rejections"] = _rows(
            db, "SELECT COUNT(*) n FROM trial_ledger WHERE rejection_class IS NOT NULL")[0]["n"]
    return agg


def classify(t: dict) -> str:
    if (t.get("status") or "").upper() == "ERROR":
        return "ERROR"
    if (t.get("reason") or "").upper().startswith("UNDERPOWERED"):
        return "SKIP_UNDERPOWERED"
    if not t.get("mined"):
        # A tick that mined nothing but still charged the LORD++ account tested SOMETHING: since
        # crucible-v13.0 that is the cohort re-adjudicating an existing pool. Calling it IDLE would
        # hide the only gate on this project with measured power.
        return "COHORT_ONLY" if (t.get("fdr_charged_total") or 0.0) > 0.0 else "IDLE"
    if not t.get("n_scored"):
        return "NO_SCORE"
    n_ho = t.get("n_holdout_tested")
    if n_ho is None:
        return "SCREENED_UNKNOWN"
    if n_ho == 0:
        return "SCREENED_ONLY"
    return "TESTED"


def _label(store: Path, checkouts: list[Path]) -> tuple[str, str]:
    """(display label, class). Canonical == a `results/` store inside a checkout of THIS repo."""
    store = store.resolve()
    for co in checkouts:
        try:
            rel = store.relative_to(co.resolve())
        except ValueError:
            continue
        if "results" in rel.parts:
            return rel.as_posix(), "canonical"
    return store.as_posix(), "scratch"


def scan(roots: list[Path], checkouts: list[Path]) -> list[dict]:
    """One record per store. `store_class` is canonical iff the store lives under a repo `results/`."""
    stores: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for db in sorted(root.rglob("orchestrator.db")):
            if ".git" in db.parts or ".mypy_cache" in db.parts:
                continue
            key = str(db.parent.resolve())
            if key in stores:
                continue
            label, klass = _label(db.parent, checkouts)
            stores[key] = {
                "store": key,
                "label": label,
                "store_class": klass,
                "ticks": read_ticks(db),
                "ledger": read_ledger(db.parent / "trial_ledger.db"),
            }
    # A ledger with no orchestrator.db beside it is a manual `crucible_hypothesis_loop.py` cycle.
    for root in roots:
        if not root.is_dir():
            continue
        for db in sorted(root.rglob("trial_ledger.db")):
            if ".git" in db.parts or ".mypy_cache" in db.parts:
                continue
            key = str(db.parent.resolve())
            if key in stores or (db.parent / "orchestrator.db").exists():
                continue
            ledger = read_ledger(db)
            if not ledger.get("n_rows"):
                continue
            stores[key] = {
                "store": key,
                "label": _label(db.parent, checkouts)[0],
                "store_class": "manual-cycle",
                "ticks": [],
                "ledger": ledger,
            }
    return sorted(stores.values(), key=lambda s: (s["store_class"] != "canonical", s["label"]))


def flatten(stores: list[dict]) -> list[dict]:
    """Tick rows with `outcome_class` and a duplicate marker.

    Rehearsal stores are byte-copies of the production one, so the same tick appears many times.
    A row is `dup=True` when an identical (tick_ts, substrate, n_preregistered) was already seen in
    a canonical store — counting those again would inflate every campaign total.
    """
    seen_canonical: set[tuple] = set()
    rows: list[dict] = []
    for s in sorted(stores, key=lambda x: (x["store_class"] != "canonical", x["store"])):
        for t in s["ticks"]:
            ident = (t["tick_ts"], t["substrate_id"], t["n_preregistered"])
            dup = ident in seen_canonical
            if s["store_class"] == "canonical":
                seen_canonical.add(ident)
            rows.append({"store": s["store"], "label": s["label"], "store_class": s["store_class"],
                         "outcome_class": classify(t), "duplicate": dup, **t})
    return sorted(rows, key=lambda r: (r["tick_ts"] or "", r["store"]))


def rollup(rows: list[dict]) -> dict:
    by_sub: dict[str, dict] = defaultdict(
        lambda: {"ticks": 0, "mined": 0, "tested": 0, "screened_only": 0, "screened_unknown": 0,
                 "cohort_only": 0, "underpowered_skips": 0, "errors": 0, "prereg": 0, "scored": 0,
                 "holdout_tested": 0, "promising": 0, "fdr": 0.0, "first": None, "last": None,
                 "mde": set()})
    for r in rows:
        if r["duplicate"]:
            continue
        d = by_sub[r["substrate_id"] or "(unknown)"]
        d["ticks"] += 1
        d["mined"] += int(bool(r["mined"]))
        d["prereg"] += r["n_preregistered"] or 0
        d["scored"] += r["n_scored"] or 0
        d["holdout_tested"] += r["n_holdout_tested"] or 0
        d["promising"] += r["n_promising"] or 0
        d["fdr"] += r["fdr_charged_total"] or 0.0
        d["tested"] += int(r["outcome_class"] == "TESTED")
        d["screened_only"] += int(r["outcome_class"] == "SCREENED_ONLY")
        d["screened_unknown"] += int(r["outcome_class"] == "SCREENED_UNKNOWN")
        d["cohort_only"] += int(r["outcome_class"] == "COHORT_ONLY")
        d["underpowered_skips"] += int(r["outcome_class"] == "SKIP_UNDERPOWERED")
        d["errors"] += int(r["outcome_class"] == "ERROR")
        ts = r["tick_ts"]
        d["first"] = ts if d["first"] is None else min(d["first"], ts)
        d["last"] = ts if d["last"] is None else max(d["last"], ts)
        mde = r["implied_mde_delta_sr"]
        if mde is not None:
            d["mde"].add(round(float(mde), 3))
    return by_sub


def _n(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def render(stores: list[dict], rows: list[dict]) -> str:
    live = [r for r in rows if not r["duplicate"]]
    out: list[str] = []
    out.append("# Crucible mining log — FACTS (auto-generated)\n")
    out.append("Regenerate with `python scripts/research/crucible_mining_log.py`. **Do not hand-edit** —\n"
               "curated campaign entries and lessons live in `crucible_mining_log.md`.\n")
    out.append(f"Stores scanned: **{len(stores)}** · tick rows: **{len(rows)}** "
               f"({len(live)} unique, {len(rows) - len(live)} rehearsal duplicates)\n")

    out.append("\n## Per-substrate rollup (unique ticks only)\n")
    out.append("| substrate | window | ticks | mined | TESTED | screened-only | screened-unknown | "
               "cohort-only | underpowered skips | prereg | scored | holdout-tested | PROMISING | "
               "FDR wealth | implied MDE |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sub, d in sorted(rollup(live).items()):
        window = f"{(d['first'] or '')[:10]} -> {(d['last'] or '')[:10]}"
        mde = ", ".join(str(m) for m in sorted(d["mde"])) or "-"
        out.append(f"| `{sub}` | {window} | {d['ticks']} | {d['mined']} | {d['tested']} | "
                   f"{d['screened_only']} | {d['screened_unknown']} | {d['cohort_only']} | "
                   f"{d['underpowered_skips']} | {d['prereg']} | {d['scored']} | "
                   f"{d['holdout_tested']} | **{d['promising']}** | {d['fdr']:.4f} | {mde} |")

    out.append("\n## Every tick, oldest first\n")
    out.append("| # | tick_ts (UTC) | substrate | outcome | prereg | scored | holdout | prom | "
               "FDR | panel_T | holdout_bars | MDE | store | reason |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(live, 1):
        reason = (r["reason"] or "").replace("|", "/").strip()
        store = r["label"]
        tag = "" if r["store_class"] == "canonical" else f" ({r['store_class']})"
        out.append(f"| {i} | {(r['tick_ts'] or '')[:19]} | `{r['substrate_id']}` | "
                   f"**{r['outcome_class']}** | {_n(r['n_preregistered'])} | {_n(r['n_scored'])} | "
                   f"{_n(r['n_holdout_tested'])} | {_n(r['n_promising'])} | "
                   f"{_n(r['fdr_charged_total'], 3)} | {_n(r['panel_T'])} | {_n(r['holdout_bars'])} | "
                   f"{_n(r['implied_mde_delta_sr'])} | `{store}`{tag} | {reason} |")

    out.append("\n## Stores\n")
    out.append("| store | class | ticks | ledger rows | versions | classified rejections |")
    out.append("|---|---|---|---|---|---|")
    for s in stores:
        lg = s["ledger"]
        out.append(f"| `{s['label']}` | {s['store_class']} | {len(s['ticks'])} | "
                   f"{lg.get('n_rows', 0)} | {', '.join(lg.get('versions', [])) or '-'} | "
                   f"{lg.get('classified_rejections', '-')} |")

    warn: list[str] = []
    at_risk = sorted({r["label"] for r in live
                      if r["store_class"] != "canonical" and r["outcome_class"] == "TESTED"})
    for st in at_risk:
        warn.append(f"**AT-RISK RECORD** — a TESTED tick lives outside the repo at `{st}`. "
                    "Its finding is unreproducible once that path is cleared.")
    vac = [r for r in live if r["outcome_class"] in ("SCREENED_ONLY", "SCREENED_UNKNOWN", "NO_SCORE")]
    if vac:
        warn.append(f"**{len(vac)} mining ticks never reached the holdout gate** "
                    "(SCREENED_ONLY / SCREENED_UNKNOWN / NO_SCORE). Their `promising=0` is not a result.")
    no_rej = [s for s in stores if s["ledger"].get("n_rows") and
              s["ledger"].get("classified_rejections") == 0]
    if no_rej:
        warn.append(f"**`rejection_class` is empty in {len(no_rej)} of {len(stores)} stores** — the "
                    "ledger cannot presently say WHERE candidates died (DECISIVE vs UNDERPOWERED).")
    out.append("\n## Flags\n")
    out.extend(f"- {w}" for w in warn) if warn else out.append("- none")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="Directory to scan (repeatable). Default: main checkout + its worktrees. "
                         "Pass scratch dirs (e.g. C:/tmp) explicitly — they hold decisive runs.")
    ap.add_argument("--out", default="docs/research/crucible_mining_log_facts.md")
    ap.add_argument("--jsonl", default=None, help="Also write one JSON object per tick row.")
    args = ap.parse_args()

    checkouts = default_roots()
    roots = [Path(r) for r in args.root] if args.root else checkouts
    stores = scan(roots, checkouts)
    rows = flatten(stores)
    md = render(stores, rows)

    # Relative paths resolve against the CWD, not the main checkout: when this runs from a worktree
    # the log belongs on THAT branch, not in the primary checkout it happens to scan.
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[mining-log] {len(stores)} stores, {len(rows)} ticks -> {out}")

    if args.jsonl:
        jl = Path(args.jsonl).resolve()
        jl.parent.mkdir(parents=True, exist_ok=True)
        with jl.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        print(f"[mining-log] jsonl -> {jl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
