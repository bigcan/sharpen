"""Community ledger of closed searches — export, schema, privacy and one-way-glass tripwires."""
from __future__ import annotations

import copy
import getpass
import hashlib
import importlib.util
import json
import platform
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from sharpen.crucible import community, gates_hash
from sharpen.crucible.ledger import TrialLedger, TrialRecord
from sharpen.crucible.manifest import RunManifest
from sharpen.crucible.orchestrator.substrate import OrchestratorStore, TickRecord
from sharpen.crucible.search_memory import candidate_hash, semantic_hash
from sharpen.sharpops import gate_registry

ROOT = Path(__file__).resolve().parents[2]
_V = "crucible-v16.0"
_SECRET = "a private prior nobody else should read"
# Fields that would turn a shared file into a fitness oracle, or name a hypothesis.
_FORBIDDEN_KEYS = {"dsr", "delta_sr_oos", "marginal_hlz_t", "formula", "economic_rationale",
                   "spec_json", "verdicts", "p_value", "fdr_wealth_charged", "rejection_class",
                   "implied_mde_at_test", "candidate_hash", "holdout_delta", "rng_seeds"}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ts(day: int) -> str:
    return f"2026-03-{day:02d}T00:00:00+00:00"


def _row(led: TrialLedger, sub: str, day: int, formula: str, *, prereg: bool, verdict: str | None,
         rejection: str | None = None, mde: float | None = None, ctype: str = "cross_sectional") -> None:
    led.record(TrialRecord(
        candidate_hash=candidate_hash(formula), crucible_version=_V,
        family="momentum" if prereg else None, candidate_type=ctype,
        spec_json='{"horizon": 5}' if prereg else None, formula=formula,
        economic_rationale=_SECRET if prereg else None, first_seen_run=f"tick-{sub}-{_ts(day)}",
        proposal_ts=_ts(day), verdict=verdict, dsr=0.42, delta_sr_oos=0.11, marginal_hlz_t=3.5,
        data_snapshot_hash=f"snap{day:02d}", rejection_class=rejection, implied_mde_at_test=mde))


def _tick(store: OrchestratorStore, out: Path, sub: str, day: int, *, holdout: int | None,
          promising: int = 0, mde: float | None = None, mined: bool = True, prereg: int = 3) -> None:
    man = RunManifest(run_id=f"tick-{sub}-{_ts(day)}", crucible_version=_V, gates_hash="519158fa1450",
                      contract="corrected", corrected_gates_hash="d1523707b860",
                      search_memory_gates_hash="26a0f6433347", data_snapshot_hash=f"snap{day:02d}",
                      agent_model_id="library-seed-v1", verdicts={"deadbeef0000": "LOGGED"})
    if mined:
        man.write(out / sub / f"day{day:02d}")
    store.record_tick(TickRecord(
        tick_ts=_ts(day), substrate_id=sub, dirty=True, mined=mined, reason="mined" if mined else "idle",
        snapshot_hash=f"snap{day:02d}", n_preregistered=prereg if mined else 0,
        n_scored=5 if mined else 0, n_holdout_tested=holdout, n_promising=promising,
        manifest_hash=man.content_hash() if mined else None, implied_mde_delta_sr=mde))


def _store(root: Path, *, reverse: bool = False) -> Path:
    """Three substrates: a dead end the test was too weak to settle, one settled decisively, and one
    that holds a hit."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        ("dead_end", 2, "rank(close)", dict(prereg=True, verdict="SCORED_NOT_SELECTED",
                                            rejection="UNDERPOWERED", mde=1.4)),
        ("dead_end", 2, "rank(volume)", dict(prereg=True, verdict="SCORED_NOT_SELECTED",
                                             rejection="UNDERPOWERED", mde=1.4)),
        ("dead_end", 2, "ts_rank(close, 10)", dict(prereg=True, verdict="NOT_TESTED")),
        ("dead_end", 2, "rank(delta(close, 5))", dict(prereg=False, verdict="LOGGED",
                                                       ctype="overlay")),
        ("dead_end", 2, "sub(high, low)", dict(prereg=True, verdict=None)),          # never scored
        ("settled", 3, "rank(ts_rank(volume, 20))", dict(prereg=True, verdict="SCORED_NOT_SELECTED",
                                                         rejection="DECISIVE", mde=0.08)),
        ("settled", 3, "rank(sub(close, open))", dict(prereg=True, verdict="SCORED_NOT_SELECTED",
                                                      rejection="DECISIVE", mde=0.08)),
        ("has_hit", 4, "add(close, volume)", dict(prereg=True, verdict="PROMISING")),
        ("has_hit", 4, "mul(-1, rank(close))", dict(prereg=True, verdict="SCORED_NOT_SELECTED",
                                                    rejection="UNDERPOWERED", mde=1.2)),
    ]
    with TrialLedger(root / "trial_ledger.db") as led:
        for sub, day, formula, kw in (reversed(rows) if reverse else rows):
            _row(led, sub, day, formula, **kw)
    with OrchestratorStore(root / "orchestrator.db") as store:
        _tick(store, root, "dead_end", 1, holdout=None, mined=False)
        _tick(store, root, "dead_end", 2, holdout=2, mde=1.4, prereg=4)
        _tick(store, root, "settled", 3, holdout=2, mde=0.08, prereg=2)
        _tick(store, root, "has_hit", 4, holdout=2, promising=1, mde=1.2, prereg=2)
    return root / "trial_ledger.db"


def _by_sub(result: community.ExportResult) -> dict[str, dict]:
    return {p["substrate"]["id"]: p for p in result.payloads}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reid(payload: dict) -> dict:
    payload["export_id"] = community.export_id(payload)
    return payload


def _keys(node) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


# ------------------------------------------------------------------------------------ round trip
def test_export_round_trip(tmp_path: Path) -> None:
    ledger = _store(tmp_path / "store")
    result = community.export_closed_searches(ledger)
    out = tmp_path / "community"
    written = community.write_exports(result.payloads, out)

    assert sorted(p.relative_to(out).as_posix() for p in written) == sorted(
        community.relative_path(p) for p in result.payloads)
    assert community.validate_tree(out) == {community.relative_path(p): [] for p in result.payloads}
    assert community.load_community_ledger(out) == sorted(result.payloads, key=community.relative_path)

    subs = _by_sub(result)
    assert set(subs) == {"dead_end", "settled"}
    dead = subs["dead_end"]
    assert dead["verdict"] == community.OPEN_UNDERPOWERED
    assert dead["verdict_reasons"] == ["underpowered_rejections", "mde_above_floor"]
    assert dead["trials"] == {"recorded": 5, "preregistered": 4, "scored": 4, "holdout_tested": 2,
                              "holdout_count_complete": True, "promising": 0}
    assert dead["rejections"] == {"decisive": 0, "underpowered": 2, "unclassified": 0}
    assert dead["power"]["mde_at_test_min"] == dead["power"]["mde_at_test_max"] == 1.4
    assert dead["power"]["mde_source"] == "ledger"
    assert dead["power"]["economic_floor"] == 0.10
    assert dead["ticks"] == {"total": 2, "mined": 1, "tested": 1, "screened_only": 0,
                             "screened_unknown": 0, "no_score": 0, "cohort_only": 0,
                             "underpowered_skips": 0, "idle": 1, "errors": 0}
    # scored hypotheses only: the never-scored pre-registration is not advertised as tested
    assert dead["hypotheses"]["semantic_hashes"] == sorted(
        semantic_hash(f) for f in ("rank(close)", "rank(volume)", "ts_rank(close, 10)",
                                   "rank(delta(close, 5))"))
    assert dead["hypotheses"]["by_type"] == {"cross_sectional": 3, "overlay": 1, "unknown": 0}
    assert dead["provenance"]["gates_hashes"] == ["519158fa1450"]
    assert dead["provenance"]["last_date"] == "2026-03-02"

    settled = subs["settled"]
    assert settled["verdict"] == community.CLOSED_DECISIVE
    assert settled["verdict_reasons"] == ["all_holdout_rejections_decisive"]
    assert settled["rejections"] == {"decisive": 2, "underpowered": 0, "unclassified": 0}


def test_export_is_deterministic(tmp_path: Path) -> None:
    """Same ledger, same bytes — across repeated exports and across row insertion order."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    first = community.write_exports(
        community.export_closed_searches(_store(tmp_path / "s1")).payloads, a)
    again = community.write_exports(
        community.export_closed_searches(tmp_path / "s1" / "trial_ledger.db").payloads, tmp_path / "a2")
    other = community.write_exports(
        community.export_closed_searches(_store(tmp_path / "s2", reverse=True)).payloads, b)
    names = [p.relative_to(a).as_posix() for p in first]
    assert names == [p.relative_to(tmp_path / "a2").as_posix() for p in again]
    assert names == [p.relative_to(b).as_posix() for p in other]
    for rel in names:
        assert (a / rel).read_bytes() == (tmp_path / "a2" / rel).read_bytes() == (b / rel).read_bytes()
        assert b"\r" not in (a / rel).read_bytes()


# ------------------------------------------------------------------------------------ schema
def test_published_schema_is_the_validators_schema() -> None:
    doc = json.loads((ROOT / "docs/schemas/community_closed_search.schema.json")
                     .read_text(encoding="utf-8"))
    assert doc == community.SCHEMA


def test_committed_ledger_is_valid() -> None:
    report = community.validate_tree(ROOT / "community" / "ledger")
    assert report, "the seeded community ledger is missing"
    assert {rel: errs for rel, errs in report.items() if errs} == {}


def _corrupt_verdict(p):
    p["verdict"], p["verdict_reasons"] = community.CLOSED_DECISIVE, ["all_holdout_rejections_decisive"]


def _corrupt_hash_order(p):
    p["hypotheses"]["semantic_hashes"].reverse()


def _corrupt_count(p):
    p["hypotheses"]["count"] += 1


def _corrupt_holdout(p):
    p["trials"]["holdout_tested"] += 5


def _corrupt_path(p):
    p["substrate"]["universe"] = r"panel built from D:\research\secret_panel.parquet"


@pytest.mark.parametrize("mutate, reid, expected", [
    (lambda p: p.pop("power"), True, {"schema: <root>: 'power' is a required property"}),
    (lambda p: p.update(dsr=0.97), True,
     {"schema: <root>: Additional properties are not allowed ('dsr' was unexpected)"}),
    (lambda p: p["hypotheses"].update(formulas=["rank(close)"]), True,
     {"schema: hypotheses: Additional properties are not allowed ('formulas' was unexpected)"}),
    (lambda p: p["trials"].update(scored=p["trials"]["scored"] - 1), False,
     {"export_id does not match the content (the file was edited after export; re-run the export "
      "instead)"}),
    (_corrupt_verdict, True,
     {"verdict CLOSED_DECISIVE does not follow from the counts (they imply OPEN_UNDERPOWERED)"}),
    (_corrupt_hash_order, True, {"hypotheses.semantic_hashes is not sorted"}),
    (_corrupt_count, True, {"hypotheses.count does not equal the number of hashes listed"}),
    (_corrupt_holdout, True,
     {"rejections and promising do not account for every holdout-tested trial"}),
    (_corrupt_path, True, {"contains a drive path: 'panel built from D:\\\\research\\\\secret_pane'"}),
], ids=["missing-block", "score-field", "formula-field", "edited-after-export", "verdict-upgraded",
        "unsorted", "count", "arithmetic", "local-path"])
def test_malformed_entry_fails_validation(tmp_path: Path, mutate, reid, expected) -> None:
    good = _by_sub(community.export_closed_searches(_store(tmp_path / "store")))["dead_end"]
    rel = community.relative_path(good)
    assert community.validate_payload(good, relpath=rel) == []      # the control: untouched passes
    bad = copy.deepcopy(good)
    mutate(bad)
    if reid:
        _reid(bad)
        rel = community.relative_path(bad) if "provenance" in bad and "substrate" in bad else rel
    assert set(community.validate_payload(bad, relpath=rel)) == expected


def test_file_in_the_wrong_place_fails(tmp_path: Path) -> None:
    good = _by_sub(community.export_closed_searches(_store(tmp_path / "store")))["dead_end"]
    assert community.validate_payload(good, relpath="settled/2026-03-02-000000000000.json") == [
        f"path should be {community.relative_path(good)}"]


def test_writer_refuses_an_invalid_payload(tmp_path: Path) -> None:
    good = _by_sub(community.export_closed_searches(_store(tmp_path / "store")))["dead_end"]
    bad = copy.deepcopy(good)
    _corrupt_verdict(bad)
    with pytest.raises(ValueError, match="does not follow from the counts"):
        community.write_exports([_reid(bad)], tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------ privacy
def test_no_private_strings_leave_the_machine(tmp_path: Path) -> None:
    marker = "PRIVATE_MARKER_DIR"
    ledger = _store(tmp_path / marker / "store")
    desc = community.SubstrateDescription(universe="  twelve   liquid futures ",
                                          data_sources=("vendor daily bars",))
    out = tmp_path / "out"
    written = community.write_exports(
        community.export_closed_searches(ledger, default_description=desc).payloads, out)
    blob = "\n".join(p.read_text(encoding="utf-8") for p in written)
    private = [marker, tmp_path.as_posix(), str(tmp_path), tmp_path.name, getpass.getuser(),
               _SECRET, "rank(close)", "horizon", "trial_ledger.db", "0.42", "has_hit"]
    if len(platform.node()) >= 4:
        private.append(platform.node())
    assert [s for s in private if s in blob] == []
    assert '"universe": "twelve liquid futures"' in blob            # the description itself survives


@pytest.mark.parametrize("text, label", [
    (r"C:\Users\me\panels\x.parquet", "a drive path"),
    ("d:/data/panel", "a drive path"),
    (r"\\nas01\share\panel", "a network path"),
    ("bars from /home/me/data/x", "a local path"),
    ("~/data/panel.parquet", "a home-directory path"),
    ("ask me@example.com", "an e-mail address"),
    ("served from 10.0.0.12", "an IP address"),
])
def test_descriptions_reject_paths_and_addresses(text: str, label: str) -> None:
    with pytest.raises(ValueError, match=label):
        community.public_text(text, "universe")


def test_ordinary_descriptions_pass() -> None:
    for text in ("S&P 500 members, top 300 by dollar volume", "Yahoo Finance daily bars",
                 "9 FX majors, hourly; 7-bar hold", "data/derived features (v2.1)"):
        assert community.public_text(text, "universe") == text
    assert community.public_text("   ", "universe") is None


# ------------------------------------------------------------------------------------ promising
def test_a_substrate_holding_a_hit_is_held_back_by_default(tmp_path: Path) -> None:
    ledger = _store(tmp_path / "store")
    result = community.export_closed_searches(ledger)
    assert "has_hit" not in _by_sub(result)
    assert [s for s, _ in result.skipped] == ["has_hit"]
    out = tmp_path / "out"
    community.write_exports(result.payloads, out)
    assert not (out / "has_hit").exists()
    assert all("has_hit" not in p.read_text(encoding="utf-8") for p in out.rglob("*.json"))
    # ... and asking for it by name changes nothing
    named = community.export_closed_searches(ledger, substrates=["has_hit"])
    assert named.payloads == [] and [s for s, _ in named.skipped] == ["has_hit"]


def test_include_promising_exports_it_as_such(tmp_path: Path) -> None:
    result = community.export_closed_searches(_store(tmp_path / "store"), include_promising=True)
    hit = _by_sub(result)["has_hit"]
    assert hit["verdict"] == community.PROMISING_FOUND and hit["trials"]["promising"] == 1
    assert result.skipped == []
    assert community.validate_payload(hit, relpath=community.relative_path(hit)) == []


def test_cannot_export_only_the_dead_end_runs_of_a_substrate_with_a_hit(tmp_path: Path) -> None:
    """Narrowing a substrate to runs that found nothing, while it holds a hit in another run, would
    publish a "nothing here" its author knows to be false — refused even with the opt-in."""
    root = tmp_path / "store"
    ledger = _store(root)
    with TrialLedger(ledger) as led:
        _row(led, "has_hit", 5, "correlation(close, volume, 10)", prereg=True,
             verdict="SCORED_NOT_SELECTED", rejection="UNDERPOWERED", mde=1.2)
    with OrchestratorStore(root / "orchestrator.db") as store:
        _tick(store, root, "has_hit", 5, holdout=1, mde=1.2, prereg=1)
    dead_run = f"tick-has_hit-{_ts(5)}"
    result = community.export_closed_searches(ledger, run_ids=[dead_run], include_promising=True)
    assert "has_hit" not in _by_sub(result)
    assert [s for s, _ in result.skipped] == ["has_hit"]
    whole = community.export_closed_searches(
        ledger, run_ids=[dead_run, f"tick-has_hit-{_ts(4)}"], include_promising=True)
    assert _by_sub(whole)["has_hit"]["verdict"] == community.PROMISING_FOUND


def test_a_hypothesis_promising_elsewhere_taints_the_substrate_that_shares_it(tmp_path: Path) -> None:
    """``add(volume, close)`` on the dead end is the same hypothesis as the PROMISING
    ``add(close, volume)``: its hash must not leave in a "found nothing" file."""
    ledger = _store(tmp_path / "store")
    assert "dead_end" in _by_sub(community.export_closed_searches(ledger))     # control
    with TrialLedger(ledger) as led:
        _row(led, "dead_end", 2, "add(volume, close)", prereg=False, verdict="LOGGED")
    result = community.export_closed_searches(ledger)
    assert sorted(s for s, _ in result.skipped) == ["dead_end", "has_hit"]
    assert set(_by_sub(result)) == {"settled"}


def test_a_promising_entry_in_a_run_manifest_holds_the_substrate_back(tmp_path: Path) -> None:
    """A cohort can pass without any single row being PROMISING; the manifest is where that shows."""
    root = tmp_path / "store"
    ledger = _store(root)
    assert "settled" in _by_sub(community.export_closed_searches(ledger))          # control
    RunManifest(run_id=f"tick-settled-{_ts(3)}", crucible_version=_V, gates_hash="519158fa1450",
                cohort_verdicts={"c0ffee000000": "PROMISING"}).write(root / "settled" / "day03")
    result = community.export_closed_searches(ledger)
    assert "settled" not in _by_sub(result)
    assert sorted(s for s, _ in result.skipped) == ["has_hit", "settled"]


def test_a_bad_date_is_refused_where_the_payload_is_built(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        community.export_closed_searches(
            _store(tmp_path / "store"),
            default_description=community.SubstrateDescription(data_start="March 2010"))


def test_run_filter_narrows_a_clean_substrate(tmp_path: Path) -> None:
    ledger = _store(tmp_path / "store")
    one = _by_sub(community.export_closed_searches(ledger, run_ids=[f"tick-dead_end-{_ts(2)}"]))
    assert set(one) == {"dead_end"} and one["dead_end"]["ticks"]["total"] == 1
    with pytest.raises(ValueError, match="not in this ledger"):
        community.export_closed_searches(ledger, substrates=["nope"])


# ------------------------------------------------------------------------------------ one-way glass
def test_export_carries_no_score_and_no_more_than_the_agent_view(tmp_path: Path) -> None:
    ledger = _store(tmp_path / "store")
    result = community.export_closed_searches(ledger, include_promising=True)
    assert _keys(result.payloads) & _FORBIDDEN_KEYS == set()
    with TrialLedger(ledger) as led:
        visible = set(led.agent_view()["semantic_hashes"])
        assert led.agent_view_columns() == ("candidate_hash", "semantic_hash", "stat_hash",
                                            "candidate_type", "family")
    for payload in result.payloads:
        assert set(payload["hypotheses"]["semantic_hashes"]) <= visible
    # per-hypothesis outcomes are never written: hashes appear only in the flat list
    for payload in result.payloads:
        flat = json.dumps({k: v for k, v in payload.items() if k != "hypotheses"})
        assert not any(h in flat for h in payload["hypotheses"]["semantic_hashes"])


def test_export_does_not_touch_the_store(tmp_path: Path) -> None:
    """Opened read-only: no migration, no back-fill, no journal — a certifying store stays as is."""
    root = tmp_path / "store"
    ledger = _store(root)
    before = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    community.export_closed_searches(ledger, include_promising=True)
    community.ledger_overview(ledger)
    after = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    assert after == before


def test_a_legacy_ledger_is_read_without_being_migrated(tmp_path: Path) -> None:
    """A ledger written before the dedup and rejection columns existed: the hash is derived in
    memory and the file keeps its old shape."""
    db = tmp_path / "old" / "trial_ledger.db"
    db.parent.mkdir()
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE trial_ledger (candidate_hash TEXT PRIMARY KEY, crucible_version TEXT "
                "NOT NULL, family TEXT, candidate_type TEXT, spec_json TEXT, formula TEXT, "
                "economic_rationale TEXT, first_seen_run TEXT, proposal_ts TEXT, verdict TEXT, "
                "dsr REAL, delta_sr_oos REAL, marginal_hlz_t REAL, data_snapshot_hash TEXT, "
                "fdr_wealth_charged REAL)")
    con.execute("INSERT INTO trial_ledger VALUES ('h1', 'crucible-v2.6', 'momentum', 'overlay', '{}', "
                "'rank(close)', 'prior', 'tick-old_sub-2026-01-05T00:00:00+00:00', "
                "'2026-01-05T00:00:00+00:00', 'LOGGED', 0.4, 0.1, 2.0, 'snap', NULL)")
    con.commit()
    con.close()
    before = _sha(db)
    payload = _by_sub(community.export_closed_searches(db))["old_sub"]
    assert payload["hypotheses"]["semantic_hashes"] == [semantic_hash("rank(close)")]
    assert payload["trials"]["holdout_tested"] is None
    assert payload["verdict_reasons"] == ["holdout_count_not_recorded", "power_not_measured"]
    assert payload["power"]["economic_floor_source"] == "generation.min_combination_uplift"
    assert _sha(db) == before
    con = sqlite3.connect(str(db))
    assert "semantic_hash" not in {r[1] for r in con.execute("PRAGMA table_info(trial_ledger)")}
    con.close()


def test_nothing_in_the_funnel_reads_the_community_ledger() -> None:
    """Community entries are advice for a person. No gate, trial count, FDR account, killed-family
    list or proposer view may be fed from them — so no funnel module may import the reader."""
    pattern = re.compile(r"\bcommunity\b")
    funnel = [p for d in ("sharpen/crucible/agentic", "sharpen/crucible/orchestrator",
                          "sharpen/crucible/lockbox", "sharpen/crucible/governance", "sharpen/signals")
              for p in (ROOT / d).rglob("*.py")]
    funnel += [ROOT / "sharpen/crucible" / n for n in ("ledger.py", "search_memory.py",
                                                       "corrected_contract.py", "manifest.py")]
    funnel.append(ROOT / "scripts/research/crucible_orchestrator.py")
    assert len(funnel) > 30
    assert [p.relative_to(ROOT).as_posix() for p in funnel
            if pattern.search(p.read_text(encoding="utf-8"))] == []


# ------------------------------------------------------------------------------------ verdict
def _verdict(**kw):
    base = dict(promising=0, holdout_tested=5, holdout_complete=True, decisive=5, underpowered=0,
                unclassified=0, mde_at_test_max=0.08, mde_substrate=0.08, floor=0.10, multiple=1.0)
    return community.decide_verdict(**{**base, **kw})


def test_only_a_complete_recorded_decisive_search_closes() -> None:
    assert _verdict() == (community.CLOSED_DECISIVE, ["all_holdout_rejections_decisive"])
    assert _verdict(mde_at_test_max=0.10)[0] == community.CLOSED_DECISIVE          # at the floor


@pytest.mark.parametrize("kw, reasons", [
    (dict(decisive=3, underpowered=2), ["underpowered_rejections"]),
    (dict(decisive=3, unclassified=2), ["unclassified_rejections"]),
    (dict(holdout_complete=False), ["holdout_count_not_recorded"]),
    (dict(holdout_tested=None, unclassified=None), ["holdout_count_not_recorded"]),
    (dict(holdout_tested=0, decisive=0), ["no_holdout_decision"]),
    (dict(mde_at_test_max=1.4), ["mde_above_floor"]),
    (dict(mde_at_test_max=None), ["power_not_measured"]),
    (dict(mde_at_test_max=None, mde_substrate=None), ["power_not_measured"]),
    (dict(mde_at_test_max=None, mde_substrate=1.7), ["mde_above_floor"]),
    (dict(decisive=0, underpowered=5, mde_at_test_max=1.4, mde_substrate=1.4),
     ["underpowered_rejections", "mde_above_floor"]),
])
def test_anything_short_of_that_stays_open(kw, reasons) -> None:
    """UNDERPOWERED is never presented as a kill: every weaker case reads OPEN, with the reason."""
    assert _verdict(**kw) == (community.OPEN_UNDERPOWERED, reasons)


def test_an_unrecorded_holdout_count_is_unknown_never_zero(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    with TrialLedger(root / "trial_ledger.db") as led:
        _row(led, "old_runs", 2, "rank(close)", prereg=True, verdict="SCORED_NOT_SELECTED")
        _row(led, "old_runs", 3, "rank(volume)", prereg=True, verdict="SCORED_NOT_SELECTED")
    with OrchestratorStore(root / "orchestrator.db") as store:
        _tick(store, root, "old_runs", 2, holdout=None, mde=None, prereg=1)
    none_known = _by_sub(community.export_closed_searches(root / "trial_ledger.db"))["old_runs"]
    assert none_known["trials"]["holdout_tested"] is None
    assert none_known["rejections"]["unclassified"] is None
    assert none_known["ticks"]["screened_unknown"] == 1
    assert none_known["verdict_reasons"] == ["holdout_count_not_recorded", "power_not_measured"]

    with OrchestratorStore(root / "orchestrator.db") as store:
        _tick(store, root, "old_runs", 3, holdout=1, mde=1.5, prereg=1)
    partly = _by_sub(community.export_closed_searches(root / "trial_ledger.db"))["old_runs"]
    assert partly["trials"]["holdout_tested"] == 1                      # a lower bound, flagged
    assert partly["trials"]["holdout_count_complete"] is False
    assert partly["power"]["mde_unmeasured_ticks"] == 1
    assert partly["verdict_reasons"] == ["holdout_count_not_recorded", "unclassified_rejections",
                                         "mde_above_floor"]


def test_reader_never_reads_an_open_search_as_a_kill(tmp_path: Path) -> None:
    result = community.export_closed_searches(_store(tmp_path / "store"))
    text = community.describe_substrate(result.payloads, "dead_end")
    assert "OPEN_UNDERPOWERED" in text and "NOT settled" in text
    assert not re.search(r"\b(closed|decisive|dead|kill|killed|nothing there)\b", text.lower()
                         .replace("open_underpowered", ""))
    settled = community.describe_substrate(result.payloads, "settled")
    assert "CLOSED_DECISIVE" in settled and "new evidence" in settled and "NOT settled" not in settled
    assert "no community record" in community.describe_substrate(result.payloads, "elsewhere")


# ------------------------------------------------------------------------------------ gates
def test_no_gate_bytes_changed(tmp_path: Path) -> None:
    """The community ledger adds no gate and edits none: every gates file still matches its
    registered hash, the frozen funnel hash is intact, and an export leaves the files it reads alone."""
    assert gate_registry.check() == {"changed": [], "unregistered": [], "missing": []}
    assert gates_hash(ROOT / "configs" / "signal_eval.gates.yaml") == "519158fa1450"
    read = (community.DEFAULT_CORRECTED_GATES, community.DEFAULT_FUNNEL_GATES,
            community.DEFAULT_SEARCH_MEMORY_GATES)
    before = [_sha(p) for p in read]
    community.export_closed_searches(_store(tmp_path / "store"))
    assert [_sha(p) for p in read] == before


# ------------------------------------------------------------------------------------ CLI
def test_export_cli(tmp_path: Path, capsys) -> None:
    ledger = _store(tmp_path / "store")
    cli = _script("crucible_export_closed_search")
    out = tmp_path / "community"
    assert cli.main(["--ledger", str(ledger), "--out", str(out), "--list"]) == 0
    listed = capsys.readouterr()
    assert not out.exists()
    assert "has_hit  [holds a PROMISING result - not exported by default]" in listed.out
    assert f"tick-dead_end-{_ts(2)}  (5 trials, 4 scored)" in listed.out

    assert cli.main(["--ledger", str(ledger), "--out", str(out), "--universe", "twelve futures",
                     "--data-source", "vendor daily bars", "--data-start", "2010-01-01",
                     "--data-end", "2026-03-01"]) == 0
    shown = capsys.readouterr()
    assert "held back   has_hit" in shown.err and "has_hit" not in shown.out
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert [f.split("/")[0] for f in files] == ["dead_end", "settled"]
    entry = json.loads((out / files[0]).read_text(encoding="utf-8"))
    assert entry["substrate"] == {"id": "dead_end", "universe": "twelve futures",
                                  "data_sources": ["vendor daily bars"], "data_start": "2010-01-01",
                                  "data_end": "2026-03-01", "search_space": None}
    with pytest.raises(SystemExit):
        cli.main(["--ledger", str(ledger), "--out", str(out), "--data-start", "March 2010"])
    with pytest.raises(SystemExit):
        cli.main(["--ledger", str(ledger), "--out", str(out), "--universe", r"C:\data\panel"])


def test_ledger_cli_validates_and_reports(tmp_path: Path, capsys) -> None:
    out = tmp_path / "community"
    community.write_exports(community.export_closed_searches(_store(tmp_path / "store")).payloads, out)
    cli = _script("crucible_community_ledger")
    assert cli.main(["validate", "--ledger", str(out)]) == 0
    assert "2 valid, 0 invalid" in capsys.readouterr().out
    assert cli.main(["check", "--ledger", str(out), "--substrate", "dead_end"]) == 0
    assert "NOT settled" in capsys.readouterr().out

    victim = next((out / "dead_end").glob("*.json"))
    data = json.loads(victim.read_text(encoding="utf-8"))
    data["verdict"] = community.CLOSED_DECISIVE
    victim.write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(["validate", "--ledger", str(out)]) == 1
    report = capsys.readouterr().out
    assert "INVALID dead_end/" in report and "1 valid, 1 invalid" in report
    # an invalid file is not something a reader should act on
    assert [e["substrate"]["id"] for e in community.load_community_ledger(out)] == ["settled"]


# ------------------------------------------------------------------------------------ end to end
@pytest.mark.slow
def test_export_from_a_real_synthetic_tick(tmp_path: Path) -> None:
    """The whole path on the real machine: one synthetic orchestrator tick writes its own ledger,
    tick log and manifest, and the export reads them as they are."""
    store = tmp_path / "store"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "research" / "crucible_orchestrator.py"),
         "--mode", "synthetic", "--nights", "1", "--t", "500", "--n", "10", "--max-proposals", "6",
         "--max-candidates", "24", "--start-ts", "2026-01-05T00:00:00+00:00", "--out", str(store),
         "--force", "--force-underpowered", "--no-governance"],
        check=True, cwd=ROOT, capture_output=True, timeout=900)
    root = store / "synthetic"
    before = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    result = community.export_closed_searches(root / "trial_ledger.db", include_promising=True)
    (payload,) = result.payloads
    assert payload["substrate"]["id"] == "synthetic"
    assert payload["ticks"]["mined"] == 1 and payload["trials"]["scored"] >= 1
    assert payload["trials"]["holdout_count_complete"] is True
    assert payload["hypotheses"]["count"] == len(payload["hypotheses"]["semantic_hashes"]) >= 1
    assert payload["provenance"]["gates_hashes"] == ["519158fa1450"]
    assert payload["provenance"]["contracts"] == ["corrected"]
    assert len(payload["provenance"]["manifest_hashes"]) == 1
    if payload["trials"]["promising"] == 0:
        assert payload["verdict"] == community.OPEN_UNDERPOWERED       # forced past the power guard
    out = tmp_path / "community"
    community.write_exports(result.payloads, out)
    assert list(community.validate_tree(out).values()) == [[]]
    assert {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*"))
            if p.is_file()} == before
