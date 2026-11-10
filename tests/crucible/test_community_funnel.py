"""Community ledger, ``funnel_closed_search`` kind — schema, verdict rule, privacy and dispatch."""
from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from sharpen.crucible import community

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "community" / "ledger"
# The twelve seeded trial-ledger entries: their ids must never move.
_SEEDS = {
    "cross_asset/2026-07-03-583da7e29138.json", "cross_asset/2026-08-01-215e6da1c963.json",
    "cross_asset/2026-08-04-b3695ef7662b.json", "cross_asset/2026-08-04-ef70cd7517a5.json",
    "intraday/2026-08-02-4d1f7d29f742.json", "intraday_fx/2026-08-04-102b401de303.json",
    "taiwan/2026-07-06-6ea9b96d7736.json", "taiwan/2026-07-12-338b9611c475.json",
    "taiwan/2026-07-13-5ddb98403652.json", "us_equity/2026-08-11-5f1c75cd5b86.json",
    "us_equity/2026-08-11-83945c013668.json", "us_equity/2026-08-11-f58005efc4f5.json",
}
_NAMES = ("secret_alpha_one", "secret_alpha_two", "secret_alpha_three")
_FAMILY = "hidden_family"
_SCORES = (1.2345671, -0.9876543, 3.1415926)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scorecard(verdicts=("LOGGED", "LOGGED", "GATE_FAIL"), *, source="preregistered") -> dict:
    cards = [{"name": n, "family": _FAMILY, "spec_hash": f"{i:x}" * 12, "verdict": v,
              "rank_key": [s], "gross": {"ic_ir": s, "net_sharpe": s},
              "caveats": [f"{n} looked fine"]}
             for i, (n, v, s) in enumerate(zip(_NAMES, verdicts, _SCORES), start=1)]
    return {"batch_name": "private_batch_name", "primary_horizon": 21, "n_trials": 2,
            "n_multiplicity": 50, "multiplicity_source": source,
            "multiplicity_provenance": "docs/private/prereg.md",
            "panel_meta": {"source": "private_vendor", "survivorship_free": False},
            "cards": cards}


def _describe() -> dict:
    return {"substrate": {"id": "toy_panel", "universe": "forty toy stocks",
                          "data_sources": ["vendor daily bars"], "data_start": "2010-01-04",
                          "data_end": "2026-03-01", "search_space": "three toy signals"},
            "campaign": {"preregistered_on": "2026-03-02", "gates_sha256_12": "0123456789ab",
                         "primary_horizon_days": 21},
            "power": {"detection_bar_ic_ir": 0.34, "economic_floor_ic_ir": 0.10,
                      "planted_control_passed": True},
            "sharpen_version": "1.1.2"}


def _export(scorecard: dict | None = None, *, include_promising: bool = False, **power) -> dict:
    d = _describe()
    d["power"].update(power)
    return community.export_funnel_search(
        scorecard or _scorecard(), substrate_id="toy_panel",
        description=community.SubstrateDescription(
            **{k: (tuple(v) if k == "data_sources" else v)
               for k, v in d["substrate"].items() if k != "id"}),
        campaign=community.FunnelCampaign(**d["campaign"]),
        power=community.FunnelPower(**d["power"]),
        sharpen_version=d["sharpen_version"], include_promising=include_promising)


def _write_inputs(tmp: Path, scorecard: dict) -> tuple[Path, Path]:
    sc, ds = tmp / "scorecard.json", tmp / "describe.json"
    sc.write_text(json.dumps(scorecard), encoding="utf-8")
    ds.write_text(json.dumps(_describe()), encoding="utf-8")
    return sc, ds


def _reid(p: dict) -> dict:
    p["export_id"] = community.export_id(p)
    return p


# ------------------------------------------------------------------------------------ schema
def test_published_funnel_schema_is_the_validators_schema() -> None:
    path = ROOT / "docs/schemas/community_funnel_closed_search.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == community.FUNNEL_SCHEMA
    # regenerated the same way as the trial-ledger kind's file
    assert path.read_bytes().replace(b"\r\n", b"\n") == (
        json.dumps(community.FUNNEL_SCHEMA, indent=2) + "\n").encode("utf-8")


def test_funnel_schema_closes_every_object() -> None:
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
            for v in node.values():
                walk(v)
    walk(community.FUNNEL_SCHEMA)


def test_the_seeded_entries_still_validate_and_keep_their_ids() -> None:
    report = community.validate_tree(LEDGER)
    assert _SEEDS <= set(report)
    for rel in _SEEDS:
        entry = json.loads((LEDGER / rel).read_text(encoding="utf-8"))
        assert entry["kind"] == community.KIND and report[rel] == []
        assert rel.endswith(f"-{community.export_id(entry)}.json")
    funnel = [rel for rel in report if rel not in _SEEDS]
    assert funnel and all(
        json.loads((LEDGER / rel).read_text(encoding="utf-8"))["kind"] == community.FUNNEL_KIND
        and report[rel] == [] for rel in funnel)


def test_describe_inputs_are_not_entries() -> None:
    assert (LEDGER / community.DESCRIBE_DIR).is_dir()
    assert not any(rel.startswith(f"{community.DESCRIBE_DIR}/")
                   for rel in community.validate_tree(LEDGER))


# ------------------------------------------------------------------------------------ verdict
def _v(**kw):
    base = dict(promising=0, detection_bar=0.08, floor=0.10, control_passed=True)
    return community.decide_funnel_verdict(**{**base, **kw})


def test_a_powered_search_with_a_recovered_control_closes() -> None:
    assert _v() == (community.CLOSED_DECISIVE, ["no_candidate_passed_at_powered_bar"])
    assert _v(detection_bar=0.10)[0] == community.CLOSED_DECISIVE              # at the floor


@pytest.mark.parametrize("kw, expected", [
    (dict(promising=1), (community.PROMISING_FOUND, ["promising_result"])),
    (dict(promising=1, detection_bar=None), (community.PROMISING_FOUND, ["promising_result"])),
    (dict(detection_bar=0.34), (community.OPEN_UNDERPOWERED, ["mde_above_floor"])),
    (dict(detection_bar=None), (community.OPEN_UNDERPOWERED, ["power_not_measured"])),
    (dict(floor=None), (community.OPEN_UNDERPOWERED, ["power_not_measured"])),
    (dict(control_passed=False), (community.OPEN_UNDERPOWERED, ["planted_control_failed"])),
    (dict(control_passed=None), (community.OPEN_UNDERPOWERED, ["power_not_measured"])),
    (dict(detection_bar=0.34, control_passed=False),
     (community.OPEN_UNDERPOWERED, ["mde_above_floor", "planted_control_failed"])),
    (dict(detection_bar=0.34, control_passed=None),
     (community.OPEN_UNDERPOWERED, ["mde_above_floor", "power_not_measured"])),
    (dict(floor=None, control_passed=False),
     (community.OPEN_UNDERPOWERED, ["planted_control_failed", "power_not_measured"])),
], ids=["promising", "promising-unmeasured", "bar-above-floor", "no-bar", "no-floor",
        "control-failed", "no-control", "above-and-failed", "above-and-no-control",
        "failed-and-no-floor"])
def test_anything_short_of_that_stays_open(kw, expected) -> None:
    assert _v(**kw) == expected


# ------------------------------------------------------------------------------------ export
def test_export_counts_and_round_trip(tmp_path: Path) -> None:
    p = _export()
    assert p["kind"] == community.FUNNEL_KIND
    assert p["trials"] == {"preregistered": 3, "scored": 2, "promising": 0}
    assert p["campaign"] == {"preregistered_on": "2026-03-02", "primary_horizon_days": 21,
                             "declared_multiplicity": 50, "gates_sha256_12": "0123456789ab"}
    assert p["hypotheses"]["spec_hashes"] == sorted(f"{i:x}" * 12 for i in (1, 2, 3))
    assert (p["verdict"], p["verdict_reasons"]) == (community.OPEN_UNDERPOWERED,
                                                    ["mde_above_floor"])
    (path,) = community.write_exports([p], tmp_path)
    assert path.relative_to(tmp_path).as_posix() == f"toy_panel/2026-03-02-{p['export_id']}.json"
    assert community.validate_tree(tmp_path) == {community.relative_path(p): []}
    assert _export() == p                                                       # deterministic


def test_an_edited_file_fails_its_id(tmp_path: Path) -> None:
    p = _export()
    bad = copy.deepcopy(p)
    bad["power"]["detection_bar_ic_ir"] = 0.05
    bad["verdict"], bad["verdict_reasons"] = (community.CLOSED_DECISIVE,
                                              ["no_candidate_passed_at_powered_bar"])
    assert community.validate_payload(bad, relpath=community.relative_path(p)) == [
        "export_id does not match the content (the file was edited after export; re-run the "
        "export instead)"]


@pytest.mark.parametrize("mutate, expected", [
    (lambda p: p.update(dsr=0.97),
     {"schema: <root>: Additional properties are not allowed ('dsr' was unexpected)"}),
    (lambda p: p["hypotheses"].update(names=["x"]),
     {"schema: hypotheses: Additional properties are not allowed ('names' was unexpected)"}),
    (lambda p: p["trials"].update(promising=1),
     {"verdict OPEN_UNDERPOWERED does not follow from the counts and power (they imply "
      "PROMISING_FOUND)"}),
    (lambda p: p["hypotheses"]["spec_hashes"].reverse(), {"hypotheses.spec_hashes is not sorted"}),
    (lambda p: p["trials"].update(scored=5),
     {"trials: promising <= scored <= preregistered does not hold"}),
    (lambda p: p["campaign"].update(declared_multiplicity=1),
     {"campaign.declared_multiplicity is smaller than the number of specs scored"}),
    (lambda p: p.update(verdict_reasons=["power_not_measured"]),
     {"verdict_reasons should be ['mde_above_floor']"}),
    (lambda p: p["substrate"].update(universe=r"built from D:\research\panel.parquet"),
     {"contains a drive path: 'built from D:\\\\research\\\\panel.parquet'"}),
], ids=["score-field", "name-field", "verdict", "unsorted", "counts", "multiplicity", "reasons",
        "local-path"])
def test_malformed_funnel_entry_fails(mutate, expected) -> None:
    good = _export()
    bad = copy.deepcopy(good)
    mutate(bad)
    _reid(bad)
    assert set(community.validate_payload(bad, relpath=community.relative_path(bad))) == expected


def test_validate_dispatches_on_kind(tmp_path: Path) -> None:
    good = _export()
    assert community.validate_payload(good, relpath=community.relative_path(good)) == []
    # a funnel record dressed as the other kind is checked against THAT schema, and fails
    as_other = _reid({**copy.deepcopy(good), "kind": community.KIND})
    assert any(e.startswith("schema:") for e in community.validate_payload(as_other))
    unknown = _reid({**copy.deepcopy(good), "kind": "mystery_search"})
    assert community.validate_payload(unknown) == [
        "kind: unknown record kind 'mystery_search' (expected one of "
        "['crucible_closed_search', 'funnel_closed_search'])"]
    path = tmp_path / "toy_panel" / f"2026-03-02-{unknown['export_id']}.json"
    path.parent.mkdir(parents=True)
    path.write_text(community.canonical_json(unknown), encoding="utf-8")
    cli = _script("crucible_community_ledger")
    assert cli.main(["validate", "--ledger", str(tmp_path)]) == 1
    assert community.load_community_ledger(tmp_path) == []


def test_check_lists_both_kinds(capsys) -> None:
    seeded = json.loads((LEDGER / "us_equity/2026-08-11-5f1c75cd5b86.json").read_text("utf-8"))
    funnel = _reid({**_export(), "substrate": {**_export()["substrate"], "id": "us_equity"}})
    text = community.describe_substrate([seeded, funnel], "us_equity")
    assert "2 shared search(es)" in text
    assert "hypotheses scored" in text and "signal-funnel batch, 2 of 3" in text
    assert "detection bar IC-IR 0.34 against a floor of 0.10" in text
    cli = _script("crucible_community_ledger")
    assert cli.main(["check", "--substrate", "us_largecap_sp500_top300"]) == 0
    assert "signal-funnel batch" in capsys.readouterr().out


# ------------------------------------------------------------------------------------ privacy
def test_a_promising_scorecard_is_refused_and_nothing_written(tmp_path: Path, capsys) -> None:
    hit = _scorecard(("LOGGED", "PROMISING", "LOGGED"))
    with pytest.raises(community.PromisingHeldBack):
        _export(hit)
    sc, ds = _write_inputs(tmp_path, hit)
    out = tmp_path / "ledger"
    cli = _script("funnel_export_closed_search")
    assert cli.main(["--scorecard", str(sc), "--describe", str(ds), "--out", str(out)]) == 1
    assert not out.exists()
    assert "secret_alpha" not in capsys.readouterr().out
    assert cli.main(["--scorecard", str(sc), "--describe", str(ds), "--out", str(out),
                     "--include-promising"]) == 0
    (written,) = out.rglob("*.json")
    entry = json.loads(written.read_text(encoding="utf-8"))
    assert (entry["verdict"], entry["trials"]["promising"]) == (community.PROMISING_FOUND, 1)


def test_a_batch_shaped_multiplicity_is_refused() -> None:
    with pytest.raises(ValueError, match="declared"):
        _export(_scorecard(source="batch"))


def test_a_horizon_that_disagrees_with_the_scorecard_is_refused() -> None:
    sc = _scorecard()
    sc["primary_horizon"] = 5
    with pytest.raises(ValueError, match="primary horizon"):
        _export(sc)


def test_no_name_score_or_card_verdict_reaches_the_file(tmp_path: Path) -> None:
    sc, ds = _write_inputs(tmp_path, _scorecard())
    out = tmp_path / "ledger"
    cli = _script("funnel_export_closed_search")
    assert cli.main(["--scorecard", str(sc), "--describe", str(ds), "--out", str(out)]) == 0
    (written,) = out.rglob("*.json")
    raw = written.read_bytes()
    for leaked in (*_NAMES, _FAMILY, "private_batch_name", "private_vendor", "docs/private",
                   "looked fine", "LOGGED", "GATE_FAIL", "rank_key", "net_sharpe"):
        assert leaked.encode() not in raw, leaked
    for score in _SCORES:
        assert str(score).encode() not in raw and f"{score:.2f}".encode() not in raw
    # the only numbers are the counts, the dates, the hashes, the version and the author's power
    assert set(re.findall(rb"-?\d+\.\d+", raw)) == {b"0.34", b"0.1", b"1.1"}

