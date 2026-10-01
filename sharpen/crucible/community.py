"""Community ledger of closed searches — export, validate, read.

A Crucible mining run that finds nothing is still a result, and the next person should not have to
pay for it again. This module turns a local trial ledger into one small JSON file per substrate that
says what was searched and — the part that matters — whether the search was actually finished:

  * ``CLOSED_DECISIVE``   every hypothesis that reached the out-of-sample test was rejected, and the
                          test could have seen an edge of the smallest size the contract accepts.
  * ``OPEN_UNDERPOWERED`` the search could not settle the question: the tests were too weak to see
                          such an edge, or nothing reached the out-of-sample test at all. This is NOT
                          evidence that nothing is there.
  * ``PROMISING_FOUND``   something passed. Never written unless the exporter opts in.

WHAT LEAVES THE MACHINE. Counts, hashes and the power of the test — nothing else. The hypothesis
list is exactly the dedup keys ``ledger_agent_view`` already exposes (``semantic_hash`` of scored
candidates); no formula, score, p-value, per-hypothesis verdict or holdout value is read into a
payload, and :data:`SCHEMA` rejects any field it does not name. The trial ledger and the tick log
are opened read-only, so an export cannot migrate or alter a store.

WHAT NEVER READS IT. Nothing in the funnel does. Community entries feed no gate, no trial count, no
FDR account, no ``killed_families()`` and no proposer view; :func:`describe_substrate` only renders
text for a human deciding where to spend compute.

PROMISING RESULTS STAY PRIVATE BY DEFAULT. A substrate that holds a PROMISING row — or shares a
hypothesis with one elsewhere in the ledger — is skipped as a whole unless ``include_promising`` is
set. It is all-or-nothing on purpose: withholding one hash would leave a visible gap in the counts,
and exporting only a substrate's dead-end runs would publish a "nothing here" that its author knows
to be false.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .ledger import PROMISING_VERDICTS
from .search_memory import REJECTION_DECISIVE, REJECTION_UNDERPOWERED, semantic_hash
from .version import gates_hash

log = logging.getLogger("crucible.community")

SCHEMA_VERSION = 1
KIND = "crucible_closed_search"

CLOSED_DECISIVE = "CLOSED_DECISIVE"
OPEN_UNDERPOWERED = "OPEN_UNDERPOWERED"
PROMISING_FOUND = "PROMISING_FOUND"
VERDICTS = (CLOSED_DECISIVE, OPEN_UNDERPOWERED, PROMISING_FOUND)

# Why a verdict was reached, in the fixed order they are written.
REASONS = (
    "promising_result",                 # PROMISING_FOUND
    "all_holdout_rejections_decisive",  # CLOSED_DECISIVE
    "holdout_count_not_recorded",       # the version that ran did not record how many reached the test
    "no_holdout_decision",              # nothing reached the out-of-sample test
    "underpowered_rejections",          # rejected by a test too weak to see the floor
    "unclassified_rejections",          # rejected, but no power was recorded for the test
    "mde_above_floor",                  # smallest detectable edge was larger than the floor
    "power_not_measured",               # no power stamp for the test
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECTED_GATES = _REPO_ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
DEFAULT_FUNNEL_GATES = _REPO_ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_SEARCH_MEMORY_GATES = _REPO_ROOT / "configs" / "crucible_search_memory.gates.yaml"
DEFAULT_LEDGER_DIR = _REPO_ROOT / "community" / "ledger"

_TICK_RUN = re.compile(r"^tick-(?P<sub>.+)-(?P<ts>\d{4}-\d{2}-\d{2}T.+)$")
_LOOP_RUN = re.compile(r"^hyp-(?P<sub>.+)-[0-9a-f]{12}$")
_SUBSTRATE_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
_VERSION = re.compile(r"^crucible-v(\d+)\.(\d+)")
# Strings a public file must not carry: local paths, addresses, hosts.
_PRIVATE_TEXT = (
    (re.compile(r"[A-Za-z]:[\\/]"), "a drive path"),
    (re.compile(r"\\\\[A-Za-z0-9_.$-]+\\"), "a network path"),
    (re.compile(r"(?:^|[\s\"'(=])/(?:home|Users|root|mnt|tmp|var|opt|etc|srv|data)/"), "a local path"),
    (re.compile(r"(?:^|[\s\"'(=])~[\\/]"), "a home-directory path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "an e-mail address"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "an IP address"),
)
_MAX_TEXT = 300


# --------------------------------------------------------------------------------------- schema
def _nullable(schema: dict) -> dict:
    return {**schema, "type": [schema["type"], "null"]}


_HASH12 = {"type": "string", "pattern": "^[0-9a-f]{12}$"}
_COUNT = {"type": "integer", "minimum": 0}
_DATE = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
_TEXT = {"type": "string", "minLength": 1, "maxLength": _MAX_TEXT}
_NUMBER = {"type": "number", "minimum": 0}
_TOKEN = {"type": "string", "pattern": r"^[A-Za-z0-9_.:+\[\]-]{1,120}$"}


def _hash_list(item: dict) -> dict:
    return {"type": "array", "items": item, "uniqueItems": True}


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": sorted(properties),
            "properties": properties}


#: The file format. ``docs/schemas/community_closed_search.schema.json`` is a JSON dump of this dict
#: (a test keeps them equal), so the published schema cannot drift from the validator.
SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:sharpen:crucible:community-closed-search:v1",
    "title": "Crucible community ledger: one closed search",
    "description": ("One substrate's mining record: what was searched, how many hypotheses reached "
                    "the out-of-sample test, how strong that test was, and whether the question is "
                    "closed. Counts and hashes only; no formula, score or per-hypothesis verdict."),
    **_object({
        "schema_version": {"const": SCHEMA_VERSION},
        "kind": {"const": KIND},
        "export_id": _HASH12,
        "verdict": {"enum": list(VERDICTS)},
        "verdict_reasons": {"type": "array", "minItems": 1, "uniqueItems": True,
                            "items": {"enum": list(REASONS)}},
        "substrate": _object({
            "id": {"type": "string", "pattern": _SUBSTRATE_ID},
            "universe": _nullable(_TEXT),
            "data_sources": {"type": "array", "items": _TEXT, "maxItems": 20, "uniqueItems": True},
            "data_start": _nullable(_DATE),
            "data_end": _nullable(_DATE),
            "search_space": _nullable(_TEXT),
        }),
        "trials": _object({
            "recorded": _COUNT,
            "preregistered": _COUNT,
            "scored": _COUNT,
            "holdout_tested": _nullable(_COUNT),
            "holdout_count_complete": {"type": "boolean"},
            "promising": _COUNT,
        }),
        "rejections": _object({
            "decisive": _COUNT,
            "underpowered": _COUNT,
            "unclassified": _nullable(_COUNT),
        }),
        "power": _object({
            "mde_at_test_min": _nullable(_NUMBER),
            "mde_at_test_max": _nullable(_NUMBER),
            "mde_substrate": _nullable(_NUMBER),
            "mde_unmeasured_ticks": _COUNT,
            "mde_source": {"enum": ["ledger", "tick_log", "none"]},
            "economic_floor": {"type": "number", "exclusiveMinimum": 0},
            "economic_floor_source": {"enum": ["guards.uplift_min",
                                               "generation.min_combination_uplift"]},
            "economic_floor_gates_hash": _HASH12,
            "decisive_mde_multiple": {"type": "number", "exclusiveMinimum": 0},
        }),
        "ticks": _nullable(_object({k: _COUNT for k in (
            "total", "mined", "tested", "screened_only", "screened_unknown", "no_score",
            "cohort_only", "underpowered_skips", "idle", "errors")})),
        "provenance": _object({
            "source": {"enum": ["trial_ledger"]},
            "runs": _COUNT,
            "first_date": _nullable(_DATE),
            "last_date": _nullable(_DATE),
            "crucible_versions": _hash_list({"type": "string",
                                             "pattern": r"^crucible-v[0-9]+\.[0-9]+(\.[0-9]+)?$"}),
            "contracts": _hash_list({"enum": ["corrected", "shipped"]}),
            "proposers": _hash_list(_TOKEN),
            "gates_hashes": _hash_list(_HASH12),
            "corrected_gates_hashes": _hash_list(_HASH12),
            "search_memory_gates_hashes": _hash_list(_HASH12),
            "cohort_gates_hashes": _hash_list(_HASH12),
            "data_snapshot_hashes": _hash_list(_TOKEN),
            "manifest_hashes": _hash_list(_HASH12),
        }),
        "hypotheses": _object({
            "count": _COUNT,
            "by_type": _object({"cross_sectional": _COUNT, "overlay": _COUNT, "unknown": _COUNT}),
            "semantic_hashes": _hash_list(_HASH12),
        }),
    }),
}


# --------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True, slots=True)
class SubstrateDescription:
    """What the exporter says about the data, in their own words. None of it is in the ledger."""

    universe: str | None = None
    data_sources: tuple[str, ...] = ()
    data_start: str | None = None
    data_end: str | None = None
    search_space: str | None = None


@dataclass(frozen=True, slots=True)
class ExportResult:
    """``payloads`` are safe to publish. ``skipped`` is LOCAL information for the person running the
    export — ``(substrate_id, reason)`` — and is never written to any file."""

    payloads: list[dict] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Row:
    substrate: str
    run: str
    version: str
    preregistered: bool
    verdict: str | None
    rejection: str | None
    mde: float | None
    semantic: str | None
    candidate_type: str | None
    snapshot: str | None
    proposal_ts: str | None


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only handle. ``TrialLedger(path)`` would MIGRATE the file (new columns, back-fills, a
    recreated view); an export must leave a certifying store byte-identical."""
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (table,)).fetchone() is not None


def _substrate_of(run: str | None) -> str | None:
    """Run ids are ``tick-<substrate>-<iso ts>`` (orchestrator) or ``hyp-<panel>-<gates hash>``
    (manual loop); the ledger has no substrate column."""
    for pattern in (_TICK_RUN, _LOOP_RUN):
        m = pattern.match(run or "")
        if m:
            return m.group("sub")
    return None


def _finite(x: object) -> float | None:
    try:
        v = float(x)                                                    # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _day(ts: str | None) -> str | None:
    return ts[:10] if ts and re.match(r"^\d{4}-\d{2}-\d{2}", ts) else None


def _version_tuple(version: str | None) -> tuple[int, int]:
    m = _VERSION.match(version or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _read_rows(ledger_db: Path) -> list[_Row]:
    con = _connect_ro(ledger_db)
    try:
        if not _has_table(con, "trial_ledger"):
            raise ValueError(f"{ledger_db.name} has no trial_ledger table")
        have = _columns(con, "trial_ledger")

        def col(name: str) -> str:
            return name if name in have else f"NULL AS {name}"

        # Score columns (dsr, delta_sr_oos, marginal_hlz_t), spec_json and the rationale are never
        # selected. ``formula`` is read only to derive the dedup key for a row written before the
        # ledger stored one, and is dropped on the next line.
        rows = con.execute(
            f"SELECT first_seen_run, crucible_version, spec_json IS NOT NULL AS prereg, verdict, "
            f"{col('rejection_class')}, {col('implied_mde_at_test')}, {col('semantic_hash')}, "
            f"{col('candidate_type')}, {col('data_snapshot_hash')}, proposal_ts, "
            f"CASE WHEN {'semantic_hash' if 'semantic_hash' in have else 'NULL'} IS NULL "
            f"THEN formula END AS formula_for_key FROM trial_ledger").fetchall()
    finally:
        con.close()
    out: list[_Row] = []
    for r in rows:
        sub = _substrate_of(r["first_seen_run"])
        if sub is None:
            continue
        sem = r["semantic_hash"]
        if sem is None and r["formula_for_key"]:
            try:
                sem = semantic_hash(r["formula_for_key"])
            except Exception:                                 # noqa: BLE001 — unparseable legacy row
                sem = None
        out.append(_Row(substrate=sub, run=str(r["first_seen_run"]),
                        version=str(r["crucible_version"] or ""),
                        preregistered=bool(r["prereg"]), verdict=r["verdict"],
                        rejection=r["rejection_class"], mde=_finite(r["implied_mde_at_test"]),
                        semantic=sem, candidate_type=r["candidate_type"],
                        snapshot=r["data_snapshot_hash"],
                        proposal_ts=r["proposal_ts"]))
    return out


def _read_ticks(orchestrator_db: Path | None) -> list[dict]:
    if orchestrator_db is None or not orchestrator_db.exists():
        return []
    con = _connect_ro(orchestrator_db)
    try:
        if not _has_table(con, "ticks"):
            return []
        have = _columns(con, "ticks")
        wanted = ("tick_ts", "substrate_id", "mined", "reason", "snapshot_hash", "n_scored",
                  "n_holdout_tested", "n_promising", "fdr_charged_total", "manifest_hash", "status",
                  "implied_mde_delta_sr", "power_interp_mode")
        sel = ", ".join(c if c in have else f"NULL AS {c}" for c in wanted)
        return [dict(r) for r in con.execute(f"SELECT {sel} FROM ticks ORDER BY id")]
    finally:
        con.close()


def _read_manifests(manifests_dir: Path | None) -> dict[str, dict]:
    """``run_id -> provenance`` from every ``run_manifest.json`` under the store. The manifest's
    per-candidate and cohort verdicts are read for ONE bit — does this run hold a PROMISING entry —
    and are never copied into a payload."""
    out: dict[str, dict] = {}
    if manifests_dir is None or not manifests_dir.is_dir():
        return out
    keep = ("crucible_version", "gates_hash", "corrected_gates_hash", "search_memory_gates_hash",
            "cohort_gates_hash", "data_snapshot_hash", "contract", "agent_model_id")
    for path in sorted(manifests_dir.rglob("run_manifest.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict) and raw.get("run_id"):
            verdicts = [*(raw.get("verdicts") or {}).values(),
                        *(raw.get("cohort_verdicts") or {}).values()]
            out[str(raw["run_id"])] = {
                **{k: raw.get(k) for k in keep},
                "promising": any(v in PROMISING_VERDICTS for v in verdicts)}
    return out


def tick_outcome(tick: dict) -> str:
    """What a tick actually did. Same classes as ``scripts/research/crucible_mining_log.py``: a tick
    that mined but never reached the out-of-sample test proves nothing, and one whose count was not
    recorded is unknown — never zero."""
    if (tick.get("status") or "").upper() == "ERROR":
        return "errors"
    if (tick.get("reason") or "").upper().startswith("UNDERPOWERED"):
        return "underpowered_skips"
    if not tick.get("mined"):
        return "cohort_only" if (tick.get("fdr_charged_total") or 0.0) > 0.0 else "idle"
    if not tick.get("n_scored"):
        return "no_score"
    n = tick.get("n_holdout_tested")
    if n is None:
        return "screened_unknown"
    return "tested" if n > 0 else "screened_only"


def _load_floors(corrected: Path, funnel: Path, search_memory: Path) -> dict:
    """The smallest edge each decision contract accepts, and the decisiveness multiple — read from
    the gates files, never restated here."""
    def read(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"gates file not found: {path.name} (pass its path explicitly)")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return {
        "corrected": float(read(corrected)["guards"]["uplift_min"]),
        "corrected_hash": gates_hash(corrected),
        "shipped": float(read(funnel)["generation"]["min_combination_uplift"]),
        "shipped_hash": gates_hash(funnel),
        "multiple": float((read(search_memory).get("decisiveness") or {})
                          .get("decisive_mde_multiple", 1.0)),
    }


def _public_date(value: str | None, what: str) -> str | None:
    if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{what} must be a YYYY-MM-DD date, got {value!r}")
    return value


def public_text(value: str | None, what: str) -> str | None:
    """A free-text description, checked for things a public file must not carry."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > _MAX_TEXT:
        raise ValueError(f"{what} is longer than {_MAX_TEXT} characters")
    for pattern, label in _PRIVATE_TEXT:
        if pattern.search(text):
            raise ValueError(f"{what} looks like it contains {label}; describe the data by name "
                             "instead of pasting where it lives")
    return text


# --------------------------------------------------------------------------------------- verdict
def decide_verdict(*, promising: int, holdout_tested: int | None, holdout_complete: bool,
                   decisive: int, underpowered: int, unclassified: int | None,
                   mde_at_test_max: float | None, mde_substrate: float | None,
                   floor: float, multiple: float) -> tuple[str, list[str]]:
    """The top-level reading of one search, from its own counts.

    ``CLOSED_DECISIVE`` needs every out-of-sample rejection to carry a DECISIVE class recorded by the
    engine at test time, a complete count of what was tested, and a power stamp AT TEST at or below
    the floor. Anything short of that is ``OPEN_UNDERPOWERED``: an export can leave a question open,
    it can never close one."""
    if promising > 0:
        return PROMISING_FOUND, ["promising_result"]
    limit = multiple * floor + 1e-12
    if (holdout_tested and holdout_complete and decisive >= 1 and underpowered == 0
            and unclassified == 0 and mde_at_test_max is not None and mde_at_test_max <= limit):
        return CLOSED_DECISIVE, ["all_holdout_rejections_decisive"]
    reasons: set[str] = set()
    if holdout_tested is None or not holdout_complete:
        reasons.add("holdout_count_not_recorded")
    elif holdout_tested == 0:
        reasons.add("no_holdout_decision")
    if underpowered > 0:
        reasons.add("underpowered_rejections")
    if unclassified:
        reasons.add("unclassified_rejections")
    best = mde_at_test_max if mde_at_test_max is not None else mde_substrate
    if best is not None and best > limit:
        reasons.add("mde_above_floor")
    if best is None or not reasons:
        reasons.add("power_not_measured")
    return OPEN_UNDERPOWERED, [r for r in REASONS if r in reasons]


# --------------------------------------------------------------------------------------- export
def canonical_json(payload: dict) -> str:
    """The one serialization: sorted keys, two-space indent, ASCII, trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def export_id(payload: dict) -> str:
    """12-hex SHA-256 of the payload without its own id — the file's identity and its integrity
    check (an edited file no longer matches its name)."""
    body = {k: v for k, v in payload.items() if k != "export_id"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:12]


def relative_path(payload: dict) -> str:
    """``<substrate>/<last date>-<export id>.json`` — the path under ``community/ledger/``."""
    date = payload["provenance"]["last_date"] or "undated"
    return f"{payload['substrate']['id']}/{date}-{payload['export_id']}.json"


def export_closed_searches(
    ledger_db: str | Path,
    *,
    orchestrator_db: str | Path | None = None,
    manifests_dir: str | Path | None = None,
    substrates: list[str] | None = None,
    run_ids: list[str] | None = None,
    include_promising: bool = False,
    descriptions: dict[str, SubstrateDescription] | None = None,
    default_description: SubstrateDescription | None = None,
    corrected_gates: str | Path = DEFAULT_CORRECTED_GATES,
    funnel_gates: str | Path = DEFAULT_FUNNEL_GATES,
    search_memory_gates: str | Path = DEFAULT_SEARCH_MEMORY_GATES,
) -> ExportResult:
    """One payload per substrate in ``ledger_db`` (optionally narrowed to ``substrates`` and to the
    runs in ``run_ids``). ``orchestrator_db`` and ``manifests_dir`` default to the ledger's own
    directory, where the orchestrator writes them. ``descriptions`` is keyed by substrate id;
    ``default_description`` covers the substrates it does not name."""
    ledger_db = Path(ledger_db)
    store = ledger_db.parent
    orchestrator_db = Path(orchestrator_db) if orchestrator_db else store / "orchestrator.db"
    manifests_dir = Path(manifests_dir) if manifests_dir else store
    floors = _load_floors(Path(corrected_gates), Path(funnel_gates), Path(search_memory_gates))
    rows = _read_rows(ledger_db)
    ticks = [t for t in _read_ticks(orchestrator_db)
             if re.match(_SUBSTRATE_ID, t["substrate_id"] or "")]
    manifests = _read_manifests(manifests_dir)
    descriptions = descriptions or {}
    wanted_runs = set(run_ids) if run_ids else None

    promising_keys = {r.semantic for r in rows if r.verdict in PROMISING_VERDICTS and r.semantic}
    found = sorted({r.substrate for r in rows} | {t["substrate_id"] for t in ticks})
    unknown = sorted(set(substrates or ()) - set(found))
    if unknown:
        raise ValueError(f"substrate(s) not in this ledger: {', '.join(unknown)}")

    result = ExportResult()
    for sub in found:
        if substrates and sub not in substrates:
            continue
        if not re.match(_SUBSTRATE_ID, sub):
            result.skipped.append((sub, "substrate id is not a plain name"))
            continue
        sub_rows = [r for r in rows if r.substrate == sub]
        sub_ticks = [t for t in ticks if t["substrate_id"] == sub]
        # The PROMISING check reads the WHOLE substrate, before any run filter (module docstring).
        holds_promising = (any(r.verdict in PROMISING_VERDICTS for r in sub_rows)
                           or any((t.get("n_promising") or 0) > 0 for t in sub_ticks)
                           or any(m["promising"] for run, m in manifests.items()
                                  if _substrate_of(run) == sub))
        shares_promising = any(r.semantic in promising_keys for r in sub_rows
                               if r.verdict is not None)
        if (holds_promising or shares_promising) and not include_promising:
            result.skipped.append((sub, "holds a PROMISING result, or a hypothesis that is PROMISING "
                                        "elsewhere in this ledger (kept private)"))
            continue
        if wanted_runs is not None:
            scoped_rows = [r for r in sub_rows if r.run in wanted_runs]
            scoped_ticks = [t for t in sub_ticks
                            if f"tick-{sub}-{t['tick_ts']}" in wanted_runs]
            dropped_promising = (
                sum(r.verdict in PROMISING_VERDICTS for r in sub_rows)
                > sum(r.verdict in PROMISING_VERDICTS for r in scoped_rows))
            if dropped_promising:
                result.skipped.append((sub, "the selected runs leave out a PROMISING result on the "
                                            "same substrate; export the whole substrate or none"))
                continue
            sub_rows, sub_ticks = scoped_rows, scoped_ticks
        if not sub_rows and not sub_ticks:
            continue
        result.payloads.append(_build_payload(
            sub, sub_rows, sub_ticks, manifests, floors,
            descriptions.get(sub) or default_description or SubstrateDescription()))
    return result


def ledger_overview(ledger_db: str | Path, *,
                    orchestrator_db: str | Path | None = None) -> list[dict]:
    """What a ledger holds, for the person choosing what to share: per substrate, its runs with
    trial counts and whether it holds a PROMISING result. LOCAL display only — never written."""
    ledger_db = Path(ledger_db)
    orchestrator_db = (Path(orchestrator_db) if orchestrator_db
                       else ledger_db.parent / "orchestrator.db")
    rows = _read_rows(ledger_db)
    ticks = _read_ticks(orchestrator_db)
    out = []
    for sub in sorted({r.substrate for r in rows}
                      | {t["substrate_id"] for t in ticks if t["substrate_id"]}):
        sub_rows = [r for r in rows if r.substrate == sub]
        mined = {f"tick-{sub}-{t['tick_ts']}" for t in ticks
                 if t["substrate_id"] == sub and t.get("mined")}
        runs = sorted({r.run for r in sub_rows} | mined)
        out.append({
            "substrate": sub,
            "holds_promising": (any(r.verdict in PROMISING_VERDICTS for r in sub_rows)
                                or any((t.get("n_promising") or 0) > 0 for t in ticks
                                       if t["substrate_id"] == sub)),
            "runs": [{"run_id": run, "trials": sum(r.run == run for r in sub_rows),
                      "scored": sum(r.run == run and r.verdict is not None for r in sub_rows)}
                     for run in runs],
        })
    return out


def _build_payload(sub: str, rows: list[_Row], ticks: list[dict], manifests: dict[str, dict],
                   floors: dict, desc: SubstrateDescription) -> dict:
    scored = [r for r in rows if r.verdict is not None]
    promising = sum(r.verdict in PROMISING_VERDICTS for r in rows)
    decisive = sum(r.rejection == REJECTION_DECISIVE for r in rows)
    underpowered = sum(r.rejection == REJECTION_UNDERPOWERED for r in rows)

    # How many hypotheses the out-of-sample test decided. The tick log records it (v12.0+); a ledger
    # written entirely by v15.0+ states it per row. A run that did not record it is UNKNOWN, never
    # zero: the count is then a lower bound and ``holdout_count_complete`` is false.
    mined = [t for t in ticks if t.get("mined")]
    known = [int(t["n_holdout_tested"]) for t in mined if t.get("n_holdout_tested") is not None]
    holdout_tested: int | None
    if mined:
        holdout_tested, holdout_complete = (sum(known) if known else None), len(known) == len(mined)
    elif ticks:
        holdout_tested, holdout_complete = 0, True
    elif rows and all(_version_tuple(r.version) >= (15, 0) for r in rows):
        holdout_tested = sum(
            1 for r in rows if r.preregistered and (
                r.verdict in PROMISING_VERDICTS or r.verdict == "SCORED_NOT_SELECTED"
                or r.rejection is not None))
        holdout_complete = True
    else:
        holdout_tested, holdout_complete = None, False
    unclassified = (None if holdout_tested is None
                    else max(0, holdout_tested - promising - decisive - underpowered))

    # Power AT TEST: the stamp on the rows the engine classified, else on the ticks that mined. A
    # stamp taken on a tick that did not mine (a later refusal, say) describes the substrate, not the
    # test, and is reported separately.
    row_mdes = [r.mde for r in rows if r.mde is not None]
    mined_mdes = [m for m in (_finite(t.get("implied_mde_delta_sr")) for t in mined)
                  if m is not None]
    mdes, mde_source = ((row_mdes, "ledger") if row_mdes
                        else (mined_mdes, "tick_log") if mined_mdes else ([], "none"))
    mde_min = round(min(mdes), 6) if mdes else None
    mde_max = round(max(mdes), 6) if mdes else None
    stamps = [m for m in (_finite(t.get("implied_mde_delta_sr")) for t in ticks) if m is not None]
    mde_substrate = round(stamps[-1], 6) if stamps else None      # ticks are in time order
    unmeasured = sum(1 for t in mined if _finite(t.get("implied_mde_delta_sr")) is None)

    tick_runs = {f"tick-{sub}-{t['tick_ts']}" for t in ticks}
    runs = sorted({r.run for r in rows} | {f"tick-{sub}-{t['tick_ts']}" for t in mined})
    mans = [manifests[r] for r in sorted({r.run for r in rows} | tick_runs) if r in manifests]
    contracts = sorted({str(m["contract"] or "shipped") for m in mans})
    if not contracts:                       # no manifests: the corrected contract arrived in v6.0
        contracts = (["corrected"] if any(_version_tuple(r.version) >= (6, 0) for r in rows)
                     else ["shipped"])
    floor_key = "shipped" if contracts == ["shipped"] else "corrected"
    floor = floors[floor_key]

    verdict, reasons = decide_verdict(
        promising=promising, holdout_tested=holdout_tested, holdout_complete=holdout_complete,
        decisive=decisive, underpowered=underpowered, unclassified=unclassified,
        mde_at_test_max=mde_max, mde_substrate=mde_substrate,
        floor=floor, multiple=floors["multiple"])

    outcomes = [tick_outcome(t) for t in ticks]
    tick_block = None if not ticks else {
        "total": len(ticks), "mined": len(mined),
        **{k: outcomes.count(k) for k in ("tested", "screened_only", "screened_unknown", "no_score",
                                          "cohort_only", "underpowered_skips", "idle", "errors")}}
    dates = sorted(d for d in ([_day(t["tick_ts"]) for t in ticks]
                               + [_day(r.proposal_ts) for r in rows]) if d)

    def distinct(values) -> list[str]:
        return sorted({str(v) for v in values if v})

    hashes = sorted({r.semantic for r in scored if r.semantic})
    types: dict[str, set] = {"cross_sectional": set(), "overlay": set(), "unknown": set()}
    for r in scored:
        if r.semantic:
            types[r.candidate_type if r.candidate_type in types else "unknown"].add(r.semantic)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "substrate": {
            "id": sub,
            "universe": public_text(desc.universe, "universe"),
            "data_sources": sorted({t for t in (public_text(s, "data source")
                                                for s in desc.data_sources) if t}),
            "data_start": _public_date(desc.data_start, "data start"),
            "data_end": _public_date(desc.data_end, "data end"),
            "search_space": public_text(desc.search_space, "search space"),
        },
        "trials": {
            "recorded": len(rows),
            "preregistered": sum(r.preregistered for r in rows),
            "scored": len(scored),
            "holdout_tested": holdout_tested,
            "holdout_count_complete": holdout_complete,
            "promising": promising,
        },
        "rejections": {"decisive": decisive, "underpowered": underpowered,
                       "unclassified": unclassified},
        "power": {
            "mde_at_test_min": mde_min,
            "mde_at_test_max": mde_max,
            "mde_substrate": mde_substrate,
            "mde_unmeasured_ticks": unmeasured,
            "mde_source": mde_source,
            "economic_floor": floor,
            "economic_floor_source": ("generation.min_combination_uplift" if floor_key == "shipped"
                                      else "guards.uplift_min"),
            "economic_floor_gates_hash": floors[f"{floor_key}_hash"],
            "decisive_mde_multiple": floors["multiple"],
        },
        "ticks": tick_block,
        "provenance": {
            "source": "trial_ledger",
            "runs": len(runs),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "crucible_versions": distinct([r.version for r in rows]
                                          + [m["crucible_version"] for m in mans]),
            "contracts": contracts,
            "proposers": distinct(m["agent_model_id"] for m in mans),
            "gates_hashes": distinct(m["gates_hash"] for m in mans),
            "corrected_gates_hashes": distinct(m["corrected_gates_hash"] for m in mans),
            "search_memory_gates_hashes": distinct(m["search_memory_gates_hash"] for m in mans),
            "cohort_gates_hashes": distinct(m["cohort_gates_hash"] for m in mans),
            "data_snapshot_hashes": distinct([r.snapshot for r in rows]
                                             + [t.get("snapshot_hash") for t in ticks]
                                             + [m["data_snapshot_hash"] for m in mans]),
            "manifest_hashes": distinct(t.get("manifest_hash") for t in ticks),
        },
        "hypotheses": {"count": len(hashes),
                       "by_type": {k: len(v) for k, v in types.items()},
                       "semantic_hashes": hashes},
    }
    payload["export_id"] = export_id(payload)
    return payload


def write_exports(payloads: list[dict], out_dir: str | Path) -> list[Path]:
    """Write each payload to ``<out_dir>/<substrate>/<date>-<id>.json``. Refuses a payload that does
    not validate, so a malformed file cannot be produced by the tool that defines the format."""
    out = Path(out_dir)
    written: list[Path] = []
    for payload in payloads:
        rel = relative_path(payload)
        errors = validate_payload(payload, relpath=rel, require_schema=False)
        if errors:
            raise ValueError(f"refusing to write {rel}: " + "; ".join(errors))
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(payload).encode("utf-8"))
        written.append(path)
    return written


# --------------------------------------------------------------------------------------- validate
def _strings(node: object):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _schema_errors(payload: object, *, required: bool) -> list[str]:
    try:
        import jsonschema                                # a dev dependency, imported where it is used
    except ImportError:
        if required:
            raise RuntimeError("validating the community ledger needs the jsonschema package "
                               "(pip install -e '.[dev]')") from None
        return []
    validator = jsonschema.Draft202012Validator(SCHEMA)
    return sorted(f"schema: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
                  for e in validator.iter_errors(payload))


def validate_payload(payload: object, *, relpath: str | None = None,
                     require_schema: bool = True) -> list[str]:
    """Every reason ``payload`` is not a well-formed ledger entry (empty list = valid).

    Beyond the schema: the id matches the content, the path matches the id, the counts add up, the
    verdict is the one those counts imply, and no string looks like a local path or an address.
    ``require_schema=False`` lets the exporter run without ``jsonschema`` installed; the checks
    below still run, and CI validates every committed file against the schema."""
    errors = _schema_errors(payload, required=require_schema)
    if errors or not isinstance(payload, dict):
        return errors or ["schema: not a JSON object"]

    if payload["export_id"] != export_id(payload):
        errors.append("export_id does not match the content (the file was edited after export; "
                      "re-run the export instead)")
    if relpath is not None and relpath.replace("\\", "/") != relative_path(payload):
        errors.append(f"path should be {relative_path(payload)}")

    t, r, p, h = payload["trials"], payload["rejections"], payload["power"], payload["hypotheses"]
    if h["count"] != len(h["semantic_hashes"]):
        errors.append("hypotheses.count does not equal the number of hashes listed")
    if sum(h["by_type"].values()) < len(h["semantic_hashes"]):
        errors.append("hypotheses.by_type does not cover every hash listed")
    if h["semantic_hashes"] != sorted(h["semantic_hashes"]):
        errors.append("hypotheses.semantic_hashes is not sorted")
    if t["scored"] > t["recorded"] or t["preregistered"] > t["recorded"]:
        errors.append("trials: scored and preregistered cannot exceed recorded")
    if t["holdout_tested"] is None and t["holdout_count_complete"]:
        errors.append("trials.holdout_count_complete cannot be true when holdout_tested is null")
    if (t["holdout_tested"] is None) != (r["unclassified"] is None):
        errors.append("rejections.unclassified must be null exactly when trials.holdout_tested is")
    elif t["holdout_tested"] is not None and (
            r["decisive"] + r["underpowered"] + r["unclassified"] + t["promising"]
            < t["holdout_tested"]):
        errors.append("rejections and promising do not account for every holdout-tested trial")
    if (p["mde_at_test_min"] is None) != (p["mde_at_test_max"] is None):
        errors.append("power: mde_at_test_min and mde_at_test_max must both be set or both null")
    elif p["mde_at_test_min"] is not None and p["mde_at_test_min"] > p["mde_at_test_max"]:
        errors.append("power: mde_at_test_min exceeds mde_at_test_max")
    if (p["mde_at_test_min"] is None) != (p["mde_source"] == "none"):
        errors.append("power: mde_source must be 'none' exactly when no MDE is given")

    verdict, reasons = decide_verdict(
        promising=t["promising"], holdout_tested=t["holdout_tested"],
        holdout_complete=t["holdout_count_complete"], decisive=r["decisive"],
        underpowered=r["underpowered"], unclassified=r["unclassified"],
        mde_at_test_max=p["mde_at_test_max"], mde_substrate=p["mde_substrate"],
        floor=p["economic_floor"], multiple=p["decisive_mde_multiple"])
    if payload["verdict"] != verdict:
        errors.append(f"verdict {payload['verdict']} does not follow from the counts (they imply "
                      f"{verdict})")
    elif payload["verdict_reasons"] != reasons:
        errors.append(f"verdict_reasons should be {reasons}")

    for text in _strings(payload):
        for pattern, label in _PRIVATE_TEXT:
            if pattern.search(text):
                errors.append(f"contains {label}: {text[:40]!r}")
                break
    return errors


def validate_tree(root: str | Path = DEFAULT_LEDGER_DIR) -> dict[str, list[str]]:
    """``{relative path: [errors]}`` for every ``*.json`` under ``root`` (errors may be empty)."""
    root = Path(root)
    report: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            report[rel] = [f"not readable JSON: {e}"]
            continue
        report[rel] = validate_payload(payload, relpath=rel)
    return report


# --------------------------------------------------------------------------------------- read
def load_community_ledger(root: str | Path = DEFAULT_LEDGER_DIR) -> list[dict]:
    """Every VALID entry under ``root``, in path order. Invalid files are skipped with a warning —
    a reader should not act on a file the validator would reject."""
    root = Path(root)
    entries: list[dict] = []
    for rel, errors in validate_tree(root).items():
        if errors:
            log.warning("community ledger: ignoring %s (%s)", rel, errors[0])
            continue
        entries.append(json.loads((root / rel).read_text(encoding="utf-8")))
    return entries


def power_phrase(entry: dict) -> str:
    """One clause on how strong the test was, next to the smallest edge the contract accepts
    (both are annualized Sharpe uplift over the base book)."""
    p = entry["power"]
    floor = f"a floor of {p['economic_floor']:.2f}"
    if p["mde_at_test_min"] is not None:
        return f"smallest detectable edge {p['mde_at_test_min']:.2f} against {floor}"
    if p["mde_substrate"] is not None:
        return (f"no power stamp at test; the substrate was later stamped at "
                f"{p['mde_substrate']:.2f} against {floor}")
    return "no power stamp recorded"


def _searched(entry: dict) -> str:
    """What the author says they searched — the only way to tell whether yours is the same search."""
    sub = entry["substrate"]
    parts = [sub["universe"], sub["search_space"]]
    if sub["data_sources"]:
        span = " to ".join(d for d in (sub["data_start"], sub["data_end"]) if d)
        parts.append("; ".join(sub["data_sources"]) + (f", {span}" if span else ""))
    return " | ".join(p for p in parts if p)


def describe_substrate(entries: list[dict], substrate_id: str) -> str:
    """Plain-text answer to "has anyone already searched this?", for a person to read.

    Advice only. Nothing here is a gate: the orchestrator never calls it, and an entry — DECISIVE or
    not — is somebody else's search on somebody else's data."""
    mine = [e for e in entries if e["substrate"]["id"] == substrate_id]
    if not mine:
        return (f"{substrate_id}: no community record. Nobody has shared a search on a substrate "
                "with this name.")
    lines = [f"{substrate_id}: {len(mine)} shared search(es)."]
    for e in sorted(mine, key=relative_path):
        t = e["trials"]
        tested = ("an unrecorded number" if t["holdout_tested"] is None else
                  f"{'' if t['holdout_count_complete'] else 'at least '}{t['holdout_tested']}")
        power = power_phrase(e)
        lines.append(f"  {relative_path(e)}: {e['verdict']} - {e['hypotheses']['count']} hypotheses "
                     f"scored, {tested} reached the out-of-sample test; {power}.")
        about = _searched(e)
        if about:
            lines.append(f"      {about}")
    verdicts = {e["verdict"] for e in mine}
    if PROMISING_FOUND in verdicts:
        lines.append("  Someone reports a result that passed here. It is their claim on their data.")
    if CLOSED_DECISIVE in verdicts:
        lines.append("  A decisive search found nothing for the hypotheses it lists. Bring new "
                     "evidence (new data, a new mechanism) before repeating it.")
    if OPEN_UNDERPOWERED in verdicts:
        lines.append("  An open search is NOT settled: the test could not tell either way. The same "
                     "hypotheses on a deeper panel are still worth running.")
    return "\n".join(lines)
