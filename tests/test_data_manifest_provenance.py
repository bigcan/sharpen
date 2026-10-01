"""Data-manifest provenance: the origin fields, their tri-state, and the anti-wipe guard.

Covers `scripts/build_data_manifest.py` (emission + inheritance) and
`scripts.validate_config.check_data_provenance` (Phase α WARN / Phase β FAIL).

Every assertion here is two-directional. A check that can only ever pass is
indistinguishable from a check that never runs, and this file guards exactly the
class of bug where an absent value is silently read as a declared one — so the
"declared" and "undeclared" cases are both asserted, and `False` is asserted to be
distinguishable from `None` rather than merely falsy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_data_manifest import (  # noqa: E402
    PROVENANCE_FIELDS,
    PROVENANCE_REQUIRED,
    _tristate,
    build_manifest,
    empty_provenance,
    main,
    merge_provenance,
    read_existing_provenance,
)
from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_data_provenance,
    validate,
)


# ---------- fixtures --------------------------------------------------------

def _write_parquet(path: Path, n: int = 400) -> Path:
    """A clean synthetic OHLCV series that passes detect_outliers.

    Anchored to *now* rather than a fixed date so the recency gate in
    `check_data_manifest` cannot start failing the wiring test as the calendar
    moves past it.
    """
    rng = np.random.default_rng(20260920)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.001, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.001, size=n))
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=end, periods=n, freq="15min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1.0, 100.0, size=n),
        }
    )
    df.to_parquet(path)
    return path


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    return _write_parquet(tmp_path / "synth_15min.parquet")


def _cfg(parquet_path: Path, *, required: bool = False) -> dict:
    """Enforcement switch lives in `gates:`, beside its `*_required` siblings."""
    cfg: dict = {"data": {"file_path": str(parquet_path), "frequency": "15min"}}
    if required:
        cfg["gates"] = {"provenance_required": True}
    return cfg


# ---------- emission: absent is not zero ------------------------------------

def test_manifest_always_carries_a_provenance_block(parquet: Path):
    m = build_manifest(parquet)
    assert "provenance" in m, "provenance block must exist even when undeclared"
    assert set(m["provenance"]) == set(PROVENANCE_FIELDS)


def test_undeclared_fields_are_none_not_false_or_empty_string(parquet: Path):
    block = build_manifest(parquet)["provenance"]
    for key in PROVENANCE_FIELDS:
        assert block[key] is None, f"{key} must default to None, got {block[key]!r}"
    # The distinction that matters: nobody-said vs fetcher-asserted-no-fallback.
    assert block["fallback_used"] is not False


def test_declared_fields_reach_the_manifest(parquet: Path):
    m = build_manifest(
        parquet,
        provenance={"vendor": "oanda", "feed": "demo", "symbol_convention": "SPX500"},
    )
    assert m["provenance"]["vendor"] == "oanda"
    assert m["provenance"]["feed"] == "demo"
    assert m["provenance"]["symbol_convention"] == "SPX500"
    assert m["provenance"]["notes"] is None  # undeclared neighbours stay undeclared


def test_fallback_used_false_is_preserved_as_false(parquet: Path):
    """An explicit False is an assertion by the fetcher and must survive the merge."""
    block = build_manifest(parquet, provenance={"fallback_used": False})["provenance"]
    assert block["fallback_used"] is False, "explicit False must not collapse back to None"


# ---------- merge / inheritance ---------------------------------------------

def test_merge_overlays_only_declared_values():
    inherited = {**empty_provenance(), "vendor": "oanda", "feed": "demo"}
    merged = merge_provenance(inherited, {"feed": "live", "notes": None})
    assert merged["vendor"] == "oanda", "omitted flag must inherit"
    assert merged["feed"] == "live", "declared flag must override"
    assert merged["notes"] is None


def test_read_existing_provenance_round_trips(tmp_path: Path):
    p = tmp_path / "x.manifest.json"
    p.write_text(json.dumps({"provenance": {"vendor": "ctrader", "feed": "demo"}}), encoding="utf-8")
    block = read_existing_provenance(p)
    assert block["vendor"] == "ctrader"
    assert block["symbol_convention"] is None  # absent key, not a crash


@pytest.mark.parametrize(
    "payload",
    ["not json at all", json.dumps({"provenance": "a string, not a dict"}), json.dumps({})],
)
def test_malformed_prior_manifest_yields_empty_block_without_raising(tmp_path: Path, payload: str):
    p = tmp_path / "bad.manifest.json"
    p.write_text(payload, encoding="utf-8")
    assert read_existing_provenance(p) == empty_provenance()


def test_missing_prior_manifest_yields_empty_block(tmp_path: Path):
    assert read_existing_provenance(tmp_path / "nope.manifest.json") == empty_provenance()


# ---------- the anti-wipe guard (both directions) ---------------------------

def test_rebuild_without_flags_preserves_provenance(parquet: Path, monkeypatch, capsys):
    """The failure this guards: refreshing a manifest silently drops its origin."""
    out = parquet.with_suffix(".manifest.json")
    monkeypatch.setattr(
        sys, "argv",
        ["build_data_manifest.py", str(parquet), "--write",
         "--vendor", "oanda", "--feed", "demo", "--symbol-convention", "SPX500"],
    )
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["build_data_manifest.py", str(parquet), "--write"])
    assert main() == 0
    capsys.readouterr()

    block = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert block["vendor"] == "oanda", "rebuild without flags must NOT wipe provenance"
    assert block["symbol_convention"] == "SPX500"


def test_clear_provenance_actually_clears(parquet: Path, monkeypatch, capsys):
    """The opposite direction: the escape hatch must really remove it."""
    out = parquet.with_suffix(".manifest.json")
    monkeypatch.setattr(
        sys, "argv",
        ["build_data_manifest.py", str(parquet), "--write", "--vendor", "oanda"],
    )
    assert main() == 0
    capsys.readouterr()
    assert json.loads(out.read_text(encoding="utf-8"))["provenance"]["vendor"] == "oanda"

    monkeypatch.setattr(
        sys, "argv",
        ["build_data_manifest.py", str(parquet), "--write", "--clear-provenance"],
    )
    assert main() == 0
    capsys.readouterr()
    assert json.loads(out.read_text(encoding="utf-8"))["provenance"] == empty_provenance()


@pytest.mark.parametrize("raw,expected", [("true", True), ("FALSE", False), ("1", True), ("no", False)])
def test_tristate_parses_both_polarities(raw: str, expected: bool):
    assert _tristate(raw) is expected


def test_tristate_rejects_anything_else():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _tristate("maybe")


# ---------- validator: Phase α WARN / Phase β FAIL --------------------------

def _manifest_with(parquet_path: Path, provenance) -> None:
    out = parquet_path.with_suffix(".manifest.json")
    m = build_manifest(parquet_path)
    if provenance is _MISSING:
        m.pop("provenance")
    else:
        m["provenance"] = provenance
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")


_MISSING = object()
_COMPLETE = {"vendor": "oanda", "feed": "demo", "symbol_convention": "SPX500",
             "fallback_used": False, "retrieved_at": None, "notes": None}


def test_missing_block_warns_but_does_not_fail_by_default(parquet: Path):
    _manifest_with(parquet, _MISSING)
    r = ValidationResult()
    check_data_provenance(_cfg(parquet), "hpo", r)
    assert r.warnings and not r.failures, "Phase α must not invalidate existing manifests"


def test_missing_block_fails_when_workstream_opts_in(parquet: Path):
    _manifest_with(parquet, _MISSING)
    r = ValidationResult()
    check_data_provenance(_cfg(parquet, required=True), "paper-deploy", r)
    assert r.failures, "provenance_required: true must escalate the same defect to FAIL"


def test_opt_in_is_honoured_from_an_external_gates_file(parquet: Path, tmp_path: Path, monkeypatch):
    """CLAUDE.md puts per-workstream gates in `configs/<ws>.gates.yaml`, not inline.

    Reading `cfg["gates"]` directly would leave enforcement OFF for exactly the
    workstreams that followed that rule — the switch would be declared in the file
    the project mandates and read by nobody. Caught in audit on 2026-09-20.
    """
    _manifest_with(parquet, _MISSING)
    gates_file = tmp_path / "ws.gates.yaml"
    gates_file.write_text(
        yaml.safe_dump({"gates": {"provenance_required": True}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)  # overlay resolves relative paths against CWD
    cfg = {
        "data": {"file_path": str(parquet), "frequency": "15min"},
        "ensemble": {"gates_file": gates_file.name},
    }
    r = ValidationResult()
    check_data_provenance(cfg, "paper-deploy", r)
    assert r.failures, "gates_file opt-in must enforce, not silently stay dormant"


def test_incomplete_block_is_caught(parquet: Path):
    _manifest_with(parquet, {**empty_provenance(), "vendor": "oanda"})
    r = ValidationResult()
    check_data_provenance(_cfg(parquet), "hpo", r)
    assert r.warnings
    assert "feed" in r.warnings[0] and "symbol_convention" in r.warnings[0]


def test_complete_block_passes_clean(parquet: Path):
    """The other direction: a correct manifest must produce NO warning."""
    _manifest_with(parquet, dict(_COMPLETE))
    r = ValidationResult()
    check_data_provenance(_cfg(parquet, required=True), "paper-deploy", r)
    assert not r.failures and not r.warnings
    assert any("data provenance OK" in p for p in r.passed)


def test_fallback_used_warns_even_when_identity_fields_are_complete(parquet: Path):
    _manifest_with(parquet, {**_COMPLETE, "fallback_used": True})
    r = ValidationResult()
    check_data_provenance(_cfg(parquet), "hpo", r)
    assert any("fallback_used" in w for w in r.warnings)


def test_fallback_used_false_does_not_warn(parquet: Path):
    """Two-directional partner of the test above — False must be silent."""
    _manifest_with(parquet, {**_COMPLETE, "fallback_used": False})
    r = ValidationResult()
    check_data_provenance(_cfg(parquet), "hpo", r)
    assert not any("fallback_used" in w for w in r.warnings)


def test_runtime_fetch_sources_are_exempt():
    r = ValidationResult()
    check_data_provenance({"data": {"source": "ccxt", "frequency": "1h"}}, "hpo", r)
    assert not r.failures and not r.warnings
    assert any("runtime fetch" in p for p in r.passed)


def test_absent_file_path_is_left_to_check_data_manifest():
    r = ValidationResult()
    check_data_provenance({"data": {}}, "hpo", r)
    assert not r.failures and not r.warnings and not r.passed


def test_required_fields_are_the_identity_fields_only():
    """fallback_used is a warning channel, not an identity field."""
    assert set(PROVENANCE_REQUIRED) == {"vendor", "feed", "symbol_convention"}
    assert "fallback_used" not in PROVENANCE_REQUIRED


def test_exemption_mirrors_check_data_manifest_exactly():
    """`ccxt` is exact there, so it must be exact here.

    A source like `ccxt_binance` is NOT exempt from the manifest requirement in
    `check_data_manifest`, so exempting it here would skip provenance on a dataset
    that does ship a manifest — two checks reading one field and disagreeing.
    """
    r = ValidationResult()
    check_data_provenance({"data": {"source": "ccxt_binance"}}, "hpo", r)
    assert not r.passed, "ccxt_binance must not take the runtime-fetch exemption"

    r = ValidationResult()
    check_data_provenance({"data": {"source": "yfinance_etf"}}, "hpo", r)
    assert any("runtime fetch" in p for p in r.passed), "yfinance* is prefix-exempt"


# ---------- the check is actually WIRED into validate() ---------------------

def test_check_is_reachable_from_the_validate_entry_point(parquet: Path, tmp_path: Path):
    """Tripwire for the dispatch line itself.

    Every other test here calls `check_data_provenance` directly, so all of them
    keep passing if the call is dropped from `validate()` — a mutation run on
    2026-09-20 confirmed exactly that. A check nobody invokes is the
    declared-but-not-wired false assurance the audit skill exists to catch, so the
    wiring gets its own test that fails when the dispatch line is removed.
    """
    _manifest_with(parquet, _MISSING)
    cfg_path = tmp_path / "wiring.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"data": {"file_path": str(parquet), "frequency": "15min"}}),
        encoding="utf-8",
    )
    r = validate(cfg_path, "hpo")
    assert any("provenance" in w for w in r.warnings), (
        "check_data_provenance is not reached from validate() — the dispatch call is missing"
    )
