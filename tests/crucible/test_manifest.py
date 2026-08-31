"""Run-manifest tripwires — the reproducibility contract (spec §5)."""
from __future__ import annotations

from pathlib import Path

from sharpen.crucible.manifest import RunManifest
from sharpen.crucible.version import CRUCIBLE_VERSION


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="gen-cross_asset-abc123",
        gates_hash="deadbeef0000",
        rng_seeds={"generation": 7},
        file_drawer_N_before=0,
        file_drawer_N_after=4495,
        verdicts={"aaaaaaaaaaaa": "LOGGED", "bbbbbbbbbbbb": "PROMISING"},
    )


def test_defaults_pin_current_version_and_nullable_data_fields() -> None:
    m = _manifest()
    assert m.crucible_version == CRUCIBLE_VERSION
    assert m.data_snapshot_hash is None      # populated in P1b
    assert m.agent_model_id is None          # populated in P2
    assert m.token_cost is None              # populated in P2


def test_write_read_round_trip(tmp_path: Path) -> None:
    m = _manifest()
    path = m.write(tmp_path)
    assert path.name == "run_manifest.json"
    back = RunManifest.read(tmp_path)         # read a directory containing the manifest
    assert back == m
    assert RunManifest.read(path) == m        # ... or the file directly


def test_content_hash_is_deterministic_and_input_sensitive() -> None:
    from dataclasses import replace

    m = _manifest()
    assert m.content_hash() == _manifest().content_hash()          # deterministic
    assert len(m.content_hash()) == 12
    assert replace(m, gates_hash="0000").content_hash() != m.content_hash()   # input-sensitive


def test_from_json_ignores_unknown_keys() -> None:
    data = _manifest().to_json()
    data["a_future_phase_field"] = 123        # forward-compat: unknown keys must not crash
    assert RunManifest.from_json(data) == _manifest()
