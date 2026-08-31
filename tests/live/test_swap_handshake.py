"""Tests for Protocol v2.3 §4.5 step 6 swap-approval handshake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sharpen.live.swap_handshake import (
    check_swap_approved,
    record_successful_load,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def _make_bundle(path: Path, content: bytes = b"bundle-bytes-v1") -> Path:
    path.write_bytes(content)
    return path


# ---------- happy path: same bundle, no handshake required -------------------


def test_same_bundle_same_sha_skips_handshake(workdir: Path):
    bundle = _make_bundle(workdir / "ensemble_v3.tar.gz")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"
    record_successful_load(bundle, last_bundle_state_path=state)

    result = check_swap_approved(
        bundle,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert result.approved
    assert not result.is_swap
    assert "same bundle" in result.reason
    # sentinel should not have been required nor consumed
    assert not sentinel.exists()


# ---------- swap detection ---------------------------------------------------


def test_swap_to_different_path_requires_handshake(workdir: Path):
    old = _make_bundle(workdir / "ensemble_v3.tar.gz", b"old-bytes")
    new = _make_bundle(workdir / "ensemble_v4.tar.gz", b"new-bytes")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"
    record_successful_load(old, last_bundle_state_path=state)

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert not result.approved
    assert result.is_swap
    assert "HANDSHAKE" in result.reason


def test_swap_in_place_rebuild_detected_via_sha256(workdir: Path):
    # Same path, different bytes (operator rebuilt the bundle in place).
    bundle = _make_bundle(workdir / "ensemble_v3.tar.gz", b"v1-bytes")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"
    record_successful_load(bundle, last_bundle_state_path=state)

    bundle.write_bytes(b"v2-bytes-rebuilt")  # in-place mutation

    result = check_swap_approved(
        bundle,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert not result.approved
    assert result.is_swap
    assert "in-place rebuild" in result.reason


def test_swap_approved_when_sentinel_present(workdir: Path):
    old = _make_bundle(workdir / "v3.tar.gz", b"old")
    new = _make_bundle(workdir / "v4.tar.gz", b"new")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"
    record_successful_load(old, last_bundle_state_path=state)
    sentinel.write_text(json.dumps({"approved_by": "op", "ticket": "OPS-42"}))

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert result.approved
    assert result.is_swap
    assert "approved by operator" in result.reason
    # Sentinel must be consumed (deleted) so the next swap requires fresh approval
    assert not sentinel.exists()


def test_consume_handshake_false_preserves_sentinel(workdir: Path):
    new = _make_bundle(workdir / "v1.tar.gz")
    state = workdir / "last_bundle.json"  # missing — first run
    sentinel = workdir / "kill.swap_approved"
    sentinel.write_text("ok")

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
        consume_handshake=False,
    )
    assert result.approved
    assert sentinel.exists()  # NOT consumed


# ---------- prop-firm gating -------------------------------------------------


def test_non_prop_firm_skips_handshake_on_swap(workdir: Path):
    old = _make_bundle(workdir / "v3.tar.gz", b"old")
    new = _make_bundle(workdir / "v4.tar.gz", b"new")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"
    record_successful_load(old, last_bundle_state_path=state)

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=False,
    )
    assert result.approved
    assert result.is_swap
    assert "non-prop-firm" in result.reason


# ---------- first-load path --------------------------------------------------


def test_first_load_no_state_file_treated_as_swap(workdir: Path):
    # No prior state → treated as a swap (from None to new bundle).
    new = _make_bundle(workdir / "v1.tar.gz")
    state = workdir / "last_bundle.json"  # does not exist
    sentinel = workdir / "kill.swap_approved"

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert not result.approved
    assert result.is_swap
    assert result.previous_bundle_path is None


def test_first_load_non_prop_firm_succeeds_without_handshake(workdir: Path):
    new = _make_bundle(workdir / "v1.tar.gz")
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"

    result = check_swap_approved(
        new,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=False,
    )
    assert result.approved


# ---------- robustness -------------------------------------------------------


def test_missing_bundle_path_rejects(workdir: Path):
    state = workdir / "last_bundle.json"
    sentinel = workdir / "kill.swap_approved"

    result = check_swap_approved(
        workdir / "does_not_exist.tar.gz",
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert not result.approved
    assert "not found" in result.reason


def test_legacy_bare_path_state_treated_as_path_only(workdir: Path):
    # Older deploys may have written a bare path string instead of JSON.
    bundle = _make_bundle(workdir / "v3.tar.gz", b"x")
    state = workdir / "last_bundle.json"
    state.write_text(str(bundle))  # legacy format

    sentinel = workdir / "kill.swap_approved"
    # Same path but sha is unknown → treated as a swap (in-place rebuild
    # cannot be ruled out). Operator handshake required.
    result = check_swap_approved(
        bundle,
        last_bundle_state_path=state,
        swap_approved_path=sentinel,
        is_prop_firm=True,
    )
    assert not result.approved
    assert result.is_swap


def test_record_successful_load_round_trip(workdir: Path):
    bundle = _make_bundle(workdir / "v1.tar.gz", b"the-bytes")
    state = workdir / "state.json"
    record_successful_load(bundle, last_bundle_state_path=state)

    payload = json.loads(state.read_text())
    assert payload["path"] == str(bundle)
    assert isinstance(payload["sha256"], str) and len(payload["sha256"]) == 64
    assert "loaded_at" in payload


def test_record_successful_load_creates_parent_dir(workdir: Path):
    bundle = _make_bundle(workdir / "v1.tar.gz")
    state = workdir / "deeply" / "nested" / "state.json"
    record_successful_load(bundle, last_bundle_state_path=state)
    assert state.exists()


def test_to_dict_keys():
    from sharpen.live.swap_handshake import SwapHandshakeResult
    r = SwapHandshakeResult(
        approved=True, reason="x", is_swap=False,
        previous_bundle_path=None, previous_bundle_sha256=None,
        new_bundle_sha256="abc",
    )
    assert set(r.to_dict().keys()) == {
        "approved", "reason", "is_swap",
        "previous_bundle_path", "previous_bundle_sha256", "new_bundle_sha256",
    }
