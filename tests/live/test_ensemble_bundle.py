"""Tests for the Protocol v2.3 atomic ensemble swap-bundle reader.

Covers extraction + SHA256 verification, MISSING-normalizer policy, and
the failure modes that the live engine relies on for all-or-nothing swap.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from finrl_pro_ds.live.ensemble_bundle import (
    BundleIntegrityError,
    extract_and_verify_bundle,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_bundle(
    tmp_path: Path,
    seeds=(42, 789, 456),
    chosen_rule: str = "ens_mean",
    schema_version: str = "v2.3",
    skip_normalizer_for: tuple[int, ...] = (),
    corrupt_ckpt_for: int | None = None,
    omit_sidecar: bool = False,
    drop_config: bool = False,
    bad_chosen_rule: bool = False,
) -> Path:
    """Build a synthetic bundle on disk and return its path.

    Mirrors the writer at ``scripts/sg1_xauusd_ensemble_eval.py:_pack_swap_bundle``
    closely enough to exercise the reader's invariants.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    ckpt_dir = staging / "checkpoints"
    norm_dir = staging / "normalizers"
    ckpt_dir.mkdir()
    norm_dir.mkdir()

    ckpt_sha: dict[str, str] = {}
    norm_sha: dict[str, str] = {}
    for s in seeds:
        seed_ckpt_dir = ckpt_dir / f"seed_{s}"
        seed_ckpt_dir.mkdir()
        ckpt_path = seed_ckpt_dir / "checkpoint_final.pth"
        ckpt_path.write_bytes(f"checkpoint-bytes-seed-{s}".encode())
        ckpt_sha[str(s)] = _sha256_file(ckpt_path)

        if s in skip_normalizer_for:
            norm_sha[str(s)] = "MISSING"
            continue
        seed_norm_dir = norm_dir / f"seed_{s}"
        seed_norm_dir.mkdir()
        norm_path = seed_norm_dir / "ema_state.pkl"
        norm_path.write_bytes(f"ema-state-seed-{s}".encode())
        norm_sha[str(s)] = _sha256_file(norm_path)

    if not drop_config:
        config_path = staging / "config.resolved.yaml"
        config_path.write_text(
            yaml.safe_dump({"trading": {"deadband_threshold": 0.25}}, sort_keys=False),
            encoding="utf-8",
        )
        config_sha = _sha256_file(config_path)
    else:
        config_sha = None

    manifest = {
        "schema_version": schema_version,
        "protocol": "v2_stage_2_5_swap_bundle",
        "workstream": "test_ws",
        "version": "v1",
        "chosen_rule": "INVALID_RULE" if bad_chosen_rule else chosen_rule,
        "seeds": [int(s) for s in seeds],
        "selected_via": "diversity_aware",
        "checkpoint_sha256": ckpt_sha,
        "normalizer_sha256": norm_sha,
        "config_sha256": config_sha,
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2))

    if corrupt_ckpt_for is not None:
        # write a different payload AFTER the manifest sha was computed → mismatch
        target = ckpt_dir / f"seed_{corrupt_ckpt_for}" / "checkpoint_final.pth"
        target.write_bytes(b"corrupted-payload")

    bundle_path = tmp_path / "ensemble_v1.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(staging, arcname=".")

    if not omit_sidecar:
        sha = _sha256_file(bundle_path)
        bundle_path.with_suffix(bundle_path.suffix + ".sha256").write_text(
            f"{sha}  {bundle_path.name}\n", encoding="utf-8"
        )
    return bundle_path


def test_extract_and_verify_happy_path(tmp_path):
    bundle = _make_bundle(tmp_path)
    contents = extract_and_verify_bundle(bundle)
    assert contents.seeds == [42, 789, 456]
    assert contents.chosen_rule == "ens_mean"
    assert contents.workstream == "test_ws"
    assert contents.version == "v1"
    for seed in contents.seeds:
        assert contents.checkpoint_paths[seed].exists()
        assert contents.normalizer_paths[seed] is not None
        assert contents.normalizer_paths[seed].exists()
    assert contents.deadband == 0.25


def test_top_level_sha256_mismatch_rejects(tmp_path):
    bundle = _make_bundle(tmp_path)
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    sidecar.write_text("0" * 64 + f"  {bundle.name}\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="SHA256 mismatch"):
        extract_and_verify_bundle(bundle)


def test_omitted_sidecar_warns_but_passes(tmp_path, caplog):
    bundle = _make_bundle(tmp_path, omit_sidecar=True)
    with caplog.at_level("WARNING"):
        contents = extract_and_verify_bundle(bundle)
    assert contents.seeds  # still loaded
    assert any("no .sha256 sidecar" in m for m in caplog.messages)


def test_per_seed_checkpoint_sha_mismatch_rejects(tmp_path):
    # corrupt_ckpt_for makes the stored ckpt different from the manifest sha,
    # but we must rebuild the sidecar so the top-level check passes first.
    bundle = _make_bundle(tmp_path, corrupt_ckpt_for=42)
    # re-bundle so .sha256 matches the (corrupt) tarball
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    sidecar.write_text(f"{_sha256_file(bundle)}  {bundle.name}\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="checkpoint SHA256 mismatch"):
        extract_and_verify_bundle(bundle)


def test_missing_normalizer_with_require_true_rejects(tmp_path):
    bundle = _make_bundle(tmp_path, skip_normalizer_for=(42,))
    with pytest.raises(BundleIntegrityError, match="MISSING in manifest"):
        extract_and_verify_bundle(bundle, require_normalizers=True)


def test_missing_normalizer_with_require_false_loads(tmp_path):
    bundle = _make_bundle(tmp_path, skip_normalizer_for=(42,))
    contents = extract_and_verify_bundle(bundle, require_normalizers=False)
    assert contents.normalizer_paths[42] is None
    assert contents.normalizer_paths[789] is not None


def test_unsupported_schema_version_rejects(tmp_path):
    bundle = _make_bundle(tmp_path, schema_version="v3.0")
    with pytest.raises(BundleIntegrityError, match="unsupported bundle schema_version"):
        extract_and_verify_bundle(bundle)


def test_invalid_chosen_rule_rejects(tmp_path):
    bundle = _make_bundle(tmp_path, bad_chosen_rule=True)
    with pytest.raises(BundleIntegrityError, match="chosen_rule"):
        extract_and_verify_bundle(bundle)


def test_missing_config_yaml_rejects(tmp_path):
    bundle = _make_bundle(tmp_path, drop_config=True)
    with pytest.raises(BundleIntegrityError, match="config.resolved.yaml"):
        extract_and_verify_bundle(bundle)


def test_path_traversal_rejected(tmp_path):
    """Tarballs with `..` paths must not escape the destination dir."""
    staging = tmp_path / "evil_staging"
    staging.mkdir()
    bundle_path = tmp_path / "evil.tar.gz"
    target = staging / "innocent.txt"
    target.write_text("ok")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(target, arcname="../escape.txt")
    with pytest.raises(Exception):  # noqa: B017 — either OSError or BundleIntegrityError, both acceptable
        extract_and_verify_bundle(bundle_path)
