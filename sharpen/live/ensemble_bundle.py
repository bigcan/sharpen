"""Protocol v2.3 atomic ensemble swap-bundle reader.

Counterpart to the writer at
``scripts/sg1_xauusd_ensemble_eval.py:_pack_swap_bundle``. Lays out the
``ensemble_v{N}.tar.gz`` contents on disk, validates per-file SHA256
hashes against ``ensemble_manifest.json``, and surfaces a
:class:`BundleContents` dataclass that the live runners feed into
``EnsembleAgent``.

Live container contract (per ``docs/protocol_v2.md`` §4 v2.3):
- Bundle SHA256 (the top-level ``.sha256`` sidecar) must match the
  archive byte-for-byte before extraction.
- Every checkpoint file under ``checkpoints/seed_<id>/`` and every
  declared (non-MISSING) normalizer file under ``normalizers/seed_<id>/``
  must hash-match the value declared in ``ensemble_manifest.json``.
- All-or-nothing: a single mismatch raises :class:`BundleIntegrityError`
  and the runner aborts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_VALID_RULES = ("ens_mean", "ens_median", "ens_agreement", "ens_majority", "ens_pf_weighted")


class BundleIntegrityError(RuntimeError):
    """Raised when bundle SHA256 / per-file SHA256 / structural checks fail."""


@dataclass
class BundleContents:
    """Resolved bundle state ready for :class:`EnsembleAgent` construction."""

    bundle_path: Path
    extract_dir: Path
    manifest: dict[str, Any]
    seeds: list[int]
    checkpoint_paths: dict[int, Path]
    normalizer_paths: dict[int, Path | None]
    resolved_config: dict[str, Any]
    chosen_rule: str
    schema_version: str
    workstream: str
    version: str
    seed_pfs: dict[int, float] = field(default_factory=dict)

    @property
    def deadband(self) -> float:
        """Deadband threshold from the resolved config (v7 default 0.25).

        Falls back to 0.25 if the config doesn't declare one — but emits a
        warning so the operator notices missing provenance.
        """
        trading = self.resolved_config.get("trading", {}) or {}
        env_cfg = self.resolved_config.get("env", {}) or {}
        for key in ("deadband_threshold", "deadband"):
            if key in trading:
                return float(trading[key])
            if key in env_cfg:
                return float(env_cfg[key])
        logger.warning(
            "bundle %s: deadband not found in resolved config; defaulting to 0.25",
            self.bundle_path.name,
        )
        return 0.25


def _sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(_chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _read_sidecar_sha256(bundle_path: Path) -> str | None:
    """Return the top-level bundle hash from ``<bundle>.sha256`` if present."""
    sidecar = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    line = sidecar.read_text(encoding="utf-8").strip().splitlines()[0]
    # writer uses "<sha>  <filename>"
    return line.split()[0]


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract with path-traversal guard.

    Python 3.12+ supports ``filter='data'``; we use it when available and fall
    back to a manual prefix check to keep the module importable on 3.11.
    """
    dest_resolved = dest.resolve()
    try:
        tar.extractall(dest, filter="data")
        return
    except TypeError:
        pass  # python < 3.12 — fall through to manual guard
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest_resolved)):
            raise BundleIntegrityError(
                f"refusing to extract {member.name!r}: escapes destination"
            )
    tar.extractall(dest)  # noqa: S202 — guarded above


def extract_and_verify_bundle(
    bundle_path: str | Path,
    dest: str | Path | None = None,
    *,
    require_normalizers: bool = True,
    purge_existing: bool = True,
) -> BundleContents:
    """Extract ``ensemble_v{N}.tar.gz`` and verify SHA256 manifest.

    Args:
        bundle_path: path to the ``.tar.gz`` swap bundle.
        dest: extraction directory; defaults to
            ``<bundle_dir>/_extracted_<bundle_stem>``. Created if absent.
        require_normalizers: if True, any seed whose
            ``normalizer_sha256[seed] == "MISSING"`` raises. Set False only
            for solo / non-EMA-Z workstreams.
        purge_existing: if True, wipes ``dest`` before extracting (avoids
            stale artifacts from a partial prior extraction).

    Returns:
        :class:`BundleContents` — caller passes the inner state to
        ``SACAgent.load`` (per seed) and ``EnsembleAgent`` (for aggregation).

    Raises:
        BundleIntegrityError on any SHA256 mismatch, missing required file,
        or manifest schema violation.
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise BundleIntegrityError(f"bundle not found: {bundle_path}")

    expected_top = _read_sidecar_sha256(bundle_path)
    actual_top = _sha256_file(bundle_path)
    if expected_top is None:
        logger.warning(
            "bundle %s: no .sha256 sidecar found; skipping top-level integrity check "
            "(actual=%s...)",
            bundle_path.name, actual_top[:12],
        )
    elif expected_top != actual_top:
        raise BundleIntegrityError(
            f"bundle SHA256 mismatch: expected {expected_top[:12]}…, "
            f"got {actual_top[:12]}… for {bundle_path}"
        )

    if dest is None:
        dest = bundle_path.parent / f"_extracted_{bundle_path.stem.removesuffix('.tar')}"
    dest = Path(dest)
    if purge_existing and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_path, "r:gz") as tar:
        _safe_extract(tar, dest)

    manifest_path = dest / "ensemble_manifest.json"
    if not manifest_path.exists():
        raise BundleIntegrityError(
            f"bundle {bundle_path} missing ensemble_manifest.json after extract"
        )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema_version = str(manifest.get("schema_version", ""))
    if not schema_version.startswith("v2."):
        raise BundleIntegrityError(
            f"unsupported bundle schema_version {schema_version!r}; expected v2.x"
        )

    chosen_rule = str(manifest.get("chosen_rule", ""))
    if chosen_rule not in _VALID_RULES:
        raise BundleIntegrityError(
            f"manifest chosen_rule={chosen_rule!r} not in {_VALID_RULES}"
        )

    seeds_raw = manifest.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise BundleIntegrityError("manifest.seeds missing or empty")
    seeds = [int(s) for s in seeds_raw]

    config_path = dest / "config.resolved.yaml"
    if not config_path.exists():
        raise BundleIntegrityError("bundle missing config.resolved.yaml")
    declared_config_sha = manifest.get("config_sha256")
    actual_config_sha = _sha256_file(config_path)
    if declared_config_sha and declared_config_sha != actual_config_sha:
        raise BundleIntegrityError(
            f"config.resolved.yaml SHA256 mismatch: "
            f"manifest={declared_config_sha[:12]}… actual={actual_config_sha[:12]}…"
        )
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    ckpt_sha = manifest.get("checkpoint_sha256", {}) or {}
    norm_sha = manifest.get("normalizer_sha256", {}) or {}

    checkpoint_paths: dict[int, Path] = {}
    normalizer_paths: dict[int, Path | None] = {}
    for s in seeds:
        ckpt = dest / "checkpoints" / f"seed_{s}" / "checkpoint_final.pth"
        if not ckpt.exists():
            raise BundleIntegrityError(
                f"seed {s}: checkpoint missing at {ckpt} after extract"
            )
        declared = ckpt_sha.get(str(s))
        actual = _sha256_file(ckpt)
        if declared is None:
            raise BundleIntegrityError(
                f"seed {s}: manifest.checkpoint_sha256 has no entry"
            )
        if declared != actual:
            raise BundleIntegrityError(
                f"seed {s}: checkpoint SHA256 mismatch "
                f"(manifest={declared[:12]}… actual={actual[:12]}…)"
            )
        checkpoint_paths[s] = ckpt

        declared_norm = norm_sha.get(str(s))
        seed_norm_dir = dest / "normalizers" / f"seed_{s}"
        if declared_norm == "MISSING":
            if require_normalizers:
                raise BundleIntegrityError(
                    f"seed {s}: normalizer marked MISSING in manifest and "
                    f"require_normalizers=True (EMA-Z LEAK-1 invariant)"
                )
            normalizer_paths[s] = None
            continue
        if declared_norm is None:
            raise BundleIntegrityError(
                f"seed {s}: manifest.normalizer_sha256 missing entry"
            )
        if not seed_norm_dir.exists():
            raise BundleIntegrityError(
                f"seed {s}: normalizer dir {seed_norm_dir} missing after extract"
            )
        norm_files = [p for p in seed_norm_dir.iterdir() if p.is_file()]
        if len(norm_files) != 1:
            raise BundleIntegrityError(
                f"seed {s}: expected exactly 1 normalizer file under "
                f"{seed_norm_dir}, found {len(norm_files)}"
            )
        norm_file = norm_files[0]
        actual_norm = _sha256_file(norm_file)
        if declared_norm != actual_norm:
            raise BundleIntegrityError(
                f"seed {s}: normalizer SHA256 mismatch "
                f"(manifest={declared_norm[:12]}… actual={actual_norm[:12]}…)"
            )
        normalizer_paths[s] = norm_file

    seed_pfs_raw = manifest.get("seed_pfs") or {}
    seed_pfs = {int(k): float(v) for k, v in seed_pfs_raw.items()} if seed_pfs_raw else {}

    contents = BundleContents(
        bundle_path=bundle_path,
        extract_dir=dest,
        manifest=manifest,
        seeds=seeds,
        checkpoint_paths=checkpoint_paths,
        normalizer_paths=normalizer_paths,
        resolved_config=resolved_config,
        chosen_rule=chosen_rule,
        schema_version=schema_version,
        workstream=str(manifest.get("workstream", "")),
        version=str(manifest.get("version", "")),
        seed_pfs=seed_pfs,
    )
    logger.info(
        "bundle verified: %s schema=%s ws=%s v=%s rule=%s seeds=%s",
        bundle_path.name, schema_version, contents.workstream,
        contents.version, chosen_rule, seeds,
    )
    return contents
