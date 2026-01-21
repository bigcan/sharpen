"""Feature cache helpers for reproducible, fast experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import pandas as pd


def _normalize_cfg(obj: Any) -> Any:
    """Convert config objects into JSON-serializable stable structures."""
    if is_dataclass(obj):
        return _normalize_cfg(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _normalize_cfg(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_cfg(x) for x in obj]
    return obj


def feature_cache_key(dataset_hash: str, feature_cfg: Mapping[str, Any]) -> str:
    norm = {
        "dataset_hash": str(dataset_hash),
        "features": _normalize_cfg(feature_cfg),
    }
    payload = json.dumps(norm, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_paths(base_dir: Path, key: str) -> Tuple[Path, Path]:
    dir_path = base_dir / key
    return dir_path / "features.parquet", dir_path / "features.meta.json"


def load_cached_features(base_dir: Path, key: str) -> tuple[pd.DataFrame | None, dict | None]:
    data_path, meta_path = cache_paths(base_dir, key)
    if not data_path.exists() or not meta_path.exists():
        return None, None
    df = pd.read_parquet(data_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, meta


def save_cached_features(base_dir: Path, key: str, df: pd.DataFrame, meta: Mapping[str, Any]) -> None:
    data_path, meta_path = cache_paths(base_dir, key)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    meta_path.write_text(json.dumps(dict(meta), indent=2, sort_keys=True), encoding="utf-8")

