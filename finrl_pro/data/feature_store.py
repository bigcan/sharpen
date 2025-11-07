"""CLI for building and registering features into the DB feature store.

Usage examples:
  python -m finrl_pro.data.feature_store build --snapshot <uuid> --config finrl_pro/configs/experiments/sp500_daily.yaml
  python -m finrl_pro.data.feature_store export --feature-set <uuid> --fmt parquet --out data/features.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd
import yaml

from finrl_pro.data.cache import feature_cache_key
from finrl_pro.data.db import DatabaseClient


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _rows_from_wide(df: pd.DataFrame, feature_set_id: str) -> Iterable[dict[str, Any]]:
    # Expect columns: date/timestamp, tic/ticker, and feature columns
    date_col = "date" if "date" in df.columns else "timestamp"
    tic_col = "tic" if "tic" in df.columns else "ticker"
    base = [date_col, tic_col]
    feat_cols = [c for c in df.columns if c not in base]
    for _, row in df.iterrows():
        for name in feat_cols:
            val = row[name]
            if pd.isna(val):
                continue
            yield {
                "timestamp": pd.to_datetime(row[date_col]).to_pydatetime(),
                "ticker": str(row[tic_col]),
                "feature_set_id": feature_set_id,
                "name": str(name),
                "value": float(val),
            }


def cmd_build(args: argparse.Namespace) -> None:
    db = DatabaseClient(dsn=args.dsn)
    db.init_schema()
    db.init_feature_store()

    cfg = _load_yaml(Path(args.config))
    tcfg = cfg.get("training", {})
    features_cfg = cfg.get("features", {})
    dataset_hash = str(tcfg.get("dataset_hash", ""))
    cache_key = feature_cache_key(dataset_hash, features_cfg)

    feature_set_id = db.get_feature_set(snapshot_id=args.snapshot, cache_key=cache_key)
    if feature_set_id:
        print(json.dumps({"status": "exists", "feature_set_id": feature_set_id, "cache_key": cache_key}))
        return

    # Placeholder: in a full pipeline, compute features from snapshot here.
    # For now, require an input parquet of features for registration.
    if not args.from_parquet:
        raise SystemExit("--from-parquet is required to register features in this lightweight CLI")

    feature_set_id = str(uuid4())
    db.insert_feature_set(
        feature_set_id=feature_set_id,
        snapshot_id=args.snapshot,
        cache_key=cache_key,
        config_json=json.dumps(features_cfg, separators=(",", ":"), sort_keys=True),
        code_hash=args.code_hash or "unknown",
        lib_versions_json=json.dumps({}),
    )
    df = pd.read_parquet(args.from_parquet)
    rows = _rows_from_wide(df, feature_set_id)
    inserted = db.upsert_feature_values(feature_set_id, rows)
    print(json.dumps({"status": "created", "feature_set_id": feature_set_id, "inserted": inserted, "cache_key": cache_key}))


def cmd_export(args: argparse.Namespace) -> None:
    db = DatabaseClient(dsn=args.dsn)
    df = db.fetch_features(feature_set_id=args.feature_set, tickers=args.tickers or [])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.fmt == "parquet":
        df.to_parquet(out)
    elif args.fmt == "csv":
        df.to_csv(out, index=False)
    else:
        raise SystemExit(f"Unsupported fmt: {args.fmt}")
    print(json.dumps({"status": "exported", "path": str(out)}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="finrl_pro.data.feature_store", description="Feature store CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Register features for a snapshot+config")
    p_build.add_argument("--snapshot", required=True, help="Snapshot UUID")
    p_build.add_argument("--config", required=True, help="Experiment YAML path with features block")
    p_build.add_argument("--from-parquet", help="Parquet file with wide feature table to register")
    p_build.add_argument("--dsn", default=None, help="DB connection string")
    p_build.add_argument("--code-hash", default=None, help="Git code hash for lineage")
    p_build.set_defaults(func=cmd_build)

    p_export = sub.add_parser("export", help="Export feature values for a feature_set")
    p_export.add_argument("--feature-set", required=True)
    p_export.add_argument("--fmt", default="parquet", choices=["parquet", "csv"])
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--dsn", default=None)
    p_export.add_argument("--tickers", nargs="*", default=[])
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

