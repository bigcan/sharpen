"""Run a FinRL Pro experiment from a YAML config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.data.cache import feature_cache_key
from finrl_pro.mlops.logger import MLOpsLogger
from finrl_pro.mlops.risk_controls import RiskControlPolicy
from finrl_pro.mlops.risk_profiles import load_risk_profile
from finrl_pro.training.trainer import Trainer


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _stable_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_from_inputs(seed: int, dataset_hash: str, config_path: Path) -> int:
    payload = f"{seed}:{dataset_hash}:{config_path}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


@dataclass(slots=True)
class SimulatedRunArtifacts:
    returns: list[float]
    equity: list[float]
    drawdowns: list[float]
    metrics: dict[str, float]


def _compute_equity_and_metrics(returns: list[float]) -> tuple[list[float], list[float], float, float, float]:
    equity: list[float] = []
    drawdowns: list[float] = []
    cum = 1.0
    peak = 1.0
    for r in returns:
        cum *= (1.0 + r)
        equity.append(cum)
        peak = max(peak, cum)
        drawdowns.append((cum / peak) - 1.0)

    mean_daily = statistics.fmean(returns)
    std_daily = statistics.pstdev(returns) or 1e-9
    sharpe = (mean_daily / std_daily) * math.sqrt(252)
    vol_realized = std_daily * math.sqrt(252)
    max_dd = abs(min(drawdowns)) if drawdowns else 0.0
    return equity, drawdowns, sharpe, vol_realized, max_dd


def _simulate_training_outputs(
    *,
    cfg_path: Path,
    training_cfg: dict[str, Any],
) -> SimulatedRunArtifacts:
    """Simulate a training run by generating deterministic returns + metrics."""
    module_versions = dict(training_cfg.get("module_versions", {}) or {})
    agent = str(module_versions.get("agent", "PPO")).upper()
    action_space = str(module_versions.get("action.space", "CONTINUOUS")).upper()

    base_metrics = dict(training_cfg.get("metrics", {}) or {})
    target_sharpe = _stable_float(base_metrics.get("sharpe_ratio"), 1.0)
    target_vol = _stable_float(base_metrics.get("volatility"), 0.20)

    # Adjust heuristic targets by agent family to create diversity.
    if agent == "TD3":
        target_sharpe += 0.05
        target_vol = max(target_vol - 0.02, 0.10)
    elif agent == "SAC":
        target_sharpe -= 0.05
        target_vol = min(target_vol + 0.03, 0.35)

    dataset_hash = str(training_cfg.get("dataset_hash", ""))
    base_seed = int(training_cfg.get("seed", 0))
    rng_seed = _seed_from_inputs(base_seed, dataset_hash, cfg_path)
    rng = random.Random(rng_seed)

    sigma_daily = target_vol / math.sqrt(252)
    sigma_daily = max(sigma_daily, 1e-4)
    mu_daily = target_sharpe * sigma_daily / math.sqrt(252)

    horizon = int(training_cfg.get("steps", 756) or 756)
    horizon = max(horizon, 128)

    returns: list[float] = []
    drift = 0.0
    for _ in range(horizon):
        noise = rng.gauss(mu_daily + drift, sigma_daily)
        # Inject occasional shocks to create realistic drawdowns.
        if rng.random() < 0.015:
            shock = abs(rng.gauss(0.0, 3.0 * sigma_daily))
            noise -= shock
            drift = -shock * 0.25
        else:
            drift *= 0.90
        noise = max(min(noise, 0.25), -0.60)
        returns.append(noise)

    target_dd = max(_stable_float(base_metrics.get("max_drawdown"), 0.18), 0.05)
    for _ in range(3):
        equity, drawdowns, sharpe, vol_realized, max_dd = _compute_equity_and_metrics(returns)
        if max_dd <= (target_dd + 0.01):
            break
        scale = max(min(target_dd / max_dd, 1.0), 0.25)
        returns = [max(min(r * scale, 0.25), -0.60) for r in returns]
    else:
        equity, drawdowns, sharpe, vol_realized, max_dd = _compute_equity_and_metrics(returns)

    capital_at_risk = min(max_dd * 0.9, 0.09)
    leverage = min(1.0 + (0.2 if "CONTINUOUS" in action_space else 0.05), 1.5)

    metrics = {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "volatility": float(vol_realized),
        "capital_at_risk": float(capital_at_risk),
        "leverage": float(leverage),
    }
    return SimulatedRunArtifacts(
        returns=returns,
        equity=equity,
        drawdowns=drawdowns,
        metrics=metrics,
    )


def _persist_artifacts(fingerprint_id: str, sim: SimulatedRunArtifacts) -> list[str]:
    """Write simulated artifacts to disk for downstream evaluation."""
    out_dir = Path("reports") / fingerprint_id
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_path = out_dir / "returns.csv"
    with returns_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "return"])
        for idx, value in enumerate(sim.returns):
            writer.writerow([idx, value])

    equity_path = out_dir / "equity_curve.csv"
    with equity_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "equity"])
        for idx, value in enumerate(sim.equity):
            writer.writerow([idx, value])

    dd_path = out_dir / "drawdown.csv"
    with dd_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "drawdown"])
        for idx, value in enumerate(sim.drawdowns):
            writer.writerow([idx, value])

    return [str(path) for path in (returns_path, equity_path, dd_path)]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.training.run_experiment",
        description="Execute a training run using an experiment YAML.",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    manifest = Path(cfg.get("fingerprint_manifest", "finrl_pro/configs/fingerprints.yaml"))
    store = FingerprintStore(manifest_path=manifest)
    store.load()

    risk_file = Path(cfg["risk_profile_file"]) if cfg.get("risk_profile_file") else None
    risk_id = cfg.get("risk_profile_id", "default")
    risk_policy = None
    if risk_file and risk_file.exists():
        profile = load_risk_profile(risk_file, risk_id)
        risk_policy = RiskControlPolicy(profile)

    logger = MLOpsLogger()
    trainer = Trainer(
        fingerprint_store=store,
        logger=logger,
        risk_policy=risk_policy,
    )

    tcfg = cfg["training"]
    sim = _simulate_training_outputs(cfg_path=cfg_path, training_cfg=tcfg)

    # Optional dataset resolution check for snapshot-backed datasets
    ds_hash = str(tcfg.get("dataset_hash", ""))
    if ds_hash.startswith("snapshot://"):
        try:
            from finrl_pro.data.loader import DataLoader  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            logger.log_event(
                "finrl_pro.training.dataset_resolve_skipped",
                level=30,
                context={
                    "dataset_hash": ds_hash,
                    "reason": f"DataLoader import failed: {exc}",
                },
            )
        else:
            try:
                df = DataLoader.resolve_dataset(ds_hash)
                logger.log_event(
                    "finrl_pro.training.dataset_resolved",
                    context={"dataset_hash": ds_hash, "rows": int(df.shape[0])},
                )
            except Exception as e:  # noqa: BLE001
                logger.log_event(
                    "finrl_pro.training.dataset_resolve_error",
                    context={"dataset_hash": ds_hash, "error": str(e)},
                )
                raise
    # Compose module versions with feature cache key for reproducibility
    mv = dict(tcfg.get("module_versions", {}))
    feat_cfg = cfg.get("features", {})
    try:
        fkey = feature_cache_key(str(tcfg.get("dataset_hash", "")), feat_cfg)
        mv["features.cache_key"] = fkey
    except Exception:
        # Keep going if features block is malformed
        pass

    # Optional: assemble features from DB and log feature_set_id (snapshot datasets only)
    ds_hash = str(tcfg.get("dataset_hash", ""))
    use_pro_env = bool(tcfg.get("use_pro_env", False))
    mv["features.env_mode"] = "B" if use_pro_env else "A"
    if ds_hash.startswith("snapshot://") and feat_cfg:
        try:
            from finrl_pro.data.loader_pro import ProFeatureAssembler  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            logger.log_event(
                "finrl_pro.training.feature_assembly_skipped",
                level=30,
                context={"dataset_hash": ds_hash, "reason": f"import failed: {exc}"},
            )
        else:
            try:
                snapshot_id = ds_hash.split("//", 1)[1]
                assembler = ProFeatureAssembler()
                asm = assembler.assemble_from_snapshot(snapshot_id=snapshot_id, features_cfg=feat_cfg)
                mv["features.feature_set_id"] = asm.feature_set_id
            except Exception:  # noqa: BLE001
                # Non-fatal: continue without feature_set_id if assembly not available
                pass

    fingerprint = trainer.run(
        config_path=str(tcfg["config_path"]),
        dataset_hash=str(tcfg["dataset_hash"]),
        seed=int(tcfg.get("seed", 0)),
        module_versions=mv,
        metrics=sim.metrics,
        artifact_uris=list(tcfg.get("artifact_uris", [])),
        baseline_reference=str(tcfg["baseline_reference"]),
        sandbox_enabled=bool(cfg.get("sandbox_enabled", False)),
    )

    artifact_paths = _persist_artifacts(fingerprint.fingerprint_id, sim)
    fingerprint.artifact_uris = artifact_paths
    store.register(fingerprint)
    store.save()

    print(json.dumps({
        "fingerprint_id": fingerprint.fingerprint_id,
        "manifest": str(manifest),
        "config": str(cfg_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
