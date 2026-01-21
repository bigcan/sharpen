"""Helpers for enforcing risk telemetry limits from ``risk_summary.json``.

These utilities load the aggregated turnover / transaction cost telemetry
emitted by ``finrl_pro_ds.training.commands.run_matrix`` and enforce the
``RiskControlProfile`` limits referenced by each experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import yaml

from finrl_pro_ds.mlops.risk_profiles import RiskControlProfile, load_risk_profile


@dataclass(slots=True)
class RiskRunTelemetry:
    """Turnover/cost telemetry for a single config + seed."""

    config: str
    fingerprint_id: str
    seed: int | None
    avg_turnover: float | None
    total_turnover: float | None
    transaction_costs_bps: float | None


@dataclass(slots=True)
class ExperimentRiskMetadata:
    """Metadata extracted from an experiment config for risk enforcement."""

    display_name: str
    config_path: Path
    risk_profile_id: str
    risk_profile_file: Path
    sandbox_enabled: bool


def normalize_config_key(config_path: str | Path) -> str:
    return str(Path(config_path).expanduser().resolve())


def load_risk_summary_runs(path: Path) -> Dict[str, RiskRunTelemetry]:
    """Return a mapping of config path -> telemetry from ``risk_summary.json``."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")) or {}
    runs: Dict[str, RiskRunTelemetry] = {}
    for cfg in payload.values():
        for row in cfg.get("runs", []) or []:
            config = str(row.get("config"))
            if not config:
                continue
            key = normalize_config_key(config)
            runs[key] = RiskRunTelemetry(
                config=config,
                fingerprint_id=str(row.get("fingerprint_id") or ""),
                seed=(row.get("seed") if isinstance(row.get("seed"), int) else None),
                avg_turnover=row.get("avg_turnover"),
                total_turnover=row.get("total_turnover"),
                transaction_costs_bps=row.get("transaction_costs_bps"),
            )
    return runs


def _resolve_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path).expanduser()
    resolved = candidate.resolve()
    if resolved.exists():
        return resolved
    base = candidate.stem.split("__", 1)[0]
    experiments_root = Path("finrl_pro_ds") / "configs" / "experiments"
    matches = list(experiments_root.rglob(f"{base}.yaml"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"Config '{config_path}' not found; expected under tmp inputs or finrl_pro_ds/configs/experiments."
    )


@lru_cache(maxsize=256)
def _load_experiment_metadata(config_path: str) -> ExperimentRiskMetadata:
    resolved = _resolve_config_path(config_path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    profile_id = str(payload.get("risk_profile_id") or "default")
    profile_file_raw = payload.get("risk_profile_file") or "finrl_pro_ds/configs/risk_profiles.yaml"
    profile_file = Path(profile_file_raw)
    if not profile_file.is_absolute():
        profile_file = (resolved.parent / profile_file).resolve()
        if not profile_file.exists():
            profile_file = (Path.cwd() / profile_file_raw).resolve()
    sandbox_enabled = bool(payload.get("sandbox_enabled", True))
    return ExperimentRiskMetadata(
        display_name=resolved.stem,
        config_path=resolved,
        risk_profile_id=profile_id,
        risk_profile_file=profile_file,
        sandbox_enabled=sandbox_enabled,
    )


@lru_cache(maxsize=256)
def _load_profile(profile_path: str, profile_id: str) -> RiskControlProfile:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Risk profile file '{profile_path}' not found.")
    return load_risk_profile(path, profile_id)


def enforce_turnover_cost_limits(
    matrix_dir: Path,
    *,
    risk_summary_path: Path | None = None,
) -> None:
    """Raise if any config violates turnover or cost caps."""

    summary_path = risk_summary_path or (matrix_dir / "risk_summary.json")
    runs = load_risk_summary_runs(summary_path)
    if not runs:
        return
    violations: list[str] = []
    for key, telemetry in runs.items():
        meta = _load_experiment_metadata(telemetry.config)
        profile = _load_profile(str(meta.risk_profile_file), meta.risk_profile_id)
        if (
            profile.max_avg_turnover is not None
            and telemetry.avg_turnover is not None
            and telemetry.avg_turnover > profile.max_avg_turnover
        ):
            violations.append(
                f"{meta.display_name} (seed {telemetry.seed}, fp {telemetry.fingerprint_id}) avg_turnover "
                f"{telemetry.avg_turnover:.4f} exceeds {profile.max_avg_turnover:.4f} for profile "
                f"'{profile.profile_id}'."
            )
        if (
            profile.max_transaction_costs_bps is not None
            and telemetry.transaction_costs_bps is not None
            and telemetry.transaction_costs_bps > profile.max_transaction_costs_bps
        ):
            violations.append(
                f"{meta.display_name} (seed {telemetry.seed}, fp {telemetry.fingerprint_id}) transaction costs "
                f"{telemetry.transaction_costs_bps:.2f}bps exceed {profile.max_transaction_costs_bps:.2f}bps for profile "
                f"'{profile.profile_id}'."
            )
    if violations:
        joined = "\n- " + "\n- ".join(violations)
        raise RuntimeError(
            "Turnover/transaction cost breaches detected in risk_summary.json:" + joined
        )


def iter_risk_telemetry(runs: Mapping[str, RiskRunTelemetry]) -> Iterable[RiskRunTelemetry]:
    return runs.values()
