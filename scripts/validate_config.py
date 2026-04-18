"""Protocol v2 config validator.

Enforces the gates defined in `docs/protocol_v2.md`. Exits non-zero on any
violation so it can be wired into pre-commit, CI, and `run_full_pipeline.py`
launch hooks.

Usage:
    python scripts/validate_config.py --config configs/<cfg>.yaml --stage <stage>
    python scripts/validate_config.py --config configs/<cfg>.yaml --stage hpo --strict

Stages: data-prep, hpo, l1-multiseed, wf, oos, paper-deploy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("validate_config")

PROTOCOL_DOC = "docs/protocol_v2.md"
VALID_STAGES = ("data-prep", "hpo", "l1-multiseed", "wf", "oos", "paper-deploy")


@dataclass
class ValidationResult:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_manifest(data_path: Path) -> dict[str, Any] | None:
    """Manifest co-located with parquet as <stem>.manifest.json."""
    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ---------- universal checks (every stage) ----------

def check_no_fee_curriculum(cfg: dict, r: ValidationResult) -> None:
    """Steady-state fees from step 0; `fee_schedule` curriculum is banned.

    See decision_steady_state_fees_pattern.md (S464).
    """
    env = cfg.get("env", {})
    if "fee_schedule" in env:
        r.fail(
            "env.fee_schedule found — steady-state fees only. "
            "Set env.taker_fee directly and remove fee_schedule. "
            "(decision_steady_state_fees_pattern.md)"
        )
    else:
        r.ok("steady-state fees (no fee_schedule curriculum)")

    if "taker_fee" not in env:
        r.fail("env.taker_fee missing — required for steady-state fees")


def check_no_hindsight_outside_hpo(cfg: dict, stage: str, r: ValidationResult) -> None:
    """BUG-03: hindsight_weight must be 0.0 outside HPO."""
    if stage == "hpo":
        return
    env = cfg.get("env", {})
    reward = env.get("reward", {})
    hw = reward.get("hindsight_weight", 0.0)
    if hw and float(hw) > 0:
        r.fail(
            f"reward.hindsight_weight={hw} but stage={stage}. "
            "Must be 0.0 outside HPO (BUG-03)."
        )
    else:
        r.ok(f"hindsight_weight=0 (stage={stage})")


def check_gates_block(cfg: dict, r: ValidationResult) -> None:
    """All numeric gates must come from a `gates` block, not be hardcoded."""
    gates = cfg.get("gates")
    if not isinstance(gates, dict) or not gates:
        r.fail(
            "Missing `gates:` block. Per-workstream gates required by Protocol v2 §4. "
            f"See {PROTOCOL_DOC}."
        )
        return
    r.ok(f"gates block present ({len(gates)} keys)")


def check_data_manifest(cfg: dict, stage: str, r: ValidationResult) -> None:
    """Stage 0 rejection rules from §3 of protocol_v2.md."""
    data = cfg.get("data", {})
    file_path = data.get("file_path")
    if not file_path:
        r.fail("data.file_path missing")
        return

    data_path = Path(file_path)
    if not data_path.exists():
        r.fail(f"data file not found: {data_path}")
        return

    manifest = load_data_manifest(data_path)
    if manifest is None:
        r.fail(
            f"data manifest not found alongside {data_path.name} "
            f"(expected {data_path.with_suffix('.manifest.json').name}). "
            "Run scripts/build_data_manifest.py first."
        )
        return

    # Manifest content gates
    if not manifest.get("clean_ohlcv_passed"):
        r.fail("data manifest: clean_ohlcv_passed=false (DATA-CLEAN invariant)")
    if manifest.get("nan_count", 0) > 0:
        r.fail(f"data manifest: nan_count={manifest['nan_count']} (must be 0)")

    # Recency
    last_ts_str = manifest.get("last_ts")
    if last_ts_str:
        last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - last_ts).days
        max_age = 1 if stage in ("oos", "paper-deploy") else 7
        if age_days > max_age:
            r.fail(
                f"data manifest: last_ts={last_ts_str} is {age_days}d old; "
                f"stage {stage} requires <{max_age}d"
            )
        else:
            r.ok(f"data recency OK ({age_days}d old, max {max_age}d for {stage})")

    # Regime coverage
    quartiles = manifest.get("regime_quartiles", {})
    for k, v in quartiles.items():
        if v < 0.10:
            r.fail(f"regime quartile {k}={v} < 0.10 (insufficient coverage)")

    # Gap policy
    max_gap = manifest.get("max_gap_bars", 0)
    freq = manifest.get("freq_seconds", 0)
    if freq:
        gap_seconds = max_gap * freq
        if gap_seconds > 86400 and not cfg.get("env", {}).get("gap_detection"):
            r.fail(
                f"data manifest: max_gap_bars={max_gap} ({gap_seconds}s) > 1 day "
                "but env.gap_detection not set (project_xauusd_rehpo_prelaunch_checklist.md)"
            )


# ---------- stage-specific checks ----------

def check_hpo(cfg: dict, r: ValidationResult) -> None:
    """Stage 1 gates from §4."""
    hpo = cfg.get("hpo", {})
    if not hpo:
        r.fail("hpo block missing")
        return

    objective = hpo.get("objective")
    if objective != "profit_factor":
        r.fail(f"hpo.objective={objective!r} — must be 'profit_factor' (BUG-01)")
    else:
        r.ok("HPO objective = profit_factor")

    # HPO budget must be declared (no unbounded HPO)
    for key in ("trials", "steps_per_trial"):
        if key not in hpo:
            r.fail(f"hpo.{key} missing — HPO budget must be declared (Protocol v2 §4)")

    # Training-health hard-fail thresholds expected in gates
    gates = cfg.get("gates", {})
    health_keys = ("entropy_floor", "q_div_max", "action_sat_max")
    missing = [k for k in health_keys if k not in gates]
    if missing:
        r.warn(
            f"gates missing training-health keys {missing} — "
            "stage 1 will fall back to project defaults"
        )


def check_l1_multiseed(cfg: dict, r: ValidationResult) -> None:
    gates = cfg.get("gates", {})
    seeds = gates.get("l1_seeds", 5)
    if seeds < 3:
        r.fail(f"gates.l1_seeds={seeds} — must be >=3 (Protocol v2 §4 stage 2)")
    if "l1_pf_cv_max" not in gates:
        r.warn("gates.l1_pf_cv_max not set — defaulting to 0.30")


def check_wf(cfg: dict, r: ValidationResult) -> None:
    gates = cfg.get("gates", {})
    if gates.get("wf_windows", 4) < 4:
        r.fail("gates.wf_windows < 4 (Protocol v2 §4 stage 3)")


def check_paper_deploy(cfg: dict, r: ValidationResult) -> None:
    """Stage 5 live-config requirements (S468 + S470)."""
    risk = cfg.get("risk", {})
    if not risk.get("static_peak"):
        r.fail("risk.static_peak must be true for paper-deploy (project_ftmo_risk_manager_fix.md)")
    if not risk.get("kill_file"):
        r.fail("risk.kill_file path required for paper-deploy")


STAGE_CHECKS = {
    "data-prep": [],
    "hpo": [check_hpo],
    "l1-multiseed": [check_l1_multiseed],
    "wf": [check_wf],
    "oos": [],
    "paper-deploy": [check_paper_deploy],
}


def validate(config_path: Path, stage: str) -> ValidationResult:
    cfg = load_yaml(config_path)
    r = ValidationResult()

    check_no_fee_curriculum(cfg, r)
    check_no_hindsight_outside_hpo(cfg, stage, r)
    check_gates_block(cfg, r)
    check_data_manifest(cfg, stage, r)

    for check in STAGE_CHECKS[stage]:
        check(cfg, r)

    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=VALID_STAGES)
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as failures (CI mode)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 2

    result = validate(args.config, args.stage)

    for msg in result.passed:
        logger.info("  PASS  %s", msg)
    for msg in result.warnings:
        logger.warning("  WARN  %s", msg)
    for msg in result.failures:
        logger.error("  FAIL  %s", msg)

    status = result.status
    if args.strict and status == "WARN":
        status = "FAIL"

    logger.info("\nstatus=%s  config=%s  stage=%s", status, args.config.name, args.stage)
    if status == "FAIL":
        logger.error("Protocol v2 violations. See %s.", PROTOCOL_DOC)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
