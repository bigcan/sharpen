"""Protocol v2 config validator.

Enforces the gates defined in `docs/protocol_v2.md`. Exits non-zero on any
violation so it can be wired into pre-commit, CI, and `run_full_pipeline.py`
launch hooks.

Usage:
    python scripts/validate_config.py --config configs/<cfg>.yaml --stage <stage>
    python scripts/validate_config.py --config configs/<cfg>.yaml --stage hpo --strict

Stages: data-prep, hpo, l1-multiseed, ensemble-confirm, wf, oos, paper-deploy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("validate_config")

PROTOCOL_DOC = "docs/protocol_v2.md"
VALID_STAGES = ("data-prep", "hpo", "l1-multiseed", "ensemble-confirm", "wf", "oos", "paper-deploy")


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

    Accepts two cost-model schemas:
      - Single-leg (GMGP1, SG-1, CMGP1):   env.taker_fee
      - Two-leg spot+perp (Funding-Arb):   environment.{spot,perp}_taker_fee_pct

    Both blocks are checked for the banned `fee_schedule:` curriculum key.
    """
    env = cfg.get("env", {}) or {}
    environment = cfg.get("environment", {}) or {}

    for block_name, block in (("env", env), ("environment", environment)):
        if "fee_schedule" in block:
            r.fail(
                f"{block_name}.fee_schedule found — steady-state fees only. "
                "Set taker_fee (or {spot,perp}_taker_fee_pct for funding-arb) "
                "directly and remove fee_schedule. "
                "(decision_steady_state_fees_pattern.md)"
            )

    one_leg = "taker_fee" in env
    two_leg = (
        "spot_taker_fee_pct" in environment
        and "perp_taker_fee_pct" in environment
    )

    if one_leg and two_leg:
        r.fail(
            "Both env.taker_fee (1-leg) and environment.{spot,perp}_taker_fee_pct "
            "(2-leg) declared. Pick one cost model."
        )
    elif one_leg:
        r.ok("steady-state fees (1-leg env.taker_fee)")
    elif two_leg:
        r.ok("steady-state fees (2-leg environment.{spot,perp}_taker_fee_pct)")
    else:
        r.fail(
            "No cost model declared. Expected env.taker_fee (1-leg) or "
            "environment.{spot_taker_fee_pct, perp_taker_fee_pct} (2-leg, "
            "funding-arb)."
        )


def check_legacy_prop_firm_block(cfg: dict, stage: str, r: ValidationResult) -> None:
    """Flag configs still using the legacy ``env.prop_firm:`` block.

    Post S495-cont the prop-firm decoupling split responsibilities:
    - Training-side DD shaping lives under ``env.risk:`` +
      :class:`finrl_pro_ds.envs.risk_shaping_wrapper.RiskShapingWrapper`.
    - Live-side profit-target tracking lives under ``challenge:`` +
      ``ChallengeStateMachine`` (live engine).

    WARN at non-paper-deploy stages (adapter still honors the legacy block);
    FAIL at ``paper-deploy`` so migrated deploys cannot ship without the new
    schema. See ``.agent/artifacts/prop_firm_decoupling_architecture.md``.
    """
    env = cfg.get("env", {}) or {}
    if "prop_firm" not in env:
        return
    msg = (
        "env.prop_firm: is deprecated — migrate to env.risk: + top-level "
        "challenge: block. The PropFirmWrapperV7 adapter is retired in Step 6. "
        "(prop_firm_decoupling_architecture.md)"
    )
    if stage == "paper-deploy":
        r.fail(msg)
    else:
        r.warn(msg)


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

    # ccxt-at-runtime pipelines (funding-arb, sync-1h crypto) don't ship a
    # pre-built parquet; skip file/manifest gating. Freshness is validated by
    # the data loader at fetch time. Exchange + frequency must still be declared.
    if data.get("source") == "ccxt":
        if not data.get("data_exchange") and not cfg.get("universe", {}).get("data_exchange"):
            r.fail("data.source=ccxt but no data_exchange declared (data.* or universe.*)")
        if not data.get("frequency"):
            r.fail("data.source=ccxt but data.frequency missing")
        r.ok(f"data source=ccxt (runtime fetch, skip manifest; stage={stage})")
        # ccxt pipelines manage gaps internally; don't require env.gap_detection.
        return

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

    # Recency — thresholds match stage semantics: data-prep = fresh ingestion,
    # hpo/l1-multiseed train on historical snapshots, wf needs moderately recent
    # windows, oos/paper-deploy must be live.
    recency_by_stage = {
        "data-prep": 7,
        "hpo": 180,
        "l1-multiseed": 180,
        "ensemble-confirm": 180,  # reuses L1 test window — same recency bound
        "wf": 90,
        "oos": 1,
        "paper-deploy": 1,
    }
    last_ts_str = manifest.get("last_ts")
    if last_ts_str:
        last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - last_ts).days
        max_age = recency_by_stage.get(stage, 7)
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


def _is_prop_firm(cfg: dict) -> bool:
    """Workstream is paper-or-live-capital per Protocol v2.2 §8 scope rules.

    Tag set mirrors the one declared in decision_protocol_v22_rlops_drift_safemode.md:
    any of `prop-firm`, `FTMO`, `Velotrade` (case-insensitive match on
    wandb.tags).
    """
    tags = [str(t).lower() for t in (cfg.get("wandb", {}).get("tags") or [])]
    return any(t in tags for t in ("prop-firm", "propfirm", "ftmo", "velotrade"))


# Protocol v2.2 §8.2 / §8.3 gate keys. Every one of these must be present in
# a prop-firm / live-capital workstream's gate YAML — missing keys reject.
# Advisory workstreams (research tier) are exempt (warn only).
_V22_DRIFT_GATE_KEYS = (
    "window_bars",
    "min_bars_before_check",
    "deadband_frac_warn",
    "deadband_frac_crit",
    "saturation_frac_warn",
    "saturation_frac_crit",
    "action_kl_warn",
    "action_kl_crit",
)
_V22_SAFE_MODE_KEYS = (
    "crit_triggers_flatten",
    "crit_repeat_window_hours",
    "crit_repeat_count_before_lockout",
)


def check_drift_safemode_gates(cfg: dict, r: ValidationResult) -> None:
    """Protocol v2.2 §8 / §8.3 drift + safe_mode gate-key presence.

    Blocking for prop-firm / live-capital; advisory otherwise. Keys live
    under the `gates:` block in the workstream YAML (co-located with the
    other per-stage thresholds). The values themselves are project-wide
    defaults and are not enforced here — only the presence of an explicit
    declaration is, so that an operator intentionally declares their
    thresholds rather than silently inheriting.

    See `decision_protocol_v22_rlops_drift_safemode.md`.
    """
    prop_firm = _is_prop_firm(cfg)
    gates = cfg.get("gates", {}) or {}
    drift_gates = gates.get("drift") or {}
    safe_gates = gates.get("safe_mode") or {}

    missing_drift = [k for k in _V22_DRIFT_GATE_KEYS if k not in drift_gates]
    missing_safe = [k for k in _V22_SAFE_MODE_KEYS if k not in safe_gates]

    if missing_drift:
        msg = (
            f"gates.drift missing v2.2 key(s): {missing_drift} — "
            f"§8.2 action-drift thresholds must be declared per workstream "
            f"(see decision_protocol_v22_rlops_drift_safemode.md)"
        )
        r.fail(msg) if prop_firm else r.warn(msg)
    else:
        r.ok(f"gates.drift has all {len(_V22_DRIFT_GATE_KEYS)} v2.2 keys")

    if missing_safe:
        msg = (
            f"gates.safe_mode missing v2.2 key(s): {missing_safe} — "
            f"§8.3 tiered safe-mode must be explicitly declared"
        )
        r.fail(msg) if prop_firm else r.warn(msg)
    else:
        r.ok(f"gates.safe_mode has all {len(_V22_SAFE_MODE_KEYS)} v2.2 keys")

    # Sanity on declared thresholds when present.
    if all(k in drift_gates for k in ("deadband_frac_warn", "deadband_frac_crit")):
        if drift_gates["deadband_frac_warn"] >= drift_gates["deadband_frac_crit"]:
            r.fail(
                f"gates.drift.deadband_frac_warn="
                f"{drift_gates['deadband_frac_warn']} must be < "
                f"deadband_frac_crit={drift_gates['deadband_frac_crit']}"
            )
    if all(k in drift_gates for k in ("saturation_frac_warn", "saturation_frac_crit")):
        if drift_gates["saturation_frac_warn"] >= drift_gates["saturation_frac_crit"]:
            r.fail(
                "gates.drift.saturation_frac_warn must be < saturation_frac_crit"
            )
    if all(k in drift_gates for k in ("action_kl_warn", "action_kl_crit")):
        if drift_gates["action_kl_warn"] >= drift_gates["action_kl_crit"]:
            r.fail("gates.drift.action_kl_warn must be < action_kl_crit")
    wb = drift_gates.get("window_bars")
    mb = drift_gates.get("min_bars_before_check")
    if wb is not None and mb is not None and mb > wb:
        r.fail(
            f"gates.drift.min_bars_before_check={mb} must be <= window_bars={wb}"
        )


def check_l1_multiseed(cfg: dict, r: ValidationResult) -> None:
    gates = cfg.get("gates", {})
    seeds = gates.get("l1_seeds", 5)
    if seeds < 3:
        r.fail(f"gates.l1_seeds={seeds} — must be >=3 (Protocol v2 §4 stage 2)")
    if "l1_pf_cv_max" not in gates:
        r.warn("gates.l1_pf_cv_max not set — defaulting to 0.30")
    # S488: prop-firm / live-capital workstreams must declare the ambiguous
    # range pre-launch (Protocol v2 §4 stage 2, pre-committed escalation rule).
    tags = cfg.get("wandb", {}).get("tags", []) or []
    prop_firm = ("prop-firm" in tags) or ("FTMO" in tags) or ("velotrade" in tags)
    if prop_firm and seeds >= 10:
        amb = gates.get("l1_pf_cv_ambiguous")
        if amb is None:
            r.warn("prop-firm workstream with l1_seeds>=10 should declare "
                   "gates.l1_pf_cv_ambiguous (default [0.22, 0.38]) for "
                   "pre-committed escalation (Protocol v2 §4 stage 2)")
        elif not (isinstance(amb, list) and len(amb) == 2 and amb[0] < amb[1]):
            r.fail(f"gates.l1_pf_cv_ambiguous={amb} invalid — must be "
                   "[low, high] with low<high")
    if prop_firm and seeds < 10:
        r.warn(f"prop-firm workstream with l1_seeds={seeds} — Protocol v2 §4 "
               "stage 2 (S488) recommends N>=10 for CV-estimator reliability")


def check_ensemble_confirm(cfg: dict, r: ValidationResult) -> None:
    """Stage 2.5 gates (Protocol v2 §4, S493 codification).

    Prop-firm / live-capital workstreams MUST pass Stage 2.5 before Stage 3.
    Non-prop-firm workstreams may opt in via `wandb.tags` or gates block, but
    the stage is advisory for them (warn, do not fail).
    """
    gates = cfg.get("gates", {}) or {}
    ens = cfg.get("ensemble", {}) or {}
    tags = cfg.get("wandb", {}).get("tags", []) or []
    prop_firm = any(t in tags for t in ("prop-firm", "propfirm", "FTMO", "velotrade"))

    # `ensemble:` block is required for Stage 2.5 configs — it names the top-3
    # seeds + aggregation rule that the eval script will load.
    if not ens:
        msg = (
            "ensemble block missing — Stage 2.5 requires `ensemble:` declaring "
            "seeds (top-3 distinct checkpoints) + rule (default ens_agreement). "
            "See configs/sg1_xauusd_ftmo_rehpo_wf_multiseed.yaml for schema."
        )
        r.fail(msg) if prop_firm else r.warn(msg)
        return

    seeds = ens.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        r.fail(
            f"ensemble.seeds={seeds!r} — must be a list of exactly 3 seed ids "
            "(top-3 from stage 2 by test PF)"
        )
    else:
        r.ok(f"ensemble.seeds = {seeds}")

    rule = ens.get("rule", "ens_agreement")
    if rule != "ens_agreement":
        r.warn(
            f"ensemble.rule={rule!r} — canonical rule is 'ens_agreement' "
            "(+17.6% SG-1, +17.15% GMGP1 uplift; see project_ensemble_protocol_"
            "confirmation_test.md). Other rules allowed for A/B but should not "
            "be promoted to deploy without independent uplift evidence."
        )

    # Gate: uplift floor must be declared per-workstream (no hardcoded default
    # in code — CLAUDE.md anti-pattern rule).
    if "ensemble_uplift_min" not in gates:
        msg = (
            "gates.ensemble_uplift_min not set — Stage 2.5 pre-committed "
            "threshold. Protocol v2 default is 1.10; add to gates YAML."
        )
        r.fail(msg) if prop_firm else r.warn(msg)
    else:
        uplift_min = gates.get("ensemble_uplift_min")
        try:
            uplift_min = float(uplift_min)
        except (TypeError, ValueError):
            r.fail(f"gates.ensemble_uplift_min={uplift_min!r} is not numeric")
            return
        if uplift_min < 1.05:
            r.warn(
                f"gates.ensemble_uplift_min={uplift_min} < 1.05 — below the "
                "AMBIGUOUS-band floor. Two-point evidence suggests ≥1.10 is "
                "the load-bearing threshold."
            )
        r.ok(f"ensemble uplift gate = {uplift_min}×")


def check_wf(cfg: dict, r: ValidationResult) -> None:
    gates = cfg.get("gates", {})
    if gates.get("wf_windows", 4) < 4:
        r.fail("gates.wf_windows < 4 (Protocol v2 §4 stage 3)")


def check_wandb_consolidation(cfg: dict, stage: str, r: ValidationResult) -> None:
    """S488+: multi-process stages must use WandB run consolidation.

    One WandB run per experiment, not per seed/window. See
    `memory/project_wandb_consolidation_plan.md`. Legacy `run_per_seed` /
    `run_per_window` keys are rejected outright.
    """
    wcfg = cfg.get("wandb", {}) or {}
    for legacy in ("run_per_seed", "run_per_window", "run_per_worker"):
        if legacy in wcfg:
            r.fail(
                f"wandb.{legacy} is retired (S488 consolidation plan). "
                f"Remove the key; launchers default to one parent run with "
                f"seed/window/worker namespaces. Pass --separate_runs to the "
                f"launcher if you genuinely need the legacy behavior."
            )
    if stage in ("l1-multiseed", "wf") and wcfg.get("consolidate") is False:
        r.fail(
            f"wandb.consolidate=false is not supported for stage '{stage}' "
            f"(S488). Remove the key or set true; use launcher flag "
            f"--separate_runs for one-off debugging."
        )


def check_paper_deploy(cfg: dict, r: ValidationResult) -> None:
    """Stage 5 live-config requirements (S468 + S470 + v2.2 §8.3)."""
    prop_firm = _is_prop_firm(cfg)
    risk = cfg.get("risk", {}) or {}
    safety = cfg.get("safety", {}) or {}
    drift = cfg.get("drift", {}) or {}

    # Challenge-phase-aware static_peak requirement. Challenge / step programs
    # use a static peak (DD from initial balance, non-trailing) — S422 fix.
    # Funded accounts use a trailing peak. Default when no challenge block
    # is declared: static (preserves legacy S422 guard).
    challenge = cfg.get("challenge", {}) or {}
    phase = challenge.get("phase")  # honored regardless of `enabled`
    eod_trailing = bool(risk.get("eod_trailing_drawdown"))
    if phase == "funded":
        # FTMO funded: trailing peak baked into `risk.static_peak=false`.
        # Velotrade funded: trailing enforced live-side via
        # `risk.eod_trailing_drawdown=true` while `risk.static_peak=true`
        # (training contract). Either pattern is acceptable as long as the
        # trailing rule is declared.
        if risk.get("static_peak") is True and not eod_trailing:
            r.fail(
                "risk.static_peak=true for funded-phase paper-deploy requires "
                "risk.eod_trailing_drawdown=true (trailing-DD live-side guard). "
                "FTMO-style: set risk.static_peak=false. "
                "Velotrade-style: keep static_peak=true + eod_trailing_drawdown=true."
            )
    else:
        if not risk.get("static_peak"):
            r.fail(
                "risk.static_peak must be true for paper-deploy "
                "(project_ftmo_risk_manager_fix.md; funded phase is the only "
                "exception and must declare challenge.phase='funded')"
            )

    # v2.2 §8.3: kill_file can live under risk.kill_file OR safety.kill_file
    # (engine reads safety.kill_file; older configs have risk.kill_file — accept
    # either but require at least one).
    if not (risk.get("kill_file") or safety.get("kill_file")):
        r.fail(
            "kill_file path required for paper-deploy — declare either "
            "`risk.kill_file` or `safety.kill_file` (engine reads the latter, "
            "watchdog reads kill_file inside the container via exec_run)"
        )

    # v2.2 §8.3 flatten-on-kill-file: prop-firm / live-capital must confirm the
    # CRIT path flattens rather than holding positions (S491 bleed-window case).
    flatten_declared = risk.get("flatten_on_kill_file") or safety.get(
        "flatten_on_kill_file",
    )
    if prop_firm and not flatten_declared:
        r.fail(
            "risk.flatten_on_kill_file (or safety.flatten_on_kill_file) must "
            "be true for prop-firm/live-capital paper-deploy — §8.3 CRIT path "
            "requires graceful flatten, not 'halt without close' (S491)"
        )

    # v2.2 §8.2 drift block: prop-firm must enable drift monitoring and point
    # at a resolvable baseline artifact. Baseline existence isn't checked here
    # (live config often runs on a different host); only declaration is.
    if prop_firm:
        if not drift.get("enabled"):
            r.fail(
                "drift.enabled must be true for prop-firm/live-capital "
                "paper-deploy — §8.2 action-drift check is blocking"
            )
        if drift.get("enabled") and not drift.get("baseline_path"):
            r.fail(
                "drift.baseline_path missing — point at the seed_report.json "
                "or ensemble_report.json emitted by Stage 2 / 2.5 writers. "
                "LOG_ONLY fallback is intentional for T5 backfill only; "
                "production deploys must resolve a baseline"
            )

    # Drift/safe_mode gate keys (see check_drift_safemode_gates) are part of
    # the universal check path so they also apply here without repetition.

    # S495-cont prop-firm decoupling (rev 2): challenge block + static_peak
    # consistency. Live deploys that were re-authored under the new schema
    # must have both env.risk.static_peak and risk.static_peak aligned
    # (train/live divergence was the S422 failure mode we're guarding).
    check_challenge_block(cfg, r)
    check_static_peak_consistency(cfg, r)


def check_challenge_block(cfg: dict, r: ValidationResult) -> None:
    """Validate the ``challenge:`` block when present (paper-deploy stage).

    Only fires when the config actually declares a challenge block; a
    legacy ``env.prop_firm:`` config hits check_legacy_prop_firm_block
    which FAILs paper-deploy on its own, so this helper is additive.

    Rules:
    - ``challenge.phase`` must be one of {step1, step2, funded, custom}.
    - ``challenge.advance_rule`` must be "manual_ack" (auto is rejected
      for capital-at-risk per ADR-2).
    - ``challenge.profit_target_pct`` must be a positive float, or null
      for the funded phase, or >= 10.0 for backtest-style full-window
      runs. funded MUST NOT set enabled=true with a finite target.
    - ``n_confirm`` and ``smoothing_window`` must be positive integers
      when present.
    """
    challenge = cfg.get("challenge")
    if not isinstance(challenge, dict):
        return
    if not challenge.get("enabled", False):
        # Disabled blocks exist on funded deploys — pass-through.
        return

    valid_phases = ("step1", "step2", "funded", "custom")
    phase = challenge.get("phase")
    if phase not in valid_phases:
        r.fail(
            f"challenge.phase must be one of {valid_phases}, got {phase!r} "
            "(prop_firm_decoupling_architecture.md Interface 3)"
        )

    advance_rule = challenge.get("advance_rule", "manual_ack")
    if advance_rule != "manual_ack":
        r.fail(
            f"challenge.advance_rule must be 'manual_ack' at paper-deploy, "
            f"got {advance_rule!r}. 'auto' is rejected for capital-at-risk "
            "per ADR-2 — operator must explicitly swap the overlay."
        )

    target = challenge.get("profit_target_pct")
    if phase == "funded":
        if target not in (None, float("inf")):
            r.fail(
                "challenge.phase=funded requires profit_target_pct=null "
                "(or disable the challenge block entirely)"
            )
    else:
        if target is None or (isinstance(target, (int, float)) and target <= 0):
            r.fail(
                f"challenge.profit_target_pct must be a positive float "
                f"for phase={phase!r}, got {target!r}"
            )

    for key in ("n_confirm", "smoothing_window"):
        if key in challenge:
            val = challenge[key]
            if not (isinstance(val, int) and val >= 1):
                r.fail(
                    f"challenge.{key} must be a positive integer, got {val!r}"
                )


def check_static_peak_consistency(cfg: dict, r: ValidationResult) -> None:
    """``env.risk.static_peak`` must equal ``risk.static_peak`` at paper-deploy.

    Train-time wrapper reads env.risk.static_peak; live-engine reads
    risk.static_peak. A mismatch was the S422 class of bug (static train,
    trailing live) — the train-time peak-accounting diverges from live,
    so the policy can trip the live DD gate on a drawdown that its
    trained-peak never saw. Keep them in sync. If only one is set,
    accept — the sibling default is compatible — but flag if both are
    set to different values.
    """
    env_risk = (cfg.get("env", {}) or {}).get("risk", {}) or {}
    risk = cfg.get("risk", {}) or {}
    env_val = env_risk.get("static_peak")
    live_val = risk.get("static_peak")
    if env_val is not None and live_val is not None and env_val != live_val:
        r.fail(
            f"env.risk.static_peak={env_val!r} != risk.static_peak={live_val!r}. "
            "Train-time wrapper and live-engine must agree (project_ftmo_risk_manager_fix.md S422)."
        )


STAGE_CHECKS = {
    "data-prep": [],
    "hpo": [check_hpo],
    "l1-multiseed": [check_l1_multiseed],
    "ensemble-confirm": [check_ensemble_confirm],
    "wf": [check_wf],
    "oos": [],
    "paper-deploy": [check_paper_deploy],
}


def validate(config_path: Path, stage: str) -> ValidationResult:
    cfg = load_yaml(config_path)
    r = ValidationResult()

    check_no_fee_curriculum(cfg, r)
    check_no_hindsight_outside_hpo(cfg, stage, r)
    check_legacy_prop_firm_block(cfg, stage, r)
    check_gates_block(cfg, r)
    check_data_manifest(cfg, stage, r)
    check_wandb_consolidation(cfg, stage, r)
    check_drift_safemode_gates(cfg, r)

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
