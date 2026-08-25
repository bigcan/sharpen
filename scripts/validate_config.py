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

# Repo root on sys.path so lazy `finrl_pro_ds.*` imports (e.g. `_resolve_base_book`'s
# `config_utils.deep_merge`) resolve when this file is invoked directly as a script
# (`python scripts/validate_config.py ...`), where sys.path[0] is `scripts/`, not the
# repo root or cwd. Mirrors the guard `execution_overlay_runner.py` already carries.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("validate_config")

PROTOCOL_DOC = "docs/protocol_v2.md"
VALID_STAGES = ("data-prep", "hpo", "l1-multiseed", "ensemble-confirm", "wf", "oos", "paper-deploy")

MANIFEST_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "schemas" / "manifest.schema.json"


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


def _resolve_base_book(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Deep-merge a ``base_book:`` parent config UNDER ``cfg`` (cfg wins).

    The execution-overlay training config (and any future inheritance-based config) carries a
    ``base_book:`` pointer instead of duplicating the base book's universe/sleeves/env/data/
    features blocks. The runner does the same merge so both see one effective config. Resolved
    relative to CWD (repo root) first, then the config's own directory. A missing/unreadable
    base_book is left to the downstream checks (e.g. missing env/data) rather than crashing here.
    """
    base_ref = cfg.get("base_book")
    if not isinstance(base_ref, str) or not base_ref:
        return cfg
    candidates = [Path(base_ref)]
    if not candidates[0].is_absolute():
        candidates.append(config_path.parent / base_ref)
    base_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if base_path is None:
        return cfg
    from finrl_pro_ds.config_utils import deep_merge

    base_cfg = load_yaml(base_path)
    return deep_merge(base_cfg, cfg)


def load_data_manifest(data_path: Path) -> dict[str, Any] | None:
    """Manifest co-located with parquet as <stem>.manifest.json."""
    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


_MANIFEST_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_manifest_schema() -> dict[str, Any] | None:
    """Load and cache the Stage 2 / 2.5 report JSON schema (Protocol v2 §2)."""
    global _MANIFEST_SCHEMA_CACHE
    if _MANIFEST_SCHEMA_CACHE is not None:
        return _MANIFEST_SCHEMA_CACHE
    if not MANIFEST_SCHEMA_PATH.exists():
        return None
    _MANIFEST_SCHEMA_CACHE = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _MANIFEST_SCHEMA_CACHE


def _candidate_report_paths(cfg: dict) -> list[Path]:
    """Surface the report files this config points at, if any.

    Live configs reference seed/ensemble reports via:
      - drift.baseline_path     (Stage 2.5 ensemble baseline for §8.2 drift)
      - agent.ensemble.bundle_path  (v2.3 atomic-swap bundle; report sits beside it)
    Adjacent seed_report.json + ensemble_report.json are checked when present.
    """
    paths: list[Path] = []
    drift = cfg.get("drift", {}) or {}
    bp = drift.get("baseline_path")
    if isinstance(bp, str) and bp:
        paths.append(Path(bp))
    agent = cfg.get("agent", {}) or {}
    ensemble = agent.get("ensemble", {}) or {}
    bundle = ensemble.get("bundle_path")
    if isinstance(bundle, str) and bundle:
        bundle_p = Path(bundle)
        for sibling in ("ensemble_report.json", "seed_report.json"):
            paths.append(bundle_p.parent / sibling)
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# ---------- universal checks (every stage) ----------

def check_no_fee_curriculum(cfg: dict, r: ValidationResult) -> None:
    """Steady-state fees from step 0; `fee_schedule` curriculum is banned.

    See decision_steady_state_fees_pattern.md (S464).

    Accepts three cost-model schemas:
      - Single-leg (GMGP1, SG-1, CMGP1):   env.taker_fee
      - Two-leg spot+perp (Funding-Arb):   environment.{spot,perp}_taker_fee_pct
      - Options-vol harvest (vrp-harvest):  env.{option_fee_pct_underlying,
        perp_taker_fee} — option fee schedule + perp delta-hedge taker fee.

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
    # Options-vol harvest: a delta-hedged short-vol book pays an option fee per leg
    # (option_fee_pct_underlying, capped at option_fee_cap_pct_premium) PLUS a perp
    # taker fee on the delta hedge. Env-type-gated so other workstreams are unaffected.
    options_legs = (
        env.get("type") == "options_vol_harvest"
        and "option_fee_pct_underlying" in env
        and "perp_taker_fee" in env
    )

    if sum([one_leg, two_leg, options_legs]) > 1:
        r.fail("Multiple cost models declared. Pick one.")
    elif one_leg:
        r.ok("steady-state fees (1-leg env.taker_fee)")
    elif two_leg:
        r.ok("steady-state fees (2-leg environment.{spot,perp}_taker_fee_pct)")
    elif options_legs:
        r.ok("steady-state fees (options: env.option_fee_pct_underlying + perp_taker_fee)")
    else:
        r.fail(
            "No cost model declared. Expected env.taker_fee (1-leg), "
            "environment.{spot_taker_fee_pct, perp_taker_fee_pct} (2-leg, funding-arb), "
            "or env.{option_fee_pct_underlying, perp_taker_fee} (options-vol harvest)."
        )


def check_max_leverage_bounds(cfg: dict, r: ValidationResult) -> None:
    """env.max_leverage must be in [0.5, 5.0] when present.

    Default 1.0 (current behavior) is only used when the key is absent — once
    declared, values outside the safe band are rejected. Lower bound rules
    out accidental zero/negative; upper bound caps the search at 5x notional
    (broker maxes go higher but we don't want HPO exploring ruin-zone leverage
    without an explicit waiver). See plan-a-new-research-lexical-sunrise.md.
    """
    env = cfg.get("env", {}) or {}
    if "max_leverage" in env:
        val = env["max_leverage"]
        try:
            lev = float(val)
        except (TypeError, ValueError):
            r.fail(f"env.max_leverage={val!r} is not numeric")
        else:
            if not (0.5 <= lev <= 5.0):
                r.fail(
                    f"env.max_leverage={lev} out of bounds [0.5, 5.0]. "
                    "Edit the value or open a waiver in the plan file."
                )
            else:
                r.ok(f"env.max_leverage = {lev}x")

    # Allocator leverage IS gross exposure (a vol-targeted book legitimately levers
    # low-vol sleeves > 1x); bound env.max_gross_exposure like max_leverage so a
    # config can't declare ruinous gross (MARGIN-CFG). env-type-gated so existing
    # single-instrument configs (which use max_leverage) are unaffected.
    if env.get("type") == "multi_asset_allocator" and "max_gross_exposure" in env:
        gval = env["max_gross_exposure"]
        try:
            gross = float(gval)
        except (TypeError, ValueError):
            r.fail(f"env.max_gross_exposure={gval!r} is not numeric")
        else:
            if not (1.0 <= gross <= 6.0):
                r.fail(
                    f"env.max_gross_exposure={gross} out of bounds [1.0, 6.0] "
                    "(MARGIN-CFG safety bound for the allocator)"
                )
            else:
                r.ok(f"env.max_gross_exposure = {gross}x (allocator)")

    # If HPO will sample leverage, the search-space bounds must also fit.
    ss = (cfg.get("hpo", {}) or {}).get("search_space", {}) or {}
    lev_ss = ss.get("max_leverage")
    if isinstance(lev_ss, dict):
        lo = lev_ss.get("low")
        hi = lev_ss.get("high")
        if lo is not None and hi is not None:
            if not (0.5 <= float(lo) and float(hi) <= 5.0 and float(lo) < float(hi)):
                r.fail(
                    f"hpo.search_space.max_leverage=[{lo}, {hi}] must lie "
                    f"within [0.5, 5.0] with low<high"
                )
            else:
                r.ok(f"max_leverage HPO range [{lo}, {hi}]")


_SLEEVE_COMBINER_MODES = ("inverse_vol", "dynamic")
_PERF_METRICS = ("sharpe", "sortino")


def check_sleeve_combiner(cfg: dict, r: ValidationResult) -> None:
    """Validate the optional ``sleeve_combiner:`` block (C1.3 — the dynamic sleeve combiner).

    Absent ⇒ no-op (inverse-vol back-compat). When present, enforce the C1.1 contract bounds
    so a config can't reach ``dynamic_sleeve_alphas`` with an out-of-band knob:
      - ``mode ∈ {inverse_vol, dynamic}``;
      - dynamic only: ``tilt_strength ∈ [0,5]`` (M-3 anti EXP-OVERFLOW),
        ``tilt_clip > 0``, ``perf_window ≥ perf_min_periods``, ``perf_metric ∈ {sharpe,sortino}``;
      - **M-1:** ``mode: dynamic`` REQUIRES ``risk_parity.target_portfolio_vol`` unset/null —
        the convex-only contract has no clean tilt for the leverage-bearing scale-to-target prior.
    """
    if "sleeve_combiner" not in cfg:
        return
    sc = cfg.get("sleeve_combiner") or {}
    if not isinstance(sc, dict):
        r.fail(f"sleeve_combiner must be a mapping, got {type(sc).__name__}")
        return
    mode = str(sc.get("mode", "inverse_vol"))
    if mode not in _SLEEVE_COMBINER_MODES:
        r.fail(f"sleeve_combiner.mode={mode!r} invalid — must be one of {_SLEEVE_COMBINER_MODES}")
        return
    if mode == "inverse_vol":
        r.ok("sleeve_combiner.mode=inverse_vol (static, back-compat)")
        return

    # --- mode: dynamic ---
    ok = True
    try:
        lam = float(sc.get("tilt_strength", 0.0))
    except (TypeError, ValueError):
        r.fail(f"sleeve_combiner.tilt_strength={sc.get('tilt_strength')!r} is not numeric")
        ok = False
    else:
        if not (0.0 <= lam <= 5.0):
            r.fail(f"sleeve_combiner.tilt_strength={lam} out of bounds [0, 5] (M-3 anti-overflow)")
            ok = False
    try:
        clip = float(sc.get("tilt_clip", 1.5))
    except (TypeError, ValueError):
        r.fail(f"sleeve_combiner.tilt_clip={sc.get('tilt_clip')!r} is not numeric")
        ok = False
    else:
        if clip <= 0:
            r.fail(f"sleeve_combiner.tilt_clip={clip} must be > 0")
            ok = False
    pw = int(sc.get("perf_window", 126))
    pmp = int(sc.get("perf_min_periods", 63))
    if pw < pmp:
        r.fail(f"sleeve_combiner.perf_window={pw} must be >= perf_min_periods={pmp}")
        ok = False
    pm = str(sc.get("perf_metric", "sharpe"))
    if pm not in _PERF_METRICS:
        r.fail(f"sleeve_combiner.perf_metric={pm!r} invalid — must be one of {_PERF_METRICS}")
        ok = False
    # M-1: convex-only — reject a target_portfolio_vol overlay under dynamic mode.
    tpv = (cfg.get("risk_parity", {}) or {}).get("target_portfolio_vol", None)
    if tpv is not None:
        r.fail(
            f"sleeve_combiner.mode=dynamic requires risk_parity.target_portfolio_vol=null "
            f"(got {tpv}) — the dynamic combiner is convex-only (M-1)."
        )
        ok = False
    if ok:
        r.ok(f"sleeve_combiner.mode=dynamic (λ={lam}, perf_window={pw}, metric={pm})")


_PROP_FIRM_INTENTIONAL_MARKER = "DO NOT MIGRATE to env.risk:"


def _config_has_intentional_legacy_marker(config_path: Path | None) -> bool:
    """Return True if the source YAML contains the intentional-legacy marker.

    Three configs keep ``env.prop_firm:`` by design (V7 vs RS A/B controls
    for the Q2 critic-loss parity gate and the GMGP1 OANDA fold-07 ensemble
    A/B). They carry an ``# INTENTIONAL: ... DO NOT MIGRATE to env.risk:``
    header comment so the validator can recognize and skip them. See
    ``project_prop_firm_config_migration_todo.md``.
    """
    if config_path is None or not config_path.exists():
        return False
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _PROP_FIRM_INTENTIONAL_MARKER in text


def check_legacy_prop_firm_block(
    cfg: dict,
    stage: str,
    r: ValidationResult,
    config_path: Path | None = None,
) -> None:
    """FAIL configs still using the legacy ``env.prop_firm:`` block.

    Post S495-cont the prop-firm decoupling split responsibilities:
    - Training-side DD shaping lives under ``env.risk:`` +
      :class:`finrl_pro_ds.envs.risk_shaping_wrapper.RiskShapingWrapper`.
    - Live-side profit-target tracking lives under ``challenge:`` +
      ``ChallengeStateMachine`` (live engine).

    Escalated S504 from WARN→FAIL at all stages now that the config-tree
    migration is done (33/36 configs migrated; 3 intentional A/B controls
    retained with the ``DO NOT MIGRATE to env.risk:`` marker comment).
    Configs carrying that marker are skipped silently. Configs without the
    marker FAIL at every stage. See
    ``.agent/artifacts/prop_firm_decoupling_architecture.md``.
    """
    env = cfg.get("env", {}) or {}
    if "prop_firm" not in env:
        return
    if _config_has_intentional_legacy_marker(config_path):
        r.ok("env.prop_firm: present — intentional-legacy marker honored")
        return
    msg = (
        "env.prop_firm: is removed — migrate to env.risk: + top-level "
        "challenge: block. The PropFirmWrapperV7 adapter is retired in Step 6. "
        "Intentional A/B controls must include "
        f"'# INTENTIONAL: ... {_PROP_FIRM_INTENTIONAL_MARKER}' header. "
        "(prop_firm_decoupling_architecture.md)"
    )
    r.fail(msg)


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
    """All numeric gates must come from a `gates` block, not be hardcoded.

    A workstream may declare gates inline (`gates:`) *or* delegate them to a
    standalone `<ws>.gates.yaml` referenced by `ensemble.gates_file` — the
    documented per-workstream pattern. Resolve the same overlay the runtime
    eval helpers read (:func:`_load_ensemble_gates_overlay`) before concluding
    the block is absent, matching the other call sites of that helper.
    """
    inline = cfg.get("gates")
    if inline is not None and not isinstance(inline, dict):
        r.fail(
            f"`gates:` must be a mapping, got {type(inline).__name__}. "
            f"Per-workstream gates required by Protocol v2 §4. See {PROTOCOL_DOC}."
        )
        return

    gates = _load_ensemble_gates_overlay(cfg)
    gates_file = (cfg.get("ensemble") or {}).get("gates_file")
    if not gates:
        detail = (
            f" `ensemble.gates_file: {gates_file}` is declared but resolved to no "
            "`gates:` keys (file missing, unreadable, or empty)."
            if gates_file
            else ""
        )
        r.fail(
            "Missing `gates:` block. Per-workstream gates required by Protocol v2 §4."
            f"{detail} See {PROTOCOL_DOC}."
        )
        return

    if inline and gates_file:
        source = f"inline + {gates_file}"
    elif gates_file:
        source = gates_file
    else:
        source = "inline"
    r.ok(f"gates block present ({len(gates)} keys, from {source})")


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

    # yfinance daily proxies (cross-asset allocator dev) + curated-futures both
    # fetch + clean (clean_ohlcv) at runtime via cross_asset_loader (ADR-5); like
    # ccxt, no pre-built parquet ships. Skip file/manifest gating, require frequency.
    source = str(data.get("source") or "")
    if source.startswith("yfinance"):
        if not data.get("frequency"):
            r.fail("data.source=yfinance* but data.frequency missing")
        else:
            r.ok(
                f"data source={source} (runtime fetch via cross_asset_loader, "
                f"skip manifest; stage={stage})"
            )
        return

    # Deribit public API (options-vol harvest): DVOL + perp + funding fetched and
    # cached at runtime via deribit_options_loader (DATA-CLEAN + manifest.json on the
    # price legs at fetch time). Like ccxt/yfinance, no pre-built parquet ships.
    if source.startswith("deribit"):
        if not data.get("frequency"):
            r.fail("data.source=deribit* but data.frequency missing")
        else:
            r.ok(
                f"data source={source} (runtime fetch via deribit_options_loader, "
                f"skip manifest; stage={stage})"
            )
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

    # ADR-4 (cross-asset allocator) / ADR-5 (options-vol harvester): a hedged,
    # multi-leg or portfolio book optimizes a PORTFOLIO risk-adjusted return
    # (Sharpe/Sortino), not single-instrument profit_factor. BUG-01 is a
    # scalping-era rule — a daily diversified / delta-hedged-vol book has PF ~1.1
    # (no discriminating power); crypto already uses objective: sortino. The
    # allow-list is env-type-gated so single-instrument workstreams remain strictly
    # bound to profit_factor.
    env_type = (cfg.get("env", {}) or {}).get("type")
    objective = hpo.get("objective")
    risk_adjusted_objectives = ("sharpe", "sortino")
    # ADR-5: options_vol_harvest is a delta-hedged short-vol book — risk-adjusted
    # return is the right objective (PF is a secondary gate), same as the allocator.
    risk_adjusted_env_types = ("multi_asset_allocator", "options_vol_harvest")
    if env_type in risk_adjusted_env_types:
        if objective not in risk_adjusted_objectives:
            r.fail(
                f"hpo.objective={objective!r} — {env_type} must use one of "
                f"{risk_adjusted_objectives} (ADR-4/ADR-5; portfolio Sharpe/Sortino, not BUG-01 PF)"
            )
        else:
            r.ok(f"HPO objective = {objective} ({env_type}; BUG-01 extended per ADR-4/ADR-5)")
    elif objective != "profit_factor":
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

    # Training-budget multiplicity (Protocol v2.5.1 §3.5) on the per-trial budget.
    _check_multiplicity(cfg, hpo.get("steps_per_trial"), "HPO per-trial", r)


def _is_prop_firm(cfg: dict) -> bool:
    """Workstream is paper-or-live-capital per Protocol v2.2 §8 scope rules.

    Tag set mirrors the one declared in decision_protocol_v22_rlops_drift_safemode.md:
    any of `prop-firm`, `FTMO`, `Velotrade` (case-insensitive match on
    wandb.tags).
    """
    tags = [str(t).lower() for t in (cfg.get("wandb", {}).get("tags") or [])]
    return any(t in tags for t in ("prop-firm", "propfirm", "ftmo", "velotrade"))


def _base_scale_minutes(cfg: dict) -> int | None:
    """Finest OHLCV scale in minutes — one env step is one base bar."""
    feats = cfg.get("features", {}) or {}
    scales = feats.get("scales") or (cfg.get("env", {}) or {}).get("scales")
    if not scales:
        return None
    try:
        base = min(int(s) for s in scales)
    except (TypeError, ValueError):
        return None
    return base if base > 0 else None


def _train_window_days(cfg: dict) -> float | None:
    """Calendar length of the train window, from explicit dates or splitter months."""
    data = cfg.get("data", {}) or {}
    ts, te = data.get("train_start_date"), data.get("train_end_date")
    if ts and te:
        try:
            return float((datetime.fromisoformat(str(te)) - datetime.fromisoformat(str(ts))).days)
        except ValueError:
            pass
    months = (cfg.get("splitter", {}) or {}).get("train_months")
    if months:
        try:
            return float(months) * 30.44
        except (TypeError, ValueError):
            return None
    return None


def _training_budget_multiplicity(cfg: dict, total_steps) -> tuple[float, float, str] | None:
    """``steps / bars_in_train_window`` for 24/7 crypto (Protocol v2.5.1 §3.5,
    decision_training_budget_multiplicity_rule). Returns ``(mult, bars, basis)``
    or ``None`` when the trading calendar can't be inferred reliably (session-bound
    FX/futures) — we skip rather than emit a wrong multiplicity."""
    try:
        steps = float(total_steps)
    except (TypeError, ValueError):
        return None
    if steps <= 0:
        return None
    # Bar-stepped envs with an explicit walk_forward bar split declare the train
    # window directly — no calendar inference needed. Without this branch the
    # allocator's 500k/1260-bar = 397x Stage-1 overfit slipped validation
    # entirely (S553-cont-34; the WF was the proof).
    env_type = str((cfg.get("env", {}) or {}).get("type") or "").lower()
    if env_type == "multi_asset_allocator":
        try:
            bars = float((cfg.get("walk_forward", {}) or {}).get("train_bars"))
        except (TypeError, ValueError):
            return None
        if bars <= 0:
            return None
        freq = str((cfg.get("data", {}) or {}).get("frequency") or "bars")
        return steps / bars, bars, f"walk_forward.train_bars @ {freq}"
    # 24/7 calendar is only certain for crypto; session-bound assets need a
    # calendar we don't model here, so skip rather than guess.
    feats = cfg.get("features", {}) or {}
    asset_class = str(
        feats.get("asset_class") or (cfg.get("data", {}) or {}).get("asset_class") or ""
    ).lower()
    if asset_class != "crypto":
        return None
    base_min = _base_scale_minutes(cfg)
    days = _train_window_days(cfg)
    if not base_min or not days or days <= 0:
        return None
    bars = days * (1440.0 / base_min)
    if bars <= 0:
        return None
    return steps / bars, bars, f"{days:.0f}d x {1440.0 / base_min:.0f} bars/day @ {base_min}m"


def _check_multiplicity(cfg: dict, total_steps, label: str, r: ValidationResult) -> None:
    """Training-budget multiplicity gate. >50x = REJECT overfit cliff; <15x =
    under-trained WARN (acceptable only when L1 N>=10 is the real filter)."""
    m = _training_budget_multiplicity(cfg, total_steps)
    if m is None:
        return
    mult, bars, basis = m
    msg = f"{label} budget multiplicity {mult:.1f}x ({basis}, {bars:.0f} train bars)"
    if mult > 50.0:
        r.fail(f"{msg} — exceeds 50x REJECT cliff (decision_training_budget_multiplicity_rule §3.5)")
    elif mult > 40.0:
        r.warn(f"{msg} — above [15,40] productive band, approaching 50x reject cliff")
    elif mult >= 15.0:
        r.ok(f"{msg} — within [15,40] productive band")
    else:
        r.warn(
            f"{msg} — below [15,40] band (under-trained); acceptable per §3.5 only when "
            "the L1 N>=10 multiseed is the real validation filter"
        )


def check_execution_cost_realism(cfg: dict, stage: str, r: ValidationResult) -> None:
    """Prop-firm configs that bear execution must model slippage. Selecting,
    grading, or deploying a policy under slippage=0 understates live cost and is
    the documented sg1-btc sim->live gap (sg1_btc_sim_live_gap.md; audit P3/P4/P10).
    WARN (not FAIL) so historical configs still validate, but the gap is surfaced."""
    if stage not in ("l1-multiseed", "wf", "oos", "paper-deploy"):
        return
    if not _is_prop_firm(cfg):
        return
    env = cfg.get("env", {}) or {}
    try:
        slip = float(env.get("slippage_base_bps", 0.0) or 0.0)
    except (TypeError, ValueError):
        slip = 0.0
    if slip > 0:
        r.ok(f"execution cost: env.slippage_base_bps={slip} modeled (stage={stage})")
    else:
        r.warn(
            f"execution cost: env.slippage_base_bps unset/0 on a prop-firm {stage} config "
            "— policy trained/selected/graded net of fee only, understating live cost "
            "(sg1_btc_sim_live_gap.md). Set to the venue marketable-limit cross."
        )


def check_execution_overlay_gates(cfg: dict, r: ValidationResult) -> None:
    """RL execution-overlay deploy-gate presence + sanity (ADR-8, exec-overlay step 5).

    Fires ONLY when the config declares an ``execution_overlay`` block (the overlay training
    config; a no-op for every other workstream — env-type-gated like
    :func:`check_max_leverage_bounds`). The overlay shapes only the *trade path* of the FIXED
    linear target and can never manufacture a directional edge, so its single deploy gate is
    the ``beat-linear`` discipline: net-of-impact implementation-shortfall uplift vs a tuned
    TWAP baseline, robust across folds + a cost-stress variant (else ``ship_snap_executor``).

    Validates the effective gates (inline ``gates:`` merged with the standalone
    ``ensemble.gates_file`` overlay, via :func:`_load_ensemble_gates_overlay`) — both the
    ``execution_beats_baseline`` decision gate and the redefined ``overlay_parity`` gate
    (completion is hard; intra-horizon lag is sanity). Thresholds live in the gates file
    (CLAUDE invariant); this checks they are present and well-formed so the runner can never
    read a malformed or absent gate. See ``.agent/artifacts/execution_overlay_architecture.md``.
    """
    if "execution_overlay" not in cfg:
        return  # not an execution-overlay config — no-op

    gates = _load_ensemble_gates_overlay(cfg)
    ebb = gates.get("execution_beats_baseline")
    if not isinstance(ebb, dict):
        r.fail(
            "execution_overlay config missing gates.execution_beats_baseline — the deploy "
            "gate (net-IS uplift vs tuned TWAP). Copy from configs/execution_overlay.gates.yaml "
            "(min_uplift_bps / cost_stress_factor / cost_stress_min_uplift_bps / wf_folds / "
            "robust_folds_required / else)."
        )
        return

    def _pos_num(block: dict, key: str, *, strict_pos: bool = False) -> float | None:
        v = block.get(key)
        if v is None:
            r.fail(f"gates.execution_beats_baseline.{key} not set (exec-overlay deploy gate)")
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            r.fail(f"gates.execution_beats_baseline.{key}={v!r} is not numeric")
            return None
        if strict_pos and f <= 0.0:
            r.fail(f"gates.execution_beats_baseline.{key}={f} must be > 0")
            return None
        return f

    min_uplift = _pos_num(ebb, "min_uplift_bps")
    stress_uplift = _pos_num(ebb, "cost_stress_min_uplift_bps")
    factor = _pos_num(ebb, "cost_stress_factor")
    if factor is not None and factor <= 1.0:
        r.fail(
            f"gates.execution_beats_baseline.cost_stress_factor={factor} must be > 1.0 "
            "(a stress variant scales impact coefficients UP; <=1.0 is not a stress)"
        )
    # Stress floor should sit at or below the primary floor (a harder bar at stress is
    # incoherent — the stressed uplift can only be <= the primary).
    if min_uplift is not None and stress_uplift is not None and stress_uplift > min_uplift:
        r.fail(
            f"gates.execution_beats_baseline.cost_stress_min_uplift_bps={stress_uplift} must be "
            f"<= min_uplift_bps={min_uplift} (the stressed uplift cannot exceed the primary)"
        )

    wf_folds = ebb.get("wf_folds")
    robust = ebb.get("robust_folds_required")
    folds_ok = True
    for key, val in (("wf_folds", wf_folds), ("robust_folds_required", robust)):
        if not (isinstance(val, int) and not isinstance(val, bool) and val >= 1):
            r.fail(f"gates.execution_beats_baseline.{key}={val!r} must be an int >= 1")
            folds_ok = False
    if folds_ok and robust > wf_folds:
        r.fail(
            f"gates.execution_beats_baseline.robust_folds_required={robust} must be "
            f"<= wf_folds={wf_folds} (cannot require more positive folds than exist)"
        )

    else_action = ebb.get("else")
    if else_action != "ship_snap_executor":
        r.fail(
            f"gates.execution_beats_baseline.else={else_action!r} must be 'ship_snap_executor' "
            "— the safe fail-open default (the snap _replay owns rung-1 parity-0)"
        )

    parity = gates.get("overlay_parity")
    if not isinstance(parity, dict):
        r.fail(
            "execution_overlay config missing gates.overlay_parity (ADR-8 redefined parity: "
            "max_completion_l1_drift hard + max_intra_horizon_drift sanity)"
        )
    else:
        for key in ("max_completion_l1_drift", "max_intra_horizon_drift"):
            v = parity.get(key)
            if v is None:
                r.fail(f"gates.overlay_parity.{key} not set (exec-overlay parity gate)")
            else:
                try:
                    if float(v) <= 0.0:
                        r.fail(f"gates.overlay_parity.{key}={v} must be > 0")
                except (TypeError, ValueError):
                    r.fail(f"gates.overlay_parity.{key}={v!r} is not numeric")

    if not r.failures:
        r.ok(
            f"execution-overlay gates OK (min_uplift={min_uplift}bps, "
            f"stress x{factor} >= {stress_uplift}bps, {robust}/{wf_folds} folds)"
        )


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
# v2.6 (S551-cont-4): feature_variance_veto sub-block. WARN-only for one
# release cycle; once all 5 active live configs declare it, promote to
# fail-loud for prop-firm tier.
_V26_DRIFT_OPTIONAL_KEYS = ("feature_variance_veto",)
_V26_VETO_REQUIRED_SUBKEYS = ("enabled", "scale", "max_veto_frac")


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

    # v2.6 (S551-cont-4): feature_variance_veto block — WARN-only for one
    # release cycle, fail-loud after rollout to all 5 live configs.
    veto = drift_gates.get("feature_variance_veto")
    if veto is None:
        msg = (
            "gates.drift.feature_variance_veto missing — v2.6 ActionDriftTracker "
            "amendment recommends declaring {enabled, scale, max_veto_frac} "
            "(see decision_drift_two_failure_modes_s551_cont_3). Defaults "
            "(enabled=true, scale='base', max_veto_frac=0.50) apply when absent."
        )
        r.warn(msg)
    else:
        if not isinstance(veto, dict):
            r.fail(
                f"gates.drift.feature_variance_veto must be a mapping; "
                f"got {type(veto).__name__}"
            )
        else:
            missing_subkeys = [k for k in _V26_VETO_REQUIRED_SUBKEYS if k not in veto]
            if missing_subkeys:
                r.warn(
                    f"gates.drift.feature_variance_veto missing v2.6 sub-key(s): "
                    f"{missing_subkeys} — defaults will apply"
                )
            mvf = veto.get("max_veto_frac")
            if mvf is not None:
                try:
                    mvf_f = float(mvf)
                except (TypeError, ValueError):
                    r.fail(
                        f"gates.drift.feature_variance_veto.max_veto_frac must be "
                        f"a float; got {mvf!r}"
                    )
                else:
                    if not 0.0 < mvf_f < 1.0:
                        r.fail(
                            f"gates.drift.feature_variance_veto.max_veto_frac="
                            f"{mvf_f} must be in (0, 1)"
                        )
            scale = veto.get("scale")
            if scale is not None and not (
                scale in ("base", "all") or isinstance(scale, int)
            ):
                r.fail(
                    f"gates.drift.feature_variance_veto.scale must be 'base', "
                    f"'all', or an int (minutes); got {scale!r}"
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

    # Training-budget multiplicity (Protocol v2.5.1 §3.5) on the per-seed budget.
    _check_multiplicity(cfg, (cfg.get("training", {}) or {}).get("total_timesteps"),
                        "L1 per-seed", r)


def _load_ensemble_gates_overlay(cfg: dict) -> dict:
    """Mirror `sg1_xauusd_ensemble_eval._load_gates_with_overlay` for validation.

    The Stage 2.5 helper merges `ensemble.gates_file` over the inline `gates:`
    block at runtime (S498-cont fix, commit `b007fb2d`). This validator must
    see the same effective gates so prop-firm configs that delegate bootstrap
    thresholds to a standalone yaml file don't FAIL validation while passing
    at runtime. Returns a NEW dict; caller's cfg is not mutated.
    """
    gates = dict((cfg.get("gates") or {}))
    gates_file = (cfg.get("ensemble") or {}).get("gates_file")
    if not gates_file:
        return gates
    gates_path = Path(gates_file)
    if not gates_path.is_absolute():
        gates_path = Path.cwd() / gates_path
    if not gates_path.exists():
        return gates
    try:
        with open(gates_path, encoding="utf-8") as f:
            standalone = yaml.safe_load(f) or {}
    except Exception:
        return gates
    standalone_gates = (standalone.get("gates") or {})
    if standalone_gates:
        gates.update(standalone_gates)
    return gates


def check_ensemble_confirm(cfg: dict, r: ValidationResult) -> None:
    """Stage 2.5 gates (Protocol v2.5 §4, S526 bootstrap-primary refinement).

    Prop-firm / live-capital workstreams MUST pass Stage 2.5 before Stage 3.
    Non-prop-firm workstreams may opt in via `wandb.tags` or gates block, but
    the stage is advisory for them (warn, do not fail).

    Protocol v2.5 (S526): block-bootstrap thresholds are PRIMARY for prop-firm
    workstreams. The legacy `ensemble_uplift_min` point gate is demoted to an
    audit-only metric (still encouraged for diff against historical decisions
    but no longer fails validation when missing).
    """
    gates = _load_ensemble_gates_overlay(cfg)
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

    # --- Rules 1-2 (v2.5): bootstrap PROMOTE thresholds REQUIRED for prop-firm.
    bootstrap_keys_hint = (
        "Copy the four `ensemble_bootstrap_p_pf_promote` / "
        "`ensemble_bootstrap_p_mdd_promote` / `ensemble_bootstrap_p_pf_ambiguous` / "
        "`ensemble_bootstrap_resamples` keys from a sister "
        "<workstream>_ensemble.gates.yaml (e.g. configs/sg1_xauusd_ensemble.gates.yaml). "
        "Defaults: 0.90 / 0.90 / 0.75 / 10000."
    )
    p_pf_promote = gates.get("ensemble_bootstrap_p_pf_promote")
    if p_pf_promote is None:
        msg = (
            "gates.ensemble_bootstrap_p_pf_promote not set — Protocol v2.5 "
            "Stage 2.5 PRIMARY gate. " + bootstrap_keys_hint
        )
        r.fail(msg) if prop_firm else r.warn(msg)

    p_mdd_promote = gates.get("ensemble_bootstrap_p_mdd_promote")
    if p_mdd_promote is None:
        msg = (
            "gates.ensemble_bootstrap_p_mdd_promote not set — Protocol v2.5 "
            "Stage 2.5 PRIMARY gate. " + bootstrap_keys_hint
        )
        r.fail(msg) if prop_firm else r.warn(msg)

    # --- Rule 3 (v2.5): resamples is recommended; warn (not fail) when missing.
    if "ensemble_bootstrap_resamples" not in gates:
        r.warn(
            "gates.ensemble_bootstrap_resamples not set — defaulting to 10000. "
            "Add explicitly to lock the bootstrap budget."
        )

    # --- Rule 6 (v2.5): sanity-bound check on p_pf_promote.
    if p_pf_promote is not None:
        try:
            p_pf_promote_f = float(p_pf_promote)
            if not (0.80 <= p_pf_promote_f <= 0.99):
                r.warn(
                    f"gates.ensemble_bootstrap_p_pf_promote={p_pf_promote_f} "
                    "outside sanity bound [0.80, 0.99]. Lower values weaken the "
                    "noise-aware promote criterion; higher values rarely fire. "
                    "Per-asset-class overrides below 0.80 require a decision memo."
                )
        except (TypeError, ValueError):
            r.fail(
                f"gates.ensemble_bootstrap_p_pf_promote={p_pf_promote!r} is not numeric"
            )

    # --- Rule 7 (v2.5): ordering invariant ambiguous < promote.
    p_pf_ambiguous = gates.get("ensemble_bootstrap_p_pf_ambiguous")
    if p_pf_ambiguous is not None and p_pf_promote is not None:
        try:
            if float(p_pf_ambiguous) >= float(p_pf_promote):
                r.fail(
                    f"gates.ensemble_bootstrap_p_pf_ambiguous={p_pf_ambiguous} "
                    f">= ensemble_bootstrap_p_pf_promote={p_pf_promote} — ambiguous "
                    "floor must be strictly below promote threshold (otherwise the "
                    "AMBIGUOUS_BOOT band is empty)."
                )
        except (TypeError, ValueError):
            r.fail(
                f"gates.ensemble_bootstrap_p_pf_ambiguous={p_pf_ambiguous!r} is not numeric"
            )

    # --- Rule 4-5 (v2.5): legacy uplift demoted to audit-only — WARN-not-FAIL.
    if "ensemble_uplift_min" not in gates:
        if prop_firm:
            r.warn(
                "gates.ensemble_uplift_min not set — demoted to audit/sanity "
                "in Protocol v2.5 (bootstrap is primary), but recommended for "
                "diff against historical v2.1/v2.3 decisions. Default 1.10."
            )
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
        r.ok(f"ensemble uplift gate (audit-only) = {uplift_min}×")

    if p_pf_promote is not None and p_mdd_promote is not None:
        r.ok(
            f"ensemble bootstrap gates = P(PF)≥{p_pf_promote}, "
            f"P(MDD)≥{p_mdd_promote}"
        )


def _parse_protocol_version(version: str) -> tuple[int, ...] | None:
    """Parse a dotted protocol-version string into an int tuple for ordering.

    Returns ``None`` when the string isn't a clean dotted-int version (e.g. a
    git sha or ``"latest"``) so callers can treat an unparseable version as
    "not satisfying" rather than crashing.
    """
    parts = str(version).strip().split(".")
    try:
        return tuple(int(p) for p in parts)
    except (TypeError, ValueError):
        return None


def _protocol_at_least(declared: str, target: str) -> bool:
    """True when ``declared`` >= ``target`` by dotted-int version ordering.

    Zero-pads to equal length so ``"2.6" == "2.6.0"``. An unparseable
    ``declared`` returns False (treated as pre-target / exempt), preserving the
    pre-N4 behavior where only recognized versions opted into enforcement.

    N4: replaces the exact-match ``protocol_version in ("2.6",)`` gate, which
    silently self-disabled the sensitivity audit once configs bumped to "2.7"+.
    """
    d = _parse_protocol_version(declared)
    t = _parse_protocol_version(target)
    if d is None or t is None:
        return False
    n = max(len(d), len(t))
    d += (0,) * (n - len(d))
    t += (0,) * (n - len(t))
    return d >= t


def check_sensitivity_audit(cfg: dict, r: ValidationResult) -> None:
    """Protocol v2.6 Stage 2.5-R Sensitivity Audit gate validation (S553).

    Validates that the L1 multiseed config declares the gate keys required for
    the post-PROMOTE sensitivity audit (`scripts/stage_2_5_r_sensitivity_audit.py`,
    landing in v2.7-A C4). The audit runs a 3×3 (deadband_threshold ×
    max_leverage) sweep on the PROMOTE'd ensemble/solo and flags edge-fragile
    policies via the ratio gate `pf_inner_min / pf_center >= floor`.

    Phase α (S553+): legacy configs (no `protocol_version` or `<= "2.5"`) are
    exempt — this check is a no-op. Configs that opt in via `protocol_version:
    "2.6"` (or any higher version) get full enforcement.

    Phase β (post-backfill): operator bumps the 5 active live workstreams to
    `protocol_version: "2.6"` AND `sensitivity_audit_required: true` after
    threshold calibration. Validator FAILs prop-firm configs missing the keys
    at that point.

    See `.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md`
    (ADR-3, ADR-5) and `mc_robustness_methods_research.md` (Method #5).
    """
    # Phase α back-compat: enforce on v2.6 opt-in and every later protocol
    # version (N4: version-tuple compare, not exact-match — see _protocol_at_least).
    protocol_version = str(cfg.get("protocol_version", "2.5"))
    if not _protocol_at_least(protocol_version, "2.6"):
        return

    gates = _load_ensemble_gates_overlay(cfg)
    prop_firm = _is_prop_firm(cfg)

    sensitivity_hint = (
        "Copy from a sister <workstream>_ensemble.gates.yaml or use Phase α "
        "defaults: edge_stability_pf_ratio_floor: 0.70, "
        "sensitivity_deadband_grid: [0.20, 0.25, 0.30], "
        "sensitivity_max_leverage_mults: [0.5, 1.0, 1.5], "
        "sensitivity_deployable_max_leverage_cap: <deployed env.max_leverage>, "
        "sensitivity_audit_required_min_deployable_neighbors: 4, "
        "sensitivity_audit_required: false (Phase α) → true (Phase β), "
        "pf_xcheck_report_only: true (Phase α) → false (enforce, post-calibration), "
        "pf_xcheck_divergence_halt: 0.30."
    )

    floor = gates.get("edge_stability_pf_ratio_floor")
    if floor is None:
        msg = f"gates.edge_stability_pf_ratio_floor not set — v2.6 PRIMARY gate. {sensitivity_hint}"
        r.fail(msg) if prop_firm else r.warn(msg)
    else:
        try:
            floor_f = float(floor)
            if not (0.50 <= floor_f <= 0.95):
                r.warn(
                    f"gates.edge_stability_pf_ratio_floor={floor_f} outside sanity "
                    "bound [0.50, 0.95]. Below 0.50 weakens the audit; above 0.95 "
                    "rarely fires."
                )
        except (TypeError, ValueError):
            r.fail(
                f"gates.edge_stability_pf_ratio_floor={floor!r} is not numeric"
            )

    grid_db = gates.get("sensitivity_deadband_grid")
    if not isinstance(grid_db, list) or len(grid_db) != 3:
        msg = (
            "gates.sensitivity_deadband_grid must be a 3-element list "
            f"(e.g. [0.20, 0.25, 0.30]); got {grid_db!r}"
        )
        r.fail(msg) if prop_firm else r.warn(msg)

    grid_ml = gates.get("sensitivity_max_leverage_mults")
    if not isinstance(grid_ml, list) or len(grid_ml) != 3:
        msg = (
            "gates.sensitivity_max_leverage_mults must be a 3-element list "
            f"(e.g. [0.5, 1.0, 1.5]); got {grid_ml!r}"
        )
        r.fail(msg) if prop_firm else r.warn(msg)

    # SENS-1 invariant: center cell must match deployed config — non-centered
    # grids invalidate the edge-stability semantics.
    deployed_deadband = (cfg.get("env") or {}).get("deadband_threshold", 0.25)
    if isinstance(grid_db, list) and len(grid_db) == 3:
        try:
            if abs(float(grid_db[1]) - float(deployed_deadband)) > 1e-9:
                r.fail(
                    f"sensitivity_deadband_grid center [{grid_db[1]}] must equal "
                    f"deployed env.deadband_threshold ({deployed_deadband}); "
                    "non-centered grids invalidate the edge-stability semantics "
                    "(SENS-1 invariant)."
                )
        except (TypeError, ValueError):
            r.fail(
                f"sensitivity_deadband_grid contains non-numeric values: {grid_db!r}"
            )

    if isinstance(grid_ml, list) and len(grid_ml) == 3:
        try:
            if abs(float(grid_ml[1]) - 1.0) > 1e-9:
                r.fail(
                    f"sensitivity_max_leverage_mults center [{grid_ml[1]}] must "
                    "equal 1.0 (center = deployed max_leverage; mults are relative)."
                )
        except (TypeError, ValueError):
            r.fail(
                f"sensitivity_max_leverage_mults contains non-numeric values: {grid_ml!r}"
            )

    cap = gates.get("sensitivity_deployable_max_leverage_cap")
    if cap is not None:
        try:
            cap_f = float(cap)
            if cap_f <= 0:
                r.fail(
                    f"sensitivity_deployable_max_leverage_cap={cap_f} must be positive"
                )
        except (TypeError, ValueError):
            r.fail(
                f"sensitivity_deployable_max_leverage_cap={cap!r} is not numeric"
            )

    min_neighbors = gates.get(
        "sensitivity_audit_required_min_deployable_neighbors", 4,
    )
    try:
        min_neighbors_i = int(min_neighbors)
        if not (1 <= min_neighbors_i <= 8):
            r.warn(
                f"sensitivity_audit_required_min_deployable_neighbors="
                f"{min_neighbors_i} outside [1, 8]. Below 1 is meaningless; "
                "above 8 is impossible in a 3×3 grid (8 neighbors)."
            )
    except (TypeError, ValueError):
        r.fail(
            f"sensitivity_audit_required_min_deployable_neighbors="
            f"{min_neighbors!r} is not integer"
        )

    required = gates.get("sensitivity_audit_required")
    if required is None:
        msg = (
            "gates.sensitivity_audit_required not set — v2.6 explicit "
            "declaration required. Set false during Phase α calibration; "
            "flip to true once threshold is locked in Phase β."
        )
        r.fail(msg) if prop_firm else r.warn(msg)
    elif not isinstance(required, bool):
        r.fail(
            f"gates.sensitivity_audit_required={required!r} must be boolean "
            "(true/false)"
        )

    # N2 (ADR-N5, Protocol v2.7-A): PF-XCHECK report-only mode + divergence
    # threshold are gates-driven. report_only=true surfaces the (H+L)/2-vs-close
    # PF divergence without halting (Phase-α calibration); flip false to enforce
    # only after locking the threshold on the retrained policy's observed
    # divergence. Promoted out of code constants per "no hardcoded gate
    # thresholds" (CLAUDE.md).
    report_only = gates.get("pf_xcheck_report_only")
    if report_only is None:
        # Optional under Phase α (code default = report-only). Once the audit is
        # required (Phase β), the report-only -> enforce decision must be a
        # deliberate, recorded config value rather than a silent code default.
        if isinstance(required, bool) and required:
            msg = (
                "gates.pf_xcheck_report_only not set while "
                "sensitivity_audit_required=true — the report-only -> enforce "
                "decision must be explicit in Phase β. Set false only after "
                "locking pf_xcheck_divergence_halt against observed divergence."
            )
            r.fail(msg) if prop_firm else r.warn(msg)
    elif not isinstance(report_only, bool):
        r.fail(
            f"gates.pf_xcheck_report_only={report_only!r} must be boolean "
            "(true/false)"
        )

    halt_div = gates.get("pf_xcheck_divergence_halt")
    if halt_div is not None:
        try:
            halt_div_f = float(halt_div)
            if not (0.0 < halt_div_f <= 1.0):
                r.warn(
                    f"gates.pf_xcheck_divergence_halt={halt_div_f} outside sane "
                    "bound (0, 1]. The CLAUDE.md PF-XCHECK invariant is 0.30; "
                    "deviate only with a logged calibration rationale."
                )
        except (TypeError, ValueError):
            r.fail(
                f"gates.pf_xcheck_divergence_halt={halt_div!r} is not numeric"
            )

    if (
        isinstance(required, bool)
        and required
        and prop_firm
        and not r.failures
    ):
        r.ok(
            f"sensitivity audit enforcement active "
            f"(floor={floor}, grid={grid_db}×{grid_ml})"
        )


def _check_stress_gate_keys(gates: dict, r: ValidationResult) -> None:
    """Validate the X3 fixed-lot stress gate keys when present (Protocol v2 Stage 3;
    audit P7-03/P7-07). Safe code defaults exist (0.5pp / 1.0x) so absence is not an
    error — this only sanity-checks declared values."""
    ddb = gates.get("stress_dd_buffer_pp")
    if ddb is not None:
        try:
            v = float(ddb)
            if not (0.0 <= v <= 10.0):
                r.warn(
                    f"gates.stress_dd_buffer_pp={v} outside sane bound [0, 10] pp "
                    "(headroom of worst fixed-lot DD from the trailing cap)"
                )
        except (TypeError, ValueError):
            r.fail(f"gates.stress_dd_buffer_pp={ddb!r} is not numeric")
    lev = gates.get("stress_leverage_max")
    if lev is not None:
        try:
            if float(lev) <= 0:
                r.fail(f"gates.stress_leverage_max={lev} must be positive")
        except (TypeError, ValueError):
            r.fail(f"gates.stress_leverage_max={lev!r} is not numeric")


def check_wf(cfg: dict, r: ValidationResult) -> None:
    gates = cfg.get("gates", {})
    if gates.get("wf_windows", 4) < 4:
        r.fail("gates.wf_windows < 4 (Protocol v2 §4 stage 3)")
    # X3 stress gate keys live in the standalone <ws>_ensemble.gates.yaml overlay
    # (same source the WF eval reads), with inline gates taking precedence.
    overlay = _load_ensemble_gates_overlay(cfg) or {}
    _check_stress_gate_keys({**overlay, **(gates or {})}, r)


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

    # Drift/safe_mode gate keys (see check_drift_safemode_gates) are wired
    # into STAGE_CHECKS for ensemble-confirm + paper-deploy only, matching
    # the v2.2 §8 spec ("post-deploy only"). Training stages (hpo / l1 /
    # wf / oos) intentionally do not enforce them — the threshold matrix
    # comes from the overlay *.gates.yaml at ensemble-confirm time.

    # S495-cont prop-firm decoupling (rev 2): challenge block + static_peak
    # consistency. Live deploys that were re-authored under the new schema
    # must have both env.risk.static_peak and risk.static_peak aligned
    # (train/live divergence was the S422 failure mode we're guarding).
    check_challenge_block(cfg, r)
    check_static_peak_consistency(cfg, r)
    # Open Question #5 (resolved 2026-04-24): challenge_target_hit_rate field
    # in the upstream L1/WF manifest. Best-effort — WARN only when the
    # baseline file is locally readable, since live deploys often run on a
    # different host and the file may not be present at validation time.
    check_drift_baseline_manifest_schema(cfg, r)

    # v2.3 (S514) §4.5 step 6 + §8.2-extension blockers for ensemble-deployed
    # prop-firm strategies.
    check_v23_swap_handshake(cfg, r)
    check_v23_agreement_decay_gates(cfg, r)
    # X6 (§4.5): offline retrain_policy gate + live gates.retrain cost-drift.
    check_retrain_gate(cfg, r)


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


def check_drift_baseline_manifest_schema(cfg: dict, r: ValidationResult) -> None:
    """Best-effort check that the drift baseline manifest carries the v2.2
    ``challenge_target_hit_rate`` block introduced by the S495 Open Question
    #5 resolution (2026-04-24).

    Only fires when:
      - the workstream is prop-firm (so the field is meaningful)
      - ``drift.enabled`` is true
      - ``drift.baseline_path`` resolves to a readable JSON file (absolute
        path or relative to repo root)

    Otherwise silent — live deploys often run on a different host where the
    baseline file is not present at validation time. Emits WARN (not FAIL)
    so in-flight workstreams whose baselines pre-date the schema extension
    can still be deployed; promote to FAIL once all migrations have
    re-emitted their seed/ensemble reports.
    """
    if not _is_prop_firm(cfg):
        return
    drift = cfg.get("drift", {}) or {}
    if not drift.get("enabled"):
        return
    baseline = drift.get("baseline_path")
    if not baseline:
        return  # check_paper_deploy already FAILs on missing baseline_path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [Path(baseline)]
    if not candidates[0].is_absolute():
        candidates.append(repo_root / baseline)
    # Live configs typically use the container path (`/app/baselines/...`); try
    # stripping that prefix and resolving relative to repo root as a best effort.
    if baseline.startswith("/app/"):
        candidates.append(repo_root / baseline[len("/app/"):])
    target = next((p for p in candidates if p.exists() and p.is_file()), None)
    if target is None:
        return  # remote-host or pre-deploy validation; silent

    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return  # malformed file isn't this check's concern

    protocol = str(manifest.get("protocol", ""))
    is_seed_report = protocol.startswith("v2.2_stage_2_seed_report")
    is_ensemble_report = protocol.startswith("v2.2_stage_2_5_ensemble_report")

    if is_seed_report and "challenge_target_hit_rate_by_seed" not in manifest:
        r.warn(
            f"drift.baseline_path={baseline} is a seed_report but lacks "
            "`challenge_target_hit_rate_by_seed` (S495 Open Question #5 "
            "resolution 2026-04-24). Re-emit via "
            "`scripts/backfill_eval_distribution_v22.py --phase-spec ...` "
            "before relying on the val-selection tiebreaker."
        )
    elif is_ensemble_report and (
        "challenge_target_hit_rate_by_seed" not in manifest
        or "ensemble_challenge_target_hit_rate" not in manifest
    ):
        r.warn(
            f"drift.baseline_path={baseline} is an ensemble_report but lacks "
            "`challenge_target_hit_rate_by_seed` and/or "
            "`ensemble_challenge_target_hit_rate` (S495 Open Question #5 "
            "resolution 2026-04-24). Re-emit via "
            "`scripts/backfill_eval_distribution_v22.py --phase-spec ...`."
        )


def check_turnover_limit_explicit(cfg: dict, r: ValidationResult) -> None:
    """S506 Option B: structural fix uses UTC-midnight-anchored turnover reset.

    The S495-cont hotfix (`risk.daily_turnover_limit: 10.0`) was added to
    several live configs to mask the call-count-based reset stretching
    "daily" across 2-3 calendar days under signal_gate / deadband gating.
    With Option B in place, the override is no longer needed and is likely
    vestigial. WARN (not FAIL) so operators with genuinely high-turnover
    strategies can still cap explicitly. See
    `.agent/artifacts/turnover_cap_structural_fix_spec.md` §5 Stage 3 for
    the documented removal order.
    """
    risk = cfg.get("risk", {}) or {}
    if "daily_turnover_limit" not in risk:
        return  # using runner default 4.0 — fine
    val = risk["daily_turnover_limit"]
    if val == 10.0:
        r.warn(
            f"risk.daily_turnover_limit={val} matches the S495-cont hotfix "
            "value. Verify whether this override is still needed after the "
            "S506 structural UTC-midnight reset "
            "(.agent/artifacts/turnover_cap_structural_fix_spec.md)."
        )


def check_report_schema(cfg: dict, r: ValidationResult) -> None:
    """Protocol v2 §2 manifest schema enforcement on Stage 2 / 2.5 reports.

    Looks up reports referenced by the config (drift.baseline_path,
    agent.ensemble.bundle_path) and validates each against
    `docs/schemas/manifest.schema.json`. Missing files are skipped silently
    (they get caught by other checks); schema mismatches WARN so backfill
    workflows aren't blocked. Schema is loaded lazily; missing schema file
    or jsonschema package is reported as a single WARN.

    Wired into ensemble-confirm + paper-deploy. Training stages
    (hpo / l1 / wf / oos) don't reference reports yet.
    """
    schema = _load_manifest_schema()
    if schema is None:
        r.warn(
            f"manifest schema not found at {MANIFEST_SCHEMA_PATH.relative_to(Path.cwd()) if MANIFEST_SCHEMA_PATH.is_relative_to(Path.cwd()) else MANIFEST_SCHEMA_PATH} "
            "— skipping report-schema validation"
        )
        return
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        r.warn("jsonschema not installed — skipping report-schema validation (install with `pip install -e .[dev]`)")
        return

    paths = _candidate_report_paths(cfg)
    if not paths:
        return  # no reports to check is fine; other checks cover required-vs-optional

    checked = 0
    for p in paths:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            r.warn(f"report not parseable as JSON: {p} ({exc})")
            continue
        try:
            jsonschema.validate(data, schema)
            checked += 1
        except jsonschema.ValidationError as exc:
            path_repr = "/".join(str(x) for x in exc.absolute_path) or "<root>"
            r.warn(
                f"report {p} fails manifest schema at {path_repr}: "
                f"{exc.message[:200]}"
            )
    if checked:
        r.ok(f"manifest schema OK on {checked} report(s)")


_V23_AGREEMENT_DECAY_KEYS = (
    "agreement_flat_delta_warn",
    "agreement_flat_delta_crit",
    "agreement_flat_window_bars",
)
_CONSENSUS_RULES = ("ens_agreement", "ens_majority")


def _resolve_ensemble_rule(cfg: dict) -> str | None:
    """Return the live aggregation rule, preferring config override over bundle.

    The live config may either pin ``agent.ensemble.aggregation_rule`` (the
    canonical case) or rely on the bundle's manifest ``chosen_rule``. We
    can't unpack the bundle here — paper-deploy validation often runs
    without the bundle file being locally present — so we only inspect
    the config-side declaration. Returns None when no rule can be inferred.
    """
    ens = (cfg.get("agent", {}) or {}).get("ensemble") or {}
    rule = ens.get("aggregation_rule") or ens.get("rule")
    if isinstance(rule, str) and rule:
        return rule
    return None


def check_v23_swap_handshake(cfg: dict, r: ValidationResult) -> None:
    """v2.3 §4.5 step 6 swap-approval handshake declaration.

    Prop-firm / live-capital paper-deploys that load an ensemble bundle
    must declare a kill_file path (already enforced upstream by
    check_paper_deploy) AND should declare an explicit
    ``safety.last_bundle_file`` so the handshake state survives container
    restarts on a non-default mount. The handshake itself is enforced at
    runtime by ``finrl_pro_ds.live.swap_handshake.check_swap_approved``;
    this validator only catches the declaration mistake.
    """
    if not _is_prop_firm(cfg):
        return
    ens = (cfg.get("agent", {}) or {}).get("ensemble") or {}
    if not ens.get("bundle_path"):
        return  # legacy retro-apply path or solo deploy — handshake N/A

    safety = cfg.get("safety", {}) or {}
    risk = cfg.get("risk", {}) or {}
    kill_file = safety.get("kill_file") or risk.get("kill_file")
    if not kill_file:
        # Already FAILed by check_paper_deploy; skip to avoid duplicate noise.
        return

    # last_bundle_file is OPTIONAL — defaults to <kill_file>.last_bundle in
    # the runner. WARN if it isn't explicit, because the operator should
    # confirm the state file lives on a persistent mount (kill_file is
    # already required to be on /app/state per paper-deploy precedent).
    last_bundle = safety.get("last_bundle_file")
    if not last_bundle:
        r.warn(
            "safety.last_bundle_file not declared — v2.3 swap handshake "
            f"will default to {kill_file}.last_bundle. Verify the path "
            f"is on a persistent mount (otherwise every container restart "
            f"is treated as a new bundle and triggers the handshake)."
        )
    else:
        r.ok(f"v2.3 swap handshake state file: {last_bundle}")

    swap_approved = safety.get("swap_approved_file")
    if swap_approved:
        r.ok(f"v2.3 swap handshake sentinel path: {swap_approved}")


def check_v23_agreement_decay_gates(cfg: dict, r: ValidationResult) -> None:
    """v2.3 §8.2-extension agreement-decay gate-key presence.

    Mandatory when:
      * workstream is prop-firm / live-capital, AND
      * live agent uses a consensus aggregation rule (``ens_agreement`` /
        ``ens_majority``).
    The agreement-decay tracker is the silent-death detector for these
    rules (see Protocol v2 §4.5 trigger #4); shipping without the
    threshold declarations means a misconfigured live deploy could go
    undetected for thousands of bars.
    """
    if not _is_prop_firm(cfg):
        return
    rule = _resolve_ensemble_rule(cfg)
    if rule not in _CONSENSUS_RULES:
        return  # non-consensus rule → tracker is no-op; gates are advisory

    gates = cfg.get("gates", {}) or {}
    drift_gates = gates.get("drift") or {}

    missing = [k for k in _V23_AGREEMENT_DECAY_KEYS if k not in drift_gates]
    if missing:
        r.fail(
            f"gates.drift missing v2.3 agreement-decay key(s): {missing} — "
            f"required for ensemble rule {rule!r} (Protocol v2 §8.2 ext / "
            f"§4.5 Stage 2.5-R trigger #4). Defaults: warn=0.20, crit=0.40, "
            f"window_bars=2000."
        )
        return
    r.ok(
        f"gates.drift has all {len(_V23_AGREEMENT_DECAY_KEYS)} v2.3 "
        f"agreement-decay keys (rule={rule})"
    )

    # Sanity on declared thresholds.
    warn = drift_gates["agreement_flat_delta_warn"]
    crit = drift_gates["agreement_flat_delta_crit"]
    if not (0 < float(warn) < float(crit) < 1.0):
        r.fail(
            f"gates.drift.agreement_flat_delta_warn={warn} / _crit={crit}: "
            f"must satisfy 0 < warn < crit < 1.0"
        )
    window = drift_gates["agreement_flat_window_bars"]
    if int(window) < 100:
        r.fail(
            f"gates.drift.agreement_flat_window_bars={window} too small — "
            f"minimum 100 (default 2000); short windows generate false CRITs"
        )


# retrain_policy keys required for the offline gate (check_retrain_triggers.py)
# to run once `enabled: true` (each is dereferenced unconditionally there).
_RETRAIN_POLICY_REQUIRED_KEYS = (
    "last_trained_date",
    "staleness_cap_days",
    "validation_pf_baseline",
    "oos_pf_floor_ratio",
    "gate_window_days",
    "backtest_config_ref",
    "data_file_path",
)


def _num(d: dict, key: str):
    """Return d[key] if it is a real (non-bool) number, else None."""
    v = d.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def check_retrain_gate(cfg: dict, r: ValidationResult) -> None:
    """Protocol v2 §4.5 retrain-trigger schema (paper-deploy stage, X6).

    Two complementary, intentionally-distinct blocks (NOT duplicative — see
    docs/protocol_v2.md §4.5):
      * ``retrain_policy:`` drives the offline cron OOS-PF / staleness /
        live-DD / feature-drift-KS gate (scripts/check_retrain_triggers.py).
      * ``gates.retrain.cost_drift_*`` drives the LIVE CostDriftTracker
        (§4.5 trigger #6 → WandB ``drift/cost_ratio``).

    Missing blocks are WARN for prop-firm (one-release-cycle grace, mirroring
    the v2.6 feature_variance_veto rollout) so live configs that predate X6
    keep validating; a declared-but-malformed value is always a FAIL (it would
    crash CostDriftTracker / check_retrain_triggers at runtime).
    """
    prop_firm = _is_prop_firm(cfg)

    # --- gates.retrain.cost_drift_* (consumed by the live CostDriftTracker) ---
    retrain_gates = (cfg.get("gates", {}) or {}).get("retrain") or {}
    if not retrain_gates:
        if prop_firm:
            r.warn(
                "gates.retrain absent — Protocol v2 §4.5 cost-drift trigger #6 "
                "(CostDriftTracker) is DISABLED for this live config. Declare "
                "gates.retrain.cost_drift_ratio (>1.0) + cost_drift_window_trades "
                "(int>=1), mirroring the canonical <ws>_ensemble.gates.yaml "
                "(S551-cont-9 both-files rule). WARN this cycle; blocking after "
                "rollout to all live configs."
            )
    else:
        ratio = retrain_gates.get("cost_drift_ratio")
        if ratio is not None and not (
            isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
            and float(ratio) > 1.0
        ):
            r.fail(
                f"gates.retrain.cost_drift_ratio must be a number > 1.0 "
                f"(CostDriftTracker rejects <=1.0); got {ratio!r}"
            )
        window = retrain_gates.get("cost_drift_window_trades")
        if window is not None and not (
            isinstance(window, int) and not isinstance(window, bool) and window >= 1
        ):
            r.fail(
                f"gates.retrain.cost_drift_window_trades must be an int >= 1; "
                f"got {window!r}"
            )
        if ratio is not None and window is not None:
            r.ok("gates.retrain cost-drift keys present and well-formed (§4.5 #6)")

    # --- retrain_policy (offline gate) ---
    policy = cfg.get("retrain_policy")
    if not policy:
        if prop_firm:
            r.warn(
                "retrain_policy absent — the offline retrain gate "
                "(check_retrain_triggers.py) returns CONFIG_MISSING for this "
                "strategy (no automated OOS/staleness/drift degradation "
                "detector). Add the block (audit P10-03). WARN this cycle; "
                "blocking after rollout to all live configs."
            )
        return
    if not policy.get("enabled", False):
        r.warn("retrain_policy.enabled is false — offline retrain gate disabled")
        return

    missing = [k for k in _RETRAIN_POLICY_REQUIRED_KEYS if k not in policy]
    if missing:
        r.fail(
            f"retrain_policy enabled but missing required key(s): {missing} — "
            f"check_retrain_triggers.py dereferences each unconditionally"
        )

    # Checkpoint resolvability (P10-03 reconcile): bundle deploys leave
    # agent.checkpoint_path empty, so the OOS backtest needs an explicit
    # retrain_policy.checkpoint_path; solo deploys may use agent.checkpoint_path.
    ckpt = policy.get("checkpoint_path") or (cfg.get("agent", {}) or {}).get(
        "checkpoint_path",
    )
    if not ckpt:
        r.fail(
            "retrain_policy enabled but no checkpoint to backtest — set "
            "retrain_policy.checkpoint_path (bundle deploys leave "
            "agent.checkpoint_path empty)"
        )

    floor = _num(policy, "oos_pf_floor_ratio")
    if floor is not None and not (0.0 < floor <= 1.0):
        r.fail(f"retrain_policy.oos_pf_floor_ratio must be in (0, 1]; got {floor}")
    dd = _num(policy, "dd_trigger_ratio")
    if dd is not None and not (0.0 < dd <= 1.0):
        r.fail(f"retrain_policy.dd_trigger_ratio must be in (0, 1]; got {dd}")
    ks = _num(policy, "feature_drift_ks_stat_threshold")
    if ks is not None and not (0.0 < ks <= 1.0):
        r.fail(
            f"retrain_policy.feature_drift_ks_stat_threshold must be in (0, 1]; "
            f"got {ks}"
        )
    baseline = _num(policy, "validation_pf_baseline")
    if baseline is not None and baseline <= 0.0:
        r.fail(f"retrain_policy.validation_pf_baseline must be > 0; got {baseline}")
    for key in ("staleness_cap_days", "gate_window_days"):
        v = _num(policy, key)
        if v is not None and v <= 0:
            r.fail(f"retrain_policy.{key} must be > 0; got {v}")
    ltd = policy.get("last_trained_date")
    if ltd is not None:
        try:
            datetime.fromisoformat(str(ltd))
        except ValueError:
            r.fail(
                f"retrain_policy.last_trained_date not ISO-parseable: {ltd!r}"
            )
    if not missing and ckpt:
        r.ok("retrain_policy block present and well-formed (offline §4.5 gate)")


def check_obs_noise_gate(cfg: dict, r: ValidationResult) -> None:
    """Protocol v2.7-B Stage 3.5 observation-noise gate validation (S553-cont-25).

    Validates that the config (via its ensemble gates overlay) declares the keys
    required for Stage 3.5 price-path randomization
    (``scripts/stage_3_5_obs_noise.py``). Stage 3.5 re-rolls the PROMOTE policy
    through multiplicative log-noise on the raw OHLC and gates on edge survival
    (median PF ratio floor + q95(|MDD|) degradation buffer, per σ-level).

    Phase α (S553+): legacy configs (``protocol_version`` unset or ``< "2.7"``)
    are exempt — this check is a no-op. Mirrors ``check_sensitivity_audit``'s
    Phase α/β rollout (ADR-9), using the N4 version-tuple compare so the gate
    self-enables at 2.7 and every later version.

    Phase β: operator bumps ``protocol_version`` to ``"2.7"`` AND flips
    ``obs_noise_required: true`` after locking ``obs_noise_pf_floor_*`` /
    ``obs_noise_mdd_buffer_pp_*`` against the de-leaked candidates' empirical
    spread. Validator then FAILs prop-firm configs missing the keys.

    See ``.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md``
    (IC-3/IC-4).
    """
    protocol_version = str(cfg.get("protocol_version", "2.6"))
    if not _protocol_at_least(protocol_version, "2.7"):
        return

    gates = _load_ensemble_gates_overlay(cfg)
    prop_firm = _is_prop_firm(cfg)
    fail_or_warn = r.fail if prop_firm else r.warn

    obs_hint = (
        "Copy the v2.7 Stage 3.5 block from a sister <workstream>_ensemble.gates.yaml "
        "or use Phase α defaults: obs_noise_sigma_levels: {\"10bps\": 0.001, "
        "\"50bps\": 0.005}, obs_noise_n_seeds: 10, obs_noise_pf_floor_10bps: 0.85, "
        "obs_noise_pf_floor_50bps: 0.70, obs_noise_mdd_buffer_pp_10bps: 0.3, "
        "obs_noise_mdd_buffer_pp_50bps: 0.7, obs_noise_required_min_folds: <wf_folds>, "
        "obs_noise_required: false (Phase α) -> true (Phase β)."
    )

    levels = gates.get("obs_noise_sigma_levels")
    if not isinstance(levels, dict) or not levels:
        fail_or_warn(
            "gates.obs_noise_sigma_levels missing or empty — v2.7 Stage 3.5 "
            "requires a {label -> log-sigma} map. " + obs_hint
        )
        return

    # Each σ-level needs a non-negative σ plus both a PF floor and an MDD buffer.
    for label in levels:
        sigma = levels[label]
        try:
            if float(sigma) < 0.0:
                r.fail(f"gates.obs_noise_sigma_levels[{label!r}]={sigma} must be >= 0")
        except (TypeError, ValueError):
            r.fail(f"gates.obs_noise_sigma_levels[{label!r}]={sigma!r} is not numeric")
        for key, lo, hi in ((f"obs_noise_pf_floor_{label}", 0.50, 0.99),
                            (f"obs_noise_mdd_buffer_pp_{label}", 0.0, 5.0)):
            v = gates.get(key)
            if v is None:
                fail_or_warn(
                    f"gates.{key} not set (v2.7 Stage 3.5 σ-level {label!r}). " + obs_hint
                )
                continue
            try:
                vf = float(v)
                if not (lo <= vf <= hi):
                    r.warn(f"gates.{key}={vf} outside sanity bound [{lo}, {hi}].")
            except (TypeError, ValueError):
                r.fail(f"gates.{key}={v!r} is not numeric")

    n_seeds = gates.get("obs_noise_n_seeds")
    if n_seeds is None:
        fail_or_warn("gates.obs_noise_n_seeds not set (v2.7 Stage 3.5). " + obs_hint)
    else:
        try:
            if int(n_seeds) < 5:
                r.warn(
                    f"gates.obs_noise_n_seeds={n_seeds} < 5 — noise q05/q95 quantile "
                    "estimates will be unstable; 10 recommended."
                )
        except (TypeError, ValueError):
            r.fail(f"gates.obs_noise_n_seeds={n_seeds!r} is not integer")

    required = gates.get("obs_noise_required")
    if required is None:
        fail_or_warn(
            "gates.obs_noise_required not set — v2.7 explicit declaration required. "
            "Set false during Phase α calibration; flip true in Phase β once the "
            "thresholds are locked."
        )
    elif not isinstance(required, bool):
        r.fail(f"gates.obs_noise_required={required!r} must be boolean (true/false)")


STAGE_CHECKS = {
    "data-prep": [],
    "hpo": [check_hpo],
    "l1-multiseed": [check_l1_multiseed],
    "ensemble-confirm": [
        check_ensemble_confirm,
        check_sensitivity_audit,
        check_drift_safemode_gates,
        check_report_schema,
    ],
    "wf": [check_wf, check_obs_noise_gate],
    "oos": [],
    "paper-deploy": [
        check_paper_deploy,
        check_turnover_limit_explicit,
        check_drift_safemode_gates,
        check_report_schema,
    ],
}


# ── XPARAM: cross-parameter structural invariants (XPARAM-01..12) ───────────
# Ported 2026-07-29 from the /audit skill addendum (FINRL.md Phase 3.5), where
# they existed only as a checklist an auditor had to remember. A declared-but-
# unwired invariant is false assurance — the audit skill's own rule — so they
# execute here instead.
#
# These are STRUCTURAL invariants: a config violating one cannot train
# correctly regardless of strategy. They are deliberately NOT in
# `configs/<workstream>.gates.yaml`, which holds *decision* thresholds (PF
# floors, DD buffers, retrain triggers) that vary per workstream. These do not
# vary. Kept as named constants so they remain greppable and single-source.
_XPARAM_MIN_LEARN_STEPS = {"sac": 200_000, "iqn": 100_000, "bdq": 100_000}  # XPARAM-01
_XPARAM_MAX_DEADBAND = 0.5  # XPARAM-05
_XPARAM_DSR_ETA_RANGE = (0.0001, 0.01)  # XPARAM-06
_XPARAM_DSR_SCALE_RANGE = (0.1, 10.0)  # XPARAM-07
_XPARAM_MAX_UPDATE_X_ENVS = 200  # XPARAM-08 (OPT-10)
_XPARAM_MIN_N_STEP = 3  # XPARAM-12
_XPARAM_PRIVATE_DIM_BY_MDP = {"v6": 4, "v7": 5}  # XPARAM-11
# Below this, a config is a backtest (total_timesteps: 0), smoke, or dev run;
# production budget invariants do not apply. Real runs are 3M-5M steps.
_XPARAM_MIN_PRODUCTION_STEPS = 100_000


def _dig(cfg: dict, path: str) -> Any:
    """Fetch a dotted config path, returning None if any level is absent."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _is_live_inference_config(cfg: dict) -> bool:
    """Live/paper trading configs use a different schema entirely.

    They carry `exchange`/`bar_clock` and no `training` block; their `network`
    values describe the loaded checkpoint, not a run to be trained. Applying
    training invariants to them produces only false positives (e.g. a live
    config's vestigial `buffer_size: 100` is not an XPARAM-03 violation).
    """
    return "training" not in cfg and ("bar_clock" in cfg or "exchange" in cfg)


def _is_production_training(cfg: dict) -> bool:
    """True only for real training runs, not backtest/smoke/dev configs.

    Backtest and eval-only configs set `training.total_timesteps: 0`; smoke and
    dev configs use deliberately tiny budgets (1K-50K). The budget invariants
    (XPARAM-01/02/03/04/08) describe production training and are meaningless
    below that scale — firing CRITICAL on a 1,000-step smoke config would make
    the gate something operators route around.
    """
    total = _dig(cfg, "training.total_timesteps")
    return isinstance(total, (int, float)) and total >= _XPARAM_MIN_PRODUCTION_STEPS


def check_xparam(cfg: dict, r: ValidationResult) -> None:
    """Cross-parameter structural invariants (XPARAM-01..12).

    Every check is guarded: an absent block is SKIPPED, never assumed. Silence
    means "not applicable to this config", not "passed". CRITICAL/HIGH findings
    call fail(); MEDIUM call warn().

    Scope: training configs only. Budget checks additionally require production
    scale — see `_is_live_inference_config` / `_is_production_training`. The
    dimensional checks (05/06/07/09/10/11/12) still run on backtest configs,
    because a backtest with the wrong `private_dim` is genuinely broken.
    """
    if _is_live_inference_config(cfg):
        return

    is_production = _is_production_training(cfg)
    agents = cfg.get("agents") or {}
    agent_name = next((a for a in ("sac", "iqn", "bdq") if a in agents), None)
    agent_cfg = (agents.get(agent_name) or {}) if agent_name else {}

    learning_starts = agent_cfg.get("learning_starts")
    buffer_size = agent_cfg.get("buffer_size")
    total_timesteps = _dig(cfg, "training.total_timesteps")
    checked: list[str] = []

    # XPARAM-01 (CRITICAL) — HPO trial must leave enough post-warmup budget.
    hpo = cfg.get("hpo") or {}
    steps_per_trial = hpo.get("steps_per_trial")
    if hpo.get("enabled") is False or not is_production:
        pass  # explicitly disabled, or not a production training run
    elif agent_name and steps_per_trial is not None and learning_starts is not None:
        floor = _XPARAM_MIN_LEARN_STEPS[agent_name]
        usable = steps_per_trial - learning_starts
        if usable < floor:
            r.fail(
                f"XPARAM-01: hpo.steps_per_trial - agents.{agent_name}.learning_starts "
                f"= {usable:,} < {floor:,} required for {agent_name.upper()}"
            )
        else:
            checked.append("01")

    # XPARAM-02 (CRITICAL) — warmup must fit inside the run.
    if is_production and learning_starts is not None and total_timesteps is not None:
        if learning_starts >= total_timesteps:
            r.fail(
                f"XPARAM-02: agents.{agent_name}.learning_starts ({learning_starts:,}) "
                f">= training.total_timesteps ({total_timesteps:,}) — agent never learns"
            )
        else:
            checked.append("02")

    # XPARAM-03 (HIGH) — warmup must fit inside the replay buffer.
    if is_production and learning_starts is not None and buffer_size is not None:
        if learning_starts >= buffer_size:
            r.fail(
                f"XPARAM-03: agents.{agent_name}.learning_starts ({learning_starts:,}) "
                f">= buffer_size ({buffer_size:,})"
            )
        else:
            checked.append("03")

    # XPARAM-04 (HIGH) — fee ramp must complete within the run.
    fee_schedule = _dig(cfg, "env.fee_schedule")
    if is_production and isinstance(fee_schedule, list) and total_timesteps is not None:
        for entry in fee_schedule:
            if not isinstance(entry, dict):
                continue
            ramp_end = entry.get("ramp_end_step")
            if ramp_end is not None and ramp_end > total_timesteps:
                r.fail(
                    f"XPARAM-04: env.fee_schedule ramp_end_step ({ramp_end:,}) > "
                    f"training.total_timesteps ({total_timesteps:,}) — fees never reach full rate"
                )
                break
        else:
            checked.append("04")

    # XPARAM-05 (CRITICAL) — deadband above this suppresses most trades.
    deadband = _dig(cfg, "env.deadband_threshold")
    if deadband is not None:
        if deadband > _XPARAM_MAX_DEADBAND:
            r.fail(
                f"XPARAM-05: env.deadband_threshold ({deadband}) > {_XPARAM_MAX_DEADBAND}"
            )
        else:
            checked.append("05")

    # XPARAM-06 / -07 (MEDIUM) — DSR reward params inside sane ranges.
    for xid, key, (lo, hi) in (
        ("06", "dsr_eta", _XPARAM_DSR_ETA_RANGE),
        ("07", "dsr_scale", _XPARAM_DSR_SCALE_RANGE),
    ):
        val = _dig(cfg, f"env.reward.{key}")
        if val is None:
            val = _dig(cfg, f"env.{key}")
        if val is None:
            continue
        if not (lo <= val <= hi):
            r.warn(f"XPARAM-{xid}: env.reward.{key} ({val}) outside [{lo}, {hi}]")
        else:
            checked.append(xid)

    # XPARAM-08 (MEDIUM) — effective update batching ceiling (OPT-10).
    update_interval = agent_cfg.get("update_interval")
    num_envs = _dig(cfg, "training.num_envs")
    if is_production and update_interval is not None and num_envs is not None:
        product = update_interval * num_envs
        if product > _XPARAM_MAX_UPDATE_X_ENVS:
            r.warn(
                f"XPARAM-08: update_interval x num_envs = {product} > "
                f"{_XPARAM_MAX_UPDATE_X_ENVS} (OPT-10)"
            )
        else:
            checked.append("08")

    # XPARAM-09 (HIGH) — features-per-scale must triple-match.
    fps = {
        "features.features_per_scale": _dig(cfg, "features.features_per_scale"),
        "env.features_per_scale": _dig(cfg, "env.features_per_scale"),
        "network.scale_encoder.input_size": _dig(cfg, "network.scale_encoder.input_size"),
    }
    present = {k: v for k, v in fps.items() if v is not None}
    if len(present) >= 2:
        if len(set(present.values())) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in present.items())
            r.fail(f"XPARAM-09: features-per-scale mismatch ({detail})")
        else:
            checked.append("09")

    # XPARAM-10 (HIGH) — scale lists must match between features and env.
    f_scales, e_scales = _dig(cfg, "features.scales"), _dig(cfg, "env.scales")
    if f_scales is not None and e_scales is not None:
        if list(f_scales) != list(e_scales):
            r.fail(f"XPARAM-10: features.scales {f_scales} != env.scales {e_scales}")
        else:
            checked.append("10")

    # XPARAM-11 (HIGH) — private_dim is fixed by MDP version.
    mdp = _dig(cfg, "env.mdp_version")
    private_dim = _dig(cfg, "network.private_dim")
    if mdp is not None and private_dim is not None:
        expected = _XPARAM_PRIVATE_DIM_BY_MDP.get(str(mdp).lower())
        if expected is None:
            pass  # unknown MDP version — not this check's business
        elif private_dim != expected:
            r.fail(
                f"XPARAM-11: network.private_dim ({private_dim}) != {expected} "
                f"required for env.mdp_version={mdp}"
            )
        else:
            checked.append("11")

    # XPARAM-12 (HIGH) — IQN multi-step return depth.
    n_step = _dig(cfg, "agents.iqn.n_step")
    if n_step is not None:
        if n_step < _XPARAM_MIN_N_STEP:
            r.fail(f"XPARAM-12: agents.iqn.n_step ({n_step}) < {_XPARAM_MIN_N_STEP}")
        else:
            checked.append("12")

    if checked:
        r.ok(f"XPARAM {'/'.join(sorted(checked))} pass ({len(checked)} applicable)")


def validate(config_path: Path, stage: str, overlays: list[str] | None = None) -> ValidationResult:
    cfg = load_yaml(config_path)
    # Resolve a `base_book:` parent (inheritance) BEFORE the deploy overlays so the universal
    # checks see the effective env/data/gates (exec-overlay config inherits the 2-sleeve book).
    cfg = _resolve_base_book(cfg, config_path)
    if overlays:
        from finrl_pro_ds.config_utils import apply_overlays
        project_root = Path(__file__).resolve().parent.parent
        cfg = apply_overlays(
            cfg, overlays,
            overlay_root=project_root / "configs" / "deploy",
            allowlist_path=project_root / "configs" / "deploy" / "ALLOWLIST.yaml",
        )
    r = ValidationResult()

    check_no_fee_curriculum(cfg, r)
    check_xparam(cfg, r)
    check_no_hindsight_outside_hpo(cfg, stage, r)
    check_max_leverage_bounds(cfg, r)
    check_sleeve_combiner(cfg, r)
    check_legacy_prop_firm_block(cfg, stage, r, config_path=config_path)
    check_gates_block(cfg, r)
    check_data_manifest(cfg, stage, r)
    check_execution_cost_realism(cfg, stage, r)
    check_execution_overlay_gates(cfg, r)
    check_wandb_consolidation(cfg, stage, r)

    for check in STAGE_CHECKS[stage]:
        check(cfg, r)

    return r


def validate_report(report_path: Path) -> ValidationResult:
    """Validate a single Stage 2 / 2.5 report against the manifest schema."""
    r = ValidationResult()
    schema = _load_manifest_schema()
    if schema is None:
        r.fail(f"manifest schema not found at {MANIFEST_SCHEMA_PATH}")
        return r
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        r.fail("jsonschema not installed (`pip install -e .[dev]`)")
        return r
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        r.fail(f"report unreadable: {exc}")
        return r
    try:
        jsonschema.validate(data, schema)
        protocol = data.get("protocol", "<legacy>")
        r.ok(f"report valid against manifest schema (protocol={protocol})")
    except jsonschema.ValidationError as exc:
        path_repr = "/".join(str(x) for x in exc.absolute_path) or "<root>"
        r.fail(f"schema mismatch at {path_repr}: {exc.message[:300]}")
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="YAML config to validate against a stage")
    parser.add_argument("--stage", choices=VALID_STAGES,
                        help="Pipeline stage; required with --config")
    parser.add_argument("--report", type=Path,
                        help="Validate a Stage 2 / 2.5 JSON report against manifest schema "
                             "(seed_report.json or ensemble_report.json)")
    parser.add_argument(
        "--overlay", action="append", default=None,
        help="Deploy overlay under configs/deploy/ (e.g. 'velotrade/step1'). "
             "Repeat for multiple; later wins. Mirrors run_live --overlay so "
             "the validator can check the EFFECTIVE deployed config, not "
             "just the base (prop-firm overlays own static_peak per "
             "decision_prop_firm_decoupling_s495).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as failures (CI mode)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.report is not None:
        if not args.report.exists():
            logger.error("report not found: %s", args.report)
            return 2
        result = validate_report(args.report)
        for msg in result.passed:
            logger.info("  PASS  %s", msg)
        for msg in result.warnings:
            logger.warning("  WARN  %s", msg)
        for msg in result.failures:
            logger.error("  FAIL  %s", msg)
        status = result.status
        if args.strict and status == "WARN":
            status = "FAIL"
        logger.info("\nstatus=%s  report=%s", status, args.report.name)
        return 1 if status == "FAIL" else 0

    if args.config is None or args.stage is None:
        parser.error("--config and --stage are required (or use --report)")

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 2

    result = validate(args.config, args.stage, overlays=args.overlay)

    for msg in result.passed:
        logger.info("  PASS  %s", msg)
    for msg in result.warnings:
        logger.warning("  WARN  %s", msg)
    for msg in result.failures:
        logger.error("  FAIL  %s", msg)

    status = result.status
    if args.strict and status == "WARN":
        status = "FAIL"

    overlay_tag = f"  overlays={','.join(args.overlay)}" if args.overlay else ""
    logger.info("\nstatus=%s  config=%s  stage=%s%s", status, args.config.name, args.stage, overlay_tag)
    if status == "FAIL":
        logger.error("Protocol v2 violations. See %s.", PROTOCOL_DOC)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
