"""Crucible P3 runner — the continuous orchestrator (spec §8, P3 row).

Runs N unattended nightly ticks over one or more substrates. Each tick applies the ``substrate_dirty``
eligibility gate (§10.1) — mining only when new data or fresh hypotheses exist — inside a per-tick
cost budget (CR-7), charging per-substrate LORD++ online-FDR wealth per test (§6.1) and appending an
auditable row to the central ``orchestrator.db`` tick log. The default proposer is the deterministic
offline ``LibrarySeedProposer`` (no LLM/network), so the whole loop reproduces bit-identically and a
synthetic run is a pure ``0 PROMISING`` null-safety pass. The harness caps at PROMISING; a deploy
read of any survivor still requires a human-triggered Tier-2 deep audit.

Because the offline proposer emits a fixed seed bank, a synthetic run mines on night 1 and then
NO-OPS every subsequent night on the unchanged panel (the clean-panel no-op the P3 gate requires) —
this is the intended demonstration that the loop only spends compute/FDR wealth when a substrate is
genuinely dirty.

Usage:
  python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4
  python scripts/research/crucible_orchestrator.py --mode real --start 2008-01-01 --nights 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env (override=False: a real shell/system export always wins) so the real-mode alt-data bridge
# picks up FRED_API_KEY / SEC_EDGAR_UA without the operator having to export them each session. A
# missing credential still degrades to a per-source skip inside the bridge — it never aborts a tick.
try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(ROOT / ".env", override=False)
except ModuleNotFoundError:
    pass

from finrl_pro_ds.crucible import (  # noqa: E402
    CRUCIBLE_VERSION,
    DataCatalog,
    Lockbox,
    OrchestratorStore,
    Substrate,
    TickBudget,
    TrialLedger,
    gates_hash,
    load_incubation_criterion,
    run_orchestrator_tick,
)
from finrl_pro_ds.crucible.agentic import LibrarySeedProposer, LlmProposer  # noqa: E402
from finrl_pro_ds.crucible.orchestrator.orchestrator import _safe  # noqa: E402
from finrl_pro_ds.crucible.orchestrator.substrate import (  # noqa: E402
    PowerGuard,
    PreparedSubstrate,
    folded_snapshot_hash,
    stamp_substrate_power,
)
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.generation.config import (  # noqa: E402
    load_cohort_config,
    load_generation_config,
    load_generation_meta,
)

log = logging.getLogger("crucible_orchestrator")
DEFAULT_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
DEFAULT_LOCKBOX_GATES = ROOT / "configs" / "crucible_lockbox.gates.yaml"
DEFAULT_COHORT_GATES = ROOT / "configs" / "crucible_cohort.gates.yaml"
DEFAULT_POWER_GATES = ROOT / "configs" / "crucible_power.gates.yaml"


def _load_power_guard(path: str, *, force: bool):
    """Load the NOW-5 substrate-power guard + its E1/E2 calibration sweep (audit C2-01/C6-07). Returns
    ``(PowerGuard | None, sweep_dict | None, sweep_hash)``. Degrades to ``(None, None, "")`` — no stamp,
    no guard — when the config or the sweep JSON is absent, so a fresh checkout (no calibration run yet)
    still works. Kept OUT of signal_eval.gates.yaml so the frozen funnel gates_hash is untouched (CRU-1)."""
    import yaml

    p = Path(path)
    if not p.exists():
        log.warning("power-gates file %s absent — substrate-power stamp/guard disabled", path)
        return None, None, ""
    cfg = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("power_guard", {})
    sweep_path = ROOT / cfg.get("calibration_sweep_path", "")
    if not sweep_path.exists():
        log.warning("calibration sweep %s absent — substrate-power stamp/guard disabled", sweep_path)
        return None, None, ""
    raw = sweep_path.read_bytes()
    sweep = json.loads(raw)
    sweep_hash = hashlib.sha256(raw).hexdigest()[:12]
    guard = PowerGuard(enabled=bool(cfg.get("enabled", True)),
                       ceiling=float(cfg.get("plausible_delta_sr_max", 0.5)),
                       action=str(cfg.get("action", "warn")), force=bool(force))
    return guard, sweep, sweep_hash


def _panel_ts(panel: Panel) -> np.ndarray:
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _synthetic_panel(t: int, n: int, *, seed: int = 0, n_feature_slots: int = 1) -> Panel:
    """A NOISE OHLCV panel carrying ``n_feature_slots`` macro feature slots (drive the overlay path).

    Default 1 (just ``macro:regime``) — byte-identical to the historic single-slot panel. Passing more
    adds mutually-independent ``macro:X{i}`` random-walk slots so the overlay pool has de-correlated
    breadth and the opt-in weak-signal COHORT gate can actually form a cohort. The extra draws happen
    AFTER the regime draw, so the 1-slot default leaves the downstream RNG stream untouched."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
    close = np.exp(base + rng.uniform(3.0, 5.0, size=n))
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2010-01-04") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    regime = (np.sin(2.0 * np.pi * np.arange(t) / 80.0) + 0.2 * rng.standard_normal(t)
              ).astype(np.float64)
    slots = {"macro:regime": regime}
    for i in range(max(0, n_feature_slots - 1)):
        slots[f"macro:X{i:02d}"] = np.cumsum(0.05 * rng.standard_normal(t)).astype(np.float64)
    return Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": "synthetic_noise"},
                 feature_slots=slots)


def _proxy_base_sleeves(panel: Panel, hold: int = 21) -> dict[str, np.ndarray]:
    """Inline TSMOM + reversal proxy books (stand-in so the loop runs end-to-end on synthetic)."""
    from finrl_pro_ds.signals.eval_harness import _ls_weights

    c = panel.close
    fwd1 = panel.forward_returns(1)

    def book(score: np.ndarray) -> np.ndarray:
        out = np.full(panel.T, np.nan)
        w = np.zeros(panel.N)
        for t in range(panel.T - 1):
            if t % hold == 0:
                w = _ls_weights(score[t], panel.active[t], min_names=6)
            out[t] = float(np.nansum(w * fwd1[t]))
        return np.nan_to_num(out)

    mom = np.full_like(c, np.nan)
    mom[252:] = c[252:] / c[:-252] - 1.0
    rev = np.full_like(c, np.nan)
    rev[21:] = -(c[21:] / c[:-21] - 1.0)
    return {"tsmom": book(mom), "rates_carry": book(rev)}


def _build_substrate(args, cfg, ek, meta, sweep, sweep_hash) -> tuple[Substrate, DataCatalog]:
    """Construct the single substrate for the chosen mode, plus its catalog (for asset classes).

    Wires the CR-8 forward-incubation lockbox (P4) by default, for both synthetic and real mode: a
    PROMISING survivor is enrolled and accrues forward evidence every tick until CLEARED/REJECTED
    (substrate.py Substrate.lockbox / .incubation_criterion — both opt-in, so ``--no-lockbox`` restores
    the pure P3 byte-identical path used by ``crucible reproduce`` / testing)."""
    out_dir = Path(args.out) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = DataCatalog(out_dir / "catalog.db")
    asset_classes = tuple(sorted({r["asset_class"] for r in catalog.list_series()})) or ("macro",)

    def prepare() -> PreparedSubstrate:
        if args.mode == "synthetic":
            panel = _synthetic_panel(args.t, args.n, n_feature_slots=args.synthetic_slots)
            base = _proxy_base_sleeves(panel, hold=ek["hold_horizon"])
        elif meta["panel"] == "taiwan":
            import dataclasses

            from finrl_pro_ds.crucible.data.altdata_bridge import bridge_altdata_feature_slots
            from finrl_pro_ds.crucible.data.taiwan_altdata import (
                TAIWAN_ALTDATA_ALIASES,
                taiwan_connectors,
            )
            from finrl_pro_ds.data.taiwan_panel_loader import load_taiwan_panel
            from finrl_pro_ds.signals.generation.base_sleeves import taiwan_base_sleeves
            panel = load_taiwan_panel(args.start, args.end)
            # Bridge TWSE/TAIFEX positioning series into feature slots, same shape as the cross_asset
            # branch below (bridge_altdata_feature_slots is already generic — only the connector list
            # + alias map differ). TaifexPositioningConnector is poll-and-accumulate (module docstring
            # in taifex_positioning.py) so real history only grows once this branch is actually run on
            # a schedule; that's expected, not a bug on any single tick.
            if not args.no_altdata_slots:
                bar_end = args.end or np.datetime_as_string(panel.dates.max(), unit="D")
                slots = bridge_altdata_feature_slots(
                    bar_dates=panel.dates, start=args.start, end=bar_end, catalog=catalog,
                    connectors=taiwan_connectors(), aliases=TAIWAN_ALTDATA_ALIASES)
                if slots:
                    panel = dataclasses.replace(
                        panel, feature_slots={**panel.feature_slots, **slots})
            base = taiwan_base_sleeves(panel, hold_horizon=ek["hold_horizon"], cost_bps=ek["cost_bps"],
                                       start=args.start, end=args.end)
        else:
            import dataclasses

            from finrl_pro_ds.crucible.data.altdata_bridge import bridge_altdata_feature_slots
            from finrl_pro_ds.data.cross_asset_panel_loader import load_cross_asset_panel
            from finrl_pro_ds.signals.generation.base_sleeves import production_base_sleeves
            panel = load_cross_asset_panel(
                args.start, args.end, config_path=ROOT / "configs" / "cross_asset_momentum.yaml")
            # Bridge accepted macro/positioning/fundamental series into feature slots so the overlay
            # proposer (family 'altdata') can score them — otherwise real mode mines only the fixed
            # cross-sectional bank. Registers accepted series into the catalog (the 'dirty' signal),
            # and is PIT-gated: a look-ahead HARD-fails inside build_feature_slots (LEAK-2).
            if not args.no_altdata_slots:
                bar_end = args.end or np.datetime_as_string(panel.dates.max(), unit="D")
                slots = bridge_altdata_feature_slots(
                    bar_dates=panel.dates, start=args.start, end=bar_end, catalog=catalog)
                if slots:
                    panel = dataclasses.replace(
                        panel, feature_slots={**panel.feature_slots, **slots})
            base = production_base_sleeves(panel, hold_horizon=ek["hold_horizon"],
                                           cost_bps=ek["cost_bps"], start=args.start, end=args.end)
        # Recompute asset classes AFTER the bridge registered its accepted series (informational —
        # feeds the manifest + proposer context; overlay generation keys off feature_slots, not this).
        prepared_classes = tuple(sorted({r["asset_class"] for r in catalog.list_series()})) or asset_classes
        # NOW-5: stamp the substrate's statistical power (panel depth + interpolated MDE) so the tick log
        # records what the funnel could even detect here, and the guard can refuse an underpowered mine.
        power = (stamp_substrate_power(panel.T, float(ek.get("holdout_frac", 0.25)), sweep, sweep_hash)
                 if sweep is not None else None)
        # NOW-6 (C2-07): fold the panel content-hash into snapshot_hash so a PRICE-bar arrival (or a
        # revision) flips the substrate dirty — the alt-data catalog hash alone missed the primary
        # mining surface, so a pure row-extension of the OHLCV panel used to look "clean" forever.
        return PreparedSubstrate(
            panel=panel, base_returns=base, timestamps=_panel_ts(panel),
            asset_classes=prepared_classes,
            snapshot_hash=folded_snapshot_hash(catalog.snapshot_hash(), panel), power=power)

    ledger = TrialLedger(out_dir / "trial_ledger.db")
    substrate_id = "synthetic" if args.mode == "synthetic" else meta["panel"]
    lockbox = None if args.no_lockbox else Lockbox(out_dir / "lockbox.db")
    incubation_criterion = (None if args.no_lockbox
                            else load_incubation_criterion(args.lockbox_config))
    # Phase 4 weak-signal COHORT gate (opt-in, disabled by default in the gates file). --no-cohort
    # detaches it entirely (the pure byte-identical pre-cohort path for reproduce/testing). The reused
    # funnel floors come from --config's generation block; the cohort knobs from --cohort-config, whose
    # OWN hash is pinned so the frozen funnel gates_hash stays untouched.
    if args.no_cohort:
        cohort_cfg = cohort_mc = cohort_ghash = None
    else:
        cohort_cfg, cohort_mc = load_cohort_config(args.config, args.cohort_config)
        cohort_ghash = gates_hash(args.cohort_config)
    # Proposer seam (CR-1, spec Part A2): default deterministic offline library (0 tokens, no network,
    # bit-reproducible); opt-in LLM proposer spends real tokens per tick, so it also carries the CR-7
    # est_tokens_per_tick (feeds the tick budget + the manifest token_cost; the offline proposer's is
    # 0). main()'s fail-closed guard already rejected --proposer llm without ANTHROPIC_API_KEY.
    proposer: LibrarySeedProposer | LlmProposer
    if args.proposer == "llm":
        proposer = LlmProposer(model=args.llm_model)
        est_tokens = int(args.est_tokens_per_tick)
    else:
        proposer = LibrarySeedProposer()
        est_tokens = 0
    sub = Substrate(substrate_id=substrate_id, prepare=prepare, ledger=ledger, cfg=cfg,
                    evolve_kwargs=ek, proposer=proposer, max_proposals=args.max_proposals,
                    est_tokens_per_tick=est_tokens, lockbox=lockbox,
                    incubation_criterion=incubation_criterion, cohort_cfg=cohort_cfg,
                    cohort_mc_kwargs=cohort_mc, cohort_gates_hash=cohort_ghash)
    return sub, catalog


def _write_recipe(out_dir: Path, substrate_id: str, tick_ts: str, args) -> None:
    """Write ``reproduce_recipe.json`` beside a mined synthetic manifest — the deterministic argv to
    re-run this single tick from scratch (``crucible reproduce`` consumes it; spec §5)."""
    sub_dir = out_dir / _safe(substrate_id) / _safe(tick_ts)
    gates_rel = os.path.relpath(args.config, ROOT)
    # --force: reproduce re-runs a single tick deterministically regardless of generation.enabled
    # (the eligibility flag gates scheduled discovery, not an explicit re-execution of a past run).
    argv = [
        "--mode", "synthetic", "--t", str(args.t), "--n", str(args.n),
        "--synthetic-slots", str(args.synthetic_slots),
        "--max-proposals", str(args.max_proposals), "--max-candidates", str(args.max_candidates),
        "--start-ts", tick_ts, "--nights", "1", "--config", args.config,
        "--lockbox-config", args.lockbox_config, "--force",
        # reproduce is a hermetic pure-P3 re-run: the power stamp (tick DB) and governance handoff
        # (governance.db/notifications) are side artifacts that don't affect the manifest/verdicts the
        # contract verifies, so they are switched off for a clean, byte-identical replay.
        "--no-power-guard", "--no-governance",
    ]
    if args.no_lockbox:
        # keep the recipe's argv faithful to how this run was actually invoked (lockbox enrollment
        # does not affect the manifest/verdicts either way, but the recipe should still match argv).
        argv.append("--no-lockbox")
    # The cohort gate DOES affect the manifest (pinned cohort verdicts + card hashes), so the recipe
    # MUST re-run with the SAME cohort config — otherwise `crucible reproduce` would silently re-run
    # with the disabled default and drop the cohort provenance it is supposed to be verifying.
    if args.no_cohort:
        argv.append("--no-cohort")
    else:
        argv += ["--cohort-config", args.cohort_config]
    recipe = {
        "kind": "synthetic_orchestrator",
        "script": "scripts/research/crucible_orchestrator.py",
        "argv": argv,
        "tick_ts": tick_ts,
        "substrate_id": substrate_id,
        "gates_path": gates_rel,
        "crucible_version": CRUCIBLE_VERSION,
        "manifest_relpath": f"{args.mode}/{_safe(substrate_id)}/{_safe(tick_ts)}/run_manifest.json",
    }
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "reproduce_recipe.json").write_text(
        json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")


def _tick_timestamps(start_ts: str | None, nights: int) -> list[str]:
    """Wall-clock-honest CR-2/CR-8 proposal timestamp per tick (NOW-7, C9-05).

    DEFAULT (no --start-ts): every tick stamps the ACTUAL UTC instant it runs — a multi-night burst in
    one process is N near-simultaneous ticks, never N fabricated FUTURE nights (the old +1d-per-night
    stamped 07-03..07-05 while executing on 07-02, future-dating CR-2 proposal_ts and pushing the
    lockbox forward boundary out). Real nightly cadence comes from the OS scheduler invoking this script
    once per night (--nights 1). REPRODUCE / explicit replay (--start-ts given): step +1 calendar day
    per night but NEVER stamp past wall-clock now (cap at now) — a recorded recipe (--start-ts <ts>
    --nights 1) re-runs bit-identically because a single past ts is returned verbatim, uncapped."""
    now = datetime.now(timezone.utc)
    if not start_ts:
        return [now.isoformat() for _ in range(nights)]
    base = datetime.fromisoformat(start_ts)
    base_cmp = base if base.tzinfo is not None else base.replace(tzinfo=timezone.utc)
    out: list[str] = []
    for i in range(nights):
        stepped_cmp = base_cmp + timedelta(days=i)
        out.append(now.isoformat() if stepped_cmp > now else (base + timedelta(days=i)).isoformat())
    return out


def _run_governance_epilogue(sub, out_dir: Path, handoff_ts: str) -> None:
    """NOW-9 (C8-02/C9-07): the CLEARED->human last mile, wired at last. After the nights loop, scan the
    lockbox for newly-CLEARED survivors and hand each off EXACTLY ONCE (idempotent GovernanceStore),
    surfacing a NOTIFY_<hash>.md + a structured log line. OFFLINE notifiers only — a live TelegramNotifier
    (@<TELEGRAM_BOT>) is a documented drop-in the operator wires deliberately, never fired from a headless run.
    Nothing here promotes or runs the Tier-2 audit (CLAUDE.md)."""
    from finrl_pro_ds.crucible.governance import (  # noqa: E402
        FileNotifier,
        GovernanceStore,
        LogNotifier,
        card_from_dir,
        scan_and_handoff,
    )

    class _FanNotifier:
        """Fan one handoff out to every configured offline notifier."""

        def __init__(self, notifiers):
            self._notifiers = notifiers

        def notify(self, handoff) -> None:
            for n in self._notifiers:
                n.notify(handoff)

    notify_dir = out_dir / "notifications"
    gov_store = GovernanceStore(out_dir / "governance.db")
    try:
        created = scan_and_handoff(
            sub.lockbox, gov_store, _FanNotifier([LogNotifier(), FileNotifier(notify_dir)]),
            handoff_ts=handoff_ts, workstream="crucible", scope=sub.substrate_id,
            substrate_id=sub.substrate_id, card_lookup=card_from_dir(out_dir), out_dir=notify_dir)
        if created:
            log.info("governance: handed off %d newly-CLEARED survivor(s) for human Tier-2 audit",
                     len(created))
    finally:
        gov_store.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible P3 — continuous orchestrator")
    ap.add_argument("--config", default=str(DEFAULT_GATES))
    ap.add_argument("--lockbox-config", default=str(DEFAULT_LOCKBOX_GATES),
                    help="CR-8 incubation criterion gates YAML (kept separate from --config so the "
                         "funnel's frozen gates_hash is untouched)")
    ap.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--nights", type=int, default=4, help="unattended ticks to run")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--t", type=int, default=900)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--synthetic-slots", type=int, default=1,
                    help="synthetic mode only: number of macro feature slots (default 1 = the historic "
                         "single-slot panel; use >=5 to give the overlay pool de-correlated breadth so "
                         "the opt-in weak-signal cohort gate can form a cohort)")
    ap.add_argument("--max-proposals", type=int, default=32)
    ap.add_argument("--proposer", choices=("library", "llm"), default="library",
                    help="library = deterministic offline seed bank (default, no cost/network, "
                         "bit-reproducible). llm = live LLM-backed proposer (spec Part A2); needs "
                         "ANTHROPIC_API_KEY, opt-in only — never the default.")
    ap.add_argument("--llm-model", default="claude-sonnet-5",
                    help="model id for --proposer llm (stamped into the manifest agent_model_id).")
    ap.add_argument("--est-tokens-per-tick", type=int, default=12000,
                    help="CR-7 per-tick token estimate charged to the tick budget for --proposer llm "
                         "(the offline library proposer always charges 0).")
    ap.add_argument("--max-candidates", type=int, default=256, help="per-tick candidate cap (CR-7)")
    ap.add_argument("--start-ts", default=None,
                    help="ISO timestamp base for reproduce/explicit replay only (CR-2/CR-8); nights step "
                         "+1d but are CAPPED at wall-clock now (never future-dated). DEFAULT (omitted): "
                         "each tick stamps the actual UTC instant it runs.")
    ap.add_argument("--out", default=str(ROOT / "results" / "crucible_orchestrator"))
    ap.add_argument("--force", action="store_true",
                    help="run even when generation.enabled is false in the gates")
    ap.add_argument("--no-lockbox", action="store_true",
                    help="disable the CR-8 forward-incubation lockbox (pure P3 byte-identical mode; "
                         "used by crucible reproduce / testing)")
    ap.add_argument("--cohort-config", default=str(DEFAULT_COHORT_GATES),
                    help="Phase-4 weak-signal cohort gates YAML (kept separate from --config so the "
                         "frozen funnel gates_hash is untouched; disabled by default in the file)")
    ap.add_argument("--no-cohort", action="store_true",
                    help="detach the weak-signal cohort gate entirely (pure pre-cohort byte-identical "
                         "path; used by crucible reproduce / testing)")
    ap.add_argument("--no-altdata-slots", action="store_true",
                    help="real mode only: skip bridging macro/positioning/fundamental connector "
                         "series into Panel feature slots (mine the cross-sectional bank only)")
    ap.add_argument("--power-gates", default=str(DEFAULT_POWER_GATES),
                    help="NOW-5 substrate-power guard YAML (separate from --config so the frozen funnel "
                         "gates_hash is untouched); stamps each tick's implied MDE and can refuse an "
                         "underpowered mine. Disabled gracefully if it or the calibration sweep is absent.")
    ap.add_argument("--force-underpowered", action="store_true",
                    help="mine even when the power guard's action is 'refuse' (distinct from --force, "
                         "which bypasses generation.enabled)")
    ap.add_argument("--no-power-guard", action="store_true",
                    help="detach the substrate-power stamp/guard entirely (no tick power columns)")
    ap.add_argument("--no-governance", action="store_true",
                    help="skip the NOW-9 CLEARED->human handoff epilogue (used by reproduce/testing so "
                         "no governance.db / notifications are written)")
    args = ap.parse_args()
    if args.proposer == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("--proposer llm requires ANTHROPIC_API_KEY in the environment (fails closed, like "
                  "the FRED/EDGAR credentials) — not set, aborting before the (possibly slow) build.")
        return 1

    cfg, ek = load_generation_config(args.config)
    meta = load_generation_meta(args.config)
    if not meta["enabled"] and not args.force:
        log.warning("generation.enabled is false in %s — no-op. Pass --force to run anyway.",
                    args.config)
        return 0

    if args.no_power_guard:
        power_gate, sweep, sweep_hash = None, None, ""
    else:
        power_gate, sweep, sweep_hash = _load_power_guard(args.power_gates, force=args.force_underpowered)
    sub, catalog = _build_substrate(args, cfg, ek, meta, sweep, sweep_hash)
    out_dir = Path(args.out) / args.mode
    store = OrchestratorStore(out_dir / "orchestrator.db")

    tick_stamps = _tick_timestamps(args.start_ts, args.nights)
    summaries = []
    try:
        for i in range(args.nights):
            tick_ts = tick_stamps[i]
            budget = TickBudget(max_candidates=args.max_candidates)   # a fresh budget per tick (CR-7)
            res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=args.config,
                                        tick_ts=tick_ts, budget=budget, out_dir=out_dir,
                                        power_gate=power_gate)
            o = res.outcomes[0]
            summaries.append({
                "night": i + 1, "tick_ts": tick_ts, "dirty": o.dirty, "mined": o.mined,
                "status": o.status, "reason": o.reason, "n_preregistered": o.n_preregistered,
                "n_scored": o.n_scored, "n_promising": o.n_promising, "fdr_num_tests": o.fdr_num_tests,
                "fdr_charged_total": round(o.fdr_charged_total, 8), "burst_target": o.burst_target,
                "budget_breached": o.budget_breached,
                # CR-8 lockbox (P4): empty/zero when the substrate has no lockbox (--no-lockbox).
                "n_enrolled": o.n_enrolled, "n_incubating": o.n_incubating,
                "n_cleared": o.n_cleared, "n_rejected": o.n_rejected})
            log.info("night %d/%d ts=%s dirty=%s mined=%s status=%s promising=%d fdr_tests=%d "
                     "lockbox(incub=%d cleared=%d rejected=%d) (%s)",
                     i + 1, args.nights, tick_ts, o.dirty, o.mined, o.status, o.n_promising,
                     o.fdr_num_tests, o.n_incubating, o.n_cleared, o.n_rejected, o.reason)
            # Drop a deterministic reproduce recipe next to each mined synthetic manifest (spec §5, P5).
            # Only synthetic mode is bit-reproducible offline; a live/networked substrate writes none.
            # An LLM proposer is non-bit-reproducible even on synthetic (its reproduce contract is
            # "replay the recorded specs", not "re-call"), so no recipe is written for it.
            if o.mined and args.mode == "synthetic" and args.proposer == "library":
                _write_recipe(out_dir, sub.substrate_id, tick_ts, args)
        # NOW-9: hand off any newly-CLEARED survivors to the human Tier-2 gate — lockbox still open here.
        if summaries and sub.lockbox is not None and not args.no_governance:
            _run_governance_epilogue(sub, out_dir, summaries[-1]["tick_ts"])
    finally:
        # NOW-2 (C9-11): always persist the summary of whatever nights completed AND release DB handles,
        # even if a night raised — a half-run must not leave an empty summary + leaked connections.
        (out_dir / "orchestrator_summary.json").write_text(
            json.dumps({"advisory": "harness caps at PROMISING; deploy read requires a Tier-2 audit",
                        "crucible_version": CRUCIBLE_VERSION, "mode": args.mode, "nights": args.nights,
                        "substrate_id": sub.substrate_id, "ticks": summaries}, indent=2),
            encoding="utf-8")
        sub.ledger.close()
        if sub.lockbox is not None:
            sub.lockbox.close()
        catalog.close()
        store.close()

    n_mined = sum(1 for s in summaries if s["mined"])
    log.info("done: %d nights, mined %d (only when dirty), %d PROMISING total -> %s",
             args.nights, n_mined, sum(s["n_promising"] for s in summaries), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
