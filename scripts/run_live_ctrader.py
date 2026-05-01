#!/usr/bin/env python3
"""Launch live/paper trading for GMGP1-XAUUSD SAC on Gold CFD via cTrader.

Requires cTrader Open API credentials in environment variables:
    CTRADER_CLIENT_ID      — OAuth 2.0 client ID
    CTRADER_CLIENT_SECRET  — OAuth 2.0 client secret
    CTRADER_ACCESS_TOKEN   — OAuth 2.0 access token
    CTRADER_ACCOUNT_ID     — Numeric trading account ID

    Paper trading: IC Markets demo account
    Live trading:  FTMO challenge account

Usage:
    # Paper trading (default)
    python scripts/run_live_ctrader.py --config configs/live_gmgp1_xauusd_ctrader.yaml

    # Live trading (requires explicit flag)
    python scripts/run_live_ctrader.py --config configs/live_gmgp1_xauusd_ctrader.yaml --mainnet

    # Dry run (log actions, no orders)
    python scripts/run_live_ctrader.py --config configs/live_gmgp1_xauusd_ctrader.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_live_ctrader")


# Required cTrader env vars
_REQUIRED_ENV_VARS = [
    "CTRADER_CLIENT_ID",
    "CTRADER_CLIENT_SECRET",
    "CTRADER_ACCESS_TOKEN",
    "CTRADER_ACCOUNT_ID",
]


def validate_config(config: dict, args) -> dict:
    """Validate and patch config with CLI overrides."""

    # --- Checkpoint existence ---
    # Three modes:
    #   - solo (agent.checkpoint_path)
    #   - v2.3 atomic-swap bundle (agent.ensemble.bundle_path → ensemble_v{N}.tar.gz)
    #   - legacy ensemble (agent.ensemble.{seeds, checkpoint_pattern, aggregation_rule})
    # The legacy form remains supported for v2.1/v2.2 retro-apply paper deploys per
    # docs/protocol_v2.md §4. New L1-retrain promotions MUST emit a bundle.
    agent_cfg = config.get("agent", {})
    ensemble_cfg = agent_cfg.get("ensemble")
    if ensemble_cfg and ensemble_cfg.get("bundle_path"):
        # v2.3 path: defer extraction + SHA256 verification to build_components,
        # so failures abort the runner with the live engine's logging context
        # rather than during config validation. Just check the bundle file
        # exists at this stage.
        bundle_path = Path(ensemble_cfg["bundle_path"])
        if not bundle_path.exists():
            logger.error(f"v2.3 swap bundle not found: {bundle_path}")
            sys.exit(1)
        logger.info(f"v2.3 swap bundle declared: {bundle_path}")
    elif ensemble_cfg:
        import glob
        seeds = list(ensemble_cfg.get("seeds", []))
        pattern = ensemble_cfg.get("checkpoint_pattern", "")
        if not seeds or not pattern:
            logger.error(
                "agent.ensemble requires either 'bundle_path' (v2.3) or "
                "'seeds' + 'checkpoint_pattern' (legacy retro-apply)"
            )
            sys.exit(1)
        resolved: dict = {}
        for s in seeds:
            # Let the caller choose which fold to load via a fixed 'fold' key
            # (defaults to 7 = fold_07, the most-recent training window).
            fold = int(ensemble_cfg.get("fold", 7))
            glob_s = pattern.format(seed=s, fold=fold)
            matches = sorted(glob.glob(glob_s))
            if not matches:
                logger.error(f"No checkpoint matches for seed {s} at {glob_s}")
                sys.exit(1)
            if len(matches) > 1:
                logger.warning(f"Seed {s} has {len(matches)} matches; picking last: {matches[-1]}")
            resolved[s] = matches[-1]
        ensemble_cfg["_resolved_paths"] = resolved
        logger.info(f"Legacy ensemble checkpoints resolved: {resolved}")
    else:
        checkpoint_path = agent_cfg.get("checkpoint_path", "")
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Set agent.checkpoint_path (solo) or agent.ensemble (multi-seed).")
            sys.exit(1)

    # --- cTrader credentials ---
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        logger.error(
            f"Missing cTrader credentials: {', '.join(missing)}\n"
            f"  Set these environment variables before running.\n"
            f"  Obtain OAuth tokens at https://openapi.ctrader.com/apps/auth"
        )
        sys.exit(1)

    # --- Mainnet safety ---
    exchange_cfg = config.setdefault("exchange", {})
    if args.mainnet:
        if config.get("safety", {}).get("require_explicit_mainnet", True):
            exchange_cfg["testnet"] = False
            logger.warning(
                "\n"
                "  +================================================+\n"
                "  |   WARNING: LIVE MODE — REAL MONEY AT RISK       |\n"
                "  |   Trading on FTMO LIVE account                  |\n"
                "  +================================================+",
            )
    else:
        exchange_cfg["testnet"] = True

    # --- Dry run ---
    if args.dry_run:
        config["dry_run"] = True
        logger.info("DRY RUN mode: no orders will be executed")
    else:
        config.setdefault("dry_run", False)

    # --- Device ---
    import torch

    device = config.get("agent", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config["agent"]["device"] = "cpu"

    return config


def build_components(config: dict):
    """Instantiate all live trading components for cTrader XAUUSD CFD."""
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    from finrl_pro_ds.cfd.execution.ctrader_broker import CTraderBroker
    from finrl_pro_ds.cfd.live.cfd_bar_clock import CFDBarClock
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine
    from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder, resolve_norm_warmup_path
    from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
        CryptoRiskConfig,
        CryptoRiskManager,
    )

    # --- Agent (solo or ensemble) ---
    agent_cfg = config.get("agent", {})
    network_cfg = config.get("network", {})
    sac_cfg = config.get("agents", {}).get("sac", {})
    device = agent_cfg.get("device", "cpu")
    sac_kwargs = {k: v for k, v in sac_cfg.items() if k not in ("checkpoint_path",)}

    ensemble_cfg = agent_cfg.get("ensemble")
    if ensemble_cfg and ensemble_cfg.get("bundle_path"):
        # v2.3 atomic-swap bundle path: extract + SHA256-verify, then load N
        # SACAgents from the bundle's checkpoints. EnsembleAgent constructor
        # is shared with the legacy path below.
        from finrl_pro_ds.agents.sac.ensemble_agent import EnsembleAgent
        from finrl_pro_ds.live import extract_and_verify_bundle, BundleIntegrityError

        bundle_path = ensemble_cfg["bundle_path"]
        try:
            bundle = extract_and_verify_bundle(
                bundle_path,
                require_normalizers=ensemble_cfg.get("require_normalizers", True),
            )
        except BundleIntegrityError as exc:
            logger.error(f"v2.3 bundle integrity failure: {exc}")
            raise
        seeds = list(bundle.seeds)
        loaded_agents = []
        for s in seeds:
            a = SACAgent(
                network_config=network_cfg, device=device, torch_compile=False,
                **sac_kwargs,
            )
            a.load(str(bundle.checkpoint_paths[s]))
            a.actor.eval()
            loaded_agents.append(a)
        # Manifest-declared seed_pfs win over any caller override; live config
        # may still pin an aggregation_rule override (e.g. swap to ens_mean
        # without rebuilding the bundle), but the canonical rule is in the
        # manifest and we log if the caller diverges.
        rule = ensemble_cfg.get("aggregation_rule") or bundle.chosen_rule
        if rule != bundle.chosen_rule:
            logger.warning(
                f"aggregation_rule override: config={rule!r} differs from "
                f"manifest.chosen_rule={bundle.chosen_rule!r}"
            )
        deadband = float(ensemble_cfg.get("deadband", bundle.deadband))
        agent = EnsembleAgent(
            agents=loaded_agents,
            seeds=seeds,
            aggregation_rule=rule,
            deadband=deadband,
            seed_pfs=bundle.seed_pfs,
        )
        logger.info(
            f"EnsembleAgent loaded from v2.3 bundle: ws={bundle.workstream} "
            f"v={bundle.version} seeds={seeds} rule={rule} deadband={deadband}",
        )
    elif ensemble_cfg:
        from finrl_pro_ds.agents.sac.ensemble_agent import EnsembleAgent
        resolved = ensemble_cfg["_resolved_paths"]  # populated in validate_config
        seeds = list(ensemble_cfg.get("seeds", []))
        # Ensemble agents share `sac_kwargs` (config.agents.sac). This assumes all
        # ensemble seeds were trained with the same HPs, which is true for the
        # current SG-1 XAUUSD ensemble (top-3 seeds all ran trial-56 params). A
        # mixed-HP ensemble would need per-seed kwargs keyed by seed id.
        loaded_agents = []
        for s in seeds:
            a = SACAgent(
                network_config=network_cfg, device=device, torch_compile=False,
                **sac_kwargs,
            )
            a.load(resolved[s])
            a.actor.eval()
            loaded_agents.append(a)
        agent = EnsembleAgent(
            agents=loaded_agents,
            seeds=seeds,
            aggregation_rule=ensemble_cfg.get("aggregation_rule", "ens_agreement"),
            deadband=float(ensemble_cfg.get("deadband",
                config.get("trading", {}).get("deadband_threshold", 0.25))),
            seed_pfs={int(k): float(v) for k, v in (ensemble_cfg.get("seed_pfs") or {}).items()},
        )
        logger.info(
            f"EnsembleAgent loaded: {len(seeds)} seeds={seeds} "
            f"rule={ensemble_cfg.get('aggregation_rule', 'ens_agreement')} "
            f"deadband={agent.deadband}",
        )
    else:
        agent = SACAgent(
            network_config=network_cfg, device=device, torch_compile=False, **sac_kwargs,
        )
        agent.load(agent_cfg["checkpoint_path"])
        agent.actor.eval()
        logger.info(f"Agent loaded from {agent_cfg['checkpoint_path']}")

    # --- Broker ---
    ex_cfg = config.get("exchange", {})
    contract_cfg = config.get("contract", {})
    trading_cfg = config.get("trading", {})

    broker = CTraderBroker(
        testnet=ex_cfg.get("testnet", True),
        symbol=contract_cfg.get("symbol", "XAUUSD"),
        lot_size=contract_cfg.get("lot_size", 100.0),
        min_lot=contract_cfg.get("min_lot", 0.01),
        tick_size=contract_cfg.get("tick_size", 0.01),
        leverage=contract_cfg.get("leverage", 30),
        order_type=contract_cfg.get("order_type", "MKT"),
        taker_fee=trading_cfg.get("taker_fee", 0.00015),
        market_fallback_timeout=contract_cfg.get("market_fallback_timeout", 10.0),
    )

    # --- Observation Builder (reused from crypto, asset-agnostic) ---
    feat_cfg = config.get("features", {})
    # S509: per-scale norm warmup buffer reproduces training freeze-and-restart
    # EMA-Z (vs live's legacy rolling EMA-Z). Resolved either explicitly via
    # `features.norm_warmup_path` or relative to the first ensemble checkpoint.
    norm_warmup_path = resolve_norm_warmup_path(config)
    obs_builder = LiveObsBuilder(
        scales=feat_cfg.get("scales", [15, 60, 240]),
        window_size=feat_cfg.get("window_size", 30),
        norm_span=feat_cfg.get("norm_span", 120),
        n_features=feat_cfg.get("features_per_scale", 8),
        bootstrap_bars=feat_cfg.get("bootstrap_bars", 30_000),
        obs_mode=feat_cfg.get("obs_mode", "window"),
        summary_feature_indices=feat_cfg.get("summary_feature_indices"),
        norm_warmup_path=norm_warmup_path,
    )

    # --- Bar Clock (CFD 24h market) ---
    clock_cfg = config.get("bar_clock", {})
    bar_clock = CFDBarClock(
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
        execution_delay_seconds=clock_cfg.get("execution_delay_seconds", 5.0),
        max_late_seconds=clock_cfg.get("max_late_seconds", 30.0),
    )

    # --- Risk Manager (reused from crypto, position-fraction-based) ---
    risk_cfg = config.get("risk", {})
    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.08),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 12),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        max_net_short_exposure=risk_cfg.get("max_net_short_exposure", -1.0),
        min_effective_bets=risk_cfg.get("min_effective_bets", 1.0),
        daily_turnover_limit=risk_cfg.get("daily_turnover_limit", 4.0),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 999.0),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.10),
        static_peak=risk_cfg.get("static_peak", False),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", False),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
    ))

    # --- Data Loader ---
    # CTraderDataLoader needs the broker's client, created after connect()
    loader = None  # Placeholder

    # --- Engine ---
    engine = LiveTradingEngine(
        agent=agent,
        broker=broker,
        obs_builder=obs_builder,
        risk_manager=risk_manager,
        bar_clock=bar_clock,
        loader=loader,
        config=config,
    )

    # --- PRISM Overlay (optional L2 position sizing) ---
    prism_cfg = config.get("prism", {})
    if prism_cfg.get("enabled", False):
        from finrl_pro_ds.crypto.live.prism_overlay import PRISMOverlay
        engine._prism_overlay = PRISMOverlay(prism_cfg)
        logger.info(f"PRISM L2 overlay enabled: {prism_cfg.get('base_url', 'http://prism-api:8001')}")

    return engine, broker


async def main_async(config: dict) -> None:
    """Async main: connect cTrader, wire data loader, start engine."""
    from finrl_pro_ds.cfd.data.ctrader_data_loader import CTraderDataLoader

    engine, broker = build_components(config)

    # FIX CT-08: Connect with retry logic to prevent Docker restart-loops
    # when the cTrader server has a ghost session or transient issue.
    max_connect_attempts = 5
    for attempt in range(1, max_connect_attempts + 1):
        try:
            # Check if a ghost session was detected on a prior attempt
            if broker._already_logged_in_seen:
                wait_secs = broker._ALREADY_LOGGED_IN_WAIT
                logger.warning(
                    "CT-08: Ghost session detected — waiting %ds before retry",
                    wait_secs,
                )
                broker._already_logged_in_seen = False
                await asyncio.sleep(wait_secs)

            await broker.connect()
            break  # Success
        except Exception as exc:
            exc_str = str(exc)
            logger.error(
                "CT-08: connect() attempt %d/%d failed: %s",
                attempt, max_connect_attempts, exc,
            )
            if attempt == max_connect_attempts:
                raise

            # FIX AUD: Special handling for server-side routing failures.
            # If the IC Markets server is down, don't spam TCP connections.
            if "CANT_ROUTE_REQUEST" in exc_str:
                delay = 300.0  # 5 minutes
                logger.warning("AUD: Server routing failure — waiting 5 min before retry")
            else:
                # Exponential backoff: 5, 10, 20, 40s
                delay = 5.0 * (2 ** (attempt - 1))

            logger.info("CT-08: Retrying in %.0fs...", delay)
            await asyncio.sleep(delay)

    # Now create the data loader with the live cTrader connection
    loader = CTraderDataLoader(
        broker=broker,
        account_id=broker._account_id,
        symbol_id=broker._symbol_id,
        symbol_digits=broker._symbol_digits,
    )
    engine.loader = loader

    # Start the trading loop
    await engine.start()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gold CFD paper/live trading via cTrader",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to live trading YAML config",
    )
    parser.add_argument(
        "--overlay", action="append", default=None,
        help="Deploy overlay under configs/deploy/ (e.g. 'ftmo/step1'). "
             "Repeat for multiple; later wins. Same allowlist as "
             "deploy_bare_metal --overlay. Falls back to STRATEGY_OVERLAY "
             "env var if --overlay not given.",
    )
    parser.add_argument(
        "--mainnet", action="store_true",
        help="Use live trading (FTMO challenge). Default: paper (IC Markets demo).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions without executing orders.",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply deploy overlays (Step 5 prop-firm decoupling). CLI flag wins
    # over STRATEGY_OVERLAY env var so an interactive operator can override
    # what the docker-compose bakes in.
    from finrl_pro_ds.config_utils import apply_overlays, parse_overlay_env
    overlay_specs = args.overlay or parse_overlay_env(os.environ.get("STRATEGY_OVERLAY"))
    if overlay_specs:
        project_root = Path(__file__).resolve().parents[1]
        overlay_root = project_root / "configs" / "deploy"
        allowlist_path = overlay_root / "ALLOWLIST.yaml"
        config = apply_overlays(
            config,
            overlay_specs,
            overlay_root=overlay_root,
            allowlist_path=allowlist_path,
        )
        logger.info(
            "Applied %d deploy overlay(s): %s",
            len(overlay_specs), ", ".join(overlay_specs),
        )

    # Validate and patch
    config = validate_config(config, args)

    contract = config.get("contract", {}).get("symbol", "XAUUSD")
    mode = "PAPER (IC Markets demo)" if config.get("exchange", {}).get("testnet", True) else "LIVE (FTMO)"
    logger.info(
        f"Starting cTrader Gold CFD trading engine\n"
        f"  Config: {config_path}\n"
        f"  Overlays: {overlay_specs or 'none'}\n"
        f"  Symbol: {contract}\n"
        f"  Mode: {mode}\n"
        f"  Dry run: {config.get('dry_run', False)}"
    )

    # Start Twisted reactor in a daemon thread. The default reactor handles
    # TCP/TLS I/O for the ctrader-open-api Client. We cannot use asyncioreactor
    # because asyncio.run() creates a new event loop, leaving the reactor bound
    # to a stale loop (deferreds never fire → connection timeout).
    # The polling bridge in CTraderBroker._deferred_to_future already handles
    # cross-thread deferred → asyncio communication safely.
    import threading

    from twisted.internet import reactor

    reactor_thread = threading.Thread(
        target=reactor.run, args=(False,), daemon=True
    )
    reactor_thread.start()

    exit_code = 0
    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_code = 130
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception:
        logger.exception("Trading engine failed")
        exit_code = 1
    finally:
        # Force PID 1 exit even if non-daemon threads (twisted reactor,
        # wandb-core subprocess, ccxt aiohttp pools) are still alive.
        # Docker `restart: unless-stopped` auto-recovers the strategy.
        # Mirrors the run_live_ib.py S489 fix; closes the orphan-netns
        # generalization gap (S498).
        logger.info(f"run_live_ctrader exiting with code {exit_code}")
        os._exit(exit_code)


if __name__ == "__main__":
    main()
