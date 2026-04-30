#!/usr/bin/env python3
"""Launch live/paper trading for GMGP1-v5 SAC on Gold futures via Interactive Brokers.

Requires IB Gateway (or TWS) running locally.
    Paper trading: port 4002
    Live trading:  port 4001

Usage:
    # Paper trading (default)
    python scripts/run_live_ib.py --config configs/live_gmgp1_gc_ib.yaml

    # Live trading (requires explicit flag)
    python scripts/run_live_ib.py --config configs/live_gmgp1_gc_ib.yaml --mainnet

    # Dry run (log actions, no orders)
    python scripts/run_live_ib.py --config configs/live_gmgp1_gc_ib.yaml --dry-run
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
logger = logging.getLogger("run_live_ib")


def validate_config(config: dict, args) -> dict:
    """Validate and patch config with CLI overrides."""

    # --- Checkpoint existence ---
    checkpoint_path = config.get("agent", {}).get("checkpoint_path", "")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Set agent.checkpoint_path in the config to your L1 checkpoint.")
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
                "  |   Using IB Gateway port %d (LIVE)              |\n"
                "  +================================================+",
                exchange_cfg.get("live_port", 4001),
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
    """Instantiate all live trading components for IB Gold futures."""
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine
    from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder, resolve_norm_warmup_path
    from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
        CryptoRiskConfig,
        CryptoRiskManager,
    )
    from finrl_pro_ds.futures.execution.ib_futures_broker import IBFuturesBroker
    from finrl_pro_ds.futures.live.cme_bar_clock import CMEBarClock
    from finrl_pro_ds.futures.live.cme_calendar import CMEGlobexCalendar

    # --- Agent ---
    agent_cfg = config.get("agent", {})
    network_cfg = config.get("network", {})
    sac_cfg = config.get("agents", {}).get("sac", {})

    agent = SACAgent(
        network_config=network_cfg,
        device=agent_cfg.get("device", "cpu"),
        torch_compile=False,
        **{k: v for k, v in sac_cfg.items() if k not in ("checkpoint_path",)},
    )
    agent.load(agent_cfg["checkpoint_path"])
    agent.actor.eval()
    logger.info(f"Agent loaded from {agent_cfg['checkpoint_path']}")

    # --- Broker ---
    ex_cfg = config.get("exchange", {})
    contract_cfg = config.get("contract", {})
    port = ex_cfg.get("live_port", 4001) if not ex_cfg.get("testnet", True) else ex_cfg.get("paper_port", 4002)

    broker = IBFuturesBroker(
        host=ex_cfg.get("host", "127.0.0.1"),
        port=port,
        client_id=ex_cfg.get("client_id", 1),
        symbol=contract_cfg.get("symbol", "MGC"),
        order_type=contract_cfg.get("order_type", "LMT"),
        limit_offset_ticks=contract_cfg.get("limit_offset_ticks", 2),
        market_fallback_timeout=contract_cfg.get("market_fallback_timeout", 30.0),
        commission_per_side=contract_cfg.get("commission_per_side", 0.62),
        roll_days_before_expiry=contract_cfg.get("roll_days_before_expiry", 5),
        market_data_timeout=contract_cfg.get("market_data_timeout", 10.0),
    )

    # --- Observation Builder (reused from crypto, asset-agnostic) ---
    feat_cfg = config.get("features", {})
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

    # --- Bar Clock (CME-aware) ---
    clock_cfg = config.get("bar_clock", {})
    calendar = CMEGlobexCalendar()
    bar_clock = CMEBarClock(
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
        execution_delay_seconds=clock_cfg.get("execution_delay_seconds", 10.0),
        max_late_seconds=clock_cfg.get("max_late_seconds", 60.0),
        calendar=calendar,
    )

    # --- Risk Manager (reused from crypto, position-fraction-based) ---
    risk_cfg = config.get("risk", {})
    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 12),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        max_net_short_exposure=risk_cfg.get("max_net_short_exposure", -1.0),
        min_effective_bets=risk_cfg.get("min_effective_bets", 1.0),
        daily_turnover_limit=risk_cfg.get("daily_turnover_limit", 4.0),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 999.0),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.20),
        static_peak=risk_cfg.get("static_peak", False),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", False),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
    ))

    # --- Data Loader ---
    # IBDataLoader needs a connected IB instance. We pass None here
    # and wire it up after broker.connect() in main_async().
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
    """Async main: connect IB, wire data loader, start engine."""
    from finrl_pro_ds.futures.data.ib_data_loader import IBDataLoader

    engine, broker = build_components(config)

    # Connect IB first (broker needs IB for contract qualification)
    await broker.connect()

    # Now create the data loader with the live IB connection
    loader = IBDataLoader(
        ib=broker._ib,
        contract_manager=broker._contract_manager,
    )
    engine.loader = loader

    # Start the trading loop
    await engine.start()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gold futures paper/live trading via IB",
    )
    parser.add_argument("--config", required=True, help="Path to live trading YAML config")
    parser.add_argument(
        "--overlay", action="append", default=None,
        help="Deploy overlay under configs/deploy/ (e.g. 'ftmo/step1'). "
             "Repeat for multiple; later wins. Same allowlist as "
             "deploy_bare_metal --overlay. Falls back to STRATEGY_OVERLAY "
             "env var if --overlay not given.",
    )
    parser.add_argument("--mainnet", action="store_true", help="Use live trading (real money). Default: paper.")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing orders.")
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

    # Build and run
    logger.info(
        f"Starting IB Gold futures trading engine\n"
        f"  Config: {config_path}\n"
        f"  Overlays: {overlay_specs or 'none'}\n"
        f"  Contract: {config.get('contract', {}).get('symbol', 'MGC')}\n"
        f"  Mode: {'PAPER' if config.get('exchange', {}).get('testnet', True) else 'LIVE'}\n"
        f"  Dry run: {config.get('dry_run', False)}",
    )
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
        # Force PID 1 exit even if non-daemon threads or wandb-core
        # subprocess is still alive. Docker restart: unless-stopped
        # auto-recovers the strategy. Fixes S489 orphan-netns hang.
        logger.info(f"run_live_ib exiting with code {exit_code}")
        os._exit(exit_code)


if __name__ == "__main__":
    main()
