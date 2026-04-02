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
    checkpoint_path = config.get("agent", {}).get("checkpoint_path", "")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Set agent.checkpoint_path in the config to your L1 checkpoint.")
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
    from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder
    from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
        CryptoRiskConfig,
        CryptoRiskManager,
    )

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
    obs_builder = LiveObsBuilder(
        scales=feat_cfg.get("scales", [15, 60, 240]),
        window_size=feat_cfg.get("window_size", 30),
        norm_span=feat_cfg.get("norm_span", 120),
        n_features=feat_cfg.get("features_per_scale", 8),
        bootstrap_bars=feat_cfg.get("bootstrap_bars", 30_000),
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
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 999.0),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.10),
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

    # Connect cTrader first (authenticates, resolves symbol, subscribes to spots)
    await broker.connect()

    # Now create the data loader with the live cTrader connection
    loader = CTraderDataLoader(
        client=broker._client,
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

    # Validate and patch
    config = validate_config(config, args)

    contract = config.get("contract", {}).get("symbol", "XAUUSD")
    mode = "PAPER (IC Markets demo)" if config.get("exchange", {}).get("testnet", True) else "LIVE (FTMO)"
    logger.info(
        f"Starting cTrader Gold CFD trading engine\n"
        f"  Config: {config_path}\n"
        f"  Symbol: {contract}\n"
        f"  Mode: {mode}\n"
        f"  Dry run: {config.get('dry_run', False)}"
    )

    # Install Twisted asyncio reactor BEFORE any Twisted imports.
    # This must happen before build_components() which imports ctrader_broker.
    try:
        from twisted.internet import asyncioreactor
        asyncioreactor.install()
    except Exception:
        pass  # May already be installed

    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
