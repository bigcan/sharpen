#!/usr/bin/env python3
"""Launch live/paper trading for Velotrade BTC SAC via DXtrade.

Requires DXtrade credentials in environment variables:
    DXTRADE_SERVER      — DXtrade server hostname
    DXTRADE_USERNAME    — Account username
    DXTRADE_PASSWORD    — Account password
    DXTRADE_ACCOUNT_ID  — Trading account ID

    Paper trading: Velotrade demo account
    Live trading:  Velotrade challenge account

Usage:
    # Paper trading (default)
    python scripts/run_live_dxtrade.py --config configs/live_velotrade_btc.yaml

    # Live trading (Velotrade challenge — requires explicit flag)
    python scripts/run_live_dxtrade.py --config configs/live_velotrade_btc.yaml --mainnet

    # Dry run (log actions, no orders)
    python scripts/run_live_dxtrade.py --config configs/live_velotrade_btc.yaml --dry-run
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
logger = logging.getLogger("run_live_dxtrade")


# Optional DXtrade env vars (not required for dry-run or stub mode)
_DXTRADE_ENV_VARS = [
    "DXTRADE_SERVER",
    "DXTRADE_USERNAME",
    "DXTRADE_PASSWORD",
    "DXTRADE_ACCOUNT_ID",
]


def validate_config(config: dict, args) -> dict:
    """Validate and patch config with CLI overrides."""

    # --- Checkpoint existence ---
    checkpoint_path = config.get("agent", {}).get("checkpoint_path", "")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Set agent.checkpoint_path in the config to your trained checkpoint.")
        sys.exit(1)

    # --- DXtrade credentials (warn if missing, allow dry-run without) ---
    missing = [v for v in _DXTRADE_ENV_VARS if not os.environ.get(v)]
    if missing and not args.dry_run:
        logger.warning(
            f"Missing DXtrade credentials: {', '.join(missing)}\n"
            f"  Running in stub mode — orders will not be executed.\n"
            f"  Set these environment variables for live/paper trading."
        )

    # --- Mainnet safety ---
    exchange_cfg = config.setdefault("exchange", {})
    if args.mainnet:
        if config.get("safety", {}).get("require_explicit_mainnet", True):
            exchange_cfg["testnet"] = False
            logger.warning(
                "\n"
                "  +================================================+\n"
                "  |   WARNING: LIVE MODE — REAL CHALLENGE ACCOUNT   |\n"
                "  |   Trading on Velotrade CHALLENGE account        |\n"
                "  |   Challenge fees are non-refundable             |\n"
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
    """Instantiate all live trading components for DXtrade BTC."""
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    from finrl_pro_ds.crypto.execution.dxtrade_broker import DXtradePerpBroker
    from finrl_pro_ds.crypto.live.bar_clock import BarClock
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
    broker = DXtradePerpBroker(
        testnet=ex_cfg.get("testnet", True),
        order_type=ex_cfg.get("order_type", "market"),
        leverage=ex_cfg.get("leverage", 6),
        taker_fee=config.get("trading", {}).get("taker_fee", 0.0005),
    )

    # --- Observation Builder ---
    feat_cfg = config.get("features", {})
    obs_builder = LiveObsBuilder(
        scales=feat_cfg.get("scales", [15, 60, 240]),
        window_size=feat_cfg.get("window_size", 30),
        norm_span=feat_cfg.get("norm_span", 120),
        n_features=feat_cfg.get("features_per_scale", 8),
        bootstrap_bars=feat_cfg.get("bootstrap_bars", 30_000),
    )

    # --- Bar Clock ---
    clock_cfg = config.get("bar_clock", {})
    bar_clock = BarClock(
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
        execution_delay_seconds=clock_cfg.get("execution_delay_seconds", 5.0),
        max_late_seconds=clock_cfg.get("max_late_seconds", 30.0),
    )

    # --- Risk Manager (with EOD trailing drawdown for prop firm) ---
    risk_cfg = config.get("risk", {})
    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.08),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", True),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 0),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        max_net_short_exposure=risk_cfg.get("max_net_short_exposure", -1.0),
        max_gross_exposure=risk_cfg.get("max_gross_exposure", 6.0),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 999.0),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.15),
    ))

    # --- Data Loader (Binance for OHLCV, DXtrade for execution) ---
    loader = None  # Will be set after connect

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
        logger.info(f"PRISM L2 overlay enabled: {prism_cfg.get('base_url')}")

    return engine, broker


async def main_async(config: dict) -> None:
    """Async main: connect DXtrade, wire data loader, start engine."""
    from finrl_pro_ds.crypto.data.crypto_loader import CryptoLoader

    engine, broker = build_components(config)

    # Connect DXtrade (may be stub mode if no credentials)
    try:
        await broker.connect()
    except NotImplementedError:
        logger.warning("DXtrade broker in stub mode — dry-run only")
        config["dry_run"] = True

    # Create data loader (Binance for OHLCV data)
    ex_cfg = config.get("exchange", {})
    data_exchange = ex_cfg.get("data_exchange", "binance")
    asset = ex_cfg.get("asset", "BTC")
    loader = CryptoLoader(
        exchange=data_exchange,
        assets=[asset],
        market_type="futures",
    )
    engine.loader = loader

    # Start the trading loop
    await engine.start()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Velotrade BTC paper/live trading via DXtrade",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to live trading YAML config",
    )
    parser.add_argument(
        "--mainnet", action="store_true",
        help="Use challenge account (Velotrade). Default: demo.",
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

    asset = config.get("exchange", {}).get("asset", "BTC")
    mode = "DEMO" if config.get("exchange", {}).get("testnet", True) else "CHALLENGE"
    logger.info(
        f"Starting DXtrade live trading engine\n"
        f"  Config: {config_path}\n"
        f"  Asset: {asset}\n"
        f"  Mode: {mode} (Velotrade)\n"
        f"  Leverage: {config.get('exchange', {}).get('leverage', 6)}x\n"
        f"  Dry run: {config.get('dry_run', False)}"
    )

    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
