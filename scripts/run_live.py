#!/usr/bin/env python3
"""Launch live/paper trading for trained SAC agents.

Usage:
    # Paper trading (testnet, default)
    python scripts/run_live.py --config configs/live_gmgp1_btc_bybit.yaml

    # Live trading (mainnet, requires explicit flag)
    python scripts/run_live.py --config configs/live_gmgp1_btc_bybit.yaml --mainnet

    # Dry run (log actions, no orders executed)
    python scripts/run_live.py --config configs/live_gmgp1_btc_bybit.yaml --dry-run
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
logger = logging.getLogger("run_live")


def validate_config(config: dict, args) -> dict:
    """Validate and patch config with CLI overrides."""

    # --- Checkpoint existence ---
    checkpoint_path = config.get("agent", {}).get("checkpoint_path", "")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # --- Mainnet safety ---
    exchange_cfg = config.setdefault("exchange", {})
    if args.mainnet:
        if not config.get("safety", {}).get("require_explicit_mainnet", True):
            logger.error(
                "safety.require_explicit_mainnet is false — refusing mainnet launch. "
                "Set to true in config to confirm intent."
            )
            sys.exit(1)
        exchange_cfg["testnet"] = False
        logger.warning(
            "\n"
            "  ╔══════════════════════════════════════════════╗\n"
            "  ║   ⚠  MAINNET MODE — REAL MONEY AT RISK  ⚠   ║\n"
            "  ╚══════════════════════════════════════════════╝",
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
    """Instantiate all live trading components from config."""
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    from finrl_pro_ds.crypto.data.crypto_loader import CryptoLoader
    from finrl_pro_ds.crypto.execution.exchange_perp_broker import ExchangePerpBroker
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

    # FIX AUD-C03: Force torch_compile=False for live inference
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
    broker = ExchangePerpBroker(
        exchange=ex_cfg.get("name", "bybit"),
        testnet=ex_cfg.get("testnet", True),
        demo=ex_cfg.get("demo", False),
        order_type=ex_cfg.get("order_type", "limit"),
        limit_offset_pct=ex_cfg.get("limit_offset_pct", 0.0005),
        market_fallback_timeout=ex_cfg.get("market_fallback_timeout", 60.0),
    )

    # --- Observation Builder ---
    feat_cfg = config.get("features", {})
    obs_builder = LiveObsBuilder(
        scales=feat_cfg.get("scales", [15, 60, 240]),
        window_size=feat_cfg.get("window_size", 30),
        norm_span=feat_cfg.get("norm_span", 120),
        n_features=feat_cfg.get("features_per_scale", 8),
        bootstrap_bars=feat_cfg.get("bootstrap_bars", 30_000),
        obs_mode=feat_cfg.get("obs_mode", "window"),
        summary_feature_indices=feat_cfg.get("summary_feature_indices"),
    )

    # --- Bar Clock ---
    clock_cfg = config.get("bar_clock", {})
    bar_clock = BarClock(
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 15),
        execution_delay_seconds=clock_cfg.get("execution_delay_seconds", 5.0),
        max_late_seconds=clock_cfg.get("max_late_seconds", 30.0),
    )

    # --- Risk Manager ---
    risk_cfg = config.get("risk", {})
    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 12),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 0.001),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.10),
        static_peak=risk_cfg.get("static_peak", False),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", False),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
    ))

    # --- Data Loader ---
    loader = CryptoLoader(
        exchange=ex_cfg.get("data_exchange", ex_cfg.get("name", "binance")),
        market_type="swap",
    )

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

    return engine


def main():
    parser = argparse.ArgumentParser(description="Launch live/paper trading")
    parser.add_argument("--config", required=True, help="Path to live trading YAML config")
    parser.add_argument(
        "--overlay", action="append", default=None,
        help="Deploy overlay under configs/deploy/ (e.g. 'velotrade/step1'). "
             "Repeat for multiple; later wins. Same allowlist as "
             "deploy_bare_metal --overlay. Falls back to STRATEGY_OVERLAY "
             "env var if --overlay not given.",
    )
    parser.add_argument("--mainnet", action="store_true", help="Use mainnet (real money). Default: testnet.")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing orders.")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
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

    # Build components and run
    engine = build_components(config)

    logger.info(
        "Starting live trading engine (overlays: %s)",
        overlay_specs or "none",
    )
    asyncio.run(engine.start())


if __name__ == "__main__":
    main()
