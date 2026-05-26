#!/usr/bin/env python3
"""Launch live/paper trading for SG-1 EURUSD SAC on EUR/USD CFD via OANDA.

Requires OANDA v20 credentials in environment variables:
    OANDA_API_TOKEN     — personal access token (Bearer auth)
    OANDA_ACCOUNT_ID    — trading account ID (e.g. 101-011-<ACCT_9>-001)
    OANDA_ENV           — "practice" (paper, default) or "live"

Per-strategy isolation is via OANDA_ACCOUNT_ID. The compose entry maps
``SG1_EURUSD_OANDA_ACCOUNT_ID`` -> ``OANDA_ACCOUNT_ID`` so multiple
strategies can share one OANDA personal token while targeting separate
sub-accounts.

Usage:
    # Paper trading (default; uses api-fxpractice.oanda.com)
    python scripts/run_live_oanda.py --config configs/live_sg1_eurusd_oanda.yaml

    # Live trading (requires explicit flag; uses api-fxtrade.oanda.com)
    python scripts/run_live_oanda.py --config configs/live_sg1_eurusd_oanda.yaml --mainnet

    # Dry run (log actions, no orders)
    python scripts/run_live_oanda.py --config configs/live_sg1_eurusd_oanda.yaml --dry-run

See `.agent/artifacts/oanda_broker_architecture.md` for the full design.
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
logger = logging.getLogger("run_live_oanda")


# Required OANDA env vars
_REQUIRED_ENV_VARS = [
    "OANDA_API_TOKEN",
    "OANDA_ACCOUNT_ID",
]


def validate_config(config: dict, args) -> dict:
    """Validate config and apply CLI overrides."""

    # --- Checkpoint / bundle existence (solo / v2.3 bundle / legacy ensemble) ---
    from finrl_pro_ds.live import resolve_agent_paths
    resolve_agent_paths(config, logger=logger)

    # --- OANDA credentials ---
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        logger.error(
            "Missing OANDA credentials: %s\n"
            "  Set these environment variables before running.\n"
            "  Generate token at OANDA Hub -> My Services -> Manage API Access.",
            ", ".join(missing),
        )
        sys.exit(1)

    # --- Mainnet safety ---
    # `testnet` in our exchange block follows the codebase convention:
    # True = paper/practice, False = live. Map --mainnet flag onto OANDA_ENV.
    exchange_cfg = config.setdefault("exchange", {})
    if args.mainnet:
        if config.get("safety", {}).get("require_explicit_mainnet", True):
            exchange_cfg["env"] = "live"
            exchange_cfg["testnet"] = False
            os.environ["OANDA_ENV"] = "live"
            logger.warning(
                "\n"
                "  +================================================+\n"
                "  |   WARNING: LIVE MODE — REAL MONEY AT RISK       |\n"
                "  |   Trading on OANDA LIVE account                 |\n"
                "  +================================================+",
            )
    else:
        exchange_cfg["env"] = "practice"
        exchange_cfg["testnet"] = True
        os.environ.setdefault("OANDA_ENV", "practice")

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
    """Instantiate all live trading components for OANDA EUR/USD CFD."""
    from finrl_pro_ds.cfd.data.oanda_data_loader import OandaDataLoader
    from finrl_pro_ds.cfd.execution.oanda_broker import OandaBroker
    from finrl_pro_ds.cfd.live.cfd_bar_clock import CFDBarClock
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine
    from finrl_pro_ds.crypto.live.live_obs_builder import (
        LiveObsBuilder,
        resolve_norm_warmup_path,
    )
    from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
        CryptoRiskConfig,
        CryptoRiskManager,
    )
    from finrl_pro_ds.live import build_agent

    # --- Agent (solo / v2.3 bundle / legacy ensemble) ---
    agent = build_agent(config, logger=logger)

    # --- Broker ---
    ex_cfg = config.get("exchange", {})
    contract_cfg = config.get("contract", {})
    trading_cfg = config.get("trading", {})

    broker = OandaBroker(
        env=ex_cfg.get("env", "practice"),
        symbol=contract_cfg.get("symbol", "EURUSD"),
        lot_size=contract_cfg.get("lot_size", 100_000.0),
        min_lot=contract_cfg.get("min_lot", 0.01),
        tick_size=contract_cfg.get("tick_size", 0.00001),
        leverage=contract_cfg.get("leverage", 1),
        order_type=contract_cfg.get("order_type", "MKT"),
        taker_fee=trading_cfg.get("taker_fee", 0.00005),
        market_fallback_timeout=contract_cfg.get("market_fallback_timeout", 10.0),
    )

    # --- Observation Builder (asset-agnostic; reused from crypto path) ---
    feat_cfg = config.get("features", {})
    norm_warmup_path = resolve_norm_warmup_path(config)
    obs_builder = LiveObsBuilder(
        scales=feat_cfg.get("scales", [3, 15, 60]),
        window_size=feat_cfg.get("window_size", 30),
        norm_span=feat_cfg.get("norm_span", 120),
        n_features=feat_cfg.get("features_per_scale", 8),
        bootstrap_bars=feat_cfg.get("bootstrap_bars", 30_000),
        obs_mode=feat_cfg.get("obs_mode", "window"),
        summary_feature_indices=feat_cfg.get("summary_feature_indices"),
        norm_warmup_path=norm_warmup_path,
        asset_class=feat_cfg.get("asset_class", "cfd_forex"),
    )

    # --- Bar Clock (CFD 24h-on-weekday market; CFDBarClock handles weekend gap) ---
    clock_cfg = config.get("bar_clock", {})
    bar_clock = CFDBarClock(
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 3),
        execution_delay_seconds=clock_cfg.get("execution_delay_seconds", 10.0),
        max_late_seconds=clock_cfg.get("max_late_seconds", 60.0),
    )

    # --- Risk Manager (position-fraction based, broker-agnostic) ---
    risk_cfg = config.get("risk", {})
    risk_manager = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.06),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 12),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        max_net_short_exposure=risk_cfg.get("max_net_short_exposure", -1.0),
        min_effective_bets=risk_cfg.get("min_effective_bets", 1.0),
        daily_turnover_limit=risk_cfg.get("daily_turnover_limit", 10.0),
        soft_throttle_start=risk_cfg.get("soft_throttle_start", 0.8),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 999.0),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.10),
        static_peak=risk_cfg.get("static_peak", False),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", False),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
        bar_interval_minutes=clock_cfg.get("base_interval_minutes", 3),
    ))

    # --- Data Loader (created upfront — OANDA needs no per-symbol handshake) ---
    loader = OandaDataLoader(broker=broker)

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

    # --- PRISM Overlay (optional L2; falsified S413+ but kept wireable) ---
    prism_cfg = config.get("prism", {})
    if prism_cfg.get("enabled", False):
        from finrl_pro_ds.crypto.live.prism_overlay import PRISMOverlay
        engine._prism_overlay = PRISMOverlay(prism_cfg)
        logger.info(
            "PRISM L2 overlay enabled: %s",
            prism_cfg.get("base_url", "http://prism-api:8001"),
        )

    return engine, broker


async def main_async(config: dict) -> None:
    """Async main: connect OANDA, wire data loader, start engine."""

    engine, broker = build_components(config)

    # Connect with exponential backoff. OANDA practice is rarely
    # transient-failing, but 5xx during maintenance windows happens.
    max_connect_attempts = 5
    for attempt in range(1, max_connect_attempts + 1):
        try:
            await broker.connect()
            break
        except Exception as exc:
            logger.error(
                "OANDA connect() attempt %d/%d failed: %s",
                attempt, max_connect_attempts, exc,
            )
            if attempt == max_connect_attempts:
                raise
            # Exponential backoff: 5, 10, 20, 40, 80s
            delay = 5.0 * (2 ** (attempt - 1))
            logger.info("Retrying in %.0fs...", delay)
            await asyncio.sleep(delay)

    # Start the trading loop
    await engine.start()


def main():
    parser = argparse.ArgumentParser(
        description="Launch EUR/USD CFD paper/live trading via OANDA v20",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to live trading YAML config",
    )
    parser.add_argument(
        "--overlay", action="append", default=None,
        help="Deploy overlay under configs/deploy/ (e.g. 'hyper_growth/step1'). "
             "Repeat for multiple; later wins. Same allowlist as "
             "deploy_bare_metal --overlay. Falls back to STRATEGY_OVERLAY "
             "env var if --overlay not given.",
    )
    parser.add_argument(
        "--mainnet", action="store_true",
        help="Use live OANDA account. Default: practice (paper).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions without executing orders.",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply deploy overlays (Step 5 prop-firm decoupling). CLI flag wins
    # over STRATEGY_OVERLAY env var.
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

    contract = config.get("contract", {}).get("symbol", "EURUSD")
    env_name = config.get("exchange", {}).get("env", "practice")
    mode = "PAPER (OANDA practice)" if env_name == "practice" else "LIVE (OANDA)"
    logger.info(
        "Starting OANDA EUR/USD CFD trading engine\n"
        "  Config: %s\n"
        "  Overlays: %s\n"
        "  Symbol: %s\n"
        "  Mode: %s\n"
        "  Dry run: %s",
        config_path, overlay_specs or "none", contract, mode,
        config.get("dry_run", False),
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
        # Force PID 1 exit even if non-daemon threads (wandb-core, httpx
        # connection pool, OANDA stream task) are still alive. Mirrors
        # run_live_ctrader.py / run_live_ib.py orphan-netns safety pattern.
        logger.info("run_live_oanda exiting with code %d", exit_code)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
