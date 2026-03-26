#!/usr/bin/env python3
"""AlphaSeek Live Trading — CLI entry point.

Usage:
    # Paper trading (Bybit testnet)
    python scripts/run_alphaseek_live.py --config configs/live_alphaseek_btc_bybit.yaml

    # Live trading (requires --mainnet flag)
    python scripts/run_alphaseek_live.py --config configs/live_alphaseek_btc_bybit.yaml --mainnet

    # Dry run (no orders, no exchange connection)
    python scripts/run_alphaseek_live.py --config configs/live_alphaseek_btc_bybit.yaml --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main(args):
    from finrl_pro_ds.alphaseek.live_engine import AlphaSeekLiveEngine
    from finrl_pro_ds.alphaseek.model_registry import ModelRegistry

    config = load_config(args.config)

    # --- Safety checks ---
    exchange_cfg = config.get("exchange", {})
    safety_cfg = config.get("safety", {})

    if args.mainnet:
        if exchange_cfg.get("testnet", True):
            logging.warning("--mainnet flag set but config has testnet=true. Overriding to mainnet.")
            exchange_cfg["testnet"] = False
        if safety_cfg.get("require_explicit_mainnet", True):
            logging.info("MAINNET mode enabled. Real money at risk.")
    else:
        exchange_cfg["testnet"] = True
        logging.info("TESTNET mode (paper trading). Use --mainnet for live.")

    config["dry_run"] = args.dry_run
    if args.dry_run:
        logging.info("DRY RUN mode — no orders will be placed.")

    # --- GPU ---
    import torch
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        logging.info("Using GPU: %s", device)
    else:
        device = "cpu"
        logging.info("Using CPU")

    # --- Load ensemble ---
    alphaseek_cfg = config.get("alphaseek", {})
    agent_configs = alphaseek_cfg.get("agents", [])

    if not agent_configs:
        logging.error("No agents configured in alphaseek.agents")
        sys.exit(1)

    # Try registry first, fall back to inline config
    models_dir = str(PROJECT_ROOT / "models" / "alphaseek")
    registry = ModelRegistry(models_dir)

    ensemble = registry.load_ensemble(config, device=device)
    logging.info("Ensemble loaded: %s", ensemble)

    # --- State builder ---
    sb_cfg = config.get("state_builder", {})
    sb_type = sb_cfg.get("type", "mock")

    if sb_type == "mock" or args.dry_run:
        from finrl_pro_ds.alphaseek._mock_state_builder import MockStateBuilder
        state_builder = MockStateBuilder(device=device)
        logging.info("Using MockStateBuilder (dry-run / mock mode)")
    elif sb_type == "live":
        from finrl_pro_ds.alphaseek.feature_engine import AlphaSeekFeatureEngine
        from finrl_pro_ds.alphaseek.lob_feed import BybitLOBFeed
        from finrl_pro_ds.alphaseek.state_builder import AlphaSeekStateBuilder

        feat_cfg = sb_cfg.get("features", {})
        feature_engine = AlphaSeekFeatureEngine(
            norm_span=feat_cfg.get("norm_span", 120),
            momentum_window=feat_cfg.get("momentum_window", 5),
            vol_window=feat_cfg.get("vol_window", 30),
        )
        state_builder = AlphaSeekStateBuilder(
            feature_engine=feature_engine,
            device=device,
            max_position=alphaseek_cfg.get("max_position", 1),
            max_holding=alphaseek_cfg.get("max_holding", 1800),
        )

        # Start LOB feed → state builder ingestion loop
        lob_feed = BybitLOBFeed(
            symbol=exchange_cfg.get("symbol", "BTC/USDT:USDT"),
            depth=sb_cfg.get("lob_depth", 20),
            testnet=exchange_cfg.get("testnet", True),
        )
        await lob_feed.connect()

        async def _feed_loop():
            """Background: ingest LOB snapshots at ~1s cadence."""
            interval = sb_cfg.get("feed_interval_s", 1.0)
            while True:
                try:
                    snapshot = await lob_feed.get_latest_snapshot()
                    state_builder.ingest_snapshot(snapshot)
                except Exception as e:
                    logging.warning("LOB feed error: %s", e)
                await asyncio.sleep(interval)

        asyncio.get_event_loop().create_task(_feed_loop())
        logging.info(
            "Using AlphaSeekStateBuilder (live LOB feed, warmup=%d ticks)",
            feature_engine.warmup_ticks,
        )
    else:
        logging.error("Unknown state_builder.type: %s", sb_type)
        sys.exit(1)

    # --- Broker ---
    if not args.dry_run:
        from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
            ExchangePerpBroker,
        )
        broker = ExchangePerpBroker(
            exchange=exchange_cfg.get("name", "bybit"),
            testnet=exchange_cfg.get("testnet", True),
            order_type=exchange_cfg.get("order_type", "limit"),
            limit_offset_pct=exchange_cfg.get("limit_offset_pct", 0.0005),
        )
        logging.info("Broker: %s (testnet=%s)", exchange_cfg.get("name"), exchange_cfg.get("testnet"))
    else:
        broker = None  # Dry run doesn't need a broker

    # --- WandB ---
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False) and not args.dry_run:
        try:
            import wandb
            tags = wandb_cfg.get("tags", ["alphaseek"])
            if args.mainnet:
                tags.append("mainnet")
            else:
                tags.append("testnet")
            wandb.init(
                entity=wandb_cfg.get("entity", "bigcan-chiwin-technology"),
                project=wandb_cfg.get("project", "FinRL-Pro-DS"),
                name=f"alphaseek-live-{exchange_cfg.get('asset', 'BTC')}",
                tags=tags,
                config=config,
            )
            logging.info("WandB initialized")
        except Exception as e:
            logging.warning("WandB init failed: %s", e)

    # --- Build engine ---
    engine = AlphaSeekLiveEngine(
        ensemble=ensemble,
        state_builder=state_builder,
        broker=broker,
        config=config,
    )

    # --- Run ---
    await engine.start()


def cli():
    parser = argparse.ArgumentParser(
        description="AlphaSeek Live Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mainnet", action="store_true",
        help="Enable mainnet (real money) trading",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run — no orders, mock state builder",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-1,
        help="GPU device ID (-1 for CPU)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
