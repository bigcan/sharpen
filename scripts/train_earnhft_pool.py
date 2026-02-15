"""
Standalone EarnHFT Pool Training Script.

Train the low-level agent pool (each with a different Q-teacher reward shaping beta).
Can be run independently from the full pipeline for iterative development.

Usage:
    python scripts/train_earnhft_pool.py --config configs/deepscalper_earnhft.yaml
    python scripts/train_earnhft_pool.py --config configs/deepscalper_earnhft.yaml --betas 0.0 1.0 5.0
    python scripts/train_earnhft_pool.py --config configs/deepscalper_earnhft.yaml --stage q_teacher
    python scripts/train_earnhft_pool.py --config configs/deepscalper_earnhft.yaml --stage pool
    python scripts/train_earnhft_pool.py --config configs/deepscalper_earnhft.yaml --stage router
"""
import argparse
import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="EarnHFT Pool Training Script")
    parser.add_argument("--config", type=str, default="configs/deepscalper_earnhft.yaml")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "q_teacher", "pool", "router"],
                        help="Which stage to run")
    parser.add_argument("--betas", nargs="*", type=float, default=None,
                        help="Override beta values for the pool")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/earnhft")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override betas if specified
    if args.betas:
        config.setdefault("agents", {}).setdefault("earnhft", {})
        config["agents"]["earnhft"]["betas"] = args.betas

    # Auto-detect device
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"Config: {args.config}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Device: {device}")

    # Load data
    data_config = config.get("data", {})
    data_path = data_config.get("file_path")
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter to training range
    if "timestamp" in df.columns:
        df = df[
            (df["timestamp"] >= data_config.get("train_start_date", "")) &
            (df["timestamp"] <= data_config.get("train_end_date", ""))
        ]
    logger.info(f"Training data: {len(df)} rows")

    # Create trainer
    from finrl_pro_ds.training.earnhft_trainer import EarnHFTTrainer
    trainer = EarnHFTTrainer(
        config=config,
        data_df=df,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.stage == "all":
        results = trainer.train()
        logger.info(f"Full pipeline results: {results}")
    elif args.stage == "q_teacher":
        q_table = trainer._train_q_teacher()
        logger.info(f"Q-table shape: {q_table.shape}")
    elif args.stage == "pool":
        # Load Q-table first
        q_path = Path(args.checkpoint_dir) / "q_table.npy"
        if not q_path.exists():
            logger.info("Q-table not found, computing...")
            trainer._train_q_teacher()
        trainer.q_table = np.load(q_path)
        results = trainer._train_pool()
        logger.info(f"Pool results: {results}")
    elif args.stage == "router":
        # Load Q-table and pool agents
        q_path = Path(args.checkpoint_dir) / "q_table.npy"
        if not q_path.exists():
            raise FileNotFoundError(f"Q-table not found at {q_path}. Run q_teacher stage first.")
        trainer.q_table = np.load(q_path)

        # Load pool agents
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        betas = config.get("agents", {}).get("earnhft", {}).get("betas", [0.0, 1.0, 5.0])
        for beta in betas:
            ckpt = Path(args.checkpoint_dir) / f"pool_beta_{beta}.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"Pool checkpoint not found at {ckpt}. Run pool stage first.")
            agent = DiscretePPOAgent(
                network_config=trainer.network_config,
                device=device,
            )
            agent.load(str(ckpt))
            trainer.pool_agents.append(agent)

        results = trainer._train_router()
        logger.info(f"Router results: {results}")


if __name__ == "__main__":
    main()
