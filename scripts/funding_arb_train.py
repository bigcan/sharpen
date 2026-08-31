"""
funding_arb_train.py — PPO training loop for the multi-exchange funding rate arb agent.

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted for FinRL-Pro_DS project structure.

Usage:
    python scripts/funding_arb_train.py
    python scripts/funding_arb_train.py --config configs/funding_arb_multi_exchange.yaml --timesteps 300000
    python scripts/funding_arb_train.py --data data/funding_arb/train_data.npy --eval-data data/funding_arb/eval_data.npy
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sharpen.crypto.envs.multi_exchange_arb_env import (
    EnvConfig,
    MultiExchangeArbEnv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env_config(cfg: dict) -> EnvConfig:
    """Build EnvConfig from YAML dict."""
    env_cfg = cfg.get("environment", {})
    defaults = EnvConfig()
    return EnvConfig(
        symbols=env_cfg.get("symbols", defaults.symbols),
        exchanges=env_cfg.get("exchanges", defaults.exchanges),
        initial_capital=env_cfg.get("initial_capital", defaults.initial_capital),
        max_open_positions=env_cfg.get("max_open_positions", defaults.max_open_positions),
        max_steps=env_cfg.get("max_steps", defaults.max_steps),
        taker_fee=env_cfg.get("taker_fee", defaults.taker_fee),
        slippage_bps=env_cfg.get("slippage_bps", defaults.slippage_bps),
    )


def make_env(market_data, env_config):
    """Factory for creating monitored environments."""
    def _init():
        env = MultiExchangeArbEnv(market_data=market_data, config=env_config)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for multi-exchange funding rate arbitrage",
    )
    parser.add_argument(
        "--config", type=str, default="configs/funding_arb_multi_exchange.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to train_data.npy")
    parser.add_argument(
        "--eval-data", type=str, default=None, help="Path to eval_data.npy",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total_timesteps",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--output", type=str, default="models/fr_arb_ppo_v1",
        help="Output directory for saved model",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config {config_path} not found, using defaults")
        cfg = {}

    train_cfg = cfg.get("training", {})
    env_config = build_env_config(cfg)

    # Load market data
    train_data = None
    if args.data and Path(args.data).exists():
        train_data = np.load(args.data)
        logger.info(f"Loaded training data: {train_data.shape}")
    else:
        logger.info("No training data provided — using synthetic data")

    eval_data = None
    if args.eval_data and Path(args.eval_data).exists():
        eval_data = np.load(args.eval_data)
        logger.info(f"Loaded eval data: {eval_data.shape}")

    # Create environments
    train_env = DummyVecEnv([make_env(train_data, env_config)])
    eval_env = DummyVecEnv([make_env(eval_data, env_config)])

    # Hyperparameters
    total_timesteps = args.timesteps or train_cfg.get("total_timesteps", 300_000)
    learning_rate = args.lr or train_cfg.get("learning_rate", 3e-4)
    net_arch = train_cfg.get("net_arch", [256, 256])

    # Output paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs/tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoint_dir),
        name_prefix="fr_arb_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5_000,
        n_eval_episodes=3,
        deterministic=True,
    )

    logger.info(f"Training PPO for {total_timesteps} timesteps")
    logger.info(f"  LR={learning_rate}, net_arch={net_arch}")
    logger.info(f"  Output: {output_dir}")

    # Create and train model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=train_cfg.get("n_steps", 2048),
        batch_size=train_cfg.get("batch_size", 64),
        n_epochs=train_cfg.get("n_epochs", 10),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_range=train_cfg.get("clip_range", 0.2),
        ent_coef=train_cfg.get("ent_coef", 0.01),
        vf_coef=train_cfg.get("vf_coef", 0.5),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = output_dir / "policy"
    model.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
