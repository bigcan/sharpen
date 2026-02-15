"""
EarnHFT Trainer — Full pipeline orchestrator.

Orchestrates the complete EarnHFT training pipeline:
  Stage 1: Q-Teacher (backward DP)
  Stage 2: Low-level agent pool training (with reward shaping)
  Stage 3: High-level router training (DQN)

Each stage runs sequentially; checkpoints are saved between stages.
"""
import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError, AttributeError):
    HAS_TORCH = False


class EarnHFTTrainer:
    """Full pipeline trainer for EarnHFT.

    Args:
        config: Full config dict (from YAML).
        data_df: LOB DataFrame for training.
        device: Torch device string.
        run_name: WandB run name prefix.
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_df: pd.DataFrame,
        device: str = "cpu",
        run_name: str = "earnhft",
        checkpoint_dir: str = "checkpoints/earnhft",
    ):
        self.config = config
        self.data_df = data_df
        self.device = device
        self.run_name = run_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Extract EarnHFT-specific config
        earnhft_cfg = config.get("agents", {}).get("earnhft", {})
        self.num_actions = earnhft_cfg.get("num_actions", 5)
        self.betas = earnhft_cfg.get("betas", [0.0, 1.0, 5.0])
        self.max_holding = earnhft_cfg.get("max_holding", 1.0)
        self.commission_fee = earnhft_cfg.get("commission_fee", 0.000175)
        self.low_level_steps = earnhft_cfg.get("low_level_steps", 200000)
        self.router_steps = earnhft_cfg.get("router_steps", 50000)
        self.chunk_length = earnhft_cfg.get("chunk_length", 3600)

        # Network config
        net_cfg = config.get("network", {})
        features_cfg = config.get("features", {})
        self.network_config = {
            "micro_config": {
                "input_size": net_cfg.get("micro_input_size", 20),
                "private_input_size": net_cfg.get("private_input_size", 3),
                "hidden_size": net_cfg.get("micro_hidden_size", 128),
                "rnn_type": net_cfg.get("rnn_type", "LSTM"),
            },
            "macro_config": {
                "input_size": net_cfg.get("macro_input_size", 11),
                "hidden_sizes": net_cfg.get("macro_hidden_sizes", [128, 64]),
            },
            "fusion_dim": net_cfg.get("fusion_dim", 128),
            "num_actions": self.num_actions,
        }

        self.pool_agents = []
        self.router = None
        self.q_table = None

    def train(self) -> Dict[str, Any]:
        """Run the full EarnHFT training pipeline.

        Returns:
            Dict with training metrics and checkpoint paths.
        """
        logger.info("=" * 60)
        logger.info("EarnHFT Training Pipeline Started")
        logger.info("=" * 60)

        results = {}

        # Stage 1: Q-Teacher
        logger.info("\n--- Stage 1: Q-Teacher ---")
        self.q_table = self._train_q_teacher()
        results["q_teacher"] = {"q_table_shape": list(self.q_table.shape)}

        # Stage 2: Low-Level Pool
        logger.info("\n--- Stage 2: Low-Level Agent Pool ---")
        pool_results = self._train_pool()
        results["pool"] = pool_results

        # Stage 3: High-Level Router
        logger.info("\n--- Stage 3: High-Level Router ---")
        router_results = self._train_router()
        results["router"] = router_results

        # Save manifest
        manifest_path = self.checkpoint_dir / "pool_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved manifest to {manifest_path}")

        logger.info("=" * 60)
        logger.info("EarnHFT Training Pipeline Complete!")
        logger.info("=" * 60)

        return results

    def _train_q_teacher(self) -> np.ndarray:
        """Stage 1: Compute Q-table via backward DP."""
        from finrl_pro_ds.agents.earnhft.q_teacher import QTeacher

        teacher = QTeacher(
            num_actions=self.num_actions,
            max_holding=self.max_holding,
            commission_fee=self.commission_fee,
        )
        q_table = teacher.compute_q_table(self.data_df)

        # Save Q-table
        q_path = self.checkpoint_dir / "q_table.npy"
        np.save(q_path, q_table)
        logger.info(f"Saved Q-table to {q_path}, shape={q_table.shape}")

        # Save optimal actions for analysis
        opt_actions = teacher.get_optimal_actions(q_table)
        opt_path = self.checkpoint_dir / "optimal_actions.npy"
        np.save(opt_path, opt_actions)

        return q_table

    def _train_pool(self) -> Dict[str, Any]:
        """Stage 2: Train low-level agents with different betas."""
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        from finrl_pro_ds.agents.earnhft.low_level_env import EarnHFTLowLevelEnv
        from finrl_pro_ds.agents.earnhft.data_utils import chunk_data

        chunks = chunk_data(self.data_df, self.chunk_length)
        pool_results = {}

        for beta in self.betas:
            logger.info(f"\nTraining agent with beta={beta}")
            agent = DiscretePPOAgent(
                network_config=self.network_config,
                lr=self.config.get("agents", {}).get("earnhft", {}).get("learning_rate", 3e-4),
                device=self.device,
            )

            total_reward = 0.0
            total_steps = 0

            for epoch in range(max(1, self.low_level_steps // (self.chunk_length * len(chunks)))):
                for chunk_idx, chunk_df in enumerate(chunks):
                    if total_steps >= self.low_level_steps:
                        break

                    # Create Q-table for this chunk
                    chunk_start = chunk_idx * self.chunk_length
                    chunk_end = chunk_start + self.chunk_length
                    chunk_q = self.q_table[chunk_start:chunk_end]

                    env = EarnHFTLowLevelEnv(
                        df=chunk_df, q_table=chunk_q,
                        num_actions=self.num_actions,
                        max_holding=self.max_holding,
                        beta=beta,
                        commission_fee=self.commission_fee,
                    )

                    obs, _ = env.reset()
                    agent.reset_hidden_state()
                    ep_reward = 0.0

                    done = False
                    while not done:
                        micro_t = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0)
                        private_t = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0)
                        macro_t = None
                        if "macro" in obs:
                            macro_t = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0)

                        actions, log_probs, values = agent.predict(micro_t, private_t, macro_t)
                        obs, reward, terminated, truncated, info = env.step(actions[0])
                        done = terminated or truncated
                        ep_reward += reward
                        total_steps += 1

                    total_reward += ep_reward

            # Save agent checkpoint
            ckpt_path = str(self.checkpoint_dir / f"pool_beta_{beta}.pth")
            agent.save(ckpt_path)
            self.pool_agents.append(agent)

            pool_results[f"beta_{beta}"] = {
                "checkpoint": ckpt_path,
                "total_reward": float(total_reward),
                "total_steps": total_steps,
            }
            logger.info(f"Beta={beta}: reward={total_reward:.4f}, steps={total_steps}")

        return pool_results

    def _train_router(self) -> Dict[str, Any]:
        """Stage 3: Train high-level router via DQN."""
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN
        from finrl_pro_ds.agents.earnhft.data_utils import aggregate_to_minute, chunk_data
        from finrl_pro_ds.agents.earnhft.low_level_env import EarnHFTLowLevelEnv

        chunks = chunk_data(self.data_df, self.chunk_length)
        pool_size = len(self.pool_agents)

        # Evaluate pool agents on each chunk to get per-minute PnLs
        logger.info("Evaluating pool agents for router training data...")
        all_minute_features = []
        all_minute_pnls = []

        for chunk_idx, chunk_df in enumerate(chunks):
            minute_df = aggregate_to_minute(chunk_df, ticks_per_minute=60)
            n_minutes = len(minute_df)
            if n_minutes == 0:
                continue

            features = minute_df.values.astype(np.float32)
            pnls = np.zeros((n_minutes, pool_size), dtype=np.float32)

            chunk_start = chunk_idx * self.chunk_length
            chunk_end = chunk_start + self.chunk_length
            chunk_q = self.q_table[chunk_start:chunk_end]

            for agent_idx, agent in enumerate(self.pool_agents):
                env = EarnHFTLowLevelEnv(
                    df=chunk_df, q_table=chunk_q,
                    num_actions=self.num_actions,
                    max_holding=self.max_holding,
                    beta=0.0,  # Evaluate on pure PnL
                    commission_fee=self.commission_fee,
                )
                obs, _ = env.reset()
                agent.reset_hidden_state()

                minute_pnl = 0.0
                tick_in_minute = 0

                for tick in range(len(chunk_df) - 1):
                    micro_t = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0)
                    private_t = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0)
                    actions, _, _ = agent.predict(micro_t, private_t, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(actions[0])

                    minute_pnl += reward
                    tick_in_minute += 1

                    if tick_in_minute >= 60:
                        minute_idx = tick // 60
                        if minute_idx < n_minutes:
                            pnls[minute_idx, agent_idx] = minute_pnl
                        minute_pnl = 0.0
                        tick_in_minute = 0

                    if terminated or truncated:
                        break

            all_minute_features.append(features)
            all_minute_pnls.append(pnls)

        if not all_minute_features:
            logger.warning("No minute data for router training!")
            return {"error": "no_data"}

        minute_features = np.concatenate(all_minute_features, axis=0)
        minute_pnls = np.concatenate(all_minute_pnls, axis=0)

        # Train DQN router
        obs_dim = minute_features.shape[1]
        self.router = RouterDQN(
            obs_dim=obs_dim,
            pool_size=pool_size,
            device=self.device,
        )

        from finrl_pro_ds.agents.earnhft.high_level_env import EarnHFTHighLevelEnv
        env = EarnHFTHighLevelEnv(minute_features, minute_pnls, pool_size)

        total_reward = 0.0
        n_episodes = max(1, self.router_steps // len(minute_features))

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.router.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.router.store_transition(obs, action, reward, next_obs, float(done))
                loss = self.router.train_step()
                obs = next_obs
                ep_reward += reward

            total_reward += ep_reward

        # Save router
        router_path = str(self.checkpoint_dir / "router.pth")
        self.router.save(router_path)

        return {
            "checkpoint": router_path,
            "total_reward": float(total_reward),
            "n_episodes": n_episodes,
            "obs_dim": obs_dim,
        }
