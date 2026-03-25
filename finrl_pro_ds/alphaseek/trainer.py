"""AlphaSeek DQN Trainer — production training loop.

Adapted from contest/reference/erl_run.py::train_agent(). Provides:
- Full training loop: explore → buffer → update_net → eval → WandB
- Production evaluation metrics: total_return, sharpe, sortino, PF, max_dd
- Checkpoint save/load for recovery
- WandB heartbeat logging (commit=True)

Usage:
    trainer = AlphaSeekTrainer(
        agent_class=AgentD3QN,
        train_sim=LOBTradeSimulator(..., segment_filter=[0,1,...,7]),
        eval_sim=EvalLOBTradeSimulator(..., segment_filter=[8]),
        config={"learning_rate": 1e-4, ...},
        gpu_id=0,
    )
    metrics = trainer.train(break_step=320_000)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Type

import numpy as np
import torch

from .agents import AgentDoubleDQN, AlphaSeekAgentConfig
from .lob_trade_simulator import EvalLOBTradeSimulator, LOBTradeSimulator
from .replay_buffer import AlphaSeekReplayBuffer

logger = logging.getLogger(__name__)


def _wandb_log(metrics: dict) -> None:
    """Log metrics to WandB with commit=True (heartbeat fix)."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, commit=True)
    except Exception:
        pass


class AlphaSeekTrainer:
    """Production DQN trainer for AlphaSeek agents.

    Wraps the explore → buffer → update_net training loop with:
    - Periodic evaluation on a held-out eval simulator
    - WandB logging with heartbeat
    - Checkpoint save/load
    - Production metrics computation
    """

    def __init__(
        self,
        agent_class: Type[AgentDoubleDQN],
        train_sim: LOBTradeSimulator,
        eval_sim: LOBTradeSimulator | EvalLOBTradeSimulator,
        config: dict,
        gpu_id: int = 0,
        cwd: str = "./alphaseek_train",
    ):
        self.agent_class = agent_class
        self.train_sim = train_sim
        self.eval_sim = eval_sim
        self.gpu_id = gpu_id
        self.cwd = cwd
        os.makedirs(cwd, exist_ok=True)

        # Build agent config from merged dict
        cfg = dict(config)
        cfg.setdefault("num_envs", train_sim.num_sims)
        cfg.setdefault("state_dim", train_sim.state_dim)
        cfg.setdefault("action_dim", train_sim.action_dim)
        self.agent_config = AlphaSeekAgentConfig.from_dict(cfg)

        # Training params
        max_step = train_sim.max_step
        self.horizon_len = cfg.get("horizon_len") or int(max_step * 4)
        self.eval_per_step = cfg.get("eval_per_step") or int(max_step)
        self.buffer_size = cfg.get("buffer_size") or int(max_step * 32)
        self.save_gap = cfg.get("save_gap", 8)

        # Agent and buffer (created in train())
        self.agent = None
        self.buffer = None

    def train(self, break_step: int) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        break_step : int
            Total environment steps before stopping.

        Returns
        -------
        dict
            Final evaluation metrics from eval simulator.
        """
        torch.set_grad_enabled(False)
        start_time = time.time()

        # Build agent
        agent = self.agent_class(
            net_dims=self.agent_config.net_dims,
            state_dim=self.agent_config.state_dim,
            action_dim=self.agent_config.action_dim,
            gpu_id=self.gpu_id,
            args=self.agent_config,
        )
        self.agent = agent

        # Init env
        state = self.train_sim.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        else:
            state = state.to(agent.device, non_blocking=True)
        agent.last_state = state.detach()

        # Init buffer
        buffer = AlphaSeekReplayBuffer(
            max_size=self.buffer_size,
            state_dim=self.agent_config.state_dim,
            action_dim=1,  # Discrete(3) → action stored as single int
            gpu_id=self.gpu_id,
            num_seqs=self.agent_config.num_envs,
        )
        self.buffer = buffer

        # Warmup: fill buffer with random exploration
        warmup_len = self.horizon_len * 3
        warmup_items = agent.explore_env(self.train_sim, warmup_len, if_random=True)
        buffer.update(warmup_items)
        logger.info(
            f"Buffer warmup complete: {len(buffer)}/{self.buffer_size} "
            f"({warmup_len} steps x {self.agent_config.num_envs} sims)"
        )

        # Training loop
        total_step = 0
        eval_count = 0
        best_return = -float("inf")
        best_metrics = {}

        while total_step <= break_step:
            # Collect
            buffer_items = agent.explore_env(self.train_sim, self.horizon_len)
            buffer.update(buffer_items)
            total_step += self.horizon_len

            # Train
            torch.set_grad_enabled(True)
            obj_critic, obj_actor = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            # SPS
            elapsed = time.time() - start_time
            sps = total_step * self.agent_config.num_envs / max(elapsed, 1e-6)

            # Periodic eval
            eval_count += self.horizon_len
            if eval_count >= self.eval_per_step:
                eval_count = 0
                metrics = self.evaluate()

                logger.info(
                    f"Step {total_step:>8d}/{break_step} | "
                    f"SPS={sps:,.0f} | "
                    f"critic={obj_critic:.4f} | "
                    f"Q_avg={obj_actor:.4f} | "
                    f"return={metrics['total_return']:.6f} | "
                    f"sharpe={metrics['sharpe']:.4f}"
                )

                # WandB
                _wandb_log({
                    "train/total_step": total_step,
                    "train/sps": sps,
                    "train/obj_critic": obj_critic,
                    "train/q_avg": obj_actor,
                    "eval/total_return": metrics["total_return"],
                    "eval/sharpe": metrics["sharpe"],
                    "eval/sortino": metrics["sortino"],
                    "eval/profit_factor": metrics["profit_factor"],
                    "eval/max_drawdown": metrics["max_drawdown"],
                    "eval/win_rate": metrics["win_rate"],
                    "eval/hold_rate": metrics["hold_rate"],
                })

                # Save best
                if metrics["total_return"] > best_return:
                    best_return = metrics["total_return"]
                    best_metrics = dict(metrics)
                    self.save_checkpoint(os.path.join(self.cwd, "best"))

        # Final save
        self.save_checkpoint(os.path.join(self.cwd, "final"))
        elapsed = time.time() - start_time
        logger.info(
            f"Training complete: {total_step} steps in {elapsed:.1f}s "
            f"({total_step * self.agent_config.num_envs / elapsed:,.0f} SPS)"
        )

        return best_metrics if best_metrics else self.evaluate()

    def evaluate(self) -> dict:
        """Run one eval episode and compute production metrics.

        Returns dict with: total_return, sharpe, sortino, profit_factor,
        max_drawdown, win_rate, hold_rate, n_trades.
        """
        if self.agent is None:
            raise RuntimeError("Must call train() or load_checkpoint() first")

        agent = self.agent
        sim = self.eval_sim

        # Run eval episode
        state = sim.reset()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        else:
            state = state.to(agent.device, non_blocking=True)

        all_rewards = []
        all_actions = []
        done = False
        step = 0

        while not done:
            with torch.no_grad():
                q1, q2 = agent.act.get_q1_q2(state)
                q_min = torch.min(q1, q2)
                action = q_min.argmax(dim=1, keepdim=True)

            state, reward, terminal, _ = sim.step(action)
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=agent.device)
            else:
                state = state.to(agent.device, non_blocking=True)

            # Average across sims for this step
            all_rewards.append(reward.mean().item())
            all_actions.append(action.float().mean().item())

            step += 1
            done = terminal.all().item() if terminal.any() else False
            if step >= sim.max_step:
                break

        rewards = np.array(all_rewards)
        actions = np.array(all_actions)

        # Compute metrics
        total_return = float(rewards.sum())

        if len(rewards) > 1 and rewards.std() > 1e-12:
            sharpe = float(rewards.mean() / rewards.std())
        else:
            sharpe = 0.0

        neg_returns = rewards[rewards < 0]
        if len(neg_returns) > 1 and neg_returns.std() > 1e-12:
            sortino = float(rewards.mean() / neg_returns.std())
        else:
            sortino = 0.0 if rewards.mean() <= 0 else float("inf")

        pos_sum = float(rewards[rewards > 0].sum()) if (rewards > 0).any() else 0.0
        neg_sum = float(abs(rewards[rewards < 0].sum())) if (rewards < 0).any() else 0.0
        profit_factor = pos_sum / max(neg_sum, 1e-12)

        # Max drawdown from cumulative equity
        cum_equity = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_equity)
        drawdown = running_max - cum_equity
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        # Trade metrics
        non_hold = rewards != 0
        win_rate = float((rewards[non_hold] > 0).sum() / max(non_hold.sum(), 1)) if non_hold.any() else 0.0
        hold_rate = float((np.abs(actions - 1.0) < 0.3).mean())  # action ~1 = hold
        n_trades = int(non_hold.sum())

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "hold_rate": hold_rate,
            "n_trades": n_trades,
            "n_steps": len(rewards),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        if self.agent is not None:
            self.agent.save_agent(path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent from checkpoint.

        Creates agent if not already initialized.
        """
        if self.agent is None:
            self.agent = self.agent_class(
                net_dims=self.agent_config.net_dims,
                state_dim=self.agent_config.state_dim,
                action_dim=self.agent_config.action_dim,
                gpu_id=self.gpu_id,
                args=self.agent_config,
            )
        self.agent.load_agent(path)
