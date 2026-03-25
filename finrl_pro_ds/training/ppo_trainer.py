"""
PPO Trainer for DeepScalper.

On-policy training loop: collect rollouts → compute GAE → PPO update.
Fundamentally different from BDQ's off-policy step-by-step pattern.
"""
import logging
import os
import time
import numpy as np
import torch
from collections import deque

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None

try:
    import optuna
except ImportError:
    optuna = None

from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent  # noqa: E402
from finrl_pro_ds.agents.ppo_scalper.rollout_buffer import RolloutBuffer  # noqa: E402


def extract_tensors(obs):
    """Helper to convert obs dict to tensors (CPU)."""
    return (
        torch.as_tensor(obs["micro"], dtype=torch.float32),
        torch.as_tensor(obs["private"], dtype=torch.float32),
        torch.as_tensor(obs["macro"], dtype=torch.float32),
    )


class PPOTrainer:
    """
    On-policy trainer for PPO agent.

    Training loop:
        for epoch in training_epochs:
            reset env
            while epoch_step < total_timesteps:
                collect rollout (T steps)
                compute GAE
                PPO update (K epochs × minibatches)
                log, checkpoint
    """

    def __init__(self, env, config, device="cuda" if torch.cuda.is_available() else "cpu",
                 run_name=None, hpo_mode=False):
        self.env = env
        self.config = config
        self.device = device
        self.hpo_mode = hpo_mode
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Read action dims from config
        action_dims = config.get("env", {}).get("action", {}).get("discrete_dims", 6)

        # PPO config
        ppo_cfg = config.get("agents", {}).get("ppo", {})

        # Inject action_space_dims into network config
        net_cfg = dict(config["network"])
        net_cfg["action_space_dims"] = action_dims

        # FIX BUG-11: Validate encoder_type before agent construction
        encoder_type = net_cfg.get("micro_config", {}).get("encoder_type", "lstm")
        assert encoder_type in ("lstm", "mlp", "tcn"), (
            f"Unknown encoder_type='{encoder_type}' in network.micro_config. "
            f"Supported: 'lstm', 'mlp', 'tcn'."
        )

        # Training params
        self.total_timesteps = config["training"]["total_timesteps"]
        self.training_epochs = config["training"].get("training_epochs", 1)
        self.log_interval = config["training"]["log_interval"]
        self.rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        self.checkpoint_interval = ppo_cfg.get("checkpoint_interval", 100000)

        # Agent
        self.agent = PPOAgent(
            network_config=net_cfg,
            lr=ppo_cfg.get("learning_rate", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.95),  # T1.2 default
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_eps=ppo_cfg.get("clip_eps", 0.2),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            ent_coef=ppo_cfg.get("ent_coef", 0.001),  # T1.3 default
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            n_epochs=ppo_cfg.get("n_epochs", 4),
            rollout_steps=self.rollout_steps,
            batch_size=ppo_cfg.get("batch_size", 256),
            action_dims=action_dims,
            lr_schedule=ppo_cfg.get("lr_schedule", "linear"),
            total_timesteps=self.total_timesteps * self.training_epochs,
            use_amp=config["training"].get("use_amp", False),
            clip_value_loss=ppo_cfg.get("clip_value_loss", True),  # AUDIT FIX FLAG-3
            target_kl=ppo_cfg.get("target_kl", None),              # AUDIT FIX FLAG-4
            torch_compile=config["training"].get("torch_compile", False),  # PERF FIX-1
            device=device,
        )

        # Rollout buffer
        window_size = config.get("env", {}).get("window_size", 15)
        micro_input = net_cfg.get("micro_config", {}).get("input_size", 30)
        private_input = net_cfg.get("micro_config", {}).get("private_input_size", 5)
        self._private_dim = private_input
        macro_input = net_cfg.get("macro_config", {}).get("input_size", 15)
        num_envs = getattr(env, "num_envs", 1)

        self.buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            num_envs=num_envs,
            micro_shape=(window_size, micro_input),
            private_shape=(window_size, private_input),
            macro_shape=(macro_input,),
            n_action_branches=1, # Tier 2: Flattened
            n_qty_actions=action_dims,
        )

        # Checkpoint directory
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Private state check — V6 (SwingScalperEnv) uses 4, V5 (DeepScalperEnv) uses 5
        priv_cfg = config.get("network", {}).get("micro_config", {}).get("private_input_size", 5)
        mdp_ver = config.get("env", {}).get("mdp_version", "v5")
        expected_priv = 4 if mdp_ver == "v6" else 5
        assert priv_cfg == expected_priv, (
            f"MDP {mdp_ver} env produces {expected_priv} private features, "
            f"config expects {priv_cfg}"
        )

        # AUDIT FIX D1: Validate window_size consistency between env and network
        net_ws = config.get("network", {}).get("micro_config", {}).get("window_size")
        if net_ws is not None:
            assert window_size == net_ws, (
                f"env.window_size ({window_size}) != micro_config.window_size ({net_ws}). "
                f"These MUST match or MLP encoder will crash with a shape mismatch."
            )

    def train(self, start_step=0, skip_reset=False, optuna_trial=None, pruning_callback=None):
        """
        On-policy training loop.

        Args:
            start_step: Resume from this step
            skip_reset: If True, reuse stored obs
            optuna_trial: Optuna Trial for HPO
            pruning_callback: Function() -> float for HPO pruning
        """
        num_envs = getattr(self.env, "num_envs", 1)
        logger.info(f"Starting PPO Training | Device: {self.device} | "
              f"Envs: {num_envs} | Rollout: {self.rollout_steps} | "
              f"Epochs: {self.training_epochs}")

        # Init state
        qty_mask = None  # Default: no masking unless env or prior state provides one
        if skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
            obs = self._current_obs
            # AUDIT FIX: Restore mask too
            if hasattr(self, '_current_qty_mask'):
                qty_mask = self._current_qty_mask
        else:
            obs, reset_info = self.env.reset()
            # AUDIT FIX FLAG-2: Robust extraction (dict or list of dicts)
            if isinstance(reset_info, dict) and "qty_action_mask" in reset_info:
                qty_mask = reset_info["qty_action_mask"]
            elif isinstance(reset_info, list) and len(reset_info) > 0 and "qty_action_mask" in reset_info[0]:
                qty_mask = np.stack([info_i["qty_action_mask"] for info_i in reset_info])

        # Shape assertions
        B = num_envs
        W = self.config.get("env", {}).get("window_size", 15)
        _micro_dim = self.config.get("network", {}).get("micro_config", {}).get("input_size", 30)
        assert obs["micro"].shape == (B, W, _micro_dim), \
            f"obs['micro'] shape mismatch: expected ({B}, {W}, {_micro_dim}), got {obs['micro'].shape}"
        assert obs["private"].shape == (B, W, self._private_dim), \
            f"obs['private'] shape mismatch: expected ({B}, {W}, {self._private_dim}), got {obs['private'].shape}"
        logger.info(f"Observation shapes verified: micro={obs['micro'].shape}, "
              f"private={obs['private'].shape}, macro={obs['macro'].shape}")

        global_step = start_step
        start_time = time.time()  # AUDIT FIX: Define start_time for SPS metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        curr_rewards = np.zeros(num_envs)
        curr_lens = np.zeros(num_envs)

        start_time = time.time()
        self._last_prune_rung = 0

        def extract_tensors(o, device=self.device):
            return (
                torch.as_tensor(o["micro"], dtype=torch.float32).to(device),
                torch.as_tensor(o["private"], dtype=torch.float32).to(device),
                torch.as_tensor(o["macro"], dtype=torch.float32).to(device),
            )


        for epoch in range(self.training_epochs):
            logger.info(f"=== Epoch {epoch+1}/{self.training_epochs} ===")

            if epoch == 0 and skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
                obs = self._current_obs
            else:
                obs, reset_info = self.env.reset()
                self.agent.reset_hidden_state()
                # AUDIT FIX FLAG-2: Robust extraction (dict or list of dicts)
                if isinstance(reset_info, dict) and "qty_action_mask" in reset_info:
                    qty_mask = reset_info["qty_action_mask"]
                elif isinstance(reset_info, list) and len(reset_info) > 0 and "qty_action_mask" in reset_info[0]:
                    qty_mask = np.stack([info_i["qty_action_mask"] for info_i in reset_info])

            epoch_step = 0

            while epoch_step < self.total_timesteps:
                # ============================================================
                # COLLECT ROLLOUT
                # ============================================================
                self.buffer.reset()

                for t in range(self.rollout_steps):
                    micro_t, private_t, macro_t = extract_tensors(obs)

                    # Agent prediction
                    actions, log_probs, values = self.agent.predict(
                        micro_t, private_t, macro_t,
                        deterministic=False, qty_mask=qty_mask
                    )

                    # Step environment
                    next_obs, rewards, term, trunc, infos = self.env.step(actions)

                    # C2 fix: NaN guard on rewards
                    if np.isnan(rewards).any():
                        nan_envs = np.where(np.isnan(rewards))[0]
                        logger.warning(f"NaN reward detected in envs {nan_envs} at step {global_step}, replacing with 0.0")
                        rewards = np.nan_to_num(rewards, nan=0.0)

                    # Store current qty_mask before extracting next one (B4: for this timestep)
                    current_qty_mask = qty_mask  # mask used for THIS action selection

                    # Extract qty mask for next step
                    if isinstance(infos, dict) and "qty_action_mask" in infos:
                        qty_mask = infos["qty_action_mask"]
                    elif isinstance(infos, list) and len(infos) > 0 and "qty_action_mask" in infos[0]:
                        qty_mask = np.stack([info_i["qty_action_mask"] for info_i in infos])
                    else:
                        qty_mask = None

                    # Done handling (only term zeroes bootstrap, same as BDQ)
                    dones_for_reset = np.logical_or(term, trunc)

                    # Mask hidden states for done envs
                    self.agent.mask_hidden_state(dones_for_reset)

                    # Use term only for done flag in buffer (Bellman bootstrap)
                    self.buffer.store(
                        obs=obs,
                        actions=actions,
                        log_probs=log_probs,
                        rewards=rewards,
                        values=values,
                        dones=term.astype(np.float32),
                        qty_masks=current_qty_mask,  # B4: store mask used for this step's action
                    )

                    # Track episodic stats
                    for i in range(num_envs):
                        curr_rewards[i] += rewards[i]
                        curr_lens[i] += 1
                        if dones_for_reset[i]:
                            self.episode_rewards.append(curr_rewards[i])
                            self.episode_lengths.append(curr_lens[i])
                            curr_rewards[i] = 0
                            curr_lens[i] = 0

                    obs = next_obs
                    global_step += num_envs
                    epoch_step += num_envs

                # ============================================================
                # COMPUTE GAE
                # ============================================================
                # Bootstrap value for the last step
                with torch.no_grad():
                    # Move to device for prediction
                    micro_t, private_t, macro_t = [t.to(self.device) for t in extract_tensors(obs)]

                    # predict() returns 3 numpy arrays: (actions, log_probs, values)
                    _, _, last_values = self.agent.predict(
                        micro_t, private_t, macro_t,
                        deterministic=True
                    )

                self.buffer.compute_gae(
                    gamma=self.agent.gamma,
                    gae_lambda=self.agent.gae_lambda,
                    last_values=last_values,
                    last_dones=term.astype(np.float32),
                )

                # ============================================================
                # PPO UPDATE
                # ============================================================
                # AUDIT FIX: Update step_count BEFORE train_step so scheduler uses current progress
                self.agent.step_count = global_step
                metrics = self.agent.train_step(self.buffer)

                # ============================================================
                # LOGGING
                # ============================================================
                if metrics and global_step > 0 and global_step % self.log_interval < (self.rollout_steps * num_envs):
                    if not self.hpo_mode and wandb and wandb.run:
                        logs = {
                            "step": global_step,
                            "train/epoch": epoch + 1,
                            "train/reward_mean": np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0,
                            "train/len_mean": np.mean(self.episode_lengths) if len(self.episode_lengths) > 0 else 0.0,
                            **{f"agent/{k}": v for k, v in metrics.items()},
                        }

                        # SPS
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            logs["train/sps"] = global_step / elapsed

                        wandb.log(logs)
                    elif self.hpo_mode and wandb and wandb.run and global_step % 10000 < (self.rollout_steps * num_envs):
                        # WandB heartbeat during HPO so fleet monitor doesn't flag as stalled
                        try:
                            elapsed = time.time() - start_time
                            sps = global_step / max(elapsed, 1e-6)
                            wandb.log({
                                "hpo/heartbeat_step": global_step,
                                "hpo/heartbeat_sps": round(sps, 1),
                            })
                        except Exception:
                            pass

                # HPO Pruning
                if optuna_trial and pruning_callback:
                    min_pruning_steps = 20000
                    prune_interval = 5000
                    current_rung = global_step // prune_interval
                    if global_step > min_pruning_steps and current_rung > self._last_prune_rung:
                        self._last_prune_rung = current_rung
                        score = pruning_callback()
                        logger.info(f"  [HPO] Step {global_step} Score: {score:.4f}")
                        optuna_trial.report(score, global_step)
                        if optuna_trial.should_prune():
                            raise optuna.TrialPruned()

                # Checkpointing
                if global_step % self.checkpoint_interval < (self.rollout_steps * num_envs):
                    self.save_checkpoint(f"checkpoint_step_{global_step}.pth")

        # Store obs for resume
        self._current_obs = obs
        self._current_qty_mask = qty_mask

        # Final save
        self.save_checkpoint("checkpoint_final.pth")

        if not self.hpo_mode and wandb and wandb.run:
            wandb.log({"step": global_step, "train/final_step": 1})

        logger.info("PPO Training Complete.")

    def save_checkpoint(self, filename):
        path = os.path.join(self.ckpt_dir, filename)
        self.agent.save(path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        self.agent.load(path)
        logger.info(f"Loaded checkpoint: {path}")
