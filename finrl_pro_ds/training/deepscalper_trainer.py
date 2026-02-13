import torch
import numpy as np
import time
import os
import wandb
import logging
from collections import deque
from datetime import datetime
import optuna

from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ

class DeepScalperTrainer:
    """
    Simplified Trainer for Single BDQ Agent Strategy (Paper Replication).
    """
    def __init__(self, env, config, device="cuda" if torch.cuda.is_available() else "cpu", run_name=None, hpo_mode=False):
        self.env = env
        self.config = config
        self.device = device
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Read Action Dims from Config — Paper-aligned: 2 branches (Price, SignedQty)
        action_config = config.get("env", {}).get("action", {})
        signed_qty_props = action_config.get(
            "signed_qty_proportions", [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]
        )
        action_dims = (
            action_config.get("price_bins", 5),
            len(signed_qty_props)
        )
        
        # Inject action_space_dims into network config so BDQ assertion is guaranteed
        net_cfg = dict(config["network"])
        net_cfg["action_space_dims"] = action_dims
        
        # Agent Init
        self.agent = DeepScalperBDQ(
            network_config=net_cfg,
            lr=config["agents"]["bdq"]["learning_rate"],
            gamma=config["agents"]["bdq"]["gamma"],
            epsilon_start=config["agents"]["bdq"].get("epsilon_start", 1.0),
            epsilon_end=config["agents"]["bdq"].get("epsilon_end", 0.01),
            buffer_size=config["agents"]["bdq"]["buffer_size"],
            batch_size=config["agents"]["bdq"]["batch_size"],
            target_update_freq=config["agents"]["bdq"]["target_update_freq"],
            auxiliary_weight=config["agents"]["bdq"].get("auxiliary_weight", 1.0),
            epsilon_decay=config["agents"]["bdq"].get("epsilon_decay", 0.99999), # FIX: Read from config
            action_dims=action_dims,  # FIX: Pass from config
            use_amp=config["training"].get("use_amp", False),
            # Paper Section 4.3: Prioritized Experience Replay
            use_per=config["agents"]["bdq"].get("use_per", False),
            per_alpha=config["agents"]["bdq"].get("per_alpha", 0.6),
            per_beta_start=config["agents"]["bdq"].get("per_beta_start", 0.4),
            per_beta_frames=config["agents"]["bdq"].get("per_beta_frames", 100000),
            device=device
        )
        
        # Training Params
        self.total_timesteps = config["training"]["total_timesteps"]
        self.training_epochs = config["training"].get("training_epochs", 1)  # Paper: ~5 epochs
        self.update_interval = config["agents"]["bdq"].get("update_interval", 1.0)
        self.log_interval = config["training"]["log_interval"]
        self.checkpoint_interval = config["agents"]["bdq"]["checkpoint_interval"]
        self.learning_starts = config["agents"]["bdq"]["learning_starts"]
        
        # Checkpoint Directory
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # HPO mode: suppress frequent WandB logging to avoid memory flooding
        self.hpo_mode = hpo_mode
        self.gradient_accumulator = 0.0  # FIX FIND-4: Accumulator for fractional update_interval

        # FIX PERF-8: Initialize cosine LR scheduler for stable late-training convergence
        # FIX N1: Resolve num_envs from env before use (was NameError)
        _num_envs = getattr(self.env, 'num_envs', 1)
        total_updates = int(self.total_timesteps * self.training_epochs * self.update_interval / _num_envs) if _num_envs > 0 else 100000
        self.agent._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.agent.optimizer, T_max=total_updates, eta_min=1e-6
        )

        # FIX FIND-5 + CRIT-1: Private size consistency check
        # Env produces 3 private features (pos, bal, remaining_time). Config must match.
        priv_cfg = config.get("network", {}).get("micro_config", {}).get("private_input_size", 3)
        assert priv_cfg == 3, f"FIND-5 Mismatch: Env produces 3 private features, config expects {priv_cfg}"
        
    def train(self, start_step=0, skip_reset=False, optuna_trial=None, pruning_callback=None):
        """Single Phase Training Loop
        
        Args:
            start_step: Resume training from this step (for chunked HPO).
            skip_reset: If True, reuse stored obs from previous chunk instead of resetting.
            optuna_trial: Optuna Trial object for HPO reporting/pruning.
            pruning_callback: Function() -> float to evaluate agent during training.
        """
        print(f"Starting Training: Single BDQ Agent | Device: {self.device} | Start Step: {start_step} | Epochs: {self.training_epochs}")
        
        # Init State - Obs is Dict: {'micro': ..., 'macro': ..., 'private': ...}
        if skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
            obs = self._current_obs
            print(f"  Resuming from stored observation (skip_reset=True)")
        else:
            obs, _ = self.env.reset()
        
        # Defensive Shape Assertions — run once at start for performance
        if global_step == start_step:
            B = self.env.num_envs
            W = self.config.get("env", {}).get("window_size", 50)
            assert obs["micro"].shape == (B, W, 27), \
                f"obs['micro'] shape mismatch: expected ({B}, {W}, 27), got {obs['micro'].shape}"
            assert obs["private"].shape == (B, W, 3), \
                f"obs['private'] shape mismatch: expected ({B}, {W}, 3), got {obs['private'].shape}"
            assert obs["macro"].ndim == 2, \
                f"obs['macro'] expected 2D (B, M), got shape {obs['macro'].shape}"
            print(f"✓ Observation shapes verified: micro={obs['micro'].shape}, "
                  f"private={obs['private'].shape}, macro={obs['macro'].shape}")
        
        # FIX PERF-2: Dynamic epsilon decay for BOTH HPO and production training.
        # Ensures exploration schedule matches actual training budget regardless of num_envs.
        num_envs = self.env.num_envs
        
        if self.hpo_mode:
            steps_per_trial = self.config.get("hpo", {}).get("steps_per_trial", 50000)
            n_calls = len(list(range(0, steps_per_trial, num_envs)))
        else:
            # Production: compute from total_timesteps
            n_calls = self.total_timesteps // num_envs

        if n_calls > 0:
            # Scale epsilon decay across ALL epochs
            total_calls = n_calls * self.training_epochs
            epsilon_end = self.config.get("agents", {}).get("bdq", {}).get("epsilon_end", 0.01)
            computed_decay = np.exp(np.log(max(epsilon_end, 1e-10)) / total_calls)
            self.agent.epsilon_decay = computed_decay
            if self.hpo_mode:
                print(f"[HPO] Overriding epsilon_decay to {computed_decay:.6f} for {steps_per_trial} steps (~{total_calls} updates, {self.training_epochs} epochs)")
            else:
                print(f"[Train] Computed epsilon_decay = {computed_decay:.6f} for {self.total_timesteps}x{self.training_epochs} steps (~{total_calls} updates)")
        
        global_step = start_step
        episode_rewards = deque(maxlen=100)
        episode_lens = deque(maxlen=100)
        
        curr_rewards = np.zeros(num_envs)
        curr_lens = np.zeros(num_envs)
        
        # Reward component accumulators for hindsight ratio tracking
        _acc_hindsight = 0.0
        _acc_total = 0.0

        
        curr_lens = np.zeros(num_envs)
        
        # FIND-3: Removed redundant 'import time'
        start_time = time.time()
        
        # Reset pruning rung tracker for fresh trial (P0 fix)
        # Initialize to 0 to skip rung 0 - avoids eval at step ~12 before learning starts (P1a fix)
        self._last_prune_rung = 0
        
        # FIX PERF-3 + N4: Hoist extract_tensors outside loop
        def extract_tensors(o, device=self.device):
            return (
                torch.as_tensor(o["micro"], dtype=torch.float32).to(device),
                torch.as_tensor(o["private"], dtype=torch.float32).to(device),
                torch.as_tensor(o["macro"], dtype=torch.float32).to(device)
            )

        # Paper Section 4.3: Epoch-based training — each epoch replays the data
        for epoch in range(self.training_epochs):
            print(f"\n=== Epoch {epoch+1}/{self.training_epochs} ===")
            
            # Reset env at start of each epoch (except first if skip_reset)
            if epoch == 0 and skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
                obs = self._current_obs
                print(f"  Resuming from stored observation (skip_reset=True)")
            else:
                obs, _ = self.env.reset()

            epoch_step = 0
            while epoch_step < self.total_timesteps:
                # Convert to torch for prediction
                micro_t, private_t, macro_t = extract_tensors(obs)
                
                # Predict (Returns numpy array of actions [B, 2])
                actions = self.agent.predict(micro_t, private_t, macro_t, deterministic=False)
                
                # 2. Step Environment
                next_obs, rewards, term, trunc, infos = self.env.step(actions)
                
                dones = np.logical_or(term, trunc)
                
                # 3. Store in Buffer
                for i in range(num_envs):
                    s = {k: v[i] for k, v in obs.items()}
                    ns = {k: v[i] for k, v in next_obs.items()}
                    a = actions[i]
                    r = rewards[i]
                    d = dones[i]
                    
                    # EXTRACT VOLATILITY TARGET FROM INFO for Hindsight/Aux
                    aux_target = 0.0
                    if isinstance(infos, dict) and "volatility_target" in infos:
                        val = infos["volatility_target"]
                        if hasattr(val, "__getitem__"):
                             aux_target = val[i]
                        else:
                             aux_target = val
                    elif isinstance(infos, list):
                         aux_target = infos[i].get("volatility_target", 0.0)
                    
                    self.agent.memory.push(s, a, float(r), ns, bool(d), float(aux_target))
                    
                    # Accumulate reward components for hindsight ratio tracking
                    if isinstance(infos, dict):
                        rh = infos.get("reward_hindsight", None)
                        rt = infos.get("reward_total", None)
                        if rh is not None and rt is not None:
                            _acc_hindsight += abs(float(rh[i]) if hasattr(rh, "__getitem__") else float(rh))
                            _acc_total += abs(float(rt[i]) if hasattr(rt, "__getitem__") else float(rt))
                    
                    # Track Episodic Stats
                    curr_rewards[i] += r
                    curr_lens[i] += 1
                    
                    if d:
                        episode_rewards.append(curr_rewards[i])
                        episode_lens.append(curr_lens[i])
                        curr_rewards[i] = 0
                        curr_lens[i] = 0
                        
                obs = next_obs
                global_step += num_envs
                epoch_step += num_envs
                
                # FIX: Decay epsilon every step batch
                self.agent.decay_epsilon()
                
                # 4. Training Step
                if global_step > self.learning_starts:
                    self.gradient_accumulator += self.update_interval
                    
                    metrics = None
                    while self.gradient_accumulator >= 1.0:
                        self.gradient_accumulator -= 1.0
                        metrics = self.agent.train_step()
                        if self.agent._lr_scheduler is not None:
                            self.agent._lr_scheduler.step()
                    
                    # LOGGING
                    if metrics and global_step > 0 and global_step % self.log_interval == 0:
                         if not self.hpo_mode:
                             logs = {
                                 "step": global_step,
                                 "train/epoch": epoch + 1,
                                 "train/reward_mean": np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
                                 "train/len_mean": np.mean(episode_lens) if len(episode_lens) > 0 else 0.0,
                                 **{f"agent/{k}": v for k, v in metrics.items()}
                             }
                             logs["agent/auxiliary_weight"] = self.agent.auxiliary_weight
                             if "loss_aux" in metrics and "loss_total" in metrics:
                                 total = metrics["loss_total"]
                                 if total > 1e-12:
                                     logs["agent/aux_loss_ratio"] = metrics["loss_aux"] / total
                             if self.agent._lr_scheduler is not None:
                                 logs["agent/learning_rate"] = self.agent._lr_scheduler.get_last_lr()[0]
                             
                             # Calculate SPS
                             current_time = time.time()
                             last_time = getattr(self, '_last_log_time', start_time)
                             last_step = getattr(self, '_last_log_step', start_step)
                             
                             elapsed = current_time - last_time
                             if elapsed > 1e-4:
                                 sps = (global_step - last_step) / elapsed
                                 logs["train/sps"] = sps
                             else:
                                 logs["train/sps"] = 0.0
                             
                             self._last_log_time = current_time
                             self._last_log_step = global_step

                             verbose = self.config["training"].get("verbose_logging", False)
                             
                             if not verbose:
                                 filtered_logs = {
                                     "step": logs["step"],
                                     "train/epoch": logs.get("train/epoch", 1),
                                     "train/reward_mean": logs.get("train/reward_mean", 0.0),
                                     "train/loss_total": logs.get("agent/loss_total", 0.0),
                                     "train/sps": logs.get("train/sps", 0.0),
                                     "train/epsilon": logs.get("agent/epsilon", 0.0),
                                     "train/len_mean": logs.get("train/len_mean", 0.0),
                                 }
                                 for k, v in logs.items():
                                     if k.startswith("eval/"):
                                         filtered_logs[k] = v
                                         
                                 wandb.log(filtered_logs)
                             else:
                                  wandb.log(logs)

                # 4a. Hindsight Ratio Logging
                if global_step > 0 and global_step % self.log_interval == 0 and not self.hpo_mode:
                    hindsight_ratio = _acc_hindsight / _acc_total if _acc_total > 1e-9 else 0.0
                    wandb.log({"reward/hindsight_ratio": hindsight_ratio}, step=global_step)
                    _acc_hindsight = 0.0
                    _acc_total = 0.0

                # 4b. HPO Pruning Check
                if optuna_trial and pruning_callback:
                    min_pruning_steps = 20000 
                    prune_interval = 5000
                    current_rung = global_step // prune_interval
                    
                    if global_step > min_pruning_steps and current_rung > getattr(self, '_last_prune_rung', -1):
                        self._last_prune_rung = current_rung
                        print(f"  [HPO] Probing agent at step {global_step} (rung {current_rung})...")
                        score = pruning_callback()
                        print(f"  [HPO] Step {global_step} Score: {score:.4f}")
                        
                        optuna_trial.report(score, global_step)
                        
                        if optuna_trial.should_prune():
                            print(f"  [HPO] Pruning trial at step {global_step}")
                            raise optuna.TrialPruned()

                # 5. Checkpointing
                if global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}.pth")
                
        # Store obs for potential resume via skip_reset=True
        self._current_obs = obs
                
        # Final Save
        self.save_checkpoint("checkpoint_final.pth")
        
        # Always log final step status to ensure graph continuity
        if not self.hpo_mode:
            print(f"Logging final metrics at step {global_step}")
            wandb.log({"step": global_step, "train/final_step": 1})
            
        print("Training Complete.")

    def save_checkpoint(self, filename):
        path = os.path.join(self.ckpt_dir, filename)
        self.agent.save(path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        self.agent.load(path)
        print(f"Loaded checkpoint: {path}")
