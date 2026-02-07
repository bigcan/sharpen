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
        
        # Read Action Dims from Config (Default: 3, 5, 5)
        action_config = config.get("env", {}).get("action", {})
        action_dims = (
            action_config.get("direction_bins", 3),
            action_config.get("price_bins", 5),
            action_config.get("volume_bins", 5)
        )
        
        # Agent Init
        self.agent = DeepScalperBDQ(
            network_config=config["network"],
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
            device=device
        )
        
        # Training Params
        self.total_timesteps = config["training"]["total_timesteps"]
        self.update_interval = config["agents"]["bdq"].get("update_interval", 1.0)
        self.log_interval = config["training"]["log_interval"]
        self.checkpoint_interval = config["agents"]["bdq"]["checkpoint_interval"]
        self.learning_starts = config["agents"]["bdq"]["learning_starts"]
        
        # Checkpoint Directory
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # HPO mode: suppress frequent WandB logging to avoid memory flooding
        self.hpo_mode = hpo_mode
        
    def train(self, start_step=0, skip_reset=False, optuna_trial=None, pruning_callback=None):
        """Single Phase Training Loop
        
        Args:
            start_step: Resume training from this step (for chunked HPO).
            skip_reset: If True, reuse stored obs from previous chunk instead of resetting.
            optuna_trial: Optuna Trial object for HPO reporting/pruning.
            pruning_callback: Function() -> float to evaluate agent during training.
        """
        print(f"Starting Training: Single BDQ Agent | Device: {self.device} | Start Step: {start_step}")
        
        # Init State - Obs is Dict: {'micro': ..., 'macro': ..., 'private': ...}
        if skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
            obs = self._current_obs
            print(f"  Resuming from stored observation (skip_reset=True)")
        else:
            obs, _ = self.env.reset()
        
        # Defensive Assertion: Ensure VectorEnv semantics (batch dimension present)
        assert len(obs["micro"].shape) == 3, \
            f"Expected obs['micro'] shape (B, Window, Features), got {obs['micro'].shape}. " \
            "Ensure env is wrapped in SyncVectorEnv even for num_envs=1."
        
        # FIX: Read num_envs from ACTUAL env, not config.
        # During HPO, env may have fewer workers than config specifies.
        num_envs = self.env.num_envs
        
        global_step = start_step
        episode_rewards = deque(maxlen=100)
        episode_lens = deque(maxlen=100)
        
        curr_rewards = np.zeros(num_envs)
        curr_lens = np.zeros(num_envs)
        
        import time
        start_time = time.time()
        
        # Reset pruning rung tracker for fresh trial (P0 fix)
        # Initialize to 0 to skip rung 0 - avoids eval at step ~12 before learning starts (P1a fix)
        self._last_prune_rung = 0
        
        while global_step < self.total_timesteps:
            # 1. Action Selection
            # We need to extract tensors from Dict Obs
            def extract_tensors(o):
                return (
                    torch.tensor(o["micro"], dtype=torch.float32).to(self.device),
                    torch.tensor(o["private"], dtype=torch.float32).to(self.device),
                    torch.tensor(o["macro"], dtype=torch.float32).to(self.device)
                )

            # Epsilon Decay handled in agent logic or here? Agent has it inside train_step usually
            # But get_action uses epsilon.
            # Convert to torch for prediction
            micro_t, private_t, macro_t = extract_tensors(obs)
            
            # Predict (Returns numpy array of actions [B, 3])
            actions = self.agent.predict(micro_t, private_t, macro_t, deterministic=False)
            
            # 2. Step Environment
            next_obs, rewards, term, trunc, infos = self.env.step(actions)
            
            dones = np.logical_or(term, trunc)
            
            # 3. Store in Buffer
            # Handle VectorEnv by iterating
            for i in range(num_envs):
                # Extract single instance data
                # obs is dict of batch arrays
                s = {k: v[i] for k, v in obs.items()}
                ns = {k: v[i] for k, v in next_obs.items()}
                a = actions[i]
                r = rewards[i]
                d = dones[i]
                
                # EXTRACT VOLATILITY TARGET FROM INFO for Hindsight/Aux
                # Gymnasium VectorEnv returns info as Dict of Arrays
                aux_target = 0.0
                if isinstance(infos, dict) and "volatility_target" in infos:
                    # Handle numpy array or list
                    val = infos["volatility_target"]
                    if hasattr(val, "__getitem__"):
                         aux_target = val[i]
                    else:
                         aux_target = val
                elif isinstance(infos, list):
                     aux_target = infos[i].get("volatility_target", 0.0)
                
                self.agent.memory.push(s, a, float(r), ns, bool(d), float(aux_target))
                
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
            
            # FIX: Decay epsilon every step batch (decoupled from train updates)
            self.agent.decay_epsilon()
            
            # 4. Training Step
            # Update every N steps (accumulated logic via update_interval)
            # If update_interval = 2.0, we update every 2 steps (0.5 updates/step? No, usually ratio)
            # Logic: If update_interval >= 1, do X updates.
            # If update_interval < 1 (e.g. 0.1), update every 10 steps.
            
            # Simpler: Just train `update_interval` times per step if >=1
            # Or train once every 1/interval steps.
            
            if global_step > self.learning_starts:
                # Update Logic: Support Frequency based on int(interval)
                # If interval=2.0, update every 2 steps.
                update_step = int(self.update_interval)
                should_update = (global_step % update_step == 0) if update_step > 0 else True
                if should_update:
                     metrics = self.agent.train_step()
                     if metrics and global_step % self.log_interval == 0:
                         # Log to WandB (skip in HPO mode to avoid flooding)
                         if not self.hpo_mode:
                             logs = {
                                 "step": global_step,
                                 "train/reward_mean": np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
                                 "train/len_mean": np.mean(episode_lens) if len(episode_lens) > 0 else 0.0,
                                 **{f"agent/{k}": v for k, v in metrics.items()}
                             }
                             
                             # Calculate SPS
                             current_time = time.time()
                             # Use getattr for safety if init failed
                             last_time = getattr(self, '_last_log_time', start_time)
                             last_step = getattr(self, '_last_log_step', start_step)
                             
                             elapsed = current_time - last_time
                             if elapsed > 1e-6:
                                 sps = (global_step - last_step) / elapsed
                                 logs["train/sps"] = sps
                             
                             # Update trackers
                             self._last_log_time = current_time
                             self._last_log_step = global_step

                             # FILTER METRICS TO REDUCE NOISE (Unless verbose_logging=True)
                             verbose = self.config["training"].get("verbose_logging", False)
                             
                             if not verbose:
                                 # Key Metrics Only ["The Big 5"]
                                 filtered_logs = {
                                     "step": logs["step"],
                                     "train/reward_mean": logs.get("train/reward_mean", 0.0),
                                     "train/loss_total": logs.get("agent/loss_total", 0.0), # Map from agent/
                                     "train/sps": logs.get("train/sps", 0.0),
                                     "train/epsilon": logs.get("agent/epsilon", 0.0),       # Map from agent/
                                     "train/len_mean": logs.get("train/len_mean", 0.0)
                                 }
                                 # Preserve any 'eval/' metrics if they happened to be mixed in (rare)
                                 for k, v in logs.items():
                                     if k.startswith("eval/"):
                                         filtered_logs[k] = v
                                         
                                 wandb.log(filtered_logs)
                             else:
                                 # Full detailed logging
                                 wandb.log(logs)
            
            # 4b. HPO Pruning Check (rung-based for vectorized envs)
            # Uses rung tracking to handle num_envs > 1 step increments
            if optuna_trial and pruning_callback:
                prune_interval = 5000
                current_rung = global_step // prune_interval
                if current_rung > getattr(self, '_last_prune_rung', -1):
                    self._last_prune_rung = current_rung
                    print(f"  [HPO] Probing agent at step {global_step} (rung {current_rung})...")
                    score = pruning_callback()
                    print(f"  [HPO] Step {global_step} Score: {score:.4f}")
                    
                    # Report to Optuna
                    optuna_trial.report(score, global_step)
                    
                    # Check Pruning
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
        print("Training Complete.")

    def save_checkpoint(self, filename):
        path = os.path.join(self.ckpt_dir, filename)
        self.agent.save(path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        self.agent.load(path)
        print(f"Loaded checkpoint: {path}")
