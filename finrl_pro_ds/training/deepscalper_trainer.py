import torch
import numpy as np
import time
import os
import wandb
import logging
from collections import deque
from datetime import datetime

from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ

class DeepScalperTrainer:
    """
    Simplified Trainer for Single BDQ Agent Strategy (Paper Replication).
    """
    def __init__(self, env, config, device="cuda" if torch.cuda.is_available() else "cpu", run_name=None):
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
            action_dims=action_dims,  # FIX: Pass from config
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
        
    def train(self):
        """Single Phase Training Loop"""
        print(f"Starting Training: Single BDQ Agent | Device: {self.device}")
        
        # Init State
        # Obs is Dict: {'micro': ..., 'macro': ..., 'private': ...}
        obs, _ = self.env.reset()
        
        # Need to ensure obs components are batched correctly (VectorEnv does this, but if Single env?)
        # If VectorEnv, obs['micro'] is (NumEnvs, Window, Feats)
        # We process row-by-row for filling buffer if NumEnvs > 1
        
        num_envs = self.config["env"].get("num_envs", 1)
        
        global_step = 0
        episode_rewards = deque(maxlen=100)
        episode_lens = deque(maxlen=100)
        
        curr_rewards = np.zeros(num_envs)
        curr_lens = np.zeros(num_envs)
        
        import time
        start_time = time.time()
        
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
                num_updates = int(self.update_interval * num_envs) # Scale by envs?
                # Actually, standard is: collected specific batch size?
                # Using config logic: "Update every 2 steps" -> update_interval = 2.0 (wait 2 steps?)
                # Config comment says: "dqn_update_interval: 2.0" -> "Update every 2 steps".
                # My logic:
                
                should_update = (global_step % int(self.update_interval) == 0)
                if should_update:
                     metrics = self.agent.train_step()
                     if metrics and global_step % self.log_interval == 0:
                         # Log to WandB
                         logs = {
                             "step": global_step,
                             "train/reward_mean": np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
                             "train/len_mean": np.mean(episode_lens) if len(episode_lens) > 0 else 0.0,
                             **{f"agent/{k}": v for k, v in metrics.items()}
                         }
                         wandb.log(logs)
            
            # 5. Checkpointing
            if global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{global_step}.pth")
                
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
