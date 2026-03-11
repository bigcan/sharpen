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

# RunningRewardNormalizer REMOVED (Sprint 7 BUG-3):
# Rewards are already in basis points (~O(1)) from the environment.
# Double-normalizing with an EMA z-score created a non-stationary target
# that fought learning stability.

class DeepScalperTrainer:
    """
    Simplified Trainer for Single BDQ Agent Strategy (Paper Replication).
    """
    def __init__(self, env, config, device="cuda" if torch.cuda.is_available() else "cpu", run_name=None, hpo_mode=False):
        self.env = env
        self.config = config
        self.device = device
        self.tracker_rewards = []
        self.tracker_lens = []

        # Sprint 7: Reward normalizer removed (BUG-3). Raw bps rewards used directly.
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Read Action Dims from Config
        action_config = config.get("env", {}).get("action", {})
        if "size_dims" in action_config and int(action_config["size_dims"]) > 0:
            # H2: Leverage-Aware Sizing — MultiDiscrete([size_dims, direction_dims])
            # BDQ "price" branch = size, "qty" branch = direction
            action_dims = (
                int(action_config["size_dims"]),
                int(action_config.get("direction_dims", 3))
            )
        elif "discrete_dims" in action_config:
            action_dims = action_config["discrete_dims"]
        else:
            # Fallback for legacy MultiDiscrete
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

        # Detect agent type from config
        # FIX J-05: Detect by section presence, not fragile type key.
        # If agents.iqn section exists, it's an IQN config.
        self._agent_type = "bdq"  # default
        if "iqn" in config.get("agents", {}):
            self._agent_type = "iqn"

        # Agent Init — dispatch by type
        if self._agent_type == "iqn":
            from finrl_pro_ds.agents.deepscalper.iqn_agent import IQNAgent
            iqn_cfg = config["agents"]["iqn"]
            self.agent = IQNAgent(
                network_config=net_cfg,
                lr=iqn_cfg["learning_rate"],
                gamma=iqn_cfg["gamma"],
                tau=iqn_cfg.get("tau", 0.005),
                batch_size=iqn_cfg.get("batch_size", 256),
                buffer_size=iqn_cfg.get("buffer_size", 500000),
                num_quantiles=iqn_cfg.get("num_quantiles", 32),
                embedding_dim=iqn_cfg.get("embedding_dim", 64),
                noisy_sigma0=iqn_cfg.get("noisy_sigma0", 0.5),
                quantile_huber_kappa=iqn_cfg.get("quantile_huber_kappa", 1.0),
                gradient_clip=iqn_cfg.get("gradient_clip", 10.0),
                auxiliary_weight=iqn_cfg.get("auxiliary_weight", 0.1),
                use_amp=config["training"].get("use_amp", False),
                amp_dtype=config["training"].get("amp_dtype", "float16"),
                use_per=iqn_cfg.get("use_per", False),
                per_alpha=iqn_cfg.get("per_alpha", 0.6),
                per_beta_start=iqn_cfg.get("per_beta_start", 0.4),
                per_beta_frames=iqn_cfg.get("per_beta_frames", 100000),
                torch_compile=config["training"].get("torch_compile", False),
                n_step=iqn_cfg.get("n_step", 1),
                multi_horizon=iqn_cfg.get("multi_horizon", False),
                gamma_short=iqn_cfg.get("gamma_short", 0.95),
                gamma_long=iqn_cfg.get("gamma_long", 0.99),
                horizon_alpha=iqn_cfg.get("horizon_alpha", 0.5),
                fee_threshold=iqn_cfg.get("fee_threshold", 0.0),  # FIX GMO1-05
                action_dims=action_dims,
                device=device,
            )
        else:
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
                epsilon_decay=config["agents"]["bdq"].get("epsilon_decay", 0.99999),
                action_dims=action_dims,
                use_amp=config["training"].get("use_amp", False),
                use_per=config["agents"]["bdq"].get("use_per", False),
                per_alpha=config["agents"]["bdq"].get("per_alpha", 0.6),
                per_beta_start=config["agents"]["bdq"].get("per_beta_start", 0.4),
                per_beta_frames=config["agents"]["bdq"].get("per_beta_frames", 100000),
                exploration_mode=config["agents"]["bdq"].get("exploration_mode", "boltzmann"),
                tau=config["agents"]["bdq"].get("tau", 0.005),
                torch_compile=config["training"].get("torch_compile", False),
                device=device
            )

        # Phase J: Configure stratified sampling for IQN
        # FIX J-02: PER and stratified sampling are mutually exclusive — PER branch
        # silently ignores stratified logic, reintroducing Hold domination.
        replay_cfg = config.get("replay", {})
        _strat_requested = replay_cfg.get("stratified_sampling", False)
        _per_enabled = self.agent.use_per if hasattr(self.agent, "use_per") else False
        if _strat_requested and _per_enabled:
            raise ValueError(
                "FIND-J-02: use_per=True and stratified_sampling=True are mutually exclusive. "
                "PER branch bypasses stratified logic entirely. Disable one."
            )
        if self._agent_type == "iqn" and _strat_requested:
            self.agent.stratified_sampling = True
            self.agent.stratified_hold_action = replay_cfg.get("stratified_hold_action", 1)
            self.agent.stratified_hold_ratio = replay_cfg.get("stratified_hold_ratio", 0.5)
            print(f"[IQN] Stratified sampling enabled: hold_action={self.agent.stratified_hold_action}, "
                  f"hold_ratio={self.agent.stratified_hold_ratio}")

        # Training Params — read from agent-type-specific config section
        _agent_cfg_key = "iqn" if self._agent_type == "iqn" else "bdq"
        _agent_cfg = config["agents"][_agent_cfg_key]
        self.total_timesteps = config["training"]["total_timesteps"]
        self.training_epochs = config["training"].get("training_epochs", 1)  # Paper: ~5 epochs
        self.update_interval = _agent_cfg.get("update_interval", 1.0)
        self.log_interval = config["training"]["log_interval"]
        self.checkpoint_interval = _agent_cfg.get("checkpoint_interval", 500000)
        self.learning_starts = _agent_cfg.get("learning_starts", 5000)

        # PERF: Auto-scale tau for high UTD (update-to-data ratio).
        # Soft target update fires every train_step(). With UTD=N, the target net
        # drifts N times faster per env step. To maintain the same effective tracking
        # rate, scale tau so (1-tau)^N is constant regardless of N.
        # Reference rate: UTD=1 with config tau. Formula: tau_eff = 1 - (1-tau)^(1/N)
        if self.update_interval > 1.0:
            raw_tau = self.agent.tau
            effective_tau = 1.0 - (1.0 - raw_tau) ** (1.0 / self.update_interval)
            self.agent.tau = effective_tau
            print(f"[UTD] update_interval={self.update_interval:.0f} -> tau auto-scaled: "
                  f"{raw_tau:.6f} -> {effective_tau:.6f} "
                  f"(effective target shift per env step: "
                  f"{1-(1-effective_tau)**self.update_interval:.4f})")

        # Checkpoint Directory
        self.ckpt_dir = os.path.join("checkpoints", self.run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # HPO mode: suppress frequent WandB logging to avoid memory flooding
        self.hpo_mode = hpo_mode
        self.gradient_accumulator = 0.0  # FIX FIND-4: Accumulator for fractional update_interval

        # Fee curriculum: sorted list of {step, taker_fee, maker_fee} tier dicts.
        # Applied only during production training (not HPO — HPO uses static base fees).
        # Uses env.call("set_fees", ...) which works for SyncVectorEnv and AsyncVectorEnv.
        raw_schedule = self.config.get("env", {}).get("fee_schedule", [])
        self._fee_schedule = sorted(raw_schedule, key=lambda x: x["step"]) if raw_schedule else []
        self._fee_tier_applied = -1  # Index of the last applied tier (-1 = none yet)

        # FIX PERF-8: Initialize cosine LR scheduler for stable late-training convergence
        # FIX N1: Resolve num_envs from env before use (was NameError)
        _num_envs = getattr(self.env, 'num_envs', 1)
        # FIX FIND-V3-05b: Subtract learning_starts warmup from T_max — no LR steps during warmup
        effective_steps = max(self.total_timesteps - self.learning_starts, 1)
        total_updates = int(effective_steps * self.training_epochs * self.update_interval / _num_envs) if _num_envs > 0 else 100000
        self.agent._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.agent.optimizer, T_max=total_updates, eta_min=1e-6
        )

        # FIX FIND-5 + CRIT-1: Private size consistency check
        # V5 Tier 2: 5 dims (pos, bal, time, order_dir, order_dist)
        # V5 H2: 6 dims (+spread_bps for leverage-aware sizing)
        # V6 Swing: 4 dims (direction, bars_since_switch, unrealized_pnl, atr)
        priv_cfg = config.get("network", {}).get("micro_config", {}).get("private_input_size", 5)
        mdp_version = config.get("env", {}).get("mdp_version", "v5")
        if mdp_version == "v6":
            expected_priv = 4
            label = "V6 Swing MDP"
        else:
            include_spread = config.get("features", {}).get("include_spread", False)
            expected_priv = 6 if include_spread else 5
            label = "H2 with spread" if include_spread else "Tier 2"
        assert priv_cfg == expected_priv, (
            f"FIND-5 Mismatch: Env produces {expected_priv} private features "
            f"({label}), config expects {priv_cfg}"
        )

    def _seed_demo_buffer(self, demo_steps: int) -> None:
        """Pre-populate replay buffer with a rule-based momentum policy (DQfD warm-start).

        Runs a simple OFI/return-momentum policy through the training env for demo_steps
        steps and pushes all transitions directly to agent.memory. This breaks the
        cold-start catastrophe: the buffer has profitable-ish trajectories before
        epsilon-greedy exploration begins, so the agent's early Q-updates see signal
        rather than pure noise.

        Policy: buy if macro[:,0] (logret_5) > 5bps, sell if < -5bps, else hold.
        Uses observable features only (no lookahead) — not oracle cheating.

        Args:
            demo_steps: Number of env steps. demo_steps * num_envs transitions are stored.
                        Set to buffer_size // num_envs to fill the buffer exactly once.
        """
        logger = logging.getLogger(__name__)

        action_cfg = self.config.get("env", {}).get("action", {})
        size_dims = int(action_cfg.get("size_dims", 0))

        if size_dims > 0:
            # H2: MultiDiscrete([size_dims, direction_dims]) — direction is branch 1
            BUY_DIR, HOLD_DIR, SELL_DIR = 0, 1, 2
            DEFAULT_SIZE_IDX = size_dims // 2  # 1x multiplier (middle of size range)
        else:
            discrete_dims = action_cfg.get("discrete_dims", 6)
            if discrete_dims == 3:
                BUY_ACTION, HOLD_ACTION, SELL_ACTION = 0, 1, 2
            else:
                BUY_ACTION, HOLD_ACTION, SELL_ACTION = 0, 2, 5

        THRESHOLD = 0.0005  # 5bps logret_5 threshold for directional signal
        num_envs = self.env.num_envs
        total_transitions = demo_steps * num_envs
        logger.info(f"[DQfD] Seeding buffer: {demo_steps} steps x {num_envs} envs = {total_transitions} transitions")
        print(f"[DQfD] Seeding replay buffer with {total_transitions} momentum-policy demos...")

        obs, _ = self.env.reset()
        n_pushed = 0

        for _ in range(demo_steps):
            signal = obs["macro"][:, 0]  # logret_5, shape (num_envs,)

            if size_dims > 0:
                # H2: Build (num_envs, 2) actions — [size_idx, dir_idx]
                dir_actions = np.where(signal > THRESHOLD, BUY_DIR,
                              np.where(signal < -THRESHOLD, SELL_DIR, HOLD_DIR))
                size_actions = np.full(num_envs, DEFAULT_SIZE_IDX, dtype=np.intp)
                actions = np.stack([size_actions, dir_actions], axis=1)  # (num_envs, 2)
            else:
                actions = np.where(signal > THRESHOLD, BUY_ACTION,
                          np.where(signal < -THRESHOLD, SELL_ACTION, HOLD_ACTION))

            next_obs, rewards, term, trunc, infos = self.env.step(actions)
            dones_for_buffer = term

            for i in range(num_envs):
                s = {k: v[i] for k, v in obs.items()}
                ns = {k: v[i] for k, v in next_obs.items()}
                aux_target = 0.0
                if isinstance(infos, list):
                    aux_target = float(infos[i].get("volatility_target", 0.0))
                elif isinstance(infos, dict) and "volatility_target" in infos:
                    val = infos["volatility_target"]
                    aux_target = float(val[i]) if hasattr(val, "__getitem__") else float(val)
                self.agent.memory.push(s, actions[i], float(rewards[i]),
                                       ns, bool(dones_for_buffer[i]), aux_target)
                n_pushed += 1

            obs = next_obs

        print(f"[DQfD] Buffer seeded: {n_pushed} transitions ({len(self.agent.memory)} in buffer). "
              f"Resuming with epsilon-greedy exploration.")

    def train(self, start_step=0, skip_reset=False, optuna_trial=None, pruning_callback=None):
        """Single Phase Training Loop

        Args:
            start_step: Resume training from this step (for chunked HPO).
            skip_reset: If True, reuse stored obs from previous chunk instead of resetting.
            optuna_trial: Optuna Trial object for HPO reporting/pruning.
            pruning_callback: Function() -> float to evaluate agent during training.
        """
        # DQfD warm-start: pre-populate buffer with rule-based demos before training.
        # Only runs on fresh start (start_step==0) — not on HPO trial resumptions.
        # FIX K01: Demo seeding uses V5 action semantics (Buy/Hold/Sell) — skip for V6 Discrete(2).
        mdp_ver = self.config.get("env", {}).get("mdp_version", "v5")
        demo_steps = self.config.get("agents", {}).get("bdq", {}).get("demo_seed_steps", 0)
        if demo_steps > 0 and start_step == 0 and mdp_ver != "v6":
            self._seed_demo_buffer(demo_steps)

        print(f"Starting Training: Single BDQ Agent | Device: {self.device} | Start Step: {start_step} | Epochs: {self.training_epochs}")

        # Init State - Obs is Dict: {'micro': ..., 'macro': ..., 'private': ...}
        if skip_reset and hasattr(self, '_current_obs') and self._current_obs is not None:
            obs = self._current_obs
            print(f"  Resuming from stored observation (skip_reset=True)")
        else:
            obs, _ = self.env.reset()

        # Defensive Shape Assertions — runs once before training loop starts
        B = self.env.num_envs
        W = self.config.get("env", {}).get("window_size", 15)
        _micro_dim = self.config.get("network", {}).get("micro_config", {}).get("input_size", 30)
        assert obs["micro"].shape == (B, W, _micro_dim), \
            f"obs['micro'] shape mismatch: expected ({B}, {W}, {_micro_dim}), got {obs['micro'].shape}"
        _priv_dim = self.config.get("network", {}).get("micro_config", {}).get("private_input_size", 5)
        assert obs["private"].shape == (B, W, _priv_dim), \
            f"obs['private'] shape mismatch: expected ({B}, {W}, {_priv_dim}), got {obs['private'].shape}"
        assert obs["macro"].ndim == 2, \
            f"obs['macro'] expected 2D (B, M), got shape {obs['macro'].shape}"
        print(f"[OK] Observation shapes verified: micro={obs['micro'].shape}, "
              f"private={obs['private'].shape}, macro={obs['macro'].shape}")

        # FIX PERF-2: Dynamic epsilon decay for BOTH HPO and production training.
        # Ensures exploration schedule matches actual training budget regardless of num_envs.
        # IQN uses NoisyNets — epsilon is always 0, skip this entire block.
        num_envs = self.env.num_envs

        if self.hpo_mode:
            steps_per_trial = self.config.get("hpo", {}).get("steps_per_trial", 50000)
            n_calls = len(list(range(0, steps_per_trial, num_envs)))
        else:
            # Production: compute from total_timesteps
            n_calls = self.total_timesteps // num_envs

        if n_calls > 0 and self._agent_type != "iqn":
            # Scale epsilon decay to reach epsilon_end at exploration_fraction of total training
            exploration_fraction = self.config.get("agents", {}).get("bdq", {}).get("exploration_fraction", 0.5)
            total_calls = n_calls * self.training_epochs
            explore_calls = max(int(total_calls * exploration_fraction), 1)
            epsilon_end = self.config.get("agents", {}).get("bdq", {}).get("epsilon_end", 0.01)
            computed_decay = np.exp(np.log(max(epsilon_end, 1e-10)) / explore_calls)
            self.agent.epsilon_decay = computed_decay
            # FIX BUG-07: Store schedule params for closed-form linear decay
            self.agent._epsilon_start = self.agent.epsilon  # current epsilon (should be epsilon_start)
            self.agent._epsilon_decay_steps = explore_calls
            self.agent.step_count = 0  # Reset step counter for decay schedule
            # FIX BUG-04: Assert epsilon_end consistency between config and agent
            assert abs(self.agent.epsilon_end - epsilon_end) < 1e-9, (
                f"Epsilon end mismatch: agent={self.agent.epsilon_end}, config={epsilon_end}. "
                f"Config was likely mutated after agent creation."
            )
            if self.hpo_mode:
                print(f"[HPO] Overriding epsilon_decay to {computed_decay:.6f} for {steps_per_trial} steps (~{explore_calls} explore updates of {total_calls} total, {self.training_epochs} epochs)")
            else:
                print(f"[Train] Computed epsilon_decay = {computed_decay:.6f} (explore over {explore_calls}/{total_calls} updates, fraction={exploration_fraction})")

        global_step = start_step
        episode_rewards = deque(maxlen=100)
        episode_lens = deque(maxlen=100)

        curr_rewards = np.zeros(num_envs)
        curr_lens = np.zeros(num_envs)

        # Reward component accumulators for hindsight ratio tracking
        _acc_hindsight = 0.0
        _acc_total = 0.0

        # FIND-3: Removed redundant 'import time'
        start_time = time.time()

        # Reset pruning rung tracker for fresh trial (P0 fix)
        # Initialize to 0 to skip rung 0 - avoids eval at step ~12 before learning starts (P1a fix)
        self._last_prune_rung = 0

        # FIX R8-AUD-01: Truncated episode handling — see buffer push block in train().

        # N-step return buffer: wraps replay buffer push for multi-step returns
        # When multi_horizon=True, use gamma_long for N-step discounting so the
        # long-horizon Bellman target is exactly correct. Short-horizon has a small
        # approximation error (~0.04*r per intermediate step) which is acceptable.
        self._nstep_buffer = None
        if self._agent_type == "iqn":
            _n_step = self.config.get("agents", {}).get("iqn", {}).get("n_step", 1)
            if _n_step > 1:
                from finrl_pro_ds.agents.deepscalper.nstep_buffer import NStepBuffer
                _nstep_gamma = self.agent.gamma_long if self.agent.multi_horizon else self.agent.gamma
                self._nstep_buffer = NStepBuffer(
                    n=_n_step, gamma=_nstep_gamma, num_envs=num_envs
                )
                print(f"[N-Step] Enabled: n={_n_step}, gamma={_nstep_gamma}, "
                      f"gamma^n={_nstep_gamma ** _n_step:.6f}"
                      f"{' (multi_horizon: using gamma_long)' if self.agent.multi_horizon else ''}")

        # FIX PERF-3 + N4: Hoist extract_tensors outside loop
        # PERF FIX-4: non_blocking H2D transfers
        def extract_tensors(o, device=self.device):
            return (
                torch.as_tensor(o["micro"], dtype=torch.float32).to(device, non_blocking=True),
                torch.as_tensor(o["private"], dtype=torch.float32).to(device, non_blocking=True),
                torch.as_tensor(o["macro"], dtype=torch.float32).to(device, non_blocking=True)
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

                # BUG-B: Reset per-epoch hidden state (stateless LSTM assumption for training, but inference needs clean slate)
                if hasattr(self.agent, "reset_hidden_state"):
                    self.agent.reset_hidden_state()

            epoch_step = 0
            qty_mask = None  # ARCH-3: Will be populated from env info after first step
            while epoch_step < self.total_timesteps:
                # Convert to torch for prediction
                micro_t, private_t, macro_t = extract_tensors(obs)

                # Predict (Returns numpy array of actions [B, 2])
                # ARCH-3: Pass qty mask to block invalid actions at position limits
                actions = self.agent.predict(micro_t, private_t, macro_t, deterministic=False, qty_mask=qty_mask)

                # 2. Step Environment
                next_obs, rewards, term, trunc, infos = self.env.step(actions)

                # ARCH-3: Extract qty_action_mask for NEXT step's predict()
                if isinstance(infos, dict) and "qty_action_mask" in infos:
                    qty_mask = infos["qty_action_mask"]
                elif isinstance(infos, list) and len(infos) > 0 and "qty_action_mask" in infos[0]:
                    qty_mask = np.stack([info_i["qty_action_mask"] for info_i in infos])
                else:
                    qty_mask = None

                # Episode boundary: term (drawdown stop) or trunc (data/episode_length).
                dones_for_reset = np.logical_or(term, trunc)  # For episodic stats + hidden state

                # BUG-B: Mask hidden states for terminated/truncated envs
                if hasattr(self.agent, "mask_hidden_state"):
                    self.agent.mask_hidden_state(dones_for_reset)
                # NOTE: dones_for_buffer is computed in the buffer push block below
                # (FIX R8-AUD-01: includes truncation to prevent Q-bootstrap from reset obs)

                # 3. Store in Buffer
                # PERF FIX-2: Extract aux_targets vectorized
                aux_targets_vec = np.zeros(num_envs, dtype=np.float32)
                if isinstance(infos, dict) and "volatility_target" in infos:
                    val = infos["volatility_target"]
                    if hasattr(val, "__getitem__"):
                        aux_targets_vec[:] = val[:num_envs]
                    else:
                        aux_targets_vec[:] = float(val)
                elif isinstance(infos, list):
                    for i in range(num_envs):
                        aux_targets_vec[i] = float(infos[i].get("volatility_target", 0.0))

                # PERF FIX-2: Batch push for FlatReplayBuffer (not PER — PER stores Python objects)
                from finrl_pro_ds.agents.deepscalper.flat_replay_buffer import FlatReplayBuffer
                _actions = actions if actions.ndim > 1 else actions.reshape(-1, 1)

                # FIX R8-AUD-01: On truncation (trunc=True, term=False), SyncVectorEnv
                # auto-resets and returns the RESET observation as next_obs. Bootstrapping
                # Q(s_reset) pollutes the Bellman target with cross-episode garbage.
                # Fix: treat truncation as terminal in the buffer (done=1.0) to zero the
                # Q-bootstrap. This is a standard approximation (SB3 does the same when
                # final_observation is unavailable). Bias is minimal: only affects the
                # last transition per episode (~0.1% of data).
                # NOTE: dones_for_reset (used for stats/hidden masking) is unchanged.
                dones_for_buffer = np.logical_or(term, trunc).astype(np.float32)

                if self._nstep_buffer is not None:
                    # N-step returns: accumulate before pushing to replay
                    # FIX GMO1-04: Pass resets (term|trunc) so n-step flushes at
                    # episode boundaries, preventing cross-episode reward mixing.
                    self._nstep_buffer.add(
                        obs, _actions, rewards.astype(np.float32),
                        next_obs, dones_for_buffer,
                        aux_targets_vec, self.agent.memory,
                        resets=dones_for_reset.astype(np.float32),
                    )
                elif isinstance(self.agent.memory, FlatReplayBuffer):
                    self.agent.memory.push_batch(
                        obs, _actions, rewards.astype(np.float32),
                        next_obs, dones_for_buffer,
                        aux_targets_vec,
                    )
                else:
                    # PER fallback: per-transition push
                    for i in range(num_envs):
                        s = {k: v[i] for k, v in obs.items()}
                        ns = {k: v[i] for k, v in next_obs.items()}
                        self.agent.memory.push(
                            s, actions[i], float(rewards[i]),
                            ns, bool(dones_for_buffer[i]), float(aux_targets_vec[i])
                        )

                # Accumulate reward components for hindsight ratio tracking
                if isinstance(infos, dict):
                    rh = infos.get("reward_hindsight", None)
                    rt = infos.get("reward_total", None)
                    if rh is not None and rt is not None:
                        for i in range(num_envs):
                            _acc_hindsight += abs(float(rh[i]) if hasattr(rh, "__getitem__") else float(rh))
                            _acc_total += abs(float(rt[i]) if hasattr(rt, "__getitem__") else float(rt))

                # Track Episodic Stats (lightweight per-env loop)
                for i in range(num_envs):
                    curr_rewards[i] += rewards[i]
                    curr_lens[i] += 1
                    if dones_for_reset[i]:
                        episode_rewards.append(curr_rewards[i])
                        episode_lens.append(curr_lens[i])
                        curr_rewards[i] = 0
                        curr_lens[i] = 0

                obs = next_obs
                global_step += num_envs
                epoch_step += num_envs

                # Fee curriculum: activate the highest eligible tier (production only).
                # Each tier defines the fees that apply from `step` onwards.
                if self._fee_schedule and not self.hpo_mode:
                    highest_eligible = -1
                    for tier_idx, tier in enumerate(self._fee_schedule):
                        if global_step >= tier["step"]:
                            highest_eligible = tier_idx
                    if highest_eligible > self._fee_tier_applied:
                        tier = self._fee_schedule[highest_eligible]
                        t_fee = float(tier.get("taker_fee", self.config["env"].get("taker_fee", 0.0005)))
                        m_fee = float(tier.get("maker_fee", self.config["env"].get("maker_fee", 0.0002)))
                        self.env.call("set_fees", t_fee, m_fee)
                        self._fee_tier_applied = highest_eligible
                        wandb.log({"train/fee_tier": highest_eligible,
                                   "train/taker_fee_bps": t_fee * 10000,
                                   "train/maker_fee_bps": m_fee * 10000,
                                   "step": global_step})
                        print(f"[FeeCurriculum] Step {global_step}: tier {highest_eligible} activated "
                              f"→ taker={t_fee*10000:.1f}bps, maker={m_fee*10000:.1f}bps")

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

                # 4b. HPO Pruning Check — dual strategy:
                #   (a) Optuna Hyperband pruner for score-based inter-trial comparison
                #   (b) Early-kill based on TD-error divergence (no learning signal)
                # PERF-OPT: Reduced min_pruning_steps from 20K to 15K and prune_interval
                # from 5K to 3K for faster trial turnover. At 500K steps/trial,
                # aggressive pruning saves ~60% of wasted compute on dead trials.
                if optuna_trial and pruning_callback:
                    min_pruning_steps = 15000
                    prune_interval = 3000
                    current_rung = global_step // prune_interval

                    if global_step > min_pruning_steps and current_rung > getattr(self, '_last_prune_rung', -1):
                        self._last_prune_rung = current_rung
                        print(f"  [HPO] Probing agent at step {global_step} (rung {current_rung})...")
                        score = pruning_callback()
                        print(f"  [HPO] Step {global_step} Score: {score:.4f}")

                        # Track score history for early-kill on flat/diverging trials
                        if not hasattr(self, '_hpo_score_history'):
                            self._hpo_score_history = []
                        self._hpo_score_history.append((global_step, score))

                        # FIX HPO-2: Disabled early-kill for swing MDP validation.
                        # PF < 0.8 threshold was calibrated for old Discrete(6) MDP.
                        # IQN+NoisyNets on binary swing MDP needs full trial duration
                        # to show signal. All K1-K4 trials were killed by this gate.
                        # TODO: Re-enable with calibrated threshold once swing MDP baseline is established.

                        optuna_trial.report(score, global_step)

                        if optuna_trial.should_prune():
                            print(f"  [HPO] Pruning trial at step {global_step}")
                            raise optuna.TrialPruned()

                # 5. Checkpointing
                if global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}.pth")

        # Store obs for potential resume via skip_reset=True
        self._current_obs = obs

        # Final Save — only if actual training occurred (guards against
        # total_timesteps=0 flows overwriting loaded weights with random init)
        if n_calls > 0:
            self.save_checkpoint("checkpoint_final.pth")
        else:
            print("⚠ Skipping checkpoint_final.pth save: no training steps executed (n_calls=0)")

        # Always log final step status to ensure graph continuity
        if not self.hpo_mode:
            print(f"Logging final metrics at step {global_step}")
            wandb.log({"step": global_step, "train/final_step": 1})

        print("Training Complete.")

    def save_checkpoint(self, filename):
        path = os.path.join(self.ckpt_dir, filename)
        self.agent.save(path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path, strict: bool = True):
        self.agent.load(path, strict=strict)
        print(f"Loaded checkpoint: {path} (strict={strict})")
