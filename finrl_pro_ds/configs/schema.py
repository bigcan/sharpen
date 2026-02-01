from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import yaml
import os
from pathlib import Path

@dataclass
class WandbConfig:
    project: str = "FinRL-Pro-DS"
    entity: str = "bigcan-chiwin-technology"
    mode: str = "online"
    tags: List[str] = field(default_factory=list)
    name: Optional[str] = None

@dataclass
class DataConfig:
    file_path: str
    ticker: str = "BTCUSDT"
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    val_start_date: Optional[str] = None
    val_end_date: Optional[str] = None
    time_feature: str = "timestamp"
    # Shared memory config is injected at runtime, not usually in static yaml
    shared_memory_config: Optional[Dict[str, Any]] = None

@dataclass
class FeatureConfig:
    micro_features: List[str] = field(default_factory=lambda: ["micro"]) # Placeholder defaults
    macro_features: List[str] = field(default_factory=lambda: ["macro"])
    private_features: List[str] = field(default_factory=lambda: ["position", "balance"])

@dataclass
class RewardConfig:
    profit_weight: float = 1.0
    volatility_penalty_weight: float = 0.0
    volatility_horizon: int = 100
    risk_aversion: float = 0.0 # From paper if applicable

@dataclass
class ActionConfig:
    direction_bins: int = 3   # Buy, Hold, Sell
    price_bins: int = 5       # Aggressiveness levels
    volume_bins: int = 5      # Size levels

@dataclass
class EnvConfig:
    symbol: str = "BTCUSDT"
    initial_balance: float = 100000.0  # USDT
    transaction_fee: float = 0.0001
    num_envs: int = 1
    window_size: int = 50
    reward: RewardConfig = field(default_factory=RewardConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

@dataclass
class NetworkConfig:
    micro_input_size: int = 20
    macro_input_size: int = 11
    private_input_size: int = 2
    hidden_size: int = 64
    embedding_dim: int = 0  # If using embeddings

@dataclass
class AgentConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    # Algo specific
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    dqn_buffer_size: int = 100000
    dqn_batch_size: int = 64
    dqn_target_update_freq: int = 1000

@dataclass
class AgentsContainerConfig:
    dqn: AgentConfig = field(default_factory=AgentConfig)
    ppo: AgentConfig = field(default_factory=AgentConfig)
    a2c: AgentConfig = field(default_factory=AgentConfig)
    gating: AgentConfig = field(default_factory=AgentConfig)

@dataclass
class TrainConfig:
    total_timesteps: int = 1000000
    batch_size: int = 4096  # Global batch size
    learning_rate: float = 1e-4 # Fallback
    log_interval: int = 1000
    checkpoint_interval: int = 10000
    torch_compile: bool = True
    use_amp: bool = True    # Mixed Precision
    use_shm: bool = True    # Shared Memory

@dataclass
class UnifiedConfig:
    data: DataConfig
    features: FeatureConfig
    env: EnvConfig
    network: NetworkConfig
    agents: AgentsContainerConfig
    training: TrainConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)

class ConfigLoader:
    @staticmethod
    def load_yaml(path: str) -> UnifiedConfig:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            raw_dict = yaml.safe_load(f)
            
        return ConfigLoader._parse_dict(raw_dict)
    
    @staticmethod
    def _parse_dict(d: Dict[str, Any]) -> UnifiedConfig:
        # Helper to safely instantiate nesting
        # This is a manual recursive parsing to ensure strict typing or defaults
        
        # 1. Data
        data_d = d.get("data", {})
        data_cfg = DataConfig(
            file_path=data_d.get("file_path", ""),
            ticker=data_d.get("ticker", "BTCUSDT"),
            train_start_date=data_d.get("train_start_date") or data_d.get("start_date"),
            train_end_date=data_d.get("train_end_date") or data_d.get("end_date"),
            val_start_date=data_d.get("val_start_date"),
            val_end_date=data_d.get("val_end_date")
        )
        
        # 2. Features
        feat_d = d.get("features", {})
        feat_cfg = FeatureConfig(
            micro_features=feat_d.get("micro_features", ["micro"]),
            macro_features=feat_d.get("macro_features", ["macro"]),
            private_features=feat_d.get("private_features", ["position", "balance"])
        )
        
        # 3. Env (Nested Reward/Action)
        env_d = d.get("env", {})
        rew_d = env_d.get("reward", {})
        act_d = env_d.get("action", {})
        
        rew_cfg = RewardConfig(**{k:v for k,v in rew_d.items() if k in RewardConfig.__annotations__})
        act_cfg = ActionConfig(**{k:v for k,v in act_d.items() if k in ActionConfig.__annotations__})
        
        env_cfg = EnvConfig(
            symbol=env_d.get("symbol", "BTCUSDT"),
            initial_balance=float(env_d.get("initial_balance", 100000.0)),
            transaction_fee=float(env_d.get("transaction_fee", 0.0001)),
            num_envs=int(env_d.get("num_envs", 1)),
            window_size=int(env_d.get("window_size", 50)),
            reward=rew_cfg,
            action=act_cfg
        )
        
        # 4. Network
        net_d = d.get("network", {})
        # Flattening simple structure from legacy or keeping explicit?
        # Let's support the structure: network: { micro_config: {...}, ... } OR flat.
        # Supporting flat for simplicity in Unified.
        net_cfg = NetworkConfig(
            micro_input_size=net_d.get("micro_input_size", 20),
            macro_input_size=net_d.get("macro_input_size", 11),
            hidden_size=net_d.get("hidden_size", 64)
        )
        
        # 5. Agents
        agents_d = d.get("agents", {})
        
        def parse_agent(ad):
            return AgentConfig(**{k:v for k,v in ad.items() if k in AgentConfig.__annotations__})
            
        agents_cfg = AgentsContainerConfig(
            dqn=parse_agent(agents_d.get("dqn", {})),
            ppo=parse_agent(agents_d.get("ppo", {})),
            a2c=parse_agent(agents_d.get("a2c", {})),
            gating=parse_agent(agents_d.get("gating", {}))
        )
        
        # 6. Training
        train_d = d.get("training", {})
        train_cfg = TrainConfig(**{k:v for k,v in train_d.items() if k in TrainConfig.__annotations__})
        
        # 7. WandB
        wandb_d = d.get("wandb", {})
        wandb_cfg = WandbConfig(**{k:v for k,v in wandb_d.items() if k in WandbConfig.__annotations__})
        
        return UnifiedConfig(
            data=data_cfg,
            features=feat_cfg,
            env=env_cfg,
            network=net_cfg,
            agents=agents_cfg,
            training=train_cfg,
            wandb=wandb_cfg
        )
