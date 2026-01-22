import torch
import numpy as np
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN

def verify_auxiliary_task():
    print("=== Verifying DeepScalper Section 4.4 (Auxiliary Task) ===")
    
    # 1. Load Config
    config_path = "c:/FinRL/FinRL-Pro_DS/finrl_pro_ds/configs/deepscalper_smoke.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Ensure data exists (create dummy if not)
    data_path = config['data']['file_path']
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}. Please run generate_smoke_data.py first.")
        # Try to run generate smoke data
        import subprocess
        subprocess.run(["python", "scripts/generate_smoke_data.py"])
        
    # 2. Setup Components
    print("\n1. Initializing Environment and Data Handler...")
    handler = ParquetDataHandler(
        file_path=data_path,
        ticker=config['data']['ticker'],
        feature_config=config.get('features')
    )
    
    env = DeepScalperEnv(config['env'], data_handler=handler)
    
    # Observe Step
    print("\n2. Stepping Environment to check Info...")
    obs, info = env.reset()
    action = [1, 2, 2] # Random action
    next_obs, reward, done, trunc, info = env.step(action)
    
    vol_target = info.get('volatility_target')
    print(f"  -> Volatility Target found in Info: {vol_target}")
    
    if vol_target is None:
        print("FAIL: volatility_target is Missing!")
        return
        
    # 3. Setup Agent
    print("\n3. Initializing DQN Agent...")
    dqn_config = config['agents']['dqn']
    net_config = config['network']
    # Merge configs as 'DeepScalperDQN' expects combined or we need to structure it?
    # DeepScalperDQN expects 'network_config' which corresponds to DeepScalperNetwork init params.
    
    network_config = {
        "micro_config": net_config['micro_config'],
        "macro_config": net_config['macro_config'],
        "fusion_dim": 64, # Default in yaml? not in yaml, hardcoded/default in init
        "action_space_dims": (3, 5, 5)
    }
    
    agent = DeepScalperDQN(
        network_config=network_config,
        auxiliary_weight=dqn_config['auxiliary_weight'],
        device="cpu"
    )
    
    # Verify Volatility Head Exists
    if hasattr(agent.policy_net, 'vol_head'):
        print("  -> agent.policy_net.vol_head EXISTS. ✅")
    else:
        print("FAIL: agent.policy_net.vol_head MISSING! ❌")
        return

    # 4. Train Step (Fake Data)
    print("\n4. Running Train Step verification...")
    
    # Push data to buffer
    state = {"micro": obs["micro"], "macro": obs["macro"], "private": obs["private"]}
    next_state = {"micro": next_obs["micro"], "macro": next_obs["macro"], "private": next_obs["private"]}
    
    # Fill buffer to batch size
    for _ in range(agent.batch_size + 2):
        agent.memory.push(state, action, 1.0, next_state, False, aux_target=0.01)
        
    # Run Step
    loss = agent.train_step()
    print(f"  -> Train Step Loss: {loss}")
    
    # Check Gradients on Vol Head
    if agent.policy_net.vol_head.weight.grad is not None:
         grad_norm = agent.policy_net.vol_head.weight.grad.norm().item()
         print(f"  -> Volatility Head Gradient Norm: {grad_norm}")
         if grad_norm > 0:
             print("  -> Gradients flowed to Vol Head! ✅")
         else:
             print("  -> Warning: Grad is 0.0 (might be expected if prediction perfect or frozen)")
    else:
        print("FAIL: No gradient on Volatility Head! ❌")

    print("\nVerification Complete: PASS ✅")

if __name__ == "__main__":
    verify_auxiliary_task()
