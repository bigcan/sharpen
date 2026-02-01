
import os
import sys
import numpy as np
import yaml
import logging

# Add project root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# Setup logging
logging.basicConfig(level=logging.INFO)

def verify_env():
    print("--- Verifying DeepScalperEnv Logic ---")
    
    # Load Config
    config_path = "configs/deepscalper_anti_churn.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_config = config['env']
    data_config = config['data']
    
    # 1. Setup Data Handler
    file_path = data_config.get('file_path')
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}. Looking in data/...")
        if os.path.exists("data/btc_lob_jan2023.parquet"):
            file_path = "data/btc_lob_jan2023.parquet"
        else:
            print("No data found. Cannot run verification.")
            return

    print(f"Loading data from {file_path}...")
    handler = ParquetDataHandler(
        file_path=file_path, 
        ticker="BTCUSDT",
        feature_config=config['features']
    )
    
    # 2. Init Env
    print("Initializing Env...")
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    obs, info = env.reset()
    
    print(f"Initial Balance: {env.balance}")
    print(f"Initial Position: {env.position}")
    print(f"Best Ask: {env.current_best_ask}")
    print(f"Micro Window Sample: {env.micro_window[-1, :4]}") # BidP, BidV, AskP, AskV
    
    # 3. Force BUY Action (Action 1, Price 4, Vol 0)
    # Price 4 = Offset 4 = Limit at Ask - 4 ticks (Deep Passive)
    # Vol 0 = 10%
    action = np.array([1, 4, 0]) 
    
    print("\n--- Step 1: Send BUY Order (Price Bin 4) ---")
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Pending Order: {env.pending_order}")
    print(f"Position: {env.position}")
    print(f"Balance: {env.balance}")
    
    # 4. Step 2: Execution (Should fill if price allows)
    print("\n--- Step 2: Execution Step ---")
    # Action Hold (0, 0, 0)
    action_hold = np.array([0, 0, 0])
    obs, reward, terminated, truncated, info = env.step(action_hold)
    
    print(f"Pending Order (Should be None): {env.pending_order}")
    print(f"Position: {env.position}")
    print(f"Balance: {env.balance}")
    print(f"Transaction Cost: {env.step_transaction_costs}")
    print(f"Reward: {reward}")
    print(f"Current Ask: {env.current_best_ask}")
    
    if env.position > 0:
        print("\nSUCCESS: Order Filled!")
    else:
        print("\nFAILURE: Order Did NOT Fill.")
        # Debugging
        print(f"Micro Window Last: {env.micro_window[-1, :4]}")
        print(f"Pre-Execution Ask (from Order): {env.current_best_ask}") # Wait, this is T+2
        
    print("\n--- Running 100 steps loop (Price Bin 4) ---")
    filled_count = 0
    for i in range(100):
        # Alternate Buy / Sell
        if env.position == 0:
            act = np.array([1, 4, 0]) # Buy Deep
        else:
            act = np.array([2, 4, 0]) # Sell Deep
            
        obs, reward, term, trunc, info = env.step(act)
        
        if env.step_transaction_costs > 0:
             filled_count += 1
             
    print(f"Total Fills in 100 steps: {filled_count}")

    print("\n--- Verifying Shared Memory Logic ---")
    if config['training'].get('use_shm', False):
        try:
            print("Creating Shared Memory...")
            shm_config = handler.create_shared_memory()
            print("SHM Created. Simulating Worker Attach...")
            
            # Create new handler instance as "Worker"
            worker_handler = ParquetDataHandler(
                file_path=file_path,
                ticker="BTCUSDT",
                feature_config=config['features'],
                shared_memory_config=shm_config
            )
            
            print("Worker Attached.")
            worker_step = worker_handler.step()
            print(f"Worker Step 0 Keys: {list(worker_step.keys())}")
            print(f"Worker Step 0 Bid1: {worker_step.get('bid_price_1')}")
            
            if float(worker_step.get('bid_price_1', 0)) == 0:
                print("CRITICAL FAILURE: Worker read 0.0 from SHM!")
            else:
                print("SUCCESS: Worker read valid data from SHM.")
                
            handler.close_shared_memory(unlink=True)
            
        except Exception as e:
            print(f"SHM Test Failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping SHM Test (use_shm=False in config).")

if __name__ == "__main__":
    verify_env()
