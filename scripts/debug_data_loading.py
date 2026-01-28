import os
import sys
import yaml
import traceback
import argparse
import multiprocessing
# Fix: Ensure finrl_pro_ds is in path
sys.path.append(os.getcwd())

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def test_data_loading(config_path, feature_config_path=None):
    print("Testing ParquetDataHandler Data Loading...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_config = config.get("data", {})
    # Override with local path for testing
    file_path = "c:/data/btc_lob_jan2023.parquet"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"Found file: {file_path}")
    
    try:
        # 1. Load Data (Main Process)
        print("Initializing ParquetDataHandler...")
        handler = ParquetDataHandler(
            file_path=file_path,
            ticker="BTCUSDT",
            feature_config=config.get("features", {})
        )
        print("Data Loaded Successfully.")
        print(f"Data Length: {handler._len}")
        
        # 2. Create Shared Memory
        print("Creating Shared Memory...")
        shm_config = handler.create_shared_memory()
        print(f"Shared Memory Created: {shm_config}")
        
        # 3. Test Reading from SHM in subprocess
        print("Testing Subprocess Access...")
        config.get("data", {})["shared_memory_config"] = shm_config
        
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=worker_fn, args=(file_path, config))
        p.start()
        p.join()
        
        if p.exitcode == 0:
            print("Subprocess Success.")
        else:
            print(f"Subprocess Failed with exit code {p.exitcode}")
            
        # 4. Cleanup
        print("Cleaning up SHM...")
        handler.close_shared_memory(unlink=True)
        print("Done.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

def worker_fn(file_path, config):
    try:
        print(f"[Worker] Loading Handler with SHM Config...")
        handler = ParquetDataHandler(
            file_path=file_path,
            ticker="BTCUSDT",
            feature_config=config.get("features", {}),
            shared_memory_config=config.get("data", {}).get("shared_memory_config")
        )
        print(f"[Worker] Data Attached. Length: {handler._len}, Cols: {len(handler._feature_cols)}")
        # Test read
        row = handler.step()
        print(f"[Worker] First row type: {type(row)}")
        if row:
             print(f"[Worker] Timestamp: {row.get('timestamp')}")
    except Exception as e:
        print(f"[Worker] FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepscalper_unified.yaml")
    args = parser.parse_args()
    
    test_data_loading(args.config)
