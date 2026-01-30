
import yaml
import sys
import pandas as pd
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def verify_config_dates():
    try:
        with open('configs/deepscalper_unified.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        data_config = config['data']
        print(f"Testing Config: {data_config}")
        
        print("\n--- Testing Data Splits ---")
        
        splits = [
            ("Train", data_config['train_start_date'], data_config['train_end_date']),
            ("Val",   data_config['val_start_date'],   data_config['val_end_date']),
            ("Test",  data_config['test_start_date'],  data_config['test_end_date'])
        ]
        
        for name, start, end in splits:
            print(f"\nScanning {name}: {start} to {end}")
            # Re-init handler for each split to simulate DeepScalperEnv usage
            handler = ParquetDataHandler(
                file_path=data_config['file_path'],
                ticker=data_config['ticker'],
                start_date=start,
                end_date=end
            )
            # data matches handler.df after load_data
            handler.load_data() 
            
            count = handler._len
            print(f"{name} Data: {count} rows")
            assert count > 0, f"{name} data is empty!"
        
        print("\nSUCCESS: All data splits loaded successfully with correct dates.")
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_config_dates()
