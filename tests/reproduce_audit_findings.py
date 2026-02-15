import gymnasium as gym
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

def test_remaining_time_bug():
    print(">>> Testing BUG-A: remaining_time reset after order fill...")
    
    # Mock config
    config = {
        "symbol": "BTCUSDT",
        "ticker_list": ["BTCUSDT"],
        "period": "1m",
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "transaction_fee": 0,
        "max_position": 1.0,
        "initial_balance": 100000.0,
        "window_size": 50,
        "reward": {"scaling": 1.0}
    }
    
    # Mock Data Handler (returns constant data to simulate environment steps)
    class MockHandler:
        def __init__(self):
            self.current_step = 0
            self._len = 100
        def reset(self):
            self.current_step = 0
        def step(self):
            self.current_step += 1
            if self.current_step > 100: return None
            # Return dummy dict with all required keys
            data = {
                'mid_price': 100.0,
                'bid_price_1': 99.5, 'bid_vol_1': 1.0,
                'ask_price_1': 100.5, 'ask_vol_1': 1.0,
                'spread_1': 1.0, 'log_ret': 0.0,
                'ofi_1': 0.0, 'ofi_2': 0.0, 'ofi_3': 0.0, 'ofi_4': 0.0, 'ofi_5': 0.0,
                'timestamp': '2023-01-01 00:00:00'
            }
            # Fill LOB keys
            for i in range(1, 6):
                data[f'bid_price_{i}'] = 99.0
                data[f'bid_vol_{i}'] = 1.0
                data[f'ask_price_{i}'] = 101.0
                data[f'ask_vol_{i}'] = 1.0
                # Normalized keys expected by _build_frame
                data[f'n_bid_price_{i}'] = 0.0
                data[f'n_bid_vol_{i}'] = 0.0
                data[f'n_ask_price_{i}'] = 0.0
                data[f'n_ask_vol_{i}'] = 0.0
            return data
            
        def get_lookahead_price(self, h): return 100.0
        def get_lookahead_volatility(self, h): return 0.0

    env = DeepScalperEnv(config, data_handler=MockHandler())
    env.reset()
    
    # Force step count to 50/100 -> remaining time should be 0.5
    env.current_step = 50
    env.total_episode_steps = 100
    
    # 1. Place a BUY order
    # Action: Price=0 (At touch), Qty=8 (Max position) -> Buy
    # This sets pending_order for T+1
    obs, _, _, _, _ = env.step(np.array([0, 8])) 
    
    # At this point, pending_order is set.
    # In the NEXT step, it will fill.
    
    # Manually ensure it fills
    env.current_best_ask = 100.5 # Buy fits here
    
    # 2. Step again to trigger execution
    # This calls _update_state -> then executes -> then OVERWRITES private_window[-1]
    obs, _, _, _, _ = env.step(np.array([4, 4])) # Hold action
    
    # Check the private window's last frame
    # Index 0=Pos, 1=Bal, 2=Time
    last_private = env.private_window[-1]
    actual_time = last_private[2]
    expected_time = 1.0 - (52 / 100) # approx 0.48
    
    print(f"Step: {env.current_step} / {env.total_episode_steps}")
    print(f"Private Window Last Frame: {last_private}")
    print(f"Remaining Time Value: {actual_time}")
    
    if np.isclose(actual_time, 1.0):
        print("❌ BUG CONFIRMED: Remaining time reset to 1.0 after fill!")
    elif np.isclose(actual_time, expected_time, atol=0.02):
        print("✅ BUG FIXED: Remaining time is correct.")
    else:
        print(f"❓ UNEXPECTED: Time {actual_time}, Expected {expected_time}")

if __name__ == "__main__":
    test_remaining_time_bug()
