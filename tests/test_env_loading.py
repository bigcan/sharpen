
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
    print("SUCCESS: DeepScalperEnv imported successfully.")
except Exception as e:
    print(f"FAILURE: Could not import DeepScalperEnv. Error: {e}")
    sys.exit(1)

# Mock config
config = {
    "margin_requirement": 1.0,
    "initial_balance": 100000.0,
    "window_size": 50,
    "reward": {"scaling": 1e-4},
    "tech_indicator_list": [], # Dummy
    "dims": {"micro": 44, "macro": 10, "action": 3} # Dummy dims
}

try:
    env = DeepScalperEnv(config=config)
    print("SUCCESS: DeepScalperEnv initialized successfully.")
    
    # Test Margin Check Logic (Unit Test style)
    print("\n--- Testing Margin Logic (Symmetric) ---")
    
    # reset env internal state for testing
    env.balance = 100000.0
    env.position = 0.0
    env.margin_requirement = 1.0 # 100% Cash
    
    # Test 1: Buy 1 unit at 1000. Requires 1000 cash.
    # Should be allowed (1000 <= 100000)
    allowed = env._check_margin(0.0, 1.0, 1000.0, 1)
    print(f"Test 1 (Buy 1 @ 1000, Bal 100k, Req 1k): {allowed} (Expected: True)")
    
    # Test 2: Sell 1 unit at 1000. Requires 1000 margin (since margin_req=1.0).
    # Should be allowed (1000 <= 100000)
    allowed = env._check_margin(0.0, 1.0, 1000.0, 2)
    print(f"Test 2 (Sell 1 @ 1000, Bal 100k, Req 1k): {allowed} (Expected: True)")
    
    # Test 3: Buy huge (1000 units @ 1000 = 1M). Bal 100k.
    # Should fail.
    allowed = env._check_margin(0.0, 1000.0, 1000.0, 1)
    print(f"Test 3 (Buy 1M, Bal 100k): {allowed} (Expected: False)")
    
    # Test 4: Sell huge (1000 units @ 1000 = 1M). Bal 100k.
    # Should fail (Symmetric!)
    allowed = env._check_margin(0.0, 1000.0, 1000.0, 2)
    print(f"Test 4 (Sell 1M, Bal 100k): {allowed} (Expected: False)")
    
    # Test 5: Leverage 2x (margin_req = 0.5)
    env.margin_requirement = 0.5
    # Buy 200k worth (200 @ 1000). Bal 100k. Req = 200k * 0.5 = 100k.
    # Should be allowed.
    allowed = env._check_margin(0.0, 200.0, 1000.0, 1)
    print(f"Test 5 (2x Lev Buy, 200k pos, 100k bal): {allowed} (Expected: True)")
    
    # Sell 200k worth.
    allowed = env._check_margin(0.0, 200.0, 1000.0, 2)
    print(f"Test 6 (2x Lev Sell, 200k pos, 100k bal): {allowed} (Expected: True)")

except Exception as e:
    print(f"FAILURE: Runtime error during test: {e}")
    sys.exit(1)
