"""
Verification script for FinRL Pro Agents.
Run this script to verify that dependencies are installed and agents are working.
Usage: python verify_agents.py
"""

import sys
import platform
import importlib.util

def check_dependency(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"[-] Missing dependency: {name}")
        return False
    print(f"[+] Found dependency: {name}")
    return True

def main():
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # 1. Check Dependencies
    print("\n--- Checking Dependencies ---")
    deps = ["numpy", "torch", "pandas", "gym"]
    missing = []
    for dep in deps:
        if not check_dependency(dep):
            missing.append(dep)
    
    if missing:
        print(f"\n[!] CRITICAL: Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        # Try to install them? No, better to ask user.
        return

    # 2. Check Agent Imports
    print("\n--- Checking Agent Imports ---")
    try:
        import torch
        import numpy as np
        from finrl_pro.agents.ppo import PPOAgent
        from finrl_pro.agents.sac import SACAgent
        from finrl_pro.agents.ddpg import DDPGAgent
        from finrl_pro.agents.td3 import TD3Agent
        from finrl_pro.agents.cql import CQLAgent
        print("[+] All agents imported successfully.")
    except ImportError as e:
        print(f"[-] Import Error: {e}")
        print("Make sure you are running this from the root directory (c:\\FinRL\\FinRL Pro V2.0\\FinRL_Podracer)")
        return

    # 3. Run Basic Tests
    print("\n--- Running Basic Agent Tests ---")
    state_dim = 10
    action_dim = 2
    batch_size = 4
    
    # PPO
    try:
        print("Testing PPO...", end=" ")
        ppo = PPOAgent(state_dim, action_dim, batch_size=batch_size)
        s = np.random.random(state_dim)
        a, _ = ppo.select_action(s)
        assert a.shape == (action_dim,)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # SAC
    try:
        print("Testing SAC...", end=" ")
        sac = SACAgent(state_dim, action_dim, batch_size=batch_size)
        s = np.random.random(state_dim)
        a = sac.select_action(s)
        assert a.shape == (action_dim,)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # DDPG
    try:
        print("Testing DDPG...", end=" ")
        ddpg = DDPGAgent(state_dim, action_dim, batch_size=batch_size)
        s = np.random.random(state_dim)
        a = ddpg.select_action(s)
        assert a.shape == (action_dim,)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # TD3
    try:
        print("Testing TD3...", end=" ")
        td3 = TD3Agent(state_dim, action_dim, batch_size=batch_size)
        s = np.random.random(state_dim)
        a = td3.select_action(s)
        assert a.shape == (action_dim,)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # CQL
    try:
        print("Testing CQL...", end=" ")
        cql = CQLAgent(state_dim, action_dim, batch_size=batch_size)
        s = np.random.random(state_dim)
        a = cql.select_action(s)
        assert a.shape == (action_dim,)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[+] Verification Complete.")

if __name__ == "__main__":
    main()
