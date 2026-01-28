import argparse
import sys
import os
import yaml
import numpy as np

# Mock Config Loader or minimal config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Starting Env Isolation Test...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()

    print("Importing Gymnasium...")
    import gymnasium as gym
    print(f"Gymnasium Version: {gym.__version__}")

    print("Importing DeepScalperEnv...")
    try:
        from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
        print("DeepScalperEnv Imported.")
    except Exception as e:
        print(f"DeepScalperEnv Import Failed: {e}")
        return

    print("Loading Config...")
    config = load_config(args.config)
    print("Config Loaded.")

    print("Initializing Environment...")
    try:
        env = DeepScalperEnv(config)
        print("Environment Initialized.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Env Init Failed: {e}")
        return

    print("Resetting Environment...")
    try:
        obs, info = env.reset()
        print("Environment Reset Successful.")
        print(f"Obs Keys: {obs.keys() if isinstance(obs, dict) else 'Not Dict'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Env Reset Failed: {e}")
        return

    print("Stepping Environment (5 steps)...")
    try:
        for i in range(5):
            # Random Action: [Direction(3), Price(5), Volume(5)]
            action = np.array([1, 2, 2]) 
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Done={terminated or truncated}")
            if terminated or truncated:
                print("Episode Done. Resetting...")
                env.reset()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Env Step Failed: {e}")
        return

    print("Env Debug SUCCESS.")

if __name__ == "__main__":
    main()
