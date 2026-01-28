import gymnasium as gym
import multiprocessing as mp
import time

def make_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":
    print("Starting Gymnasium AsyncVectorEnv Verification...")
    try:
        # Test 1: Sync
        print("Test 1: SyncVectorEnv")
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(2)])
        envs.reset()
        envs.close()
        print("SyncVectorEnv Success.")
        
        # Test 2: Async Spawn
        print("Test 2: AsyncVectorEnv (spawn)")
        envs = gym.vector.AsyncVectorEnv([make_env for _ in range(2)], context="spawn")
        print("Created Async Envs.")
        obs, info = envs.reset()
        print(f"Reset Done. Obs shape: {obs.shape}")
        envs.close()
        print("AsyncVectorEnv Success.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
