
import yaml
import sys
import os

# Mock the sanitize function from train_deepscalper.py
def sanitize_config(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            sanitize_config(v)
        elif k in ["learning_rate", "gamma", "entropy_coef", "gae_lambda", "clip_epsilon", "max_grad_norm"]:
            try:
                cfg[k] = float(v)
            except:
                pass

def debug_load(config_path):
    print(f"Loading {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nSelect Raw Keys:", list(config.keys()))
    
    if "training" in config:
        print("\nRaw Training Config:", config["training"])
    else:
        print("\n[ERROR] 'training' key missing from root config!")
        
    # Simulate train_deepscalper.py logic
    training_config = config.get("training", {})
    training_config["agents"] = config.get("agents", {})
    
    # Sanitize "agents" block (as done in script)
    agents_config = config.get("agents", {})
    sanitize_config(agents_config)
    
    print("\nProcessed Training Config keys:", list(training_config.keys()))
    print(f"torch_compile: {training_config.get('torch_compile', 'Not Set')}")
    print(f"total_timesteps: {training_config.get('total_timesteps', 'Not Set')}")

if __name__ == "__main__":
    # Point to the production config
    path = "configs/deepscalper_prod_rtx5090.yaml"
    debug_load(path)
