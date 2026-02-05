def merge_configs(base, overrides):
    """Deep merge dictionaries."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base

base = {
    "agents": {
        "bdq": {
            "batch_size": 256,
            "learning_rate": 0.0001
        }
    },
    "env": {"reward": {"hindsight_horizon": 180}}
}

overrides = {
    "agents": {
        "bdq": {
            "batch_size": 64,
            "learning_rate": 0.0003
        }
    },
    "env": {"reward": {"hindsight_horizon": 120}}
}

merged = merge_configs(base, overrides)
print(merged)

assert merged["agents"]["bdq"]["batch_size"] == 64
assert merged["agents"]["bdq"]["learning_rate"] == 0.0003
assert merged["env"]["reward"]["hindsight_horizon"] == 120
print("Merge Successful")
