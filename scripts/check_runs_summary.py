import json
import os

run_ids = ["f1s6uv1q", "tbxmglp4"]
keys = [
    "hpo/n_trials", "hpo/best_trial", "hpo/best_profit_factor", "hpo/best_params",
    "train/status", "train/final_step",
    "backtest_val/status", "backtest_val/error",
    "backtest_test/status", "backtest_test/error",
    "backtest_val/sharpe_ratio", "backtest_test/sharpe_ratio",
    "train/checkpoint_path",
]

for rid in run_ids:
    path = os.path.join("results", f"run_data_{rid}.json")
    if not os.path.exists(path):
        print(f"--- {rid}: FILE NOT FOUND ---")
        continue
    with open(path) as f:
        data = json.load(f)
    s = data.get("summary", {})
    print(f"=== {data.get('name', rid)} ({rid}) ===")
    print(f"  State: {data.get('state')}")
    for k in keys:
        val = s.get(k, "N/A")
        if isinstance(val, str) and len(val) > 80:
            val = val[:80] + "..."
        print(f"  {k}: {val}")
    print()
