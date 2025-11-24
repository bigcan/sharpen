"""Daily retraining automation for FinRL Pro strategies.

This script automates the "daily loop":
1. Identify the active/latest strategy config.
2. Update the dataset (simulated fetch of new bars).
3. Trigger a retraining run (using the Trainer).
4. Register the new fingerprint as the current production candidate.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml

from finrl_pro.training.run_experiment import main as run_experiment_main

logger = logging.getLogger(__name__)


def _get_latest_config(configs_dir: Path) -> Optional[Path]:
    """Find the most recently modified experiment config."""
    yaml_files = list(configs_dir.glob("*.yaml"))
    if not yaml_files:
        return None
    return max(yaml_files, key=lambda p: p.stat().st_mtime)


def retrain(
    config_path: str,
    force_new_seed: bool = True,
    output_dir: str = "reports/retrain",
) -> None:
    """Execute retraining workflow."""
    cfg = Path(config_path)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"--- Starting Retraining for {cfg.name} ---")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")

    # 1. In a real scenario, we would fetch new data here.
    # For now, we assume the dataset URI in the config points to a source
    # that gets updated (e.g., 'snapshot://latest' or a CSV file).
    
    # 2. Modify config for new run (e.g. new seed)
    # We create a temporary runtime config
    with open(cfg, "r") as f:
        config_data = yaml.safe_load(f)
    
    if force_new_seed:
        import random
        new_seed = random.randint(1000, 9999)
        print(f"[+] Rotating seed to {new_seed}")
        config_data["training"]["seed"] = new_seed
        
    # Update output artifacts tag if needed, or rely on fingerprinting
    # ...
    
    # Ensure training block exists
    if "training" not in config_data:
        config_data["training"] = {}

    # Enable model registration for the retrained model
    config_data["training"]["register_model"] = True

    # Determine model name if not present
    if "model_name" not in config_data["training"]:
        base_name = cfg.stem
        # Sanitize name for MLflow (alphanumeric, underscores, dashes, periods, spaces)
        safe_name = "".join(c if c.isalnum() or c in "_-." else "_" for c in base_name)
        config_data["training"]["model_name"] = safe_name
        print(f"[+] Configured for Model Registry: {safe_name}")

    tmp_config = Path(f"tmp_retrain_{cfg.name}")
    with open(tmp_config, "w") as f:
        yaml.dump(config_data, f)

    # 3. Run Experiment
    try:
        # Delegate to the main runner
        # We pass the temp config path
        run_experiment_main(["--config", str(tmp_config)])
        print("[+] Retraining completed successfully.")
    except Exception as e:
        print(f"[-] Retraining failed: {e}")
        raise
    finally:
        if tmp_config.exists():
            tmp_config.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Retraining Bot")
    parser.add_argument("--config", help="Path to specific strategy config")
    parser.add_argument("--auto-latest", action="store_true", help="Pick latest config from defaults")
    args = parser.parse_args()

    target_config = args.config
    if args.auto_latest:
        base_dir = Path("finrl_pro/configs/experiments")
        latest = _get_latest_config(base_dir)
        if latest:
            target_config = str(latest)
            print(f"[+] Auto-selected latest config: {target_config}")
    
    if not target_config:
        print("[-] No config specified. Use --config or --auto-latest.")
        return

    retrain(target_config)


if __name__ == "__main__":
    main()
