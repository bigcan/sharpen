#!/usr/bin/env python3
"""
Generate the DeepScalper Control Plane notebook from config files.

This script ensures the notebook is always up-to-date with the current
MLOps pipeline configuration. Run it before major releases or after
significant config changes.

Usage:
    python scripts/generate_control_plane.py
"""

import json
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "deepscalper_production.yaml"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "DeepScalper_Control_Plane.ipynb"


def load_config():
    """Load the unified config file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_notebook(config: dict) -> dict:
    """Build the notebook structure from config."""
    
    training = config.get("training", {})
    batch_size = training.get("batch_size", "N/A")
    update_interval = training.get("update_interval", "N/A")
    torch_compile = training.get("torch_compile", False)
    use_amp = training.get("use_amp", False)
    total_timesteps = training.get("total_timesteps", "N/A")
    
    # Determine profile name based on settings
    if torch_compile and use_amp and batch_size >= 8192:
        profile = "Synapse V9.5 (Blackwell RTX 5090)"
    else:
        profile = "V1 Baseline"
    
    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    cells = [
        # Header
        {
            "cell_type": "markdown",
            "id": "header",
            "metadata": {},
            "source": [
                "# 🚀 DeepScalper Mission Control Plane\n",
                f"**Project:** FinRL-Pro_DS | **Profile:** {profile}\n",
                f"*Auto-generated on {generated_ts}*\n",
                "\n",
                "### 📋 Workflow Stages:\n",
                "1. **Environment Setup**: Verify paths & load `.env`.\n",
                "2. **Configuration Review**: Inspect current settings.\n",
                "3. **Mission Launch**: Deploy to GPUHub.\n",
                "4. **Monitoring**: Check remote processes.\n",
                "5. **Analysis**: Log findings.\n",
                "\n",
                "### 🏎️ Current Configuration:\n",
                "| Parameter | Value |\n",
                "|---|---|\n",
                f"| `batch_size` | **{batch_size}** |\n",
                f"| `update_interval` | **{update_interval}** |\n",
                f"| `torch_compile` | **{torch_compile}** |\n",
                f"| `use_amp` | **{use_amp}** |\n",
                f"| `total_timesteps` | **{total_timesteps:,}** |"
            ]
        },
        # Environment Setup
        {
            "cell_type": "markdown",
            "id": "env_header",
            "metadata": {},
            "source": ["## 1. Environment Setup"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "env_setup",
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, yaml\n",
                "from datetime import datetime\n",
                "from dotenv import load_dotenv\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath('../')\n",
                "if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)\n",
                "os.chdir(PROJECT_ROOT)\n",
                "load_dotenv()\n",
                "\n",
                "print(f'✅ CWD: {os.getcwd()}')\n",
                "print(f'✅ Host: {os.getenv(\"GPUHUB_HOST\", \"NOT SET\")}')"
            ]
        },
        # Config Review
        {
            "cell_type": "markdown",
            "id": "config_header",
            "metadata": {},
            "source": ["## 2. Configuration Review"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "config_review",
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('configs/deepscalper_unified.yaml') as f:\n",
                "    cfg = yaml.safe_load(f)\n",
                "t = cfg.get('training', {})\n",
                "print(f'Batch: {t.get(\"batch_size\")} | Update: {t.get(\"update_interval\")}')\n",
                "print(f'Compile: {t.get(\"torch_compile\")} | AMP: {t.get(\"use_amp\")}')"
            ]
        },
        # Mission Launch
        {
            "cell_type": "markdown",
            "id": "launch_header",
            "metadata": {},
            "source": [
                "## 3. Mission Launch\n",
                "> Auth via `.env` (no `--host`/`--key` needed)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "launch_params",
            "metadata": {},
            "outputs": [],
            "source": [
                "MISSION = 'pipeline'  # 'train', 'hpo', 'pipeline'\n",
                "EXTRA = ''  # e.g. '--trials 20 --steps 2000000' for HPO\n",
                "print(f'🚀 Mission: {MISSION}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "launch_exec",
            "metadata": {},
            "outputs": [],
            "source": [
                "SCRIPTS = {'hpo': 'scripts/tune_deepscalper.py', 'train': 'scripts/train_deepscalper_v3.py', 'pipeline': 'scripts/run_full_pipeline.py'}\n",
                "cmd = f'python scripts/deploy_bare_metal.py --script {SCRIPTS[MISSION]} --config configs/deepscalper_production.yaml --upload_data --data_file btc_lob_jan2023.parquet'\n",
                "if EXTRA: cmd += f' --extra_args \"{EXTRA}\"'\n",
                "print(cmd)\n",
                "# !{cmd}  # Uncomment to run"
            ]
        },
        # Monitoring
        {
            "cell_type": "markdown",
            "id": "monitor_header",
            "metadata": {},
            "source": ["## 4. Monitoring"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "monitor_ps",
            "metadata": {},
            "outputs": [],
            "source": ["!python scripts/remote_cmd.py 'pgrep -a python'"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "monitor_logs",
            "metadata": {},
            "outputs": [],
            "source": ["!python scripts/remote_cmd.py 'tail -n 30 /workspace/DeepScalper/run.log'"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "monitor_gpu",
            "metadata": {},
            "outputs": [],
            "source": ["!python scripts/remote_cmd.py 'nvidia-smi'"]
        },
        # Post-Flight
        {
            "cell_type": "markdown",
            "id": "post_header",
            "metadata": {},
            "source": ["## 5. Post-Flight Logging"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "post_log",
            "metadata": {},
            "outputs": [],
            "source": [
                f"print(f'''## {{datetime.now().strftime('%Y-%m-%d %H:%M')}} - {{RUN_NAME}}\n",
                f"**Config:** Batch {batch_size} | AMP {use_amp} | Compile {torch_compile}\n",
                "**Outcome:** [PASS/FAIL]\n",
                "''')"
            ]
        }
    ]
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.5"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def main():
    print(f"📖 Loading config from {CONFIG_PATH}")
    config = load_config()
    
    print("🔧 Building notebook...")
    notebook = build_notebook(config)
    
    print(f"💾 Writing to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Control Plane notebook regenerated!")


if __name__ == "__main__":
    main()
