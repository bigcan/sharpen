import os
import shutil
import yaml
import subprocess
from finrl_pro_ds.configs.schema import ConfigLoader
from finrl_pro_ds.mlops.logger import MLOpsLogger

logger = MLOpsLogger()

def deploy_to_remote(target: str, config_path: str, dry_run: bool = False):
    """
    Orchestrate deployment to GPUHub or RunPod.
    
    1. Validate Config for Remote Execution.
    2. Package Code + Config.
    3. Generate target-specific startup script.
    4. Push/Execute.
    """
    logger.log_event("deployment.start", context={"target": target, "dry_run": dry_run})
    
    # 1. Load & Validate
    try:
        config = ConfigLoader.load_yaml(config_path)
    except Exception as e:
        logger.error(f"Config Validation Failed: {e}")
        return

    # 2. Package
    dist_dir = "dist/deploy_package"
    if os.path.exists(dist_dir):
        # We need to be careful. rmtree sometimes fails on windows due to file locks.
        # But we really want a clean state.
        try:
            shutil.rmtree(dist_dir)
        except Exception as e:
            logger.warning(f"Could not remove {dist_dir} fully: {e}")
            
    os.makedirs(dist_dir, exist_ok=True)
    
    # Copy essential source
    # Ignore __pycache__ and .git
    ignore_func = shutil.ignore_patterns("__pycache__", ".git", "*.pyc")
    
    shutil.copytree("finrl_pro_ds", f"{dist_dir}/finrl_pro_ds", dirs_exist_ok=True, ignore=ignore_func)
    shutil.copytree("scripts", f"{dist_dir}/scripts", dirs_exist_ok=True, ignore=ignore_func)
    shutil.copytree("configs", f"{dist_dir}/configs", dirs_exist_ok=True, ignore=ignore_func)
    
    # Copy requirements if exists
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", dist_dir)
        
    logger.info(f"Code packaged to {dist_dir}")
    
    # 3. Target Specifics
    if target == "gpuhub":
        _prepare_gpuhub(dist_dir, config, dry_run)
    elif target == "runpod":
        _prepare_runpod(dist_dir, config, dry_run)
        
def _prepare_gpuhub(dist_dir, config, dry_run):
    """Prepare GPUHub specific start.sh and sync logic"""
    startup_script = f"""#!/bin/bash
# GPUHub Startup for DeepScalper {config.wandb.project}
echo "Starting DeepScalper Training..."
pip install -r requirements.txt
python -m finrl_pro_ds.cli train --config configs/deepscalper_unified.yaml
echo "Training Complete."
"""
    with open(f"{dist_dir}/start.sh", "w") as f:
        f.write(startup_script)
        
    msg = f"GPUHub Package Ready at {dist_dir}. To deploy: 'gpuhub push {dist_dir}' (Mock Command)"
    print(msg)
    if not dry_run:
        # Popen(['gpuhub', 'push', ...]) 
        pass

def _prepare_runpod(dist_dir, config, dry_run):
    """Prepare RunPod Dockerfile or wrapper"""
    dockerfile = f"""
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "-m", "finrl_pro_ds.cli", "train", "--config", "configs/deepscalper_unified.yaml"]
"""
    with open(f"{dist_dir}/Dockerfile", "w") as f:
        f.write(dockerfile)
        
    msg = f"RunPod Dockerfile Ready at {dist_dir}. To build & push: 'docker build -t deepscalper:latest {dist_dir}'"
    print(msg)
