#!/bin/bash
cd /workspace/DeepScalper
export PATH=/root/miniconda3/bin:$PATH
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2 | tr -d '[:space:]')

python -u -c "
import yaml, torch, sys, os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print('Step 1: Load config')
cfg = yaml.safe_load(open('configs/phase_b13_inventory_penalty_5min.yaml'))
cfg['wandb']['mode'] = 'disabled'
print('OK')

print('Step 2: Create single env')
from scripts.run_full_pipeline import make_env
env = make_env(cfg)
obs, info = env.reset()
print(f'OK - obs shapes: micro={obs[\"micro\"].shape}, macro={obs[\"macro\"].shape}')
env.close()

print('Step 3: Create vector env (SyncVectorEnv, 24 envs)')
from scripts.run_full_pipeline import create_vector_env
venv = create_vector_env(cfg, num_envs=24, use_sync=True)
print(f'OK - vector env created')
venv.close()

print('Step 4: Create trainer')
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
venv2 = create_vector_env(cfg, num_envs=24, use_sync=True)
device = 'cuda'
trainer = DeepScalperTrainer(venv2, cfg, device=device, run_name='debug_test')
print(f'OK - trainer created')

print('Step 5: First training step')
trainer.train_step()
print(f'OK - first train step')
venv2.close()

print('ALL STEPS PASSED')
" 2>&1
