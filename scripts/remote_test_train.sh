#!/bin/bash
cd /workspace/DeepScalper
export PATH=/root/miniconda3/bin:$PATH
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2 | tr -d '[:space:]')

# Quick test: disable torch_compile via env var, short run
export DEEPSCALPER_NO_COMPILE=1
python -u -c "
import yaml, torch

cfg = yaml.safe_load(open('configs/phase_b13_inventory_penalty_5min.yaml'))

# Patch for quick smoke test
cfg['training']['torch_compile'] = False
cfg['training']['total_timesteps'] = 2000
cfg['wandb']['mode'] = 'disabled'

from scripts.run_full_pipeline import run_training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
torch.set_float32_matmul_precision('high')

try:
    run_training(cfg, run_name='smoke_test_b13', device=device)
    print('TRAINING SMOKE TEST: PASSED')
except Exception as e:
    print(f'TRAINING SMOKE TEST: FAILED - {e}')
    import traceback
    traceback.print_exc()
" 2>&1 | tail -40
