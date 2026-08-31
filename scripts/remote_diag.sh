#!/bin/bash
cd /workspace/DeepScalper
export PATH=/root/miniconda3/bin:$PATH
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2 | tr -d '[:space:]')

echo "=== B13 log (last 10 lines) ==="
tail -10 logs/b13_inventory_penalty.log 2>/dev/null
echo ""
echo "=== B14 log (last 10 lines) ==="
tail -10 logs/b14_crra_utility.log 2>/dev/null
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo ""
echo "=== Python processes ==="
ps aux | grep python | grep -v grep
echo ""
echo "=== Quick env test ==="
python -c "
from sharpen.envs.deep_scalper_env import DeepScalperEnv
import yaml
cfg = yaml.safe_load(open('configs/phase_b13_inventory_penalty_5min.yaml'))
env = DeepScalperEnv(cfg['env'])
print(f'Env OK: inv_pen={env.inventory_penalty_bps}, crra={env.crra_gamma}')
" 2>&1
