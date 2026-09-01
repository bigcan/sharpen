#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
echo "=== NVIDIA Driver ==="
nvidia-smi | head -5
echo ""
echo "=== CUDA Version ==="
nvcc --version 2>/dev/null || echo "nvcc not found"
echo ""
echo "=== PyTorch CUDA ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    # Quick tensor test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x
    print(f'Tensor test OK: {y.shape}')
else:
    print('CUDA NOT AVAILABLE')
" 2>&1
echo ""
echo "=== torch.compile test ==="
python -c "
import torch
@torch.compile
def f(x): return x * 2
x = torch.randn(10, device='cuda')
result = f(x)
print(f'torch.compile test OK: {result.shape}')
" 2>&1
echo ""
echo "=== SyncVectorEnv test ==="
cd /workspace/DeepScalper
python -c "
import yaml
from sharpen.training.deepscalper_trainer import DeepScalperTrainer
cfg = yaml.safe_load(open('configs/phase_b13_inventory_penalty_5min.yaml'))
# Try creating trainer (this is where segfault happens)
try:
    trainer = DeepScalperTrainer(cfg, run_name='segfault_test')
    print('Trainer created OK')
except Exception as e:
    print(f'Trainer creation failed: {e}')
" 2>&1
