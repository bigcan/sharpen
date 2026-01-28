import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CWD: {os.getcwd()}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Testing tensor creation...")
        x = torch.randn(1000, 1000, device="cuda")
        print("Testing matmul...")
        y = torch.matmul(x, x)
        print("Matmul successful")
        print(f"Result mean: {y.mean().item()}")
    except Exception as e:
        print(f"CUDA Operation Failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("CUDA not available")

print("Exiting verify_cuda.py")
