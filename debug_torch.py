import sys
import torch
print(f"Sys.path: {sys.path}")
print(f"Torch path: {torch.__path__}")
try:
    import torch.utils
    print("torch.utils imported")
except ImportError as e:
    print(f"Failed to import torch.utils: {e}")
