import os
import platform
import sys
from pathlib import Path

print("=" * 45)
print("VISTA-XL Environment Check")
print("=" * 45)

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("GPU training: Use Kaggle GPU for this project.")
except ImportError:
    print("PyTorch: NOT INSTALLED")

root = Path(__file__).resolve().parent.parent
dataset = root / "Dataset"

for split in ("training_data", "validation_data", "test_data"):
    split_path = dataset / split
    if split_path.exists():
        count = sum(
            1 for p in split_path.iterdir()
            if p.is_dir() and (p / "image.png").exists()
            and (p / "mask.png").exists()
        )
        print(f"{split}: FOUND ({count} image-mask pairs)")
    else:
        print(f"{split}: NOT FOUND")
