import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vista_xl import VISTAXL

checkpoint_path = ROOT / "checkpoints" / "vista_xl_best.pth"

print("Device: CPU")
print("Creating VISTA-XL model...")

model = VISTAXL(pretrained=False)

print("Loading checkpoint...")
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

print("Checkpoint loaded successfully!")
print("Epoch:", checkpoint.get("epoch"))
print("Validation Dice:", checkpoint.get("val_dice"))
print("Validation IoU:", checkpoint.get("val_iou"))
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

print("Running dummy inference...")
x = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    y = model(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(y.shape))
print("Output finite:", bool(torch.isfinite(y).all()))
print("VISTA-XL loading and inference test PASSED.")
