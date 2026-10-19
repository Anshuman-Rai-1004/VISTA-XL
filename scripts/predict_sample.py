import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vista_xl import VISTAXL

DATA_DIR = ROOT / "Dataset" / "validation_data"
CHECKPOINT = ROOT / "checkpoints" / "vista_xl_best.pth"
OUTPUT_DIR = ROOT / "results" / "sample_prediction"
IMAGE_SIZE = 224

# Pick the first validation sample with both files available.
samples = sorted(
    p.parent for p in DATA_DIR.rglob("image.png")
    if (p.parent / "mask.png").is_file()
)

if not samples:
    raise FileNotFoundError(f"No image/mask pairs found in {DATA_DIR}")

sample_dir = samples[0]
image_path = sample_dir / "image.png"
mask_path = sample_dir / "mask.png"

print("Sample:", sample_dir.name)
print("Loading VISTA-XL...")
model = VISTAXL(pretrained=False)

checkpoint = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

# RGB input, resized to 224x224, normalized to [0, 1].
original = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

resized = original.resize(
    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
)
image_array = np.asarray(resized, dtype=np.float32) / 255.0
input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

with torch.inference_mode():
    logits = model(input_tensor)
    probability = torch.sigmoid(logits)[0, 0].numpy()

prediction = probability >= 0.5
ground_truth = (
    np.asarray(
        mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    ) >= 128
)

intersection = np.logical_and(prediction, ground_truth).sum()
dice = (2 * intersection) / (
    prediction.sum() + ground_truth.sum() + 1e-8
)
union = np.logical_or(prediction, ground_truth).sum()
iou = intersection / (union + 1e-8)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"{sample_dir.name}_prediction.png"

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(resized)
axes[0].set_title("Original Image")
axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth")
axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("VISTA-XL Prediction")
axes[3].imshow(resized)
axes[3].imshow(prediction, cmap="autumn", alpha=0.45)
axes[3].set_title("Prediction Overlay")

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"{sample_dir.name} | Dice: {dice:.4f} | IoU: {iou:.4f}"
)
fig.tight_layout()
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Dice: {dice:.4f}")
print(f"IoU: {iou:.4f}")
print("Saved visualization:", output_path)
print("Real-image inference completed.")
