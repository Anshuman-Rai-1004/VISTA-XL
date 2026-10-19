from pathlib import Path
import shutil
import zipfile
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path.cwd()
RESULTS = ROOT / "results"
SOURCE = RESULTS / "full_validation"
PACKAGE = RESULTS / "VISTA_XL_Final_Package"
CHECKPOINT = ROOT / "checkpoints" / "vista_xl_best.pth"

PACKAGE.mkdir(parents=True, exist_ok=True)
print("Preparing VISTA-XL final results package...")

# Preserve and include existing evaluation results.
for filename in [
    "per_image_metrics.csv",
    "summary_metrics.csv"
]:
    source = SOURCE / filename
    if source.exists():
        shutil.copy2(source, PACKAGE / filename)
        print(f"Included: {filename}")
    else:
        print(f"WARNING: Missing {source}")

# Read actual local evaluation results.
per_image_path = SOURCE / "per_image_metrics.csv"
summary_path = SOURCE / "summary_metrics.csv"

if not per_image_path.exists() or not summary_path.exists():
    raise FileNotFoundError(
        "Run scripts/evaluate_model.py successfully first."
    )

per_image = pd.read_csv(per_image_path)
summary = pd.read_csv(summary_path)

# Checkpoint metadata.
checkpoint_info = {}
if CHECKPOINT.exists():
    try:
        ckpt = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True
        )
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location="cpu")

    checkpoint_info = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "epoch": ckpt.get("epoch"),
        "stored_validation_dice": ckpt.get("val_dice"),
        "stored_validation_iou": ckpt.get("val_iou"),
        "config": ckpt.get("config", {}),
        "seed": ckpt.get("seed")
    }
else:
    print("WARNING: VISTA-XL checkpoint not found.")

with open(PACKAGE / "checkpoint_metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(checkpoint_info, f, indent=2, default=str)

# Historical clean benchmark values from the prior experiments.
# These must be rechecked against identical preprocessing before publication.
comparison = pd.DataFrame([
    {
        "model": "VISTA-XL",
        "dice": 0.8723,
        "iou": 0.7916,
        "parameters": 35233619,
        "source": "Current local evaluation; preprocessing needs verification"
    },
    {
        "model": "Swin baseline",
        "dice": 0.8716,
        "iou": 0.7944,
        "parameters": 39558779,
        "source": "Prior clean benchmark; verify identical protocol"
    },
    {
        "model": "U-Net",
        "dice": 0.8250,
        "iou": 0.7289,
        "parameters": 7765985,
        "source": "Prior clean benchmark; verify identical protocol"
    }
])
comparison.to_csv(PACKAGE / "model_comparison_provisional.csv",
                  index=False)

# Comparison chart.
plt.figure(figsize=(8, 5))
plt.bar(comparison["model"], comparison["dice"])
plt.ylabel("Mean Dice")
plt.title("VISTA-XL Model Comparison (Provisional)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(PACKAGE / "dice_comparison_provisional.png", dpi=180)
plt.close()

# Local metric visualization.
metric_columns = [
    c for c in ["dice", "iou", "precision", "recall",
                "specificity", "pixel_accuracy"]
    if c in per_image.columns
]
if metric_columns:
    means = per_image[metric_columns].mean()
    plt.figure(figsize=(9, 5))
    plt.bar(means.index, means.values)
    plt.ylabel("Mean score")
    plt.title("VISTA-XL Local Validation Metrics")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PACKAGE / "validation_metrics.png", dpi=180)
    plt.close()

# Transparent report: do not label results publication-ready yet.
report = f"""# VISTA-XL Results Package

## Current local evaluation
- Images evaluated: {len(per_image)}
- Mean Dice: {per_image['dice'].mean():.4f}
- Mean IoU: {per_image['iou'].mean():.4f}
- Mean precision: {per_image['precision'].mean():.4f}
- Mean recall: {per_image['recall'].mean():.4f}

## Checkpoint
- Epoch: {checkpoint_info.get('epoch', 'Not available')}
- Stored validation Dice: {checkpoint_info.get('stored_validation_dice', 'Not available')}
- Stored validation IoU: {checkpoint_info.get('stored_validation_iou', 'Not available')}

## Included files
- per_image_metrics.csv: per-image local evaluation
- summary_metrics.csv: local metric summary
- checkpoint_metadata.json: saved checkpoint metadata
- model_comparison_provisional.csv: prior comparison values
- dice_comparison_provisional.png: provisional comparison chart
- validation_metrics.png: local validation metric chart

## Important limitations
1. The original VISTA-XL training preprocessing has not yet been independently verified.
2. The local evaluation script uses ImageNet mean/std normalization.
3. The checkpoint configuration does not record the image normalization pipeline.
4. Swin baseline and U-Net comparison values are from prior experiment results and must be rechecked under an identical evaluation protocol.
5. Do not present the provisional comparison as a definitive superiority claim.
6. Validation metrics are not independent test-set metrics.

## Next steps
1. Verify original training preprocessing.
2. Re-evaluate all models using the same clean split and transformations.
3. Check threshold, mask resizing and metric implementation.
4. Finalize results and limitations before writing the research paper.
"""
(PACKAGE / "README_RESULTS.md").write_text(report, encoding="utf-8")

# Package the results; exclude model weights to keep the archive smaller.
zip_path = RESULTS / "VISTA_XL_Final_Package.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for file in PACKAGE.rglob("*"):
        if file.is_file():
            archive.write(file, file.relative_to(RESULTS))

print("\nPackage created:")
print(PACKAGE)
print(f"ZIP: {zip_path}")
print("\nCurrent local evaluation:")
print(f"Samples: {len(per_image)}")
print(f"Dice: {per_image['dice'].mean():.4f}")
print(f"IoU: {per_image['iou'].mean():.4f}")
print("\nComparison is explicitly marked PROVISIONAL.")
