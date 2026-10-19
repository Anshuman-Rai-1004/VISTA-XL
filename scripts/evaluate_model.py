from pathlib import Path
import sys
import time
import csv
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vista_xl import VISTAXL

DATA_DIR = ROOT / "Dataset" / "validation_data"
CHECKPOINT = ROOT / "checkpoints" / "vista_xl_best.pth"
OUT_DIR = ROOT / "results" / "full_validation"
IMAGE_SIZE = 224
THRESHOLD = 0.5
EXCLUDED_IDS = {"ISIC_0012127"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValidationDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for folder in sorted(DATA_DIR.iterdir()):
            if not folder.is_dir() or folder.name in EXCLUDED_IDS:
                continue

            image_path = folder / "image.png"
            mask_path = folder / "mask.png"

            if image_path.exists() and mask_path.exists():
                self.samples.append(
                    (folder.name, image_path, mask_path)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )
        mask = torch.from_numpy(
            (np.array(mask) >= 128).astype(np.float32)
        ).unsqueeze(0)

        return sample_id, image, mask


def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Dataset: {DATA_DIR}", flush=True)
    print(f"Checkpoint: {CHECKPOINT}", flush=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    dataset = ValidationDataset()
    print(f"Validation samples found: {len(dataset)}", flush=True)

    if len(dataset) == 0:
        raise RuntimeError("No valid validation samples found.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Creating VISTA-XL model...", flush=True)
    model = VISTAXL(pretrained=False).to(DEVICE)

    print("Loading checkpoint...", flush=True)
    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Starting evaluation...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()

    with torch.inference_mode():
        for index, (sample_ids, images, masks) in enumerate(
            tqdm(loader, desc="Evaluating"), start=1
        ):
            images = images.to(DEVICE)
            masks_np = masks.squeeze(0).squeeze(0).numpy() >= 0.5

            image_start = time.time()
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probabilities = torch.sigmoid(logits)
            predictions = (
                probabilities.squeeze().cpu().numpy() >= THRESHOLD
            )
            inference_seconds = time.time() - image_start

            metrics = calculate_metrics(predictions, masks_np)
            metrics["sample_id"] = sample_ids[0]
            metrics["inference_seconds"] = inference_seconds
            rows.append(metrics)

            if index % 10 == 0 or index == len(dataset):
                print(
                    f"Processed {index}/{len(dataset)} images",
                    flush=True
                )

    frame = pd.DataFrame(rows)
    per_image_path = OUT_DIR / "per_image_metrics.csv"
    summary_path = OUT_DIR / "summary_metrics.csv"

    frame.to_csv(per_image_path, index=False)

    metric_names = [
        "dice", "iou", "precision", "recall",
        "specificity", "pixel_accuracy", "inference_seconds"
    ]
    summary_rows = []

    for name in metric_names:
        summary_rows.append({
            "metric": name,
            "mean": frame[name].mean(),
            "std": frame[name].std(ddof=1)
        })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n=== VISTA-XL Validation Summary ===", flush=True)
    print(f"Samples evaluated: {len(frame)}", flush=True)

    for name in metric_names:
        mean = frame[name].mean()
        std = frame[name].std(ddof=1)
        print(
            f"{name}: {mean:.4f} ({mean * 100:.2f}%), "
            f"std={std:.4f}",
            flush=True
        )

    print(
        f"Total elapsed time: {time.time() - start:.2f} seconds",
        flush=True
    )
    print(f"Per-image CSV: {per_image_path}", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)
    print(
        "\nIMPORTANT: This script uses ImageNet normalization. "
        "Confirm it matches the original training preprocessing "
        "before treating these metrics as final.",
        flush=True
    )


if __name__ == "__main__":
    main()
