from pathlib import Path
from PIL import Image
import numpy as np

root = Path(__file__).resolve().parent.parent / "Dataset"
checked = 0

for split in ("training_data", "validation_data"):
    for sample in sorted((root / split).iterdir()):
        if not sample.is_dir():
            continue

        image_path = sample / "image.png"
        mask_path = sample / "mask.png"

        try:
            with Image.open(image_path) as im:
                im.load()
                image_size = im.size

            with Image.open(mask_path) as im:
                im.load()
                mask = np.array(im.convert("L"))

            print(
                f"OK: {split}/{sample.name} | "
                f"image={image_size} | "
                f"mask_values={np.unique(mask).tolist()}"
            )
            checked += 1

        except Exception as exc:
            print(f"ERROR: {sample}: {exc}")

        if checked >= 10:
            break

    if checked >= 10:
        break

print(f"\nSuccessfully checked: {checked} samples")
