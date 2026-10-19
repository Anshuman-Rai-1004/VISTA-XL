"""End-to-end API test using a real validation image."""
import json
import base64
import time
import urllib.request
from pathlib import Path

val_img = Path("Dataset/validation_data/ISIC_0001769/image.png")
print("Test image exists:", val_img.exists())

boundary = "BOUNDARY123"
with open(val_img, "rb") as f:
    img_data = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=body,
    method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0
    print(f"SUCCESS in {elapsed:.2f}s")
    print(f"inference_ms:   {result['inference_ms']:.1f}")
    print(f"lesion_pct:     {result['lesion_pct']:.2f}%")
    print(f"mask size:      {result['mask_w']}x{result['mask_h']}")
    print(f"epoch:          {result['epoch']}")
    print(f"val_dice:       {result['val_dice']}")
    print(f"mask_b64 len:   {len(result['mask_b64'])} chars")
    print(f"overlay_b64 len:{len(result['overlay_b64'])} chars")

    Path("results/api_test").mkdir(parents=True, exist_ok=True)
    with open("results/api_test/mask.png", "wb") as f:
        f.write(base64.b64decode(result["mask_b64"]))
    with open("results/api_test/overlay.png", "wb") as f:
        f.write(base64.b64decode(result["overlay_b64"]))
    print("Saved outputs to results/api_test/")
except Exception as e:
    print(f"ERROR: {e}")
