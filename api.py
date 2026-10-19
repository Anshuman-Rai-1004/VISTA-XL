"""
VISTA-XL FastAPI Inference Server
----------------------------------
Loads the trained Vista-XL checkpoint once at startup and serves
POST /predict — accepts a multipart image upload, runs inference,
returns JSON with base64-encoded mask/overlay images and metadata.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────
# Model is loaded lazily on first request (or at startup).
# torch is imported at the top level; only the heavy checkpoint
# and VISTA-XL class are deferred until _load_model_once().
# ──────────────────────────────────────────────────────────────
_model = None
_checkpoint_meta: dict = {}
_model_error: str | None = None

CHECKPOINT_PATH = ROOT / "checkpoints" / "vista_xl_best.pth"
IMAGE_SIZE = 224
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ACCEPTED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}


def _load_model_once():
    """Import torch + model and load checkpoint exactly once."""
    global _model, _checkpoint_meta, _model_error

    if _model is not None:
        return
    if _model_error is not None:
        raise RuntimeError(_model_error)

    try:
        from src.models.vista_xl import VISTAXL

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        model = VISTAXL(pretrained=False)
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval()

        _model = model
        _checkpoint_meta = {
            "epoch": ckpt.get("epoch"),
            "val_dice": ckpt.get("val_dice"),
            "val_iou": ckpt.get("val_iou"),
        }
    except Exception as exc:
        _model_error = str(exc)
        raise


# ──────────────────────────────────────────────────────────────
# Attempt eager load at startup (non-fatal — users still get a
# clear error message if the checkpoint is unavailable)
# ──────────────────────────────────────────────────────────────
try:
    _load_model_once()
    print(f"[VISTA-XL] Model loaded. Epoch={_checkpoint_meta.get('epoch')}, "
          f"val_dice={_checkpoint_meta.get('val_dice')}")
except Exception as exc:
    print(f"[VISTA-XL] WARNING: Model not loaded at startup — {exc}")


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VISTA-XL Inference API",
    description="Skin-lesion segmentation API. POST an image, receive a binary mask.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    mask_b64: str
    overlay_b64: str
    input_w: int
    input_h: int
    model_res: int
    mask_w: int
    mask_h: int
    lesion_pct: float
    inference_ms: float
    epoch: int | None
    val_dice: float | None
    val_iou: float | None
    threshold: float
    architecture: str


# ──────────────────────────────────────────────────────────────
# Helper functions (mirroring app.py to preserve exact logic)
# ──────────────────────────────────────────────────────────────

def _prepare_image(pil_image: Image.Image):
    original = pil_image.convert("RGB")
    orig_w, orig_h = original.size

    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return original, resized, tensor, orig_w, orig_h


def _predict(tensor):
    with torch.inference_mode():
        logits = _model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = prob >= THRESHOLD
    return mask


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    color = np.zeros_like(base)
    color[..., 0] = 233  # coral R
    color[..., 1] = 123  # coral G
    color[..., 2] = 119  # coral B
    alpha = 0.42
    selected = mask.astype(bool)
    base[selected] = base[selected] * (1 - alpha) + color[selected] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    loaded = _model is not None
    return {
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "error": _model_error,
        "checkpoint": str(CHECKPOINT_PATH),
        "epoch": _checkpoint_meta.get("epoch"),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # ── Validate upload ──
    content_type = (file.content_type or "").lower()
    if content_type not in ACCEPTED_MIME and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Use PNG, JPG, or BMP.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) / 1024 / 1024:.1f} MB). Max 20 MB.",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── Load model (may already be loaded) ──
    try:
        _load_model_once()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")

    # ── Open image ──
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.verify()  # check integrity
        pil_image = Image.open(io.BytesIO(raw))  # re-open after verify
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    # ── Inference ──
    try:
        original, resized, tensor, orig_w, orig_h = _prepare_image(pil_image)

        t0 = time.perf_counter()
        mask = _predict(tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        mask_img    = _mask_to_pil(mask)
        overlay_img = _make_overlay(resized, mask)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    lesion_pct = float(mask.mean() * 100.0)

    return PredictResponse(
        mask_b64    = _pil_to_b64(mask_img),
        overlay_b64 = _pil_to_b64(overlay_img),
        input_w     = orig_w,
        input_h     = orig_h,
        model_res   = IMAGE_SIZE,
        mask_w      = mask_img.width,
        mask_h      = mask_img.height,
        lesion_pct  = lesion_pct,
        inference_ms= inference_ms,
        epoch       = _checkpoint_meta.get("epoch"),
        val_dice    = _checkpoint_meta.get("val_dice"),
        val_iou     = _checkpoint_meta.get("val_iou"),
        threshold   = THRESHOLD,
        architecture= "Swin-Tiny + mLSTM + U-Net Decoder",
    )
