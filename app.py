from pathlib import Path
import sys
import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.vista_xl import VISTAXL

CHECKPOINT = ROOT / "checkpoints" / "vista_xl_best.pth"
IMAGE_SIZE = 224
THRESHOLD = 0.5

st.set_page_config(
    page_title="VISTA-XL | Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.hero {
    padding: 1.5rem 1.8rem;
    border: 1px solid #293548;
    border-radius: 16px;
    background: linear-gradient(120deg, #101b2d, #182b40);
    margin-bottom: 1.5rem;
}
.hero h1 {margin: 0; color: #f3f7ff; font-size: 2.2rem;}
.hero p {color: #c1d0e3; margin: .5rem 0 0 0;}
.metric-card {
    border: 1px solid #293548;
    padding: 1rem;
    border-radius: 12px;
    background: rgba(100,120,150,.07);
}
.small-note {color: #8b9bb0; font-size: .85rem;}
</style>

<div class="hero">
    <h1>VISTA-XL</h1>
    <p>Skin Lesion Segmentation · Swin Transformer + mLSTM</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading VISTA-XL checkpoint...")
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = VISTAXL(pretrained=False)
    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def prepare_image(pil_image):
    original = pil_image.convert("RGB")
    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0

    # Matches the ImageNet normalization used by the local evaluator.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return original, resized, tensor


def predict(model, tensor):
    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = probability >= THRESHOLD
    return probability, mask


def make_mask_image(mask):
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def make_overlay(image, mask):
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    color = np.zeros_like(base)
    color[..., 0] = 255
    alpha = 0.42
    selected = mask.astype(bool)
    base[selected] = (
        base[selected] * (1 - alpha) + color[selected] * alpha
    )
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def calculate_metrics(prediction, truth):
    pred = prediction.astype(bool)
    gt = truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(dice), float(iou)


with st.sidebar:
    st.header("Controls")
    st.write("**Model:** VISTA-XL")
    st.write("**Input size:** 224 × 224")
    st.write("**Threshold:** 0.50")
    st.write("**Device:** CPU")
    st.divider()
    st.caption(
        "Upload a dermoscopic image. A ground-truth mask is optional "
        "and is only needed to calculate Dice and IoU."
    )

uploaded_file = st.file_uploader(
    "Upload a skin-lesion image",
    type=["png", "jpg", "jpeg", "bmp"],
)

ground_truth_file = st.file_uploader(
    "Optional: upload ground-truth mask",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Use a binary mask where lesion pixels are white and background is black.",
)

if uploaded_file is None:
    st.info("Upload an image to start segmentation.")
    st.markdown("""
    **Workflow**
    1. Upload a dermoscopic image.
    2. Run VISTA-XL inference.
    3. Inspect the predicted mask and overlay.
    4. Optionally compare with a ground-truth mask.
    """)
else:
    try:
        uploaded_image = Image.open(uploaded_file)
        original, resized, tensor = prepare_image(uploaded_image)

        with st.spinner("Running VISTA-XL inference on CPU..."):
            model, checkpoint = load_model()
            probability, prediction = predict(model, tensor)
            overlay = make_overlay(resized, prediction)
            mask_image = make_mask_image(prediction)

        st.success("Segmentation completed.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input image")
            st.image(resized, use_container_width=True)
        with col2:
            st.subheader("Predicted mask")
            st.image(mask_image, use_container_width=True)
        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        lesion_percentage = float(prediction.mean() * 100)
        st.markdown("### Prediction summary")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted lesion area", f"{lesion_percentage:.2f}%")
        metric_cols[1].metric("Output resolution", f"{IMAGE_SIZE} × {IMAGE_SIZE}")
        metric_cols[2].metric("Threshold", f"{THRESHOLD:.2f}")

        st.caption(
            "Predicted lesion area is the percentage of pixels classified "
            "as lesion, not a clinical severity score."
        )

        if ground_truth_file is not None:
            truth_image = Image.open(ground_truth_file).convert("L")
            truth_image = truth_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.NEAREST,
            )
            truth = np.asarray(truth_image) >= 128
            dice, iou = calculate_metrics(prediction, truth)

            st.markdown("### Ground-truth evaluation")
            gt_col1, gt_col2, gt_col3 = st.columns(3)
            with gt_col1:
                st.image(truth_image, caption="Ground-truth mask",
                         use_container_width=True)
            with gt_col2:
                st.metric("Dice", f"{dice:.4f}")
            with gt_col3:
                st.metric("IoU", f"{iou:.4f}")

        st.markdown("### Export results")

        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")

        overlay_buffer = io.BytesIO()
        overlay.save(overlay_buffer, format="PNG")

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download predicted mask",
                data=mask_buffer.getvalue(),
                file_name="vista_xl_predicted_mask.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "Download overlay",
                data=overlay_buffer.getvalue(),
                file_name="vista_xl_overlay.png",
                mime="image/png",
                use_container_width=True,
            )

        with st.expander("Model checkpoint details"):
            st.write("Checkpoint:", str(CHECKPOINT))
            st.write("Saved epoch:", checkpoint.get("epoch", "Unknown"))
            st.write(
                "Stored validation Dice:",
                checkpoint.get("val_dice", "Unknown"),
            )
            st.write(
                "Stored validation IoU:",
                checkpoint.get("val_iou", "Unknown"),
            )

        st.warning(
            "Research prototype only. Predictions are not a diagnosis "
            "and must not be used as a substitute for professional medical assessment."
        )

    except Exception as exc:
        st.error(f"Could not run segmentation: {exc}")
        st.exception(exc)
