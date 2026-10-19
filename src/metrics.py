
import torch


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> dict:
    """
    Calculate per-image Dice and IoU, then return batch means.

    Args:
        logits: Raw model outputs, shape (B, 1, H, W).
        targets: Binary masks, shape (B, 1, H, W), values 0 or 1.
        threshold: Probability threshold for foreground prediction.
        smooth: Small constant to avoid division by zero.

    Returns:
        Dictionary containing mean Dice and IoU.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, "
            f"targets={targets.shape}"
        )

    probabilities = torch.sigmoid(logits.float())
    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (predictions * targets).sum(dim=1)
    pred_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        pred_sum + target_sum + smooth
    )

    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }
