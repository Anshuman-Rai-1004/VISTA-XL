
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Binary cross-entropy plus soft Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits={logits.shape}, "
                f"targets={targets.shape}"
            )

        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice.mean()

        return bce + dice_loss
