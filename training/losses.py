
import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss computed per-sample across the batch.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        batch_size = logits.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (probs * targets).sum(dim=1)
        total = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Dynamically scales down the loss for easy-to-classify background pixels.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class

        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedFocalDiceLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss.
    Loss = focal_weight * Focal + dice_weight * Dice
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
        positive_class_weight: float = 1.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.register_buffer(
            "positive_class_weight", torch.tensor(float(positive_class_weight)))
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_class_weight)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * bce_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """Tversky loss with configurable false-positive/false-negative penalties."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        true_positive = (probabilities * targets).sum(dim=1)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=1)
        score = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive +
            self.beta * false_negative + self.smooth)
        return (1.0 - score).mean()


class CombinedBCETverskyLoss(nn.Module):
    """BCE plus Tversky loss, weighted toward reducing false positives."""

    def __init__(self, bce_weight: float = 0.5, tversky_weight: float = 0.5,
                 alpha: float = 0.7, beta: float = 0.3,
                 positive_class_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.register_buffer(
            "positive_class_weight", torch.tensor(float(positive_class_weight)))
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_class_weight)
        return self.bce_weight * bce + self.tversky_weight * self.tversky(logits, targets)
