"""
pipeline_losses.py
==================
Loss functions and evaluation metrics for CN2VF fire/smoke/neutral pipeline.

Contains:
- FocalLoss   : Down-weights easy samples, focuses on hard (Smoke ↔ Neutral confusion)
- WeightedCE  : Standard CrossEntropy with class weights
- CN2VFClassificationLoss : Wrapper that routes to either loss
- Metric utilities: F1-score computation used for model selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduces the loss contribution from easy samples
    - alpha provides per-class weighting (class imbalance correction)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma:     Focusing parameter. Higher → harder samples get more weight.
                       Recommended: 2.0 (original paper default)
            alpha:     Per-class weight tensor of shape [num_classes].
                       Pass compute_class_weights() output here.
            reduction: "mean" | "sum" | "none"
        """
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha          # stored on correct device via .to(device) later
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw unnormalised scores
            targets: [B]   ground truth class indices
        """
        # Compute cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(
            input=logits,
            target=targets,
            weight=self.alpha,
            reduction="none",
        )
        # p_t = probability of true class
        p_t = torch.exp(-ce_loss)
        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ──────────────────────────────────────────────────────────────
# Unified Classification Loss Wrapper
# ──────────────────────────────────────────────────────────────
class CN2VFClassificationLoss(nn.Module):
    """
    Wrapper that provides a unified interface for either:
      - Weighted CrossEntropy  (use_focal=False)
      - Focal Loss             (use_focal=True)

    Both respect per-class weights from compute_class_weights().
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            class_weights: [num_classes] tensor from compute_class_weights()
            use_focal:     If True use Focal Loss, else Weighted CrossEntropy
            focal_gamma:   Focal loss gamma (only used when use_focal=True)
        """
        super().__init__()
        self.use_focal = use_focal

        if use_focal:
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_weights)
            print(f"Loss: Focal Loss  (gamma={focal_gamma})")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("Loss: Weighted CrossEntropy")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)

    def to(self, device):
        """Ensure alpha weights are on the right device."""
        super().to(device)
        if self.use_focal and self.loss_fn.alpha is not None:
            self.loss_fn.alpha = self.loss_fn.alpha.to(device)
        return self


# ──────────────────────────────────────────────────────────────
# Metric Utilities
# ──────────────────────────────────────────────────────────────
def compute_f1_macro(
    y_true: list,
    y_pred: list,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> float:
    """
    Compute macro-averaged F1-score (pure PyTorch, no sklearn dependency at runtime).
    This is used inside the training loop for fast model selection.
    sklearn is used only in the evaluation module for detailed reports.

    Args:
        y_true:      List of ground-truth class indices
        y_pred:      List of predicted class indices
        num_classes: Number of classes
        eps:         Small value to avoid division by zero

    Returns:
        macro F1 score (float)
    """
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_pred_t = torch.tensor(y_pred, dtype=torch.long)

    f1_scores = []
    for cls in range(num_classes):
        tp = ((y_pred_t == cls) & (y_true_t == cls)).sum().float()
        fp = ((y_pred_t == cls) & (y_true_t != cls)).sum().float()
        fn = ((y_pred_t != cls) & (y_true_t == cls)).sum().float()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1.item())

    return float(sum(f1_scores) / num_classes)


def compute_per_class_f1(
    y_true: list,
    y_pred: list,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> dict:
    """
    Return per-class precision, recall, F1.
    Keys: class_name → {"precision", "recall", "f1"}
    """
    from pipeline_dataset import IDX_TO_CLASS

    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_pred_t = torch.tensor(y_pred, dtype=torch.long)

    results = {}
    for cls in range(num_classes):
        tp = ((y_pred_t == cls) & (y_true_t == cls)).sum().float()
        fp = ((y_pred_t == cls) & (y_true_t != cls)).sum().float()
        fn = ((y_pred_t != cls) & (y_true_t == cls)).sum().float()

        precision = (tp / (tp + fp + eps)).item()
        recall    = (tp / (tp + fn + eps)).item()
        f1 = 2 * precision * recall / (precision + recall + eps)

        results[IDX_TO_CLASS[cls]] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }
    return results
