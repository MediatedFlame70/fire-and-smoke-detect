"""
pipeline_evaluate.py
====================
Evaluation, visualization, and error analysis for the CN2VF pipeline.

Functions:
- evaluate()          : Full evaluation with sklearn metrics
- plot_training_curves()    : Loss + accuracy curves
- plot_confusion_matrix()   : Heatmap with per-class annotations
- analyze_errors()          : Visualise Smoke ↔ Neutral misclassifications
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for server / headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from pipeline_dataset import CLASS_NAMES, IDX_TO_CLASS


# ──────────────────────────────────────────────────────────────
# evaluate()
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:      torch.nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    save_dir:   Optional[str] = None,
) -> Dict:
    """
    Run full evaluation on a DataLoader.

    Returns dict with keys:
        accuracy, per_class (precision/recall/f1), macro_f1,
        y_true (list), y_pred (list)

    Also prints sklearn classification_report and saves confusion matrix.
    """
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["class_label"]

        output = model(images)
        # Support both dict output (existing CN2VFNet) and plain tensor
        if isinstance(output, dict):
            logits = output["cls_logits"]
        else:
            logits = output

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    # ── Metrics ──────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report   = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(report)

    # Sklearn report also gives per-class f1; replicate in dict
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1[i]),   4),
        }

    results = {
        "accuracy":  round(accuracy, 4),
        "macro_f1":  round(macro_f1, 4),
        "per_class": per_class,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "confusion_matrix": cm,
    }

    # ── Save confusion matrix ─────────────────────────────────
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(cm, save_path=str(Path(save_dir) / "confusion_matrix.png"))

    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# plot_training_curves()
# ──────────────────────────────────────────────────────────────
def plot_training_curves(
    history:   Dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training vs validation loss and accuracy curves side by side.

    Args:
        history: dict with keys "train_loss", "val_loss",
                              "train_acc",  "val_acc"
        save_path: if given, save figure to this path
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CN2VF Training Curves", fontsize=14, fontweight="bold")

    # — Loss —
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — Accuracy —
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc")
    ax.set_title("Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Training curves saved -> {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# plot_confusion_matrix()
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm:        np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Render a color-coded confusion matrix heatmap with per-cell counts and row-norm %.

    Args:
        cm:        confusion_matrix output (3×3 array)
        save_path: path to save figure
    """
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Row-Normalised (%)"],
        ["d",      ".2%"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            ax=ax,
            cbar=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Confusion matrix saved -> {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# analyze_errors()
# ──────────────────────────────────────────────────────────────
def analyze_errors(
    model:        torch.nn.Module,
    dataloader:   DataLoader,
    device:       torch.device,
    save_dir:     str,
    max_images:   int = 16,
    focus_pair:   Tuple[int, int] = (1, 2),   # (Smoke=1, Neutral=2)
) -> None:
    """
    Identify and visualise misclassified samples, with special focus on the
    Smoke ↔ Neutral confusion pair.

    Args:
        model:       Trained model
        dataloader:  Validation DataLoader
        device:      Torch device
        save_dir:    Directory to save visualisation grids
        max_images:  Max misclassified images to display per plot
        focus_pair:  Tuple of (class_a, class_b) to highlight separately
    """
    import torchvision.transforms.functional as TF_fn
    from PIL import Image

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    misclassified   = []        # all wrong predictions
    focus_errors    = []        # only Smoke <-> Neutral errors

    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD  = torch.tensor([0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["class_label"]
            filenames = batch["filename"]

            output = model(images)
            logits = output["cls_logits"] if isinstance(output, dict) else output
            probs  = torch.softmax(logits, dim=1).cpu()
            preds  = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                true_cls = labels[i].item()
                pred_cls = preds[i].item()

                if true_cls == pred_cls:
                    continue

                # Un-normalise image for display
                img_t = images[i].cpu()
                img_t = img_t * STD[:, None, None] + MEAN[:, None, None]
                img_t = img_t.clamp(0, 1)

                entry = {
                    "image":     img_t,
                    "true_cls":  true_cls,
                    "pred_cls":  pred_cls,
                    "confidence": probs[i, pred_cls].item(),
                    "filename":  filenames[i],
                }
                misclassified.append(entry)

                # Track focus pair (Smoke ↔ Neutral)
                pair = {true_cls, pred_cls}
                if pair == set(focus_pair):
                    focus_errors.append(entry)

    print(f"\nError Analysis: {len(misclassified)} total misclassifications")
    print(f"  Smoke <-> Neutral confusion: {len(focus_errors)} samples")

    # ── Plot all misclassified ────────────────────────────────
    _save_error_grid(
        misclassified[:max_images],
        title="All Misclassifications",
        save_path=str(save_path / "errors_all.png"),
    )

    # ── Plot focus-pair errors ────────────────────────────────
    if focus_errors:
        _save_error_grid(
            focus_errors[:max_images],
            title="Smoke <-> Neutral Confusion",
            save_path=str(save_path / "errors_smoke_neutral.png"),
        )


def _save_error_grid(
    entries:   List[Dict],
    title:     str,
    save_path: str,
    ncols:     int = 4,
) -> None:
    """Render a grid of misclassified images with labels."""
    if not entries:
        print(f"  No entries for: {title}")
        return

    n     = len(entries)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    axes = np.array(axes).flatten()

    for i, entry in enumerate(entries):
        ax = axes[i]
        img_np = entry["image"].permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        true_name = IDX_TO_CLASS[entry["true_cls"]]
        pred_name = IDX_TO_CLASS[entry["pred_cls"]]
        conf      = entry["confidence"]

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1%})",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[OK] Error grid saved -> {save_path}")
    plt.close(fig)
