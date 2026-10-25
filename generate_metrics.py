"""
generate_metrics.py
===================
Generates four-class classification and class-specific box metrics for CN2VF-Net.

For a classification model these metrics are defined as:
  Argmax accuracy: always choose the most probable class.
  Threshold metrics: reject predictions below confidence 0.50; rejected true
                     examples remain false negatives in recall/F1.
  Coverage: fraction of the split receiving a prediction at that threshold.
  Macro AP: mean one-vs-rest average precision for image classes.
  Box AP50: VOC-style AP at IoU >= 0.50 for one enclosing box per present
           class per image. This is not COCO mAP or multi-instance detection AP.

Output graphs (saved to output/metrics_report/):
  1. metrics_summary.png   - argmax accuracy and threshold metrics + coverage
  2. pr_curves.png         - per-class Precision-Recall curves with AP values
  3. confusion_matrix.png  - thresholded counts including rejected predictions
  4. per_class_bars.png    - thresholded per-class Precision / Recall / F1
  5. metrics_by_threshold.png - P / R / F1 and coverage across thresholds
  6. bbox_metrics.png        - Fire/Smoke union-box AP50, mean IoU, recall@0.50
  7. metrics_summary.json    - machine-readable classification and box metrics

Usage:
    python generate_metrics.py
    python generate_metrics.py --checkpoint "output/Final model/checkpoints/best_model.pth"
    python generate_metrics.py --data-root ./fire_smoke
"""

import argparse
import json
import sys
from pathlib import Path

# Import PyTorch before plotting/scientific packages: on some Windows setups,
# those packages preload native DLLs that prevent PyTorch's c10.dll initializing.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# ── project imports ───────────────────────────────────────────
from dataset import CombinedFireSmokeDataset
from cn2vf_net import CN2VFNet

# ─────────────────────────────────────────────────────────────
DEFAULT_IMG_SIZE = 448
OUT_DIR     = Path("output/metrics_report")
BATCH_SIZE  = 4
NUM_WORKERS = 0
CLASS_NAMES = ["Fire only", "Smoke only", "Both", "None"]
NUM_CLASSES = len(CLASS_NAMES)
COLORS      = ["#E63946", "#F4A261", "#7B61A8", "#2A9D8F"]
MODEL_TYPE  = "cn2vf_dualbox_v2"
FIRE_INDEX  = 0
SMOKE_INDEX = 1
BOTH_INDEX  = 2
# ─────────────────────────────────────────────────────────────


def latest_checkpoint() -> Path:
    """Choose the newest checkpoint compatible with the four-class model."""
    run_checkpoints = list(Path("output").glob("cn2vf_net_*/checkpoints/best_model.pth"))
    for path in sorted(run_checkpoints, key=lambda item: item.parents[1].name, reverse=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("model_type") == MODEL_TYPE:
            return path
    raise FileNotFoundError(
        "No compatible four-class dual-box best_model.pth found. Train the prepared model first."
    )


# ══════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ck, dict) or ck.get("model_type") != MODEL_TYPE:
        raise ValueError(
            "Checkpoint architecture does not match the prepared four-class dual-box model. "
            "Use a best_model.pth produced by the new training script."
        )
    model = CN2VFNet(num_classes=NUM_CLASSES)
    state = ck["model_state_dict"]
    model.load_state_dict(state)
    model.preprocessing_type = ck.get("preprocessing", "stretch")
    model.img_size = int(ck.get("img_size", DEFAULT_IMG_SIZE))
    model.to(device).eval()
    print(f"[OK] Loaded checkpoint  epoch={ck.get('epoch','?')}  val_acc={ck.get('val_acc','?')}")
    return model


# ══════════════════════════════════════════════════════════════
# Inference – returns raw softmax probabilities + ground truth
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, val_loader, device):
    """
    Returns targets and scores for image classes plus class-specific boxes.
    """
    y_true, y_probs = [], []
    fire_present, smoke_present = [], []
    fire_boxes_true, smoke_boxes_true = [], []
    fire_boxes_pred, smoke_boxes_pred = [], []
    for batch in val_loader:
        imgs   = batch["image"].to(device)
        labels = batch["class_label"]
        out    = model(imgs)
        probs  = torch.softmax(out["cls_logits"], dim=1).cpu().numpy()
        y_true.extend(labels.tolist())
        y_probs.append(probs)
        fire_present.extend(batch["fire_present"].numpy().tolist())
        smoke_present.extend(batch["smoke_present"].numpy().tolist())
        fire_boxes_true.extend(batch["fire_bbox"].numpy().tolist())
        smoke_boxes_true.extend(batch["smoke_bbox"].numpy().tolist())
        fire_boxes_pred.extend(out["fire_bbox"].cpu().numpy().tolist())
        smoke_boxes_pred.extend(out["smoke_bbox"].cpu().numpy().tolist())
    return {
        "y_true": y_true,
        "y_probs": np.vstack(y_probs),
        "fire_present": np.asarray(fire_present, dtype=bool),
        "smoke_present": np.asarray(smoke_present, dtype=bool),
        "fire_boxes_true": np.asarray(fire_boxes_true, dtype=np.float32),
        "smoke_boxes_true": np.asarray(smoke_boxes_true, dtype=np.float32),
        "fire_boxes_pred": np.asarray(fire_boxes_pred, dtype=np.float32),
        "smoke_boxes_pred": np.asarray(smoke_boxes_pred, dtype=np.float32),
    }


# ══════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════
DEFAULT_THRESHOLD = 0.50


def predictions_at_threshold(y_probs, threshold: float):
    """Return argmax predictions, replacing low-confidence ones with -1 (abstain)."""
    y_probs = np.asarray(y_probs)
    y_pred = np.argmax(y_probs, axis=1)
    confidence = np.max(y_probs, axis=1)
    y_pred_threshold = y_pred.copy()
    y_pred_threshold[confidence < threshold] = -1
    return y_pred, y_pred_threshold


def compute_metrics_at_threshold(y_true, y_probs, threshold: float):
    """Compute fixed-label macro metrics; rejected true examples count as false negatives."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_argmax, y_pred_threshold = predictions_at_threshold(y_probs, threshold)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_threshold,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    accepted = y_pred_threshold != -1
    coverage = float(np.mean(accepted)) if len(accepted) else 0.0
    selective_accuracy = (
        float(accuracy_score(y_true_arr[accepted], y_pred_threshold[accepted]))
        if np.any(accepted) else 0.0
    )

    return {
        "per_class_p": per_p.tolist(),
        "per_class_r": per_r.tolist(),
        "per_class_f1": per_f1.tolist(),
        "support": support.tolist(),
        "macro_precision": float(np.mean(per_p)),
        "macro_recall": float(np.mean(per_r)),
        "macro_f1": float(np.mean(per_f1)),
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "threshold_accuracy": float(accuracy_score(y_true_arr, y_pred_threshold)),
        "y_pred_argmax": y_pred_argmax.tolist(),
        "y_pred_threshold": y_pred_threshold.tolist(),
    }


def compute_all_metrics(y_true, y_probs, inference=None):
    """
    Compute argmax accuracy, thresholded metrics, and one-vs-rest Average Precision.

    Returns a dict with all values.
    """
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    # ── Per-class APs (full PR curve) ─────────────────────────
    aps = []
    pr_data = []   # list of (precision_arr, recall_arr, ap)
    for c in range(NUM_CLASSES):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])
        ap = float(average_precision_score(y_bin[:, c], y_probs[:, c]))
        aps.append(ap)
        pr_data.append((prec, rec, ap))

    mean_ap = float(np.mean(aps))

    operating = compute_metrics_at_threshold(y_true, y_probs, DEFAULT_THRESHOLD)
    y_pred_argmax, y_pred_threshold = predictions_at_threshold(y_probs, DEFAULT_THRESHOLD)
    box_metrics = compute_box_metrics(inference) if inference is not None else None

    return {
        "threshold":   DEFAULT_THRESHOLD,
        "argmax_accuracy": float(accuracy_score(y_true, y_pred_argmax)),
        "precision":   operating["macro_precision"],
        "recall":      operating["macro_recall"],
        "mean_ap":      mean_ap,
        "macro_f1":    operating["macro_f1"],
        "coverage":    operating["coverage"],
        "selective_accuracy": operating["selective_accuracy"],
        "threshold_accuracy": operating["threshold_accuracy"],
        "per_class_p":  operating["per_class_p"],
        "per_class_r":  operating["per_class_r"],
        "per_class_f1": operating["per_class_f1"],
        "support":      operating["support"],
        "per_class_ap": aps,
        "pr_data":      pr_data,
        "y_true":       list(y_true),
        "y_pred_argmax": operating["y_pred_argmax"],
        "y_pred_threshold": operating["y_pred_threshold"],
        "box_metrics": box_metrics,
    }


def box_iou_cxcywh(box_a, box_b):
    """IoU for normalized center-x, center-y, width, height boxes."""
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def box_ap50(present, boxes_true, boxes_pred, scores, iou_threshold=0.50):
    """VOC-style AP50 with one merged ground-truth/predicted box per image/class."""
    present = np.asarray(present, dtype=bool)
    gt_count = int(present.sum())
    if gt_count == 0:
        return {"ap50": 0.0, "mean_iou": 0.0, "recall50": 0.0, "gt_count": 0}

    ious = np.asarray([
        box_iou_cxcywh(boxes_pred[i], boxes_true[i]) if present[i] else 0.0
        for i in range(len(present))
    ])
    mean_iou = float(ious[present].mean())
    recall50 = float(np.mean(ious[present] >= iou_threshold))

    order = np.argsort(-np.asarray(scores))
    true_positives = np.asarray(
        [bool(present[i] and ious[i] >= iou_threshold) for i in order], dtype=np.float64
    )
    false_positives = 1.0 - true_positives
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / gt_count
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

    # Integrate the monotonically interpolated precision envelope. Unmatched
    # ground-truth boxes remain in gt_count, so misses reduce maximum recall.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    return {"ap50": ap, "mean_iou": mean_iou, "recall50": recall50, "gt_count": gt_count}


def compute_box_metrics(inference):
    probs = inference["y_probs"]
    # Image-class probabilities are converted to object-presence scores.
    fire_scores = probs[:, FIRE_INDEX] + probs[:, BOTH_INDEX]
    smoke_scores = probs[:, SMOKE_INDEX] + probs[:, BOTH_INDEX]
    fire_metrics = box_ap50(
        inference["fire_present"], inference["fire_boxes_true"],
        inference["fire_boxes_pred"], fire_scores,
    )
    smoke_metrics = box_ap50(
        inference["smoke_present"], inference["smoke_boxes_true"],
        inference["smoke_boxes_pred"], smoke_scores,
    )
    active = [metric["ap50"] for metric in (fire_metrics, smoke_metrics) if metric["gt_count"] > 0]
    return {
        "fire": fire_metrics,
        "smoke": smoke_metrics,
        "mean_ap50": float(np.mean(active)) if active else 0.0,
        "definition": "One enclosing ground-truth box and one predicted box per image per present class",
    }


# ══════════════════════════════════════════════════════════════
# Plot 1 – Metrics Summary Dashboard (2 × 3)
# ══════════════════════════════════════════════════════════════
def plot_metrics_summary(metrics: dict, save_dir: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Fire/Smoke Classification Metrics", fontsize=16,
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)
    headline = [
        ("Argmax accuracy", metrics["argmax_accuracy"], "#4C72B0", "Always choose the highest-probability class"),
        ("Threshold accuracy", metrics["threshold_accuracy"], "#8172B2", "Rejected predictions count as incorrect"),
        ("Macro precision", metrics["precision"], "#DD8452", f"Fixed class set @ confidence ≥ {metrics['threshold']:.2f}"),
        ("Macro recall", metrics["recall"], "#55A868", f"Rejected true examples count as misses"),
        ("Macro F1", metrics["macro_f1"], "#C44E52", "Mean per-class F1 at the operating threshold"),
        ("Coverage", metrics["coverage"], "#64B5CD", "Fraction receiving a prediction at threshold"),
    ]

    for idx, (name, val, col, subtitle) in enumerate(headline):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        # Background gauge-style bar
        ax.barh([0], [1.0], color="#e8e8e8", height=0.5, zorder=1)
        ax.barh([0], [val],  color=col,      height=0.5, zorder=2, alpha=0.85)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.8)
        ax.axis("off")

        # Big number
        ax.text(0.5, 0.35, f"{val:.4f}", ha="center", va="center",
                fontsize=34, fontweight="bold", color=col,
                transform=ax.transAxes)

        # Label
        ax.text(0.5, 0.88, name, ha="center", va="center",
                fontsize=14, fontweight="bold", color="#222222",
                transform=ax.transAxes)

        # Subtitle
        ax.text(0.5, -0.18, subtitle, ha="center", va="center",
                fontsize=9, color="#555555", style="italic",
                transform=ax.transAxes)

        # Per-class breakdown inside gauge
        pc_vals = {
            "Macro precision": metrics["per_class_p"],
            "Macro recall": metrics["per_class_r"],
            "Macro F1": metrics["per_class_f1"],
        }
        key = name
        if key in pc_vals:
            vals_pc = pc_vals[key]
            txt = "  ".join(
                f"{CLASS_NAMES[i]}:{vals_pc[i]:.3f}" for i in range(NUM_CLASSES)
            )
            ax.text(0.5, -0.38, txt, ha="center", va="center",
                    fontsize=8.5, color="#333333",
                    transform=ax.transAxes)

    plt.savefig(save_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'metrics_summary.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 2 – Per-class Precision-Recall Curves
# ══════════════════════════════════════════════════════════════
def plot_pr_curves(metrics: dict, save_dir: Path):
    pr_data = metrics["pr_data"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    fig.suptitle("Precision-Recall Curves  |  CN2VF", fontsize=13, fontweight="bold")

    for i, (cls, col, ax) in enumerate(zip(CLASS_NAMES, COLORS, axes)):
        prec, rec, ap = pr_data[i]
        ax.plot(rec, prec, color=col, lw=2.2)
        ax.fill_between(rec, prec, alpha=0.18, color=col)
        ax.axhline(metrics["per_class_p"][i], color=col, lw=1.2,
                   linestyle="--", alpha=0.6, label=f"P@{metrics['threshold']:.2f}={metrics['per_class_p'][i]:.3f}")
        ax.axvline(metrics["per_class_r"][i], color="#888", lw=1.0,
                   linestyle=":", alpha=0.6, label=f"R@{metrics['threshold']:.2f}={metrics['per_class_r'][i]:.3f}")
        ax.set_title(f"{cls}  —  AP = {ap:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall",    fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'pr_curves.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 3 – Confusion Matrix
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, save_dir: Path, threshold: float):
    # Four true classes by five outcomes; rejected predictions are explicit.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        true_label = int(true_label)
        predicted_label = int(predicted_label)
        if 0 <= true_label < NUM_CLASSES:
            cm[true_label, predicted_label if 0 <= predicted_label < NUM_CLASSES else NUM_CLASSES] += 1
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix @ confidence ≥ {threshold:.2f}  |  CN2VF",
                 fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Row-Normalised (%)"],
        ["d", ".2%"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES + ["No prediction"], yticklabels=CLASS_NAMES,
                    linewidths=0.6, ax=ax, cbar=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted outcome"); ax.set_ylabel("True label")

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'confusion_matrix.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 4 – Per-class Precision / Recall / F1 bars
# ══════════════════════════════════════════════════════════════
def plot_per_class_bars(y_true, y_pred, save_dir: Path, threshold: float):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    x = np.arange(len(CLASS_NAMES))
    w = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w, prec, w, label="Precision", color="#4C72B0", alpha=0.88)
    b2 = ax.bar(x,     rec,  w, label="Recall",    color="#DD8452", alpha=0.88)
    b3 = ax.bar(x + w, f1,   w, label="F1-Score",  color="#55A868", alpha=0.88)

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p + 0.012, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i,     r + 0.012, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.012, f"{f:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Per-Class Precision / Recall / F1 @ confidence ≥ {threshold:.2f}  |  CN2VF",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    for i, s in enumerate(sup):
        ax.text(i, -0.07, f"n={s}", ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(save_dir / "per_class_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'per_class_bars.png'}")


def plot_bbox_metrics(metrics: dict, save_dir: Path):
    box_metrics = metrics.get("box_metrics")
    if not box_metrics:
        return
    names = ["Fire", "Smoke"]
    entries = [box_metrics["fire"], box_metrics["smoke"]]
    series = {
        "Box AP50": [entry["ap50"] for entry in entries],
        "Mean IoU": [entry["mean_iou"] for entry in entries],
        "Recall @ IoU 0.50": [entry["recall50"] for entry in entries],
    }
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (metric_name, scores) in enumerate(series.items()):
        bars = ax.bar(x + (offset - 1) * width, scores, width, label=metric_name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Box Localization Metrics | one enclosing box per image/class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "bbox_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'bbox_metrics.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 5 – Metrics vs Confidence Threshold
# ══════════════════════════════════════════════════════════════
def plot_metrics_by_threshold(y_true, y_probs, metrics: dict, save_dir: Path):
    thresholds = np.arange(0.05, 1.00, 0.05)
    macro_ps, macro_rs, macro_f1s = [], [], []
    coverages = []

    for thr in thresholds:
        threshold_metrics = compute_metrics_at_threshold(y_true, y_probs, float(thr))
        macro_ps.append(threshold_metrics["macro_precision"])
        macro_rs.append(threshold_metrics["macro_recall"])
        macro_f1s.append(threshold_metrics["macro_f1"])
        coverages.append(threshold_metrics["coverage"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, macro_ps,  "b-o", markersize=4, lw=2, label="Precision (macro)")
    ax.plot(thresholds, macro_rs,  "r-o", markersize=4, lw=2, label="Recall (macro)")
    ax.plot(thresholds, macro_f1s, "g-o", markersize=4, lw=2, label="F1 (macro)")
    ax.axvline(metrics["threshold"], color="gray", lw=1.2, linestyle="--", alpha=0.7,
               label=f"operating thr={metrics['threshold']:.2f}")

    # Show accepted prediction coverage separately; rejected examples remain
    # in the recall denominator above and are counted as false negatives.
    coverage_ax = ax.twinx()
    coverage_ax.plot(thresholds, coverages, color="#555555", marker=".", linestyle=":",
                     lw=1.5, label="Coverage")
    coverage_ax.set_ylim(0, 1.05)
    coverage_ax.set_ylabel("Coverage", fontsize=11, color="#555555")

    # Annotate headline metrics at the operating threshold.
    ax.axhline(metrics["precision"], color="blue",  lw=1, linestyle=":", alpha=0.5)
    ax.axhline(metrics["recall"],    color="red",   lw=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("Macro score", fontsize=11)
    ax.set_title("Precision / Recall / F1 and Coverage vs Confidence Threshold  |  CN2VF",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    coverage_handles, coverage_labels = coverage_ax.get_legend_handles_labels()
    ax.legend(handles + coverage_handles, labels + coverage_labels, fontsize=9, loc="best")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_dir / "metrics_by_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'metrics_by_threshold.png'}")


# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════
def print_summary(y_true, y_probs, metrics: dict):
    y_pred_threshold = metrics["y_pred_threshold"]

    print("\n" + "=" * 62)
    print("  CN2VF  –  METRICS REPORT")
    print("=" * 62)
    print(f"  Samples          : {len(y_true)}")
    print(f"  Argmax Accuracy  : {metrics['argmax_accuracy'] * 100:.2f}%  (always predicts one class)")
    print(f"  Threshold        : {metrics['threshold']:.2f}")
    print(f"  Coverage          : {metrics['coverage'] * 100:.2f}%  (fraction receiving a prediction)")
    print(f"  Selective Accuracy: {metrics['selective_accuracy'] * 100:.2f}%  (among accepted predictions)")
    print(f"  Accuracy incl. rejects: {metrics['threshold_accuracy'] * 100:.2f}%  (rejects count as incorrect)")
    print("-" * 62)
    print(f"  {'Metric':<20}  {'Value':>10}  {'Definition'}")
    print("-" * 62)
    print(f"  {'Macro Precision':<20}  {metrics['precision']:>10.4f}  @ threshold; fixed four-class set")
    print(f"  {'Macro Recall':<20}  {metrics['recall']:>10.4f}  rejects count as false negatives")
    print(f"  {'Macro F1':<20}  {metrics['macro_f1']:>10.4f}  mean per-class F1 @ threshold")
    print(f"  {'Macro AP':<20}  {metrics['mean_ap']:>10.4f}  mean one-vs-rest PR area (ranking)")
    print("-" * 62)
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'AP':>10}  {'Support':>8}")
    print(f"  {'-'*67}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10}  {metrics['per_class_p'][i]:>10.4f}  "
              f"{metrics['per_class_r'][i]:>10.4f}  "
              f"{metrics['per_class_f1'][i]:>10.4f}  "
              f"{metrics['per_class_ap'][i]:>10.4f}  "
              f"{metrics['support'][i]:>8d}")
    print("=" * 62)

    print("\n  Threshold Classification Report (rejections count as misses):")
    print(classification_report(y_true, y_pred_threshold,
                                labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    if metrics.get("box_metrics"):
        box = metrics["box_metrics"]
        print("\n  Box metrics (one enclosing GT box per class/image):")
        print(f"  Fire  AP50={box['fire']['ap50']:.4f}  meanIoU={box['fire']['mean_iou']:.4f}  "
              f"recall@IoU.50={box['fire']['recall50']:.4f}  GT={box['fire']['gt_count']}")
        print(f"  Smoke AP50={box['smoke']['ap50']:.4f}  meanIoU={box['smoke']['mean_iou']:.4f}  "
              f"recall@IoU.50={box['smoke']['recall50']:.4f}  GT={box['smoke']['gt_count']}")
        print(f"  Mean box AP50={box['mean_ap50']:.4f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main(checkpoint: str, data_roots: list, split: str = "test", img_size=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load model
    model = load_model(checkpoint, device)
    eval_img_size = int(img_size or model.img_size)

    # Evaluate the already-separated held-out split; do not rebuild a split
    # from train+validation samples.
    val_ds = CombinedFireSmokeDataset(data_roots, split=split, img_size=eval_img_size)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    print(f"{split.capitalize()} samples : {len(val_ds)}")

    # Run inference
    print(f"\nRunning inference on {split} set …")
    inference = run_inference(model, val_loader, device)
    y_true, y_probs = inference["y_true"], inference["y_probs"]
    # Compute all metrics
    print("Computing metrics …")
    metrics = compute_all_metrics(y_true, y_probs, inference)

    # Console report
    print_summary(y_true, y_probs, metrics)

    # Plots
    print(f"\nGenerating graphs -> {OUT_DIR}/")
    plot_metrics_summary(metrics, OUT_DIR)
    plot_pr_curves(metrics, OUT_DIR)
    plot_confusion_matrix(y_true, metrics["y_pred_threshold"], OUT_DIR, metrics["threshold"])
    plot_per_class_bars(y_true, metrics["y_pred_threshold"], OUT_DIR, metrics["threshold"])
    plot_metrics_by_threshold(y_true, y_probs, metrics, OUT_DIR)
    plot_bbox_metrics(metrics, OUT_DIR)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=int)
    for true_label, pred_label in zip(y_true, metrics["y_pred_threshold"]):
        cm[int(true_label), int(pred_label) if int(pred_label) >= 0 else NUM_CLASSES] += 1
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split,
        "image_size": eval_img_size,
        "num_images": len(y_true),
        "class_names": CLASS_NAMES,
        "confidence_threshold": metrics["threshold"],
        "classification": {
            "argmax_accuracy": metrics["argmax_accuracy"],
            "threshold_accuracy_rejects_wrong": metrics["threshold_accuracy"],
            "coverage": metrics["coverage"],
            "selective_accuracy": metrics["selective_accuracy"],
            "macro_precision": metrics["precision"],
            "macro_recall": metrics["recall"],
            "macro_f1": metrics["macro_f1"],
            "macro_average_precision": metrics["mean_ap"],
            "per_class": {
                name: {
                    "precision": metrics["per_class_p"][i],
                    "recall": metrics["per_class_r"][i],
                    "f1": metrics["per_class_f1"][i],
                    "average_precision": metrics["per_class_ap"][i],
                    "support": metrics["support"][i],
                }
                for i, name in enumerate(CLASS_NAMES)
            },
            "threshold_confusion_matrix": {
                "rows": CLASS_NAMES,
                "columns": CLASS_NAMES + ["No prediction"],
                "counts": cm.tolist(),
            },
        },
        "localization": metrics["box_metrics"],
    }
    report_path = OUT_DIR / "metrics_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Saved: {report_path}")

    print(f"\n[OK] All graphs saved to: {OUT_DIR.resolve()}")
    print(f"\n  Threshold macro precision : {metrics['precision']:.4f}")
    print(f"  Threshold macro recall    : {metrics['recall']:.4f}")
    print(f"  Threshold macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Threshold coverage        : {metrics['coverage']:.4f}")
    print(f"  Macro Average Precision   : {metrics['mean_ap']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate classification metrics for CN2VF-Net"
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to best_model.pth (default: newest training run)")
    parser.add_argument("--data-roots", "--data-root", dest="data_roots", nargs="+",
                        default=["./d-fire", "./fire_smoke"],
                        help="Dataset roots to combine (e.g. ./d-fire ./fire_smoke)")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Evaluate validation during development or held-out test for final reporting")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override checkpoint image size (normally read from checkpoint)")
    args = parser.parse_args()
    checkpoint = args.checkpoint or str(latest_checkpoint())
    print(f"Using checkpoint: {checkpoint}")
    main(checkpoint, args.data_roots, args.split, args.img_size)
