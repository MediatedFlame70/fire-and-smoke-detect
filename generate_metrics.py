"""
generate_metrics.py
===================
Generates classification metrics for CN2VF-Net on a held-out split.

For a classification model these metrics are defined as:
  Precision  : TP / (TP + FP)  at confidence threshold >= 0.5
  Recall     : TP / (TP + FN)  at confidence threshold >= 0.5
  Macro AP   : mean one-vs-rest area under the precision-recall curve.
  Macro F1   : mean of per-class F1 scores at confidence threshold 0.50.
               These are image-classification metrics, not box-IoU detection mAP.

Output graphs (saved to output/metrics_report/):
  1. metrics_summary.png   - dashboard: accuracy / macro P / macro R / macro F1
  2. pr_curves.png         - per-class Precision-Recall curves with AP values
  3. confusion_matrix.png  - counts + row-normalised heatmaps
  4. per_class_bars.png    - Precision / Recall / F1 per class bar chart
  5. metrics_by_threshold.png - how P, R, F1 vary with confidence threshold

Usage:
    python generate_metrics.py
    python generate_metrics.py --checkpoint "output/Final model/checkpoints/best_model.pth"
    python generate_metrics.py --data-root ./fire_smoke
"""

import argparse
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
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# ── project imports ───────────────────────────────────────────
from dataset import FireSmokeDataset
from cn2vf_net import CN2VFNet

# ─────────────────────────────────────────────────────────────
DATA_ROOT   = "./fire_smoke"
IMG_SIZE    = 448
OUT_DIR     = Path("output/metrics_report")
BATCH_SIZE  = 16
NUM_WORKERS = 0
COLORS      = ["#E63946", "#F4A261", "#2A9D8F"]   # Fire / Smoke / Neutral
CLASS_NAMES = ["Fire", "Smoke", "Neutral"]
# ─────────────────────────────────────────────────────────────


def latest_checkpoint() -> Path:
    """Choose the best checkpoint from the newest training run."""
    run_checkpoints = list(Path("output").glob("cn2vf_net_*/checkpoints/best_model.pth"))
    if run_checkpoints:
        return max(run_checkpoints, key=lambda path: path.parents[1].name)

    legacy_checkpoint = Path("output/Final model/checkpoints/best_model.pth")
    if legacy_checkpoint.exists():
        return legacy_checkpoint
    raise FileNotFoundError(
        "No best_model.pth found. Train the model first or pass --checkpoint <path>."
    )


# ══════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ck    = torch.load(checkpoint_path, map_location=device)
    model = CN2VFNet(num_classes=3)
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state)
    model.preprocessing_type = ck.get("preprocessing", "stretch")
    model.to(device).eval()
    print(f"[OK] Loaded checkpoint  epoch={ck.get('epoch','?')}  val_acc={ck.get('val_acc','?')}")
    return model


# ══════════════════════════════════════════════════════════════
# Inference – returns raw softmax probabilities + ground truth
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, val_loader, device):
    """
    Returns:
        y_true  list[int]          ground-truth class indices
        y_probs np.ndarray [N, 3]  softmax probabilities
    """
    y_true, y_probs = [], []
    for batch in val_loader:
        imgs   = batch["image"].to(device)
        labels = batch["class_label"]
        out    = model(imgs)
        logits = out["cls_logits"] if isinstance(out, dict) else out
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        y_true.extend(labels.tolist())
        y_probs.append(probs)
    return y_true, np.vstack(y_probs)


# ══════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════
def compute_ap(precision, recall):
    """
    Area under precision-recall curve using sklearn auc().
    sklearn's precision_recall_curve returns recall in *decreasing* order;
    we reverse so recall is increasing before calling auc().
    """
    # reverse so recall is ascending (required by sklearn auc)
    rec  = recall[::-1]
    prec = precision[::-1]
    # remove duplicate recall values (keep max precision)
    _, unique_idx = np.unique(rec, return_index=True)
    rec  = rec[unique_idx]
    prec = prec[unique_idx]
    if len(rec) < 2:
        return 0.0
    return float(auc(rec, prec))


def compute_metrics_at_threshold(y_true, y_probs, threshold: float):
    """
    Compute per-class and macro Precision, Recall at a given
    confidence threshold.  Samples below threshold are treated as
    'no prediction' (counted as FN for that class).
    Returns per_p, per_r, macro_p, macro_r  (all floats).
    """
    y_true_arr = np.array(y_true)
    n_cls = y_probs.shape[1]
    per_p, per_r = [], []

    for c in range(n_cls):
        # predicted as class c with confidence >= threshold
        pred_mask  = (np.argmax(y_probs, axis=1) == c) & (y_probs[:, c] >= threshold)
        true_mask  = y_true_arr == c

        tp = int(np.sum(pred_mask & true_mask))
        fp = int(np.sum(pred_mask & ~true_mask))
        fn = int(np.sum(~pred_mask & true_mask))

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        per_p.append(p)
        per_r.append(r)

    return per_p, per_r, float(np.mean(per_p)), float(np.mean(per_r))


def compute_all_metrics(y_true, y_probs):
    """
    Compute classification precision/recall at confidence 0.50, macro F1,
    and mean one-vs-rest AP over full precision-recall curves.

    Returns a dict with all values.
    """
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    # ── Per-class APs (full PR curve) ─────────────────────────
    aps = []
    pr_data = []   # list of (precision_arr, recall_arr, ap)
    for c in range(3):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])
        ap = compute_ap(prec, rec)
        aps.append(ap)
        pr_data.append((prec, rec, ap))

    mean_ap = float(np.mean(aps))

    # ── Precision / Recall @ threshold 0.50 ───────────────────
    per_p, per_r, macro_p, macro_r = compute_metrics_at_threshold(y_true, y_probs, 0.50)
    macro_f1 = float(np.mean([
        2 * p * r / (p + r + 1e-12) for p, r in zip(per_p, per_r)
    ]))

    return {
        "precision":    macro_p,
        "recall":       macro_r,
        "mean_ap":      mean_ap,
        "macro_f1":     macro_f1,
        "per_class_p":  per_p,
        "per_class_r":  per_r,
        "per_class_ap": aps,
        "pr_data":      pr_data,
        "y_true":       list(y_true),
        "y_pred":       np.argmax(y_probs, axis=1).tolist(),
    }


# ══════════════════════════════════════════════════════════════
# Plot 1 – Metrics Summary Dashboard (2 × 2)
# ══════════════════════════════════════════════════════════════
def plot_metrics_summary(metrics: dict, save_dir: Path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Fire/Smoke Classification Metrics", fontsize=16,
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    accuracy = accuracy_score(metrics["y_true"], metrics["y_pred"])
    headline = [
        ("Accuracy",  accuracy,              "#4C72B0", "Overall correct predictions"),
        ("Precision", metrics["precision"], "#DD8452", "Macro Precision @ conf ≥ 0.50"),
        ("Recall",    metrics["recall"],    "#55A868", "Macro Recall @ conf ≥ 0.50"),
        ("Macro F1",  metrics["macro_f1"], "#C44E52", "Mean class F1 @ conf ≥ 0.50"),
    ]

    for idx, (name, val, col, subtitle) in enumerate(headline):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

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
            "Precision":  metrics["per_class_p"],
            "Recall":     metrics["per_class_r"],
            "Macro F1":   [2 * p * r / (p + r + 1e-12) for p, r in zip(metrics["per_class_p"], metrics["per_class_r"])],
            "Accuracy":   [accuracy] * len(CLASS_NAMES),
        }
        key = name
        if key in pc_vals:
            vals_pc = pc_vals[key]
            txt = "  ".join(
                f"{CLASS_NAMES[i][0]}:{vals_pc[i]:.3f}" for i in range(3)
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Precision-Recall Curves  |  CN2VF", fontsize=13, fontweight="bold")

    for i, (cls, col, ax) in enumerate(zip(CLASS_NAMES, COLORS, axes)):
        prec, rec, ap = pr_data[i]
        ax.plot(rec, prec, color=col, lw=2.2)
        ax.fill_between(rec, prec, alpha=0.18, color=col)
        ax.axhline(metrics["per_class_p"][i], color=col, lw=1.2,
                   linestyle="--", alpha=0.6, label=f"P={metrics['per_class_p'][i]:.3f}")
        ax.axvline(metrics["per_class_r"][i], color="#888", lw=1.0,
                   linestyle=":", alpha=0.6, label=f"R={metrics['per_class_r'][i]:.3f}")
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
def plot_confusion_matrix(y_true, y_pred, save_dir: Path):
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix  |  CN2VF", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Row-Normalised (%)"],
        ["d", ".2%"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.6, ax=ax, cbar=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'confusion_matrix.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 4 – Per-class Precision / Recall / F1 bars
# ══════════════════════════════════════════════════════════════
def plot_per_class_bars(y_true, y_pred, save_dir: Path):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, prec, w, label="Precision", color="#4C72B0", alpha=0.88)
    b2 = ax.bar(x,     rec,  w, label="Recall",    color="#DD8452", alpha=0.88)
    b3 = ax.bar(x + w, f1,   w, label="F1-Score",  color="#55A868", alpha=0.88)

    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p + 0.012, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i,     r + 0.012, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.012, f"{f:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Class  Precision / Recall / F1  |  CN2VF",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    for i, s in enumerate(sup):
        ax.text(i, -0.07, f"n={s}", ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(save_dir / "per_class_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'per_class_bars.png'}")


# ══════════════════════════════════════════════════════════════
# Plot 5 – Metrics vs Confidence Threshold
# ══════════════════════════════════════════════════════════════
def plot_metrics_by_threshold(y_true, y_probs, metrics: dict, save_dir: Path):
    thresholds = np.arange(0.05, 1.00, 0.05)
    macro_ps, macro_rs, macro_f1s = [], [], []

    for thr in thresholds:
        y_pred_thr = []
        for i in range(len(y_true)):
            best_cls  = int(np.argmax(y_probs[i]))
            best_conf = float(y_probs[i, best_cls])
            y_pred_thr.append(best_cls if best_conf >= thr else -1)

        # filter out abstained predictions for sklearn
        valid = [(yt, yp) for yt, yp in zip(y_true, y_pred_thr) if yp != -1]
        if not valid:
            macro_ps.append(0.0); macro_rs.append(0.0); macro_f1s.append(0.0)
            continue
        yt_v, yp_v = zip(*valid)
        p, r, f, _ = precision_recall_fscore_support(
            yt_v, yp_v, average="macro", zero_division=0
        )
        macro_ps.append(p); macro_rs.append(r); macro_f1s.append(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, macro_ps,  "b-o", markersize=4, lw=2, label="Precision (macro)")
    ax.plot(thresholds, macro_rs,  "r-o", markersize=4, lw=2, label="Recall (macro)")
    ax.plot(thresholds, macro_f1s, "g-o", markersize=4, lw=2, label="F1 (macro)")
    ax.axvline(0.50, color="gray", lw=1.2, linestyle="--", alpha=0.7, label="thr=0.50")

    # Annotate headline metrics at 0.50
    ax.axhline(metrics["precision"], color="blue",  lw=1, linestyle=":", alpha=0.5)
    ax.axhline(metrics["recall"],    color="red",   lw=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("Score",                fontsize=11)
    ax.set_title("Precision / Recall / F1  vs  Confidence Threshold  |  CN2VF",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_dir / "metrics_by_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'metrics_by_threshold.png'}")


# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════
def print_summary(y_true, y_probs, metrics: dict):
    y_pred = np.argmax(y_probs, axis=1).tolist()
    acc    = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 62)
    print("  CN2VF  –  METRICS REPORT")
    print("=" * 62)
    print(f"  Overall Accuracy : {acc * 100:.2f}%")
    print("-" * 62)
    print(f"  {'Metric':<20}  {'Value':>10}  {'Description'}")
    print("-" * 62)
    print(f"  {'Precision':<20}  {metrics['precision']:>10.4f}  macro @ conf >= 0.50")
    print(f"  {'Recall':<20}  {metrics['recall']:>10.4f}  macro @ conf >= 0.50")
    print(f"  {'Macro F1':<20}  {metrics['macro_f1']:>10.4f}  mean class F1 @ conf >= 0.50")
    print(f"  {'Macro AP':<20}  {metrics['mean_ap']:>10.4f}  mean one-vs-rest PR area")
    print("-" * 62)
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<10}  {'Precision':>10}  {'Recall':>10}  {'AP':>10}")
    print(f"  {'-'*46}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10}  {metrics['per_class_p'][i]:>10.4f}  "
              f"{metrics['per_class_r'][i]:>10.4f}  "
              f"{metrics['per_class_ap'][i]:>10.4f}")
    print("=" * 62)

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                digits=4, zero_division=0))


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main(checkpoint: str, data_root: str, split: str = "test"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load model
    model = load_model(checkpoint, device)

    # Evaluate the already-separated held-out split; do not rebuild a split
    # from train+validation samples.
    val_ds = FireSmokeDataset(
        root_dir=data_root,
        split=split,
        img_size=IMG_SIZE,
        preserve_aspect=getattr(model, "preprocessing_type", "stretch") == "letterbox-v1",
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    print(f"{split.capitalize()} samples : {len(val_ds)}")

    # Run inference
    print(f"\nRunning inference on {split} set …")
    y_true, y_probs = run_inference(model, val_loader, device)
    y_pred = np.argmax(y_probs, axis=1).tolist()

    # Compute all metrics
    print("Computing metrics …")
    metrics = compute_all_metrics(y_true, y_probs)

    # Console report
    print_summary(y_true, y_probs, metrics)

    # Plots
    print(f"\nGenerating graphs -> {OUT_DIR}/")
    plot_metrics_summary(metrics, OUT_DIR)
    plot_pr_curves(metrics, OUT_DIR)
    plot_confusion_matrix(y_true, y_pred, OUT_DIR)
    plot_per_class_bars(y_true, y_pred, OUT_DIR)
    plot_metrics_by_threshold(y_true, y_probs, metrics, OUT_DIR)

    print(f"\n[OK] All graphs saved to: {OUT_DIR.resolve()}")
    print(f"\n  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  Macro AP   : {metrics['mean_ap']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate classification metrics for CN2VF-Net"
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to best_model.pth (default: newest training run)")
    parser.add_argument("--data-root",  default=DATA_ROOT,
                        help="Dataset root directory")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Evaluate validation during development or held-out test for final reporting")
    args = parser.parse_args()
    checkpoint = args.checkpoint or str(latest_checkpoint())
    print(f"Using checkpoint: {checkpoint}")
    main(checkpoint, args.data_root, args.split)
