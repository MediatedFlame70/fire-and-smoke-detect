from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

from .utils import ensure_dir, save_json


def _read_gt_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, x, y, w, h = parts
        boxes.append((int(c), float(x), float(y), float(w), float(h)))
    return boxes


def _to_xyxy(box: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, int, int, int, int]:
    c, x, y, w, h = box
    x1 = int((x - w / 2.0) * width)
    y1 = int((y - h / 2.0) * height)
    x2 = int((x + w / 2.0) * width)
    y2 = int((y + h / 2.0) * height)
    return c, x1, y1, x2, y2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _plot_confusion_matrix(y_true: list[int], y_pred: list[int], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Fire", "Smoke"], yticklabels=["Fire", "Smoke"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (Matched Detections)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_training_curves(run_dir: Path, out_dir: Path) -> None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    numeric = [c for c in df.columns if c != "epoch" and np.issubdtype(df[c].dtype, np.number)]
    if not numeric:
        return

    plt.figure(figsize=(12, 7))
    for col in numeric:
        if "loss" in col or "mAP" in col or "precision" in col or "recall" in col:
            plt.plot(df["epoch"], df[col], label=col)
    plt.legend(loc="best", fontsize=8)
    plt.title("YOLOv8 Training Curves")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png")
    plt.close()


def _write_metrics_summary(eval_dir: Path, summary: dict[str, float | int | str | dict[str, int]]) -> None:
    lines = [
        "# YOLOv8 Evaluation Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {float(summary['precision']):.4f} |",
        f"| Recall | {float(summary['recall']):.4f} |",
        f"| mAP50 | {float(summary['mAP50']):.4f} |",
        f"| mAP50-95 | {float(summary['mAP50-95']):.4f} |",
        "",
        f"Matched pairs: {int(summary['matched_pairs'])}",
        f"Smoke/fire confusions: {summary['smoke_fire_confusions']}",
    ]
    (eval_dir / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _print_metrics_summary(summary: dict[str, float | int | str | dict[str, int]]) -> None:
    print("\nYOLOv8 evaluation metrics")
    print(f"  Precision:  {float(summary['precision']):.4f}")
    print(f"  Recall:     {float(summary['recall']):.4f}")
    print(f"  mAP50:      {float(summary['mAP50']):.4f}")
    print(f"  mAP50-95:   {float(summary['mAP50-95']):.4f}")
    print(f"  Matched pairs: {int(summary['matched_pairs'])}")


def evaluate_model(weights: Path, data_yaml: Path, output_root: Path | None = None, conf: float = 0.25) -> Path:
    weights = Path(weights).resolve()
    data_yaml = Path(data_yaml).resolve()
    output_root = (output_root or Path("output") / "yolov8").resolve()

    eval_dir = ensure_dir(output_root / "evaluation")
    errors_dir = ensure_dir(eval_dir / "error_analysis")

    model = YOLO(str(weights))
    val_metrics = model.val(data=str(data_yaml), split="val", imgsz=640)

    # Resolve val image directory from yaml.
    import yaml

    yaml_obj = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(yaml_obj["path"])
    val_images = (base / yaml_obj["val"]).resolve()
    val_labels = val_images.parent.parent / "val" / "labels"
    if not val_labels.exists():
        val_labels = base / "val" / "labels"

    y_true: list[int] = []
    y_pred: list[int] = []
    confusion_cases: dict[str, int] = defaultdict(int)

    for image_path in val_images.glob("*"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        gt_boxes = [_to_xyxy(b, w, h) for b in _read_gt_boxes(val_labels / f"{image_path.stem}.txt")]
        pred = model.predict(source=str(image_path), conf=conf, verbose=False)[0]

        preds: list[tuple[int, int, int, int, int, float]] = []
        for b in pred.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item())
            preds.append((cls_id, x1, y1, x2, y2, score))

        used = set()
        for gt_cls, gx1, gy1, gx2, gy2 in gt_boxes:
            best_idx = -1
            best_iou = 0.0
            for i, (pcls, px1, py1, px2, py2, _) in enumerate(preds):
                if i in used:
                    continue
                iou = _iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= 0.3:
                pcls, px1, py1, px2, py2, pconf = preds[best_idx]
                used.add(best_idx)
                y_true.append(gt_cls)
                y_pred.append(pcls)

                if gt_cls != pcls and {gt_cls, pcls} == {0, 1}:
                    confusion_cases[f"{gt_cls}->{pcls}"] += 1
                    vis = image.copy()
                    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(
                        vis,
                        f"GT:{gt_cls} P:{pcls} conf:{pconf:.2f}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    out_name = f"{image_path.stem}_gt{gt_cls}_pred{pcls}.jpg"
                    cv2.imwrite(str(errors_dir / out_name), vis)

    if y_true and y_pred:
        _plot_confusion_matrix(y_true, y_pred, eval_dir / "confusion_matrix.png")
        report_text = classification_report(y_true, y_pred, target_names=["Fire", "Smoke"], digits=4)
    else:
        report_text = "No matched detections available for confusion matrix."

    report_path = eval_dir / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    run_dir = output_root / "runs"
    trial_dirs = sorted([d for d in run_dir.glob("trial_*") if d.is_dir()])
    if trial_dirs:
        latest_trial = trial_dirs[-1]
        _plot_training_curves(latest_trial, eval_dir)

    summary = {
        "weights": str(weights),
        "map50": float(getattr(val_metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(val_metrics.box, "map", 0.0)),
        "mAP50": float(getattr(val_metrics.box, "map50", 0.0)),
        "mAP50-95": float(getattr(val_metrics.box, "map", 0.0)),
        "precision": float(getattr(val_metrics.box, "mp", 0.0)),
        "recall": float(getattr(val_metrics.box, "mr", 0.0)),
        "matched_pairs": len(y_true),
        "smoke_fire_confusions": dict(confusion_cases),
        "error_images_dir": str(errors_dir),
    }
    save_json(eval_dir / "evaluation_metrics.json", summary)
    _write_metrics_summary(eval_dir, summary)
    _print_metrics_summary(summary)

    return eval_dir
