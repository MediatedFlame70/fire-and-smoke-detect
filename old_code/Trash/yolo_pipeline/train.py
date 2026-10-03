from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from .utils import ensure_dir, save_json


def _metric_from_results(val_results) -> dict[str, float]:
    return {
        "map50": float(getattr(val_results.box, "map50", 0.0)),
        "map50_95": float(getattr(val_results.box, "map", 0.0)),
        "precision": float(getattr(val_results.box, "mp", 0.0)),
        "recall": float(getattr(val_results.box, "mr", 0.0)),
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def train_model(
    data_yaml: Path,
    model_name: str = "yolov8m.pt",
    output_root: Path | None = None,
    epochs: int = 100,
) -> Path:
    data_yaml = Path(data_yaml).resolve()
    output_root = (output_root or Path("output") / "yolov8").resolve()
    project_dir = ensure_dir(output_root / "runs")

    search_space = [
        {"lr0": 0.0010, "batch": 16, "momentum": 0.90},
        {"lr0": 0.0007, "batch": 16, "momentum": 0.93},
        {"lr0": 0.0012, "batch": 24, "momentum": 0.92},
        {"lr0": 0.0005, "batch": 32, "momentum": 0.94},
    ]

    trial_records: list[dict[str, float | str]] = []
    best_score = -1.0
    best_weights: Path | None = None

    for idx, cfg in enumerate(search_space):
        run_name = f"trial_{idx+1}"
        model = YOLO(model_name)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=cfg["batch"],
            optimizer="AdamW",
            lr0=cfg["lr0"],
            lrf=0.01,
            momentum=cfg["momentum"],
            weight_decay=0.0005,
            label_smoothing=0.1,
            cos_lr=True,
            patience=20,
            amp=True,
            dropout=0.10,
            mosaic=1.0,
            mixup=0.20,
            hsv_h=0.03,
            hsv_s=0.75,
            hsv_v=0.45,
            degrees=0.0,
            translate=0.10,
            scale=0.65,
            fliplr=0.50,
            perspective=0.0,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            workers=4,
            pretrained=True,
            close_mosaic=10,
            verbose=True,
        )

        val_metrics = model.val(data=str(data_yaml), split="val", imgsz=640)
        metrics = _metric_from_results(val_metrics)
        metrics["f1"] = _f1(metrics["precision"], metrics["recall"])

        score = 0.5 * metrics["map50"] + 0.35 * metrics["map50_95"] + 0.15 * metrics["f1"]
        trial_record = {
            "run_name": run_name,
            "score": score,
            **cfg,
            **metrics,
        }
        trial_records.append(trial_record)

        run_dir = project_dir / run_name
        weights_path = run_dir / "weights" / "best.pt"
        if score > best_score and weights_path.exists():
            best_score = score
            best_weights = weights_path

    if best_weights is None:
        raise RuntimeError("Training completed but no best weights were found")

    best_model_dir = ensure_dir(output_root / "best_model")
    final_best = best_model_dir / "best.pt"
    final_best.write_bytes(best_weights.read_bytes())

    pd.DataFrame(trial_records).sort_values("score", ascending=False).to_csv(
        output_root / "tuning_results.csv", index=False
    )
    save_json(output_root / "tuning_results.json", {"trials": trial_records, "best": str(final_best)})

    return final_best
