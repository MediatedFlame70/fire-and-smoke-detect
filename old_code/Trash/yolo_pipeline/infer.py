from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .utils import ensure_dir


def predict(weights: Path, source: str, output_root: Path | None = None, conf: float = 0.25) -> Path:
    weights = Path(weights).resolve()
    output_root = (output_root or Path("output") / "yolov8").resolve()
    pred_dir = ensure_dir(output_root / "predictions")

    model = YOLO(str(weights))
    model.predict(
        source=source,
        conf=conf,
        imgsz=640,
        save=True,
        project=str(pred_dir),
        name="inference",
        exist_ok=True,
        verbose=True,
    )
    return pred_dir / "inference"
