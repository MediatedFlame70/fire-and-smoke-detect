from pathlib import Path

from yolo_pipeline import evaluate_model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    weights = root / "output" / "yolov8" / "best_model" / "best.pt"
    data_yaml = root / "output" / "yolov8" / "prepared_data" / "dataset.yaml"
    report = evaluate_model(weights=weights, data_yaml=data_yaml)
    print(f"Evaluation report: {report}")
