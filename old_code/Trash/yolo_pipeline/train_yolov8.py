from pathlib import Path

from yolo_pipeline import load_data, train_model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data_yaml = load_data(dataset_root=root / "fire_smoke")
    best_weights = train_model(data_yaml=data_yaml, model_name="yolov8m.pt")
    print(f"Best weights: {best_weights}")
