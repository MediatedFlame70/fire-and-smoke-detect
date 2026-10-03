import argparse
from pathlib import Path

from yolo_pipeline import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Fire/Smoke prediction")
    parser.add_argument("source", type=str, help="Image, directory, video, or webcam source")
    parser.add_argument("--weights", type=str, default="output/yolov8/best_model/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = predict(
        weights=Path(args.weights),
        source=args.source,
        conf=args.conf,
    )
    print(f"Predictions saved to: {output_dir}")
