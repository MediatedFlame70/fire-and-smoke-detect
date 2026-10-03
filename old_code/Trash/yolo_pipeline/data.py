from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .utils import ensure_dir, save_json

RANDOM_SEED = 42


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    boxes: list[tuple[int, float, float, float, float]]
    dominant_class: int


def _read_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        boxes.append((cls_id, x, y, w, h))
    return boxes


def _collect_raw_samples(dataset_root: Path) -> list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]]:
    samples: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []
    for split_dir in sorted(dataset_root.glob("*")):
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in images_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = _read_label_file(label_path)
            if not boxes:
                continue
            samples.append((image_path, label_path, boxes))
    return samples


def _infer_binary_mapping(samples: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]]) -> dict[int, int]:
    id_votes: dict[int, Counter[str]] = defaultdict(Counter)
    for image_path, _, boxes in samples:
        lower_name = image_path.name.lower()
        has_fire_hint = "fire" in lower_name
        has_smoke_hint = "smoke" in lower_name
        for cls_id, *_ in boxes:
            if has_fire_hint and not has_smoke_hint:
                id_votes[cls_id]["fire"] += 1
            elif has_smoke_hint and not has_fire_hint:
                id_votes[cls_id]["smoke"] += 1

    mapping: dict[int, int] = {}
    observed_ids = sorted({box[0] for _, _, boxes in samples for box in boxes})
    for cls_id in observed_ids:
        smoke_votes = id_votes[cls_id]["smoke"]
        fire_votes = id_votes[cls_id]["fire"]
        if smoke_votes > fire_votes:
            mapping[cls_id] = 1
        else:
            mapping[cls_id] = 0

    # Ensure both classes exist in mapping; fallback to min id = fire, max id = smoke.
    if len(set(mapping.values())) == 1 and len(observed_ids) > 1:
        mapping[min(observed_ids)] = 0
        mapping[max(observed_ids)] = 1
    return mapping


def _make_visualization(image_path: Path, boxes: list[tuple[int, float, float, float, float]], out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w = image.shape[:2]
    color_map = {0: (0, 0, 255), 1: (255, 0, 0)}
    name_map = {0: "Fire", 1: "Smoke"}

    for cls_id, x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        color = color_map.get(cls_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, name_map.get(cls_id, str(cls_id)), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite(str(out_path), image)


def _save_sample(out_images_dir: Path, out_labels_dir: Path, sample: Sample) -> None:
    dst_image = out_images_dir / sample.image_path.name
    dst_label = out_labels_dir / f"{sample.image_path.stem}.txt"
    shutil.copy2(sample.image_path, dst_image)

    lines = [f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls_id, x, y, w, h in sample.boxes]
    dst_label.write_text("\n".join(lines), encoding="utf-8")


def _augment_and_oversample_smoke(train_images: Path, train_labels: Path, max_factor: int = 2) -> int:
    created = 0
    label_files = sorted(train_labels.glob("*.txt"))
    smoke_files: list[Path] = []
    for label_file in label_files:
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()
        has_smoke = any(line.split()[0] == "1" for line in lines if line.strip())
        if has_smoke:
            smoke_files.append(label_file)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(smoke_files)

    for label_file in smoke_files:
        stem = label_file.stem
        source_image = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = train_images / f"{stem}{ext}"
            if candidate.exists():
                source_image = candidate
                break
        if source_image is None:
            continue

        image = cv2.imread(str(source_image))
        if image is None:
            continue

        for idx in range(max_factor):
            alpha = rng.uniform(0.85, 1.2)
            beta = rng.randint(-15, 20)
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            if idx % 2 == 0:
                adjusted = cv2.GaussianBlur(adjusted, (5, 5), 0)

            out_stem = f"{stem}_smkaug{idx}"
            out_image = train_images / f"{out_stem}{source_image.suffix}"
            out_label = train_labels / f"{out_stem}.txt"
            cv2.imwrite(str(out_image), adjusted)
            shutil.copy2(label_file, out_label)
            created += 1

    return created


def load_data(
    dataset_root: Path,
    output_root: Path | None = None,
    train_ratio: float = 0.8,
) -> Path:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset_root = dataset_root.resolve()
    output_root = (output_root or Path("output") / "yolov8").resolve()
    prepared_root = ensure_dir(output_root / "prepared_data")

    raw_samples = _collect_raw_samples(dataset_root)
    if not raw_samples:
        raise RuntimeError(f"No labeled samples found in {dataset_root}")

    class_mapping = _infer_binary_mapping(raw_samples)

    clean_samples: list[Sample] = []
    dropped_for_mapping = 0
    for image_path, label_path, boxes in raw_samples:
        remapped: list[tuple[int, float, float, float, float]] = []
        for cls_id, x, y, w, h in boxes:
            if cls_id not in class_mapping:
                continue
            mapped_cls = class_mapping[cls_id]
            remapped.append((mapped_cls, x, y, w, h))

        if not remapped:
            dropped_for_mapping += 1
            continue

        class_counts = Counter([b[0] for b in remapped])
        dominant = 1 if class_counts[1] >= class_counts[0] else 0
        clean_samples.append(Sample(image_path=image_path, label_path=label_path, boxes=remapped, dominant_class=dominant))

    if len(clean_samples) < 20:
        raise RuntimeError("Too few clean samples after remapping labels")

    stratify_labels = [s.dominant_class for s in clean_samples]
    train_samples, val_samples = train_test_split(
        clean_samples,
        train_size=train_ratio,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        out_images = ensure_dir(prepared_root / split_name / "images")
        out_labels = ensure_dir(prepared_root / split_name / "labels")
        for sample in split_samples:
            _save_sample(out_images, out_labels, sample)

    oversampled = _augment_and_oversample_smoke(
        train_images=prepared_root / "train" / "images",
        train_labels=prepared_root / "train" / "labels",
        max_factor=2,
    )

    # Save bbox visual checks.
    vis_dir = ensure_dir(output_root / "verification" / "bbox_samples")
    for i, sample in enumerate(random.sample(clean_samples, k=min(24, len(clean_samples)))):
        _make_visualization(sample.image_path, sample.boxes, vis_dir / f"sample_{i:03d}.jpg")

    dataset_yaml = {
        "path": str(prepared_root),
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": ["fire", "smoke"],
    }
    yaml_path = prepared_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "raw_samples": len(raw_samples),
        "clean_samples": len(clean_samples),
        "dropped_for_mapping": dropped_for_mapping,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "oversampled_smoke_samples": oversampled,
        "mapping": {str(k): int(v) for k, v in class_mapping.items()},
    }
    save_json(output_root / "verification" / "data_summary.json", summary)

    return yaml_path
