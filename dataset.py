"""
Dataset loader for the fire/smoke detection dataset.

The workspace dataset is stored with one normalized text-label file per image:

dataset_root/
    train/
        images/
        labels/
    val/
        images/
        labels/

Each label file contains one or more rows in the form:

class_id x_center y_center width height

All coordinates are normalized to the range [0, 1].
"""

import json
import random
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from preprocessing import letterbox_bbox, letterbox_image


class FireSmokeDataset(Dataset):
    """Dataset loader that supports normalized text labels and legacy JSON annotations."""

    CLASS_NAMES = ["Fire", "Smoke", "Both", "None"]
    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_MAPPING = {"fire": 0, "smoke": 1, "both": 2, "none": 3, "neutral": 3}
    OBJECT_ALIASES = {
        "fire": "fire", "flame": "fire", "smoke": "smoke",
        "both": "both", "fire_smoke": "both", "fire+smoke": "both",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 448,
        annotation_file: Optional[str] = None,
        transforms: Optional[T.Compose] = None,
        preserve_aspect: bool = True,
        augment: Optional[bool] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.preserve_aspect = preserve_aspect
        self.augment = split == "train" if augment is None else bool(augment)
        self.object_class_names = self._load_object_class_names()

        self.img_dir = self._resolve_image_dir()
        self.label_dir = self._resolve_label_dir()

        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms

        if annotation_file:
            self.samples = self._load_json_samples(Path(annotation_file))
        elif self.label_dir is not None:
            self.samples = self._load_label_samples(self.label_dir)
        else:
            json_path = self.root_dir / "annotations" / f"{split}.json"
            self.samples = self._load_json_samples(json_path)

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_object_class_names(self) -> Dict[int, str]:
        """Read common data.yaml names forms; default to fire=0, smoke=1."""
        mapping = {0: "fire", 1: "smoke"}
        if self.root_dir.name.lower().replace("-", "_") == "fire_smoke":
            mapping[2] = "none"
        candidates = [
            self.root_dir / "data.yaml",
            self.root_dir / "data.yml",
            self.root_dir / "dataset.yaml",
            self.root_dir / "dataset.yml",
        ]
        yaml_path = next((path for path in candidates if path.is_file()), None)
        if yaml_path is None:
            return mapping

        text = yaml_path.read_text(encoding="utf-8")
        names_match = re.search(r"(?ms)^\s*names\s*:\s*(.*?)(?=^\S|\Z)", text)
        if not names_match:
            return mapping

        names_block = names_match.group(1).strip()
        inline_map = re.search(r"\{(.*?)\}", names_block, flags=re.S)
        if inline_map:
            entries = re.findall(
                r"['\"]?(\d+)['\"]?\s*:\s*['\"]?([A-Za-z_+ -]+)['\"]?",
                inline_map.group(1),
            )
            if entries:
                return {int(index): name.strip().lower() for index, name in entries}

        inline = re.search(r"\[(.*?)\]", names_block, flags=re.S)
        if inline:
            names = [part.strip().strip("'\" ") for part in inline.group(1).split(",")]
            return {index: name.lower() for index, name in enumerate(names) if name}

        entries = re.findall(r"(?m)^\s*(\d+)\s*:\s*['\"]?([A-Za-z_ -]+)['\"]?\s*$", names_block)
        if entries:
            return {int(index): name.strip().lower() for index, name in entries}

        entries = re.findall(r"(?m)^\s*-\s*['\"]?([A-Za-z_ -]+)['\"]?\s*$", names_block)
        if entries:
            return {index: name.strip().lower() for index, name in enumerate(entries)}
        return mapping

    def _resolve_image_dir(self) -> Path:
        split_names = [self.split, "valid"] if self.split == "val" else [self.split]
        candidates = [
            candidate
            for split_name in split_names
            for candidate in (
                self.root_dir / split_name / "images",
                self.root_dir / "images" / split_name,
                self.root_dir / split_name,
            )
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise ValueError(f"Image directory not found for split '{self.split}' under {self.root_dir}")

    def _resolve_label_dir(self) -> Optional[Path]:
        split_names = [self.split, "valid"] if self.split == "val" else [self.split]
        candidates = [
            candidate
            for split_name in split_names
            for candidate in (
                self.root_dir / split_name / "labels",
                self.root_dir / "labels" / split_name,
            )
        ]

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        return None

    def _load_json_samples(self, anno_path: Path) -> List[Dict]:
        if not anno_path.exists():
            raise ValueError(f"Annotation file not found: {anno_path}")

        with open(anno_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if isinstance(data, list):
            records = data
        elif "annotations" in data:
            records = data["annotations"]
        elif "images" in data:
            records = data["images"]
        else:
            raise ValueError(f"Unknown annotation format in {anno_path}")

        samples: List[Dict] = []
        for record in records:
            filename = record.get("filename") or record.get("file_name") or record.get("image")
            if not filename:
                continue

            image_path = self.img_dir / filename
            if not image_path.exists():
                image_path = self.img_dir.parent / filename
            if not image_path.exists():
                continue

            samples.append(
                {
                    "source": "json",
                    "filename": filename,
                    "image_path": image_path,
                    "record": record,
                }
            )

        return samples

    def _load_label_samples(self, label_dir: Path) -> List[Dict]:
        samples: List[Dict] = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path for path in self.img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(
                    f"Missing label file for {image_path.name}: {label_path}. "
                    "Use an empty .txt file for a confirmed no-fire/no-smoke image; "
                    "do not leave missing annotations ambiguous."
                )
            objects = self._parse_label_file(label_path)
            samples.append(
                {
                    "source": "text_label",
                    "filename": image_path.name,
                    "image_path": image_path,
                    "label_path": label_path,
                    "objects": objects,
                }
            )

        if not samples:
            raise ValueError(f"No images found in {self.img_dir}")

        return samples

    def _find_image_for_label(self, label_stem: str) -> Optional[Path]:
        extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

        for extension in extensions:
            image_path = self.img_dir / f"{label_stem}{extension}"
            if image_path.exists():
                return image_path

        return None

    def _parse_label_file(self, label_path: Path) -> List[Dict]:
        objects: List[Dict] = []

        with open(label_path, "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                except ValueError:
                    continue

                if class_id not in self.object_class_names:
                    raise ValueError(
                        f"Class id {class_id} in {label_path} has no mapping in data.yaml. "
                        "Add a names mapping for each object class (fire/smoke)."
                    )
                class_name = self.object_class_names[class_id].strip().lower().replace(" ", "_")
                class_name = self.OBJECT_ALIASES.get(class_name, class_name)
                if class_name not in {"fire", "smoke", "both", "none", "neutral"}:
                    raise ValueError(
                        f"Class id {class_id} maps to unsupported name {class_name!r} in "
                        f"{label_path}; expected fire or smoke."
                    )
                objects.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox": bbox,
                    }
                )

        return objects

    @staticmethod
    def _union_boxes(boxes: List[List[float]]) -> List[float]:
        """Return one enclosing normalized [cx, cy, w, h] box for a class."""
        x_min = min(max(0.0, cx - width / 2) for cx, cy, width, height in boxes)
        y_min = min(max(0.0, cy - height / 2) for cx, cy, width, height in boxes)
        x_max = max(min(1.0, cx + width / 2) for cx, cy, width, height in boxes)
        y_max = max(min(1.0, cy + height / 2) for cx, cy, width, height in boxes)
        return [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]

    def _objects_for_sample(self, sample: Dict, orig_w: int, orig_h: int):
        """Return normalized fire/smoke boxes for text-label or JSON samples."""
        raw_objects = []
        if sample["source"] == "text_label":
            raw_objects = sample["objects"]
        else:
            record = sample["record"]
            annotations = record.get("objects") or record.get("annotations")
            if isinstance(annotations, list):
                raw_objects = annotations
            else:
                label = record.get("class", record.get("category", record.get("label", "none")))
                bbox = record.get("bbox", record.get("bounding_box"))
                if bbox is not None:
                    raw_objects = [{"class_name": label, "bbox": bbox}]

        boxes = {"fire": [], "smoke": []}
        for obj in raw_objects:
            if "class_name" in obj:
                name = str(obj["class_name"]).lower()
            elif "class_id" in obj:
                try:
                    class_id = int(obj["class_id"])
                    name = self.object_class_names.get(class_id, self.CLASS_NAMES[class_id].lower()
                                                       if 0 <= class_id < self.NUM_CLASSES else "")
                except (TypeError, ValueError):
                    name = ""
            else:
                name = str(obj.get("class", obj.get("category", obj.get("label", "")))).lower()
            name = name.strip().lower().replace(" ", "_")
            name = self.OBJECT_ALIASES.get(name, name)
            if name.startswith("class_"):
                try:
                    name = self.object_class_names.get(int(name.split("_", 1)[1]), name)
                except ValueError:
                    pass
            if name in {"fire", "smoke"}:
                targets = [name]
            elif name == "both":
                targets = ["fire", "smoke"]
            else:
                continue

            bbox = obj.get("bbox", obj.get("bounding_box"))
            if bbox is None:
                continue
            box = self._normalize_bbox(list(bbox), orig_w, orig_h).tolist()
            for target in targets:
                boxes[target].append(box)

        return boxes

    def sample_class_label(self, sample: Dict) -> int:
        """Derive the mutually exclusive image class from object presence."""
        if sample["source"] == "text_label":
            names = {obj["class_name"] for obj in sample["objects"]}
            fire = "fire" in names or "both" in names
            smoke = "smoke" in names or "both" in names
        else:
            record = sample["record"]
            # Avoid opening image files when classifying the metadata for weights.
            label = str(record.get("class", record.get("category", record.get("label", "none")))).lower()
            if label in {"fire", "smoke", "both", "none", "neutral"}:
                return self.CLASS_MAPPING[label]
            if label.isdigit() and int(label) in range(self.NUM_CLASSES):
                return int(label)
            raw = record.get("objects") or record.get("annotations") or []
            names = set()
            for obj in raw:
                if not isinstance(obj, dict):
                    continue
                if "class_name" in obj:
                    raw_name = str(obj["class_name"]).lower().strip().replace(" ", "_")
                    names.add(self.OBJECT_ALIASES.get(raw_name, raw_name))
                elif "class_id" in obj:
                    try:
                        object_id = int(obj["class_id"])
                        names.add(self.object_class_names.get(
                            object_id,
                            self.CLASS_NAMES[object_id].lower() if object_id in range(self.NUM_CLASSES) else "",
                        ))
                    except (TypeError, ValueError):
                        pass
                else:
                    raw_name = str(obj.get("class", obj.get("category", ""))).lower().strip().replace(" ", "_")
                    names.add(self.OBJECT_ALIASES.get(raw_name, raw_name))
            fire = "fire" in names or "both" in names
            smoke = "smoke" in names or "both" in names
        return 2 if fire and smoke else 0 if fire else 1 if smoke else 3
    
    def _default_transforms(self) -> T.Compose:
        """Default image preprocessing pipeline with advanced hard-negative & color augmentations."""
        if self.augment:
            return T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomAutocontrast(p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _normalize_bbox(self, bbox: List[float], orig_w: int, orig_h: int) -> torch.Tensor:
        if all(0.0 <= value <= 1.0 for value in bbox):
            return torch.tensor(bbox, dtype=torch.float32)

        x, y, w, h = bbox
        x_norm = max(0.0, min(1.0, x / orig_w))
        y_norm = max(0.0, min(1.0, y / orig_h))
        w_norm = max(0.0, min(1.0, w / orig_w))
        h_norm = max(0.0, min(1.0, h / orig_h))

        return torch.tensor([x_norm, y_norm, w_norm, h_norm], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "image": Tensor [3, img_size, img_size],
                "class_label": Tensor (Fire=0, Smoke=1, Both=2, None=3),
                "fire_bbox": Tensor [4] normalized [cx, cy, w, h],
                "smoke_bbox": Tensor [4] normalized [cx, cy, w, h],
                "fire_present" / "smoke_present": boolean tensors,
                "filename": str
            }
        """
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        object_boxes = self._objects_for_sample(sample, orig_w, orig_h)
        fire_present = bool(object_boxes["fire"])
        smoke_present = bool(object_boxes["smoke"])
        class_label = 2 if fire_present and smoke_present else 0 if fire_present else 1 if smoke_present else 3
        fire_bbox = self._union_boxes(object_boxes["fire"]) if fire_present else [0.0] * 4
        smoke_bbox = self._union_boxes(object_boxes["smoke"]) if smoke_present else [0.0] * 4

        if self.preserve_aspect:
            image = letterbox_image(image, size=self.img_size)
            if fire_present:
                fire_bbox = letterbox_bbox(fire_bbox, orig_w, orig_h, self.img_size)
            if smoke_present:
                smoke_bbox = letterbox_bbox(smoke_bbox, orig_w, orig_h, self.img_size)
        else:
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize((self.img_size, self.img_size), resampling)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            if fire_present:
                fire_bbox[0] = 1.0 - fire_bbox[0]
            if smoke_present:
                smoke_bbox[0] = 1.0 - smoke_bbox[0]

        image_tensor = self.transforms(image)

        return {
            "image": image_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "fire_bbox": torch.tensor(fire_bbox, dtype=torch.float32),
            "smoke_bbox": torch.tensor(smoke_bbox, dtype=torch.float32),
            "fire_present": torch.tensor(fire_present, dtype=torch.bool),
            "smoke_present": torch.tensor(smoke_present, dtype=torch.bool),
            "filename": sample["filename"],
        }


class CombinedFireSmokeDataset(Dataset):
    """Combine multiple annotated roots while reserving validation data safely.

    Roots that have no validation split contribute a deterministic 10% holdout
    from their training samples. Their test split is never used for training or
    validation.
    """

    def __init__(self, roots: List[str], split: str, img_size: int = 448):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if not roots:
            raise ValueError("At least one dataset root is required")

        self.split = split
        self.datasets = []
        self.samples = []
        self._sample_refs = []

        for root in roots:
            root_path = Path(root)
            has_val = self._has_split(root_path, "val")
            if split == "test":
                if not self._has_split(root_path, "test"):
                    raise ValueError(f"Held-out test split not found under {root_path}")
                source = FireSmokeDataset(root, "test", img_size=img_size, augment=False)
                selected = range(len(source))
            elif split == "val" and has_val:
                source = FireSmokeDataset(root, "val", img_size=img_size, augment=False)
                selected = range(len(source))
            elif split == "train":
                source = FireSmokeDataset(root, "train", img_size=img_size, augment=True)
                if has_val:
                    selected = range(len(source))
                else:
                    train_indices, _ = self._partition(source)
                    selected = train_indices
            else:
                # Validation set absent: use only a fixed subset of train, with
                # augmentation disabled so model selection remains repeatable.
                source = FireSmokeDataset(root, "train", img_size=img_size, augment=False)
                _, selected = self._partition(source)

            dataset_index = len(self.datasets)
            self.datasets.append(source)
            for sample_index in selected:
                self._sample_refs.append((dataset_index, sample_index))
                self.samples.append((dataset_index, source.samples[sample_index]))

        if not self._sample_refs:
            raise ValueError(f"No samples found for combined {split} split")

    @staticmethod
    def _has_split(root: Path, split: str) -> bool:
        split_names = [split, "valid"] if split == "val" else [split]
        return any(
            (root / name / "images").is_dir()
            or (root / "images" / name).is_dir()
            for name in split_names
        )

    @staticmethod
    def _partition(dataset, fraction: float = 0.10, seed: int = 2026):
        """Create a repeatable class-stratified train/validation partition."""
        by_class = {}
        for index, sample in enumerate(dataset.samples):
            label = dataset.sample_class_label(sample)
            by_class.setdefault(label, []).append(index)

        train_indices, val_indices = [], []
        rng = random.Random(seed)
        for indices in by_class.values():
            indices = list(indices)
            rng.shuffle(indices)
            heldout_count = max(1, round(len(indices) * fraction)) if len(indices) > 1 else 0
            val_indices.extend(indices[:heldout_count])
            train_indices.extend(indices[heldout_count:])
        return sorted(train_indices), sorted(val_indices)

    def sample_class_label(self, sample):
        dataset_index, underlying_sample = sample
        return self.datasets[dataset_index].sample_class_label(underlying_sample)

    def __len__(self):
        return len(self._sample_refs)

    def __getitem__(self, index):
        dataset_index, sample_index = self._sample_refs[index]
        return self.datasets[dataset_index][sample_index]


def create_dataloaders(
    data_roots: List[str],
    batch_size: int = 8,
    num_workers: int = 2,
    img_size: int = 448,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_roots: Dataset root directories to combine
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        img_size: Image size (default 448)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = CombinedFireSmokeDataset(data_roots, "train", img_size=img_size)
    val_dataset = CombinedFireSmokeDataset(data_roots, "val", img_size=img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_dataset>")
        print("Example: python dataset.py ./fire_smoke_data")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    try:
        train_loader, val_loader = create_dataloaders([data_root], batch_size=4)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test first batch
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Class labels: {batch['class_label']}")
        print(f"  Fire bbox shape: {batch['fire_bbox'].shape}")
        print(f"  Smoke bbox shape: {batch['smoke_bbox'].shape}")
        print(f"  Image class labels: {batch['class_label']}")
        
        print("\n✓ Dataset loading test passed!")
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
