"""
pipeline_dataset.py
===================
Enhanced dataset handling for CN2VF fire/smoke/neutral classification.

Features:
- Stratified 80/20 train-val split from a flat labeled directory
- Class distribution analysis and imbalance reporting
- Rich augmentation pipeline (rotation, flips, brightness, Gaussian noise)
- Class weight computation for weighted loss / Focal Loss
- Compatible with the existing normalized text-label and folder-based datasets
"""

import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedShuffleSplit


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["Fire", "Smoke", "Neutral"]
CLASS_TO_IDX = {"fire": 0, "smoke": 1, "neutral": 2, "none": 2}
IDX_TO_CLASS = {0: "Fire", 1: "Smoke", 2: "Neutral"}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────
# Gaussian Noise Transform (not in torchvision by default)
# ──────────────────────────────────────────────────────────────
class AddGaussianNoise:
    """Add Gaussian noise to a tensor image. Critical for smoke/neutral disambiguation."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


# ──────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ──────────────────────────────────────────────────────────────
def build_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Rich augmentation for training.
    Gaussian noise is especially important to help the model distinguish
    hazy smoke textures from neutral/clear backgrounds.
    """
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),          # Slightly larger for random crop
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),                         # Rare grayscale helps generalization
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(mean=0.0, std=0.015),             # Smoke boundary robustness
    ])


def build_val_transforms(img_size: int = 224) -> T.Compose:
    """Clean deterministic preprocessing for validation/test."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────
# Sample Discovery
# ──────────────────────────────────────────────────────────────
def discover_samples_folder_layout(data_root: str) -> List[Dict]:
    """
    Discover samples from a folder-per-class layout:

        data_root/
            fire/      *.jpg  *.png  ...
            smoke/     ...
            neutral/   ...
    """
    root = Path(data_root)
    samples = []
    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = root / class_name
        if not class_dir.exists():
            # Try capitalised variant
            class_dir = root / class_name.capitalize()
        if not class_dir.exists():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTENSIONS:
                samples.append({
                    "image_path": img_path,
                    "class_idx": class_idx,
                    "class_name": IDX_TO_CLASS[class_idx],
                })
    return samples


def _case_insensitive_image_find(image_dir: Path, stem: str) -> Optional[Path]:
    """Find image file regardless of extension case (.jpg / .JPG / .png / etc.)."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        # Try lowercase then uppercase
        for variant in [ext, ext.upper()]:
            candidate = image_dir / (stem + variant)
            if candidate.exists():
                return candidate
    return None


def discover_samples_label_layout(data_root: str, split: str = "train") -> List[Dict]:
    """
    Discover samples from the existing text-label layout:

        data_root/
            train/images/  train/labels/
            val/images/    val/labels/
    """
    root = Path(data_root)
    label_dir = root / split / "labels"
    image_dir = root / split / "images"

    if not label_dir.exists():
        return []

    samples = []
    for label_path in sorted(label_dir.glob("*.txt")):
        # Find matching image (case-insensitive extension search)
        img_path = _case_insensitive_image_find(image_dir, label_path.stem)
        if img_path is None:
            continue

        # Parse the label and take the highest-priority class in the file.
        class_idx = _parse_primary_class_label(label_path)
        samples.append({
            "image_path": img_path,
            "class_idx": class_idx,
            "class_name": IDX_TO_CLASS[class_idx],
        })
    return samples


def _parse_primary_class_label(label_path: Path) -> int:
    """Return the highest-priority (Fire > Smoke > Neutral) class in a text label file."""
    priority = {0: 0, 1: 1, 2: 2}  # lower = higher priority
    best = 2  # default neutral
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
                if priority.get(cid, 99) < priority.get(best, 99):
                    best = cid
            except ValueError:
                continue
    return best


# ──────────────────────────────────────────────────────────────
# load_data()
# ──────────────────────────────────────────────────────────────
def load_data(
    data_root: str,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all samples, perform stratified 80/20 split, and report class distribution.

    Priority: folder layout → train split → validation split combined
    Returns:
        train_samples, val_samples  (each a list of dicts with keys:
            image_path, class_idx, class_name)
    """
    # 1. Try folder-per-class layout
    samples = discover_samples_folder_layout(data_root)

    # 2. Fall back to the text-label layout (merge train and validation, then re-split).
    if not samples:
        print("[load_data] Folder layout not found – trying the text-label layout …")
        train_s = discover_samples_label_layout(data_root, split="train")
        val_s   = discover_samples_label_layout(data_root, split="val")
        samples = train_s + val_s

    if not samples:
        raise ValueError(
            f"No image samples found under '{data_root}'.\n"
            "Expected either:\n"
            "  data_root/fire/  data_root/smoke/  data_root/neutral/\n"
            "or text-label layout:  data_root/train/images+labels  data_root/val/images+labels"
        )

    # 3. Stratified split
    labels = [s["class_idx"] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(sss.split(samples, labels))
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

    # 4. Report class distribution
    _report_class_distribution(train_samples, val_samples)

    return train_samples, val_samples


def _report_class_distribution(train: List[Dict], val: List[Dict]):
    """Print class distribution and imbalance ratio."""
    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 55)

    def _counts(samples):
        c = Counter(s["class_name"] for s in samples)
        return {name: c.get(name, 0) for name in CLASS_NAMES}

    tr = _counts(train)
    vl = _counts(val)

    print(f"{'Class':<10} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 40)
    for name in CLASS_NAMES:
        total = tr[name] + vl[name]
        print(f"{name:<10} {tr[name]:>8} {vl[name]:>8} {total:>8}")
    print(f"{'TOTAL':<10} {sum(tr.values()):>8} {sum(vl.values()):>8} {sum(tr.values())+sum(vl.values()):>8}")

    all_counts = [tr[n] + vl[n] for n in CLASS_NAMES]
    if max(all_counts) > 0:
        ratio = max(all_counts) / max(min(all_counts), 1)
        print(f"\nImbalance ratio (max/min): {ratio:.2f}x")
        if ratio > 2.0:
            print("[!] Class imbalance detected - class weights will be applied automatically.")
        else:
            print("[OK] Class distribution is reasonably balanced.")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────────────────────
# Class Weight Computation
# ──────────────────────────────────────────────────────────────
def compute_class_weights(train_samples: List[Dict], device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropy or Focal Loss.
    Formula: w_c = total_samples / (num_classes * count_c)
    """
    counts = Counter(s["class_idx"] for s in train_samples)
    total  = len(train_samples)
    n_cls  = len(CLASS_NAMES)

    weights = []
    for i in range(n_cls):
        cnt = counts.get(i, 1)
        weights.append(total / (n_cls * cnt))

    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Class weights -> Fire: {weights[0]:.3f} | Smoke: {weights[1]:.3f} | Neutral: {weights[2]:.3f}")
    return w_tensor


# ──────────────────────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────────────────────
class FireSmokeClassifyDataset(Dataset):
    """
    Classification-only dataset for CN2VF pipeline.
    Returns image tensor + class label (no bbox needed for classification-focused training).
    """

    def __init__(self, samples: List[Dict], transforms: T.Compose):
        self.samples    = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = self.transforms(img)
        return {
            "image":      img_tensor,
            "class_label": torch.tensor(sample["class_idx"], dtype=torch.long),
            "filename":   sample["image_path"].name,
        }


# ──────────────────────────────────────────────────────────────
# preprocess() / DataLoader factory
# ──────────────────────────────────────────────────────────────
def preprocess(
    train_samples: List[Dict],
    val_samples:   List[Dict],
    img_size:      int = 224,
    batch_size:    int = 32,
    num_workers:   int = 2,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build augmented train DataLoader and clean val DataLoader.

    Args:
        train_samples:        Output of load_data()
        val_samples:          Output of load_data()
        img_size:             Resize target (224 or 256)
        batch_size:           Batch size
        num_workers:          DataLoader workers
        use_weighted_sampler: Oversample minority classes during training

    Returns:
        train_loader, val_loader
    """
    train_ds = FireSmokeClassifyDataset(train_samples, build_train_transforms(img_size))
    val_ds   = FireSmokeClassifyDataset(val_samples,   build_val_transforms(img_size))

    # Optional: WeightedRandomSampler to oversample minority classes
    sampler = None
    if use_weighted_sampler:
        counts  = Counter(s["class_idx"] for s in train_samples)
        sample_weights = [1.0 / counts[s["class_idx"]] for s in train_samples]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(train_samples),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
