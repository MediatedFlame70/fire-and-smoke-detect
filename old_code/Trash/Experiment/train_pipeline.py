"""
train_pipeline.py
=================
Complete, optimised CN2VF training pipeline for Fire / Smoke / Neutral classification.

Usage:
    python train_pipeline.py --data-root ./fire_smoke_data

    # Hyperparameter sweep example:
    python train_pipeline.py --data-root ./fire_smoke_data --tune

Key features implemented:
  ✓ Stratified 80/20 split with class imbalance analysis
  ✓ Rich augmentation (rotation, flips, brightness, Gaussian noise)
  ✓ Focal Loss with class weights (reduces Smoke ↔ Neutral confusion)
  ✓ AdamW optimizer + CosineAnnealingLR / ReduceLROnPlateau
  ✓ Early stopping (patience=10) based on validation macro F1
  ✓ Best model saved on F1 (not accuracy)
  ✓ GridSearch hyperparameter tuning
  ✓ Full sklearn evaluation + confusion matrix + error analysis
  ✓ predict(image) → Fire / Smoke / Neutral + confidence
"""

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ── Project modules ───────────────────────────────────────────
from cn2vf_net import CN2VFNet
from pipeline_dataset import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    compute_class_weights,
    load_data,
    preprocess,
)
from pipeline_evaluate import (
    analyze_errors,
    evaluate,
    plot_confusion_matrix,
    plot_training_curves,
)
from pipeline_losses import CN2VFClassificationLoss, compute_f1_macro


# ═══════════════════════════════════════════════════════════════
# CN2VF Classification Wrapper
# ═══════════════════════════════════════════════════════════════
class CN2VFClassifier(nn.Module):
    """
    Wraps CN2VFNet for pure classification.
    Dropout is inserted between GAP and the final linear layer via
    a clean forward() override (no monkey-patching).
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.backbone = CN2VFNet(num_classes=num_classes)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> Dict:
        # Run all backbone stages up to (but not including) the head's final linear
        x = self.backbone.stem(x)
        c1 = self.backbone.cnn_stage1(x)
        c2 = self.backbone.cnn_stage2(c1)
        c3 = self.backbone.cnn_stage3(c2)

        tokens, h1, w1 = self.backbone.patch_embed(c3)
        t1 = self.backbone.vit_stage1(tokens)
        t2_tokens, h2, w2 = self.backbone.token_down(t1, h1, w1)
        t2_tokens = self.backbone.vit_stage2(t2_tokens)

        b, n, c = t2_tokens.shape
        t2_map = t2_tokens.transpose(1, 2).reshape(b, c, h2, w2)
        fused = self.backbone.fusion(c2, c3, t2_map)

        # Detection head with dropout injected before the linear classifer
        head = self.backbone.head
        feat = head.pre(fused)
        feat = head.pool(feat).flatten(1)    # [B, C]
        feat = self.dropout(feat)            # Dropout BEFORE FC
        cls_logits = head.cls(feat)          # [B, num_classes]
        bbox       = torch.sigmoid(head.box(feat))  # [B, 4]

        return {"cls_logits": cls_logits, "bbox": bbox}

    def get_cls_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return raw classification logits only."""
        return self.forward(x)["cls_logits"]


# ═══════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Monitors validation macro F1-score and stops training when no improvement
    is seen for `patience` epochs. Saves the best model state dict.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_f1    = -1.0
        self.counter    = 0
        self.best_state: Optional[dict] = None
        self.best_epoch = 0

    def step(self, val_f1: float, model_state: dict, epoch: int) -> bool:
        """
        Returns True if training should stop.
        Saves best model state internally.
        """
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1    = val_f1
            self.counter    = 0
            self.best_state = deepcopy(model_state)
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# ═══════════════════════════════════════════════════════════════
# Single Epoch Helpers
# ═══════════════════════════════════════════════════════════════
def _train_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> Tuple[float, float, List[int], List[int]]:
    """Run one training epoch. Returns (avg_loss, accuracy_pct, y_true, y_pred)."""
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(loader, desc=f"  [Train] E{epoch}", leave=False, ncols=90)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["class_label"].to(device)

        optimizer.zero_grad()
        output = model(images)
        logits = output["cls_logits"] if isinstance(output, dict) else output

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    avg_loss = total_loss / len(loader)
    acc_pct  = 100.0 * correct / max(total, 1)
    return avg_loss, acc_pct, y_true, y_pred


@torch.no_grad()
def _val_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> Tuple[float, float, float, List[int], List[int]]:
    """Run one validation pass. Returns (avg_loss, accuracy_pct, macro_f1, y_true, y_pred)."""
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(loader, desc=f"  [Val]   E{epoch}", leave=False, ncols=90)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["class_label"].to(device)

        output = model(images)
        logits = output["cls_logits"] if isinstance(output, dict) else output

        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss  = total_loss / len(loader)
    acc_pct   = 100.0 * correct / max(total, 1)
    macro_f1  = compute_f1_macro(y_true, y_pred)
    return avg_loss, acc_pct, macro_f1, y_true, y_pred


# ═══════════════════════════════════════════════════════════════
# train()
# ═══════════════════════════════════════════════════════════════
def train(
    data_root:       str,
    output_dir:      str      = "./output/pipeline",
    img_size:        int      = 224,
    batch_size:      int      = 32,
    lr:              float    = 1e-3,
    weight_decay:    float    = 1e-4,
    dropout:         float    = 0.4,
    epochs:          int      = 100,
    patience:        int      = 10,
    use_focal:       bool     = True,
    focal_gamma:     float    = 2.0,
    scheduler_type:  str      = "cosine",   # "cosine" | "plateau"
    num_workers:     int      = 2,
    seed:            int      = 42,
    run_tag:         str      = "",
) -> Dict:
    """
    Full training run. Returns dict with best val F1 and output path.

    Args:
        data_root:       Path to dataset root
        output_dir:      Where to save checkpoints/plots
        img_size:        224 or 256
        batch_size:      16 or 32
        lr:              Initial learning rate (1e-3, 1e-4, 1e-5)
        weight_decay:    AdamW weight decay
        dropout:         Dropout before classification head (0.3–0.5)
        epochs:          Max epochs (early stopping may terminate sooner)
        patience:        Early stopping patience (epochs without F1 improvement)
        use_focal:       True → Focal Loss | False → Weighted CrossEntropy
        focal_gamma:     Focal Loss gamma
        scheduler_type:  "cosine" → CosineAnnealingLR | "plateau" → ReduceLROnPlateau
        num_workers:     DataLoader workers
        seed:            Reproducibility seed
        run_tag:         Short tag appended to output dir name (useful in grid search)
    """
    # ── Reproducibility ────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CN2VF Training Pipeline")
    print(f"  Device: {device} | LR: {lr} | BS: {batch_size} | Dropout: {dropout}")
    print(f"{'='*60}")

    # ── Output directory ───────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    run_dir = Path(output_dir) / f"run_{ts}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print(f"  Output: {run_dir}\n")

    # ── Data ───────────────────────────────────────────────────
    train_samples, val_samples = load_data(data_root)
    train_loader, val_loader   = preprocess(
        train_samples, val_samples,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Class weights ─────────────────────────────────────────
    class_weights = compute_class_weights(train_samples, device)

    # ── Model ─────────────────────────────────────────────────
    model = CN2VFClassifier(num_classes=3, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────
    criterion = CN2VFClassificationLoss(
        class_weights=class_weights,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Scheduler ─────────────────────────────────────────────
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

    # ── Early stopping ────────────────────────────────────────
    early_stop = EarlyStopping(patience=patience)

    # ── History ───────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1":     [],
    }

    # ── Training loop ─────────────────────────────────────────
    print(f"\n  Starting training (max {epochs} epochs, early_stop patience={patience})")
    print(f"  Model selection criterion: Validation Macro F1-Score\n")

    for epoch in range(1, epochs + 1):
        # — Train —
        tr_loss, tr_acc, _, _ = _train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # — Validate —
        vl_loss, vl_acc, vl_f1, vl_true, vl_pred = _val_epoch(
            model, val_loader, criterion, device, epoch
        )

        # — LR scheduler step —
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(vl_f1)          # ReduceLROnPlateau monitors F1

        current_lr = optimizer.param_groups[0]["lr"]

        # — Record history —
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        # — Print summary —
        star = "*" if vl_f1 > early_stop.best_f1 else " "
        print(
            f"  E{epoch:03d} | "
            f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.1f}% | "
            f"VlLoss={vl_loss:.4f} VlAcc={vl_acc:.1f}% F1={vl_f1:.4f} | "
            f"LR={current_lr:.2e} {star}"
        )

        # — Early stopping check —
        stop = early_stop.step(vl_f1, model.state_dict(), epoch)

        if stop:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            print(f"  Best epoch: {early_stop.best_epoch} | Best F1: {early_stop.best_f1:.4f}")
            break

    # ── Restore best model ────────────────────────────────────
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
        print(f"\n  Restored best model from epoch {early_stop.best_epoch}")

    # ── Save best model checkpoint ────────────────────────────
    best_ckpt_path = ckpt_dir / "best_model.pth"
    torch.save({
        "epoch":           early_stop.best_epoch,
        "model_state_dict": model.state_dict(),
        "val_f1":          early_stop.best_f1,
        "config": {
            "img_size": img_size, "dropout": dropout,
            "use_focal": use_focal, "lr": lr, "batch_size": batch_size,
        },
    }, best_ckpt_path)
    print(f"  [OK] Best model saved -> {best_ckpt_path}")

    # ── Plots ─────────────────────────────────────────────────
    plot_training_curves(history, save_path=str(plot_dir / "training_curves.png"))

    # ── Full evaluation on val set ────────────────────────────
    print("\n  === Final Evaluation on Validation Set ===")
    eval_results = evaluate(model, val_loader, device, save_dir=str(plot_dir))

    # ── Error analysis ────────────────────────────────────────
    print("\n  === Error Analysis ===")
    analyze_errors(model, val_loader, device, save_dir=str(plot_dir / "error_analysis"))

    # ── Save run config + results ─────────────────────────────
    summary = {
        "best_epoch":   early_stop.best_epoch,
        "best_val_f1":  early_stop.best_f1,
        "accuracy":     eval_results["accuracy"],
        "macro_f1":     eval_results["macro_f1"],
        "per_class":    eval_results["per_class"],
        "config": {
            "data_root": data_root, "img_size": img_size,
            "batch_size": batch_size, "lr": lr, "dropout": dropout,
            "use_focal": use_focal, "focal_gamma": focal_gamma,
            "scheduler_type": scheduler_type,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [OK] Run summary saved -> {run_dir / 'summary.json'}")
    print(f"  Best Val F1 : {early_stop.best_f1:.4f}")
    print(f"  Accuracy    : {eval_results['accuracy'] * 100:.2f}%")
    print("=" * 60)

    return {"best_val_f1": early_stop.best_f1, "run_dir": str(run_dir), **summary}


# ═══════════════════════════════════════════════════════════════
# Hyperparameter Tuning (Grid Search)
# ═══════════════════════════════════════════════════════════════
def hyperparameter_search(data_root: str, output_dir: str = "./output/hparam_search") -> Dict:
    """
    Manual grid search over learning rate, batch size, and dropout.
    Selects best config based on validation macro F1.

    Grid:
        lr          : [1e-3, 1e-4, 1e-5]
        batch_size  : [16, 32]
        dropout     : [0.3, 0.5]
    """
    grid = {
        "lr":         [1e-3, 1e-4, 1e-5],
        "batch_size": [16, 32],
        "dropout":    [0.3, 0.5],
    }

    keys   = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Grid Search  ({len(combos)} combinations)")
    print(f"{'='*60}\n")

    best_f1     = -1.0
    best_config = {}
    all_results = []

    for i, combo in enumerate(combos, 1):
        config = dict(zip(keys, combo))
        tag = f"lr{config['lr']}_bs{config['batch_size']}_dp{config['dropout']}"
        print(f"\n  [{i}/{len(combos)}] Config: {config}")

        result = train(
            data_root=data_root,
            output_dir=output_dir,
            epochs=30,                      # shorter sweep epochs
            patience=7,
            run_tag=tag,
            **config,
        )
        all_results.append({**config, "val_f1": result["best_val_f1"]})

        if result["best_val_f1"] > best_f1:
            best_f1     = result["best_val_f1"]
            best_config = config

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH COMPLETE")
    print(f"  Best Val F1  : {best_f1:.4f}")
    print(f"  Best Config  : {best_config}")
    print(f"{'='*60}\n")

    # Save grid results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "grid_search_results.json", "w") as f:
        json.dump({"best_config": best_config, "best_f1": best_f1, "all_results": all_results}, f, indent=2)

    return best_config


# ═══════════════════════════════════════════════════════════════
# predict()
# ═══════════════════════════════════════════════════════════════
def predict(
    image,                          # str path OR PIL.Image OR torch.Tensor [3,H,W]
    checkpoint_path: str = None,
    model:           nn.Module = None,
    device:          torch.device = None,
    img_size:        int = 224,
    confidence_threshold: float = 0.5,
) -> Dict:
    """
    Predict class for a single image.

    Args:
        image:                str path, PIL Image, or pre-processed tensor [3,H,W]
        checkpoint_path:      Path to best_model.pth (used if model is None)
        model:                Pre-loaded model (optional; avoids reload overhead)
        device:               Torch device (auto-detected if None)
        img_size:             Must match training img_size
        confidence_threshold: Below this, result marked as low-confidence

    Returns:
        {
            "class":       "Fire" | "Smoke" | "Neutral",
            "class_idx":   int,
            "confidence":  float  (0.0 – 1.0),
            "probabilities": {"Fire": float, "Smoke": float, "Neutral": float},
            "is_alert":    bool   (True for Fire or Smoke above threshold),
            "low_confidence": bool,
        }
    """
    # ── Device ────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Provide either `model` or `checkpoint_path`.")
        model = CN2VFClassifier(num_classes=3)
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model = model.to(device)

    model.eval()

    # ── Preprocess ────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
    elif isinstance(image, Image.Image):
        img_tensor = transform(image).unsqueeze(0).to(device)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img_tensor = image.unsqueeze(0).to(device)
        else:
            img_tensor = image.to(device)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ── Inference ─────────────────────────────────────────────
    with torch.no_grad():
        output = model(img_tensor)
        logits = output["cls_logits"] if isinstance(output, dict) else output
        probs  = torch.softmax(logits, dim=1)[0].cpu()

    pred_idx  = int(probs.argmax().item())
    conf      = float(probs[pred_idx].item())
    class_name = IDX_TO_CLASS[pred_idx]

    return {
        "class":      class_name,
        "class_idx":  pred_idx,
        "confidence": round(conf, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        },
        "is_alert":        pred_idx in (0, 1) and conf >= confidence_threshold,
        "low_confidence":  conf < confidence_threshold,
    }


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CN2VF Optimised Training Pipeline – Fire / Smoke / Neutral"
    )

    # Dataset
    parser.add_argument("--data-root",  type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--img-size",   type=int, default=224,
                        help="Input image size (224 or 256)")

    # Training
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--patience",   type=int,   default=10)

    # Loss
    parser.add_argument("--no-focal",   action="store_true",
                        help="Use Weighted CrossEntropy instead of Focal Loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Scheduler
    parser.add_argument("--scheduler",  type=str, default="cosine",
                        choices=["cosine", "plateau"])

    # Output
    parser.add_argument("--output-dir", type=str, default="./output/pipeline")

    # Hyperparameter search
    parser.add_argument("--tune",       action="store_true",
                        help="Run GridSearch hyperparameter sweep then full train with best config")

    # Misc
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed",        type=int, default=42)

    args = parser.parse_args()

    if args.tune:
        # ── Grid search then final train ───────────────────────
        best_cfg = hyperparameter_search(
            data_root=args.data_root,
            output_dir=args.output_dir + "/hparam",
        )
        print("\n  Running full training with best hyperparameters …\n")
        train(
            data_root=args.data_root,
            output_dir=args.output_dir + "/best_run",
            img_size=args.img_size,
            epochs=args.epochs,
            patience=args.patience,
            use_focal=not args.no_focal,
            focal_gamma=args.focal_gamma,
            scheduler_type=args.scheduler,
            num_workers=args.num_workers,
            seed=args.seed,
            **best_cfg,
        )
    else:
        # ── Single training run ────────────────────────────────
        train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
            use_focal=not args.no_focal,
            focal_gamma=args.focal_gamma,
            scheduler_type=args.scheduler,
            num_workers=args.num_workers,
            seed=args.seed,
        )
