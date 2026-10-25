"""
Training script for CN2VF-Net on Fire and Smoke Detection Dataset.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dataloaders
from cn2vf_net import CN2VFNet, FocalLoss, ciou_loss


class CN2VFLoss(nn.Module):
    """
    Combined loss for CN2VF-Net:
    - Class-weighted focal classification loss
    - Presence-masked fire and smoke Complete-IoU box losses
    """
    
    def __init__(self, cls_weight=1.0, bbox_weight=1.5, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.cls_loss_fn = FocalLoss(
            class_weights=class_weights, label_smoothing=label_smoothing
        )
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with four class logits and separate fire/smoke boxes
            targets: four-way image label, class-specific boxes, and presence masks
        
        Returns:
            total_loss, loss_dict
        """
        cls_logits = predictions["cls_logits"]
        target_labels = targets["class_label"]
        cls_loss = self.cls_loss_fn(cls_logits, target_labels)

        fire_mask = targets["fire_present"].bool()
        smoke_mask = targets["smoke_present"].bool()
        fire_loss = (
            ciou_loss(predictions["fire_bbox"][fire_mask], targets["fire_bbox"][fire_mask])
            if fire_mask.any() else predictions["fire_bbox"].sum() * 0.0
        )
        smoke_loss = (
            ciou_loss(predictions["smoke_bbox"][smoke_mask], targets["smoke_bbox"][smoke_mask])
            if smoke_mask.any() else predictions["smoke_bbox"].sum() * 0.0
        )
        bbox_loss = fire_loss + smoke_loss
        total_loss = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        
        loss_dict = {
            "total": total_loss.item(),
            "cls": cls_loss.item(),
            "bbox": bbox_loss.item(),
            "fire_bbox": fire_loss.item(),
            "smoke_bbox": smoke_loss.item(),
        }
        
        return total_loss, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer,
                    multi_scale=False, scaler=None):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        targets = {
            "class_label": batch["class_label"].to(device),
            "fire_bbox": batch["fire_bbox"].to(device),
            "smoke_bbox": batch["smoke_bbox"].to(device),
            "fire_present": batch["fire_present"].to(device),
            "smoke_present": batch["smoke_present"].to(device),
        }
        if multi_scale:
            size = (384, 416, 448, 480, 512)[torch.randint(0, 5, ()).item()]
            if images.shape[-1] != size:
                images = F.interpolate(images, size=(size, size), mode="bilinear",
                                       align_corners=False, antialias=True)
        
        # Forward pass with Automatic Mixed Precision (AMP)
        use_amp = scaler is not None and scaler.is_enabled()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass with Scaler
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Metrics
        running_loss += loss_dict["total"]
        running_cls_loss += loss_dict["cls"]
        running_bbox_loss += loss_dict["bbox"]
        
        # Classification accuracy
        _, predicted = torch.max(predictions["cls_logits"], 1)
        total += targets["class_label"].size(0)
        correct += (predicted == targets["class_label"]).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "cls": f"{loss_dict['cls']:.4f}",
            "bbox": f"{loss_dict['bbox']:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
        
        # TensorBoard logging
        global_step = epoch * len(dataloader) + batch_idx
        if writer and batch_idx % 10 == 0:
            writer.add_scalar("Train/Loss", loss_dict["total"], global_step)
            writer.add_scalar("Train/ClsLoss", loss_dict["cls"], global_step)
            writer.add_scalar("Train/BBoxLoss", loss_dict["bbox"], global_step)
            writer.add_scalar("Train/Accuracy", 100 * correct / total, global_step)
    
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_bbox_loss = running_bbox_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(4, 4, dtype=torch.long)
    
    pbar = tqdm(dataloader, desc=f"Validation")
    
    for batch in pbar:
        images = batch["image"].to(device)
        targets = {
            "class_label": batch["class_label"].to(device),
            "fire_bbox": batch["fire_bbox"].to(device),
            "smoke_bbox": batch["smoke_bbox"].to(device),
            "fire_present": batch["fire_present"].to(device),
            "smoke_present": batch["smoke_present"].to(device),
        }
        
        # Forward pass
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        
        # Metrics
        running_loss += loss_dict["total"]
        running_cls_loss += loss_dict["cls"]
        running_bbox_loss += loss_dict["bbox"]
        
        # Classification accuracy
        _, predicted = torch.max(predictions["cls_logits"], 1)
        total += targets["class_label"].size(0)
        correct += (predicted == targets["class_label"]).sum().item()
        encoded = targets["class_label"].cpu() * 4 + predicted.cpu()
        confusion += torch.bincount(encoded, minlength=16).reshape(4, 4)
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_bbox_loss = running_bbox_loss / len(dataloader)
    accuracy = 100 * correct / total
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(0).clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    macro_f1 = f1.mean().item()
    
    # TensorBoard logging
    if writer:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/ClsLoss", avg_cls_loss, epoch)
        writer.add_scalar("Val/BBoxLoss", avg_bbox_loss, epoch)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        writer.add_scalar("Val/MacroF1", macro_f1, epoch)
    
    return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy, macro_f1


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path,
                    val_macro_f1=None, img_size=448):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "model_type": "cn2vf_dualbox_v2",
        "num_classes": 4,
        "class_names": ["Fire", "Smoke", "Both", "None"],
        "preprocessing": "letterbox-v1",
        "img_size": int(img_size),
    }
    torch.save(checkpoint, save_path)
    print(f"[OK] Checkpoint saved: {save_path}")


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("CUDA CuDNN benchmark enabled for accelerated training.")
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    if args.fp16 and device.type == "cuda":
        print("FP16 Automatic Mixed Precision (AMP) enabled.")
    
    # Resume in the existing run directory; otherwise create a fresh run.
    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        output_dir = resume_path.parent.parent
        print(f"Resuming from checkpoint: {resume_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"cn2vf_net_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save training config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / "tensorboard") if args.use_tensorboard else None
    
    # Load dataset
    print(f"\nLoading datasets from: {', '.join(args.data_roots)}")
    train_loader, val_loader = create_dataloaders(
        data_roots=args.data_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing CN2VF-Net...")
    model = CN2VFNet(num_classes=4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    counts = torch.zeros(4, dtype=torch.float32)
    for sample in train_loader.dataset.samples:
        label = train_loader.dataset.sample_class_label(sample)
        counts[label] += 1
    # Normalized square-root dampened inverse frequency weighting for 4-way classification
    raw_weights = (counts.sum() / (4.0 * counts.clamp_min(1))).pow(0.5)
    class_weights = (raw_weights / raw_weights.mean()).clamp(min=0.5, max=3.5).to(device)
    print(f"Training class counts: {counts.int().tolist()}")
    print(f"Class loss weights:    {class_weights.cpu().tolist()}")

    criterion = CN2VFLoss(
        cls_weight=args.cls_weight,
        bbox_weight=args.bbox_weight,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_type") != "cn2vf_dualbox_v2":
            raise ValueError(
                "This is not a four-class dual-box checkpoint. Start a fresh run for the merged dataset; "
                "the previous three-class checkpoint is not compatible."
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"Checkpoint is from epoch {start_epoch - 1}; continuing at epoch {start_epoch}.")
    
    # Learning rate scheduler
    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=args.lr * args.min_lr_ratio,
    )
    if warmup_epochs:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = cosine

    if resume_checkpoint is not None:
        # Older checkpoints do not store scheduler state; advance it to the
        # checkpoint epoch so the resumed learning rate follows the schedule.
        for group in optimizer.param_groups:
            group["initial_lr"] = args.lr
            group["lr"] = args.lr
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
                category=UserWarning,
            )
            for _ in range(start_epoch - 1):
                scheduler.step()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_f1 = float(resume_checkpoint.get("val_macro_f1", -1.0)) if resume_checkpoint else -1.0
    best_val_acc = float(resume_checkpoint.get("val_acc", 0.0)) if resume_checkpoint else 0.0
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    if resume_checkpoint is not None and best_checkpoint_path.is_file():
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        best_val_f1 = float(best_checkpoint.get("val_macro_f1", best_val_f1))
        best_val_acc = float(best_checkpoint.get("val_acc", best_val_acc))
    epochs_without_improvement = 0
    
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}, which is >= --epochs {args.epochs}. "
            "Set --epochs to a larger total epoch count."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # Train
        train_loss, train_cls_loss, train_bbox_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            multi_scale=args.multi_scale, scaler=scaler,
        )
        
        # Validate
        val_loss, val_cls_loss, val_bbox_loss, val_acc, val_macro_f1 = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, bbox: {train_bbox_loss:.4f})")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} (cls: {val_cls_loss:.4f}, bbox: {val_bbox_loss:.4f})")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  Val Macro F1: {val_macro_f1:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save checkpoints
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth", val_macro_f1, args.img_size
            )
        
        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                checkpoint_dir / "best_model.pth", val_macro_f1, args.img_size
            )
            print(f"  [BEST] Validation macro F1: {val_macro_f1:.4f} (accuracy {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: validation macro F1 did not improve for {args.patience} epochs.")
                break
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_loss, val_acc,
        checkpoint_dir / "final_model.pth", val_macro_f1, args.img_size
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CN2VF-Net for Fire/Smoke Detection")
    
    # Dataset
    parser.add_argument("--data-roots", "--data-root", dest="data_roots", nargs="+", required=True,
                        help="One or more dataset roots (e.g. .\\d-fire .\\fire_smoke)")
    parser.add_argument("--img-size", type=int, default=448,
                        help="Input image size (default: 448)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=25,
                        help="Total number of training epochs (default: 20)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume model and optimizer state from a checkpoint, continuing its run")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16 for RTX GPU acceleration)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Enable FP16 Automatic Mixed Precision (AMP) for 2-3x speedup")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable FP16 mixed precision")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.05)")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear learning-rate warm-up before cosine decay")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01,
                        help="Final LR as a fraction of the initial LR")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stop after this many epochs without macro-F1 improvement")
    
    # Loss weights
    parser.add_argument("--cls-weight", type=float, default=1.0,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--bbox-weight", type=float, default=1.0,
                        help="Weight for each present-class Complete-IoU loss (default: 1.5)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Randomly resize each training batch to 384–512 pixels")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers (default: 0 for Windows stability)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    main(args)
