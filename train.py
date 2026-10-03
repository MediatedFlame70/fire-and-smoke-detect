"""
Training script for CN2VF-Net on Fire and Smoke Detection Dataset.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import create_dataloaders
from cn2vf_net import CN2VFNet, FocalLoss, ciou_loss


class CN2VFLoss(nn.Module):
    """
    Combined loss for CN2VF-Net:
    - Class-weighted focal classification loss
    - Complete-IoU bounding-box loss
    """
    
    def __init__(self, cls_weight=1.0, bbox_weight=2.0, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.cls_loss_fn = FocalLoss(
            class_weights=class_weights, label_smoothing=label_smoothing
        )
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with "cls_logits" [B, 3] and "bbox" [B, 4]
            targets: dict with "class_label" [B] and "bbox" [B, 4]
        
        Returns:
            total_loss, loss_dict
        """
        cls_logits = predictions["cls_logits"]
        pred_bbox = predictions["bbox"]
        
        target_labels = targets["class_label"]
        target_bbox = targets["bbox"]
        
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_logits, target_labels)
        
        bbox_loss = ciou_loss(pred_bbox, target_bbox)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        
        loss_dict = {
            "total": total_loss.item(),
            "cls": cls_loss.item(),
            "bbox": bbox_loss.item(),
        }
        
        return total_loss, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
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
            "bbox": batch["bbox"].to(device),
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
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
    confusion = torch.zeros(3, 3, dtype=torch.long)
    
    pbar = tqdm(dataloader, desc=f"Validation")
    
    for batch in pbar:
        images = batch["image"].to(device)
        targets = {
            "class_label": batch["class_label"].to(device),
            "bbox": batch["bbox"].to(device),
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
        encoded = targets["class_label"].cpu() * 3 + predicted.cpu()
        confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
        
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


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path, val_macro_f1=None):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "model_type": "cn2vf",
        "preprocessing": "letterbox-v1",
    }
    torch.save(checkpoint, save_path)
    print(f"[OK] Checkpoint saved: {save_path}")


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
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
    print(f"\nLoading dataset from: {args.data_root}")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing CN2VF-Net...")
    model = CN2VFNet(num_classes=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    counts = torch.zeros(3, dtype=torch.float32)
    for sample in train_loader.dataset.samples:
        if sample["source"] == "text_label":
            label = train_loader.dataset._select_primary_object(sample["objects"])["class_id"]
        else:
            record = sample["record"]
            name = str(record.get("class") or record.get("category") or record.get("label") or "neutral").lower()
            label = train_loader.dataset.CLASS_MAPPING.get(name, 2)
        if 0 <= label < 3:
            counts[label] += 1
    class_weights = (counts.sum() / (3.0 * counts.clamp_min(1))).sqrt()
    class_weights = (class_weights / class_weights.mean()).clamp(max=2.5).to(device)
    print(f"Training class counts: {counts.int().tolist()}")
    print(f"Class loss weights:    {class_weights.cpu().tolist()}")

    criterion = CN2VFLoss(
        cls_weight=args.cls_weight,
        bbox_weight=args.bbox_weight,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_f1 = -1.0
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # Train
        train_loss, train_cls_loss, train_bbox_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
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
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pth", val_macro_f1
            )
        
        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                checkpoint_dir / "best_model.pth", val_macro_f1
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
        checkpoint_dir / "final_model.pth", val_macro_f1
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
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--img-size", type=int, default=448,
                        help="Input image size (default: 448)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear learning-rate warm-up before cosine decay")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01,
                        help="Final LR as a fraction of the initial LR")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stop after this many epochs without macro-F1 improvement")
    
    # Loss weights
    parser.add_argument("--cls-weight", type=float, default=1.0,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--bbox-weight", type=float, default=2.0,
                        help="Complete-IoU bounding-box loss weight (default: 2.0)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    main(args)
