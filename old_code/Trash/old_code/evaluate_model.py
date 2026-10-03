"""
Model Evaluation Script
Generates confusion matrix and calculates evaluation metrics for CN2VF-Net
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    auc
)
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from cn2vf_net import CN2VFNet
from dataset_voc import FireSmokePascalVOCDataset


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    # Convert to [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model and collect predictions
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_ious = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            cls_logits = outputs["cls_logits"]
            bbox_pred = outputs["bbox"]
            
            probs = torch.softmax(cls_logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            
            # Collect data
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(targets["labels"].numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Calculate IoU for bounding boxes
            for i in range(len(images)):
                pred_bbox = bbox_pred[i].cpu().numpy()
                true_bbox = targets["bboxes"][i].numpy()
                iou = calculate_iou(pred_bbox, true_bbox)
                all_ious.append(iou)
    
    # Debug output
    print(f"\nDebug Info:")
    print(f"Predictions (first 20): {np.array(all_preds)[:20]}")
    print(f"True labels (first 20): {np.array(all_labels)[:20]}")
    print(f"Unique predictions: {np.unique(all_preds)}")
    print(f"Unique true labels: {np.unique(all_labels)}")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_ious)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CN2VF-Net', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save normalized confusion matrix (percentages)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix - CN2VF-Net', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """
    Plot per-class precision, recall, F1-score
    """
    # Get unique classes present in the data
    unique_classes = sorted(set(y_true) | set(y_pred))
    present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=unique_classes)
    
    x = np.arange(len(present_class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#FF6B6B')
    ax.bar(x, recall, width, label='Recall', color='#4ECDC4')
    ax.bar(x + width, f1, width, label='F1-Score', color='#45B7D1')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(present_class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


def plot_iou_distribution(ious, save_path):
    """
    Plot IoU distribution histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=20, color='#45B7D1', edgecolor='black', alpha=0.7)
    plt.axvline(ious.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean IoU: {ious.mean():.3f}')
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Bounding Box IoU Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"IoU distribution plot saved to {save_path}")
    plt.close()


def calculate_metrics(y_true, y_pred, y_probs, ious, class_names):
    """
    Calculate all evaluation metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
    
    # Bounding box metrics
    metrics['mean_iou'] = float(ious.mean())
    metrics['median_iou'] = float(np.median(ious))
    metrics['iou_std'] = float(ious.std())
    metrics['iou_at_50'] = float((ious >= 0.5).sum() / len(ious))  # % predictions with IoU >= 0.5
    metrics['iou_at_75'] = float((ious >= 0.75).sum() / len(ious))  # % predictions with IoU >= 0.75
    
    return metrics


def print_metrics_report(metrics, class_names):
    """
    Print formatted metrics report
    """
    print("\n" + "="*70)
    print("CN2VF-NET EVALUATION METRICS")
    print("="*70)
    
    print("\n--- OVERALL METRICS ---")
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\nMacro Average:")
    print(f"  Precision:        {metrics['precision_macro']:.4f}")
    print(f"  Recall:           {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:         {metrics['f1_macro']:.4f}")
    print(f"\nWeighted Average:")
    print(f"  Precision:        {metrics['precision_weighted']:.4f}")
    print(f"  Recall:           {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:         {metrics['f1_weighted']:.4f}")
    
    print("\n--- PER-CLASS METRICS ---")
    for class_name in class_names:
        if class_name in metrics['per_class']:
            cm = metrics['per_class'][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision:        {cm['precision']:.4f}")
            print(f"  Recall:           {cm['recall']:.4f}")
            print(f"  F1-Score:         {cm['f1_score']:.4f}")
    
    print("\n--- BOUNDING BOX METRICS ---")
    print(f"Mean IoU:           {metrics['mean_iou']:.4f}")
    print(f"Median IoU:         {metrics['median_iou']:.4f}")
    print(f"IoU Std Dev:        {metrics['iou_std']:.4f}")
    print(f"IoU @ 0.50:         {metrics['iou_at_50']:.4f} ({metrics['iou_at_50']*100:.2f}%)")
    print(f"IoU @ 0.75:         {metrics['iou_at_75']:.4f} ({metrics['iou_at_75']*100:.2f}%)")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CN2VF-Net model')
    parser.add_argument('--model-path', type=str, default='output/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-root', type=str, 
                        default='Datacluster Fire and Smoke Sample',
                        help='Path to dataset root')
    parser.add_argument('--split-file', type=str, default='val_split.txt',
                        help='Validation split file')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--img-size', type=int, default=448,
                        help='Input image size')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = CN2VFNet(num_classes=args.num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading validation dataset from {args.data_root}...")
    
    # Read split file
    with open(args.split_file, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    
    val_dataset = FireSmokePascalVOCDataset(
        image_dir=f"{args.data_root}/{args.data_root}",
        annotation_dir="Annotations",
        image_list=image_list,
        img_size=args.img_size,
        is_train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([item["image"] for item in batch]),
            {
                "labels": torch.tensor([item["class_label"].item() for item in batch]),
                "bboxes": torch.stack([item["bbox"] for item in batch])
            }
        )
    )
    
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # Class names
    class_names = ['Fire', 'Smoke', 'Neutral']
    
    # Evaluate
    print("\nRunning evaluation...")
    y_pred, y_true, y_probs, ious = evaluate_model(model, val_loader, device, class_names)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_probs, ious, class_names)
    
    # Print report
    print_metrics_report(metrics, class_names)
    
    # Save metrics to JSON
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Normalized confusion matrix
    cm_norm_path = output_dir / 'confusion_matrix_normalized.png'
    plot_normalized_confusion_matrix(y_true, y_pred, class_names, cm_norm_path)
    
    # Per-class metrics
    metrics_plot_path = output_dir / 'per_class_metrics.png'
    plot_per_class_metrics(y_true, y_pred, class_names, metrics_plot_path)
    
    # IoU distribution
    iou_plot_path = output_dir / 'iou_distribution.png'
    plot_iou_distribution(ious, iou_plot_path)
    
    # Save sklearn classification report
    unique_classes = sorted(set(y_true) | set(y_pred))
    present_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
    report = classification_report(y_true, y_pred, labels=unique_classes, target_names=present_class_names, zero_division=0)
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("CN2VF-NET CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")
    
    print(f"\n✓ Evaluation complete! All results saved to {output_dir}/")


if __name__ == '__main__':
    main()
