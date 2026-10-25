"""
Inference script for CN2VF-Net on single images or batch of images.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from cn2vf_net import CN2VFNet
from preprocessing import letterbox_image


class FireSmokePredictor:
    """Predictor for fire and smoke detection."""
    
    CLASS_NAMES = ["Fire only", "Smoke only", "Fire + Smoke", "None"]
    CLASS_COLORS = [(255, 0, 0), (169, 169, 169), (255, 128, 0), (0, 255, 0)]
    
    def __init__(self, checkpoint_path, device="cuda"):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "cn2vf_dualbox_v2":
            raise ValueError("This predictor requires a four-class dual-box CN2VF checkpoint.")
        self.preprocessing_type = checkpoint.get("preprocessing", "stretch") if isinstance(checkpoint, dict) else "stretch"
        self.img_size = int(checkpoint.get("img_size", 448))
        self.model = CN2VFNet(num_classes=4)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val Accuracy: {checkpoint.get('val_acc', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transforms = T.Compose([
            *([T.Resize((self.img_size, self.img_size))] if self.preprocessing_type == "stretch" else []),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("✓ Model loaded successfully")
    
    @torch.no_grad()
    def predict(self, image_path, confidence_threshold=0.5):
        """
        Predict fire/smoke detection on a single image.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence to consider detection valid
        
        Returns:
            {
                "class": str,
                "confidence": float,
                "bbox": [x, y, w, h],  # normalized coords
                "is_detection": bool  # True if fire or smoke detected with high confidence
            }
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        prepared_image = letterbox_image(image, size=self.img_size) if self.preprocessing_type == "letterbox-v1" else image
        image_tensor = self.transforms(prepared_image).unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(image_tensor)
        
        # Parse results
        class_probs = torch.softmax(output["cls_logits"], dim=1)
        predicted_class_idx = torch.argmax(class_probs, dim=1).item()
        confidence = class_probs[0, predicted_class_idx].item()
        fire_bbox = output["fire_bbox"][0].cpu().tolist()
        smoke_bbox = output["smoke_bbox"][0].cpu().tolist()

        # Convert each predicted padded-canvas box to original-image coordinates.
        if self.preprocessing_type == "letterbox-v1":
            scale = min(self.img_size / orig_w, self.img_size / orig_h)
            resized_w = max(1, round(orig_w * scale))
            resized_h = max(1, round(orig_h * scale))
            left = (self.img_size - resized_w) // 2
            top = (self.img_size - resized_h) // 2

            def unletterbox(box):
                cx, cy, box_w, box_h = (float(value) for value in box)
                x1 = max(0.0, min(float(orig_w), ((cx - box_w / 2) * self.img_size - left) * orig_w / resized_w))
                y1 = max(0.0, min(float(orig_h), ((cy - box_h / 2) * self.img_size - top) * orig_h / resized_h))
                x2 = max(0.0, min(float(orig_w), ((cx + box_w / 2) * self.img_size - left) * orig_w / resized_w))
                y2 = max(0.0, min(float(orig_h), ((cy + box_h / 2) * self.img_size - top) * orig_h / resized_h))
                return [(x1 + x2) / (2 * orig_w), (y1 + y2) / (2 * orig_h),
                        (x2 - x1) / orig_w, (y2 - y1) / orig_h]

            fire_bbox = unletterbox(fire_bbox)
            smoke_bbox = unletterbox(smoke_bbox)
        
        class_name = self.CLASS_NAMES[predicted_class_idx]
        fire_confidence = float((class_probs[0, 0] + class_probs[0, 2]).item())
        smoke_confidence = float((class_probs[0, 1] + class_probs[0, 2]).item())
        fire_detected = fire_confidence >= confidence_threshold
        smoke_detected = smoke_confidence >= confidence_threshold
        is_detection = fire_detected or smoke_detected
        
        return {
            "class": class_name,
            "class_idx": predicted_class_idx,
            "confidence": confidence,
            "fire_confidence": fire_confidence,
            "smoke_confidence": smoke_confidence,
            "fire_bbox": fire_bbox,
            "smoke_bbox": smoke_bbox,
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected,
            "is_detection": is_detection,
            "image_size": (orig_w, orig_h),
        }
    
    def visualize(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        Visualize detection results on image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            PIL Image with visualization
        """
        # Get prediction
        result = self.predict(image_path, confidence_threshold)
        
        # Load original image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        orig_w, orig_h = result["image_size"]
        detections = []
        if result["fire_detected"]:
            detections.append(("Fire", result["fire_confidence"], result["fire_bbox"], self.CLASS_COLORS[0]))
        if result["smoke_detected"]:
            detections.append(("Smoke", result["smoke_confidence"], result["smoke_bbox"], self.CLASS_COLORS[1]))

        if detections:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except OSError:
                font = ImageFont.load_default()

            for label_name, confidence, box, color in detections:
                x, y, w, h = box
                # Convert normalized coords to absolute.
                x1 = int((x - w/2) * orig_w)
                y1 = int((y - h/2) * orig_h)
                x2 = int((x + w/2) * orig_w)
                y2 = int((y + h/2) * orig_h)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                label = f"{label_name}: {confidence*100:.1f}%"
                bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)
        else:
            label = f"{result['class']}: {result['confidence']*100:.1f}%"
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()
            draw.text((10, 10), label, fill=self.CLASS_COLORS[result["class_idx"]], font=font)
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
            print(f"✓ Visualization saved to: {output_path}")
        
        return image


def main():
    parser = argparse.ArgumentParser(description="CN2VF-Net Inference for Fire/Smoke Detection")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations (optional)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (only print results)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FireSmokePredictor(args.checkpoint, device=args.device)
    
    # Get image paths
    image_path = Path(args.image)
    if image_path.is_dir():
        image_paths = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        print(f"Found {len(image_paths)} images in directory")
    else:
        image_paths = [image_path]
    
    # Create output directory if needed
    if args.output_dir and not args.no_viz:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    print("\n" + "="*70)
    detections_count = 0
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        
        # Get prediction
        result = predictor.predict(str(img_path), args.confidence)
        
        # Print results
        print(f"  Class:      {result['class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Fire score: {result['fire_confidence']*100:.2f}%  box={result['fire_bbox']}")
        print(f"  Smoke score:{result['smoke_confidence']*100:.2f}%  box={result['smoke_bbox']}")
        
        if result["is_detection"]:
            labels = ", ".join(name for name, detected in (
                ("Fire", result["fire_detected"]), ("Smoke", result["smoke_detected"])
            ) if detected)
            print(f"  ALERT: {labels} detected!")
            detections_count += 1
        
        # Visualize if requested
        if not args.no_viz:
            output_path = None
            if args.output_dir:
                output_path = Path(args.output_dir) / f"pred_{img_path.name}"
            
            predictor.visualize(str(img_path), output_path, args.confidence)
    
    print("\n" + "="*70)
    print(f"Processed {len(image_paths)} images")
    print(f"Detections (Fire/Smoke): {detections_count}")
    print("="*70)


if __name__ == "__main__":
    main()
