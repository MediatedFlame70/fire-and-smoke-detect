"""
Flask Web Application for CN2VF-Net Fire and Smoke Detection
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import base64
from io import BytesIO
from datetime import datetime
import glob

from cn2vf_net import CN2VFNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Fire", "Smoke", "Neutral"]
CLASS_COLORS = ["#FF0000", "#A9A9A9", "#00FF00"]  # Red, Gray, Green


def find_best_model():
    """Find the most recent best_model.pth in output directory"""
    output_dir = Path("output")
    if not output_dir.exists():
        return None
    
    # Find all best_model.pth files
    model_paths = list(output_dir.glob("*/checkpoints/best_model.pth"))
    if not model_paths:
        return None
    
    # Return the most recent one
    return str(sorted(model_paths, key=lambda x: x.stat().st_mtime)[-1])


def load_model(model_path=None):
    """Load the trained model"""
    global model, MODEL_PATH
    
    if model_path is None:
        model_path = find_best_model()
    
    if model_path is None:
        return False
    
    try:
        model = CN2VFNet(num_classes=3)
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        MODEL_PATH = model_path
        print(f"✓ Model loaded from: {model_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def get_training_status():
    """Get current training status from output directory"""
    output_dir = Path("output")
    if not output_dir.exists():
        return {"status": "not_started", "message": "No training runs found"}
    
    # Find most recent training run
    runs = sorted(output_dir.glob("cn2vf_net_*"), key=lambda x: x.stat().st_mtime)
    if not runs:
        return {"status": "not_started", "message": "No training runs found"}
    
    latest_run = runs[-1]
    config_file = latest_run / "config.json"
    checkpoints_dir = latest_run / "checkpoints"
    
    if not config_file.exists():
        return {"status": "unknown", "message": "Training configuration not found"}
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Check for checkpoints
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pth"))
        best_model = checkpoints_dir / "best_model.pth"
        
        if best_model.exists():
            checkpoint = torch.load(best_model, map_location="cpu")
            return {
                "status": "completed",
                "run_name": latest_run.name,
                "epochs_completed": checkpoint.get("epoch", "unknown"),
                "total_epochs": config.get("epochs", "unknown"),
                "best_val_acc": checkpoint.get("val_acc", "unknown"),
                "config": config,
                "model_path": str(best_model)
            }
        else:
            # Training in progress
            epoch_checkpoints = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
            if epoch_checkpoints:
                latest_checkpoint = epoch_checkpoints[-1]
                checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                return {
                    "status": "training",
                    "run_name": latest_run.name,
                    "epochs_completed": checkpoint.get("epoch", "unknown"),
                    "total_epochs": config.get("epochs", "unknown"),
                    "val_acc": checkpoint.get("val_acc", "unknown"),
                    "config": config
                }
    
    return {"status": "started", "run_name": latest_run.name, "config": config}


def preprocess_image(image):
    """Preprocess image for model input"""
    transforms = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms(image).unsqueeze(0)


def predict_image(image_path, confidence_threshold=0.5):
    """Run inference on an image"""
    if model is None:
        return None
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    image_tensor = preprocess_image(image).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Parse results
    class_probs = torch.softmax(output["cls_logits"], dim=1)
    predicted_class_idx = torch.argmax(class_probs, dim=1).item()
    confidence = class_probs[0, predicted_class_idx].item()
    bbox_norm = output["bbox"][0].cpu().numpy()
    
    # --- FALSE POSITIVE FILTER (Color Heuristic) ---
    # If the model predicts Fire but there are no fire-colored pixels in the bounding box, reject it.
    if predicted_class_idx == 0:  # Fire
        import cv2
        import numpy as np
        
        img_np = np.array(image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        x_c, y_c, bw, bh = bbox_norm
        x1 = int(max(0, (x_c - bw/2) * orig_w))
        y1 = int(max(0, (y_c - bh/2) * orig_h))
        x2 = int(min(orig_w, (x_c + bw/2) * orig_w))
        y2 = int(min(orig_h, (y_c + bh/2) * orig_h))
        
        crop = hsv[y1:y2, x1:x2]
        
        if crop.size > 0:
            # OpenCV HSV: H=0-179, S=0-255, V=0-255
            # Fire is typically Red/Orange (H: 0-22 or 170-179) with decent saturation and high brightness
            lower1 = np.array([0, 100, 150])
            upper1 = np.array([22, 255, 255])
            lower2 = np.array([170, 100, 150])
            upper2 = np.array([179, 255, 255])
            
            mask = cv2.inRange(crop, lower1, upper1) | cv2.inRange(crop, lower2, upper2)
            fire_pixels = cv2.countNonZero(mask)
            total_pixels = crop.shape[0] * crop.shape[1]
            
            # If less than 0.5% of the bounding box has fire colors, override to Neutral
            if fire_pixels / max(1, total_pixels) < 0.005:
                predicted_class_idx = 2  # Override to Neutral
                
                # Swap probabilities logically so it makes sense in the UI
                old_fire_prob = float(class_probs[0, 0].item())
                old_neutral_prob = float(class_probs[0, 2].item())
                
                # Make neutral prob dominant
                class_probs[0, 2] = max(0.9, old_fire_prob) 
                class_probs[0, 0] = old_neutral_prob
                confidence = float(class_probs[0, 2].item())

    class_name = CLASS_NAMES[predicted_class_idx]
    is_detection = (predicted_class_idx in [0, 1]) and (confidence >= confidence_threshold)
    
    # Get all class probabilities
    all_probs = {
        CLASS_NAMES[i]: float(class_probs[0, i].item())
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        "class": class_name,
        "class_idx": predicted_class_idx,
        "confidence": float(confidence),
        "all_probabilities": all_probs,
        "bbox": bbox_norm.tolist(),
        "is_detection": is_detection,
        "image_size": (orig_w, orig_h),
        "color": CLASS_COLORS[predicted_class_idx]
    }


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/status')
def status():
    """Get training and model status"""
    training_status = get_training_status()
    model_loaded = model is not None
    
    return jsonify({
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "device": str(device),
        "training": training_status
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not model:
        if not load_model():
            return jsonify({"error": "Model not loaded. Please train the model first."}), 503
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Run prediction
    confidence_threshold = float(request.form.get('confidence', 0.5))
    result = predict_image(filepath, confidence_threshold)
    
    if result is None:
        return jsonify({"error": "Prediction failed"}), 500
    
    # Convert image to base64 for display
    with Image.open(filepath) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
    
    result['image_data'] = f"data:image/jpeg;base64,{img_str}"
    result['filename'] = filename
    result['upload_path'] = filepath
    
    return jsonify(result)


@app.route('/results')
def results():
    """Results and visualization page"""
    return render_template('results.html')


@app.route('/api/training-metrics')
def training_metrics():
    """Get training metrics from TensorBoard logs"""
    training_status = get_training_status()
    if training_status["status"] not in ["training", "completed"]:
        return jsonify({"error": "No training data available"}), 404
    
    # TODO: Parse TensorBoard event files for detailed metrics
    # For now, return basic info
    return jsonify(training_status)


@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Manually load a specific model"""
    data = request.json
    model_path = data.get('model_path')
    
    if load_model(model_path):
        return jsonify({"success": True, "message": "Model loaded successfully"})
    else:
        return jsonify({"success": False, "message": "Failed to load model"}), 500


if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("CN2VF-Net Fire and Smoke Detection Web Application")
    print("=" * 70)

    # Try to load model automatically
    print("\nLooking for trained model...")
    if load_model():
        print("✓ Model loaded and ready for inference")
    else:
        print("⚠ No trained model found. Please train the model first.")
        print("  The web interface will still be available.")

    print(f"\n🌐 Starting web server on http://localhost:5000")
    print("=" * 70)

    # When launched via `streamlit run`, Streamlit executes the script in a thread
    # where registering signal handlers (used by Flask's reloader) fails with
    # "ValueError: signal only works in main thread". Detect Streamlit and skip
    # calling `app.run()` in that case — Streamlit already serves the app if
    # needed, or the Flask app can be started separately.
    if 'streamlit' in sys.modules or any('STREAMLIT' in k for k in (sys.argv + list(sys.modules.keys()))):
        print("Detected Streamlit environment — skipping Flask app.run() to avoid signal errors.")
    else:
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        except ValueError as e:
            print(f"Skipping app.run() due to environment error: {e}")
