# Quick Start Guide

## Step-by-Step Setup and Training

## YOLOv8 Fire/Smoke Pipeline (Recommended)

### 1. Prepare and Validate Data

This command verifies labels, auto-remaps source class IDs to binary classes (`fire`, `smoke`), creates a balanced 80/20 train/val split, saves bbox visual checks, and applies smoke-focused oversampling/augmentation.

```bash
python train_yolov8.py
```

Artifacts are saved under `output/yolov8/`.

### 2. Evaluate and Run Error Analysis

```bash
python evaluate_yolov8.py
```

Outputs include:
- `output/yolov8/evaluation/evaluation_metrics.json`
- `output/yolov8/evaluation/confusion_matrix.png`
- `output/yolov8/evaluation/error_analysis/` (Smoke<->Fire confusion examples)

### 3. Run Inference (Image/Video)

```bash
python predict_yolov8.py ./test_images
python predict_yolov8.py ./test_wildfire.mp4
```

### 4. Launch Streamlit App

```bash
streamlit run streamlit_yolov8_app.py
```

Then open the local URL shown in terminal and upload an image/video.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
2. Download and extract to a folder (e.g., `./fire_smoke_data`)
3. Ensure structure:
   ```
   fire_smoke_data/
     ├── train/
     │   ├── images/
     │   └── labels/
     └── val/
       ├── images/
       └── labels/
   ```

     Each `.txt` label file should use YOLO rows in the form `class_id x_center y_center width height` with normalized coordinates.

### 3. Test Dataset Loading

```bash
python dataset.py ./fire_smoke_data
```

Expected output:
```
Loaded X samples for train split
Loaded Y samples for val split
Train batches: ...
Val batches: ...
✓ Dataset loading test passed!
```

### 4. Run Model Tests

```bash
python test_cn2vf_net.py
```

Expected: All 15 tests should pass ✓

### 5. Start Training

**Basic training (CPU or GPU):**
```bash
python train.py \
  --data-root ./fire_smoke_data \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --output-dir ./output
```

**With TensorBoard logging:**
```bash
python train.py \
  --data-root ./fire_smoke_data \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --use-tensorboard \
  --output-dir ./output
```

**Advanced options:**
```bash
python train.py \
  --data-root ./fire_smoke_data \
  --epochs 150 \
  --batch-size 16 \
  --lr 2e-4 \
  --cls-weight 1.0 \
  --bbox-weight 5.0 \
  --use-iou \
  --use-tensorboard \
  --save-interval 5 \
  --num-workers 4 \
  --output-dir ./output
```

### 6. Monitor Training (if using TensorBoard)

Open a new terminal:
```bash
tensorboard --logdir ./output/cn2vf_net_<timestamp>/tensorboard
```

Then open browser: http://localhost:6006

### 7. Inference on New Images

**Single image:**
```bash
python inference.py \
  --checkpoint ./output/cn2vf_net_<timestamp>/checkpoints/best_model.pth \
  --image test_image.jpg \
  --output-dir ./predictions
```

**Batch of images:**
```bash
python inference.py \
  --checkpoint ./output/cn2vf_net_<timestamp>/checkpoints/best_model.pth \
  --image ./test_images/ \
  --output-dir ./predictions \
  --confidence 0.7
```

## Expected Training Time

- **CPU**: ~30-60 minutes per epoch (depends on dataset size)
- **GPU (RTX 3060+)**: ~5-10 minutes per epoch

## Expected Performance

After 100 epochs:
- **Validation Accuracy**: 85-95% (depends on dataset quality)
- **Inference Speed**: 
  - CPU: ~100-200ms per image
  - GPU: ~10-20ms per image

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "Annotation file not found"
**Solution**: Check dataset structure and annotation paths in `dataset.py`

### Issue: Out of memory during training
**Solution**: Reduce batch size
```bash
python train.py --batch-size 4 ...
```

### Issue: Low validation accuracy
**Solutions**:
1. Train longer (150-200 epochs)
2. Increase learning rate (2e-4)
3. Adjust loss weights (try `--bbox-weight 10.0`)
4. Check dataset annotation quality

## Files Overview

- `cn2vf_net.py` - Model architecture (1.26M parameters)
- `dataset.py` - Dataset loader with preprocessing
- `train.py` - Training script with validation
- `inference.py` - Inference on new images
- `test_cn2vf_net.py` - Comprehensive test suite
- `README.md` - Full documentation

## Next Steps After Training

1. **Evaluate on test set** - Create evaluation script
2. **Export to ONNX** - For production deployment
3. **Optimize for inference** - Quantization, pruning
4. **Deploy to drone** - Integration with drone platform

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review test_cn2vf_net.py for component testing
3. Validate dataset loading with dataset.py
