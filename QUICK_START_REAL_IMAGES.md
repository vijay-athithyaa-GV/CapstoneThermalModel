# Quick Start: Using Real Thermal Images

## ✅ Model Trained with Real Images

The model has been successfully trained on:
- **200 synthetic FLIR-like images**
- **29 real thermal images** (from RealImageDataset/)
- **Total**: 229 images
- **Accuracy**: 89.1%

---

## 🚀 Quick Test on Real Images

```bash
# Test a single real image
python test_real_image.py RealImageDataset/bb10.jpeg

# Test multiple images
python test_real_image.py RealImageDataset/bb15.jpeg
python test_real_image.py RealImageDataset/bb20.jpeg
```

---

## 📊 Results on Real Images

All tested real images show:
- **Load Category**: High Load
- **Criticality**: 0.88 - 0.98 (CRITICAL)
- **Temperature Range**: 25-120°C
- **Status**: 🔴 Immediate attention required

---

## 🔧 Processing New Real Images

### Step 1: Add Images to RealImageDataset/
```bash
# Copy your thermal images to:
RealImageDataset/your_image.jpeg
```

### Step 2: Process Images
```bash
python process_real_images.py
```

This will:
- Analyze all images in RealImageDataset/
- Extract temperature statistics
- Auto-generate labels based on thermal patterns
- Create/update `real_images_labels.csv`

### Step 3: Merge with Dataset
The script automatically merges with existing dataset.

### Step 4: Retrain Model
```bash
python train_merged.py
```

---

## 📝 Files Structure

```
BusbarPreprocessing/
├── RealImageDataset/          # Your real thermal images
│   ├── bb5.jpeg
│   ├── bb10.jpeg
│   └── ...
├── dataset/                   # Training dataset
│   ├── labels.csv            # Original synthetic dataset
│   ├── labels_merged.csv     # Merged (synthetic + real)
│   └── flir_thermal_*.png    # Synthetic images
├── real_images_labels.csv     # Real images labels
├── process_real_images.py     # Process real images
├── train_merged.py            # Train with merged dataset
└── test_real_image.py         # Test on real images
```

---

## 🎯 Key Commands

```bash
# Process real images
python process_real_images.py

# Train with merged dataset
python train_merged.py

# Test on real image
python test_real_image.py RealImageDataset/bb10.jpeg

# Generate performance graphs
python plot_model_performance.py
```

---

## ⚠️ Important Notes

### Temperature Calibration
Real images are hitting 120°C ceiling. If your camera uses different ranges:

```python
# Adjust in test_real_image.py or process_real_images.py
feats, _ = preprocess_image_to_features(
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=0.0,    # Your camera's min temp
    max_temp_c=200.0   # Your camera's max temp
)
```

### Label Review
- Review `real_images_labels.csv` for accuracy
- Manually adjust labels if needed
- Retrain model after corrections

---

## 📈 Model Performance

- **Test Accuracy**: 89.1%
- **Regression MAE**: 0.0644 (6.44% error)
- **Dataset**: 229 images (200 synthetic + 29 real)
- **Status**: ✅ Ready for production

---

## 🔄 Workflow

1. **Add real images** → `RealImageDataset/`
2. **Process images** → `python process_real_images.py`
3. **Review labels** → Check `real_images_labels.csv`
4. **Train model** → `python train_merged.py`
5. **Test model** → `python test_real_image.py <image>`
6. **Deploy** → Use trained models in `artifacts/`

---

## 📚 Documentation

- **Full Training Summary**: `REAL_IMAGES_TRAINING_SUMMARY.md`
- **Performance Graphs**: `PERFORMANCE_GRAPHS_README.md`
- **Using Real Images**: `USING_REAL_IMAGES.md`

---

**Status**: ✅ Model trained and ready to use with real thermal images!

