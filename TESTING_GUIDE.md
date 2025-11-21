# How to Test the Model

This guide explains all the ways you can test the Busbar Heat Detection model.

---

## 🚀 Quick Start

### Prerequisites

1. **Model must be trained first** (if not already done):
   ```bash
   python train_criticality_based.py --csv "dataset/labels.csv" --roots "Lowload" "HighLoad"
   ```

2. **Model files should exist** in `artifacts_criticality/`:
   - `regressor.joblib` - The trained model
   - `model_config.joblib` - Model configuration (thresholds)

---

## 📸 Method 1: Test Single Image (Recommended)

### Basic Usage

```bash
python test_criticality_based.py "path/to/your/image.jpg"
```

### Examples

**Test a Low Load image:**
```bash
python test_criticality_based.py "Lowload/IMG-20251105-WA0117.jpg"
```

**Test a High Load image:**
```bash
python test_criticality_based.py "HighLoad/IMG-20251105-WA0086.jpg"
```

**Test with custom model directory:**
```bash
python test_criticality_based.py "my_image.jpg" "artifacts_criticality"
```

**Test with warn-only mode** (processes non-thermal images too):
```bash
python test_criticality_based.py "my_image.jpg" --warn-only
```

### What You'll See

The script provides detailed output:

```
======================================================================
Testing with Criticality-Based Model
======================================================================

Image: HighLoad/IMG-20251105-WA0086.jpg
Model: artifacts_criticality

[1] Loading image...
  ✓ Image loaded: 160×120 pixels, 3 channels

[2] Validating thermal image...
  ✓ Image appears to be a thermal image

[3] Loading trained model...
  ✓ Model loaded successfully
  Thresholds: Low<0.33, Med<0.67, High≥0.67

[4] Extracting features from image...
  ✓ Features extracted: (6,)

  Temperature Statistics:
    Min: 20.0°C
    Max: 120.0°C
    Mean: 65.3°C
    Std: 18.5°C

[5] Running inference...
  ✓ Prediction complete

  Criticality Score: 0.8234
  Derived Category: High Load

======================================================================
RESULTS
======================================================================

  Criticality Score: 0.8234
  Load Category:     High Load

  Interpretation:
    Criticality range: 0.67 - 1.0 → High Load
    🔴 CRITICAL - Immediate attention required (Very Critical)

  JSON Output:
    {'criticality_score': 0.8234, 'load_category': 'High Load'}

======================================================================
How it works:
  1. Model predicts criticality score: 0.8234
  2. Classification derived from criticality:
     - 0.8234 → High Load
======================================================================
```

### Output Format

The function returns a dictionary:
```python
{
    "criticality_score": 0.8234,  # Float between 0.0 and 1.0
    "load_category": "High Load"   # "Low Load", "Medium Load", or "High Load"
}
```

---

## 📊 Method 2: Evaluate Model Performance

### Comprehensive Performance Evaluation

This generates detailed performance graphs and metrics:

```bash
python evaluate_model_performance.py \
    --csv "dataset/labels.csv" \
    --roots "Lowload" "HighLoad" \
    --model_dir "artifacts_criticality" \
    --output_dir "performance_evaluation"
```

### What It Does

1. **Loads all images** from the CSV file
2. **Runs predictions** on all images
3. **Computes metrics**:
   - Classification accuracy
   - Confusion matrix
   - Regression metrics (MAE, RMSE, R²)
   - Feature importance
4. **Generates graphs**:
   - Classification performance (confusion matrix, precision/recall)
   - Regression performance (scatter plots, residuals)
   - Criticality distribution
   - Learning curves and feature importance

### Output

Graphs saved to `performance_evaluation/`:
- `01_classification_performance.png`
- `02_regression_performance.png`
- `03_criticality_distribution.png`
- `04_learning_features.png`

---

## 🌐 Method 3: Test via REST API

### Start the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Test with curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/image.jpg"
```

### Test with Python

```python
import requests

with open("HighLoad/IMG-20251105-WA0086.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(result)
```

### API Response

```json
{
    "load_category": "High Load",
    "criticality_score": 0.8234
}
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

---

## 🧪 Method 4: Test Programmatically (Python)

### Simple Test Function

```python
from busbar.features import preprocess_image_to_features
from busbar.model_criticality_based import CriticalityBasedModel
import cv2

# Load image
img = cv2.imread("HighLoad/IMG-20251105-WA0086.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract features
feats, debug = preprocess_image_to_features(
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=20.0,
    max_temp_c=120.0
)

# Load model
model = CriticalityBasedModel.load("artifacts_criticality")

# Predict
criticality = model.predict_criticality(feats.reshape(1, -1))[0]
load_category = model.criticality_to_class(criticality)

print(f"Criticality: {criticality:.4f}")
print(f"Category: {load_category}")
```

### Batch Testing

```python
import os
from pathlib import Path
from busbar.features import preprocess_image_to_features
from busbar.model_criticality_based import CriticalityBasedModel
import cv2

# Load model once
model = CriticalityBasedModel.load("artifacts_criticality")

# Test all images in a folder
image_dir = Path("HighLoad")
results = []

for image_path in image_dir.glob("*.jpg"):
    # Load and process
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats, _ = preprocess_image_to_features(img_rgb, mode="rgb_pseudocolor")
    
    # Predict
    criticality = model.predict_criticality(feats.reshape(1, -1))[0]
    load_category = model.criticality_to_class(criticality)
    
    results.append({
        "image": image_path.name,
        "criticality": criticality,
        "category": load_category
    })
    print(f"{image_path.name}: {criticality:.4f} → {load_category}")

# Summary
print(f"\nTested {len(results)} images")
print(f"High Load: {sum(1 for r in results if r['category'] == 'High Load')}")
print(f"Medium Load: {sum(1 for r in results if r['category'] == 'Medium Load')}")
print(f"Low Load: {sum(1 for r in results if r['category'] == 'Low Load')}")
```

---

## ✅ Testing Checklist

### Before Testing

- [ ] Model is trained (`artifacts_criticality/regressor.joblib` exists)
- [ ] Test images are thermal images (not regular photos)
- [ ] Images are in supported formats (PNG, JPG, JPEG)

### During Testing

- [ ] Check that image loads correctly
- [ ] Verify thermal image validation passes
- [ ] Confirm features are extracted (6-D vector)
- [ ] Check that prediction completes without errors

### After Testing

- [ ] Verify criticality score is between 0.0 and 1.0
- [ ] Check that load category matches expected range
- [ ] Review temperature statistics (should be reasonable: 20-120°C)
- [ ] Compare results with known labels (if available)

---

## 🔍 Understanding Test Results

### Criticality Score Interpretation

| Score Range | Category | Meaning |
|-------------|----------|---------|
| 0.0 - 0.33 | Low Load | Normal operation, no immediate concern |
| 0.33 - 0.67 | Medium Load | Elevated temperature, monitor closely |
| 0.67 - 1.0 | High Load | Critical overheating, immediate action required |

### Example Results

**Low Load Example:**
```
Criticality Score: 0.15
Load Category: Low Load
Interpretation: 🟢 LOW RISK - Normal operation
```

**Medium Load Example:**
```
Criticality Score: 0.52
Load Category: Medium Load
Interpretation: 🟡 MEDIUM RISK - Monitor closely
```

**High Load Example:**
```
Criticality Score: 0.85
Load Category: High Load
Interpretation: 🔴 CRITICAL - Immediate attention required
```

---

## 🐛 Troubleshooting

### Error: Model directory not found

**Problem**: `ERROR: Model directory not found: artifacts_criticality`

**Solution**: Train the model first:
```bash
python train_criticality_based.py --csv "dataset/labels.csv" --roots "Lowload" "HighLoad"
```

### Error: Image not found

**Problem**: `ERROR: Image not found: path/to/image.jpg`

**Solution**: 
- Check that the image path is correct
- Use absolute path or relative path from project root
- Verify image file exists

### Error: Cannot read image

**Problem**: `ERROR: Could not read image. Supported formats: PNG, JPG, JPEG`

**Solution**:
- Check image format (must be PNG, JPG, or JPEG)
- Verify image file is not corrupted
- Try opening the image in an image viewer first

### Error: Image does not appear to be a thermal image

**Problem**: `REJECTED: Image does not appear to be a thermal image`

**Solution**:
- Verify the image is actually a thermal image
- Use `--warn-only` flag to process anyway:
  ```bash
  python test_criticality_based.py "image.jpg" --warn-only
  ```

### Error: Features extracted but prediction fails

**Problem**: Error during inference

**Solution**:
- Check that model was trained with same feature extraction settings
- Verify model file is not corrupted
- Re-train the model if necessary

---

## 📝 Quick Test Examples

### Test with sample images

```bash
# Test Low Load image
python test_criticality_based.py "Lowload/IMG-20251105-WA0117.jpg"

# Test High Load image
python test_criticality_based.py "HighLoad/IMG-20251105-WA0086.jpg"

# Test from RealImageDataset
python test_criticality_based.py "RealImageDataset/bb10.jpeg"
```

### Test multiple images

```bash
# Windows PowerShell
Get-ChildItem "HighLoad\*.jpg" | ForEach-Object { python test_criticality_based.py $_.FullName }

# Linux/Mac
for img in HighLoad/*.jpg; do python test_criticality_based.py "$img"; done
```

---

## 🎯 Best Practices

1. **Always validate thermal images**: The model is designed for thermal images, not regular photos
2. **Check temperature statistics**: Min/max/mean should be reasonable (20-120°C)
3. **Compare with known labels**: If you have ground truth, compare predictions
4. **Use performance evaluation**: Run `evaluate_model_performance.py` for comprehensive metrics
5. **Test edge cases**: Try images with different temperature ranges and patterns

---

## 📚 Additional Resources

- **Model Overview**: See `MODEL_OVERVIEW.md`
- **Model Flow**: See `MODEL_FLOW_EXPLANATION.md`
- **Preprocessing**: See `PREPROCESSING_EXPLANATION.md`
- **Training Guide**: See `TRAINING_GUIDE.md`
- **Performance Evaluation**: See `PERFORMANCE_EVALUATION_GUIDE.md`

---

## 💡 Tips

- **Start simple**: Test with a single known image first
- **Use verbose output**: The test script provides detailed information
- **Check debug info**: Temperature statistics help verify preprocessing
- **Compare results**: Test multiple images to see consistency
- **Validate inputs**: Ensure images are actually thermal images

Happy testing! 🔥

