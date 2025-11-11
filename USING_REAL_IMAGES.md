# Using Real Thermal Images with the Model

## Quick Start

### Method 1: Simple Test Script (Recommended)

```bash
python test_real_image.py <path_to_your_image>
```

**Example:**
```bash
# Test with a FLIR image
python test_real_image.py my_thermal_image.jpg

# Test with full path
python test_real_image.py C:/Users/YourName/Desktop/thermal_photo.png

# Specify custom model directory
python test_real_image.py image.jpg custom_artifacts_folder
```

### Method 2: Using the Inference Script

```bash
python infer.py <image_path> artifacts
```

**Example:**
```bash
python infer.py my_thermal_image.jpg artifacts
```

---

## Image Requirements

### Supported Formats
- **PNG** (recommended)
- **JPG/JPEG**
- **Any format supported by OpenCV**

### Image Types
The model works with:
- ✅ **RGB pseudo-color thermal images** (like FLIR camera outputs)
- ✅ **Grayscale thermal images**
- ✅ **Any thermal image where colors represent temperature**

### Image Size
- **Any size** - The model automatically processes any dimensions
- **Recommended**: Similar to training data (240×320 or larger)
- The model extracts features regardless of size

---

## Step-by-Step Guide

### 1. Prepare Your Image

Place your thermal image in a convenient location:
```
C:/Users/YourName/Desktop/my_thermal.jpg
```

Or use an existing image from the dataset:
```
dataset/flir_thermal_0015.png
```

### 2. Run Inference

**Option A: Detailed Output (Recommended)**
```bash
python test_real_image.py my_thermal.jpg
```

This shows:
- Image loading status
- Feature extraction details
- Temperature statistics
- Prediction results
- Risk interpretation

**Option B: Quick JSON Output**
```bash
python infer.py my_thermal.jpg artifacts
```

Output:
```json
{'load_category': 'High Load', 'criticality_score': 0.82}
```

### 3. Interpret Results

**Load Categories:**
- `"Low Load"` - Normal operation, safe temperatures
- `"Medium Load"` - Elevated temperatures, monitor
- `"High Load"` - Critical temperatures, take action

**Criticality Scores:**
- **0.0 - 0.3**: 🟢 Low risk - Normal operation
- **0.3 - 0.6**: 🟡 Medium risk - Monitor closely
- **0.6 - 0.8**: 🟠 High risk - Take action
- **0.8 - 1.0**: 🔴 Critical - Immediate attention required

---

## Examples

### Example 1: Test with Dataset Image
```bash
python test_real_image.py dataset/flir_thermal_0015.png
```

**Expected Output:**
```
======================================================================
Testing Real Thermal Image
======================================================================

Image: dataset/flir_thermal_0015.png
Model: artifacts

[1] Loading image...
  ✓ Image loaded: 320×240 pixels, 3 channels

[2] Loading trained model...
  ✓ Model loaded successfully

[3] Extracting features from image...
  ✓ Features extracted: (6,)
    Feature values: [14.5, 2.3, 0.48, 0.05, 1.3, 0.2]

  Temperature Statistics:
    Min: 25.3°C
    Max: 108.2°C
    Mean: 45.6°C
    Std: 18.2°C

[4] Running inference...
  ✓ Prediction complete

======================================================================
RESULTS
======================================================================

  Load Category:     High Load
  Criticality Score: 0.7696

  Interpretation:
    🟠 HIGH RISK - Take action

  JSON Output:
    {'load_category': 'High Load', 'criticality_score': 0.7696}
```

### Example 2: Using Python API

```python
from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel
import cv2

# Load image
img = cv2.imread("my_thermal_image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load model
model = MultiHeadModel.load("artifacts")

# Extract features
feats, _ = preprocess_image_to_features(
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=20.0,
    max_temp_c=120.0
)

# Predict
load_category, criticality = model.predict(feats.reshape(1, -1))

print(f"Load Category: {load_category[0]}")
print(f"Criticality: {criticality[0]:.4f}")
```

### Example 3: Batch Processing

```python
import os
from pathlib import Path
from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel
import cv2

model = MultiHeadModel.load("artifacts")

# Process all images in a folder
image_folder = "my_thermal_images"
results = []

for img_file in Path(image_folder).glob("*.jpg"):
    img = cv2.imread(str(img_file))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    feats, _ = preprocess_image_to_features(
        img_rgb, mode="rgb_pseudocolor", min_temp_c=20, max_temp_c=120
    )
    
    load_cat, crit = model.predict(feats.reshape(1, -1))
    
    results.append({
        "file": img_file.name,
        "load_category": load_cat[0],
        "criticality": float(crit[0])
    })
    
    print(f"{img_file.name}: {load_cat[0]}, {crit[0]:.4f}")
```

---

## Troubleshooting

### Issue: "Image not found"
**Solution**: Check the file path. Use absolute path if needed:
```bash
python test_real_image.py "C:/full/path/to/image.jpg"
```

### Issue: "Could not read image"
**Solution**: 
- Check file format (PNG, JPG supported)
- Verify file is not corrupted
- Try converting to PNG format

### Issue: "Model directory not found"
**Solution**: 
- Ensure you've run `python train.py` first
- Check that `artifacts/` folder exists
- Specify correct path: `python test_real_image.py image.jpg path/to/artifacts`

### Issue: Unexpected predictions
**Possible causes:**
- Image temperature range different from training (20-120°C)
- Image format/colormap different from FLIR
- Image quality/resolution issues

**Solutions:**
- Adjust `min_temp_c` and `max_temp_c` parameters
- Try different `mode` ("auto", "grayscale", "rgb_pseudocolor")
- Ensure image represents thermal data correctly

---

## Advanced Usage

### Custom Temperature Range

If your thermal images use a different temperature range:

```python
from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel
import cv2

img = cv2.imread("my_image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model = MultiHeadModel.load("artifacts")

# Adjust temperature range to match your camera
feats, _ = preprocess_image_to_features(
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=0.0,    # Your minimum temperature
    max_temp_c=150.0   # Your maximum temperature
)

load_cat, crit = model.predict(feats.reshape(1, -1))
```

### Using ONNX Models

```bash
python infer.py my_image.jpg artifacts --onnx
```

### API Endpoint

Start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then POST your image:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@my_thermal_image.jpg"
```

---

## Tips for Best Results

1. **Image Quality**: Use high-resolution images for better feature extraction
2. **Temperature Range**: Match your camera's range to training data (20-120°C)
3. **Color Mapping**: Ensure thermal colormap is similar to FLIR (purple/blue to orange/yellow)
4. **Multiple Tests**: Test several images to verify consistent results
5. **Calibration**: If you have temperature calibration data, use it to adjust min/max temps

---

## Next Steps

- Test with your own thermal images
- Adjust temperature ranges if needed
- Fine-tune model with your specific camera data
- Deploy API for production use

