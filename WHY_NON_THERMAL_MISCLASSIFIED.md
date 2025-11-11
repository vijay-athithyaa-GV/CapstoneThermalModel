# Why Non-Thermal Images Are Misclassified

## The Problem

When you feed a **non-thermal image** (regular photo, diagram, etc.) to the model, it still processes it and may classify it as "High Load" or another category. This happens because:

---

## Root Cause Analysis

### 1. **Preprocessing Assumes Thermal Images**

The preprocessing pipeline converts **ANY RGB image** to a temperature matrix:

```python
# Current process (in preprocess_image_to_features):
1. RGB Image → Grayscale (using luminance weights)
2. Normalize to [0, 1]
3. Map to temperature range [20°C, 120°C]
```

**Problem**: This works on ANY image, not just thermal images!

**Example**:
- Regular photo with bright colors → High grayscale values → Mapped to high temperatures (80-120°C)
- Diagram with red/orange colors → High grayscale → High temperatures
- Result: Model sees "high temperature" features → Predicts "High Load"

### 2. **No Input Validation**

The current code doesn't check if the input is actually a thermal image before processing:

```python
# Current test_real_image.py:
img = cv2.imread(image_path)  # Loads ANY image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
feats, _ = preprocess_image_to_features(img_rgb)  # Processes ANY image
model.predict(feats)  # Classifies ANY image
```

**No validation** → Model processes everything → Misclassifies non-thermal images

### 3. **Model Trained Only on Thermal Images**

The model was trained **exclusively** on thermal images:
- FLIR camera outputs
- Pseudo-color thermal images
- Temperature-encoded images

**Result**: Model learned patterns specific to thermal images. When given a regular photo:
- Model tries to interpret it as thermal data
- Arbitrary RGB values get mapped to temperatures
- Model makes predictions based on incorrect assumptions

---

## Why It Gives "High Load" for Non-Thermal Images

### Common Scenarios:

1. **Bright/Colorful Images**:
   - High RGB values → High grayscale → High temperature mapping
   - Model sees "high temperature" → Predicts "High Load"

2. **Red/Orange Images**:
   - Red/orange colors → High R channel → High grayscale
   - Thermal images often use red for hot → Model interprets as hot
   - Predicts "High Load"

3. **High Contrast Images**:
   - Sharp edges → High variance in Hjorth features
   - Model interprets as high activity/complexity
   - May predict "High Load"

4. **Diagrams/System Drawings**:
   - Often use red/orange for warnings/hot components
   - Model sees these colors → Interprets as thermal hot spots
   - Predicts "High Load"

---

## The Solution

### 1. **Thermal Image Validation** (Implemented)

Added `busbar/thermal_validator.py` that checks:
- ✅ Color palette (thermal images have specific color ranges)
- ✅ Gradient smoothness (thermal images have smooth transitions)
- ✅ Color diversity (thermal images have limited palette)
- ✅ Intensity distribution (thermal images have specific brightness patterns)

### 2. **Enhanced Test Script** (Created)

`test_with_validation.py`:
- ✅ Validates image before processing
- ✅ Rejects non-thermal images (or warns)
- ✅ Only processes validated thermal images

---

## How to Use the Fix

### Option 1: Use Validated Test Script (Recommended)

```bash
# Strict mode (rejects non-thermal images)
python test_with_validation.py your_image.jpg

# Warn-only mode (warns but still processes)
python test_with_validation.py your_image.jpg --warn-only
```

### Option 2: Update Existing Scripts

Add validation to `test_real_image.py`:

```python
from busbar.thermal_validator import validate_before_processing

# After loading image:
should_proceed, msg = validate_before_processing(image_path, img_rgb, strict=True)
if not should_proceed:
    print(f"REJECTED: {msg}")
    return None
```

---

## Technical Details

### How Preprocessing Works (Current)

```
Regular Photo (RGB)
    ↓
Convert to Grayscale
    ↓
Normalize [0, 1]
    ↓
Map to Temperature [20°C, 120°C]  ← PROBLEM: Works on ANY image!
    ↓
Extract Hjorth Features
    ↓
Model Prediction
```

### What Should Happen

```
Input Image
    ↓
Validate: Is this a thermal image?
    ↓
    ├─ NO → REJECT (don't process)
    └─ YES → Continue
            ↓
        Preprocess (convert to temperature)
            ↓
        Extract Features
            ↓
        Model Prediction
```

---

## Validation Criteria

The validator checks:

1. **Color Palette** (40% weight):
   - Thermal images: Red/orange (warm) OR blue/purple (cool)
   - Non-thermal: Often has greens, diverse colors

2. **Gradient Smoothness** (30% weight):
   - Thermal images: Smooth temperature transitions
   - Non-thermal: Sharp edges, high contrast

3. **Saturation** (20% weight):
   - Thermal images: Moderate to high saturation
   - Non-thermal: Variable saturation

4. **Brightness** (10% weight):
   - Thermal images: Moderate brightness
   - Non-thermal: Can be very dark or very bright

**Confidence Score**: 0.0 (not thermal) to 1.0 (definitely thermal)
**Threshold**: 0.3 (default) - images below this are rejected

---

## Examples

### ✅ Valid Thermal Image
```
Image: HighLoad/IMG-20251105-WA0086.jpg
Validation: VALIDATED: Looks like thermal image (confidence: 0.85)
Result: Processed → High Load, 0.9958
```

### ❌ Invalid (Regular Photo)
```
Image: regular_photo.jpg
Validation: REJECTED: May not be thermal: unusual color palette, sharp edges (confidence: 0.15)
Result: NOT PROCESSED
```

### ⚠️ Borderline (Warn Mode)
```
Image: borderline_image.jpg
Validation: WARNING: May not be thermal: low confidence (confidence: 0.25). Proceeding anyway...
Result: Processed (but may be inaccurate)
```

---

## Best Practices

1. **Always validate** before processing
2. **Use strict mode** for production
3. **Check confidence scores** - low confidence = unreliable
4. **Train model** only on thermal images
5. **Document** that model is for thermal images only

---

## Summary

**Why non-thermal images are misclassified:**
- Preprocessing converts ANY image to temperature
- No validation before processing
- Model trained only on thermal images

**Solution:**
- Added thermal image validator
- Enhanced test script with validation
- Rejects non-thermal images before processing

**Use:**
```bash
python test_with_validation.py your_image.jpg
```

This will now **reject non-thermal images** and only process actual thermal/IR camera images!

