# Model Summary - Quick Reference

## 🔄 End-to-End Flow

```
THERMAL IMAGE (RGB, 120×160×3)
    ↓
[1] Convert to Temperature Matrix (120×160, °C)
    ↓
[2] Extract Column Signals (160 signals × 120 values)
    ↓
[3] Compute Hjorth Parameters per Column
    - Activity (variance)
    - Mobility (rate of change)
    - Complexity (irregularity)
    ↓
[4] Aggregate to 6 Features
    [mean(activity), std(activity),
     mean(mobility), std(mobility),
     mean(complexity), std(complexity)]
    ↓
[5] Normalize Features (StandardScaler)
    ↓
[6] RandomForest Models
    ├─→ Classifier → "High Load"
    └─→ Regressor → 0.82
    ↓
OUTPUT: {"load_category": "High Load", "criticality_score": 0.82}
```

## 📥 INPUT

- **Type**: Thermal image (RGB pseudo-color or grayscale)
- **Format**: PNG, JPG (via OpenCV)
- **Size**: Any dimensions (e.g., 120×160, 640×480)
- **Content**: Temperature encoded as colors (blue=cold, red=hot)

## 📤 OUTPUT

```json
{
  "load_category": "High Load",      // "Low Load" | "Medium Load" | "High Load"
  "criticality_score": 0.82           // 0.0 (safe) to 1.0 (critical)
}
```

## 🔑 Key Transformations

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| Image → Temp | RGB (H×W×3) | Temp (H×W) | Extract temperature values |
| Extract Signals | Temp (H×W) | Signals (W×H) | Get vertical profiles |
| Hjorth Params | Signals (W×H) | 3×(W,) | Characterize patterns |
| Aggregate | 3×(W,) | (6,) | Reduce dimensionality |
| Normalize | (6,) | (6,) | Standardize features |
| Predict | (6,) | 2 outputs | Classification + Regression |

## 🧠 Model Architecture

**Two RandomForest Models** sharing same 6-D input:

- **Classifier**: 300 trees → 3 classes (Low/Medium/High)
- **Regressor**: 300 trees → continuous score (0-1)

## 💡 Why This Design?

1. **Signal Processing**: Hjorth parameters capture spatial temperature patterns
2. **Efficient**: 6 features vs 19,200 pixels (99.97% reduction)
3. **Domain-Specific**: Column-wise analysis matches busbar geometry
4. **Dual Output**: Single pipeline predicts both category and risk score
5. **Fast**: No GPU needed, runs on CPU

## 📊 Feature Meaning

- **Activity**: Overall temperature variation
- **Mobility**: Rate of temperature change
- **Complexity**: Irregularity of temperature patterns
- **Mean/Std**: Captures both average and variability

## 🎯 Interpretation Guide

### Load Categories
- **Low Load**: Normal operation, safe temperatures
- **Medium Load**: Elevated temperatures, monitor
- **High Load**: Critical temperatures, take action

### Criticality Scores
- **0.0 - 0.3**: Low risk ✅
- **0.3 - 0.6**: Medium risk ⚠️
- **0.6 - 0.8**: High risk 🔴
- **0.8 - 1.0**: Critical 🚨

---

**For detailed explanation, see [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md)**

