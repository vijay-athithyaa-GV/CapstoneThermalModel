# Model Architecture Comparison

## Overview

There are **two different model architectures** in this project:

1. **CriticalityBasedModel** (Currently Used) - **1 MODEL**
2. **MultiHeadModel** (Original) - **2 MODELS**

---

## 1. CriticalityBasedModel (Current) - 1 MODEL

### Architecture
- **Single Model**: RandomForestRegressor
- **Output**: Criticality score (0-1)
- **Classification**: Derived from criticality using thresholds (not a separate model)

### How It Works

```
Input Image
    ↓
Feature Extraction (6-D Hjorth features)
    ↓
RandomForestRegressor (1 MODEL)
    ↓
Criticality Score (0-1)
    ↓
Classification (derived from thresholds):
    - 0.0 - 0.33 → Low Load
    - 0.33 - 0.67 → Medium Load
    - 0.67 - 1.0 → High Load
```

### Code Location
- **Model**: `busbar/model_criticality_based.py`
- **Training**: `train_criticality_based.py`
- **Testing**: `test_criticality_based.py`
- **Artifacts**: `artifacts_criticality/`

### Files Saved
```
artifacts_criticality/
  ├── regressor.joblib        (RandomForestRegressor)
  └── model_config.joblib     (Thresholds: 0.33, 0.67)
```

### Advantages
- ✅ **Simpler**: Only 1 model to train and maintain
- ✅ **Consistent**: Classification always matches criticality
- ✅ **Faster**: Only 1 prediction needed
- ✅ **Smaller**: Less storage and memory

### Disadvantages
- ❌ **Less flexible**: Classification cannot be independent of criticality
- ❌ **Threshold-dependent**: Classification depends on fixed thresholds

---

## 2. MultiHeadModel (Original) - 2 MODELS

### Architecture
- **Two Models**: 
  - RandomForestClassifier (for classification)
  - RandomForestRegressor (for criticality)
- **Output**: Both classification and criticality (independent predictions)

### How It Works

```
Input Image
    ↓
Feature Extraction (6-D Hjorth features)
    ↓
    ├─→ RandomForestClassifier (MODEL 1)
    │       ↓
    │   Load Category (Low/Medium/High)
    │
    └─→ RandomForestRegressor (MODEL 2)
            ↓
        Criticality Score (0-1)
```

### Code Location
- **Model**: `busbar/model.py`
- **Training**: `train.py`, `train_from_csv.py`
- **Testing**: `test_real_image.py`, `infer.py`
- **API**: `api.py`
- **Artifacts**: `artifacts/`

### Files Saved
```
artifacts/
  ├── classifier.joblib       (RandomForestClassifier)
  ├── regressor.joblib        (RandomForestRegressor)
  └── label_encoder.joblib    (Label encoding)
```

### Advantages
- ✅ **Flexible**: Classification and criticality are independent
- ✅ **More accurate**: Each model specializes in its task
- ✅ **Probabilities**: Can get classification probabilities

### Disadvantages
- ❌ **More complex**: 2 models to train and maintain
- ❌ **Slower**: 2 predictions needed
- ❌ **Larger**: More storage and memory
- ❌ **Inconsistency risk**: Classification and criticality might not match

---

## Comparison Table

| Feature | CriticalityBasedModel | MultiHeadModel |
|---------|----------------------|----------------|
| **Number of Models** | 1 (Regressor) | 2 (Classifier + Regressor) |
| **Classification** | Derived from criticality | Independent prediction |
| **Criticality** | Direct prediction | Independent prediction |
| **Consistency** | Always consistent | May be inconsistent |
| **Speed** | Faster (1 prediction) | Slower (2 predictions) |
| **Storage** | Smaller (~1 file) | Larger (~3 files) |
| **Flexibility** | Less flexible | More flexible |
| **Accuracy** | Good (90.9% test acc) | Good (depends on data) |
| **Current Usage** | ✅ Active | ❌ Not used |

---

## Which Model Is Currently Used?

### Current System (Active)
- **Model**: `CriticalityBasedModel` (1 model)
- **Training**: `train_criticality_based.py`
- **Testing**: `test_criticality_based.py`
- **Artifacts**: `artifacts_criticality/`

### Original System (Not Used)
- **Model**: `MultiHeadModel` (2 models)
- **Training**: `train.py`, `train_from_csv.py`
- **Testing**: `test_real_image.py`, `infer.py`
- **API**: `api.py` (still uses MultiHeadModel)
- **Artifacts**: `artifacts/`

---

## Answer to Your Question

### **Does this use 2 models?**

**Currently: NO** - The system uses **1 model** (CriticalityBasedModel).

The current implementation uses:
- **1 RandomForestRegressor** to predict criticality
- Classification is **derived** from criticality (not a separate model)

However, there is also a **MultiHeadModel** implementation that uses 2 models, but it's not currently being used for the criticality-based system.

---

## Code Examples

### CriticalityBasedModel (1 Model)

```python
from busbar.model_criticality_based import CriticalityBasedModel

# Load model (1 regressor)
model = CriticalityBasedModel.load("artifacts_criticality")

# Predict (1 prediction)
criticality = model.predict_criticality(features)  # 1 model call
category = model.criticality_to_class(criticality)  # Derived, not a model
```

### MultiHeadModel (2 Models)

```python
from busbar.model import MultiHeadModel

# Load model (2 models)
model = MultiHeadModel.load("artifacts")

# Predict (2 predictions)
category, criticality = model.predict(features)  # 2 model calls
```

---

## Recommendation

**Use CriticalityBasedModel (1 model)** because:
1. ✅ Simpler and faster
2. ✅ Classification always matches criticality
3. ✅ Good accuracy (90.9%)
4. ✅ Currently working well for your use case

**Use MultiHeadModel (2 models)** only if:
- You need classification probabilities
- You need classification independent of criticality
- You have specific requirements for separate models

---

## Summary

- **Current System**: **1 MODEL** (CriticalityBasedModel)
- **Original System**: **2 MODELS** (MultiHeadModel) - Not currently used
- **Recommendation**: Stick with 1 model (simpler, faster, consistent)

