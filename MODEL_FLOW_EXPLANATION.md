# Complete Model Flow Explanation

This document explains the **complete end-to-end flow** of the Busbar Heat Detection model, showing exactly where each step is implemented in the codebase.

---

## 🔄 Complete Pipeline Overview

```
INPUT: Thermal Image (RGB, PNG/JPG)
    ↓
[STAGE 1] Image Loading & Validation
    File: test_criticality_based.py, api.py
    ↓
[STAGE 2] Image → Temperature Matrix Conversion
    File: busbar/preprocessing.py
    Function: image_to_temperature_matrix()
    ↓
[STAGE 3] Extract Column Signals
    File: busbar/preprocessing.py
    Function: extract_column_signals()
    ↓
[STAGE 4] Compute Hjorth Parameters
    File: busbar/preprocessing.py
    Functions: hjorth_parameters(), hjorth_parameters_per_signal()
    ↓
[STAGE 5] Aggregate Features (6-D Vector)
    File: busbar/features.py
    Function: aggregate_hjorth_features()
    ↓
[STAGE 6] Feature Extraction (Main Function)
    File: busbar/features.py
    Function: preprocess_image_to_features()
    ↓
[STAGE 7] Model Prediction
    File: busbar/model_criticality_based.py
    Functions: predict_criticality(), criticality_to_class()
    ↓
OUTPUT: {load_category, criticality_score}
```

---

## 📁 Detailed Implementation by Stage

### **STAGE 1: Image Loading & Validation**

**Location**: `test_criticality_based.py` (lines 43-65), `api.py` (lines 27-33)

**What happens:**
1. Load image from file using OpenCV
2. Convert BGR to RGB
3. Validate that image is actually a thermal image

**Code:**
```python
# test_criticality_based.py, lines 43-56
img = cv2.imread(image_path)  # Load image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR→RGB
should_proceed, validation_msg = validate_before_processing(image_path, img_rgb, strict=strict)
```

**Validation Function:**
- **File**: `busbar/thermal_validator.py`
- **Function**: `validate_before_processing()`
- **Purpose**: Checks if image has thermal image characteristics (color distribution, temperature range)

**Output**: RGB image array (H×W×3), validated

---

### **STAGE 2: Image → Temperature Matrix**

**Location**: `busbar/preprocessing.py` (lines 7-67)

**Function**: `image_to_temperature_matrix()`

**What happens:**
1. Detects image mode (RGB pseudo-color, grayscale, or raw16)
2. Converts RGB to grayscale using luminance weights
3. Normalizes to [0, 1] range
4. Maps to temperature range [min_temp_c, max_temp_c] (default: 20°C to 120°C)

**Code:**
```python
# busbar/preprocessing.py, lines 54-65
if inferred_mode == "rgb_pseudocolor":
    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    # Convert RGB to grayscale using luminance weights
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Map to temperature range
    temp_c = gray * (max_temp_c - min_temp_c) + min_temp_c
    return temp_c.astype(np.float32)
```

**Input**: RGB image (H×W×3), uint8 [0-255] or float [0-1]

**Output**: Temperature matrix (H×W), float32, values in °C

**Example**: 120×160 RGB image → 120×160 temperature matrix with values like 30.5°C, 85.2°C

---

### **STAGE 3: Extract Column Signals**

**Location**: `busbar/preprocessing.py` (lines 70-74)

**Function**: `extract_column_signals()`

**What happens:**
1. Takes temperature matrix (H, W)
2. Transposes to extract each column as a vertical signal
3. Each column becomes a 1D temperature profile

**Code:**
```python
# busbar/preprocessing.py, lines 70-74
def extract_column_signals(temp_c: np.ndarray) -> np.ndarray:
    if temp_c.ndim != 2:
        raise ValueError("temp_c must be 2D (H, W)")
    signals = temp_c.T  # Transpose: columns become rows
    return signals.astype(np.float32)
```

**Input**: Temperature matrix (H×W)

**Output**: Signals array (W×H) - W signals, each of length H

**Why columns?**
- Busbars are typically vertical structures
- Vertical temperature profiles capture heat distribution patterns
- Each column represents a spatial location along busbar width

**Example**: 120×160 temperature matrix → 160 signals × 120 values each

---

### **STAGE 4: Compute Hjorth Parameters**

**Location**: `busbar/preprocessing.py` (lines 77-118)

**Functions**: 
- `hjorth_parameters()` (lines 77-101) - computes for single signal
- `hjorth_parameters_per_signal()` (lines 104-118) - computes for all signals

**What happens:**
For each column signal, compute three Hjorth parameters:

#### 4.1 Activity (Variance)
```python
# busbar/preprocessing.py, line 82
var0 = np.var(x)  # Variance of signal = Activity
```
- **Meaning**: Temperature variation in the signal
- **High activity**: Large temperature variations
- **Low activity**: Uniform temperature

#### 4.2 Mobility (Rate of Change)
```python
# busbar/preprocessing.py, lines 86-88
dx = np.diff(x)  # First derivative
var1 = np.var(dx)  # Variance of first derivative
mobility = math.sqrt(var1 / var0)  # Mobility
```
- **Meaning**: Standard deviation of first derivative
- **High mobility**: Rapid temperature changes
- **Low mobility**: Gradual temperature changes

#### 4.3 Complexity (Signal Irregularity)
```python
# busbar/preprocessing.py, lines 93-99
ddx = np.diff(dx)  # Second derivative
var2 = np.var(ddx)  # Variance of second derivative
complexity = math.sqrt(var2 / var1) / mobility  # Complexity
```
- **Meaning**: Ratio of mobility of first derivative to mobility of signal
- **High complexity**: Complex temperature patterns
- **Low complexity**: Simple temperature patterns

**Code Flow:**
```python
# busbar/preprocessing.py, lines 104-118
def hjorth_parameters_per_signal(signals: np.ndarray):
    n_signals = signals.shape[0]
    activity = np.zeros(n_signals, dtype=np.float32)
    mobility = np.zeros(n_signals, dtype=np.float32)
    complexity = np.zeros(n_signals, dtype=np.float32)
    
    for i in range(n_signals):
        a, m, c = hjorth_parameters(signals[i])  # Compute for each signal
        activity[i] = a
        mobility[i] = m
        complexity[i] = c
    
    return activity, mobility, complexity
```

**Input**: Signals array (W×H) - 160 signals

**Output**: 
- `activity`: array of length W (160 values)
- `mobility`: array of length W (160 values)
- `complexity`: array of length W (160 values)

**Example**: 160 signals → 160 activity values, 160 mobility values, 160 complexity values

---

### **STAGE 5: Aggregate Features (6-D Vector)**

**Location**: `busbar/features.py` (lines 12-27)

**Function**: `aggregate_hjorth_features()`

**What happens:**
1. Computes mean and standard deviation of each Hjorth parameter
2. Creates 6-dimensional feature vector

**Code:**
```python
# busbar/features.py, lines 12-27
def aggregate_hjorth_features(
    activity: np.ndarray,
    mobility: np.ndarray,
    complexity: np.ndarray,
) -> np.ndarray:
    def safe_stats(x: np.ndarray):
        if x.size == 0:
            return 0.0, 0.0
        return float(np.nanmean(x)), float(np.nanstd(x))
    
    a_mean, a_std = safe_stats(activity)      # Mean and std of activity
    m_mean, m_std = safe_stats(mobility)       # Mean and std of mobility
    c_mean, c_std = safe_stats(complexity)    # Mean and std of complexity
    
    features = np.array([a_mean, a_std, m_mean, m_std, c_mean, c_std], dtype=np.float32)
    return features
```

**Input**: 
- `activity`: array of 160 values
- `mobility`: array of 160 values
- `complexity`: array of 160 values

**Output**: 6-dimensional feature vector:
```
[mean(activity), std(activity),
 mean(mobility), std(mobility),
 mean(complexity), std(complexity)]
```

**Example**: 
- Input: 160 activity values, 160 mobility values, 160 complexity values
- Output: `[0.45, 0.12, 0.23, 0.08, 0.67, 0.15]` (6 values)

---

### **STAGE 6: Feature Extraction (Main Function)**

**Location**: `busbar/features.py` (lines 30-54)

**Function**: `preprocess_image_to_features()`

**What happens:**
This is the **main entry point** that orchestrates all preprocessing steps:

1. Calls `image_to_temperature_matrix()` (Stage 2)
2. Calls `extract_column_signals()` (Stage 3)
3. Calls `hjorth_parameters_per_signal()` (Stage 4)
4. Calls `aggregate_hjorth_features()` (Stage 5)
5. Returns features and debug information

**Code:**
```python
# busbar/features.py, lines 30-54
def preprocess_image_to_features(
    image: np.ndarray,
    mode: str = "auto",
    min_temp_c: float = 20.0,
    max_temp_c: float = 120.0,
    calibration_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Stage 2: Image → Temperature
    temp_c = image_to_temperature_matrix(
        image=image,
        mode=mode,
        min_temp_c=min_temp_c,
        max_temp_c=max_temp_c,
        calibration_fn=calibration_fn,
    )
    
    # Stage 3: Extract Column Signals
    signals = extract_column_signals(temp_c)
    
    # Stage 4: Compute Hjorth Parameters
    activity, mobility, complexity = hjorth_parameters_per_signal(signals)
    
    # Stage 5: Aggregate to 6-D Features
    features = aggregate_hjorth_features(activity, mobility, complexity)
    
    # Debug information
    debug = {
        "temp_c": temp_c,
        "signals": signals,
        "activity": activity,
        "mobility": mobility,
        "complexity": complexity,
    }
    return features, debug
```

**Input**: RGB image (H×W×3)

**Output**: 
- `features`: 6-D feature vector (numpy array)
- `debug`: Dictionary with intermediate results for debugging

**Usage:**
```python
# test_criticality_based.py, lines 80-85
feats, debug = preprocess_image_to_features(
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=20.0,
    max_temp_c=120.0
)
```

---

### **STAGE 7: Model Prediction**

**Location**: `busbar/model_criticality_based.py` (lines 50-73)

**Functions**: 
- `predict_criticality()` (lines 50-52) - predicts criticality score
- `criticality_to_class()` (lines 54-61) - converts to load category
- `predict()` (lines 63-73) - main prediction function

**What happens:**

#### 7.1 Predict Criticality Score
```python
# busbar/model_criticality_based.py, lines 50-52
def predict_criticality(self, X: np.ndarray) -> np.ndarray:
    """Predict criticality scores only"""
    return np.clip(self.reg.predict(X), 0.0, 1.0)
```

**Model Architecture:**
```python
# busbar/model_criticality_based.py, lines 34-37
self.reg = Pipeline([
    ("scaler", StandardScaler()),  # Normalize features
    ("rf", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state))
])
```

**Process:**
1. **StandardScaler**: Normalizes 6-D features (mean=0, std=1)
2. **RandomForest Regressor**: 300 trees, predicts criticality (0.0-1.0)
3. **Clipping**: Ensures output is in [0.0, 1.0] range

**Input**: 6-D feature vector (1×6 array)

**Output**: Criticality score (float, 0.0-1.0)

#### 7.2 Derive Classification from Criticality
```python
# busbar/model_criticality_based.py, lines 54-61
def criticality_to_class(self, criticality: float) -> str:
    """Convert criticality score to load category"""
    if criticality < self.low_threshold:      # < 0.33
        return "Low Load"
    elif criticality < self.medium_threshold:  # 0.33 - 0.67
        return "Medium Load"
    else:                                     # >= 0.67
        return "High Load"
```

**Thresholds** (default):
- **Low Load**: criticality < 0.33
- **Medium Load**: 0.33 ≤ criticality < 0.67
- **High Load**: criticality ≥ 0.67

#### 7.3 Main Prediction Function
```python
# busbar/model_criticality_based.py, lines 63-73
def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict both classification and criticality
    
    Returns:
        (load_categories, criticality_scores)
        Classification is derived from criticality scores
    """
    criticality = self.predict_criticality(X)  # Predict criticality first
    categories = np.array([self.criticality_to_class(c) for c in criticality])  # Derive classification
    return categories, criticality
```

**Usage:**
```python
# test_criticality_based.py, lines 105-109
criticality = model.predict_criticality(feats.reshape(1, -1))[0]  # Predict criticality
load_category = model.criticality_to_class(criticality)  # Derive classification
```

**Input**: 6-D feature vector (1×6)

**Output**: 
- `load_category`: "Low Load", "Medium Load", or "High Load"
- `criticality_score`: float (0.0-1.0)

---

## 🔄 Complete Code Flow Example

Here's how all stages connect in actual usage:

### **Inference Flow** (`test_criticality_based.py`)

```python
# 1. Load image
img = cv2.imread(image_path)  # Line 44
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Line 52

# 2. Validate
validate_before_processing(image_path, img_rgb, strict=strict)  # Line 56

# 3. Extract features (calls all preprocessing stages)
feats, debug = preprocess_image_to_features(  # Line 80
    img_rgb,
    mode="rgb_pseudocolor",
    min_temp_c=20.0,
    max_temp_c=120.0
)
# This internally calls:
#   - image_to_temperature_matrix()      → Stage 2
#   - extract_column_signals()            → Stage 3
#   - hjorth_parameters_per_signal()      → Stage 4
#   - aggregate_hjorth_features()         → Stage 5

# 4. Load model
model = CriticalityBasedModel.load(model_dir)  # Line 70

# 5. Predict
criticality = model.predict_criticality(feats.reshape(1, -1))[0]  # Line 106
load_category = model.criticality_to_class(criticality)  # Line 109

# 6. Return result
return {"load_category": load_category, "criticality_score": criticality}  # Line 144
```

### **Training Flow** (`train_criticality_based.py`)

```python
# 1. Load dataset
df = pd.read_csv(csv_path)  # Line 171

# 2. Extract features for each image
for _, row in df.iterrows():
    feats = compute_features_for_row(row)  # Line 36
    # This calls preprocess_image_to_features() internally
    X_list.append(feats)
    y_reg.append(float(row["criticality"]))

# 3. Prepare data
X = np.vstack(X_list)  # Line 46
y_reg = np.array(y_reg)  # Line 47

# 4. Train/test split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)  # Line 61

# 5. Train model
model = CriticalityBasedModel(random_state=42, low_threshold=0.33, medium_threshold=0.67)  # Line 84
model.fit(X_train, y_reg_train)  # Line 87

# 6. Evaluate
y_reg_pred = model.predict_criticality(X_test)  # Line 90
y_cls_pred = np.array([model.criticality_to_class(c) for c in y_reg_pred])  # Line 91

# 7. Save model
model.save(args.artifacts_dir)  # Line 211
```

---

## 📊 Data Flow Summary

| Stage | Input Shape | Output Shape | File | Function |
|-------|------------|--------------|------|----------|
| **1. Load Image** | File path | (H, W, 3) RGB | `test_criticality_based.py` | `cv2.imread()` |
| **2. Image→Temp** | (H, W, 3) RGB | (H, W) °C | `busbar/preprocessing.py` | `image_to_temperature_matrix()` |
| **3. Extract Signals** | (H, W) °C | (W, H) signals | `busbar/preprocessing.py` | `extract_column_signals()` |
| **4. Hjorth Params** | (W, H) signals | 3×(W,) arrays | `busbar/preprocessing.py` | `hjorth_parameters_per_signal()` |
| **5. Aggregate** | 3×(W,) arrays | (6,) vector | `busbar/features.py` | `aggregate_hjorth_features()` |
| **6. Main Feature** | (H, W, 3) RGB | (6,) vector | `busbar/features.py` | `preprocess_image_to_features()` |
| **7. Predict** | (6,) vector | (1,) score | `busbar/model_criticality_based.py` | `predict_criticality()` |
| **8. Classify** | (1,) score | string | `busbar/model_criticality_based.py` | `criticality_to_class()` |

---

## 🔑 Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `busbar/preprocessing.py` | Image processing & Hjorth parameters | `image_to_temperature_matrix()`, `extract_column_signals()`, `hjorth_parameters()` |
| `busbar/features.py` | Feature extraction orchestration | `preprocess_image_to_features()`, `aggregate_hjorth_features()` |
| `busbar/model_criticality_based.py` | Model definition & prediction | `CriticalityBasedModel`, `predict_criticality()`, `criticality_to_class()` |
| `busbar/thermal_validator.py` | Input validation | `validate_before_processing()` |
| `test_criticality_based.py` | Inference script | `test_criticality_based()` |
| `train_criticality_based.py` | Training script | `train_evaluate()`, `main()` |
| `api.py` | REST API server | `predict_api()` |

---

## 🎯 Summary

The model flow is:

1. **Image Loading** → OpenCV loads RGB image
2. **Temperature Extraction** → RGB converted to temperature matrix
3. **Signal Extraction** → Column-wise temperature profiles
4. **Hjorth Parameters** → Activity, Mobility, Complexity computed
5. **Feature Aggregation** → 6-D feature vector created
6. **Model Prediction** → RandomForest predicts criticality (0-1)
7. **Classification** → Load category derived from criticality

All preprocessing is in `busbar/preprocessing.py` and `busbar/features.py`, while the model logic is in `busbar/model_criticality_based.py`.

