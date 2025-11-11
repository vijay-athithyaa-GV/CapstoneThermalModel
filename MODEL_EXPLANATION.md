# Busbar Heat Detection Model - Complete End-to-End Explanation

## Overview

This model processes thermal images from IR cameras to classify busbar thermal load and predict criticality scores. It uses signal processing (Hjorth parameters) combined with machine learning (RandomForest) for multi-output prediction.

---

## 📥 INPUT

### Primary Input: Thermal Image

**Format:**
- **Type**: RGB image (pseudo-color thermal image) or grayscale
- **Dimensions**: Any size (e.g., 120×160, 640×480, etc.)
- **Channels**: 3 (RGB) or 1 (grayscale)
- **Data Type**: uint8 (0-255) or float (0-1)
- **File Format**: PNG, JPG, JPEG (via OpenCV)

**Example:**
```
Input: RGB thermal image (120×160×3)
       Each pixel represents temperature encoded as color
       Blue = cold, Red/Yellow = hot
```

---

## 🔄 PROCESSING PIPELINE

The model processes the input through 5 main stages:

### Stage 1: Image → Temperature Matrix

**Function**: `image_to_temperature_matrix()`

**What happens:**
1. Reads RGB thermal image (pseudo-color)
2. Converts RGB to grayscale using luminance weights:
   ```
   gray = 0.2126×R + 0.7152×G + 0.0722×B
   ```
3. Normalizes grayscale values to [0, 1]
4. Maps to temperature range [min_temp_c, max_temp_c] (default: 20°C to 120°C)
   ```
   temp_c = gray × (max_temp - min_temp) + min_temp
   ```

**Output:**
- **Shape**: (H, W) - 2D temperature matrix
- **Units**: Degrees Celsius (°C)
- **Example**: 120×160 matrix with values like 30.5°C, 85.2°C, etc.

```
Input:  RGB Image (120×160×3)
        ↓
Output: Temperature Matrix (120×160)
        Each value = temperature in °C
```

---

### Stage 2: Extract Column Signals

**Function**: `extract_column_signals()`

**What happens:**
1. Takes temperature matrix (H, W)
2. Extracts each **column** as a vertical temperature profile
3. Each column becomes a 1D signal representing temperature variation along height

**Output:**
- **Shape**: (W, H) - W signals, each of length H
- **Meaning**: Each row is one vertical temperature signal

```
Input:  Temperature Matrix (120×160)
        ↓
        Extract columns vertically
        ↓
Output: Signals Array (160×120)
        - 160 signals (one per column)
        - Each signal has 120 temperature values
```

**Why columns?**
- Busbars are typically vertical structures
- Vertical temperature profiles capture heat distribution patterns
- Each column represents a spatial location along the busbar width

---

### Stage 3: Compute Hjorth Parameters

**Function**: `hjorth_parameters_per_signal()`

**What happens:**
For each column signal, compute three Hjorth parameters:

#### 3.1 Activity
```
Activity = variance(signal)
```
- **Meaning**: Measures overall signal power/variance
- **High Activity**: Large temperature variations
- **Low Activity**: Uniform temperature

#### 3.2 Mobility
```
Mobility = sqrt(variance(diff(signal)) / variance(signal))
```
- **Meaning**: Measures rate of change relative to signal power
- **High Mobility**: Rapid temperature changes
- **Low Mobility**: Gradual temperature changes

#### 3.3 Complexity
```
Complexity = sqrt(variance(diff²(signal)) / variance(diff(signal))) / Mobility
```
- **Meaning**: Measures signal irregularity/roughness
- **High Complexity**: Irregular, complex temperature patterns
- **Low Complexity**: Smooth, predictable patterns

**Output:**
- **Activity array**: (160,) - one value per column
- **Mobility array**: (160,) - one value per column
- **Complexity array**: (160,) - one value per column

```
Input:  Signals (160×120)
        ↓
        For each signal:
          - Compute Activity (variance)
          - Compute Mobility (rate of change)
          - Compute Complexity (irregularity)
        ↓
Output: Activity: (160,)
        Mobility: (160,)
        Complexity: (160,)
```

**Why Hjorth Parameters?**
- Originally used in EEG signal analysis
- Capture temporal/spatial patterns in signals
- Good for characterizing temperature distribution patterns
- Compact representation of signal characteristics

---

### Stage 4: Feature Aggregation

**Function**: `aggregate_hjorth_features()`

**What happens:**
Aggregates 160 column-wise Hjorth parameters into 6 statistics:

1. **mean(activity)** - Average signal power across columns
2. **std(activity)** - Variability of signal power
3. **mean(mobility)** - Average rate of change
4. **std(mobility)** - Variability of rate of change
5. **mean(complexity)** - Average signal irregularity
6. **std(complexity)** - Variability of irregularity

**Output:**
- **Shape**: (6,) - 6-dimensional feature vector
- **Type**: float32 array

```
Input:  Activity: (160,)
        Mobility: (160,)
        Complexity: (160,)
        ↓
        Compute statistics:
          - mean & std for each parameter
        ↓
Output: Feature Vector (6,)
        [mean_activity, std_activity,
         mean_mobility, std_mobility,
         mean_complexity, std_complexity]
```

**Why aggregation?**
- Reduces dimensionality from 160×3 = 480 values to 6 values
- Captures overall image characteristics
- Makes model training feasible
- Preserves important statistical properties

---

### Stage 5: Feature Normalization

**Function**: `StandardScaler()` (inside model pipeline)

**What happens:**
1. Normalizes features to zero mean and unit variance
2. Formula: `(x - mean) / std`
3. Ensures all features are on similar scales

**Output:**
- **Shape**: (6,) - normalized feature vector
- **Range**: Approximately [-3, +3] (standardized)

```
Input:  Feature Vector (6,)
        [raw values, e.g., 45.2, 12.3, 0.85, 0.23, 1.2, 0.4]
        ↓
        StandardScaler: (x - μ) / σ
        ↓
Output: Normalized Vector (6,)
        [standardized values, e.g., 0.5, -0.3, 1.2, -0.8, 0.1, 0.4]
```

---

## 🧠 MODEL ARCHITECTURE

### Multi-Head Model Structure

The model consists of **two separate RandomForest models** sharing the same input features:

```
                    Normalized Features (6,)
                            │
                            ├─────────────────┐
                            │                 │
                            ▼                 ▼
                    ┌──────────────┐  ┌──────────────┐
                    │  Classifier  │  │  Regressor   │
                    │  (RF)        │  │  (RF)        │
                    └──────────────┘  └──────────────┘
                            │                 │
                            ▼                 ▼
                    Load Category      Criticality
                    (Low/Med/High)     Score (0-1)
```

### Classifier Head

**Purpose**: Classify thermal load into 3 categories

**Architecture:**
- **Input**: 6-D normalized feature vector
- **Model**: RandomForestClassifier
  - **Trees**: 300 decision trees
  - **Max Depth**: None (unlimited)
  - **Output**: 3 class probabilities
- **Classes**: 
  - 0 = "Low Load"
  - 1 = "Medium Load"
  - 2 = "High Load"

**Process:**
1. Each tree votes for a class
2. Majority vote determines final class
3. Label encoder converts numeric class to string

### Regressor Head

**Purpose**: Predict continuous criticality score

**Architecture:**
- **Input**: Same 6-D normalized feature vector
- **Model**: RandomForestRegressor
  - **Trees**: 300 decision trees
  - **Max Depth**: None (unlimited)
  - **Output**: Single float value
- **Range**: [0.0, 1.0]
  - 0.0 = No risk
  - 1.0 = Critical

**Process:**
1. Each tree predicts a score
2. Average of all tree predictions
3. Clipped to [0.0, 1.0] range

---

## 📤 OUTPUT

### Final Output Format

**Type**: Dictionary/JSON

**Structure:**
```json
{
  "load_category": "High Load",
  "criticality_score": 0.82
}
```

### Output Components

#### 1. Load Category (Classification)
- **Type**: String
- **Values**: 
  - `"Low Load"`
  - `"Medium Load"`
  - `"High Load"`
- **Source**: Classifier head prediction

#### 2. Criticality Score (Regression)
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Meaning**:
  - **0.0 - 0.3**: Low risk, normal operation
  - **0.3 - 0.6**: Medium risk, monitor closely
  - **0.6 - 0.8**: High risk, take action
  - **0.8 - 1.0**: Critical, immediate attention required
- **Source**: Regressor head prediction

---

## 🔍 COMPLETE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Thermal Image (RGB, 120×160×3)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Image → Temperature Matrix                             │
│ - Convert RGB to grayscale                                       │
│ - Map to temperature range [20°C, 120°C]                          │
│ Output: (120×160) temperature matrix                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Extract Column Signals                                 │
│ - Extract each column as vertical temperature profile            │
│ Output: (160×120) signals array                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Compute Hjorth Parameters                              │
│ For each column signal:                                         │
│   - Activity = variance(signal)                                 │
│   - Mobility = sqrt(var(diff)/var(signal))                      │
│   - Complexity = sqrt(var(diff²)/var(diff)) / Mobility          │
│ Output: Activity(160,), Mobility(160,), Complexity(160,)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Feature Aggregation                                    │
│ - mean(activity), std(activity)                                 │
│ - mean(mobility), std(mobility)                                 │
│ - mean(complexity), std(complexity)                             │
│ Output: 6-D feature vector                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: Normalization                                          │
│ - StandardScaler: (x - μ) / σ                                   │
│ Output: Normalized 6-D feature vector                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ├──────────────────┬──────────────────┐
                         ▼                  ▼                  │
         ┌──────────────────────┐  ┌──────────────────────┐  │
         │ RandomForest          │  │ RandomForest          │  │
         │ Classifier            │  │ Regressor             │  │
         │ (300 trees)           │  │ (300 trees)          │  │
         └──────────────────────┘  └──────────────────────┘  │
                         │                  │                  │
                         ▼                  ▼                  │
         ┌──────────────────────┐  ┌──────────────────────┐  │
         │ "High Load"          │  │ 0.82                 │  │
         └──────────────────────┘  └──────────────────────┘  │
                         │                  │                  │
                         └──────────┬───────┘                  │
                                    ▼                          │
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT:                                                          │
│ {                                                                │
│   "load_category": "High Load",                                 │
│   "criticality_score": 0.82                                      │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 DATA TRANSFORMATIONS SUMMARY

| Stage | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| **Input** | (H, W, 3) | - | RGB thermal image |
| **Stage 1** | (H, W, 3) | (H, W) | Temperature matrix (°C) |
| **Stage 2** | (H, W) | (W, H) | Column signals |
| **Stage 3** | (W, H) | 3×(W,) | Hjorth parameters |
| **Stage 4** | 3×(W,) | (6,) | Aggregated features |
| **Stage 5** | (6,) | (6,) | Normalized features |
| **Model** | (6,) | - | Predictions |
| **Output** | - | - | JSON with category & score |

---

## 🎯 KEY INSIGHTS

### Why This Approach?

1. **Signal Processing First**: Hjorth parameters capture spatial patterns in temperature distribution
2. **Dimensionality Reduction**: From image (19,200 pixels) → 6 features
3. **Domain Knowledge**: Column-wise analysis matches busbar geometry
4. **Multi-Output**: Single model predicts both classification and regression
5. **Interpretable Features**: Hjorth parameters have physical meaning

### Feature Interpretation

- **High mean(activity)**: Large temperature variations across busbar
- **High std(activity)**: Inconsistent temperature distribution
- **High mean(mobility)**: Rapid temperature changes (potential hotspots)
- **High mean(complexity)**: Irregular, unpredictable patterns (concerning)

### Model Strengths

- **Fast Inference**: Only 6 features → quick prediction
- **Robust**: RandomForest handles non-linear relationships
- **No Deep Learning**: No GPU required, runs on CPU
- **Interpretable**: Can analyze feature importance

### Model Limitations

- **Small Feature Set**: Only 6 features may miss complex patterns
- **Fixed Preprocessing**: Assumes specific temperature mapping
- **No Spatial Context**: Aggregation loses spatial relationships
- **Limited to Training Data**: Performance depends on dataset quality

---

## 🔬 EXAMPLE WALKTHROUGH

### Input Image
```
RGB Image: 120×160×3
- Represents thermal image from IR camera
- Colors encode temperature (blue=cold, red=hot)
```

### Processing Steps

1. **Temperature Matrix**: 
   ```
   120×160 matrix
   Values: 25.3°C, 67.8°C, 92.1°C, ...
   ```

2. **Column Signals**:
   ```
   160 signals, each 120 values long
   Signal 0: [25.3, 26.1, 28.5, ..., 30.2]
   Signal 1: [27.1, 28.9, 31.2, ..., 32.5]
   ...
   ```

3. **Hjorth Parameters**:
   ```
   Activity: [12.3, 15.7, 18.2, ..., 11.9]  (160 values)
   Mobility: [0.45, 0.52, 0.48, ..., 0.43]  (160 values)
   Complexity: [1.2, 1.5, 1.3, ..., 1.1]   (160 values)
   ```

4. **Aggregated Features**:
   ```
   [mean_activity=14.5, std_activity=2.3,
    mean_mobility=0.48, std_mobility=0.05,
    mean_complexity=1.3, std_complexity=0.2]
   ```

5. **Normalized Features**:
   ```
   [0.52, -0.31, 1.15, -0.82, 0.18, 0.41]
   ```

6. **Model Predictions**:
   ```
   Classifier: "High Load" (class 2)
   Regressor: 0.82 (criticality score)
   ```

7. **Final Output**:
   ```json
   {
     "load_category": "High Load",
     "criticality_score": 0.82
   }
   ```

---

## 📝 SUMMARY

**Input**: Thermal image (RGB or grayscale) from IR camera

**Processing**: 
1. Convert to temperature matrix
2. Extract vertical column signals
3. Compute Hjorth parameters (Activity, Mobility, Complexity)
4. Aggregate to 6-D feature vector
5. Normalize features

**Model**: Two RandomForest models (classifier + regressor)

**Output**: 
- Load category: "Low Load", "Medium Load", or "High Load"
- Criticality score: 0.0 (safe) to 1.0 (critical)

The model transforms a thermal image into a compact 6-D feature representation using signal processing, then uses machine learning to predict both classification and regression outputs simultaneously.

