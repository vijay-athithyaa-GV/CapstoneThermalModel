# Preprocessing Steps Explanation

This document explains **each step** in `busbar/preprocessing.py` and **why it's required** for the busbar heat detection model.

---

## 📋 Overview

The preprocessing pipeline converts a thermal image into features that the machine learning model can understand. It transforms raw pixel values into meaningful temperature signals and then extracts statistical features.

---

## 🔍 Step-by-Step Explanation

### **STEP 1: Image to Temperature Matrix Conversion**

**Function**: `image_to_temperature_matrix()` (lines 7-67)

#### What It Does:

```python
def image_to_temperature_matrix(
    image: np.ndarray,
    mode: str = "auto",
    min_temp_c: float = 20.0,
    max_temp_c: float = 120.0,
    calibration_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
```

This function converts a thermal image (RGB pseudo-color or grayscale) into a **temperature matrix** where each pixel value represents an actual temperature in Celsius.

#### Why It's Required:

1. **Thermal images are color-coded**: Thermal cameras display temperature as colors (blue=cold, red/yellow=hot), but the model needs actual temperature values
2. **Standardization**: Different thermal cameras use different color schemes, so we need a consistent temperature representation
3. **Physical meaning**: Temperature values (°C) have physical meaning that the model can learn from

#### How It Works:

**For RGB Pseudo-Color Images** (most common):
```python
# Lines 54-65
if inferred_mode == "rgb_pseudocolor":
    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0  # Normalize to [0, 1]
    
    # Convert RGB to grayscale using luminance weights
    # These weights match human eye sensitivity to colors
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    
    # Map grayscale [0, 1] to temperature range [20°C, 120°C]
    temp_c = gray * (max_temp_c - min_temp_c) + min_temp_c
    return temp_c.astype(np.float32)
```

**Why luminance weights?**
- Green contributes most to perceived brightness (0.7152)
- Red contributes moderately (0.2126)
- Blue contributes least (0.0722)
- This matches how human eyes perceive color brightness
- Ensures consistent temperature extraction regardless of color scheme

**Example:**
```
Input: RGB pixel [255, 100, 50] (reddish = hot)
    ↓ Normalize to [1.0, 0.39, 0.20]
    ↓ Convert to grayscale: 0.2126×1.0 + 0.7152×0.39 + 0.0722×0.20 = 0.50
    ↓ Map to temperature: 0.50 × (120-20) + 20 = 70°C
Output: 70.0°C
```

#### Why Temperature Range Matters:

- **min_temp_c = 20°C**: Typical ambient/room temperature
- **max_temp_c = 120°C**: Critical overheating threshold for electrical systems
- This range covers normal operation to critical conditions
- The model learns patterns within this realistic temperature range

---

### **STEP 2: Extract Column Signals**

**Function**: `extract_column_signals()` (lines 70-74)

#### What It Does:

```python
def extract_column_signals(temp_c: np.ndarray) -> np.ndarray:
    if temp_c.ndim != 2:
        raise ValueError("temp_c must be 2D (H, W)")
    signals = temp_c.T  # Transpose: columns become rows
    return signals.astype(np.float32)
```

This function extracts **vertical temperature profiles** by taking each column of the temperature matrix as a separate signal.

#### Why It's Required:

1. **Busbar structure**: Busbars are typically **vertical** electrical conductors
   - Heat flows vertically along the busbar
   - Vertical temperature profiles capture the heat distribution pattern

2. **Signal processing**: Hjorth parameters work on **1D signals** (time series)
   - We need to convert 2D temperature data into 1D signals
   - Each column becomes a "time series" representing temperature variation along height

3. **Spatial analysis**: Each column represents a different horizontal position
   - Column 0 = left side of busbar
   - Column 80 = middle of busbar
   - Column 159 = right side of busbar
   - Analyzing all columns captures heat distribution across the entire busbar

#### How It Works:

**Input**: Temperature matrix (H×W)
```
Temperature Matrix (120×160):
┌─────────────────────────┐
│ 30°C  32°C  35°C  ...   │  Row 0 (top)
│ 35°C  40°C  45°C  ...   │  Row 1
│ 40°C  50°C  60°C  ...   │  Row 2
│ ...   ...   ...   ...   │
│ 25°C  28°C  30°C  ...   │  Row 119 (bottom)
└─────────────────────────┘
```

**Output**: Signals array (W×H) - each row is one column signal
```
Signals Array (160×120):
┌─────────────────────────┐
│ 30°C  35°C  40°C  ...   │  Signal 0 (leftmost column)
│ 32°C  40°C  50°C  ...   │  Signal 1
│ 35°C  45°C  60°C  ...   │  Signal 2
│ ...   ...   ...   ...   │
│ ...   ...   ...   ...   │  Signal 159 (rightmost column)
└─────────────────────────┘
```

**Why transpose?**
- Original: `temp_c[i, j]` = temperature at row i, column j
- Transposed: `signals[j, i]` = temperature at column j, row i
- Now each row is a complete vertical temperature profile

#### Real-World Example:

Imagine a busbar with a hot spot in the middle:
```
Column 0 (left):    [30, 32, 35, 38, 40, 38, 35, 32, 30]  ← Normal
Column 80 (middle): [30, 35, 45, 70, 85, 70, 45, 35, 30]  ← Hot spot!
Column 159 (right): [30, 32, 35, 38, 40, 38, 35, 32, 30]  ← Normal
```

The middle column signal will have:
- **High activity** (large temperature variation)
- **High mobility** (rapid temperature changes)
- **High complexity** (complex temperature pattern)

This is exactly what the model needs to detect!

---

### **STEP 3: Compute Hjorth Parameters**

**Function**: `hjorth_parameters()` (lines 77-101) - for single signal
**Function**: `hjorth_parameters_per_signal()` (lines 104-118) - for all signals

#### What It Does:

Hjorth parameters are **signal processing features** that characterize the statistical properties of a signal. For each temperature signal (column), we compute three parameters:

1. **Activity** (variance)
2. **Mobility** (rate of change)
3. **Complexity** (signal irregularity)

#### Why It's Required:

1. **Dimensionality reduction**: 
   - Input: 160 signals × 120 values = 19,200 temperature values
   - Output: 3 parameters × 160 signals = 480 values
   - Still too many! (We'll aggregate further in next step)

2. **Pattern recognition**:
   - Raw temperature values vary with absolute temperature
   - Hjorth parameters capture **relative patterns** regardless of absolute values
   - Example: A 30°C→50°C jump has same "activity" as 80°C→100°C jump

3. **Robustness**:
   - Works with different temperature ranges
   - Works with different image sizes
   - Works with different camera calibrations

4. **Interpretability**:
   - Each parameter has physical meaning
   - Activity = temperature variation
   - Mobility = rate of temperature change
   - Complexity = pattern irregularity

#### How It Works:

##### 3.1 Activity (Variance)

```python
# Line 82
var0 = np.var(x)  # Variance of the signal
```

**Formula**: `Activity = Var(x) = E[(x - μ)²]`

**What it measures**: How much the temperature varies in this column

**Why it matters**:
- **High activity**: Large temperature variations → possible hot spot
- **Low activity**: Uniform temperature → normal operation

**Example**:
```python
Signal 1: [30, 30, 30, 30, 30]  → Activity = 0.0 (uniform)
Signal 2: [30, 50, 30, 50, 30]  → Activity = 100.0 (high variation)
```

##### 3.2 Mobility (Rate of Change)

```python
# Lines 86-88
dx = np.diff(x)  # First derivative (differences between consecutive values)
var1 = np.var(dx)  # Variance of first derivative
mobility = math.sqrt(var1 / var0)  # Normalized by activity
```

**Formula**: `Mobility = √(Var(dx) / Var(x))`

**What it measures**: How rapidly temperature changes along the column

**Why it matters**:
- **High mobility**: Rapid temperature changes → sharp gradients → potential problem
- **Low mobility**: Gradual temperature changes → normal heat distribution

**Example**:
```python
Signal 1: [30, 32, 34, 36, 38]  → Mobility = low (gradual change)
Signal 2: [30, 30, 70, 70, 30]  → Mobility = high (rapid change)
```

**Physical meaning**: 
- High mobility = steep temperature gradient
- Indicates localized heating (bad for electrical systems)

##### 3.3 Complexity (Signal Irregularity)

```python
# Lines 93-99
ddx = np.diff(dx)  # Second derivative (rate of change of rate of change)
var2 = np.var(ddx)  # Variance of second derivative
complexity = math.sqrt(var2 / var1) / mobility  # Normalized complexity
```

**Formula**: `Complexity = √(Var(d²x) / Var(dx)) / Mobility`

**What it measures**: How irregular/complex the temperature pattern is

**Why it matters**:
- **High complexity**: Irregular, complex patterns → multiple heat sources or anomalies
- **Low complexity**: Simple, smooth patterns → normal operation

**Example**:
```python
Signal 1: [30, 35, 40, 45, 50]  → Complexity = low (smooth, linear)
Signal 2: [30, 50, 35, 60, 40]  → Complexity = high (irregular, complex)
```

**Physical meaning**:
- High complexity = complex heat distribution
- Could indicate multiple problems or non-uniform loading

#### Why These Three Parameters?

Hjorth parameters were originally developed for **EEG signal analysis** (brain waves) in 1970. They work well for any time series or spatial signal because they capture:

1. **Amplitude characteristics** (Activity)
2. **Frequency characteristics** (Mobility)
3. **Pattern characteristics** (Complexity)

For thermal analysis:
- **Activity** → Overall temperature variation
- **Mobility** → Rate of temperature change (gradients)
- **Complexity** → Pattern irregularity (anomalies)

#### Complete Example:

```python
# Temperature signal from a column with a hot spot
signal = [30, 32, 35, 40, 70, 85, 70, 40, 35, 32, 30]

# Step 1: Activity (variance)
activity = np.var(signal) = 400.0  # High! Large variation

# Step 2: Mobility (rate of change)
dx = np.diff(signal) = [2, 3, 5, 30, 15, -15, -30, -5, -3, -2]
mobility = sqrt(var(dx) / var(signal)) = 0.5  # High! Rapid changes

# Step 3: Complexity (irregularity)
ddx = np.diff(dx) = [1, 2, 25, -15, -30, 0, 15, -25, -2, -1]
complexity = sqrt(var(ddx) / var(dx)) / mobility = 1.2  # High! Complex pattern
```

This signal clearly shows a hot spot (high activity, high mobility, high complexity)!

---

## 🔄 Why This Pipeline?

### Problem: Raw Images Are Too Complex

- **Input**: 120×160×3 RGB image = 57,600 values
- **Problem**: Too many dimensions, too much noise, too much variation
- **Solution**: Extract meaningful features

### Solution: Multi-Stage Feature Extraction

```
Stage 1: Image → Temperature
    Why: Convert colors to physical values (temperature)
    Result: 120×160 = 19,200 temperature values

Stage 2: Temperature → Signals
    Why: Extract vertical profiles (busbar structure)
    Result: 160 signals × 120 values = 19,200 values (same, but organized)

Stage 3: Signals → Hjorth Parameters
    Why: Extract statistical features (pattern characteristics)
    Result: 160 signals × 3 parameters = 480 values

Stage 4: (In features.py) Aggregate → 6 Features
    Why: Summarize all signals into single feature vector
    Result: 6 values (mean/std of each parameter)
```

### Why Each Step is Necessary

1. **Temperature Conversion**: 
   - Without it: Model sees colors, not temperatures
   - With it: Model sees physical values it can learn from

2. **Column Extraction**:
   - Without it: Model sees 2D image (hard to analyze)
   - With it: Model sees 1D signals (easy to analyze with signal processing)

3. **Hjorth Parameters**:
   - Without it: Model sees raw temperature values (too many, too noisy)
   - With it: Model sees statistical features (few, meaningful)

4. **Aggregation** (next step):
   - Without it: Model sees 480 values (still too many)
   - With it: Model sees 6 values (perfect for machine learning)

---

## 🎯 Real-World Example

Let's trace through a complete example:

### Input: Thermal Image of Busbar with Hot Spot

```
RGB Image (120×160×3):
- Left side: Blue colors (cold, ~30°C)
- Middle: Red colors (hot, ~85°C)
- Right side: Blue colors (cold, ~30°C)
```

### Step 1: Image → Temperature Matrix

```
Temperature Matrix (120×160):
- Left columns: ~30°C
- Middle columns: ~85°C (hot spot!)
- Right columns: ~30°C
```

### Step 2: Extract Column Signals

```
Signal 0 (left):   [30, 30, 30, ..., 30]  ← Uniform, low variation
Signal 80 (middle): [30, 35, 50, 70, 85, 70, 50, 35, 30, ...]  ← High variation!
Signal 159 (right): [30, 30, 30, ..., 30]  ← Uniform, low variation
```

### Step 3: Compute Hjorth Parameters

```
Signal 0:
  Activity = 0.0 (no variation)
  Mobility = 0.0 (no change)
  Complexity = 0.0 (simple)

Signal 80:
  Activity = 400.0 (high variation - hot spot!)
  Mobility = 0.8 (rapid changes)
  Complexity = 1.2 (complex pattern)

Signal 159:
  Activity = 0.0 (no variation)
  Mobility = 0.0 (no change)
  Complexity = 0.0 (simple)
```

### Step 4: Aggregate (in features.py)

```
Mean Activity = (0.0 + 400.0 + ... + 0.0) / 160 = 25.0
Std Activity = 15.0
Mean Mobility = 0.05
Std Mobility = 0.10
Mean Complexity = 0.08
Std Complexity = 0.15
```

### Step 5: Model Prediction

```
6-D Feature Vector: [25.0, 15.0, 0.05, 0.10, 0.08, 0.15]
    ↓
RandomForest Regressor
    ↓
Criticality Score: 0.82 (high!)
    ↓
Load Category: "High Load" (critical!)
```

---

## ✅ Summary: Why Each Step is Required

| Step | Input | Output | Why Required |
|------|-------|--------|--------------|
| **1. Image→Temp** | RGB colors | Temperature (°C) | Convert visual representation to physical values |
| **2. Extract Signals** | 2D temperature | 1D signals | Extract vertical profiles matching busbar structure |
| **3. Hjorth Params** | Raw signals | Statistical features | Reduce dimensionality, extract patterns, add robustness |
| **4. Aggregate** | Many parameters | 6 features | Final dimensionality reduction for ML model |

**Without these steps**: Model would need to learn from 57,600 raw pixel values (impossible!)

**With these steps**: Model learns from 6 meaningful features (perfect!)

---

## 🔬 Mathematical Foundation

### Why Hjorth Parameters Work

Hjorth parameters are based on **statistical moments** of the signal and its derivatives:

1. **Activity** = Second moment (variance) of signal
2. **Mobility** = Ratio of first moment of derivative to signal
3. **Complexity** = Ratio of second moment of second derivative to first derivative

These capture:
- **Scale** (how big are variations?)
- **Frequency** (how fast do things change?)
- **Shape** (how complex is the pattern?)

For thermal analysis:
- **Scale** → Temperature variation magnitude
- **Frequency** → Temperature gradient steepness
- **Shape** → Heat distribution pattern complexity

---

## 🎓 Key Takeaways

1. **Temperature conversion** is essential because models need physical values, not colors
2. **Column extraction** matches the physical structure of busbars (vertical conductors)
3. **Hjorth parameters** provide robust, interpretable features that work across different conditions
4. **Each step reduces complexity** while preserving important information
5. **The pipeline is domain-specific** - designed for thermal analysis of vertical electrical components

This preprocessing pipeline transforms raw thermal images into features that a machine learning model can effectively learn from!

