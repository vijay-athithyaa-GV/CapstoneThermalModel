# Busbar Heat Detection Model - Overview

## 🎯 What This Model Does

This is a **machine learning system** that analyzes thermal infrared images of electrical busbars to:

1. **Classify Thermal Load**: Categorizes busbar conditions as **Low Load**, **Medium Load**, or **High Load**
2. **Predict Criticality Score**: Provides a continuous risk score from **0.0** (no risk) to **1.0** (critical)

The model is designed for **preventive maintenance** and **safety monitoring** of electrical systems by detecting overheating conditions before they become critical.

---

## 🔬 How It Works

### Input
- **Thermal infrared images** from IR cameras (FLIR, thermal imaging cameras)
- Images can be RGB pseudo-color thermal images or grayscale
- Supported formats: PNG, JPG, JPEG

### Processing Pipeline

```
Thermal Image (RGB)
    ↓
[Stage 1] Convert to Temperature Matrix (°C)
    - Extracts temperature values from color-coded thermal image
    - Maps colors to temperature range (20°C to 120°C)
    ↓
[Stage 2] Extract Column-Wise Temperature Signals
    - Each column becomes a vertical temperature profile
    - Captures heat distribution patterns along busbar height
    ↓
[Stage 3] Compute Hjorth Parameters (Signal Processing)
    - Activity: Temperature variation (variance)
    - Mobility: Rate of temperature change
    - Complexity: Signal complexity measure
    ↓
[Stage 4] Aggregate Features
    - Mean and standard deviation of each parameter
    - Creates 6-dimensional feature vector
    ↓
[Stage 5] RandomForest Regressor Prediction
    - Predicts criticality score (0.0 - 1.0)
    ↓
[Stage 6] Derive Classification from Criticality
    - Low Load: criticality < 0.33
    - Medium Load: 0.33 ≤ criticality < 0.67
    - High Load: criticality ≥ 0.67
    ↓
Output: {load_category, criticality_score}
```

---

## 📊 Model Architecture

### Model Type
- **Algorithm**: RandomForest Regressor (300 trees)
- **Input Features**: 6-D Hjorth parameter vector
  - `mean(activity)`, `std(activity)`
  - `mean(mobility)`, `std(mobility)`
  - `mean(complexity)`, `std(complexity)`
- **Output**: Criticality score (0.0 - 1.0)
- **Classification**: Derived from criticality using configurable thresholds

### Key Features
- **Single Model Architecture**: Uses one regressor (simpler, faster, consistent)
- **Signal Processing**: Hjorth parameters for robust feature extraction
- **Thermal Image Validation**: Automatically validates input images before processing
- **Production Ready**: ONNX export, REST API support

---

## 🎯 Model Performance

### Accuracy Metrics (Real Data)
- **Test Accuracy**: 90.9%
- **Mean Absolute Error (MAE)**: 0.0176
- **Root Mean Squared Error (RMSE)**: 0.0243
- **R² Score**: 0.9952
- **Cross-Validation MAE**: 0.0679 ± 0.0301

### Dataset
- **Training Samples**: 44 images
- **Test Samples**: 11 images
- **Classes**: Low Load (23), High Load (32)
- **Feature Importance**: `mean(activity)` is most important (87.5%)

---

## 💡 What the Model Predicts

### 1. Criticality Score (0.0 - 1.0)
- **0.0 - 0.33**: Low risk - Normal operation
- **0.33 - 0.67**: Medium risk - Monitor closely
- **0.67 - 1.0**: High risk - Immediate attention required

### 2. Load Category
- **Low Load**: Normal operating conditions, no immediate concern
- **Medium Load**: Elevated temperature, should be monitored
- **High Load**: Critical overheating, requires immediate action

### Example Output
```json
{
    "load_category": "High Load",
    "criticality_score": 0.85
}
```

**Interpretation**: The busbar shows high thermal load with a criticality score of 0.85, indicating a critical condition requiring immediate attention.

---

## 🔧 Technical Details

### Hjorth Parameters Explained

The model uses **Hjorth parameters** (signal processing features) to characterize temperature distributions:

1. **Activity**: Variance of the signal (temperature variation)
   - High activity = large temperature variations
   - Low activity = uniform temperature

2. **Mobility**: Standard deviation of the first derivative (rate of change)
   - High mobility = rapid temperature changes
   - Low mobility = gradual temperature changes

3. **Complexity**: Ratio of mobility of first derivative to mobility of signal
   - High complexity = complex temperature patterns
   - Low complexity = simple temperature patterns

These parameters are computed for each column of the temperature matrix and aggregated (mean, std) to create a 6-D feature vector.

### Why This Approach?

1. **Robust to Image Variations**: Hjorth parameters capture signal characteristics regardless of absolute temperature values
2. **Spatial Pattern Recognition**: Column-wise analysis captures vertical heat distribution patterns
3. **Dimensionality Reduction**: 6 features from potentially thousands of pixels
4. **Interpretable Features**: Each feature has physical meaning related to temperature distribution

---

## 🚀 Use Cases

### 1. Preventive Maintenance
- Monitor busbar conditions over time
- Detect early signs of overheating
- Schedule maintenance before failures occur

### 2. Safety Monitoring
- Real-time monitoring of electrical systems
- Alert operators to critical conditions
- Prevent electrical fires and equipment damage

### 3. Quality Control
- Verify proper installation and operation
- Ensure thermal performance meets specifications
- Document thermal conditions for compliance

### 4. Research & Development
- Analyze thermal behavior of new designs
- Compare different busbar configurations
- Optimize thermal management strategies

---

## 📁 Key Files

### Core Model Files
- `busbar/model_criticality_based.py`: Main model class
- `busbar/features.py`: Feature extraction (Hjorth parameters)
- `busbar/preprocessing.py`: Image to temperature conversion
- `busbar/thermal_validator.py`: Input validation

### Training & Testing
- `train_criticality_based.py`: Model training script
- `test_criticality_based.py`: Single image inference
- `evaluate_model_performance.py`: Performance evaluation
- `build_labels_from_folders.py`: Dataset preparation

### API & Deployment
- `api.py`: FastAPI REST API server
- `artifacts_criticality/`: Trained model files

---

## 🔍 Model Validation

The model includes **thermal image validation** to ensure input images are actually thermal images:

- Checks color distribution (thermal images have specific color patterns)
- Validates temperature range (rejects non-thermal images)
- Warns about potentially incorrect inputs

This prevents misclassification of regular photos as thermal images.

---

## 📈 Model Training

### Training Process
1. **Load Dataset**: Reads images and labels from CSV
2. **Extract Features**: Computes 6-D Hjorth features for each image
3. **Train/Test Split**: 80/20 split with stratification
4. **Train Model**: Fits RandomForest regressor on training data
5. **Evaluate**: Computes accuracy, MAE, RMSE, R²
6. **Save Model**: Saves model to `artifacts_criticality/`

### Training Data Format
- Images organized in folders: `Lowload/`, `HighLoad/`
- CSV file with columns: `filepath`, `label`, `criticality`
- Criticality scores assigned based on load category:
  - Low Load: 0.0 - 0.33
  - Medium Load: 0.33 - 0.67
  - High Load: 0.67 - 1.0

---

## 🎓 Summary

This model is a **production-ready machine learning system** for thermal analysis of electrical busbars. It combines:

- **Signal Processing** (Hjorth parameters) for robust feature extraction
- **Machine Learning** (RandomForest) for accurate prediction
- **Thermal Image Processing** for temperature extraction
- **Validation** for input quality assurance

The model achieves **90.9% accuracy** on real thermal images and provides both classification and continuous criticality scores for comprehensive thermal load assessment.

---

*For detailed technical documentation, see `MODEL_EXPLANATION.md`*
*For usage instructions, see `README.md`*

