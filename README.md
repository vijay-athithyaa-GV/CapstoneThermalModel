# Busbar Heat Detection System 🔥

A complete end-to-end machine learning system for classifying thermal load (Low/Medium/High) and predicting criticality scores from thermal infrared images. This system uses signal processing (Hjorth parameters) combined with Random Forest regression to provide accurate thermal load assessment for busbar systems.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Training](#training)
- [Evaluation](#evaluation)
- [API](#api)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system processes thermal infrared images from IR cameras (FLIR, thermal imaging cameras) to:

1. **Classify Thermal Load**: Categorize busbar conditions as **Low Load**, **Medium Load**, or **High Load**
2. **Predict Criticality Score**: Provide a continuous risk score from **0.0** (no risk) to **1.0** (critical)

The model uses a **criticality-based approach** where classification is derived from the predicted criticality score using configurable thresholds:
- **Low Load**: criticality < 0.33
- **Medium Load**: 0.33 ≤ criticality < 0.67
- **High Load**: criticality ≥ 0.67

### Key Highlights

- ✅ **Single Model Architecture**: Uses one RandomForest regressor (simpler, faster, consistent)
- ✅ **Signal Processing**: Hjorth parameters for robust feature extraction
- ✅ **Thermal Image Validation**: Automatically validates input images before processing
- ✅ **Production Ready**: ONNX export, REST API, comprehensive evaluation
- ✅ **High Accuracy**: 90.9% test accuracy on real thermal images

---

## ✨ Features

### Core Capabilities

- **Thermal Image Processing**: Converts RGB pseudo-color thermal images to temperature matrices
- **Feature Extraction**: Extracts 6-D Hjorth parameter features (activity, mobility, complexity)
- **Multi-Output Prediction**: Simultaneous classification and regression
- **Model Persistence**: Save/load models in joblib and ONNX formats
- **REST API**: FastAPI server for production deployment
- **Comprehensive Evaluation**: Performance graphs, metrics, and analysis

### Advanced Features

- **Thermal Image Validator**: Validates input images to reject non-thermal images
- **Label-Based Training**: Assigns criticality based on folder labels (recommended)
- **Synthetic Data Generation**: Generate realistic thermal images for training
- **Performance Visualization**: Grid-based evaluation graphs with dotted styling
- **Cross-Validation**: Robust model evaluation with 5-fold CV

---

## 🏗️ Model Architecture

### Processing Pipeline

```
Thermal Image (RGB)
    ↓
[Stage 1] Image → Temperature Matrix (°C)
    ↓
[Stage 2] Extract Column-Wise Temperature Signals
    ↓
[Stage 3] Compute Hjorth Parameters (Activity, Mobility, Complexity)
    ↓
[Stage 4] Aggregate Features (mean, std) → 6-D Feature Vector
    ↓
[Stage 5] RandomForest Regressor → Criticality Score (0-1)
    ↓
[Stage 6] Derive Classification from Criticality
    ↓
Output: {load_category, criticality_score}
```

### Model Details

- **Model Type**: RandomForest Regressor (300 trees)
- **Input Features**: 6-D Hjorth parameter vector
  - `mean(activity)`, `std(activity)`
  - `mean(mobility)`, `std(mobility)`
  - `mean(complexity)`, `std(complexity)`
- **Output**: Criticality score (0.0 - 1.0)
- **Classification**: Derived from criticality using thresholds
- **Normalization**: StandardScaler applied to features

### Hjorth Parameters

The model uses **Hjorth parameters** (signal processing features) to characterize temperature distributions:

1. **Activity**: Variance of the signal (temperature variation)
2. **Mobility**: Standard deviation of the first derivative (rate of change)
3. **Complexity**: Ratio of mobility of first derivative to mobility of signal (signal complexity)

These parameters are computed for each column of the temperature matrix and aggregated (mean, std) to create a 6-D feature vector.

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd BusbarPreprocessing
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
# or
.\.venv\Scripts\activate.bat  # Command Prompt

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import busbar; print('Installation successful!')"
```

---

## 🚀 Quick Start

### 1. Prepare Dataset

Organize your thermal images into folders:

```
Lowload/     # Low Load images
HighLoad/    # High Load images
```

### 2. Build Labels CSV

```bash
python build_labels_from_folders.py \
    --low_dir "Lowload" \
    --high_dir "HighLoad" \
    --out_csv "dataset/labels_user_fixed.csv"
```

### 3. Train Model

```bash
python train_criticality_based.py \
    --csv "dataset/labels_user_fixed.csv" \
    --roots "Lowload" "HighLoad" \
    --artifacts_dir "artifacts_criticality"
```

### 4. Test Model

```bash
python test_criticality_based.py "Lowload/IMG-20251105-WA0117.jpg"
```

### 5. Evaluate Performance

```bash
python evaluate_model_performance.py \
    --csv "dataset/labels_user_fixed.csv" \
    --roots "Lowload" "HighLoad" \
    --model_dir "artifacts_criticality" \
    --output_dir "performance_evaluation"
```

---

## 📖 Usage Guide

### Training with Your Own Data

#### Option 1: Folder-Based Training (Recommended)

```bash
# 1. Organize images into folders
Lowload/    # Low Load images
HighLoad/   # High Load images
MediumLoad/ # Medium Load images (optional)

# 2. Build labels CSV
python build_labels_from_folders.py \
    --low_dir "Lowload" \
    --high_dir "HighLoad" \
    --medium_dir "MediumLoad" \
    --out_csv "dataset/labels.csv"

# 3. Train model
python train_criticality_based.py \
    --csv "dataset/labels.csv" \
    --roots "Lowload" "HighLoad" "MediumLoad"
```

#### Option 2: CSV-Based Training

Create a CSV file with columns: `filepath`, `label`, `criticality`

```csv
filepath,label,criticality
image1.jpg,Low Load,0.15
image2.jpg,High Load,0.85
image3.jpg,Medium Load,0.50
```

Then train:

```bash
python train_criticality_based.py \
    --csv "dataset/labels.csv" \
    --roots "Lowload" "HighLoad"
```

### Inference on Single Image

```bash
# Basic usage
python test_criticality_based.py "path/to/image.jpg"

# With validation (rejects non-thermal images)
python test_criticality_based.py "path/to/image.jpg"

# Warn-only mode (processes all images)
python test_criticality_based.py "path/to/image.jpg" --warn-only
```

### Batch Inference

```bash
# Test on all images in dataset
python test_model.py
```

### Generate Synthetic Data

```bash
# Generate FLIR-like thermal images
python generate_flir_like.py \
    --output_dir "dataset" \
    --n_images 200 \
    --csv_path "dataset/labels_synthetic.csv"
```

---

## 🎓 Training

### Training Process

1. **Load Dataset**: Reads images and labels from CSV
2. **Extract Features**: Computes 6-D Hjorth features for each image
3. **Train/Test Split**: 80/20 split with stratification
4. **Train Model**: Fits RandomForest regressor on training data
5. **Evaluate**: Computes accuracy, MAE, RMSE, R²
6. **Save Model**: Saves model to `artifacts_criticality/`

### Training Scripts

- **`train_criticality_based.py`**: Main training script (recommended)
- **`train_from_csv.py`**: Flexible CSV-based training
- **`train_merged.py`**: Training on merged datasets

### Training Options

```bash
python train_criticality_based.py \
    --csv "dataset/labels.csv" \
    --roots "Lowload" "HighLoad" \
    --low_threshold 0.33 \
    --medium_threshold 0.67 \
    --artifacts_dir "artifacts_criticality"
```

### Training Output

- **Model Files**: `artifacts_criticality/regressor.joblib`, `model_config.joblib`
- **Training Plot**: `training_criticality_based_results.png`
- **Metrics**: Console output with accuracy, MAE, RMSE, R²

---

## 📊 Evaluation

### Performance Evaluation

Generate comprehensive evaluation graphs:

```bash
python evaluate_model_performance.py \
    --csv "dataset/labels_user_fixed.csv" \
    --roots "Lowload" "HighLoad" \
    --model_dir "artifacts_criticality" \
    --output_dir "performance_evaluation"
```

### Generated Graphs

1. **Classification Performance** (2x2 grid)
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - Classification Distribution
   - Metrics Summary

2. **Regression Performance** (2x2 grid)
   - True vs Predicted Scatter
   - Residual Plot
   - Error Distribution
   - Metrics Summary

3. **Criticality Distribution** (2x2 grid)
   - True Distribution
   - Predicted Distribution
   - Overlay Comparison
   - Box Plot Comparison

4. **Learning & Features** (2x2 grid)
   - Learning Curve
   - Feature Importance
   - Cross-Validation Scores
   - Summary Statistics

### Performance Metrics

- **Accuracy**: Classification accuracy (0-1)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **R²**: Coefficient of Determination (higher is better, max=1.0)
- **CV MAE**: Cross-Validation MAE (robust estimate)

---

## 🌐 API

### Start API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST `/predict`

Predict thermal load and criticality from image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
    "load_category": "High Load",
    "criticality_score": 0.85
}
```

### API Usage (Python)

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(result)
```

---

## 📁 Project Structure

```
BusbarPreprocessing/
├── busbar/                          # Core package
│   ├── __init__.py                  # Package initialization
│   ├── preprocessing.py             # Image → temperature conversion
│   ├── features.py                  # Hjorth feature extraction
│   ├── dataset.py                   # Dataset loading utilities
│   ├── model_criticality_based.py   # CriticalityBasedModel (current)
│   ├── model.py                     # MultiHeadModel (original)
│   ├── onnx_utils.py                # ONNX export/inference
│   └── thermal_validator.py         # Thermal image validation
│
├── dataset/                         # Training data
│   ├── labels_user_fixed.csv        # Labels CSV
│   └── *.png, *.jpg                 # Thermal images
│
├── artifacts_criticality/           # Trained models (current)
│   ├── regressor.joblib             # RandomForest regressor
│   └── model_config.joblib          # Model configuration
│
├── performance_evaluation/          # Evaluation graphs
│   ├── 01_classification_performance.png
│   ├── 02_regression_performance.png
│   ├── 03_criticality_distribution.png
│   └── 04_learning_features.png
│
├── train_criticality_based.py       # Main training script
├── test_criticality_based.py        # Single image inference
├── evaluate_model_performance.py    # Performance evaluation
├── build_labels_from_folders.py     # Build labels from folders
├── api.py                           # FastAPI server
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Key Files

- **`busbar/`**: Core package with preprocessing, features, models
- **`train_criticality_based.py`**: Main training script
- **`test_criticality_based.py`**: Inference script
- **`evaluate_model_performance.py`**: Performance evaluation
- **`build_labels_from_folders.py`**: Dataset preparation
- **`api.py`**: REST API server

---

## 🎯 Performance

### Model Performance (Real Data)

- **Test Accuracy**: 90.9%
- **MAE**: 0.0176
- **RMSE**: 0.0243
- **R²**: 0.9952
- **CV MAE**: 0.0679 ± 0.0301

### Dataset

- **Training Samples**: 44 images
- **Test Samples**: 11 images
- **Features**: 6-D Hjorth parameters
- **Classes**: Low Load (23), High Load (32)

### Feature Importance

- **Most Important**: `mean(activity)` (87.5%)
- **Other Features**: `std(activity)`, `mean(complexity)`, `mean(mobility)`, etc.

---

## 📚 Documentation

### Main Documentation

- **[MODEL_EXPLANATION.md](MODEL_EXPLANATION.md)**: Complete end-to-end model explanation
- **[MODEL_SUMMARY.md](MODEL_SUMMARY.md)**: Quick reference guide
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Detailed training instructions
- **[PERFORMANCE_EVALUATION_GUIDE.md](PERFORMANCE_EVALUATION_GUIDE.md)**: Evaluation guide
- **[MODEL_ARCHITECTURE_COMPARISON.md](MODEL_ARCHITECTURE_COMPARISON.md)**: Model comparison

### Specialized Guides

- **[CRITICALITY_BASED_MODEL.md](CRITICALITY_BASED_MODEL.md)**: Criticality-based model details
- **[TRAINING_FIX_SUMMARY.md](TRAINING_FIX_SUMMARY.md)**: Training fix documentation
- **[VALIDATOR_IMPROVEMENTS.md](VALIDATOR_IMPROVEMENTS.md)**: Thermal validator improvements
- **[SUPPORTED_IMAGE_FORMATS.md](SUPPORTED_IMAGE_FORMATS.md)**: Image format support

### API Documentation

- **FastAPI Docs**: `http://localhost:8000/docs` (when API is running)
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `Model directory not found: artifacts_criticality`

**Solution**: Train the model first:
```bash
python train_criticality_based.py --csv "dataset/labels.csv" --roots "Lowload" "HighLoad"
```

#### 2. Image Not Found

**Error**: `Cannot read image: path/to/image.jpg`

**Solution**: Check image path and format (PNG, JPG, JPEG supported)

#### 3. Low Accuracy

**Possible Causes**:
- Insufficient training data
- Incorrect criticality assignments
- Poor image quality

**Solution**:
- Increase dataset size
- Verify labels and criticality assignments
- Check image quality and preprocessing

#### 4. Non-Thermal Image Rejection

**Error**: `REJECTED: Image does not appear to be a thermal image`

**Solution**: 
- Use `--warn-only` flag to process anyway
- Verify image is actually a thermal image
- Check thermal validator settings

#### 5. Memory Issues

**Error**: `MemoryError` or `OSError: No space left on device`

**Solution**:
- Reduce dataset size
- Process images in batches
- Free up disk space

### Getting Help

1. Check documentation in `docs/` folder
2. Review error messages carefully
3. Verify dataset format and labels
4. Check model training logs

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes**: Follow code style and add tests
4. **Commit your changes**: `git commit -m "Add your feature"`
5. **Push to the branch**: `git push origin feature/your-feature`
6. **Create a Pull Request**: Describe your changes

### Code Style

- Follow PEP 8 Python style guide
- Add docstrings to all functions and classes
- Include type hints where possible
- Write tests for new features

### Testing

```bash
# Run tests
python test_model.py
python test_criticality_based.py "test_image.jpg"
```

---

## 📄 License

This project is provided as-is for research and development purposes.

**License**: MIT License (see LICENSE file for details)

---

## 🙏 Acknowledgments

- **Hjorth Parameters**: Signal processing features for time series analysis
- **Random Forest**: Scikit-learn implementation
- **FastAPI**: Modern web framework for APIs
- **OpenCV**: Image processing library
- **Matplotlib**: Visualization library

---

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

## 📈 Version History

### v1.0.0 (Current)

- ✅ Criticality-based model (single regressor)
- ✅ Thermal image validation
- ✅ Comprehensive evaluation graphs
- ✅ REST API support
- ✅ ONNX export
- ✅ Label-based training
- ✅ Performance: 90.9% accuracy

### Future Enhancements

- [ ] Support for more thermal image formats
- [ ] Real-time inference optimization
- [ ] Mobile app integration
- [ ] Cloud deployment guides
- [ ] Additional evaluation metrics
- [ ] Model ensemble support

---

## 🔗 Related Projects

- **Thermal Image Processing**: OpenCV, scikit-image
- **Machine Learning**: Scikit-learn, XGBoost
- **API Framework**: FastAPI, Uvicorn
- **Model Export**: ONNX, ONNX Runtime

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{busbar_heat_detection,
  title = {Busbar Heat Detection System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/busbar-heat-detection}
}
```

---

**⭐ Star this repository if you find it useful!**

---

*Last updated: November 2025*
