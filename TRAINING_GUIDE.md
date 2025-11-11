# Complete Training and Testing Guide

## Prerequisites
- Python 3.8+ installed
- Virtual environment support

---

## Step 1: Environment Setup

### 1.1 Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Prepare Dataset

### Option A: Generate Synthetic Dataset (Quick Start)
```bash
python generate_synthetic.py
```
This creates:
- `dataset/` folder with 24 synthetic thermal images
- `dataset/labels.csv` with labels and criticality scores

### Option B: Use Your Own Dataset

1. Create `dataset/` folder
2. Place your thermal images in `dataset/`
3. Create `dataset/labels.csv` with columns:
   ```csv
   filepath,label,criticality
   image1.png,Low Load,0.1
   image2.png,Medium Load,0.5
   image3.png,High Load,0.9
   ```
   - `filepath`: relative path from `dataset/` folder
   - `label`: "Low Load", "Medium Load", or "High Load"
   - `criticality`: float between 0.0 (no risk) and 1.0 (critical)

---

## Step 3: Train the Model

### 3.1 Run Training Script
```bash
python train.py
```

**What happens:**
1. Loads dataset from `dataset/labels.csv`
2. Extracts 6-D Hjorth features from each image
3. Splits data: 80% train, 20% test
4. Performs 5-fold cross-validation
5. Trains RandomForest classifier (Low/Medium/High)
6. Trains RandomForest regressor (criticality 0-1)
7. Evaluates on test set
8. Saves models to `artifacts/` folder

### 3.2 Expected Output
```
Loaded 24 rows
CV Accuracy (mean±std): 0.XXX ± 0.XXX
Test Accuracy: 0.XXX
Regression MAE: 0.XXXX
Regression MSE: 0.XXXX
Saved joblib models to artifacts
ONNX exported: artifacts/classifier.onnx artifacts/regressor.onnx
```

**Saved Artifacts:**
- `artifacts/classifier.joblib` - Classification model
- `artifacts/regressor.joblib` - Regression model
- `artifacts/label_encoder.joblib` - Label encoder
- `artifacts/classifier.onnx` - ONNX classifier
- `artifacts/regressor.onnx` - ONNX regressor

### 3.3 View Training Plots
The training script generates:
- **Confusion Matrix**: Classification performance
- **Learning Curve**: Training vs validation accuracy
- **Regression Residuals**: Prediction error distribution

---

## Step 4: Test the Model

### 4.1 Single Image Inference (Joblib)

**Test on a single image:**
```bash
python infer.py dataset/synthetic_000.png artifacts
```

**Expected output:**
```json
{'load_category': 'Low Load', 'criticality_score': 0.556}
```

### 4.2 Single Image Inference (ONNX)

**Test using ONNX models:**
```bash
python infer.py dataset/synthetic_000.png artifacts --onnx
```

### 4.3 Batch Testing

**Run comprehensive test suite:**
```bash
python test_model.py
```

This will:
- Test all images in the dataset
- Compare joblib vs ONNX predictions
- Generate accuracy metrics
- Create a test report

---

## Step 5: API Server (Optional)

### 5.1 Start the API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5.2 Test API Endpoint

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@dataset/synthetic_000.png"
```

**Using Python:**
```python
import requests

with open("dataset/synthetic_000.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
print(response.json())
```

**Expected Response:**
```json
{
  "load_category": "Low Load",
  "criticality_score": 0.556
}
```

---

## Step 6: Understanding the Model

### 6.1 Feature Extraction Pipeline

1. **Image → Temperature Matrix**: Converts thermal image to temperature values (°C)
2. **Column Signals**: Extracts vertical temperature profiles
3. **Hjorth Parameters**: Computes Activity, Mobility, Complexity per column
4. **Feature Aggregation**: Creates 6-D feature vector:
   - mean(activity), std(activity)
   - mean(mobility), std(mobility)
   - mean(complexity), std(complexity)

### 6.2 Model Architecture

- **Multi-Head Model**: Two separate RandomForest models
  - **Classifier**: 3 classes (Low/Medium/High Load)
  - **Regressor**: Continuous criticality score (0-1)
- **Normalization**: StandardScaler applied to features
- **Hyperparameters**: 300 trees, no max depth limit

---

## Troubleshooting

### Issue: "Cannot find path 'artifacts'"
**Solution**: Run `train.py` first to generate models

### Issue: "FileNotFoundError: Cannot read image"
**Solution**: Check image path and format (PNG, JPG supported)

### Issue: Low accuracy on test set
**Solutions**:
- Increase dataset size (more images per class)
- Check label quality in CSV
- Adjust hyperparameters in `busbar/model.py`

### Issue: ONNX inference fails
**Solution**: Ensure ONNX models were exported successfully during training

---

## Next Steps

1. **Improve Dataset**: Add more diverse thermal images
2. **Hyperparameter Tuning**: Adjust RandomForest parameters
3. **Feature Engineering**: Experiment with additional features
4. **Model Comparison**: Try XGBoost or neural networks
5. **Deployment**: Package for production use

---

## File Structure
```
BusbarPreprocessing/
├── busbar/              # Core package
│   ├── preprocessing.py  # Image → temperature conversion
│   ├── features.py      # Hjorth feature extraction
│   ├── dataset.py       # Dataset loading utilities
│   ├── model.py         # MultiHeadModel class
│   └── onnx_utils.py    # ONNX export/inference
├── dataset/             # Training images and labels.csv
├── artifacts/           # Trained models (created after training)
├── train.py            # Training script
├── infer.py            # Inference script
├── generate_synthetic.py # Synthetic data generator
├── test_model.py       # Comprehensive test suite
├── api.py              # FastAPI server
└── requirements.txt    # Dependencies
```

---

## Quick Reference Commands

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate data
python generate_synthetic.py

# Train
python train.py

# Test single image
python infer.py dataset/synthetic_000.png artifacts

# Test with ONNX
python infer.py dataset/synthetic_000.png artifacts --onnx

# Run test suite
python test_model.py

# Start API
uvicorn api:app --host 0.0.0.0 --port 8000
```

