# Quick Start Reference Card

## Complete Workflow (Copy & Paste)

```powershell
# Step 1: Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Step 2: Generate synthetic dataset
python generate_synthetic.py

# Step 3: Train the model
python train.py

# Step 4: Test the model (comprehensive)
python test_model.py

# Step 5: Test single image
python infer.py dataset/synthetic_000.png artifacts

# Step 6: Start API server
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Individual Commands

### Setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data Generation
```bash
python generate_synthetic.py
```

### Training
```bash
python train.py
```

### Testing
```bash
# Full test suite
python test_model.py

# Single image (joblib)
python infer.py dataset/synthetic_000.png artifacts

# Single image (ONNX)
python infer.py dataset/synthetic_000.png artifacts --onnx
```

### API
```bash
# Start server
uvicorn api:app --host 0.0.0.0 --port 8000

# Test endpoint (PowerShell)
$file = Get-Item "dataset\synthetic_000.png"
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -InFile $file.FullName -ContentType "multipart/form-data"
```

## Expected Outputs

### Training Output
```
Loaded 24 rows
CV Accuracy (mean±std): 0.XXX ± 0.XXX
Test Accuracy: 0.XXX
Regression MAE: 0.XXXX
Regression MSE: 0.XXXX
Saved joblib models to artifacts
ONNX exported: artifacts/classifier.onnx artifacts/regressor.onnx
```

### Inference Output
```json
{'load_category': 'Low Load', 'criticality_score': 0.556}
```

### API Response
```json
{
  "load_category": "Low Load",
  "criticality_score": 0.556
}
```

## File Locations

- **Dataset**: `dataset/` folder
- **Labels**: `dataset/labels.csv`
- **Models**: `artifacts/` folder (created after training)
- **Test Results**: `test_results.json` (created after testing)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "artifacts not found" | Run `python train.py` first |
| "Cannot read image" | Check image path and format |
| Low accuracy | Increase dataset size or check labels |
| ONNX errors | Re-run training to regenerate ONNX files |

## Next Steps

1. Replace synthetic data with real thermal images
2. Adjust hyperparameters in `busbar/model.py`
3. Experiment with different models (XGBoost, etc.)
4. Deploy API to production server

