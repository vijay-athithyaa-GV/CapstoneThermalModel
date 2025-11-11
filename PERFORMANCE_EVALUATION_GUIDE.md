# Performance Evaluation Guide

## Overview

The `evaluate_model_performance.py` script generates comprehensive performance evaluation graphs in a **grid layout with dotted styling** for the criticality-based model.

## Generated Graphs

### 1. Classification Performance Analysis (2x2 Grid)
**File:** `performance_evaluation/01_classification_performance.png`

Contains:
- **Confusion Matrix**: Shows classification accuracy and error patterns
- **Precision, Recall, F1-Score**: Per-class metrics with bar charts
- **Classification Distribution**: True vs Predicted class counts
- **Metrics Summary**: Text summary of all classification metrics

### 2. Regression Performance Analysis (2x2 Grid)
**File:** `performance_evaluation/02_regression_performance.png`

Contains:
- **True vs Predicted Scatter**: Shows prediction accuracy with R² score
- **Residual Plot**: Shows prediction errors (residuals)
- **Error Distribution**: Histogram of prediction errors
- **Metrics Summary**: MAE, MSE, RMSE, R² scores

### 3. Criticality Distribution Analysis (2x2 Grid)
**File:** `performance_evaluation/03_criticality_distribution.png`

Contains:
- **True Criticality Distribution**: Histogram of true criticality scores
- **Predicted Criticality Distribution**: Histogram of predicted scores
- **Overlay Comparison**: True vs Predicted overlaid
- **Box Plot Comparison**: Statistical comparison of distributions

### 4. Model Learning & Feature Analysis (2x2 Grid)
**File:** `performance_evaluation/04_learning_features.png`

Contains:
- **Learning Curve**: Shows model performance vs training set size
- **Feature Importance**: Bar chart of feature importance
- **Cross-Validation Scores**: CV performance across folds
- **Summary Statistics**: Complete model performance summary

## Usage

### Basic Usage

```bash
python evaluate_model_performance.py \
    --csv "dataset/labels_user_fixed.csv" \
    --roots "Lowload" "HighLoad" \
    --model_dir "artifacts_criticality" \
    --output_dir "performance_evaluation"
```

### Command Line Arguments

- `--csv`: Path to labels CSV file (default: `dataset/labels_user_fixed.csv`)
- `--roots`: Directories to search for images (default: `Lowload HighLoad`)
- `--model_dir`: Directory containing trained model (default: `artifacts_criticality`)
- `--output_dir`: Output directory for graphs (default: `performance_evaluation`)
- `--low_threshold`: Low/Medium threshold (default: `0.33`)
- `--medium_threshold`: Medium/High threshold (default: `0.67`)

## Graph Features

### Grid Layout
- All graphs use **2x2 grid layouts** for comprehensive visualization
- Consistent styling across all figures
- Professional appearance suitable for reports

### Dotted Grid Styling
- **Dotted grid lines** (`linestyle='--'`) for better readability
- Grid alpha: 0.3-0.5 for subtle appearance
- Grid enabled on all axes for easy reading

### Color Scheme
- **Steel blue**: True values, primary data
- **Coral**: Predicted values, secondary data
- **Green/Orange**: Threshold lines (Low/Med, Med/High)
- **Red**: Perfect prediction lines, error indicators

### Metrics Displayed

#### Classification Metrics
- Accuracy
- Precision (per-class)
- Recall (per-class)
- F1-Score (per-class)
- Confusion Matrix

#### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Residuals Analysis

#### Model Analysis
- Learning Curves
- Feature Importance
- Cross-Validation Scores
- Data Statistics

## Output

### Generated Files

```
performance_evaluation/
├── 01_classification_performance.png  (402 KB)
├── 02_regression_performance.png      (505 KB)
├── 03_criticality_distribution.png    (372 KB)
└── 04_learning_features.png           (66 KB)
```

### Console Output

The script prints:
- Dataset loading progress
- Feature extraction progress
- Model loading confirmation
- Graph generation progress
- Final metrics summary

## Example Output

```
======================================================================
Model Performance Evaluation
======================================================================

[1] Loading dataset from dataset/labels_user_fixed.csv...
  ✓ Loaded 55 images

[2] Extracting features from 55 images...
  ✓ Features shape: (55, 6)
  ✓ Criticality range: 0.105 - 0.995

[3] Loading model from artifacts_criticality...
  ✓ Model loaded (thresholds: 0.33, 0.67)

[4] Generating performance evaluation graphs...
  Test Accuracy: 1.000
  MAE: 0.0176, RMSE: 0.0243, R²: 0.9952
  ✓ Saved: performance_evaluation/01_classification_performance.png
  ✓ Saved: performance_evaluation/02_regression_performance.png
  ✓ Saved: performance_evaluation/03_criticality_distribution.png
  ✓ Saved: performance_evaluation/04_learning_features.png

✓ All performance evaluation graphs generated successfully!
  Output directory: performance_evaluation/
  Generated 4 comprehensive evaluation figures

======================================================================
Evaluation Complete!
======================================================================

Final Metrics:
  Accuracy: 1.0000
  MAE: 0.0176
  RMSE: 0.0243
  R²: 0.9952
  CV MAE: 0.0679 ± 0.0301
======================================================================
```

## Interpretation Guide

### Classification Performance
- **Confusion Matrix**: Diagonal values = correct predictions
- **Precision**: Higher is better (fewer false positives)
- **Recall**: Higher is better (fewer false negatives)
- **F1-Score**: Balance between precision and recall

### Regression Performance
- **R² Score**: Closer to 1.0 = better fit
  - R² > 0.9: Excellent
  - R² > 0.7: Good
  - R² < 0.7: Moderate
- **MAE/RMSE**: Lower is better
- **Residuals**: Should be randomly distributed around 0

### Learning Curves
- **Training vs Validation**: Gap indicates overfitting
- **Convergence**: Both should stabilize at high performance
- **Variance**: Shaded regions show model stability

### Feature Importance
- **Higher bars**: More important features
- **Consistent importance**: Model uses all features
- **Dominant features**: May indicate over-reliance on specific features

## Tips

1. **High Resolution**: All graphs are saved at 300 DPI for publication quality
2. **Grid Layout**: Easy to compare multiple metrics at once
3. **Dotted Grid**: Improves readability without cluttering
4. **Color Coding**: Consistent colors across all graphs
5. **Threshold Lines**: Visual indication of classification boundaries

## Troubleshooting

### If graphs don't generate:
1. Check that model directory exists and contains trained model
2. Verify CSV file path and image directories
3. Ensure matplotlib backend is set to 'Agg' (non-interactive)
4. Check that output directory is writable

### If graphs are empty:
1. Verify dataset has sufficient data (minimum 10 samples)
2. Check that features were extracted successfully
3. Ensure model was trained on compatible data

## Next Steps

1. Review generated graphs for model performance
2. Identify areas for improvement
3. Adjust thresholds if needed
4. Retrain model if performance is unsatisfactory
5. Generate new evaluation graphs after improvements

---

## Summary

The performance evaluation script generates **4 comprehensive evaluation figures** in a **grid layout with dotted styling**, providing:
- ✅ Classification performance analysis
- ✅ Regression performance analysis
- ✅ Criticality distribution analysis
- ✅ Model learning and feature analysis

All graphs use consistent styling, dotted grids, and professional formatting suitable for reports and presentations.

