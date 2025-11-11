# Model Performance Graphs

## Generated Visualizations

All performance graphs have been generated and saved in the `performance_plots/` directory.

### 📊 Available Graphs

1. **confusion_matrix.png**
   - Confusion matrix showing classification accuracy
   - Displays true vs predicted labels for all classes
   - Includes overall accuracy metric

2. **classification_report.png**
   - Bar chart comparing Precision, Recall, and F1-Score
   - Shows performance metrics for each class (Low/Medium/High Load)
   - Helps identify which classes perform best/worst

3. **regression_performance.png**
   - 4-panel analysis of regression performance:
     - True vs Predicted scatter plot (with R² score)
     - Residual plot (errors vs predictions)
     - Error distribution histogram
     - Metrics summary (MAE, MSE, RMSE, R²)

4. **learning_curves.png**
   - Learning curves for both classification and regression
   - Shows training vs validation performance
   - Helps identify overfitting/underfitting
   - Classification: Accuracy over training size
   - Regression: MAE over training size

5. **feature_importance.png**
   - Feature importance for both classifier and regressor
   - Shows which Hjorth parameters are most important
   - Helps understand what the model relies on for predictions

6. **prediction_distribution.png**
   - 4-panel distribution analysis:
     - Classification distribution (true vs predicted counts)
     - True criticality score distribution
     - Predicted criticality score distribution
     - Overlay comparison of true vs predicted

7. **roc_curves.png**
   - ROC (Receiver Operating Characteristic) curves
   - Multi-class ROC analysis
   - Shows AUC (Area Under Curve) for each class
   - Higher AUC = better classification performance

---

## How to View

### Option 1: Open Directly
Navigate to `performance_plots/` folder and open any PNG file with:
- Image viewer
- Web browser
- Any image viewing application

### Option 2: View in Python
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('performance_plots/confusion_matrix.png')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Option 3: Generate Again
```bash
python plot_model_performance.py
```

---

## Graph Details

### Confusion Matrix
- **Purpose**: Shows classification accuracy per class
- **Interpretation**: 
  - Diagonal values = correct predictions
  - Off-diagonal = misclassifications
  - Higher diagonal = better performance

### Classification Report
- **Precision**: Of all predicted as class X, how many were actually X?
- **Recall**: Of all actual class X, how many were predicted as X?
- **F1-Score**: Harmonic mean of precision and recall
- **Best**: All metrics close to 1.0

### Regression Performance
- **R² Score**: How well model explains variance (1.0 = perfect)
- **MAE**: Average absolute error (lower = better)
- **MSE/RMSE**: Squared errors (lower = better)
- **Residuals**: Should be randomly distributed around zero

### Learning Curves
- **Training Score**: Performance on training data
- **Validation Score**: Performance on validation data
- **Good Model**: Both curves converge, validation close to training
- **Overfitting**: Large gap between training and validation
- **Underfitting**: Both scores are low

### Feature Importance
- Shows which of the 6 Hjorth features are most predictive
- Higher importance = more influential in predictions
- Helps understand model decision-making

### Prediction Distribution
- Shows how predictions are distributed
- Good model: Predicted distribution matches true distribution
- Helps identify bias or systematic errors

### ROC Curves
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier
- **AUC > 0.8**: Good classifier
- Higher curve = better performance

---

## Expected Performance

Based on training results:
- **Classification Accuracy**: ~96.5%
- **Regression MAE**: ~0.038 (3.8% error)
- **R² Score**: Should be > 0.9

---

## Regenerating Graphs

To regenerate all graphs:
```bash
python plot_model_performance.py
```

This will:
1. Load the dataset
2. Load the trained model
3. Generate predictions
4. Create all 7 performance graphs
5. Save to `performance_plots/` directory

---

## Customization

Edit `plot_model_performance.py` to:
- Change graph styles
- Modify figure sizes
- Add custom metrics
- Change color schemes
- Adjust DPI/resolution

---

## Notes

- All graphs are saved as PNG files (300 DPI)
- Graphs use professional styling with seaborn
- All plots include proper labels and legends
- Metrics are clearly displayed on graphs

---

**Status**: ✅ All performance graphs generated successfully!

