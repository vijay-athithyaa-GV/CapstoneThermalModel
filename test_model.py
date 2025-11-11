"""
Comprehensive test suite for Busbar Heat Detection Model
Tests all images in dataset, compares joblib vs ONNX, and generates metrics
"""
import os
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report

from busbar.dataset import DatasetConfig, load_dataset, build_feature_table
from busbar.model import MultiHeadModel
from busbar.onnx_utils import onnx_predict


def test_joblib_inference(model_dir: str, X: np.ndarray, y_cls_true: np.ndarray, y_reg_true: np.ndarray):
    """Test joblib model inference"""
    print("\n" + "="*60)
    print("Testing Joblib Model")
    print("="*60)
    
    model = MultiHeadModel.load(model_dir)
    y_cls_pred, y_reg_pred = model.predict(X)
    
    acc = accuracy_score(y_cls_true, y_cls_pred)
    mae = mean_absolute_error(y_reg_true, y_reg_pred)
    mse = mean_squared_error(y_reg_true, y_reg_pred)
    
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")
    print(f"Regression RMSE: {np.sqrt(mse):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_cls_true, y_cls_pred))
    
    return {
        "accuracy": float(acc),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "predictions_cls": y_cls_pred.tolist(),
        "predictions_reg": y_reg_pred.tolist()
    }


def test_onnx_inference(model_dir: str, X: np.ndarray, y_cls_true: np.ndarray, y_reg_true: np.ndarray):
    """Test ONNX model inference"""
    print("\n" + "="*60)
    print("Testing ONNX Model")
    print("="*60)
    
    model = MultiHeadModel.load(model_dir)
    clf_path = str(Path(model_dir) / "classifier.onnx")
    reg_path = str(Path(model_dir) / "regressor.onnx")
    
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        print("ERROR: ONNX models not found. Run train.py first.")
        return None
    
    y_cls_pred, y_reg_pred = onnx_predict(clf_path, reg_path, X, model.label_encoder)
    
    acc = accuracy_score(y_cls_true, y_cls_pred)
    mae = mean_absolute_error(y_reg_true, y_reg_pred)
    mse = mean_squared_error(y_reg_true, y_reg_pred)
    
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")
    print(f"Regression RMSE: {np.sqrt(mse):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_cls_true, y_cls_pred))
    
    return {
        "accuracy": float(acc),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "predictions_cls": y_cls_pred.tolist(),
        "predictions_reg": y_reg_pred.tolist()
    }


def compare_models(joblib_results: dict, onnx_results: dict):
    """Compare joblib vs ONNX predictions"""
    print("\n" + "="*60)
    print("Model Comparison (Joblib vs ONNX)")
    print("="*60)
    
    if onnx_results is None:
        print("Skipping comparison - ONNX results not available")
        return
    
    # Classification comparison
    joblib_cls = np.array(joblib_results["predictions_cls"])
    onnx_cls = np.array(onnx_results["predictions_cls"])
    cls_agreement = np.mean(joblib_cls == onnx_cls)
    
    print(f"\nClassification Agreement: {cls_agreement:.4f} ({cls_agreement*100:.2f}%)")
    
    # Regression comparison
    joblib_reg = np.array(joblib_results["predictions_reg"])
    onnx_reg = np.array(onnx_results["predictions_reg"])
    reg_diff = np.abs(joblib_reg - onnx_reg)
    
    print(f"\nRegression Differences:")
    print(f"  Mean absolute difference: {np.mean(reg_diff):.6f}")
    print(f"  Max absolute difference: {np.max(reg_diff):.6f}")
    print(f"  Std of differences: {np.std(reg_diff):.6f}")
    
    # Accuracy comparison
    print(f"\nAccuracy Comparison:")
    print(f"  Joblib: {joblib_results['accuracy']:.4f}")
    print(f"  ONNX:   {onnx_results['accuracy']:.4f}")
    print(f"  Difference: {abs(joblib_results['accuracy'] - onnx_results['accuracy']):.6f}")
    
    # MAE comparison
    print(f"\nMAE Comparison:")
    print(f"  Joblib: {joblib_results['mae']:.4f}")
    print(f"  ONNX:   {onnx_results['mae']:.4f}")
    print(f"  Difference: {abs(joblib_results['mae'] - onnx_results['mae']):.6f}")


def test_individual_images(model_dir: str, df: pd.DataFrame, n_samples: int = 5):
    """Test individual images and show detailed predictions"""
    print("\n" + "="*60)
    print(f"Individual Image Predictions (showing {n_samples} samples)")
    print("="*60)
    
    model = MultiHeadModel.load(model_dir)
    
    from busbar.dataset import compute_features_for_row
    
    samples = df.sample(min(n_samples, len(df)), random_state=42)
    
    print(f"\n{'Image':<30} {'True Label':<15} {'Pred Label':<15} {'True Score':<12} {'Pred Score':<12} {'Match':<8}")
    print("-" * 100)
    
    correct = 0
    for idx, row in samples.iterrows():
        try:
            feats = compute_features_for_row(row)
            y_cls_pred, y_reg_pred = model.predict(feats.reshape(1, -1))
            
            true_label = row["label"]
            true_score = row["criticality"]
            pred_label = y_cls_pred[0]
            pred_score = y_reg_pred[0]
            match = "✓" if true_label == pred_label else "✗"
            
            if match == "✓":
                correct += 1
            
            print(f"{row['filepath']:<30} {true_label:<15} {pred_label:<15} {true_score:<12.4f} {pred_score:<12.4f} {match:<8}")
        except Exception as e:
            print(f"{row['filepath']:<30} ERROR: {str(e)}")
    
    print(f"\nCorrect predictions: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")


def main():
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    csv_path = os.path.join(dataset_root, "labels.csv")
    model_dir = os.path.join(project_dir, "artifacts")
    
    # Check if artifacts exist
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory '{model_dir}' not found.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)
    
    # Load dataset
    print("Loading dataset...")
    cfg = DatasetConfig(root_dir=dataset_root, csv_path=csv_path)
    df = load_dataset(cfg)
    print(f"Loaded {len(df)} images")
    
    # Build feature table
    print("\nExtracting features from images...")
    X, y_cls_true, y_reg_true = build_feature_table(df)
    print(f"Feature shape: {X.shape}")
    
    # Test joblib model
    joblib_results = test_joblib_inference(model_dir, X, y_cls_true, y_reg_true)
    
    # Test ONNX model
    onnx_results = test_onnx_inference(model_dir, X, y_cls_true, y_reg_true)
    
    # Compare models
    compare_models(joblib_results, onnx_results)
    
    # Test individual images
    test_individual_images(model_dir, df, n_samples=10)
    
    # Save results
    results = {
        "joblib": joblib_results,
        "onnx": onnx_results,
        "dataset_size": len(df),
        "feature_dim": X.shape[1]
    }
    
    results_path = os.path.join(project_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Test results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

