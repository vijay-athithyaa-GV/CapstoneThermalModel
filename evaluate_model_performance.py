"""
Comprehensive Model Performance Evaluation with Grid Layout
Generates multiple performance graphs in a grid structure with dotted styling
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

from busbar.dataset import compute_features_for_row
from busbar.model_criticality_based import CriticalityBasedModel


# Set style with dotted grid
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 1.2

# Set color palette
sns.set_palette("husl")
LABELS = ["Low Load", "Medium Load", "High Load"]


def load_data_and_model(csv_path: str, search_roots: list, model_dir: str, 
                        low_threshold: float = 0.33, medium_threshold: float = 0.67):
    """Load dataset and trained model"""
    print(f"\n[1] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} images")
    
    # Resolve absolute paths
    def resolve_abs_paths(df, search_roots):
        def finder(rel_or_name):
            if os.path.isabs(rel_or_name) and os.path.exists(rel_or_name):
                return rel_or_name
            for root in search_roots:
                cand = os.path.join(root, rel_or_name)
                if os.path.exists(cand):
                    return cand
            base = os.path.basename(rel_or_name)
            for root in search_roots:
                cand = os.path.join(root, base)
                if os.path.exists(cand):
                    return cand
            return rel_or_name
        df = df.copy()
        df["abs_path"] = df["filepath"].apply(finder)
        return df
    
    df = resolve_abs_paths(df, search_roots)
    
    # Extract features
    print(f"\n[2] Extracting features from {len(df)} images...")
    X_list = []
    y_reg = []
    y_cls = []
    failed = 0
    
    for _, row in df.iterrows():
        try:
            feats = compute_features_for_row(row)
            X_list.append(feats)
            y_reg.append(float(row["criticality"]))
            if "label" in row:
                y_cls.append(row["label"])
        except Exception as e:
            failed += 1
            continue
    
    if failed > 0:
        print(f"  ⚠ Skipped {failed} images due to errors")
    
    X = np.vstack(X_list).astype(np.float32)
    y_reg = np.clip(np.array(y_reg, dtype=np.float32), 0.0, 1.0)
    
    # Derive classification from criticality
    def crit_to_class(crit):
        if crit < low_threshold:
            return "Low Load"
        elif crit < medium_threshold:
            return "Medium Load"
        else:
            return "High Load"
    
    y_cls = np.array([crit_to_class(c) for c in y_reg])
    
    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Criticality range: {y_reg.min():.3f} - {y_reg.max():.3f}")
    
    # Load model
    print(f"\n[3] Loading model from {model_dir}...")
    model = CriticalityBasedModel.load(model_dir)
    print(f"  ✓ Model loaded (thresholds: {model.low_threshold}, {model.medium_threshold})")
    
    return X, y_reg, y_cls, model


def create_comprehensive_evaluation(X, y_reg, y_cls, model, output_dir: str = "performance_evaluation"):
    """Create comprehensive evaluation plots in grid layout"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train/test split for evaluation
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    # Get predictions
    y_reg_pred = model.predict_criticality(X_test)
    y_cls_pred = np.array([model.criticality_to_class(c) for c in y_reg_pred])
    
    # Calculate metrics
    acc = accuracy_score(y_cls_test, y_cls_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    print(f"\n[4] Generating performance evaluation graphs...")
    print(f"  Test Accuracy: {acc:.3f}")
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Feature names
    feature_names = [
        "mean(activity)", "std(activity)",
        "mean(mobility)", "std(mobility)",
        "mean(complexity)", "std(complexity)"
    ]
    
    # ========== FIGURE 1: Classification Performance (2x2 Grid) ==========
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle('Classification Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 1.1 Confusion Matrix
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes1[0, 0],
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='gray')
    axes1[0, 0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes1[0, 0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes1[0, 0].set_title(f'Confusion Matrix (Accuracy: {acc:.3f})', fontsize=12, fontweight='bold', pad=10)
    axes1[0, 0].grid(True, linestyle='--', alpha=0.3)
    
    # 1.2 Classification Report
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_cls_test, y_cls_pred, labels=LABELS, average=None, zero_division=0
    )
    x = np.arange(len(LABELS))
    width = 0.25
    axes1[0, 1].bar(x - width, precision, width, label='Precision', alpha=0.8, color='steelblue')
    axes1[0, 1].bar(x, recall, width, label='Recall', alpha=0.8, color='coral')
    axes1[0, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='mediumseagreen')
    axes1[0, 1].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes1[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes1[0, 1].set_title('Precision, Recall, F1-Score by Class', fontsize=12, fontweight='bold', pad=10)
    axes1[0, 1].set_xticks(x)
    axes1[0, 1].set_xticklabels(LABELS, rotation=45, ha='right')
    axes1[0, 1].set_ylim([0, 1.1])
    axes1[0, 1].legend(loc='upper right')
    axes1[0, 1].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 1.3 Classification Distribution
    unique_true, counts_true = np.unique(y_cls_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_cls_pred, return_counts=True)
    x_pos = np.arange(len(LABELS))
    width_bar = 0.35
    axes1[1, 0].bar(x_pos - width_bar/2, 
                    [counts_true[np.where(unique_true == label)[0][0]] if label in unique_true else 0 
                     for label in LABELS],
                    width_bar, label='True', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1)
    axes1[1, 0].bar(x_pos + width_bar/2,
                    [counts_pred[np.where(unique_pred == label)[0][0]] if label in unique_pred else 0 
                     for label in LABELS],
                    width_bar, label='Predicted', alpha=0.8, color='coral', edgecolor='black', linewidth=1)
    axes1[1, 0].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes1[1, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes1[1, 0].set_title('True vs Predicted Distribution', fontsize=12, fontweight='bold', pad=10)
    axes1[1, 0].set_xticks(x_pos)
    axes1[1, 0].set_xticklabels(LABELS, rotation=45, ha='right')
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 1.4 Metrics Summary
    axes1[1, 1].axis('off')
    metrics_text = f"""
    Classification Metrics Summary
    
    Accuracy:           {acc:.4f}
    
    Per-Class Metrics:
    """
    for i, label in enumerate(LABELS):
        if label in unique_true:
            metrics_text += f"\n  {label:15s}"
            metrics_text += f"\n    Precision: {precision[i]:.4f}"
            metrics_text += f"\n    Recall:    {recall[i]:.4f}"
            metrics_text += f"\n    F1-Score:  {f1[i]:.4f}"
    axes1[1, 1].text(0.1, 0.5, metrics_text, fontsize=10,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{output_dir}/01_classification_performance.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/01_classification_performance.png")
    plt.close()
    
    # ========== FIGURE 2: Regression Performance (2x2 Grid) ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Regression Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 2.1 True vs Predicted Scatter
    axes2[0, 0].scatter(y_reg_test, y_reg_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='steelblue')
    axes2[0, 0].plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()],
                    'r--', lw=2, label='Perfect Prediction')
    axes2[0, 0].axvline(model.low_threshold, color='green', linestyle='--', alpha=0.5, label=f'Low/Med ({model.low_threshold})')
    axes2[0, 0].axvline(model.medium_threshold, color='orange', linestyle='--', alpha=0.5, label=f'Med/High ({model.medium_threshold})')
    axes2[0, 0].set_xlabel('True Criticality Score', fontsize=11, fontweight='bold')
    axes2[0, 0].set_ylabel('Predicted Criticality Score', fontsize=11, fontweight='bold')
    axes2[0, 0].set_title(f'True vs Predicted (R² = {r2:.4f})', fontsize=12, fontweight='bold', pad=10)
    axes2[0, 0].legend(loc='upper left')
    axes2[0, 0].grid(True, linestyle='--', alpha=0.3)
    
    # 2.2 Residuals Plot
    residuals = y_reg_test - y_reg_pred
    axes2[0, 1].scatter(y_reg_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='coral')
    axes2[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes2[0, 1].set_xlabel('Predicted Criticality Score', fontsize=11, fontweight='bold')
    axes2[0, 1].set_ylabel('Residuals (True - Predicted)', fontsize=11, fontweight='bold')
    axes2[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold', pad=10)
    axes2[0, 1].grid(True, linestyle='--', alpha=0.3)
    
    # 2.3 Error Distribution
    axes2[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue', linewidth=1)
    axes2[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes2[1, 0].axvline(x=residuals.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.4f}')
    axes2[1, 0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes2[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes2[1, 0].set_title(f'Error Distribution (MAE = {mae:.4f})', fontsize=12, fontweight='bold', pad=10)
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 2.4 Metrics Summary
    axes2[1, 1].axis('off')
    metrics_text = f"""
    Regression Performance Metrics
    
    Mean Absolute Error (MAE):  {mae:.4f}
    Mean Squared Error (MSE):   {mse:.4f}
    Root Mean Squared Error:    {rmse:.4f}
    R² Score:                   {r2:.4f}
    
    Interpretation:
    • Lower MAE/MSE/RMSE = Better
    • Higher R² = Better (max = 1.0)
    • R² = {r2:.4f} indicates {'excellent' if r2 > 0.9 else 'good' if r2 > 0.7 else 'moderate'} fit
    """
    axes2[1, 1].text(0.1, 0.5, metrics_text, fontsize=11,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{output_dir}/02_regression_performance.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/02_regression_performance.png")
    plt.close()
    
    # ========== FIGURE 3: Criticality Distribution (2x2 Grid) ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    fig3.suptitle('Criticality Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 3.1 True Criticality Distribution
    axes3[0, 0].hist(y_reg_test, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)
    axes3[0, 0].axvline(model.low_threshold, color='green', linestyle='--', lw=2, label=f'Low/Med ({model.low_threshold})')
    axes3[0, 0].axvline(model.medium_threshold, color='orange', linestyle='--', lw=2, label=f'Med/High ({model.medium_threshold})')
    axes3[0, 0].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes3[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes3[0, 0].set_title('True Criticality Distribution', fontsize=12, fontweight='bold', pad=10)
    axes3[0, 0].legend()
    axes3[0, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 3.2 Predicted Criticality Distribution
    axes3[0, 1].hist(y_reg_pred, bins=30, alpha=0.7, color='coral', edgecolor='black', linewidth=1)
    axes3[0, 1].axvline(model.low_threshold, color='green', linestyle='--', lw=2, label=f'Low/Med ({model.low_threshold})')
    axes3[0, 1].axvline(model.medium_threshold, color='orange', linestyle='--', lw=2, label=f'Med/High ({model.medium_threshold})')
    axes3[0, 1].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes3[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes3[0, 1].set_title('Predicted Criticality Distribution', fontsize=12, fontweight='bold', pad=10)
    axes3[0, 1].legend()
    axes3[0, 1].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 3.3 Overlay Comparison
    axes3[1, 0].hist(y_reg_test, bins=30, alpha=0.5, color='steelblue', edgecolor='black', linewidth=1, label='True')
    axes3[1, 0].hist(y_reg_pred, bins=30, alpha=0.5, color='coral', edgecolor='black', linewidth=1, label='Predicted')
    axes3[1, 0].axvline(model.low_threshold, color='green', linestyle='--', alpha=0.7)
    axes3[1, 0].axvline(model.medium_threshold, color='orange', linestyle='--', alpha=0.7)
    axes3[1, 0].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes3[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes3[1, 0].set_title('True vs Predicted Overlay', fontsize=12, fontweight='bold', pad=10)
    axes3[1, 0].legend()
    axes3[1, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 3.4 Box Plot Comparison
    data_to_plot = [y_reg_test, y_reg_pred]
    bp = axes3[1, 1].boxplot(data_to_plot, tick_labels=['True', 'Predicted'], patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(linestyle='--'),
                             capprops=dict(linestyle='--'))
    axes3[1, 1].axhline(model.low_threshold, color='green', linestyle='--', alpha=0.5, label=f'Low/Med')
    axes3[1, 1].axhline(model.medium_threshold, color='orange', linestyle='--', alpha=0.5, label=f'Med/High')
    axes3[1, 1].set_ylabel('Criticality Score', fontsize=11, fontweight='bold')
    axes3[1, 1].set_title('Criticality Distribution (Box Plot)', fontsize=12, fontweight='bold', pad=10)
    axes3[1, 1].legend()
    axes3[1, 1].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{output_dir}/03_criticality_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/03_criticality_distribution.png")
    plt.close()
    
    # ========== FIGURE 4: Learning Curves & Feature Importance (2x2 Grid) ==========
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 13))
    fig4.suptitle('Model Learning & Feature Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # 4.1 Learning Curve
    print("    Computing learning curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        model.reg, X_train, y_reg_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_absolute_error'
    )
    train_scores = -train_scores
    val_scores = -val_scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes4[0, 0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training MAE', linewidth=2, markersize=6)
    axes4[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    axes4[0, 0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation MAE', linewidth=2, markersize=6)
    axes4[0, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    axes4[0, 0].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    axes4[0, 0].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes4[0, 0].set_title('Learning Curve', fontsize=12, fontweight='bold', pad=10)
    axes4[0, 0].legend(loc='best', fontsize=9)
    axes4[0, 0].grid(True, linestyle='--', alpha=0.3)
    
    # 4.2 Feature Importance
    print("    Computing feature importance...")
    feature_importance = model.reg.named_steps['rf'].feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    bars = axes4[0, 1].barh(range(len(feature_names)), feature_importance[indices], 
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    axes4[0, 1].set_yticks(range(len(feature_names)))
    axes4[0, 1].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    axes4[0, 1].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes4[0, 1].set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10)
    axes4[0, 1].grid(True, linestyle='--', alpha=0.3, axis='x')
    axes4[0, 1].invert_yaxis()
    # Add value labels on bars
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        axes4[0, 1].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 4.3 Cross-Validation Scores
    print("    Computing cross-validation scores...")
    cv_scores = cross_val_score(model.reg, X_train, y_reg_train, cv=5, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_scores = -cv_scores
    bars_cv = axes4[1, 0].bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, 
                    color='mediumseagreen', edgecolor='black', linewidth=1)
    axes4[1, 0].axhline(cv_scores.mean(), color='red', linestyle='--', lw=2, 
                        label=f'Mean: {cv_scores.mean():.4f}')
    axes4[1, 0].axhline(cv_scores.mean() + cv_scores.std(), color='orange', linestyle=':', lw=1.5, 
                        label=f'±1 std', alpha=0.7)
    axes4[1, 0].axhline(cv_scores.mean() - cv_scores.std(), color='orange', linestyle=':', lw=1.5, alpha=0.7)
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars_cv, cv_scores)):
        height = bar.get_height()
        axes4[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    axes4[1, 0].set_xlabel('Fold', fontsize=11, fontweight='bold')
    axes4[1, 0].set_ylabel('MAE', fontsize=11, fontweight='bold')
    axes4[1, 0].set_title(f'Cross-Validation Scores\n(Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})',
                         fontsize=12, fontweight='bold', pad=10)
    axes4[1, 0].legend(loc='best', fontsize=9)
    axes4[1, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    axes4[1, 0].set_xticks(range(1, len(cv_scores) + 1))
    
    # 4.4 Summary Statistics
    axes4[1, 1].axis('off')
    # Create a more compact summary with better formatting
    summary_lines = [
        "Model Performance Summary",
        "",
        "Test Set Metrics:",
        f"  Accuracy:     {acc:.4f}",
        f"  MAE:          {mae:.4f}",
        f"  RMSE:         {rmse:.4f}",
        f"  R²:           {r2:.4f}",
        "",
        "Cross-Validation:",
        f"  CV MAE:       {cv_scores.mean():.4f}",
        f"  Std Dev:      {cv_scores.std():.4f}",
        "",
        "Data Statistics:",
        f"  Train:        {len(X_train)} samples",
        f"  Test:         {len(X_test)} samples",
        f"  Features:     {X.shape[1]}",
        "",
        "Classification Thresholds:",
        f"  Low Load:     < {model.low_threshold}",
        f"  Medium Load:  {model.low_threshold} - {model.medium_threshold}",
        f"  High Load:    ≥ {model.medium_threshold}"
    ]
    summary_text = "\n".join(summary_lines)
    axes4[1, 1].text(0.05, 0.95, summary_text, fontsize=9.5,
                    verticalalignment='top', horizontalalignment='left',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=12))
    
    # Adjust spacing
    plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.3, wspace=0.3)
    plt.savefig(f"{output_dir}/04_learning_features.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_dir}/04_learning_features.png")
    plt.close()
    
    print(f"\n✓ All performance evaluation graphs generated successfully!")
    print(f"  Output directory: {output_dir}/")
    print(f"  Generated {4} comprehensive evaluation figures")
    
    return {
        "accuracy": acc,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "cv_mae_mean": cv_scores.mean(),
        "cv_mae_std": cv_scores.std()
    }


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Evaluate model performance with comprehensive graphs")
    ap.add_argument("--csv", type=str, default="dataset/labels_user_fixed.csv",
                    help="Path to labels CSV")
    ap.add_argument("--roots", type=str, nargs="*", default=["Lowload", "HighLoad"],
                    help="Directories to search for images")
    ap.add_argument("--model_dir", type=str, default="artifacts_criticality",
                    help="Directory containing trained model")
    ap.add_argument("--output_dir", type=str, default="performance_evaluation",
                    help="Output directory for graphs")
    ap.add_argument("--low_threshold", type=float, default=0.33,
                    help="Low/Medium threshold (default: 0.33)")
    ap.add_argument("--medium_threshold", type=float, default=0.67,
                    help="Medium/High threshold (default: 0.67)")
    args = ap.parse_args()
    
    print("="*70)
    print("Model Performance Evaluation")
    print("="*70)
    
    # Load data and model
    X, y_reg, y_cls, model = load_data_and_model(
        args.csv, args.roots, args.model_dir,
        args.low_threshold, args.medium_threshold
    )
    
    # Generate evaluation graphs
    metrics = create_comprehensive_evaluation(X, y_reg, y_cls, model, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  CV MAE: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

