"""
Generate comprehensive performance graphs for the trained model
Creates visualizations for classification, regression, and feature analysis
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve, validation_curve

from busbar.dataset import DatasetConfig, load_dataset, build_feature_table
from busbar.model import MultiHeadModel
from busbar.features import preprocess_image_to_features

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, labels, save_path="performance_plots/confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_classification_report(y_true, y_pred, labels, save_path="performance_plots/classification_report.png"):
    """Plot classification metrics"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_regression_performance(y_true, y_pred, save_path="performance_plots/regression_performance.png"):
    """Plot regression performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot: True vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('True Criticality Score', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Criticality Score', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'True vs Predicted (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Criticality Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (True - Predicted)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Error distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f'Error Distribution (MAE = {mae:.4f})', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 4. Metrics summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    Regression Performance Metrics
    
    Mean Absolute Error (MAE):  {mae:.4f}
    Mean Squared Error (MSE):   {mse:.4f}
    Root Mean Squared Error:    {rmse:.4f}
    R² Score:                   {r2:.4f}
    
    Interpretation:
    • Lower MAE/MSE/RMSE = Better
    • Higher R² = Better (max = 1.0)
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Regression Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_learning_curves(model, X, y_cls, y_reg, save_path="performance_plots/learning_curves.png"):
    """Plot learning curves for both classification and regression"""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    
    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classification learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model.clf, X, y_cls_enc, cv=skf, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    axes[0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    axes[0].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Classification Learning Curve', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)
    
    # Regression learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model.reg, X, y_reg, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_absolute_error'
    )
    
    train_scores = -train_scores  # Convert to positive
    val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[1].plot(train_sizes, train_mean, 'o-', color='green', label='Training MAE', linewidth=2)
    axes[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='green')
    axes[1].plot(train_sizes, val_mean, 'o-', color='orange', label='Validation MAE', linewidth=2)
    axes[1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='orange')
    axes[1].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[1].set_title('Regression Learning Curve', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Learning Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path="performance_plots/feature_importance.png"):
    """Plot feature importance from RandomForest"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Classification feature importance
    clf_importance = model.clf.named_steps['rf'].feature_importances_
    indices_clf = np.argsort(clf_importance)[::-1]
    
    axes[0].barh(range(len(feature_names)), clf_importance[indices_clf], color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_yticklabels([feature_names[i] for i in indices_clf], fontsize=10)
    axes[0].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes[0].set_title('Classification Feature Importance', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Regression feature importance
    reg_importance = model.reg.named_steps['rf'].feature_importances_
    indices_reg = np.argsort(reg_importance)[::-1]
    
    axes[1].barh(range(len(feature_names)), reg_importance[indices_reg], color='coral', alpha=0.8)
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_yticklabels([feature_names[i] for i in indices_reg], fontsize=10)
    axes[1].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes[1].set_title('Regression Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_prediction_distribution(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, 
                                labels, save_path="performance_plots/prediction_distribution.png"):
    """Plot distribution of predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Classification distribution
    unique_true, counts_true = np.unique(y_true_cls, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred_cls, return_counts=True)
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, [counts_true[np.where(unique_true == label)[0][0]] 
                                  if label in unique_true else 0 for label in labels],
                   width, label='True', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, [counts_pred[np.where(unique_pred == label)[0][0]] 
                                  if label in unique_pred else 0 for label in labels],
                   width, label='Predicted', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Classification Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Regression distribution - True
    axes[0, 1].hist(y_true_reg, bins=30, alpha=0.7, color='steelblue', edgecolor='black', label='True')
    axes[0, 1].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('True Criticality Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Regression distribution - Predicted
    axes[1, 0].hist(y_pred_reg, bins=30, alpha=0.7, color='coral', edgecolor='black', label='Predicted')
    axes[1, 0].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Predicted Criticality Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Overlay comparison
    axes[1, 1].hist(y_true_reg, bins=30, alpha=0.5, color='steelblue', edgecolor='black', label='True')
    axes[1, 1].hist(y_pred_reg, bins=30, alpha=0.5, color='coral', edgecolor='black', label='Predicted')
    axes[1, 1].set_xlabel('Criticality Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('True vs Predicted Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_roc_curves(model, X, y_true_cls, labels, save_path="performance_plots/roc_curves.png"):
    """Plot ROC curves for multi-class classification"""
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true_cls)
    y_pred_proba = model.clf.predict_proba(X)
    
    # Binarize the output
    y_bin = label_binarize(y_true_enc, classes=np.arange(len(labels)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute ROC curve and AUC for each class
    colors = ['blue', 'red', 'green']
    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, 
               label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Generate all performance plots"""
    print("="*70)
    print("Generating Model Performance Graphs")
    print("="*70)
    
    # Load dataset
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    csv_path = os.path.join(dataset_root, "labels.csv")
    model_dir = os.path.join(project_dir, "artifacts")
    
    print("\n[1] Loading dataset and model...")
    cfg = DatasetConfig(root_dir=dataset_root, csv_path=csv_path)
    df = load_dataset(cfg)
    print(f"  ✓ Loaded {len(df)} images")
    
    # Build features
    print("\n[2] Extracting features...")
    X, y_cls_true, y_reg_true = build_feature_table(df)
    print(f"  ✓ Features shape: {X.shape}")
    
    # Load model
    print("\n[3] Loading trained model...")
    model = MultiHeadModel.load(model_dir)
    print("  ✓ Model loaded")
    
    # Get predictions
    print("\n[4] Generating predictions...")
    y_cls_pred, y_reg_pred = model.predict(X)
    print("  ✓ Predictions generated")
    
    # Feature names
    feature_names = [
        "mean(activity)",
        "std(activity)",
        "mean(mobility)",
        "std(mobility)",
        "mean(complexity)",
        "std(complexity)"
    ]
    
    labels = ["Low Load", "Medium Load", "High Load"]
    
    # Create output directory
    output_dir = "performance_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Generating Performance Graphs...")
    print("="*70)
    
    # Generate all plots
    plot_confusion_matrix(y_cls_true, y_cls_pred, labels, 
                          f"{output_dir}/confusion_matrix.png")
    
    plot_classification_report(y_cls_true, y_cls_pred, labels,
                              f"{output_dir}/classification_report.png")
    
    plot_regression_performance(y_reg_true, y_reg_pred,
                               f"{output_dir}/regression_performance.png")
    
    plot_learning_curves(model, X, y_cls_true, y_reg_true,
                        f"{output_dir}/learning_curves.png")
    
    plot_feature_importance(model, feature_names,
                           f"{output_dir}/feature_importance.png")
    
    plot_prediction_distribution(y_cls_true, y_cls_pred, y_reg_true, y_reg_pred,
                                labels, f"{output_dir}/prediction_distribution.png")
    
    plot_roc_curves(model, X, y_cls_true, labels,
                   f"{output_dir}/roc_curves.png")
    
    print("\n" + "="*70)
    print("✓ All performance graphs generated successfully!")
    print(f"  Saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()

