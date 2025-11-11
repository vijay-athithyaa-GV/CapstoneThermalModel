"""
Train model with merged dataset (synthetic + real images)
Handles images from multiple directories
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from busbar.dataset import DatasetConfig, load_dataset, build_feature_table
from busbar.model import MultiHeadModel
from busbar.onnx_utils import export_to_onnx


LABELS = ["Low Load", "Medium Load", "High Load"]


def load_merged_dataset(merged_csv_path: str, dataset_root: str, real_images_root: str) -> pd.DataFrame:
    """
    Load merged dataset with images from multiple directories
    """
    df = pd.read_csv(merged_csv_path)
    
    # Create absolute paths
    def get_abs_path(filepath):
        # Check if it's a real image
        real_img_path = os.path.join(real_images_root, os.path.basename(filepath))
        if os.path.exists(real_img_path):
            return real_img_path
        # Otherwise, it's in the dataset folder
        return os.path.join(dataset_root, filepath)
    
    df["abs_path"] = df["filepath"].apply(get_abs_path)
    
    # Verify all images exist
    missing = []
    for idx, row in df.iterrows():
        if not os.path.exists(row["abs_path"]):
            missing.append(row["filepath"])
    
    if missing:
        print(f"Warning: {len(missing)} images not found:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    return df


def train_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train and evaluate model"""
    print(f"\nBuilding feature table from {len(df)} images...")
    X, y_cls_str, y_reg = build_feature_table(df)

    # Train/test split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls_str, y_reg, test_size=test_size, random_state=random_state, stratify=y_cls_str
    )

    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    base_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state))
    ])
    cv_acc = cross_val_score(base_clf, X_train, LabelEncoder().fit_transform(y_cls_train), cv=skf, scoring='accuracy')

    # Train final model
    print("\nTraining MultiHeadModel...")
    model = MultiHeadModel(random_state=random_state)
    model.fit(X_train, y_cls_train, y_reg_train)

    # Evaluate
    y_cls_pred, y_reg_pred = model.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=LABELS)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)

    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model.clf, X_train, LabelEncoder().fit_transform(y_cls_train), cv=skf, scoring='accuracy', train_sizes=np.linspace(0.2, 1.0, 5)
    )

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(ax=axs[0], colorbar=False)
    axs[0].set_title(f"Confusion Matrix (acc={acc:.3f})")

    axs[1].plot(train_sizes, train_scores.mean(axis=1), label='train')
    axs[1].plot(train_sizes, val_scores.mean(axis=1), label='cv')
    axs[1].set_title("Learning Curve (Classification)")
    axs[1].set_xlabel("Train size")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    axs[2].hist(y_reg_test - y_reg_pred, bins=20)
    axs[2].set_title(f"Regression residuals (MAE={mae:.3f}, MSE={mse:.3f})")
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training plot: training_results.png")
    plt.close()

    print(f"\n{'='*70}")
    print("Training Results")
    print(f"{'='*70}")
    print(f"CV Accuracy (mean±std): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")
    print(f"{'='*70}")

    return model, {"cv_acc": cv_acc, "test_acc": acc, "mae": mae, "mse": mse}


def main():
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    real_images_root = os.path.join(project_dir, "RealImageDataset")
    merged_csv = os.path.join(dataset_root, "labels_merged.csv")
    
    # Check if merged dataset exists
    if not os.path.exists(merged_csv):
        print(f"Error: Merged dataset not found: {merged_csv}")
        print("Please run 'python process_real_images.py' first")
        return
    
    # Load merged dataset
    print("="*70)
    print("Loading Merged Dataset (Synthetic + Real Images)")
    print("="*70)
    df = load_merged_dataset(merged_csv, dataset_root, real_images_root)
    print(f"Loaded {len(df)} images")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    
    # Train and evaluate
    model, metrics = train_evaluate(df)
    
    # Save model
    model_dir = os.path.join(project_dir, "artifacts")
    model.save(model_dir)
    print(f"\n✓ Saved joblib models to {model_dir}")
    
    # Export to ONNX
    clf_onnx, reg_onnx = export_to_onnx(model, model_dir)
    print(f"✓ ONNX exported: {clf_onnx}, {reg_onnx}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

