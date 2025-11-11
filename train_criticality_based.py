"""
Train criticality-based model where classification is derived from criticality score
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from busbar.dataset import build_feature_table
from busbar.model_criticality_based import CriticalityBasedModel
from busbar.onnx_utils import export_to_onnx


LABELS = ["Low Load", "Medium Load", "High Load"]


def train_evaluate(df: pd.DataFrame, low_threshold: float = 0.33, medium_threshold: float = 0.67, random_state: int = 42):
    """Train and evaluate criticality-based model"""
    print(f"\nBuilding feature table from {len(df)} images...")
    
    # Build features - need to adapt build_feature_table or create our own
    from busbar.dataset import compute_features_for_row
    X_list = []
    y_reg = []
    for _, row in df.iterrows():
        try:
            feats = compute_features_for_row(row)
            X_list.append(feats)
            y_reg.append(float(row["criticality"]))
        except Exception as e:
            print(f"Warning: Skipping {row.get('filepath', 'unknown')}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No images could be processed")
    
    X = np.vstack(X_list).astype(np.float32)
    y_reg = np.clip(np.array(y_reg, dtype=np.float32), 0.0, 1.0)

    # Derive classification from criticality for evaluation
    def crit_to_class(crit):
        if crit < low_threshold:
            return "Low Load"
        elif crit < medium_threshold:
            return "Medium Load"
        else:
            return "High Load"
    
    y_cls_from_crit = np.array([crit_to_class(c) for c in y_reg])

    # Train/test split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=random_state
    )
    
    y_cls_test = np.array([crit_to_class(c) for c in y_reg_test])

    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    print(f"\nCriticality thresholds:")
    print(f"  Low Load: < {low_threshold}")
    print(f"  Medium Load: {low_threshold} - {medium_threshold}")
    print(f"  High Load: >= {medium_threshold}")

    # Cross-validation on regression
    reg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=random_state))
    ])
    cv_scores = cross_val_score(reg_pipeline, X_train, y_reg_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Train model
    print("\nTraining CriticalityBasedModel...")
    model = CriticalityBasedModel(random_state=random_state, 
                                  low_threshold=low_threshold,
                                  medium_threshold=medium_threshold)
    model.fit(X_train, y_reg_train)

    # Evaluate
    y_reg_pred = model.predict_criticality(X_test)
    y_cls_pred = np.array([model.criticality_to_class(c) for c in y_reg_pred])
    
    acc = accuracy_score(y_cls_test, y_cls_pred)
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=LABELS)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(ax=axs[0], colorbar=False)
    axs[0].set_title(f"Confusion Matrix (acc={acc:.3f})")
    
    # Criticality distribution
    axs[1].hist(y_reg_test, bins=30, alpha=0.5, label='True', color='blue')
    axs[1].hist(y_reg_pred, bins=30, alpha=0.5, label='Predicted', color='red')
    axs[1].axvline(low_threshold, color='green', linestyle='--', label=f'Low/Med threshold ({low_threshold})')
    axs[1].axvline(medium_threshold, color='orange', linestyle='--', label=f'Med/High threshold ({medium_threshold})')
    axs[1].set_xlabel('Criticality Score')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Criticality Distribution')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    # Residuals
    axs[2].scatter(y_reg_test, y_reg_pred, alpha=0.6)
    axs[2].plot([0, 1], [0, 1], 'r--', label='Perfect')
    axs[2].axvline(low_threshold, color='green', linestyle='--', alpha=0.5)
    axs[2].axvline(medium_threshold, color='orange', linestyle='--', alpha=0.5)
    axs[2].set_xlabel('True Criticality')
    axs[2].set_ylabel('Predicted Criticality')
    axs[2].set_title(f'True vs Predicted (MAE={mae:.4f})')
    axs[2].legend()
    axs[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_criticality_based_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training plot: training_criticality_based_results.png")
    plt.close()

    print(f"\n{'='*70}")
    print("Training Results")
    print(f"{'='*70}")
    print(f"CV MAE (mean±std): {cv_mae:.4f} ± {cv_scores.std():.4f}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")
    print(f"\nClassification from Criticality:")
    print(f"  Low Load: criticality < {low_threshold}")
    print(f"  Medium Load: {low_threshold} ≤ criticality < {medium_threshold}")
    print(f"  High Load: criticality ≥ {medium_threshold}")
    print(f"{'='*70}")

    return model, {"cv_mae": cv_mae, "test_acc": acc, "mae": mae, "mse": mse}


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Train criticality-based model")
    ap.add_argument("--csv", type=str, default="dataset/labels_3class.csv",
                    help="Path to labels CSV")
    ap.add_argument("--roots", type=str, nargs="*", default=["Low load", "HighLoad", "dataset"],
                    help="Directories to search for images")
    ap.add_argument("--low_threshold", type=float, default=0.33,
                    help="Criticality threshold for Low/Medium (default: 0.33)")
    ap.add_argument("--medium_threshold", type=float, default=0.67,
                    help="Criticality threshold for Medium/High (default: 0.67)")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts_criticality")
    args = ap.parse_args()
    
    # Load dataset
    project_dir = os.getcwd()
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(project_dir, args.csv)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns or "criticality" not in df.columns:
        raise SystemExit("CSV must include columns: filepath,criticality")
    
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
    
    df = resolve_abs_paths(df, args.roots)
    
    print("="*70)
    print("Training Criticality-Based Model")
    print("="*70)
    print(f"\nLoaded {len(df)} images from {csv_path}")
    print("Label distribution:")
    if "label" in df.columns:
        print(df["label"].value_counts().to_string())
    
    # Train
    model, metrics = train_evaluate(df, 
                                    low_threshold=args.low_threshold,
                                    medium_threshold=args.medium_threshold)
    
    # Save
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.artifacts_dir)
    print(f"\n✓ Saved model to {args.artifacts_dir}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

