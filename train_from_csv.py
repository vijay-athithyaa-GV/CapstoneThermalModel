import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from busbar.dataset import build_feature_table
from busbar.model import MultiHeadModel
from busbar.onnx_utils import export_to_onnx

LABELS = ["Low Load", "Medium Load", "High Load"]


def resolve_abs_paths(df: pd.DataFrame, search_roots: list[str]) -> pd.DataFrame:
    def finder(rel_or_name: str) -> str:
        # If absolute path exists, keep
        if os.path.isabs(rel_or_name) and os.path.exists(rel_or_name):
            return rel_or_name
        # Try each root
        for root in search_roots:
            cand = os.path.join(root, rel_or_name)
            if os.path.exists(cand):
                return cand
        # As fallback, try basename in each root
        base = os.path.basename(rel_or_name)
        for root in search_roots:
            cand = os.path.join(root, base)
            if os.path.exists(cand):
                return cand
        return rel_or_name
    df = df.copy()
    df["abs_path"] = df["filepath"].apply(finder)
    return df


def train_evaluate(df: pd.DataFrame, random_state: int = 42):
    X, y_cls_str, y_reg = build_feature_table(df)
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls_str, y_reg, test_size=0.2, random_state=random_state, stratify=y_cls_str
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    base_clf = Pipeline([("scaler", StandardScaler()),
                         ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state))])
    cv_acc = cross_val_score(base_clf, X_train, LabelEncoder().fit_transform(y_cls_train), cv=skf, scoring="accuracy")

    model = MultiHeadModel(random_state=random_state)
    model.fit(X_train, y_cls_train, y_reg_train)

    y_cls_pred, y_reg_pred = model.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=sorted(np.unique(y_cls_str)))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(np.unique(y_cls_str))).plot(ax=axs[0], colorbar=False)
    axs[0].set_title(f"Confusion Matrix (acc={acc:.3f})")
    axs[1].hist(y_reg_test - y_reg_pred, bins=20)
    axs[1].set_title(f"Regression residuals (MAE={mae:.3f}, MSE={mse:.3f})")
    plt.tight_layout()
    plt.savefig("training_from_csv_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"CV Accuracy (mean±std): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train model from a labels CSV produced from folders")
    ap.add_argument("--csv", type=str, required=True, help="Path to labels CSV (filepath,label,criticality)")
    ap.add_argument("--roots", type=str, nargs="*", default=[],
                    help="One or more directories to search for images (e.g., 'HighLoad' 'Low load' 'dataset')")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "filepath" not in df.columns or "label" not in df.columns or "criticality" not in df.columns:
        raise SystemExit("CSV must include columns: filepath,label,criticality")
    df = resolve_abs_paths(df, args.roots)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())

    model = train_evaluate(df)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.artifacts_dir)
    clf_onnx, reg_onnx = export_to_onnx(model, args.artifacts_dir)
    print("Saved artifacts to", args.artifacts_dir)
    print("ONNX:", clf_onnx, reg_onnx)


if __name__ == "__main__":
    main()


