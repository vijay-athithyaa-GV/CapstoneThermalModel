import os
from pathlib import Path

import numpy as np
import pandas as pd
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


def ensure_synthetic_dataset(dataset_root: str, csv_path: str):
    if os.path.exists(csv_path):
        return
    os.makedirs(dataset_root, exist_ok=True)
    records = []
    for i, (label, score) in enumerate([
        ("Low Load", 0.1), ("Medium Load", 0.5), ("High Load", 0.9),
        ("Low Load", 0.2), ("Medium Load", 0.6), ("High Load", 0.85),
    ]):
        img = (plt.cm.inferno(np.random.rand(120,160))[..., :3]*255).astype(np.uint8)
        import cv2
        fname = f"synthetic_{i}.png"
        cv2.imwrite(os.path.join(dataset_root, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        records.append({"filepath": fname, "label": label, "criticality": score})
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Synthetic dataset created at {dataset_root}")


def train_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y_cls_str, y_reg = build_feature_table(df)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls_str, y_reg, test_size=test_size, random_state=random_state, stratify=y_cls_str
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    base_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state))
    ])
    cv_acc = cross_val_score(base_clf, X_train, LabelEncoder().fit_transform(y_cls_train), cv=skf, scoring='accuracy')

    model = MultiHeadModel(random_state=random_state)
    model.fit(X_train, y_cls_train, y_reg_train)

    y_cls_pred, y_reg_pred = model.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=LABELS)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)

    train_sizes, train_scores, val_scores = learning_curve(
        model.clf, X_train, LabelEncoder().fit_transform(y_cls_train), cv=skf, scoring='accuracy', train_sizes=np.linspace(0.2, 1.0, 5)
    )

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
    plt.show()

    print(f"CV Accuracy (mean±std): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Regression MAE: {mae:.4f}")
    print(f"Regression MSE: {mse:.4f}")

    return model, {"cv_acc": cv_acc, "test_acc": acc, "mae": mae, "mse": mse}


def main():
    project_dir = os.getcwd()
    dataset_root = os.path.join(project_dir, "dataset")
    csv_path = os.path.join(dataset_root, "labels.csv")
    ensure_synthetic_dataset(dataset_root, csv_path)

    cfg = DatasetConfig(root_dir=dataset_root, csv_path=csv_path)
    df = load_dataset(cfg)
    print(f"Loaded {len(df)} rows")

    model, _ = train_evaluate(df)

    model_dir = os.path.join(project_dir, "artifacts")
    model.save(model_dir)
    print(f"Saved joblib models to {model_dir}")

    clf_onnx, reg_onnx = export_to_onnx(model, model_dir)
    print("ONNX exported:", clf_onnx, reg_onnx)


if __name__ == "__main__":
    main()


