from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from .features import preprocess_image_to_features


@dataclass
class DatasetConfig:
    root_dir: str
    csv_path: str
    image_col: str = "filepath"
    label_col: str = "label"
    score_col: str = "criticality"


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(Path(config.csv_path))
    required = {config.image_col, config.label_col, config.score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    root = Path(config.root_dir)
    df["abs_path"] = df[config.image_col].apply(lambda p: str(root / p))
    return df


def compute_features_for_row(row, min_temp_c=20.0, max_temp_c=120.0):
    img_path = row["abs_path"]
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats, _ = preprocess_image_to_features(
        img_rgb,
        mode="rgb_pseudocolor",
        min_temp_c=min_temp_c,
        max_temp_c=max_temp_c,
    )
    return feats


def build_feature_table(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list = []
    y_cls = []
    y_reg = []
    for _, row in df.iterrows():
        feats = compute_features_for_row(row)
        X_list.append(feats)
        y_cls.append(row["label"])  # string label
        y_reg.append(float(row["criticality"]))
    X = np.vstack(X_list).astype(np.float32)
    y_cls = np.array(y_cls)
    y_reg = np.clip(np.array(y_reg, dtype=np.float32), 0.0, 1.0)
    return X, y_cls, y_reg


