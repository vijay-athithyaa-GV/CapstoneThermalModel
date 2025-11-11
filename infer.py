import os
import sys
from pathlib import Path

import cv2
import numpy as np

from busbar.features import preprocess_image_to_features
from busbar.model import MultiHeadModel
from busbar.onnx_utils import onnx_predict


def infer_joblib(image_path: str, model_dir: str):
    model = MultiHeadModel.load(model_dir)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    feats, _ = preprocess_image_to_features(rgb, mode="rgb_pseudocolor", min_temp_c=20, max_temp_c=120)
    y_cls, y_reg = model.predict(feats.reshape(1, -1))
    print({"load_category": str(y_cls[0]), "criticality_score": float(y_reg[0])})


def infer_onnx(image_path: str, model_dir: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    feats, _ = preprocess_image_to_features(rgb, mode="rgb_pseudocolor", min_temp_c=20, max_temp_c=120)

    model = MultiHeadModel.load(model_dir)
    cls, reg = onnx_predict(
        str(Path(model_dir)/"classifier.onnx"),
        str(Path(model_dir)/"regressor.onnx"),
        feats.reshape(1, -1),
        model.label_encoder,
    )
    print({"load_category": str(cls[0]), "criticality_score": float(reg[0])})


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer.py <image_path> <model_dir> [--onnx]")
        sys.exit(1)
    image_path = sys.argv[1]
    model_dir = sys.argv[2]
    use_onnx = len(sys.argv) > 3 and sys.argv[3] == "--onnx"
    if use_onnx:
        infer_onnx(image_path, model_dir)
    else:
        infer_joblib(image_path, model_dir)


