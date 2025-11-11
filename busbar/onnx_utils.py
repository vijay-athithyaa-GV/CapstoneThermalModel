from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from .model import MultiHeadModel


def export_to_onnx(model: MultiHeadModel, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    initial_type = [("input", FloatTensorType([None, 6]))]

    onnx_clf = convert_sklearn(model.clf, initial_types=initial_type)
    onnx_reg = convert_sklearn(model.reg, initial_types=initial_type)

    clf_path = str(Path(out_dir)/"classifier.onnx")
    reg_path = str(Path(out_dir)/"regressor.onnx")
    with open(clf_path, "wb") as f:
        f.write(onnx_clf.SerializeToString())
    with open(reg_path, "wb") as f:
        f.write(onnx_reg.SerializeToString())
    return clf_path, reg_path


def _pick_first_ndarray(outputs):
    for out in outputs:
        if isinstance(out, np.ndarray):
            return out
        if isinstance(out, (list, tuple)):
            for inner in out:
                if isinstance(inner, np.ndarray):
                    return inner
    raise RuntimeError("No ndarray output found from ONNX inference.")


def onnx_predict(clf_path: str, reg_path: str, X: np.ndarray, label_encoder):
    sess_clf = ort.InferenceSession(clf_path, providers=["CPUExecutionProvider"])
    sess_reg = ort.InferenceSession(reg_path, providers=["CPUExecutionProvider"])

    input_name = sess_clf.get_inputs()[0].name
    clf_outputs = sess_clf.run(None, {input_name: X.astype(np.float32)})
    proba = _pick_first_ndarray(clf_outputs)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    cls_idx = proba.argmax(axis=1)
    cls_str = label_encoder.inverse_transform(cls_idx)

    input_name_reg = sess_reg.get_inputs()[0].name
    reg_outputs = sess_reg.run(None, {input_name_reg: X.astype(np.float32)})
    reg_out = _pick_first_ndarray(reg_outputs).reshape(-1)
    reg_out = np.clip(reg_out, 0.0, 1.0)
    return cls_str, reg_out


