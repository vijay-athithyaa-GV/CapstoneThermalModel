from pathlib import Path
from typing import Tuple

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MultiHeadModel:
    def __init__(self, random_state: int = 42):
        self.label_encoder = LabelEncoder()
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state))
        ])
        self.reg = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state))
        ])

    def fit(self, X: np.ndarray, y_cls_str: np.ndarray, y_reg: np.ndarray):
        y_cls_enc = self.label_encoder.fit_transform(y_cls_str)
        self.clf.fit(X, y_cls_enc)
        self.reg.fit(X, y_reg)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_cls_enc = self.clf.predict(X)
        y_cls_str = self.label_encoder.inverse_transform(y_cls_enc)
        y_reg = np.clip(self.reg.predict(X), 0.0, 1.0)
        return y_cls_str, y_reg

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def save(self, folder: str):
        Path(folder).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoder, str(Path(folder)/"label_encoder.joblib"))
        joblib.dump(self.clf, str(Path(folder)/"classifier.joblib"))
        joblib.dump(self.reg, str(Path(folder)/"regressor.joblib"))

    @staticmethod
    def load(folder: str) -> "MultiHeadModel":
        model = MultiHeadModel()
        model.label_encoder = joblib.load(str(Path(folder)/"label_encoder.joblib"))
        model.clf = joblib.load(str(Path(folder)/"classifier.joblib"))
        model.reg = joblib.load(str(Path(folder)/"regressor.joblib"))
        return model


