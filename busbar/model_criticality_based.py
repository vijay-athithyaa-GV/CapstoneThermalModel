"""
Criticality-Based Model
Classification is derived from criticality score:
- 0.0 - 0.33 → Low Load
- 0.33 - 0.67 → Medium Load
- 0.67 - 1.0 → High Load
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class CriticalityBasedModel:
    """
    Model that predicts criticality score (0-1), then derives classification from it.
    Classification is not independent - it's based on criticality ranges.
    """
    def __init__(self, random_state: int = 42, 
                 low_threshold: float = 0.33,
                 medium_threshold: float = 0.67):
        """
        Args:
            random_state: Random seed
            low_threshold: Criticality < this = Low Load (default: 0.33)
            medium_threshold: Criticality < this = Medium Load (default: 0.67)
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.reg = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state))
        ])

    def fit(self, X: np.ndarray, y_reg: np.ndarray):
        """
        Train only the regressor (criticality prediction)
        
        Args:
            X: Feature matrix (n_samples, 6)
            y_reg: Criticality scores (0-1)
        """
        self.reg.fit(X, y_reg)
        return self

    def predict_criticality(self, X: np.ndarray) -> np.ndarray:
        """Predict criticality scores only"""
        return np.clip(self.reg.predict(X), 0.0, 1.0)

    def criticality_to_class(self, criticality: float) -> str:
        """Convert criticality score to load category"""
        if criticality < self.low_threshold:
            return "Low Load"
        elif criticality < self.medium_threshold:
            return "Medium Load"
        else:
            return "High Load"

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both classification and criticality
        
        Returns:
            (load_categories, criticality_scores)
            Classification is derived from criticality scores
        """
        criticality = self.predict_criticality(X)
        categories = np.array([self.criticality_to_class(c) for c in criticality])
        return categories, criticality

    def save(self, folder: str):
        """Save model to disk"""
        Path(folder).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.reg, str(Path(folder)/"regressor.joblib"))
        joblib.dump({
            "low_threshold": self.low_threshold,
            "medium_threshold": self.medium_threshold
        }, str(Path(folder)/"model_config.joblib"))
        print(f"Saved model to {folder}")

    @staticmethod
    def load(folder: str) -> "CriticalityBasedModel":
        """Load model from disk"""
        model = CriticalityBasedModel()
        model.reg = joblib.load(str(Path(folder)/"regressor.joblib"))
        config = joblib.load(str(Path(folder)/"model_config.joblib"))
        model.low_threshold = config["low_threshold"]
        model.medium_threshold = config["medium_threshold"]
        return model

