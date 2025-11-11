from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .preprocessing import (
    image_to_temperature_matrix,
    extract_column_signals,
    hjorth_parameters_per_signal,
)


def aggregate_hjorth_features(
    activity: np.ndarray,
    mobility: np.ndarray,
    complexity: np.ndarray,
) -> np.ndarray:
    def safe_stats(x: np.ndarray):
        if x.size == 0:
            return 0.0, 0.0
        return float(np.nanmean(x)), float(np.nanstd(x))

    a_mean, a_std = safe_stats(activity)
    m_mean, m_std = safe_stats(mobility)
    c_mean, c_std = safe_stats(complexity)

    features = np.array([a_mean, a_std, m_mean, m_std, c_mean, c_std], dtype=np.float32)
    return features


def preprocess_image_to_features(
    image: np.ndarray,
    mode: str = "auto",
    min_temp_c: float = 20.0,
    max_temp_c: float = 120.0,
    calibration_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    temp_c = image_to_temperature_matrix(
        image=image,
        mode=mode,
        min_temp_c=min_temp_c,
        max_temp_c=max_temp_c,
        calibration_fn=calibration_fn,
    )
    signals = extract_column_signals(temp_c)
    activity, mobility, complexity = hjorth_parameters_per_signal(signals)
    features = aggregate_hjorth_features(activity, mobility, complexity)
    debug = {
        "temp_c": temp_c,
        "signals": signals,
        "activity": activity,
        "mobility": mobility,
        "complexity": complexity,
    }
    return features, debug


