import math
from typing import Callable, Optional, Tuple

import numpy as np


def image_to_temperature_matrix(
    image: np.ndarray,
    mode: str = "auto",
    min_temp_c: float = 20.0,
    max_temp_c: float = 120.0,
    calibration_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    inferred_mode = mode
    if mode == "auto":
        if image.ndim == 2:
            inferred_mode = "grayscale"
        elif image.ndim == 3 and image.shape[2] == 3:
            inferred_mode = "rgb_pseudocolor"
        else:
            raise ValueError("Unsupported image shape for auto mode: " + str(image.shape))

    if inferred_mode == "raw16":
        if image.ndim != 2:
            raise ValueError("raw16 expects single-channel 16-bit image")
        raw = image.astype(np.float32)
        if raw.max() == raw.min():
            scaled = np.zeros_like(raw, dtype=np.float32)
        else:
            scaled = (raw - raw.min()) / (raw.max() - raw.min())
        temp_c = scaled * (max_temp_c - min_temp_c) + min_temp_c
        if calibration_fn is not None:
            temp_c = calibration_fn(raw)
        return temp_c.astype(np.float32)

    if inferred_mode == "grayscale":
        if image.ndim != 2:
            raise ValueError("grayscale expects single-channel image")
        gray = image.astype(np.float32)
        denom = (gray.max() - gray.min())
        scaled = np.zeros_like(gray, dtype=np.float32) if denom == 0 else (gray - gray.min()) / denom
        if calibration_fn is None:
            temp_c = scaled * (max_temp_c - min_temp_c) + min_temp_c
        else:
            temp_c = calibration_fn(gray)
        return temp_c.astype(np.float32)

    if inferred_mode == "rgb_pseudocolor":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb_pseudocolor expects 3-channel RGB image")
        rgb = image.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        if calibration_fn is None:
            temp_c = gray * (max_temp_c - min_temp_c) + min_temp_c
        else:
            temp_c = calibration_fn(gray)
        return temp_c.astype(np.float32)

    raise ValueError(f"Unsupported mode: {mode}")


def extract_column_signals(temp_c: np.ndarray) -> np.ndarray:
    if temp_c.ndim != 2:
        raise ValueError("temp_c must be 2D (H, W)")
    signals = temp_c.T
    return signals.astype(np.float32)


def hjorth_parameters(signal_1d: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal_1d must be 1D")

    var0 = np.var(x)
    if var0 <= 1e-12:
        return 0.0, 0.0, 0.0

    dx = np.diff(x)
    var1 = np.var(dx)
    mobility = math.sqrt(var1 / var0) if var1 > 0 else 0.0

    if len(dx) < 2:
        return float(var0), float(mobility), 0.0

    ddx = np.diff(dx)
    var2 = np.var(ddx)

    if var1 <= 1e-12 or mobility == 0.0:
        complexity = 0.0
    else:
        complexity = math.sqrt(var2 / var1) / mobility if var2 > 0 else 0.0

    return float(var0), float(mobility), float(complexity)


def hjorth_parameters_per_signal(signals: np.ndarray):
    if signals.ndim != 2:
        raise ValueError("signals must be 2D (N_signals, length)")

    n_signals = signals.shape[0]
    activity = np.zeros(n_signals, dtype=np.float32)
    mobility = np.zeros(n_signals, dtype=np.float32)
    complexity = np.zeros(n_signals, dtype=np.float32)

    for i in range(n_signals):
        a, m, c = hjorth_parameters(signals[i])
        activity[i] = a
        mobility[i] = m
        complexity[i] = c
    return activity, mobility, complexity


