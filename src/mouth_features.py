"""
src/mouth_features.py
=====================
Mouth Feature Calculation Module (Mouth Aspect Ratio - MAR).

Responsibility:
- Calculate MAR representing vertical mouth opening over horizontal width
- Agnostic to driver status or classification
"""

import numpy as np
from src.landmark_detector import extract_mouth_landmarks


def compute_mar(pixel_landmarks):
    """
    Calculates Mouth Aspect Ratio (MAR) using inner lip points & corners.
    Returns:
        mar (float): Numerical MAR ratio.
    """
    mouth_data = extract_mouth_landmarks(pixel_landmarks)
    if mouth_data is None:
        return 0.0

    p_left = np.array(mouth_data["corner_left"])
    p_right = np.array(mouth_data["corner_right"])

    v_sum = 0.0
    for p_top, p_bot in zip(mouth_data["top_pts"], mouth_data["bottom_pts"]):
        v_sum += np.linalg.norm(np.array(p_top) - np.array(p_bot))

    h = np.linalg.norm(p_left - p_right)

    if h == 0:
        return 0.0

    mar = v_sum / (2.0 * h)
    return float(mar)
