"""
src/eye_features.py
===================
Eye Feature Calculation & Monotonic Eye-Closure Duration Tracker.

Responsibility:
- Calculate Left EAR, Right EAR, and Average EAR
- Determine Eye Closed State based on configurable EAR threshold
- Measure continuous Eye Closure Duration in seconds using monotonic clock
"""

import time
import numpy as np
from src.landmark_detector import extract_eye_landmarks
from config.settings import EYE_CLOSED_EAR_THRESHOLD


def calculate_single_ear(eye_points):
    """
    Calculates Eye Aspect Ratio (EAR) given 6 landmark points:
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]
    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h = np.linalg.norm(np.array(p1) - np.array(p4))

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


class EyeFeatureCalculator:
    """
    Calculates EAR values and tracks continuous eye-closure durations per face.
    Uses monotonic time (time.perf_counter()) for frame-rate independent timing.
    """
    def __init__(self, ear_threshold=EYE_CLOSED_EAR_THRESHOLD):
        self.ear_threshold = ear_threshold
        self.left_close_start_time = None
        self.right_close_start_time = None
        self.both_close_start_time = None

    def compute_eye_features(self, pixel_landmarks, current_timestamp=None):
        """
        Calculates EAR values, closure states, and elapsed closure durations.

        Returns dict:
            left_ear, right_ear, ear,
            left_eye_closed (int 0/1), right_eye_closed (int 0/1), both_eyes_closed (int 0/1),
            left_eye_closed_duration (float seconds),
            right_eye_closed_duration (float seconds),
            both_eyes_closed_duration (float seconds)
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        if not pixel_landmarks:
            self.reset_timers()
            return {
                "left_ear": None,
                "right_ear": None,
                "ear": None,
                "left_eye_closed": 0,
                "right_eye_closed": 0,
                "both_eyes_closed": 0,
                "left_eye_closed_duration": 0.0,
                "right_eye_closed_duration": 0.0,
                "both_eyes_closed_duration": 0.0
            }

        left_pts, right_pts = extract_eye_landmarks(pixel_landmarks)
        left_ear = calculate_single_ear(left_pts)
        right_ear = calculate_single_ear(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        left_closed = int(left_ear < self.ear_threshold)
        right_closed = int(right_ear < self.ear_threshold)
        both_closed = int(left_closed and right_closed)

        # 1. Left Eye Timer
        if left_closed:
            if self.left_close_start_time is None:
                self.left_close_start_time = current_timestamp
            left_duration = current_timestamp - self.left_close_start_time
        else:
            self.left_close_start_time = None
            left_duration = 0.0

        # 2. Right Eye Timer
        if right_closed:
            if self.right_close_start_time is None:
                self.right_close_start_time = current_timestamp
            right_duration = current_timestamp - self.right_close_start_time
        else:
            self.right_close_start_time = None
            right_duration = 0.0

        # 3. Both Eyes Timer
        if both_closed:
            if self.both_close_start_time is None:
                self.both_close_start_time = current_timestamp
            both_duration = current_timestamp - self.both_close_start_time
        else:
            self.both_close_start_time = None
            both_duration = 0.0

        return {
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "ear": float(avg_ear),
            "left_eye_closed": left_closed,
            "right_eye_closed": right_closed,
            "both_eyes_closed": both_closed,
            "left_eye_closed_duration": float(left_duration),
            "right_eye_closed_duration": float(right_duration),
            "both_eyes_closed_duration": float(both_duration)
        }

    def reset_timers(self):
        self.left_close_start_time = None
        self.right_close_start_time = None
        self.both_close_start_time = None
