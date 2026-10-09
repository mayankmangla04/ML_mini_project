"""
src/feature_extractor.py
========================
Central Feature Extractor Module for Driver Safety Monitor.

Responsibility:
- Combine Eye Features (EAR, Eye Closure State, Monotonic Closure Duration)
- Combine Mouth Features (MAR)
- Combine Head Pose (Yaw, Pitch, Roll in degrees)
- Output standardized numerical feature vector schema for ML training & inference
- Agnostic to driver status or classification predictions
"""

import time
from src.mouth_features import compute_mar
from src.head_pose import estimate_head_pose
from config.settings import FEATURE_SMOOTHING_ALPHA


class FeatureExtractor:
    """
    Central Feature Extractor combining eye, mouth, and head pose features.
    """
    def __init__(self, alpha=FEATURE_SMOOTHING_ALPHA):
        self.alpha = alpha
        self.smooth_states = {}  # face_id -> dict of smoothed features

    def extract_features(self, face_data, frame_w, frame_h, current_timestamp=None):
        """
        Extracts feature vector dictionary for a single detected face.

        Parameters:
            face_data (dict): Dict containing 'face_id', 'pixel_landmarks', 'eye_calculator'
            frame_w (int): Frame width
            frame_h (int): Frame height
            current_timestamp (float): Monotonic timestamp

        Returns:
            features (dict): Standardized feature dictionary schema.
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        if not face_data or not face_data.get("pixel_landmarks"):
            return self.get_empty_features()

        face_id = face_data["face_id"]
        pixel_landmarks = face_data["pixel_landmarks"]
        eye_calculator = face_data["eye_calculator"]

        # 1. Compute Eye Features (EAR, Closed States, Closure Durations)
        eye_feats = eye_calculator.compute_eye_features(pixel_landmarks, current_timestamp)

        # 2. Compute Mouth Features (MAR)
        mar = compute_mar(pixel_landmarks)

        # 3. Compute 3D Head Pose Angles (Yaw, Pitch, Roll in degrees)
        yaw, pitch, roll = estimate_head_pose(pixel_landmarks, frame_w, frame_h)

        raw_features = {
            "left_ear": eye_feats["left_ear"],
            "right_ear": eye_feats["right_ear"],
            "ear": eye_feats["ear"],
            "mar": mar,
            "left_eye_closed": eye_feats["left_eye_closed"],
            "right_eye_closed": eye_feats["right_eye_closed"],
            "both_eyes_closed": eye_feats["both_eyes_closed"],
            "left_eye_closed_duration": eye_feats["left_eye_closed_duration"],
            "right_eye_closed_duration": eye_feats["right_eye_closed_duration"],
            "both_eyes_closed_duration": eye_feats["both_eyes_closed_duration"],
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll
        }

        # 4. Apply Exponential Moving Average (EMA) smoothing to continuous features
        smoothed_features = self._apply_smoothing(face_id, raw_features)

        return smoothed_features

    def _apply_smoothing(self, face_id, raw_features):
        if face_id not in self.smooth_states:
            self.smooth_states[face_id] = {}

        state = self.smooth_states[face_id]
        smoothed = {}

        # Continuous numerical keys to smooth
        smooth_keys = {"left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll"}

        for key, val in raw_features.items():
            if key in smooth_keys and val is not None and isinstance(val, (int, float)):
                if key not in state or state[key] is None:
                    state[key] = float(val)
                else:
                    state[key] = self.alpha * float(val) + (1.0 - self.alpha) * state[key]
                smoothed[key] = float(state[key])
            else:
                smoothed[key] = val

        return smoothed

    @staticmethod
    def get_empty_features():
        """
        Returns empty feature dictionary schema when face is missing.
        """
        return {
            "left_ear": None,
            "right_ear": None,
            "ear": None,
            "mar": None,
            "left_eye_closed": 0,
            "right_eye_closed": 0,
            "both_eyes_closed": 0,
            "left_eye_closed_duration": 0.0,
            "right_eye_closed_duration": 0.0,
            "both_eyes_closed_duration": 0.0,
            "yaw": None,
            "pitch": None,
            "roll": None
        }
