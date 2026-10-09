"""
src/model_predictor.py
======================
Model Predictor Wrapper for Driver Fatigue Binary Classification.

Responsibility:
- Load saved Random Forest model (models/fatigue_binary_model.joblib) and metadata (models/model_metadata.json)
- Enforce strict feature ordering: [left_ear, right_ear, ear, mar, yaw, pitch, roll]
- Dynamically map class probabilities using model.classes_
- Validate live feature vector for finite numbers before inference
- Provide optional probability smoothing for stable real-time display
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "fatigue_binary_model.joblib")
DEFAULT_METADATA_PATH = str(BASE_DIR / "models" / "model_metadata.json")

REQUIRED_FEATURES = ["left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll"]


class ModelPredictor:
    """
    Wrapper for loading trained ML model and performing real-time binary prediction.
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH, smoothing_alpha=0.35):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.smoothing_alpha = smoothing_alpha

        self.model = None
        self.metadata = None
        self.is_loaded = False
        self.class_labels = ["Drowsy", "Not Drowsy"]
        self.smoothed_probabilities = None

        self._load_model_and_metadata()

    def _load_model_and_metadata(self):
        """
        Loads model file and metadata JSON cleanly with error handling.
        """
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at '{self.model_path}'. Live ML inference disabled.")
            return

        try:
            self.model = joblib.load(self.model_path)
            print(f"Loaded ML model cleanly from '{self.model_path}'")
        except Exception as e:
            print(f"Error loading ML model file '{self.model_path}': {e}")
            return

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                print(f"Loaded model metadata cleanly from '{self.metadata_path}'")
                
                # Validate metadata features
                meta_features = self.metadata.get("features", [])
                if meta_features != REQUIRED_FEATURES:
                    print(f"Warning: Metadata features {meta_features} mismatch expected {REQUIRED_FEATURES}")
            except Exception as e:
                print(f"Warning: Could not read metadata file '{self.metadata_path}': {e}")

        # Extract class labels from trained classifier
        if hasattr(self.model, "classes_"):
            self.class_labels = list(self.model.classes_)
            print(f"Classifier Class Ordering: {self.class_labels}")

        self.is_loaded = True

    def predict(self, feature_dict):
        """
        Performs binary classification on live numerical feature vector dictionary.

        Parameters:
            feature_dict (dict): Dictionary containing keys:
                'left_ear', 'right_ear', 'ear', 'mar', 'yaw', 'pitch', 'roll'

        Returns dict:
            'is_available' (bool): True if inference succeeded
            'predicted_label' (str): 'Drowsy' or 'Not Drowsy' (or 'N/A')
            'confidence' (float): Confidence percentage (0-100)
            'probabilities' (dict): {'Drowsy': prob, 'Not Drowsy': prob}
            'status' (str): Descriptive status message
        """
        if not self.is_loaded or self.model is None:
            return {
                "is_available": False,
                "predicted_label": "N/A",
                "confidence": 0.0,
                "probabilities": {"Drowsy": 0.0, "Not Drowsy": 0.0},
                "status": "Model Not Loaded"
            }

        # 1. Validate feature vector presence and non-null values
        if not feature_dict:
            self.smoothed_probabilities = None
            return {
                "is_available": False,
                "predicted_label": "N/A",
                "confidence": 0.0,
                "probabilities": {"Drowsy": 0.0, "Not Drowsy": 0.0},
                "status": "No Face Detected"
            }

        # Check required numerical features
        vector = []
        for feat in REQUIRED_FEATURES:
            val = feature_dict.get(feat)
            if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                self.smoothed_probabilities = None
                return {
                    "is_available": False,
                    "predicted_label": "N/A",
                    "confidence": 0.0,
                    "probabilities": {"Drowsy": 0.0, "Not Drowsy": 0.0},
                    "status": "Invalid/Missing Features"
                }
            vector.append(float(val))

        # 2. Format 2D numpy array in exact training feature order
        X = np.array([vector], dtype=np.float64)

        # 3. Model Inference
        try:
            raw_pred = self.model.predict(X)[0]
            raw_probs = self.model.predict_proba(X)[0]
        except Exception as e:
            return {
                "is_available": False,
                "predicted_label": "N/A",
                "confidence": 0.0,
                "probabilities": {"Drowsy": 0.0, "Not Drowsy": 0.0},
                "status": f"Inference Error: {e}"
            }

        # 4. Map probabilities dynamically to class labels using self.model.classes_
        prob_dict = {}
        for cls, prob in zip(self.model.classes_, raw_probs):
            prob_dict[str(cls)] = float(prob)

        # Ensure both Drowsy and Not Drowsy keys exist
        for label in ["Drowsy", "Not Drowsy"]:
            if label not in prob_dict:
                prob_dict[label] = 0.0

        # 5. Apply optional temporal probability smoothing for stable UI rendering
        if self.smoothed_probabilities is None:
            self.smoothed_probabilities = prob_dict.copy()
        else:
            for label in prob_dict:
                self.smoothed_probabilities[label] = (
                    self.smoothing_alpha * prob_dict[label] + 
                    (1.0 - self.smoothing_alpha) * self.smoothed_probabilities[label]
                )

        # Display label based on smoothed probabilities (or raw prediction if equal)
        smoothed_drowsy_prob = self.smoothed_probabilities.get("Drowsy", 0.0)
        smoothed_not_drowsy_prob = self.smoothed_probabilities.get("Not Drowsy", 0.0)

        if smoothed_drowsy_prob > smoothed_not_drowsy_prob:
            disp_label = "Drowsy"
            confidence = smoothed_drowsy_prob * 100.0
        else:
            disp_label = "Not Drowsy"
            confidence = smoothed_not_drowsy_prob * 100.0

        return {
            "is_available": True,
            "predicted_label": disp_label,
            "raw_predicted_label": str(raw_pred),
            "confidence": round(confidence, 1),
            "probabilities": {k: round(v * 100.0, 1) for k, v in self.smoothed_probabilities.items()},
            "raw_probabilities": {k: round(v * 100.0, 1) for k, v in prob_dict.items()},
            "status": "OK"
        }
