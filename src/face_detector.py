"""
src/face_detector.py
====================
Multi-Face Detection Module using MediaPipe Tasks API.

Responsibility:
- Detect ALL visible faces in the webcam frame
- Identify bounding box coordinates and primary driver face (largest face area)
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from config.settings import (
    MODEL_PATH, MODEL_URL, MAX_NUM_FACES,
    MIN_FACE_DETECTION_CONFIDENCE, MIN_FACE_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE
)


class FaceDetector:
    """
    Detects multiple faces using MediaPipe Tasks API.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=MAX_NUM_FACES,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            return

        if os.path.exists("face_landmarker.task"):
            import shutil
            shutil.copy("face_landmarker.task", self.model_path)
            return

        print(f"Downloading MediaPipe face landmarker model to '{self.model_path}'...")
        try:
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe model asset: {e}")

    def detect_faces(self, frame):
        """
        Detects all visible faces in a BGR image frame.

        Returns:
            detected_faces (list of dict): List of face objects containing:
                'bbox': (min_x, min_y, max_x, max_y, area),
                'center': (center_x, center_y),
                'pixel_landmarks': [(x, y), ...]
        """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return []

        detected_faces = []
        for face_landmarks in detection_result.face_landmarks:
            pixel_landmarks = []
            all_x = []
            all_y = []

            for lm in face_landmarks:
                px = int(lm.x * img_w)
                py = int(lm.y * img_h)
                pixel_landmarks.append((px, py))
                all_x.append(px)
                all_y.append(py)

            min_x, max_x = max(0, min(all_x)), min(img_w, max(all_x))
            min_y, max_y = max(0, min(all_y)), min(img_h, max(all_y))
            area = (max_x - min_x) * (max_y - min_y)
            center = (int((min_x + max_x) / 2.0), int((min_y + max_y) / 2.0))

            detected_faces.append({
                "bbox": (min_x, min_y, max_x, max_y, area),
                "center": center,
                "pixel_landmarks": pixel_landmarks
            })

        return detected_faces
