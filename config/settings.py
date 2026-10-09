"""
config/settings.py
==================
Centralized configuration parameters for Driver Fatigue & Distraction Safety Monitor.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# MediaPipe Model Configuration
MODEL_PATH = str(BASE_DIR / "models" / "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Camera Configuration
CAMERA_INDICES = [0, 1, 2]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Multi-Face Detection Configuration
MAX_NUM_FACES = 4
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Feature Extraction Parameters
EYE_CLOSED_EAR_THRESHOLD = 0.21  # EAR below this threshold is measured as eye closed
FEATURE_SMOOTHING_ALPHA = 0.35   # EMA smoothing factor for raw features
TRACKING_TIMEOUT_SECONDS = 1.0   # Time before removing a lost face from tracker

# Temporal Monitoring Configuration
TEMPORAL_WINDOW_SECONDS = 60.0   # Rolling window duration for PERCLOS, blinks, and yawns
YAWN_MAR_THRESHOLD = 0.55        # MAR threshold for yawning detection
YAWN_MIN_DURATION_SECONDS = 1.2  # Minimum continuous time MAR > threshold to count a yawn
BLINK_MAX_DURATION_SECONDS = 0.5  # Maximum duration of eye closure to count as a single blink

# Head Orientation Deviation Thresholds (Degrees)
HEAD_DEVIATION_YAW_THRESHOLD = 25.0
HEAD_DEVIATION_PITCH_THRESHOLD = 20.0
HEAD_DEVIATION_ROLL_THRESHOLD = 20.0

# Audio & Warning System Configuration
ENABLE_AUDIO_WARNINGS = True     # Master toggle for audio alert sound
WARNING_COOLDOWN_SECONDS = 4.0   # Minimum seconds between audio alerts
WARNING_DROWSY_PROB_THRESHOLD = 65.0  # Smoothed Drowsy probability % threshold for Warning state
PERCLOS_WARNING_THRESHOLD = 35.0      # PERCLOS % threshold for Warning state
