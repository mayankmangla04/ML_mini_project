"""
src/camera_app.py
=================
Main Real-Time Application for Continuous Driver Safety & ML Monitoring.

Pipeline Architecture:
Webcam Frame
  ↓
Face Detection & Multi-Face Tracking (FaceDetector, FaceTracker)
  ↓
Instantaneous Feature Extraction (FeatureExtractor: EAR, MAR, Head Pose)
  ↓
Binary ML Model Inference (ModelPredictor: Raw Drowsy / Not Drowsy)
  ↓
Temporal Aggregation (TemporalMonitor: PERCLOS, Blinks, Yawns, Head Deviation, Smoothed Score)
  ↓
Driver Warning Manager (WarningManager: Multi-state Warning & Audio Cooldown Alerts)
  ↓
Dual-Panel Monitoring HUD (Visualizer)
"""

import sys
import time
import cv2
import numpy as np

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_tracker import FaceTracker
from src.feature_extractor import FeatureExtractor
from src.model_predictor import ModelPredictor
from src.temporal_monitor import TemporalMonitor
from src.warning_manager import WarningManager
from src.visualizer import (
    draw_face_bounding_boxes, render_measurement_hud, render_continuous_monitoring_hud
)
from config.settings import MODEL_PATH, ENABLE_AUDIO_WARNINGS


def run():
    """
    Launches the continuous driver monitoring application.
    """
    print("==================================================")
    print(" CONTINUOUS DRIVER SAFETY MONITORING PROTOTYPE ")
    print("==================================================")

    # 1. Initialize Computer Vision Components
    try:
        face_detector = FaceDetector(model_path=MODEL_PATH)
        print(f"MediaPipe Face Landmarker : OK ({MODEL_PATH})")
    except Exception as e:
        print(f"ERROR: Failed to initialize FaceDetector: {e}")
        sys.exit(1)

    face_tracker = FaceTracker()
    feature_extractor = FeatureExtractor()

    # 2. Initialize ML Model Predictor
    print("Initializing Trained Random Forest ML Classifier...")
    model_predictor = ModelPredictor()
    if model_predictor.is_loaded:
        print("ML Model Predictor        : READY (Random Forest Baseline)")
    else:
        print("ML Model Predictor        : DISABLED (Model file or metadata missing)")

    # 3. Initialize Temporal Monitor & Warning Manager
    temporal_monitor = TemporalMonitor(window_seconds=60.0)
    warning_manager = WarningManager(enable_audio=ENABLE_AUDIO_WARNINGS)
    print("Temporal Monitor (60s)    : READY (PERCLOS, Blinks, Yawns, Head Deviation)")
    print(f"Driver Warning Manager    : READY (Audio Enabled: {warning_manager.enable_audio})")

    # 4. Initialize Camera
    camera = Camera()
    if not camera.is_opened():
        print("ERROR: Could not open any working webcam device!")
        sys.exit(1)

    print(f"Camera Connected          : Index {camera.camera_index} | Backend {camera.backend}")
    print("--------------------------------------------------")
    print("Monitoring driver behaviour over time (EAR, MAR, Head Pose, PERCLOS, Blinks, Yawns, ML Predictions)...")
    print("Press 'Q' on the video window or close window to exit.")
    print("==================================================")

    window_name = "Driver Safety Monitor - Continuous Driver Monitoring Prototype"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("Warning: Camera frame lost.")
                break

            # Mirror frame horizontally for intuitive driver preview
            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape
            res_str = f"{img_w}x{img_h}"
            current_timestamp = time.perf_counter()

            # 1. Detect ALL visible faces in frame
            detected_faces = face_detector.detect_faces(frame)

            # 2. Update multi-face tracking & identify primary driver candidate
            tracked_results, primary_driver_id = face_tracker.update(detected_faces, current_timestamp)

            # 3. Extract numerical feature vectors for all tracked faces
            primary_driver_features = FeatureExtractor.get_empty_features()

            for face_data in tracked_results:
                face_id = face_data["face_id"]
                features = feature_extractor.extract_features(face_data, img_w, img_h, current_timestamp)

                if face_id == primary_driver_id:
                    primary_driver_features = features

            # 4. Perform ML Model Binary Prediction for Primary Driver
            ml_prediction = model_predictor.predict(primary_driver_features)

            # 5. Update Temporal Aggregator (PERCLOS %, Blinks, Yawns, Head Deviation, Prediction Smoothing)
            temporal_metrics = temporal_monitor.update(primary_driver_features, ml_prediction, current_timestamp)

            # 6. Update Warning Manager (Multi-State Warnings & Audio Alerts)
            warning_info = warning_manager.update(temporal_metrics, ml_prediction, current_timestamp)

            # 7. Render clean bounding boxes (NO landmark lines across face)
            draw_face_bounding_boxes(frame, tracked_results, primary_driver_id)

            # 8. Render Left Measurement HUD Panel (Instantaneous Features)
            render_measurement_hud(frame, primary_driver_features, res_str)

            # 9. Render Right Continuous Monitoring HUD Panel (ML, PERCLOS, Blinks, Yawns, Warnings)
            render_continuous_monitoring_hud(frame, ml_prediction, temporal_metrics, warning_info)

            # Display video frame
            cv2.imshow(window_name, frame)

            # Check key press ('q' or 'Q')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("User pressed 'Q'. Exiting application...")
                break

            # Check window close button ('X')
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting...")
                break

    finally:
        # Resource cleanup
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Webcam released and windows closed. Continuous driver monitor stopped cleanly.")


if __name__ == "__main__":
    run()
