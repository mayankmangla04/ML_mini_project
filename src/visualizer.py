"""
src/visualizer.py
=================
Visualization & Compact HUD Display Module for Continuous Driver Safety Monitor.

Responsibility:
- Priority #1: Keep driver's face and webcam feed clearly visible & unobstructed
- Render subtle, thin corner bracket annotations around detected faces (NO landmark lines)
- Render Compact Instantaneous Features HUD Panel (Top-Left)
- Render Compact Continuous Monitoring & ML HUD Panel (Top-Right)
- Dark, nearly opaque backgrounds for high text contrast without cluttering center camera view
"""

import cv2
import numpy as np


def draw_face_bounding_boxes(frame, tracked_results, primary_driver_id):
    """
    Draws subtle, thin corner brackets and small face tags.
    NO landmark lines drawn across the face.
    """
    for item in tracked_results:
        face_id = item["face_id"]
        min_x, min_y, max_x, max_y, _ = item["bbox"]

        is_primary = (face_id == primary_driver_id)
        box_color = (0, 220, 180) if is_primary else (235, 160, 0)
        thickness = 1

        # Subtle corner brackets (10px length)
        corner_len = 10
        cv2.line(frame, (min_x, min_y), (min_x + corner_len, min_y), box_color, thickness)
        cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_len), box_color, thickness)

        cv2.line(frame, (max_x, min_y), (max_x - corner_len, min_y), box_color, thickness)
        cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_len), box_color, thickness)

        cv2.line(frame, (min_x, max_y), (min_x + corner_len, max_y), box_color, thickness)
        cv2.line(frame, (min_x, max_y), (min_x, max_y - corner_len), box_color, thickness)

        cv2.line(frame, (max_x, max_y), (max_x - corner_len, max_y), box_color, thickness)
        cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_len), box_color, thickness)

        # Small, subtle tag above face box
        label_text = f"Driver" if is_primary else f"Face {face_id}"
        cv2.putText(frame, label_text, (min_x, max(14, min_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, box_color, 1, cv2.LINE_AA)


def render_measurement_hud(frame, features, res_str):
    """
    Renders a Compact Instantaneous Features Panel (Top-Left corner).
    Leaves the center of the webcam view completely open for the driver.
    """
    img_h, img_w, _ = frame.shape
    panel_w = min(220, int(img_w * 0.35))
    panel_h = 245
    pad_x, pad_y = 10, 10

    # Nearly opaque dark container for crisp text contrast
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + panel_w, pad_y + panel_h), (12, 15, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.rectangle(frame, (pad_x, pad_y), (pad_x + panel_w, pad_y + panel_h), (40, 55, 75), 1)

    # Panel Title
    cv2.putText(frame, "INSTANT FEATURES", (pad_x + 8, pad_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 225, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (pad_x + 8, pad_y + 24), (pad_x + panel_w - 8, pad_y + 24), (50, 70, 90), 1)

    face_detected = (features.get("ear") is not None)

    # Face Status Indicator
    y_curr = pad_y + 40
    cv2.putText(frame, "Face:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 190, 210), 1, cv2.LINE_AA)
    status_str = "DETECTED" if face_detected else "NOT DETECTED"
    status_color = (0, 255, 120) if face_detected else (0, 80, 255)
    cv2.putText(frame, status_str, (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

    y_curr += 18
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    def fmt_num(val, unit="", decimals=3):
        if val is None or not face_detected:
            return "--"
        return f"{val:.{decimals}f}{unit}"

    # Left / Right EAR
    y_curr += 18
    l_ear = fmt_num(features.get("left_ear"), decimals=3)
    r_ear = fmt_num(features.get("right_ear"), decimals=3)
    cv2.putText(frame, "EAR (L/R):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{l_ear} / {r_ear}", (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # Average EAR
    y_curr += 18
    cv2.putText(frame, "EAR (Avg):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_num(features.get("ear")), (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    # MAR
    y_curr += 18
    cv2.putText(frame, "MAR:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_num(features.get("mar")), (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    # Eye Closure Durations
    y_curr += 18
    b_close = fmt_num(features.get("both_eyes_closed_duration"), "s", decimals=2)
    cv2.putText(frame, "Eye Closed:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, b_close, (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    # Head Pose Angles (Yaw, Pitch, Roll)
    y_curr += 18
    cv2.putText(frame, "Yaw (Turn):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_num(features.get("yaw"), " deg", decimals=1), (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.putText(frame, "Pitch (Tilt):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_num(features.get("pitch"), " deg", decimals=1), (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.putText(frame, "Roll (Side):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_num(features.get("roll"), " deg", decimals=1), (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer
    y_curr += 18
    cv2.putText(frame, f"Stream: {res_str}", (pad_x + 8, y_curr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 150, 170), 1, cv2.LINE_AA)


def render_continuous_monitoring_hud(frame, ml_prediction, temporal_metrics, warning_info):
    """
    Renders a Compact Continuous Monitoring & ML Panel (Top-Right corner).
    Leaves the center of the webcam view completely open for the driver.
    """
    img_h, img_w, _ = frame.shape
    panel_w = min(230, int(img_w * 0.36))
    panel_h = 245
    pad_x = max(10, img_w - panel_w - 10)
    pad_y = 10

    # Nearly opaque dark container for crisp text contrast
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + panel_w, pad_y + panel_h), (12, 15, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.rectangle(frame, (pad_x, pad_y), (pad_x + panel_w, pad_y + panel_h), (40, 55, 75), 1)

    # Header Title
    cv2.putText(frame, "CONTINUOUS MONITORING & ML", (pad_x + 8, pad_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (pad_x + 8, pad_y + 24), (pad_x + panel_w - 8, pad_y + 24), (50, 70, 90), 1)

    y_curr = pad_y + 40

    # ML Predictions (Raw vs Smoothed)
    is_ml_valid = ml_prediction.get("is_available", False)
    raw_label = ml_prediction.get("predicted_label", "N/A")
    raw_probs = ml_prediction.get("probabilities", {})
    raw_drowsy_prob = raw_probs.get("Drowsy", 0.0)

    smoothed_state = temporal_metrics.get("smoothed_state", "N/A")
    smoothed_drowsy_prob = temporal_metrics.get("smoothed_prob_drowsy", 0.0)

    # Line 1: Raw ML
    raw_color = (0, 80, 255) if raw_label == "Drowsy" else ((0, 255, 120) if is_ml_valid else (130, 150, 170))
    cv2.putText(frame, "Raw ML:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{raw_label} ({raw_drowsy_prob:.0f}%)", (pad_x + 95, y_curr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, raw_color, 1, cv2.LINE_AA)

    # Line 2: Smoothed ML
    y_curr += 18
    smooth_color = (0, 80, 255) if smoothed_state == "Drowsy" else ((0, 255, 120) if is_ml_valid else (130, 150, 170))
    cv2.putText(frame, "Smooth ML:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{smoothed_state} ({smoothed_drowsy_prob:.0f}%)", (pad_x + 95, y_curr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, smooth_color, 1, cv2.LINE_AA)

    y_curr += 18
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    # Temporal Metrics (PERCLOS, Blinks, Yawns, Head Deviation)
    y_curr += 18
    perclos = temporal_metrics.get("perclos", 0.0)
    blink_count = temporal_metrics.get("blink_count", 0)
    blink_rate = temporal_metrics.get("blink_rate_bpm", 0.0)
    yawn_count = temporal_metrics.get("yawn_count", 0)
    head_dev_dur = temporal_metrics.get("head_deviation_duration", 0.0)

    perclos_color = (0, 80, 255) if perclos >= 30.0 else ((0, 255, 255) if perclos >= 15.0 else (255, 255, 255))
    cv2.putText(frame, "PERCLOS (60s):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{perclos:.1f}%", (pad_x + 115, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, perclos_color, 1, cv2.LINE_AA)

    y_curr += 18
    cv2.putText(frame, "Blinks (60s):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{blink_count} ({blink_rate:.1f}/m)", (pad_x + 115, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.putText(frame, "Yawns (60s):", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{yawn_count}", (pad_x + 115, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    y_curr += 18
    cv2.putText(frame, "Head Dev:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    head_color = (0, 140, 255) if head_dev_dur >= 3.0 else (255, 255, 255)
    cv2.putText(frame, f"{head_dev_dur:.1f} s", (pad_x + 115, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.36, head_color, 1, cv2.LINE_AA)

    y_curr += 18
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    # System Warning Status
    y_curr += 18
    warn_state = warning_info.get("state", "UNAVAILABLE")
    warn_reason = warning_info.get("reason", "Observation Unavailable")

    if warn_state == "WARNING":
        warn_color = (0, 69, 255)  # Red
    elif warn_state == "ADVISORY":
        warn_color = (0, 200, 255) # Yellow/Orange
    elif warn_state == "NORMAL":
        warn_color = (0, 255, 120) # Green
    else:
        warn_color = (130, 150, 170)# Grey

    cv2.putText(frame, "Warning:", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 190, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, warn_state, (pad_x + 95, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.38, warn_color, 1, cv2.LINE_AA)

    y_curr += 16
    # Truncate reason if needed to fit panel width
    short_reason = (warn_reason[:22] + "..") if len(warn_reason) > 24 else warn_reason
    cv2.putText(frame, f"Trig: {short_reason}", (pad_x + 8, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 170, 190), 1, cv2.LINE_AA)

    y_curr += 16
    cv2.line(frame, (pad_x + 8, y_curr), (pad_x + panel_w - 8, y_curr), (40, 55, 70), 1)

    # Metadata Baseline Disclaimer
    y_curr += 16
    cv2.putText(frame, "Baseline Acc: 64.12% (Subject-Group)", (pad_x + 8, y_curr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 200, 255), 1, cv2.LINE_AA)
