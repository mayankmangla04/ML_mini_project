"""
src/head_pose.py
================
3D Head Pose Orientation Estimation Module using OpenCV solvePnP.

Responsibility:
- Calculate numerical 3D head orientation angles: Yaw, Pitch, Roll (in degrees)
- Agnostic to driver status or classification
"""

import cv2
import numpy as np

# Generic 3D Facial Model Reference Points (in mm)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
], dtype=np.float64)

HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(pixel_landmarks, frame_w, frame_h):
    """
    Calculates numerical 3D head pose orientation angles (Yaw, Pitch, Roll) in degrees.

    Returns:
        yaw (float): Horizontal turn in degrees.
        pitch (float): Vertical tilt in degrees.
        roll (float): Side-to-side tilt in degrees.
    """
    if not pixel_landmarks or max(HEAD_POSE_INDICES) >= len(pixel_landmarks):
        return 0.0, 0.0, 0.0

    image_points = np.array([
        pixel_landmarks[1],    # Nose tip
        pixel_landmarks[152],  # Chin
        pixel_landmarks[33],   # Left eye outer corner
        pixel_landmarks[263],  # Right eye outer corner
        pixel_landmarks[61],   # Left mouth corner
        pixel_landmarks[291]   # Right mouth corner
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    # Zero-center angles when looking straight ahead
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

    return float(yaw), float(pitch), float(roll)
