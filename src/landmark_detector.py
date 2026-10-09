"""
src/landmark_detector.py
========================
Facial Landmark Index Definitions and Subregion Extraction.

Responsibility:
- Maintain consistent MediaPipe 468/478 landmark index maps
- Extract specific region points (Eyes, Mouth, Nose, Pose anchors)
"""

# MediaPipe 468/478 Face Mesh Landmark Indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

LIPS_INNER_TOP = [82, 13, 312]
LIPS_INNER_BOTTOM = [87, 14, 317]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

# Landmark indices needed for solvePnP head pose estimation
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]


def extract_eye_landmarks(pixel_landmarks):
    """
    Extracts Left Eye and Right Eye 6-point coordinate lists.
    """
    if not pixel_landmarks:
        return [], []

    left_pts = [pixel_landmarks[i] for i in LEFT_EYE_INDICES if i < len(pixel_landmarks)]
    right_pts = [pixel_landmarks[i] for i in RIGHT_EYE_INDICES if i < len(pixel_landmarks)]
    return left_pts, right_pts


def extract_mouth_landmarks(pixel_landmarks):
    """
    Extracts inner top/bottom lip points and corners for MAR calculation.
    """
    needed = LIPS_INNER_TOP + LIPS_INNER_BOTTOM + [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT]
    if not pixel_landmarks or max(needed) >= len(pixel_landmarks):
        return None

    return {
        "corner_left": pixel_landmarks[MOUTH_CORNER_LEFT],
        "corner_right": pixel_landmarks[MOUTH_CORNER_RIGHT],
        "top_pts": [pixel_landmarks[i] for i in LIPS_INNER_TOP],
        "bottom_pts": [pixel_landmarks[i] for i in LIPS_INNER_BOTTOM]
    }
