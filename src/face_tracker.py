"""
src/face_tracker.py
===================
Temporary Multi-Face Tracking and Independent Per-Face State Manager.

Responsibility:
- Associate detected faces across consecutive frames using centroid distance
- Maintain stable temporary face IDs (Face 1, Face 2, ...)
- Maintain independent EyeFeatureCalculator per face (independent eye-closure timers)
- Select primary driver face based on largest bounding box area
- Handle faces appearing/disappearing gracefully
"""

import time
import numpy as np
from src.eye_features import EyeFeatureCalculator
from config.settings import TRACKING_TIMEOUT_SECONDS


class TrackedFace:
    """
    State representation for a single tracked face.
    """
    def __init__(self, face_id, center, bbox):
        self.face_id = face_id
        self.center = center
        self.bbox = bbox
        self.last_seen = time.perf_counter()
        self.eye_calculator = EyeFeatureCalculator()


class FaceTracker:
    """
    Tracks multiple faces and manages independent per-face state & timing.
    """
    def __init__(self, timeout=TRACKING_TIMEOUT_SECONDS, max_distance=100.0):
        self.timeout = timeout
        self.max_distance = max_distance
        self.next_face_id = 1
        self.tracked_faces = {}  # face_id -> TrackedFace

    def update(self, detected_faces, current_timestamp=None):
        """
        Updates tracking state with faces detected in the current frame.

        Parameters:
            detected_faces (list of dict): List from FaceDetector.detect_faces()
            current_timestamp (float): Current monotonic timestamp

        Returns:
            tracked_results (list of dict): List of face features & tracking metadata.
            primary_driver_id (int or None): Face ID of the primary driver candidate.
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        # 1. Match detected faces with existing tracked faces using centroid distance
        matched_pair_indices = []
        unmatched_detections = list(range(len(detected_faces)))
        existing_ids = list(self.tracked_faces.keys())

        if existing_ids and detected_faces:
            # Build distance matrix between existing face centers and new detection centers
            dist_matrix = np.zeros((len(existing_ids), len(detected_faces)), dtype=np.float64)
            for i, fid in enumerate(existing_ids):
                t_center = np.array(self.tracked_faces[fid].center)
                for j, det in enumerate(detected_faces):
                    d_center = np.array(det["center"])
                    dist_matrix[i, j] = np.linalg.norm(t_center - d_center)

            # Greedy Hungarian-style matching
            for _ in range(min(len(existing_ids), len(detected_faces))):
                min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                min_dist = dist_matrix[min_idx]
                if min_dist > self.max_distance:
                    break

                i, j = min_idx
                fid = existing_ids[i]
                if j in unmatched_detections:
                    matched_pair_indices.append((fid, j))
                    unmatched_detections.remove(j)

                dist_matrix[i, :] = np.inf
                dist_matrix[:, j] = np.inf

        # 2. Update matched faces
        matched_det_indices = set()
        for fid, j in matched_pair_indices:
            matched_det_indices.add(j)
            det = detected_faces[j]
            t_face = self.tracked_faces[fid]
            t_face.center = det["center"]
            t_face.bbox = det["bbox"]
            t_face.last_seen = current_timestamp

        # 3. Create new tracked faces for unmatched detections
        for j in unmatched_detections:
            det = detected_faces[j]
            new_id = self.next_face_id
            self.next_face_id += 1
            t_face = TrackedFace(new_id, det["center"], det["bbox"])
            t_face.last_seen = current_timestamp
            self.tracked_faces[new_id] = t_face

        # 4. Remove stale faces that timed out
        stale_ids = [
            fid for fid, t_face in self.tracked_faces.items()
            if (current_timestamp - t_face.last_seen) > self.timeout
        ]
        for fid in stale_ids:
            del self.tracked_faces[fid]

        # 5. Determine Primary Driver (largest bounding box area among active faces)
        primary_driver_id = None
        max_area = 0

        # Build output results for currently active detections
        results = []
        for det_idx, det in enumerate(detected_faces):
            # Find associated face_id
            assigned_id = None
            for fid, j in matched_pair_indices:
                if j == det_idx:
                    assigned_id = fid
                    break

            if assigned_id is None:
                # Find matching new face ID
                for fid, t_face in self.tracked_faces.items():
                    if t_face.center == det["center"]:
                        assigned_id = fid
                        break

            if assigned_id is None:
                continue

            t_face = self.tracked_faces[assigned_id]
            area = det["bbox"][4]
            if area > max_area:
                max_area = area
                primary_driver_id = assigned_id

            results.append({
                "face_id": assigned_id,
                "bbox": det["bbox"],
                "pixel_landmarks": det["pixel_landmarks"],
                "eye_calculator": t_face.eye_calculator
            })

        return results, primary_driver_id
