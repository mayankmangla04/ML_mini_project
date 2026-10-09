"""
extract_dataset_features.py
============================
Offline Feature Extraction Script for Driver Drowsiness ML Pipeline.

Workflow:
- Recursively scan dataset class folders: 'Drowsy' and 'Non Drowsy' (or 'non drowsy')
- Map class labels:
    - Original 'Drowsy'     -> CSV label 'Drowsy'
    - Original 'Non Drowsy' -> CSV label 'Not Drowsy'
- Detect face and 468/478 facial landmarks using MediaPipe FaceLandmarker
- Extract primary face (largest bounding box area)
- Extract numerical features using exact live application formulas:
    - left_ear
    - right_ear
    - ear
    - mar
    - yaw
    - pitch
    - roll
- Validate numerical values (finite, plausible bounds)
- Export to data/processed/driver_image_features.csv
"""

import os
import sys
import glob
import math
import argparse
import time
import cv2
import numpy as np
import pandas as pd

from src.face_detector import FaceDetector
from src.landmark_detector import extract_eye_landmarks
from src.eye_features import calculate_single_ear
from src.mouth_features import compute_mar
from src.head_pose import estimate_head_pose
from config.settings import MODEL_PATH


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract facial features from offline driver image dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Driver Drowsiness Dataset (DDD)",
        help="Path to dataset root folder containing class subdirectories."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join("data", "processed", "driver_image_features.csv"),
        help="Path where output feature CSV will be saved."
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Maximum images to process per class (default: None, processes all images)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output CSV if it already exists."
    )
    return parser.parse_args()


def find_class_folders(dataset_dir):
    """
    Locates subfolders for 'Drowsy' and 'Non Drowsy' classes.
    Returns dict: {'Drowsy': path, 'Not Drowsy': path}
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    class_paths = {}
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for d in subdirs:
        d_lower = d.lower().strip()
        full_path = os.path.join(dataset_dir, d)

        if d_lower in ["drowsy"]:
            class_paths["Drowsy"] = full_path
        elif d_lower in ["non drowsy", "nondrowsy", "non_drowsy", "not drowsy", "not_drowsy"]:
            class_paths["Not Drowsy"] = full_path

    if "Drowsy" not in class_paths or "Not Drowsy" not in class_paths:
        raise ValueError(
            f"Could not find both class folders in '{dataset_dir}'. "
            f"Found subdirectories: {subdirs}. Expected 'Drowsy' and 'Non Drowsy'."
        )

    return class_paths


def get_image_files(folder_path, max_images=None):
    """
    Recursively scans folder_path for image files matching SUPPORTED_EXTENSIONS.
    If max_images is specified, samples deterministically.
    """
    all_files = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(os.path.join(root, f))

    all_files = sorted(all_files)
    total_count = len(all_files)

    if max_images is not None and max_images > 0 and max_images < total_count:
        # Deterministic sampling across the dataset
        indices = np.linspace(0, total_count - 1, max_images, dtype=int)
        sampled_files = [all_files[i] for i in indices]
        return sampled_files, total_count

    return all_files, total_count


def validate_feature_vector(left_ear, right_ear, ear, mar, yaw, pitch, roll):
    """
    Validates that extracted numerical features are finite numbers and plausible.
    Returns (is_valid, reason)
    """
    features = [left_ear, right_ear, ear, mar, yaw, pitch, roll]
    feature_names = ["left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll"]

    for name, val in zip(feature_names, features):
        if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return False, f"Non-finite {name}: {val}"

    # Plausible range checks
    if not (0.0 <= left_ear <= 1.0):
        return False, f"left_ear out of range [0, 1]: {left_ear}"
    if not (0.0 <= right_ear <= 1.0):
        return False, f"right_ear out of range [0, 1]: {right_ear}"
    if not (0.0 <= ear <= 1.0):
        return False, f"ear out of range [0, 1]: {ear}"
    if not (0.0 <= mar <= 2.5):
        return False, f"mar out of range [0, 2.5]: {mar}"
    if not (-180.0 <= yaw <= 180.0):
        return False, f"yaw out of range [-180, 180]: {yaw}"
    if not (-180.0 <= pitch <= 180.0):
        return False, f"pitch out of range [-180, 180]: {pitch}"
    if not (-180.0 <= roll <= 180.0):
        return False, f"roll out of range [-180, 180]: {roll}"

    return True, "valid"


def extract_features_from_image(img, face_detector):
    """
    Extracts raw numerical features from a single BGR image array.

    Returns:
        feature_dict (dict or None), status_reason (str)
    """
    if img is None or img.size == 0:
        return None, "unreadable_or_empty_image"

    img_h, img_w, _ = img.shape
    if img_h == 0 or img_w == 0:
        return None, "zero_dimension_image"

    detected_faces = face_detector.detect_faces(img)
    if not detected_faces:
        return None, "no_face_detected"

    # Select primary face: largest bounding box area
    primary_face = max(detected_faces, key=lambda f: f["bbox"][4])
    pixel_landmarks = primary_face.get("pixel_landmarks")

    if not pixel_landmarks or len(pixel_landmarks) < 468:
        return None, "insufficient_landmarks"

    # 1. Compute Eye Features (EAR)
    left_pts, right_pts = extract_eye_landmarks(pixel_landmarks)
    if len(left_pts) < 6 or len(right_pts) < 6:
        return None, "missing_eye_landmarks"

    left_ear = calculate_single_ear(left_pts)
    right_ear = calculate_single_ear(right_pts)
    ear = (left_ear + right_ear) / 2.0

    # 2. Compute Mouth Features (MAR)
    mar = compute_mar(pixel_landmarks)

    # 3. Compute 3D Head Pose Angles (Yaw, Pitch, Roll in degrees)
    yaw, pitch, roll = estimate_head_pose(pixel_landmarks, img_w, img_h)

    # Validate numerical features
    is_valid, reason = validate_feature_vector(left_ear, right_ear, ear, mar, yaw, pitch, roll)
    if not is_valid:
        return None, f"invalid_numerical_{reason}"

    features = {
        "left_ear": round(float(left_ear), 6),
        "right_ear": round(float(right_ear), 6),
        "ear": round(float(ear), 6),
        "mar": round(float(mar), 6),
        "yaw": round(float(yaw), 6),
        "pitch": round(float(pitch), 6),
        "roll": round(float(roll), 6)
    }

    return features, "success"


def main():
    args = parse_args()

    print("==================================================")
    print(" OFFLINE FEATURE EXTRACTION PIPELINE ")
    print("==================================================")
    print(f"Dataset Directory : {args.dataset_dir}")
    print(f"Output CSV Path   : {args.output_csv}")
    print(f"Max Per Class     : {args.max_images_per_class if args.max_images_per_class else 'All available'}")
    print("--------------------------------------------------")

    # 1. Locate Class Folders
    class_paths = find_class_folders(args.dataset_dir)
    print(f"Found 'Drowsy' folder    : {class_paths['Drowsy']}")
    print(f"Found 'Not Drowsy' folder: {class_paths['Not Drowsy']}")

    # 2. Collect Image Files
    image_queue = []
    class_total_counts = {}
    class_sampled_counts = {}

    for csv_label, folder_path in class_paths.items():
        files, total_count = get_image_files(folder_path, args.max_images_per_class)
        class_total_counts[csv_label] = total_count
        class_sampled_counts[csv_label] = len(files)
        for fpath in files:
            image_queue.append((fpath, csv_label))

    print("Image Discovery Summary:")
    for label in ["Drowsy", "Not Drowsy"]:
        total = class_total_counts[label]
        sampled = class_sampled_counts[label]
        if sampled < total:
            print(f"  Class '{label}': {total} images on disk | Limit applied: sampled {sampled} images")
        else:
            print(f"  Class '{label}': {total} images on disk | Processing all {sampled} images")
    print(f"Total Images to Process: {len(image_queue)}")
    print("--------------------------------------------------")

    # 3. Initialize Face Detector
    print("Initializing MediaPipe Face Landmarker...")
    face_detector = FaceDetector(model_path=MODEL_PATH)
    print("Face Landmarker Initialized OK.")
    print("--------------------------------------------------")

    # 4. Process Images & Extract Features
    records = []
    skipped_stats = {}
    success_stats = {"Drowsy": 0, "Not Drowsy": 0}

    start_time = time.time()
    total_images = len(image_queue)

    print("Extracting facial features...")
    for idx, (fpath, label) in enumerate(image_queue, 1):
        img = cv2.imread(fpath)
        features, status = extract_features_from_image(img, face_detector)

        if features is not None:
            rel_path = os.path.relpath(fpath, start=args.dataset_dir).replace("\\", "/")
            record = {
                "left_ear": features["left_ear"],
                "right_ear": features["right_ear"],
                "ear": features["ear"],
                "mar": features["mar"],
                "yaw": features["yaw"],
                "pitch": features["pitch"],
                "roll": features["roll"],
                "label": label,
                "source_image": rel_path
            }
            records.append(record)
            success_stats[label] += 1
        else:
            skipped_stats[status] = skipped_stats.get(status, 0) + 1

        # Progress reporting
        if idx % 500 == 0 or idx == total_images:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta_seconds = (total_images - idx) / rate if rate > 0 else 0
            print(f"Progress: [{idx}/{total_images}] ({idx/total_images*100:.1f}%) | "
                  f"Speed: {rate:.1f} img/s | Elapsed: {elapsed:.1f}s | ETA: {eta_seconds:.1f}s", flush=True)

    elapsed_total = time.time() - start_time

    # 5. Export to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(records)

    # Column ordering requirement:
    # left_ear, right_ear, ear, mar, yaw, pitch, roll, label, source_image
    columns = ["left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll", "label", "source_image"]
    df = df[columns]

    df.to_csv(args.output_csv, index=False)
    print("==================================================")
    print(" FEATURE EXTRACTION COMPLETE ")
    print("==================================================")
    print(f"Output CSV Path             : {os.path.abspath(args.output_csv)}")
    print(f"Total Images Processed      : {total_images}")
    print(f"Total Successful Extractions: {len(records)}")
    print(f"Total Skipped Images        : {total_images - len(records)}")
    print(f"Processing Time             : {elapsed_total:.2f} seconds ({total_images/elapsed_total:.1f} img/s)")
    print("--------------------------------------------------")
    print("Successful Class Counts:")
    for label, count in success_stats.items():
        print(f"  - {label}: {count} samples")
    print("--------------------------------------------------")
    print("Skipped Breakdown:")
    if skipped_stats:
        for reason, count in sorted(skipped_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count} images")
    else:
        print("  - None! 100% of images were processed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
