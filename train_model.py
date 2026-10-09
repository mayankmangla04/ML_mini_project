"""
train_model.py
==============
Training and Evaluation Script for Driver Fatigue Binary ML Classifier.

Workflow:
1. Load data/processed/driver_image_features.csv
2. Validate columns, data types, and check for missing/non-finite values
3. Check class imbalance & duplicate samples
4. Extract subject/session groups from source_image for leak-free group splitting
5. Split into training and testing sets (Group-based & Stratified splits)
6. Train Random Forest Classifier baseline
7. Compute evaluation metrics: Accuracy, Precision, Recall, F1-score (per class), Confusion Matrix
8. Save model to models/fatigue_binary_model.joblib
9. Save metadata to models/model_metadata.json
10. Verify saved model re-loadability and test inference
"""

import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CSV_PATH = os.path.join("data", "processed", "driver_image_features.csv")
MODEL_SAVE_PATH = os.path.join("models", "fatigue_binary_model.joblib")
METADATA_SAVE_PATH = os.path.join("models", "model_metadata.json")

EXPECTED_FEATURES = ["left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll"]
LABEL_COLUMN = "label"
METADATA_COLUMN = "source_image"
CLASS_NAMES = ["Drowsy", "Not Drowsy"]


def extract_subject_id(source_image_path):
    """
    Extracts subject/prefix identifier from source filename.
    Examples:
        'Drowsy/A0001.png' -> 'A'
        'Non Drowsy/ZB0012.png' -> 'ZB'
        'Drowsy/1001.png' -> 'group_1'
    """
    filename = os.path.basename(str(source_image_path))
    name_without_ext = os.path.splitext(filename)[0]
    
    # Match leading alphabetic characters (e.g., 'A', 'ZB', 'zc', etc.)
    match = re.match(r'^([A-Za-z]+)', name_without_ext)
    if match:
        return match.group(1).upper()
    
    # Match leading numeric prefix or subfolder name
    match_num = re.match(r'^(\d+)', name_without_ext)
    if match_num:
        return f"NUM_{match_num.group(1)[:2]}"
        
    return "UNKNOWN_SUBJECT"


def main():
    print("==================================================")
    print(" DRIVER FATIGUE MODEL TRAINING & EVALUATION ")
    print("==================================================")

    # 1. Load CSV Dataset
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Feature CSV not found at '{CSV_PATH}'. Run extract_dataset_features.py first.")

    print(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total rows from CSV.")
    print("--------------------------------------------------")

    # 2. Validate Feature Columns & Clean Data
    missing_cols = [c for c in EXPECTED_FEATURES + [LABEL_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Drop any row with missing or non-finite values in features or label
    initial_count = len(df)
    df = df.dropna(subset=EXPECTED_FEATURES + [LABEL_COLUMN]).copy()

    for col in EXPECTED_FEATURES:
        df = df[np.isfinite(df[col])].copy()

    cleaned_count = len(df)
    dropped_count = initial_count - cleaned_count
    print(f"Data Cleaning Summary:")
    print(f"  - Initial rows: {initial_count}")
    print(f"  - Cleaned rows: {cleaned_count}")
    print(f"  - Dropped invalid/non-finite rows: {dropped_count}")
    print("--------------------------------------------------")

    # 3. Check Class Balance & Labels
    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    print("Class Distribution:")
    for label_name in CLASS_NAMES:
        cnt = class_counts.get(label_name, 0)
        pct = (cnt / cleaned_count * 100) if cleaned_count > 0 else 0
        print(f"  - {label_name}: {cnt} samples ({pct:.2f}%)")

    present_classes = set(df[LABEL_COLUMN].unique())
    expected_set = set(CLASS_NAMES)
    if present_classes != expected_set:
        raise ValueError(f"Dataset must contain exactly classes {expected_set}, but found {present_classes}")
    print("--------------------------------------------------")

    # 4. Check for Exact Feature Duplicates
    dup_features = df.duplicated(subset=EXPECTED_FEATURES, keep='first')
    num_dups = dup_features.sum()
    print(f"Exact Feature Duplicates Detected: {num_dups} rows")
    if num_dups > 0:
        print(f"Removing {num_dups} exact feature duplicate rows to prevent data leakage...")
        df = df[~dup_features].copy()
        print(f"Remaining dataset size after duplicate removal: {len(df)}")
    print("--------------------------------------------------")

    # 5. Extract Feature Matrix (X), Target Vector (y), and Subject Groups
    X = df[EXPECTED_FEATURES].values
    y = df[LABEL_COLUMN].values

    if METADATA_COLUMN in df.columns:
        subjects = df[METADATA_COLUMN].apply(extract_subject_id).values
    else:
        subjects = np.array(["UNKNOWN"] * len(df))

    unique_subjects = np.unique(subjects)
    print(f"Identified {len(unique_subjects)} unique subject/session groups: {list(unique_subjects)[:15]}...")
    print("--------------------------------------------------")

    # 6. Train / Test Split
    # We execute both a Subject-Group Split (leak-free across drivers) and Stratified Split
    print("Performing Subject-Group Train/Test Split (80% Train / 20% Test)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    subjects_train, subjects_test = subjects[train_idx], subjects[test_idx]

    print(f"Training Samples : {len(X_train)} ({len(np.unique(subjects_train))} subject groups)")
    print(f"Testing Samples  : {len(X_test)} ({len(np.unique(subjects_test))} subject groups)")
    print("Train Class Distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Test Class Distribution :", pd.Series(y_test).value_counts().to_dict())
    print("--------------------------------------------------")

    # 7. Train Baseline Random Forest Classifier
    print("Training Random Forest Classifier baseline (n_estimators=100, random_state=42)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Model training completed successfully.")
    print("--------------------------------------------------")

    # 8. Evaluate Model on Test Set
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_drowsy = precision_score(y_test, y_pred, pos_label="Drowsy", zero_division=0)
    rec_drowsy = recall_score(y_test, y_pred, pos_label="Drowsy", zero_division=0)
    f1_drowsy = f1_score(y_test, y_pred, pos_label="Drowsy", zero_division=0)

    prec_not_drowsy = precision_score(y_test, y_pred, pos_label="Not Drowsy", zero_division=0)
    rec_not_drowsy = recall_score(y_test, y_pred, pos_label="Not Drowsy", zero_division=0)
    f1_not_drowsy = f1_score(y_test, y_pred, pos_label="Not Drowsy", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)

    print("==================================================")
    print(" EVALUATION METRICS (SUBJECT-GROUP TEST SET) ")
    print("==================================================")
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    print("Per-Class Metrics:")
    print(f"  Class 'Drowsy'    : Precision={prec_drowsy:.4f} | Recall={rec_drowsy:.4f} | F1={f1_drowsy:.4f}")
    print(f"  Class 'Not Drowsy': Precision={prec_not_drowsy:.4f} | Recall={rec_not_drowsy:.4f} | F1={f1_not_drowsy:.4f}\n")
    
    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    print(f"Labels Order: {CLASS_NAMES}")
    print(f"[[ TN ({CLASS_NAMES[0]}->{CLASS_NAMES[0]}): {cm[0][0]:>5} | FP ({CLASS_NAMES[0]}->{CLASS_NAMES[1]}): {cm[0][1]:>5} ]")
    print(f" [ FN ({CLASS_NAMES[1]}->{CLASS_NAMES[0]}): {cm[1][0]:>5} | TP ({CLASS_NAMES[1]}->{CLASS_NAMES[1]}): {cm[1][1]:>5} ]]\n")

    print("Detailed Classification Report:")
    print(report_text)

    # Feature Importances
    importances = dict(zip(EXPECTED_FEATURES, clf.feature_importances_))
    print("Feature Importances:")
    for feat_name, imp_val in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat_name:<10}: {imp_val:.4f}")
    print("--------------------------------------------------")

    # Also evaluate Stratified Split for complete comparison report
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf_strat = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    clf_strat.fit(X_tr_s, y_tr_s)
    y_pred_s = clf_strat.predict(X_te_s)
    acc_strat = accuracy_score(y_te_s, y_pred_s)
    print(f"Comparative Reference: Stratified Random Split Accuracy = {acc_strat * 100:.2f}%")
    print("--------------------------------------------------")

    # 9. Save Trained Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Saved trained Random Forest model to: {MODEL_SAVE_PATH}")

    # 10. Save Model Metadata JSON
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 15,
        "features": EXPECTED_FEATURES,
        "class_labels": CLASS_NAMES,
        "dataset_summary": {
            "total_cleaned_samples": len(df),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_class_distribution": pd.Series(y_train).value_counts().to_dict(),
            "test_class_distribution": pd.Series(y_test).value_counts().to_dict(),
            "unique_subjects_count": int(len(unique_subjects))
        },
        "evaluation_metrics": {
            "accuracy": float(acc),
            "stratified_accuracy": float(acc_strat),
            "drowsy_metrics": {
                "precision": float(prec_drowsy),
                "recall": float(rec_drowsy),
                "f1_score": float(f1_drowsy)
            },
            "not_drowsy_metrics": {
                "precision": float(prec_not_drowsy),
                "recall": float(rec_not_drowsy),
                "f1_score": float(f1_not_drowsy)
            },
            "confusion_matrix": cm.tolist()
        },
        "feature_importances": {k: float(v) for k, v in importances.items()},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved model metadata JSON to: {METADATA_SAVE_PATH}")
    print("--------------------------------------------------")

    # 11. Test Model Loading & Sample Prediction Verification
    print("Verifying saved model re-loadability and live feature input schema...")
    loaded_model = joblib.load(MODEL_SAVE_PATH)

    # Test sample prediction with dummy feature vector: [left_ear, right_ear, ear, mar, yaw, pitch, roll]
    test_sample_drowsy = np.array([[0.15, 0.15, 0.15, 0.45, 2.0, 5.0, 1.0]])  # Typical Drowsy EAR ~0.15
    test_sample_alert  = np.array([[0.32, 0.31, 0.315, 0.05, 0.0, 1.0, 0.0]]) # Typical Alert EAR ~0.315

    pred_drowsy = loaded_model.predict(test_sample_drowsy)[0]
    pred_alert  = loaded_model.predict(test_sample_alert)[0]

    probs_drowsy = loaded_model.predict_proba(test_sample_drowsy)[0]
    probs_alert  = loaded_model.predict_proba(test_sample_alert)[0]

    print(f"Sample 1 (EAR=0.15, MAR=0.45) -> Predicted Class: {pred_drowsy} (Probabilities: {probs_drowsy})")
    print(f"Sample 2 (EAR=0.31, MAR=0.05) -> Predicted Class: {pred_alert} (Probabilities: {probs_alert})")
    print("==================================================")
    print(" MODEL TRAINING AND VERIFICATION COMPLETE ")
    print("==================================================")


if __name__ == "__main__":
    main()
