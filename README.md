# Driver Fatigue & Distraction Safety Monitor
## A Continuous Computer Vision & Machine Learning Driver Monitoring Prototype

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Project Overview

The **Driver Fatigue & Distraction Safety Monitor** is a real-time computer vision and machine learning application designed as a college mini-project prototype. 

The system captures a live webcam video feed, detects the driver's face and facial landmarks using **MediaPipe Tasks API**, extracts numerical facial measurements (Eye Aspect Ratio, Mouth Aspect Ratio, 3D Head Pose Angles), classifies driver state using a trained **Random Forest Machine Learning model**, aggregates temporal metrics (PERCLOS, Blinks, Yawns, Head Deviation) over a rolling 60-second window, and presents the data on an ultra-compact dual-panel HUD overlay.

---

## System Pipeline Architecture

```text
WEBCAM FRAME (Camera Feed)
  │
  ▼
FACE DETECTION & MULTI-FACE TRACKING (MediaPipe FaceLandmarker + Centroid Tracker)
  │
  ▼
INSTANTANEOUS FEATURE EXTRACTION (FeatureExtractor)
  ├── Left EAR, Right EAR, Avg EAR  (Eye Aspect Ratio)
  ├── MAR                           (Mouth Aspect Ratio)
  └── Yaw, Pitch, Roll              (3D Head Pose Angles via OpenCV solvePnP)
  │
  ▼
RANDOM FOREST ML CLASSIFIER (ModelPredictor)
  ├── Raw Frame Binary Score: Drowsy / Not Drowsy
  └── Probabilities: P(Drowsy), P(Not Drowsy)
  │
  ▼
TEMPORAL AGGREGATION & MONITORING (TemporalMonitor - 60s Rolling Window)
  ├── PERCLOS %                     (% of valid window samples with eyes closed)
  ├── Blink Counter & Rate          (Discrete blinks & Blinks/min)
  ├── Yawn Event Counter            (MAR ≥ 0.55 sustained for ≥ 1.2s)
  ├── Head Orientation Deviation    (Continuous head-away time in seconds)
  └── Prediction Smoothing          (Exponential Moving Average score)
  │
  ▼
DRIVER WARNING MANAGER (WarningManager)
  ├── Multi-State Evaluation: NORMAL / ADVISORY / WARNING
  └── Non-blocking Asynchronous Audio Beep Alert (with 4.0s Cooldown)
  │
  ▼
COMPACT DUAL-PANEL HUD DISPLAY (Visualizer)
  ├── Left Panel : Instantaneous Feature Vector
  └── Right Panel: Continuous Monitoring, ML Prediction, & Warning Status
```

---

## Features Implemented

- **Multi-Face Detection & Tracking**: Detects multiple faces in frame and automatically identifies the primary driver based on bounding box area.
- **Subtle Annotations**: Draws thin, subtle corner brackets around detected faces without drawing facial landmark lines across the driver's face.
- **Instantaneous Feature Vector**: Calculates `left_ear`, `right_ear`, `ear`, `mar`, `yaw`, `pitch`, and `roll` for every frame.
- **Trained Machine Learning Model**: Bundles a pre-trained Random Forest Classifier (`models/fatigue_binary_model.joblib`) for binary classification (`Drowsy` vs `Not Drowsy`).
- **Rolling-Window Temporal Monitoring**: Aggregates PERCLOS %, blink counts & rates, yawning events, and head pose deviation over a rolling 60-second window.
- **Temporal Prediction Smoothing**: Smooths frame-by-frame model outputs while preserving access to raw predictions.
- **Multi-State Warning & Audio Alerts**: Transitions between `NORMAL`, `ADVISORY`, and `WARNING` states with non-blocking audio alerts and a 4.0-second cooldown period.
- **Ultra-Compact Non-Intrusive HUD**: Side HUD panels leave the center webcam feed 100% open and unobstructed for the driver.

---

## Technologies Used

- **Language**: Python 3.10+ / 3.14
- **Computer Vision**: OpenCV (`opencv-python`), MediaPipe (`mediapipe`)
- **Machine Learning**: Scikit-Learn (`scikit-learn`), Joblib (`joblib`)
- **Data & Math**: NumPy (`numpy`), Pandas (`pandas`)

---

## Repository Directory Structure

```text
Driver_Fatigue_Project/
│
├── main.py                      # Main entrypoint launcher
├── webcam_test.py               # Standalone webcam application launcher
├── extract_dataset_features.py  # Offline feature extraction pipeline script
├── train_model.py               # ML Model training & evaluation script
│
├── config/
│   └── settings.py              # Centralized configuration parameters
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── camera.py                # OpenCV camera wrapper with auto-backend selection
│   ├── camera_app.py            # Main real-time application pipeline loop
│   ├── eye_features.py          # EAR calculation & eye closure tracker
│   ├── face_detector.py         # MediaPipe FaceLandmarker detection wrapper
│   ├── face_tracker.py          # Multi-face centroid tracking & primary driver selection
│   ├── feature_extractor.py     # Central feature vector extractor
│   ├── head_pose.py             # 3D Head Pose estimation via OpenCV solvePnP
│   ├── landmark_detector.py     # Landmark index mapping & region extraction
│   ├── model_predictor.py       # ML Model inference & probability mapper
│   ├── mouth_features.py        # MAR calculation module
│   ├── temporal_monitor.py      # Rolling-window PERCLOS, blinks, yawns & head pose dev
│   ├── warning_manager.py       # Warning state manager & audio alert sound player
│   └── visualizer.py            # Compact dual-panel HUD & face annotation renderer
│
├── models/
│   ├── fatigue_binary_model.joblib # Trained Random Forest ML Classifier (6.98 MB)
│   ├── model_metadata.json         # Model training metadata & feature schema
│   └── face_landmarker.task        # MediaPipe FaceLandmarker task model asset (3.58 MB)
│
├── data/
│   └── processed/               # Directory for generated feature CSVs
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions (excludes venv, raw datasets, caches)
└── README.md                    # Project documentation
```

---

## Installation & Setup (Windows)

### Step 1: Clone the Repository
```powershell
git clone https://github.com/mayankmangla04/ML_mini_project.git
cd ML_mini_project
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\activate
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running the Application

Ensure your webcam is connected, then launch the application with either command:

```powershell
python webcam_test.py
```
or
```powershell
python main.py
```

### Keyboard Controls:
- Press **`Q`** on the video preview window or click the window close **`X`** button to exit cleanly.

---

## Model Training & Performance Details

### Trained Model Specifications:
- **Model File**: [`models/fatigue_binary_model.joblib`](file:///c:/ML_mini_project/Driver_Fatigue_Project/models/fatigue_binary_model.joblib)
- **Metadata File**: [`models/model_metadata.json`](file:///c:/ML_mini_project/Driver_Fatigue_Project/models/model_metadata.json)
- **Classifier Type**: `RandomForestClassifier` (100 estimators, max depth 15)
- **Input Feature Vector Schema (7 Features)**:
  ```python
  ["left_ear", "right_ear", "ear", "mar", "yaw", "pitch", "roll"]
  ```

### Evaluation Results:
- **Subject-Group Test Accuracy**: **64.12%**
  > *Evaluated on 2,550 test samples from 6 completely unseen driver subject groups (GroupShuffleSplit) to test true driver generalizability.*
- **Stratified Split Reference Accuracy**: **89.54%**

---

## Re-Extracting Features & Retraining the Model

If you wish to re-extract features from an offline dataset or retrain the ML model:

1. **Obtain the Driver Drowsiness Dataset (DDD)** and place the dataset folder in the project root:
   ```text
   Driver Drowsiness Dataset (DDD)/
   ├── Drowsy/
   └── Non Drowsy/
   ```

2. **Run Offline Feature Extraction**:
   ```powershell
   python extract_dataset_features.py --max-images-per-class 5000
   ```
   *(Generates `data/processed/driver_image_features.csv`)*

3. **Train & Save Model**:
   ```powershell
   python train_model.py
   ```
   *(Saves updated model to `models/fatigue_binary_model.joblib` and metadata to `models/model_metadata.json`)*

---

## Troubleshooting & Common Issues

- **Camera Index Error (`Could not open webcam`)**:
  - Open `config/settings.py` and adjust `CAMERA_INDICES = [0, 1, 2]` to match your camera device index.
- **Missing Dependencies**:
  - Run `pip install -r requirements.txt` inside your activated virtual environment.
- **MediaPipe Model Download Failure**:
  - The MediaPipe landmarker asset is pre-bundled at `models/face_landmarker.task`. If missing, `FaceDetector` will automatically attempt to download it from Google storage.

---

## System Limitations & Disclaimer

- **College Prototype Status**: This application is an educational college mini-project prototype designed for demonstration and research purposes.
- **Subject-Group Accuracy Baseline**: The baseline Random Forest classifier achieved 64.12% test accuracy on unseen driver subjects.
- **Not for Real-World Driving**: **This prototype is NOT certified, validated, or safe for real-world automated driver safety monitoring, commercial vehicle use, or autonomous driving control.**
