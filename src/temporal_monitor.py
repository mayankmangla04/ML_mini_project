"""
src/temporal_monitor.py
========================
Temporal Monitoring & Aggregation Module for Continuous Driver Safety Monitor.

Responsibility:
- Aggregate frame-level feature observations over a rolling 60-second time window using real timestamps
- Calculate PERCLOS (% of valid window samples with eyes closed)
- Detect discrete Blink events & calculate Blink Frequency (blinks/min)
- Detect discrete Yawning events using MAR duration thresholding
- Measure Head Pose Orientation Deviation duration (Head Away Time in seconds)
- Perform temporal prediction score smoothing (EMA)
- Explicitly handle missing frames, tracking interruptions, and face loss without inventing fake values
"""

import time
from collections import deque
from config.settings import (
    EYE_CLOSED_EAR_THRESHOLD, TEMPORAL_WINDOW_SECONDS,
    YAWN_MAR_THRESHOLD, YAWN_MIN_DURATION_SECONDS, BLINK_MAX_DURATION_SECONDS,
    HEAD_DEVIATION_YAW_THRESHOLD, HEAD_DEVIATION_PITCH_THRESHOLD, HEAD_DEVIATION_ROLL_THRESHOLD,
    FEATURE_SMOOTHING_ALPHA
)


class TemporalMonitor:
    """
    Tracks and aggregates temporal driver behaviour metrics over a rolling time window.
    """
    def __init__(self, window_seconds=TEMPORAL_WINDOW_SECONDS, smoothing_alpha=FEATURE_SMOOTHING_ALPHA):
        self.window_seconds = window_seconds
        self.smoothing_alpha = smoothing_alpha

        # Deque storing observation dicts:
        # {'timestamp', 'ear', 'mar', 'yaw', 'pitch', 'roll', 'is_closed', 'prob_drowsy'}
        self.observations = deque()

        # Discrete event counters (timestamped tuples)
        self.blink_events = deque()  # timestamps of completed blinks
        self.yawn_events = deque()   # timestamps of completed yawns

        # Active event state trackers
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.head_dev_start_time = None

        # Temporal smoothed prediction states
        self.smoothed_prob_drowsy = None
        self.smoothed_prob_not_drowsy = None

    def update(self, features, ml_prediction, current_timestamp=None):
        """
        Updates temporal state with the current frame's features and ML prediction.

        Parameters:
            features (dict): Standard feature vector from FeatureExtractor
            ml_prediction (dict): Output from ModelPredictor.predict()
            current_timestamp (float): Monotonic timestamp in seconds

        Returns dict:
            Aggregated temporal metrics for HUD rendering & warning manager.
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        # Check face detection and measurement availability
        is_face_valid = (
            features is not None and 
            features.get("ear") is not None and 
            ml_prediction.get("is_available", False)
        )

        if not is_face_valid:
            self._handle_missing_face()
            return self.get_metrics(current_timestamp, is_valid=False)

        ear = float(features["ear"])
        mar = float(features["mar"])
        yaw = float(features["yaw"])
        pitch = float(features["pitch"])
        roll = float(features["roll"])

        raw_probs = ml_prediction.get("probabilities", {})
        raw_prob_drowsy = float(raw_probs.get("Drowsy", 0.0))
        raw_prob_not_drowsy = float(raw_probs.get("Not Drowsy", 0.0))

        is_eye_closed = (ear < EYE_CLOSED_EAR_THRESHOLD)

        # 1. Update Eye Closure State & Discrete Blink Event Detection
        if is_eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_timestamp
            current_close_duration = current_timestamp - self.eye_closed_start_time
        else:
            if self.eye_closed_start_time is not None:
                duration = current_timestamp - self.eye_closed_start_time
                # Valid blink: eye closure between 0.08s and 0.50s
                if 0.08 <= duration <= BLINK_MAX_DURATION_SECONDS:
                    self.blink_events.append(current_timestamp)
                self.eye_closed_start_time = None
            current_close_duration = 0.0

        # 2. Update Yawning Event Detection (MAR >= 0.55 sustained for >= 1.2s)
        if mar >= YAWN_MAR_THRESHOLD:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_timestamp
            yawn_dur = current_timestamp - self.yawn_start_time
        else:
            if self.yawn_start_time is not None:
                yawn_dur = current_timestamp - self.yawn_start_time
                if yawn_dur >= YAWN_MIN_DURATION_SECONDS:
                    self.yawn_events.append(current_timestamp)
                self.yawn_start_time = None
            yawn_dur = 0.0

        # 3. Update Head Pose Orientation Deviation Duration (Head Away Time)
        is_head_deviated = (
            abs(yaw) > HEAD_DEVIATION_YAW_THRESHOLD or
            abs(pitch) > HEAD_DEVIATION_PITCH_THRESHOLD or
            abs(roll) > HEAD_DEVIATION_ROLL_THRESHOLD
        )
        if is_head_deviated:
            if self.head_dev_start_time is None:
                self.head_dev_start_time = current_timestamp
            head_dev_duration = current_timestamp - self.head_dev_start_time
        else:
            self.head_dev_start_time = None
            head_dev_duration = 0.0

        # 4. Record current observation into rolling deque
        self.observations.append({
            "timestamp": current_timestamp,
            "ear": ear,
            "mar": mar,
            "is_closed": is_eye_closed,
            "prob_drowsy": raw_prob_drowsy
        })

        # 5. Prune observations and events older than rolling window_seconds
        cutoff_time = current_timestamp - self.window_seconds

        while self.observations and self.observations[0]["timestamp"] < cutoff_time:
            self.observations.popleft()

        while self.blink_events and self.blink_events[0] < cutoff_time:
            self.blink_events.popleft()

        while self.yawn_events and self.yawn_events[0] < cutoff_time:
            self.yawn_events.popleft()

        # 6. Apply EMA Probability Smoothing
        if self.smoothed_prob_drowsy is None:
            self.smoothed_prob_drowsy = raw_prob_drowsy
            self.smoothed_prob_not_drowsy = raw_prob_not_drowsy
        else:
            alpha = self.smoothing_alpha
            self.smoothed_prob_drowsy = alpha * raw_prob_drowsy + (1.0 - alpha) * self.smoothed_prob_drowsy
            self.smoothed_prob_not_drowsy = alpha * raw_prob_not_drowsy + (1.0 - alpha) * self.smoothed_prob_not_drowsy

        return self.get_metrics(current_timestamp, is_valid=True, current_close_dur=current_close_duration, head_dev_dur=head_dev_duration)

    def _handle_missing_face(self):
        """
        Resets active event start timers when driver face is lost or unobserved.
        """
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.head_dev_start_time = None

    def get_metrics(self, current_timestamp=None, is_valid=True, current_close_dur=0.0, head_dev_dur=0.0):
        """
        Calculates and returns aggregated metrics over the rolling window.
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        cutoff_time = current_timestamp - self.window_seconds
        valid_obs = [obs for obs in self.observations if obs["timestamp"] >= cutoff_time]
        total_obs = len(valid_obs)

        # 1. PERCLOS Calculation (% of valid window samples with eyes closed)
        if total_obs > 0:
            closed_count = sum(1 for obs in valid_obs if obs["is_closed"])
            perclos = (closed_count / float(total_obs)) * 100.0
        else:
            perclos = 0.0

        # 2. Blink Count & Frequency (Blinks per Minute)
        recent_blinks = [t for t in self.blink_events if t >= cutoff_time]
        blink_count = len(recent_blinks)

        # Window elapsed time calculation
        if total_obs > 1:
            window_elapsed = max(1.0, current_timestamp - valid_obs[0]["timestamp"])
        else:
            window_elapsed = self.window_seconds

        blink_rate_bpm = (blink_count / window_elapsed) * 60.0 if window_elapsed > 0 else 0.0

        # 3. Yawn Count
        recent_yawns = [t for t in self.yawn_events if t >= cutoff_time]
        yawn_count = len(recent_yawns)

        # 4. Determine Smoother Display Prediction State
        if is_valid and self.smoothed_prob_drowsy is not None:
            if self.smoothed_prob_drowsy > self.smoothed_prob_not_drowsy:
                smoothed_state = "Drowsy"
                smoothed_conf = self.smoothed_prob_drowsy
            else:
                smoothed_state = "Not Drowsy"
                smoothed_conf = self.smoothed_prob_not_drowsy
        else:
            smoothed_state = "N/A"
            smoothed_conf = 0.0

        return {
            "is_valid": is_valid,
            "perclos": round(perclos, 1),
            "blink_count": blink_count,
            "blink_rate_bpm": round(blink_rate_bpm, 1),
            "current_close_duration": round(current_close_dur, 2),
            "yawn_count": yawn_count,
            "head_deviation_duration": round(head_dev_dur, 1),
            "smoothed_state": smoothed_state,
            "smoothed_prob_drowsy": round(self.smoothed_prob_drowsy if self.smoothed_prob_drowsy else 0.0, 1),
            "smoothed_prob_not_drowsy": round(self.smoothed_prob_not_drowsy if self.smoothed_prob_not_drowsy else 0.0, 1),
            "smoothed_confidence": round(smoothed_conf, 1),
            "observation_count": total_obs
        }

    def reset(self):
        """
        Resets all temporal monitoring state.
        """
        self.observations.clear()
        self.blink_events.clear()
        self.yawn_events.clear()
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.head_dev_start_time = None
        self.smoothed_prob_drowsy = None
        self.smoothed_prob_not_drowsy = None
