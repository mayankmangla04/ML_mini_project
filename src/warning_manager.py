"""
src/warning_manager.py
======================
Optional Driver Warning Manager Module.

Responsibility:
- Evaluate multi-frame temporal evidence (PERCLOS, Smoothed ML Drowsy Prob, Eye Closure Duration)
- Transition between warning states: NORMAL, ADVISORY, WARNING
- Trigger optional audio alert beep with strict cooldown enforcement (no spam per frame)
- Suppress warnings when face observation is invalid or unavailable
- Master toggle to enable/disable audio completely
"""

import time
import sys
import threading
from config.settings import (
    ENABLE_AUDIO_WARNINGS, WARNING_COOLDOWN_SECONDS,
    WARNING_DROWSY_PROB_THRESHOLD, PERCLOS_WARNING_THRESHOLD
)

# Attempt winsound import on Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class WarningManager:
    """
    Manages multi-state driver warnings and optional audio alert sound notifications.
    """
    def __init__(self, enable_audio=ENABLE_AUDIO_WARNINGS, cooldown_seconds=WARNING_COOLDOWN_SECONDS):
        self.enable_audio = enable_audio
        self.cooldown_seconds = cooldown_seconds
        self.last_audio_trigger_time = 0.0

        self.current_state = "NORMAL"  # "NORMAL", "ADVISORY", "WARNING"
        self.reason = "Driver Attentive"

    def update(self, temporal_metrics, ml_prediction, current_timestamp=None):
        """
        Evaluates temporal evidence and updates warning state.

        Parameters:
            temporal_metrics (dict): Metrics from TemporalMonitor
            ml_prediction (dict): Prediction result from ModelPredictor
            current_timestamp (float): Current timestamp

        Returns dict:
            'state': 'NORMAL' / 'ADVISORY' / 'WARNING'
            'reason': Description string
            'audio_played': bool
        """
        if current_timestamp is None:
            current_timestamp = time.perf_counter()

        is_valid = temporal_metrics.get("is_valid", False)
        if not is_valid:
            self.current_state = "UNAVAILABLE"
            self.reason = "Observation Unavailable"
            return {
                "state": self.current_state,
                "reason": self.reason,
                "audio_played": False
            }

        smoothed_prob_drowsy = temporal_metrics.get("smoothed_prob_drowsy", 0.0)
        perclos = temporal_metrics.get("perclos", 0.0)
        eye_close_dur = temporal_metrics.get("current_close_duration", 0.0)

        audio_played = False

        # Evaluate Warning Thresholds
        if eye_close_dur >= 2.0 or smoothed_prob_drowsy >= WARNING_DROWSY_PROB_THRESHOLD or perclos >= PERCLOS_WARNING_THRESHOLD:
            self.current_state = "WARNING"
            if eye_close_dur >= 2.0:
                self.reason = f"Sustained Eye Closure ({eye_close_dur:.1f}s)"
            elif smoothed_prob_drowsy >= WARNING_DROWSY_PROB_THRESHOLD:
                self.reason = f"High Drowsiness Score ({smoothed_prob_drowsy:.1f}%)"
            else:
                self.reason = f"High PERCLOS ({perclos:.1f}%)"

            # Check audio trigger cooldown
            if self.enable_audio and (current_timestamp - self.last_audio_trigger_time) >= self.cooldown_seconds:
                self._play_alert_sound_async(frequency=1200, duration=350)
                self.last_audio_trigger_time = current_timestamp
                audio_played = True

        elif smoothed_prob_drowsy >= 45.0 or perclos >= 20.0 or eye_close_dur >= 1.0:
            self.current_state = "ADVISORY"
            self.reason = "Early Signs of Drowsiness"
        else:
            self.current_state = "NORMAL"
            self.reason = "Driver Attentive"

        return {
            "state": self.current_state,
            "reason": self.reason,
            "audio_played": audio_played
        }

    def _play_alert_sound_async(self, frequency=1200, duration=350):
        """
        Plays audio alert beep asynchronously in a background thread so the webcam loop never lags.
        """
        if not HAS_WINSOUND:
            return

        def sound_thread():
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass

        t = threading.Thread(target=sound_thread, daemon=True)
        t.start()

    def set_audio_enabled(self, enabled):
        self.enable_audio = bool(enabled)
