"""
src/camera.py
=============
Webcam Stream Management Module.

Responsibility:
- Safely open webcam using DirectShow (CAP_DSHOW) or default backend
- Capture video frames continuously
- Handle camera errors and clean resource release
"""

import cv2
import numpy as np
from config.settings import CAMERA_INDICES, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """
    Manages OpenCV VideoCapture instance.
    """
    def __init__(self, camera_indices=CAMERA_INDICES):
        self.cap = None
        self.camera_index = None
        self.backend = None
        self._initialize_camera(camera_indices)

    def _initialize_camera(self, camera_indices):
        # Try DirectShow first for Windows fast startup
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and np.mean(frame) > 1.0:
                    self.cap = cap
                    self.camera_index = idx
                    self.backend = "CAP_DSHOW"
                    break
                cap.release()

        # Fallback to default backend
        if self.cap is None:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cap = cap
                self.camera_index = 0
                self.backend = "DEFAULT"

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """
        Reads a frame from the webcam stream.
        Returns:
            ret (bool): Success flag.
            frame (ndarray): BGR image array.
        """
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self):
        """
        Releases the webcam hardware resource cleanly.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
