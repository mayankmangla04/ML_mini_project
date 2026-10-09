"""
main.py
=======
Driver Fatigue & Distraction Safety Monitor
Entry point for Computer Vision & Feature Extraction Pipeline.

Pipeline:
Webcam -> Face Detection -> Facial Landmarks -> Feature Extraction -> Numerical Feature Vector
"""

from src.camera_app import run

if __name__ == "__main__":
    run()
