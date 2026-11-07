"""Webcam gaze tracking with UniGaze ViT-H, MediaPipe and a per-screen calibration. Run it with `python -m gaze` (see README.md)."""

import os

if os.environ.get("XDG_SESSION_TYPE") == "wayland":   # Qt's windows on Wayland, for every entry point (gaze, gaze.record, ...)
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
