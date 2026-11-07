"""The webcam: opening it, reading it in a thread, writing its images in the background, and its true focal and distortion."""

import json
import pathlib
import queue
import sys
import threading
import time

import cv2

from . import face

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMERA = ROOT / "settings" / "camera.json"


def open_camera(config):
    capture = cv2.VideoCapture(int(config.get("camera_index", 0)), cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0)
    if config.get("camera_mjpg", True):   # most USB and laptop cameras give 1080p at 30 fps only in MJPG
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.get("camera_width", 1920)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.get("camera_height", 1080)))
    capture.set(cv2.CAP_PROP_FPS, int(config.get("camera_fps", 30)))
    return capture


def load_intrinsics(path=CAMERA):
    """settings/camera.json, written by gaze/checkerboard.py: {"focal_px", "k1_k2", "image": [w, h], ...}."""
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"STOP: no {path}: measure the camera first with the checkerboard (python -m gaze checkerboard).")
    return json.loads(path.read_text())


def straightener(intrinsics):
    return face.Straightener(intrinsics["focal_px"], intrinsics["k1_k2"], *intrinsics["image"])


class Camera(threading.Thread):
    """Reads the camera; keeps the newest frame and the monotonic time it arrived."""

    def __init__(self, capture):
        super().__init__(daemon=True, name="camera")
        self.capture, self.lock, self.running = capture, threading.Lock(), True
        self.frame, self.number, self.t_ns = None, 0, 0

    def run(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame, self.t_ns = frame, time.monotonic_ns()
                    self.number += 1
            else:
                time.sleep(0.005)

    def newest(self, after):
        with self.lock:
            return (self.frame, self.number, self.t_ns) if self.number > after else (None, after, 0)


class Writer:
    """Writes the JPEG images (quality face.JPEG_QUALITY) in the background, with a few threads (a queue of 128; counts the
    frames it had to drop and the images it could not write)."""

    def __init__(self, threads=3):
        self.jobs, self.lock = queue.Queue(maxsize=128), threading.Lock()
        self.dropped, self.written, self.failed = 0, 0, 0
        self.threads = [threading.Thread(target=self.run, daemon=True, name=f"jpeg{k}") for k in range(threads)]

    def start(self):
        for t in self.threads:
            t.start()

    def put(self, path, frame):
        """False if the queue is full: the frame is dropped."""
        try:
            self.jobs.put_nowait((path, frame))
            return True
        except queue.Full:
            with self.lock:
                self.dropped += 1
            return False

    def run(self):
        while True:
            path, frame = self.jobs.get()
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, face.JPEG_QUALITY])
            with self.lock:
                if ok:
                    self.written += 1
                else:
                    self.failed += 1

    def close(self, timeout=120):
        for _ in self.threads:
            self.jobs.put((None, None))
        for t in self.threads:
            t.join(timeout=timeout)
