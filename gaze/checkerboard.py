"""The camera's true focal length and lens distortion, from a checkerboard shown to it, once per camera.

    python -m gaze checkerboard

Show OpenCV's checkerboard (9 x 6 inner corners, opencv/doc/pattern.png) on a phone or on paper, strongly tilted every way
(30 to 50 degrees), near and far, in every corner of the image, for about 30 s. A view is kept when it differs from the kept
ones (place or tilt), up to 25; Escape ends.
OpenCV's calibrateCamera (Zhang 2000) on the kept views: one focal for x and y, principal point at the image centre, radial
distortion k1 k2. The squares' size does not matter. Spread: the same fit on 200 random halves of the views.
Written to settings/camera.json (and the kept corners to settings/camera_corners.npz, to fit again without showing the board).
"""

import datetime
import json
import sys
import time

import cv2
import numpy as np

from . import camera as cam
from . import geometry

MAX_VIEWS, MIN_VIEWS = 25, 8
PATTERN = (9, 6)   # inner corners of OpenCV's pattern.png
FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
         | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)


def find(gray, pattern):
    """Inner corners (n x 2, px) of the checkerboard, or None: found on the half-size image, then refined on the full one."""
    found, corners = cv2.findChessboardCornersSB(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                 pattern, flags=cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, (corners * 2.0 + 0.5).astype(np.float32), (4, 4), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
    return corners.reshape(-1, 2)


def descriptor(corners, size):
    """Where the board is and how big it looks (normalized)."""
    c = corners.mean(axis=0) / np.array(size)
    span = (corners.max(axis=0) - corners.min(axis=0)) / np.array(size)
    return np.concatenate([c, span])


def calibrate(views, pattern, size):
    """(focal px, RMS reprojection px, k1 k2) from a list of corner arrays."""
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    w, h = size
    K = np.array([[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
    rms, K, dist, _, _ = cv2.calibrateCamera([board] * len(views), [v.astype(np.float32) for v in views], (w, h), K, np.zeros(5), flags=FLAGS)
    return float(K[0, 0]), float(rms), dist.ravel()[:2]


def spread(views, pattern, size, rounds=200):
    rng = np.random.default_rng(20261027)
    return float(np.std([calibrate([views[i] for i in rng.choice(len(views), len(views) // 2, replace=False)], pattern, size)[0]
                         for _ in range(rounds)]))


def capture_views(pattern):
    from PyQt5 import QtCore, QtGui, QtWidgets
    capture = cam.open_camera(geometry.load_config())
    if not capture.isOpened():
        print("The camera does not open (camera_index in settings/config.json?)", flush=True)
        return None, None
    app = QtWidgets.QApplication(sys.argv[:1])
    label = QtWidgets.QLabel()
    label.setWindowTitle("Camera focal: show the checkerboard, tilted every way. Escape to finish.")
    state = {"views": [], "descriptors": [], "size": None, "last": 0.0}

    def tick():
        ok, frame = capture.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        state["size"] = (w, h)
        corners = find(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
        shown = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(shown, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
            d = descriptor(corners, (w, h))
            new = all(np.abs(d - e).max() > 0.04 for e in state["descriptors"])
            if new and time.monotonic() - state["last"] > 0.4 and len(state["views"]) < MAX_VIEWS:
                state["views"].append(corners)
                state["descriptors"].append(d)
                state["last"] = time.monotonic()
                print(f"view {len(state['views'])}", flush=True)
        lines = [f"checkerboard {'FOUND' if corners is not None else 'not found'} ; views kept {len(state['views'])}/{MAX_VIEWS}",
                 "Tilt it left, right, up, down (30-50 deg); near and far; in the corners of the image",
                 f"Escape when the count reaches {MAX_VIEWS} (or {MIN_VIEWS} at least)"]
        for k, line in enumerate(lines):
            cv2.putText(shown, line, (20, 50 + 50 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(shown, (w // 2, h // 2)), cv2.COLOR_BGR2RGB))
        label.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888)))

    class Keys(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
                app.quit()
            return False

    keys = Keys()
    label.installEventFilter(keys)
    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(1)
    label.show()
    try:
        app.exec_()
    finally:
        capture.release()
        print("Camera released.", flush=True)
    return state["views"], state["size"]


def main():
    pattern = PATTERN
    views, size = capture_views(pattern)
    if views is None:
        return 1
    if len(views) < MIN_VIEWS:
        print(f"Only {len(views)} views: at least {MIN_VIEWS} are needed (15 to 25 advised). Nothing is written.", flush=True)
        return 1
    focal, rms, dist = calibrate(views, pattern, size)
    sd = spread(views, pattern, size)
    fov = 2 * np.degrees(np.arctan(size[0] / 2 / focal))
    print(f"[focal] {focal:.1f} px ± {sd:.1f} (halves of the views) ; reprojection error {rms:.2f} px ; {len(views)} views ; "
          f"horizontal field {fov:.1f}° ; distortion k1 {dist[0]:+.3f} k2 {dist[1]:+.3f}", flush=True)
    cam.CAMERA.parent.mkdir(parents=True, exist_ok=True)
    cam.CAMERA.write_text(json.dumps({"at": datetime.datetime.now().isoformat(timespec="seconds"), "focal_px": focal, "spread_px": sd,
                                      "rms_px": rms, "views": len(views), "pattern": list(pattern), "image": list(size),
                                      "k1_k2": dist.tolist(), "horizontal_field_deg": fov}, indent=1))
    np.savez(cam.CAMERA.with_name("camera_corners.npz"), corners=np.array(views), pattern=np.array(pattern), size=np.array(size))
    print(f"-> {cam.CAMERA}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
