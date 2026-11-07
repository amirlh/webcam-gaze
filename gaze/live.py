"""The live gaze: camera -> undistortion -> MediaPipe -> UniGaze ViT-H -> the calibration's ridge + the person's adaptation.

    live.run(setup, adaptation, model, show="dot"|"data", csv_path=None)

show="dot": a full-screen window with the gaze point (green dot). Keys: F filtered / raw point (saccade filter), G the crosses,
I the gaze's numbers, Escape quits.
show="data": no window; the gaze's numbers are printed twice a second in the terminal; Ctrl+C stops.
csv_path: every gaze point is also written to this CSV file (see README.md for its columns).

MediaPipe and the network run in two threads: the network always takes the newest face, so the point's rate is the network's
own (about 20 per second with ViT-H on a laptop RTX GPU).
"""

import csv
import sys
import threading
import time
import traceback

import numpy as np

from . import camera as cam
from . import face, geometry, method, protocol
from .saccade_filter import SaccadeFilter

COLUMNS = ["time_s", "delay_ms", "screen_x", "screen_y", "screen_x_filtered", "screen_y_filtered", "screen_x_px", "screen_y_px",
           "screen_x_cm", "screen_y_cm", "eyes_x_cm", "eyes_y_cm", "eyes_z_cm", "gaze_dx", "gaze_dy", "gaze_dz",
           "hit_x_cm", "hit_y_cm", "hit_z_cm"]


def gaze_sample(screen, frozen, offset, landmarks, direction_cam, focal, size, config):
    """Everything about one gaze point (None if the gaze does not meet the screen's plane)."""
    fr = method.features(landmarks[None], direction_cam, focal, size, screen)   # as for the calibration's frames
    E, ray = fr["E"], fr["ray"]
    alpha = method.apply(frozen, method.columns(fr)) + offset
    d = geometry.gaze_direction(alpha, ray)
    P = geometry.intersect_screen(E, d)
    if not np.all(np.isfinite(P)):
        return None
    xy = screen.cm_to_norm(P)[0]
    return {"screen": xy, "screen_px": (xy + 0.5) * np.array([config.get("screen_width_px", np.nan), config.get("screen_height_px", np.nan)]),
            "screen_cm": (xy + 0.5) * np.array([screen.width_cm, screen.height_cm]),
            "eyes_cm": screen.to_camera_frame(E[0]), "gaze": screen.to_camera_frame(d[0]), "hit_cm": screen.to_camera_frame(P[0])}


def describe(s, rate=None, delay=None):
    line = (f"screen ({s['screen'][0]:+.3f}, {s['screen'][1]:+.3f}) = ({s['screen_cm'][0]:.1f}, {s['screen_cm'][1]:.1f}) cm"
            + (f" = ({s['screen_px'][0]:.0f}, {s['screen_px'][1]:.0f}) px" if np.all(np.isfinite(s["screen_px"])) else "")
            + f" | eyes ({s['eyes_cm'][0]:+.1f}, {s['eyes_cm'][1]:+.1f}, {s['eyes_cm'][2]:.1f}) cm"
            + f" | direction ({s['gaze'][0]:+.3f}, {s['gaze'][1]:+.3f}, {s['gaze'][2]:+.3f})"
            + f" | hit ({s['hit_cm'][0]:+.1f}, {s['hit_cm'][1]:+.1f}, {s['hit_cm'][2]:+.1f}) cm")
    if rate is not None:
        line += f" | {rate} points/s" + (f", {delay:.0f} ms" if delay is not None else "")
    return line


def run(setup, adaptation, model, show="dot", csv_path=None):
    frozen, offset = adaptation["frozen"], adaptation["offset"]
    config, intrinsics = setup["config"], setup["intrinsics"]
    screen, focal = geometry.Screen(config), intrinsics["focal_px"]
    straight = cam.straightener(intrinsics)
    detector = face.landmarker()
    capture = cam.open_camera(config)
    if not capture.isOpened():
        print("The camera does not open.", flush=True)
        return 1
    camera = cam.Camera(capture)
    camera.start()
    state = {"lock": threading.Lock(), "running": True, "error": None, "face": None, "number": 0, "sample": None,
             "camera": 0, "points": 0, "delays": [], "filter": SaccadeFilter(), "rows": []}
    t_start = time.monotonic_ns()

    def faces():
        seen = 0
        try:
            while state["running"]:
                frame, seen, t_ns = camera.newest(seen)
                if frame is None:
                    time.sleep(0.001)
                    continue
                frame = straight(frame)
                found = face.detect(detector, frame)
                with state["lock"]:
                    state["camera"] += 1
                    if found is not None and found[1]:
                        state["face"], state["number"] = (frame, found[0], t_ns), state["number"] + 1
        except Exception as error:  # noqa: BLE001
            state["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    def gaze():
        done = 0
        try:
            while state["running"]:
                with state["lock"]:
                    found, number = state["face"], state["number"]
                if found is None or number == done:
                    time.sleep(0.001)
                    continue
                done = number
                frame, landmarks, t_ns = found
                direction = face.gaze_cam(model, [face.as_recorded(frame)], [landmarks], focal)   # as the calibration's JPEGs
                s = gaze_sample(screen, frozen, offset, landmarks, direction, focal, (frame.shape[1], frame.shape[0]), config)
                if s is None:
                    continue
                delay = (time.monotonic_ns() - t_ns) / 1e6
                with state["lock"]:
                    s["filtered"] = state["filter"].step(s["screen"])
                    state["sample"] = s
                    state["points"] += 1
                    state["delays"].append(delay)
                    if csv_path is not None:
                        state["rows"].append([round((t_ns - t_start) / 1e9, 4), round(delay, 1), *s["screen"], *s["filtered"], *s["screen_px"],
                                              *s["screen_cm"], *s["eyes_cm"], *s["gaze"], *s["hit_cm"]])
        except Exception as error:  # noqa: BLE001
            state["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()

    out = open(csv_path, "w", newline="") if csv_path is not None else None
    table = csv.writer(out) if out else None
    if table:
        table.writerow(COLUMNS)

    def flush_rows():
        with state["lock"]:
            rows, state["rows"] = state["rows"], []
        if table:
            table.writerows([[f"{v:.6g}" if isinstance(v, float) else v for v in r] for r in rows])
            out.flush()

    def statistics():
        with state["lock"]:
            cam_rate, points, delays, s = state["camera"], state["points"], state["delays"], state["sample"]
            state["camera"], state["points"], state["delays"] = 0, 0, []
        return cam_rate, points, (float(np.median(delays)) if delays else None), s

    threads = [threading.Thread(target=f, daemon=True, name=f.__name__) for f in (faces, gaze)]
    for t in threads:
        t.start()
    try:
        if show == "data":
            print("The gaze's numbers, twice a second (Ctrl+C stops). Screen: normalized (-0.5 to 0.5), then from its top-left "
                  "corner; eyes, direction and hit: camera frame." + (f" Every point is written to {csv_path}." if csv_path else ""), flush=True)
            last = time.monotonic()
            while not state["error"]:
                time.sleep(0.5)
                flush_rows()
                cam_rate, points, delay, s = statistics()
                elapsed = time.monotonic() - last
                last = time.monotonic()
                if s is not None:
                    print(describe(s, round(points / elapsed), delay), flush=True)
                else:
                    print(f"no gaze yet (camera + MediaPipe {cam_rate / elapsed:.0f} images/s)", flush=True)
            print(f"ERROR: {state['error']}", flush=True)
        else:
            dot_window(state, statistics, flush_rows)
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        camera.running = False
        for t in (*threads, camera):   # nothing may still be reading when the camera and MediaPipe go
            t.join(timeout=2.0)
        flush_rows()
        if out:
            out.close()
        capture.release()
        detector.close()
        print("Camera released." + (f" Gaze points: {csv_path}" if csv_path else ""), flush=True)
    return 0


def dot_window(state, statistics, flush_rows):
    from PyQt5 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication(sys.argv[:1])

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
            self.setGeometry(app.primaryScreen().geometry())
            self.setCursor(QtCore.Qt.BlankCursor)
            self.filtered, self.grid, self.numbers, self.status = True, True, False, ""
            self.redraw = QtCore.QTimer(self)
            self.redraw.timeout.connect(self.update)
            self.redraw.start(16)
            self.stats = QtCore.QTimer(self)
            self.stats.timeout.connect(self.tick)
            self.stats.start(1000)

        def tick(self):
            flush_rows()
            cam_rate, points, delay, _ = statistics()
            self.status = (f"camera + MediaPipe {cam_rate} images/s · gaze {points} points/s · delay image -> point "
                           f"{f'{delay:.0f} ms' if delay is not None else '-'} · {'saccade filter' if self.filtered else 'raw point'}")
            if state["error"]:
                self.status = f"ERROR: {state['error']}"
            print(self.status, flush=True)

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Escape:
                self.close()
            elif event.key() == QtCore.Qt.Key_F:
                self.filtered = not self.filtered
            elif event.key() == QtCore.Qt.Key_G:
                self.grid = not self.grid
            elif event.key() == QtCore.Qt.Key_I:
                self.numbers = not self.numbers

        def paintEvent(self, event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
            w, h = self.width(), self.height()
            if self.grid:
                p.setPen(QtGui.QPen(QtGui.QColor(170, 170, 170), 1.5))
                for x, y in protocol.TARGETS:
                    u, v = w * (x + 0.5), h * (y + 0.5)
                    p.drawLine(QtCore.QPointF(u - 10, v), QtCore.QPointF(u + 10, v))
                    p.drawLine(QtCore.QPointF(u, v - 10), QtCore.QPointF(u, v + 10))
            with state["lock"]:
                s = state["sample"]
            if s is not None:
                x, y = np.clip(s["filtered"] if self.filtered else s["screen"], -0.5, 0.5)
                p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1.5))
                p.setBrush(QtGui.QColor(26, 158, 63))
                p.drawEllipse(QtCore.QPointF(w * (x + 0.5), h * (y + 0.5)), 14, 14)
            p.setPen(QtGui.QColor(60, 60, 60))
            p.setFont(QtGui.QFont("Sans", 14))
            p.drawText(self.rect().adjusted(0, 24, 0, 0), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                       "F: filtered / raw · G: crosses · I: numbers · Escape: quit")
            if self.numbers and s is not None:
                p.setFont(QtGui.QFont("Monospace", 11))
                p.drawText(self.rect().adjusted(40, 60, -40, 0), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
                           describe(s).replace(" | ", "\n"))
            p.setFont(QtGui.QFont("Sans", 12))
            p.setPen(QtGui.QColor(120, 120, 120))
            p.drawText(self.rect().adjusted(0, 0, 0, -24), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, self.status)

        def closeEvent(self, event):
            event.accept()
            app.quit()

    window = Window()
    window.showFullScreen()
    app.exec_()
