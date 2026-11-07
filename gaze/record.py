"""Records a calibration (gaze/protocol.py): full-screen dots, the camera's images and MediaPipe's landmarks.

    python -m gaze.record --mode calibration|short --folder DIR

Called by `python -m gaze calibrate` and `python -m gaze session`. Each sub-block is announced on the screen: Space (or a
click) starts it, R redoes the previous one, Escape stops. Between sub-blocks only, the camera's image is shown in the top
right corner (mirrored, with the iris centres MediaPipe finds); it is hidden while the dots run.
Every image is undistorted (settings/camera.json) before MediaPipe and before it is written. Written in DIR:
<name>_take<k>.npz per sub-block (frame_id, t_s, state: 1 face and eyes open, 2 eyes closed, 0 no face; landmarks 478 x 3;
dot_index, -1 between dots; dot_xy; since_onset_s; info: JSON), images/<name>_take<k>/<frame>.jpg (JPEG quality 100, about
1 MB each) and takes.jsonl. A frame's dot is the one on the screen when the camera delivered it.
"""

import argparse
import datetime
import json
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

from . import camera as cam
from . import face, geometry, protocol

DOT_RADIUS, CENTRE_RADIUS = 20, 3   # pixels
PREVIEW = (480, 270)   # pixels, in the top right corner


def preview_of(frame, found):
    """The small mirrored RGB image with the iris centres; found: face.detect's result."""
    small = cv2.flip(cv2.resize(frame, PREVIEW, interpolation=cv2.INTER_AREA), 1)
    if found is not None:
        for i in (468, 473):
            x, y = found[0][i, :2]
            cv2.circle(small, (int(PREVIEW[0] * (1 - x)), int(PREVIEW[1] * y)), 4, (0, 255, 0) if found[1] else (0, 0, 255), -1)
    return {"rgb": np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), "face": found is not None,
            "open": bool(found[1]) if found is not None else False}


class Take:
    def __init__(self, name, attempt, folder):
        self.name, self.attempt = name, attempt
        self.points, self.nose, self.text = protocol.SUBBLOCKS[name]
        self.label = f"{name}_take{attempt}"
        self.image_dir = folder / "images" / self.label
        self.rows, self.onsets_ns, self.started_ns, self.interrupted = [], [], time.monotonic_ns(), False

    def save(self, folder, size, extra):
        t = np.array([r["t_ns"] for r in self.rows], dtype=np.int64)
        index = np.full(len(t), -1)
        since = np.full(len(t), np.nan)
        onsets = np.array(self.onsets_ns + [self.end_ns], dtype=np.int64)
        for k in range(len(self.onsets_ns)):   # the dots shown (fewer if the take was interrupted)
            m = (t >= onsets[k]) & (t < onsets[k + 1])
            index[m], since[m] = k, (t[m] - onsets[k]) / 1e9
        xy = np.array([self.points[k] if k >= 0 else (np.nan, np.nan) for k in index], dtype=float).reshape(-1, 2)
        n = len(t)
        info = {"name": self.name, "attempt": self.attempt, "points": self.points, "nose_mark": self.nose,
                "image_size": list(size), "interrupted": self.interrupted, **extra}
        np.savez(folder / f"{self.label}.npz", info=json.dumps(info), frame_id=np.array([r["frame"] for r in self.rows], dtype=np.int64),
                 t_s=(t - self.started_ns) / 1e9, state=np.array([r["state"] for r in self.rows], dtype=np.int8),
                 landmarks=np.array([r["landmarks"] for r in self.rows], dtype=np.float32).reshape(n, 478, 3),
                 dot_index=index, dot_xy=xy, since_onset_s=since)
        valid = int(np.sum((np.array([r["state"] for r in self.rows]) == 1) & (index >= 0))) if n else 0
        return n, valid


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=list(protocol.MODES), required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    config, intrinsics = geometry.load_config(), cam.load_intrinsics()
    names = protocol.MODES[args.mode]
    folder = pathlib.Path(args.folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Folder: {folder}", flush=True)

    from PyQt5 import QtCore, QtGui, QtWidgets
    detector = face.landmarker()
    straight = cam.straightener(intrinsics)
    capture = cam.open_camera(config)
    if not capture.isOpened():
        print("The camera does not open (another program uses it? camera_index in settings/config.json?)", flush=True)
        return 1
    camera, writer = cam.Camera(capture), cam.Writer()
    camera.start()
    writer.start()
    app = QtWidgets.QApplication(sys.argv[:1])
    state = {"take": None, "index": 0, "done": [], "log": [], "stop": False, "size": None, "fps": 0, "preview": None, "error": None}
    lock = threading.Lock()   # a frame is added to a take only while that take is still running

    def attempt_of(name):
        k = 1
        while (folder / f"{name}_take{k}.npz").exists():
            k += 1
        return k

    def processing():
        """Camera -> undistortion -> MediaPipe, for the frames that arrive while a sub-block runs; the preview between them."""
        seen, count, t0 = 0, 0, time.monotonic()
        while not state["stop"]:
            frame, seen, t_ns = camera.newest(seen)
            if frame is None:
                time.sleep(0.002)
                continue
            size = (frame.shape[1], frame.shape[0])
            if list(size) != list(intrinsics["image"]):
                state["error"] = f"the camera gives {size}, settings/camera.json was measured at {intrinsics['image']}"
                continue
            state["size"] = size
            count += 1
            if time.monotonic() - t0 >= 1.0:
                state["fps"], count, t0 = count, 0, time.monotonic()
            take = state["take"]
            if take is None:
                if count % 3 == 0:   # the preview, about 10 times a second
                    state["preview"] = preview_of(frame, face.detect(detector, frame))
                continue
            frame = straight(frame)
            found = face.detect(detector, frame)
            row = {"frame": seen, "t_ns": t_ns, "state": 0, "landmarks": np.zeros((478, 3))}
            if found is not None:
                row["landmarks"], row["state"] = found[0], 1 if found[1] else 2
            with lock:
                if state["take"] is take and writer.put(take.image_dir / f"{seen:06d}.jpg", frame):
                    take.rows.append(row)

    threading.Thread(target=processing, daemon=True, name="mediapipe").start()

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
            self.setGeometry(app.primaryScreen().geometry())
            self.setCursor(QtCore.Qt.BlankCursor)
            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.next_point)
            self.redraw = QtCore.QTimer(self)
            self.redraw.timeout.connect(self.update)
            self.redraw.start(33)

        def start_take(self):
            if state["error"]:
                return
            name = names[state["index"]]
            state["take"] = Take(name, attempt_of(name), folder)
            print(f"{state['take'].label}: recording", flush=True)
            self.point = -1
            self.next_point()

        def next_point(self):
            take = state["take"]
            self.point += 1
            if self.point < len(take.points):
                take.onsets_ns.append(time.monotonic_ns())
                self.timer.start(int(protocol.POINT_S * 1000))
            else:
                self.finish_take()
            self.update()

        def finish_take(self, interrupted=False):
            take = state["take"]
            with lock:
                state["take"] = None
            take.end_ns, take.interrupted = time.monotonic_ns(), interrupted
            n, valid = take.save(folder, state["size"], {"mode": args.mode, "camera": intrinsics})
            print(f"{take.label}: {n} images, {valid} valid ({100 * valid / max(n, 1):.0f} %)" + (" — INTERRUPTED" if interrupted else ""), flush=True)
            state["log"].append({"name": take.name, "attempt": take.attempt, "frames": n, "valid": valid, "interrupted": interrupted})
            if not interrupted:
                state["done"].append(state["index"])
                state["index"] += 1
            if state["index"] >= len(names):
                print("Done. You can go back to the terminal.", flush=True)
                self.close()

        def keyPressEvent(self, event):
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                if state["take"] is not None:
                    self.timer.stop()
                    self.finish_take(interrupted=True)
                self.close()
            elif state["take"] is None and key == QtCore.Qt.Key_Space:
                self.start_take()
            elif state["take"] is None and key == QtCore.Qt.Key_R and state["done"]:
                state["index"] = state["done"].pop()   # redo the previous sub-block (a new take)
                self.update()

        def mousePressEvent(self, event):
            if state["take"] is None:
                self.start_take()

        def paintEvent(self, event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
            w, h = self.width(), self.height()
            take = state["take"]
            name = take.name if take else (names[state["index"]] if state["index"] < len(names) else None)
            nose = protocol.SUBBLOCKS[name][1] if name else None
            if nose is not None:
                u, v = w * (nose[0] + 0.5), h * (nose[1] + 0.5)
                p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150), 3))
                p.drawLine(QtCore.QPointF(u - 25, v), QtCore.QPointF(u + 25, v))
                p.drawLine(QtCore.QPointF(u, v - 25), QtCore.QPointF(u, v + 25))
            if take is not None and 0 <= self.point < len(take.points):
                x, y = take.points[self.point]
                c = QtCore.QPointF(w * (x + 0.5), h * (y + 0.5))
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(220, 30, 30))
                p.drawEllipse(c, DOT_RADIUS, DOT_RADIUS)
                p.setBrush(QtGui.QColor(0, 0, 0))
                p.drawEllipse(c, CENTRE_RADIUS, CENTRE_RADIUS)
            elif name is not None:
                p.setPen(QtGui.QColor(30, 30, 30))
                p.setFont(QtGui.QFont("Sans", 18))
                text = protocol.SUBBLOCKS[name][2] + "\n\nSpace (or a click) when you are ready.  R: redo the previous one.  Escape: stop."
                if state["error"]:
                    text = f"ERROR: {state['error']}.\nMeasure the camera again (python -m gaze checkerboard). Escape: stop."
                p.drawText(self.rect().adjusted(60, 0, -60, -120), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, text)
                p.setFont(QtGui.QFont("Sans", 12))
                p.setPen(QtGui.QColor(120, 120, 120))
                p.drawText(self.rect().adjusted(0, 0, 0, -20), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                           f"{name} ({state['index'] + 1}/{len(names)}) · camera {state['fps']} images/s · {state['size']}")
                self.draw_preview(p)

        def draw_preview(self, p):
            shot = state["preview"]
            if shot is None:
                return
            rgb = shot["rgb"]
            image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888)
            box = QtCore.QRectF(self.width() - PREVIEW[0] - 20, 20, *PREVIEW)
            p.drawImage(box, image)
            p.setFont(QtGui.QFont("Sans", 11))
            p.setPen(QtGui.QColor(20, 120, 20) if shot["open"] else QtGui.QColor(200, 30, 30))
            p.drawText(QtCore.QPointF(box.left(), box.bottom() + 22),
                       "face found, eyes open" if shot["open"] else "face found, eyes closed" if shot["face"] else "NO FACE")

        def closeEvent(self, event):
            event.accept()
            app.quit()

    window = Window()
    window.showFullScreen()
    window.raise_()
    window.activateWindow()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        app.exec_()
    finally:
        state["stop"] = True
        camera.running = False
        camera.join(timeout=2.0)   # no read may still be running when the camera is released
        capture.release()
        writer.close()
        with open(folder / "takes.jsonl", "a") as f:
            f.write(json.dumps({"started": stamp, "mode": args.mode, "takes": state["log"], "images_written": writer.written,
                                "images_dropped": writer.dropped, "images_failed": writer.failed, "image_size": state["size"]}) + "\n")
        print(f"Images: {writer.written} written, {writer.dropped} dropped, {writer.failed} not written. Camera released.", flush=True)
    return 1 if writer.failed else 0


if __name__ == "__main__":
    sys.exit(main())
