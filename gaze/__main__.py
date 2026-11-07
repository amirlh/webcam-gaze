"""Webcam gaze tracking: the one command.

    python -m gaze checkerboard                    the camera's focal and distortion (once per camera, ~1 min)
    python -m gaze calibrate [--checkerboard]      the long calibration (once per screen and camera, ~6 min + ~5 min of computing)
    python -m gaze session [--data] [--csv]        the short calibration (9 points, ~20 s), then the live gaze

calibrate: if settings/camera.json does not exist (or with --checkerboard), the checkerboard first; then the 13 sub-blocks of
gaze/protocol.py into data/calibration_<date>/, the network on their images, the ridge; the folder is written to
settings/current_calibration.txt.
session: the 9 points into data/session_<date>/, the network on them, the adaptation chosen on them, then the live gaze: the
green dot on the screen, or with --data the gaze's numbers in the terminal (no window). --csv: every gaze point is also written
to data/session_<date>/gaze.csv (always with --data). --calibration DIR: another calibration than the current one. --half: the
network in float16 (faster, slightly different from the calibration's float32).
"""

import argparse
import datetime
import os
import pathlib
import subprocess
import sys

from . import camera as cam
from . import face, geometry, method, protocol

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = ROOT / "settings" / "current_calibration.txt"


def step(module, *args):
    """A part of the program in its own process (its own Qt window), from the repository's folder."""
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT).returncode


def new_folder(kind):
    folder = DATA / f"{kind}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    folder.mkdir(parents=True)
    return folder


def calibrate(args):
    config = geometry.load_config()   # stops before any camera if settings/config.json is missing
    if args.checkerboard or not cam.CAMERA.exists():
        print("1. The camera's focal: show the checkerboard to the camera, strongly tilted every way.", flush=True)
        if step("gaze.checkerboard") or not cam.CAMERA.exists():
            sys.exit("STOP: no camera focal (settings/camera.json).")
    intrinsics = cam.load_intrinsics()
    folder = new_folder("calibration")
    print(f"2. The long calibration: 13 sub-blocks, about 6 min, camera on -> {folder}", flush=True)
    if step("gaze.record", "--mode", "calibration", "--folder", folder):
        sys.exit("STOP: the recording failed (see the message above).")
    runs = method.load_runs(folder, protocol.CALIBRATION)
    print(f"3. The network ({face.MODEL}) on the calibration's images", flush=True)
    method.compute_outputs(face.Model(), folder, runs, intrinsics["focal_px"])
    result = method.calibrate(folder, runs, intrinsics, config)
    print(f"4. The correction: lambda {result['lambda']:g}, validation {result['validation_deg']:.2f}° ({result['validation_cm']:.2f} cm on "
          f"the screen) on {result['frames']} frames", flush=True)
    CURRENT.write_text(os.path.relpath(folder, ROOT) + "\n")
    print(f"-> {CURRENT.relative_to(ROOT)}: {CURRENT.read_text().strip()}", flush=True)
    return 0


def session(args):
    intrinsics = cam.load_intrinsics()
    if not args.calibration and not CURRENT.exists():
        sys.exit("STOP: no calibration yet: python -m gaze calibrate.")
    setup = method.load_calibration(ROOT / (args.calibration or CURRENT.read_text().strip()), intrinsics)
    config = geometry.load_config()
    if any(setup["config"].get(k, 0) != config.get(k, 0) for k in geometry.SCREEN_KEYS):
        sys.exit("STOP: the screen or the camera's place in settings/config.json changed since this calibration: make the long "
                 "calibration again (python -m gaze calibrate).")
    setup["config"] = config
    folder = new_folder("session")
    print(f"1. The short calibration: 9 points, about 20 s, camera on -> {folder}", flush=True)
    if step("gaze.record", "--mode", "short", "--folder", folder):
        sys.exit("STOP: the recording failed (see the message above).")
    runs = method.load_runs(folder, protocol.SHORT)
    print(f"2. The network ({face.MODEL}) on the 9 points", flush=True)
    model = face.Model()
    method.compute_outputs(model, folder, runs, intrinsics["focal_px"])   # float32, as the calibration
    if args.half:
        model.to_half()
    session_frames = method.frames(runs, protocol.SHORT, intrinsics["focal_px"], geometry.Screen(config), folder / "outputs")
    adaptation = method.adapt_person(setup, session_frames)
    print(f"3. The adaptation, chosen on the 9 points: {adaptation['way']} ("
          + ", ".join(f"{w} {adaptation['error_deg'][w]:.2f}° ({v:.2f} cm)" for w, v in adaptation["error_cm"].items()) + ")", flush=True)
    print("4. The live gaze" + (" (numbers in the terminal, Ctrl+C stops)" if args.data else " (Escape quits)"), flush=True)
    from . import live
    return live.run(setup, adaptation, model, show="data" if args.data else "dot",
                    csv_path=folder / "gaze.csv" if (args.data or args.csv) else None)


def main():
    parser = argparse.ArgumentParser(prog="python -m gaze", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("checkerboard", help="the camera's focal and distortion", allow_abbrev=False)
    c = commands.add_parser("calibrate", help="the long calibration", allow_abbrev=False)
    c.add_argument("--checkerboard", action="store_true", help="measure the camera's focal again first")
    s = commands.add_parser("session", help="the 9 points, then the live gaze", allow_abbrev=False)
    s.add_argument("--data", action="store_true", help="the gaze's numbers in the terminal instead of the dot")
    s.add_argument("--csv", action="store_true", help="also write every gaze point to the session's gaze.csv")
    s.add_argument("--calibration", help="another calibration folder than settings/current_calibration.txt")
    s.add_argument("--half", action="store_true", help="the live network in float16")
    args = parser.parse_args()
    if args.command == "checkerboard":
        return step("gaze.checkerboard")
    return {"calibrate": calibrate, "session": session}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
