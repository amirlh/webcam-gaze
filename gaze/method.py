"""The method: the frames of a recording, the network on them, the correction (a ridge) and the adaptation to a person.

Per frame (the valid frames: face found, eyes open, from protocol.SKIP_S to protocol.POINT_S after the dot appeared):
    E       the eyes' centre, screen frame (cm): MediaPipe's eye corners in pixels, at the depth of the head's pose
    net     the network's gaze as 2 angles in the eye -> camera frame ("ray")
    iris    MediaPipe's 8 eye numbers (each iris's centre x, y and its upper and lower eyelids, in eye widths)
    alpha   the label: the direction to the dot in the same frame
E, ray, net and iris come from features(), the same function for the recordings and for the live gaze.
A ridge maps [net, iris] to alpha; P = E + t R_ray d(alpha) on the screen. Its lambda is chosen by cross-validation: the mean
of two scores, (each sub-block left out in turn) and (5 groups of dot positions left out in turn), the median screen error of
each, the sub-blocks weighing equally. The same score in degrees is given beside it: the angle at the eyes between the gaze and the
direction to the dot (E at the pose's distance, which a webcam's average face puts a few per cent too far: the degrees are
then a few per cent too small).
At each session, the person's 9 points choose one of three adaptations, each point left out in turn: none (the calibration's
ridge as it is), offset (plus a constant offset of the 2 angles), refit (the ridge fitted again on the calibration's frames and
the 9 points', the 9 points weighing as much as the whole calibration, the same lambda).
"""

import collections
import concurrent.futures
import json
import pathlib
import sys

import cv2
import numpy as np

from . import face, geometry, protocol

LAMBDAS = 10.0 ** np.arange(-2.0, 5.01, 0.5)
TARGET_FOLDS, SEED = 5, 20261026
WAYS = ("none", "offset", "refit")
BATCH, READERS = 32, 8


# --- Recordings -----------------------------------------------------------------------------

def label_of(run):
    return f"{run['info']['name']}_take{run['info']['attempt']}"


def load_runs(folder, names):
    """The last complete take of each of these sub-blocks of a recording folder."""
    runs = {}
    for path in sorted(pathlib.Path(folder).glob("*_take*.npz")):
        if path.stem.split("_take")[0] not in names:
            continue
        d = np.load(path)
        info = json.loads(str(d["info"]))
        if info["interrupted"]:
            continue
        if info["name"] not in runs or info["attempt"] > runs[info["name"]]["info"]["attempt"]:
            runs[info["name"]] = {"info": info, **{k: d[k] for k in d.files if k != "info"}}
    missing = [n for n in names if n not in runs]
    if missing:
        raise SystemExit(f"STOP: {', '.join(missing)} missing in {folder} (window closed before the end?).")
    return runs


def valid(run):
    s = run["since_onset_s"]
    return (run["state"] == 1) & (run["dot_index"] >= 0) & np.isfinite(s) & (s >= protocol.SKIP_S) & (s <= protocol.POINT_S)


def compute_outputs(model, folder, runs, focal):
    """The network on every frame with a face and open eyes: <folder>/outputs/<take>.npz (frame_id, gaze_cam: unit, camera
    frame). The images were undistorted when recorded. The images are read and normalised by READERS threads while the
    network runs on the previous batch."""
    out = pathlib.Path(folder) / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(label_of(r), int(i), pathlib.Path(folder) / "images" / label_of(r) / f"{int(i):06d}.jpg", lm.astype(float))
            for r in runs.values() for i, lm, s in zip(r["frame_id"], r["landmarks"], r["state"]) if s == 1]
    if not jobs:
        return

    def prepare(job):
        image = cv2.imread(str(job[2]))
        if image is None:
            raise SystemExit(f"STOP: {job[2]} cannot be read (disk full during the recording?): record again.")
        return face.normalised_face(image, job[3], focal)

    chunks = [jobs[k:k + BATCH] for k in range(0, len(jobs), BATCH)]
    rows = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(READERS) as pool, concurrent.futures.ThreadPoolExecutor(1) as ahead:

        def read(chunk):
            return list(pool.map(prepare, chunk))

        future = ahead.submit(read, chunks[0])
        for k, chunk in enumerate(chunks):
            faces, rotations = zip(*future.result())
            if k + 1 < len(chunks):
                future = ahead.submit(read, chunks[k + 1])
            for job, g in zip(chunk, face.to_camera(model, faces, rotations)):
                rows[job[0]].append((job[1], g))
            if (k + 1) % 40 == 0 or k + 1 == len(chunks):
                print(f"  network: {min((k + 1) * BATCH, len(jobs))}/{len(jobs)} images", flush=True)
    for label, r in rows.items():
        np.savez(out / f"{label}.npz", frame_id=np.array([x[0] for x in r]), gaze_cam=np.array([x[1] for x in r]))


def network_outputs(outputs, label):
    """The network's gaze (unit, camera frame) of a take, by frame id; {} if it was not computed."""
    path = pathlib.Path(outputs) / f"{label}.npz"
    if not path.exists():
        return {}
    d = np.load(path)
    return dict(zip(d["frame_id"].tolist(), d["gaze_cam"]))


# --- Frames ---------------------------------------------------------------------------------

def features(landmarks, gaze_cam, focal, size, screen):
    """What the ridge needs from n frames (landmarks n x 478 x 3, the network's gaze n x 3 in the camera frame): E, the ray
    frame, the 8 eye numbers and the network's 2 angles (see the module's docstring)."""
    depth = np.array([face.eye_depth_cm(x, focal, *size) for x in landmarks])
    E = screen.to_screen_frame(geometry.eye_position(geometry.eye_centre_px(landmarks, *size), depth, focal, *size))
    ray = geometry.ray_frame(E)
    return {"E": E, "ray": ray, "iris": geometry.iris_features(landmarks, size),
            "net": geometry.direction_to_angles(np.einsum("nji,nj->ni", ray, screen.to_screen_frame(gaze_cam)))}


def frames(runs, names, focal, screen, outputs):
    """The valid frames of these sub-blocks, with their features, dot and label."""
    parts = []
    for name in names:
        run = runs[name]
        k = valid(run)
        by_id = network_outputs(outputs, label_of(run))
        v = np.array([by_id.get(int(i), (np.nan,) * 3) for i in run["frame_id"][k]], dtype=float).reshape(-1, 3)
        part = features(run["landmarks"][k].astype(float), v, focal, run["info"]["image_size"], screen)
        part["P"] = screen.norm_to_cm(run["dot_xy"][k])
        part.update({"target": run["dot_xy"][k], "dot_index": run["dot_index"][k], "subblock": np.full(int(k.sum()), name),
                     "alpha": geometry.label_angles(part["P"], part["E"], part["ray"])})
        parts.append(part)
    fr = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if not np.isfinite(fr["net"]).all():
        raise SystemExit("STOP: some frames have no network output.")
    return fr


def columns(fr):
    return np.hstack([fr["net"], fr["iris"]])


def subset(fr, mask):
    return {k: v[mask] for k, v in fr.items()}


# --- The ridge ------------------------------------------------------------------------------

def subblock_weights(names):
    w = np.zeros(len(names))
    for name in set(names.tolist()):
        m = names == name
        w[m] = 1.0 / m.sum()
    return w * len(w) / w.sum()


class Ridge:
    """Weighted ridge on standardized columns, for one or every lambda."""

    def __init__(self, x, y, w):
        keep = np.isfinite(x).all(axis=1)
        x, y, w = x[keep], y[keep], w[keep]
        self.mean = np.average(x, axis=0, weights=w)
        sd = np.sqrt(np.average((x - self.mean) ** 2, axis=0, weights=w))
        self.sd = np.where(sd < 1e-6, 1.0, sd)
        self.y_mean = np.average(y, axis=0, weights=w)
        root = np.sqrt(w)[:, None]
        xs, ys = (x - self.mean) / self.sd * root, (y - self.y_mean) * root
        self.values, self.vectors = np.linalg.eigh(xs.T @ xs)
        self.projected = self.vectors.T @ (xs.T @ ys)

    def coef(self, lam):
        return self.vectors @ (self.projected / (self.values + lam)[:, None])

    def predict(self, x, lam):
        return apply(self.frozen(lam), x)

    def frozen(self, lam):
        return {"mean": self.mean, "sd": self.sd, "coef": self.coef(lam), "y_mean": self.y_mean}


def apply(frozen, x):
    return ((x - frozen["mean"]) / frozen["sd"]) @ frozen["coef"] + frozen["y_mean"]


def points(alpha, fr):
    return geometry.predict_point(alpha, fr["E"], fr["ray"])


def angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def errors_deg(p, fr):
    """Each frame's angle (degrees) at the eyes between the predicted point and the dot."""
    return angle_deg(p - fr["E"], fr["P"] - fr["E"])


def validate(cal, x):
    """(best lambda, its cross-validation score in cm, the same score in degrees); lambda is chosen on the cm."""
    w = subblock_weights(cal["subblock"])

    def errors(test, alpha):
        """Each test frame's error on the screen (cm) and at the eyes (degrees)."""
        part = subset(cal, test)
        p = points(alpha, part)
        return np.linalg.norm(p[:, :2] - part["P"][:, :2], axis=1), errors_deg(p, part)

    groups = list(dict.fromkeys(cal["subblock"].tolist()))
    by_subblock = np.zeros((2, len(LAMBDAS), len(groups)))   # (cm, degrees) x lambdas x sub-block left out
    for j, group in enumerate(groups):
        test = cal["subblock"] == group
        ridge = Ridge(x[~test], cal["alpha"][~test], w[~test])
        for i, lam in enumerate(LAMBDAS):
            by_subblock[:, i, j] = [np.nanmedian(e) for e in errors(test, ridge.predict(x[test], lam))]
    positions = sorted({tuple(t) for t in cal["target"].round(3)})
    fold_of = dict(zip(positions, np.random.default_rng(SEED).permutation(len(positions)) % TARGET_FOLDS))
    folds = np.array([fold_of[tuple(t)] for t in cal["target"].round(3)])
    by_position = np.full((2, len(LAMBDAS), TARGET_FOLDS, len(groups)), np.nan)   # ... x group left out x sub-block
    for f in range(TARGET_FOLDS):
        test = folds == f
        ridge = Ridge(x[~test], cal["alpha"][~test], w[~test])
        names = cal["subblock"][test]
        for i, lam in enumerate(LAMBDAS):
            for u, err in enumerate(errors(test, ridge.predict(x[test], lam))):
                by_position[u, i, f] = [np.nanmedian(err[names == s]) if np.any(names == s) else np.nan for s in groups]
    score = (by_subblock.mean(axis=2) + np.nanmean(by_position, axis=(2, 3))) / 2   # (cm, degrees) x lambdas
    best = int(np.nanargmin(score[0]))
    return float(LAMBDAS[best]), float(score[0, best]), float(score[1, best])


# --- The calibration ------------------------------------------------------------------------

def calibrate(folder, runs, intrinsics, config):
    """The ridge on the long calibration, its lambda chosen by cross-validation; saved to <folder>/model.npz."""
    screen = geometry.Screen(config)
    cal = frames(runs, protocol.CALIBRATION, intrinsics["focal_px"], screen, pathlib.Path(folder) / "outputs")
    x = columns(cal)
    lam, score, score_deg = validate(cal, x)
    w = subblock_weights(cal["subblock"])
    frozen = Ridge(x, cal["alpha"], w).frozen(lam)
    np.savez(pathlib.Path(folder) / "model.npz", lam=lam, validation_cm=score, validation_deg=score_deg, intrinsics=json.dumps(intrinsics),
             config=json.dumps(config), x=x, alpha=cal["alpha"], w=w, **{f"frozen_{k}": v for k, v in frozen.items()})
    return {"lambda": lam, "validation_cm": score, "validation_deg": score_deg, "frames": len(x)}


def load_calibration(folder, intrinsics):
    path = pathlib.Path(folder) / "model.npz"
    if not path.exists():
        raise SystemExit(f"STOP: no {path}: make the long calibration first (python -m gaze calibrate).")
    d = np.load(path)
    measured = json.loads(str(d["intrinsics"]))
    if measured["focal_px"] != intrinsics["focal_px"] or measured["k1_k2"] != intrinsics["k1_k2"]:
        raise SystemExit("STOP: settings/camera.json changed since this calibration (another checkerboard): make the long "
                         "calibration again, two focals never mix.")
    return {"lambda": float(d["lam"]), "intrinsics": measured, "config": json.loads(str(d["config"])),
            "frozen": {k: d[f"frozen_{k}"] for k in ("mean", "sd", "coef", "y_mean")}, "x": d["x"], "alpha": d["alpha"], "w": d["w"]}


# --- The adaptation -------------------------------------------------------------------------

def adapt(way, setup, x, alpha, dot_index):
    """One way to adapt, fitted on a person's frames (x, alpha; dot_index: each frame's dot)."""
    if way == "none":
        return {"frozen": setup["frozen"], "offset": np.zeros(2)}
    if way == "offset":
        residual = alpha - apply(setup["frozen"], x)
        per_point = [np.nanmedian(residual[dot_index == i], axis=0) for i in np.unique(dot_index)]
        return {"frozen": setup["frozen"], "offset": np.nanmean(per_point, axis=0)}
    if way == "refit":
        w_person = np.full(len(x), setup["w"].sum() / max(len(x), 1))   # the 9 points weigh as much as the whole calibration
        ridge = Ridge(np.vstack([setup["x"], x]), np.vstack([setup["alpha"], alpha]), np.concatenate([setup["w"], w_person]))
        return {"frozen": ridge.frozen(setup["lambda"]), "offset": np.zeros(2)}
    raise ValueError(way)


def adapt_person(setup, session):
    """The way chosen on the 9 points only (session: their frames; each point left out in turn, on the cm), then fitted on
    all of them: {"frozen", "offset", "way", "error_cm", "error_deg"}."""
    x = columns(session)
    error_cm, error_deg = {}, {}
    for way in WAYS:
        errors, angles = [], []
        for i in np.unique(session["dot_index"]):
            out = session["dot_index"] == i
            params = adapt(way, setup, x[~out], session["alpha"][~out], session["dot_index"][~out])
            p = points(apply(params["frozen"], x[out]) + params["offset"], subset(session, out))
            errors.append(np.nanmedian(np.linalg.norm(p[:, :2] - session["P"][out, :2], axis=1)))
            angles.append(np.nanmedian(errors_deg(p, subset(session, out))))
        error_cm[way], error_deg[way] = float(np.mean(errors)), float(np.mean(angles))
    if not np.isfinite(list(error_cm.values())).all():
        sys.exit(f"STOP: too few of the 9 points have valid frames (face found, eyes open): do the 9 points again ({error_cm}).")
    way = min(WAYS, key=lambda w: error_cm[w])
    return {**adapt(way, setup, x, session["alpha"], session["dot_index"]), "way": way, "error_cm": error_cm, "error_deg": error_deg}
