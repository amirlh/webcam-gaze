"""The geometry: the screen, the eyes, the gaze's angles and its point on the screen.

Camera frame (OpenCV): origin at the lens, x to the right of the image, y down, z from the camera towards the user, in cm.
The screen is the plane z = 0 of the screen frame: the camera frame turned by the camera's tilt (settings/config.json,
"camera_tilt_deg", about the camera's x axis; 0 for a camera in the plane of the screen, as a laptop's).
Every gaze quantity is moved into the screen frame first.

    d = R_ray d_v(alpha)     R_ray: the frame whose -z points from the eyes to the lens; d_v: the gaze in that frame
    P = E + t d, P_z = 0     E: the eyes' centre, P: the point looked at on the screen
Label of a frame whose looked-at point P is known: d = (P - E) / |P - E|, alpha = angles(R_ray^T d).
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EYE_CORNERS = [33, 133, 362, 263]
CONFIG = ROOT / "settings" / "config.json"
# the settings that place the screen and the camera: a calibration holds only for these values
SCREEN_KEYS = ("screen_width_cm", "screen_height_cm", "camera_from_left_edge_cm", "camera_above_top_edge_cm", "camera_tilt_deg")


def load_config(path=None):
    path = pathlib.Path(path or CONFIG)
    if not path.exists():
        raise SystemExit(f"STOP: no {path}. Copy settings/config_example.json to settings/config.json and write your screen's "
                         "and camera's measurements in it (see README.md).")
    return json.loads(path.read_text())


class Screen:
    """Normalized screen coordinates (centre 0, edges +-0.5, x right, y down, as the user sees it) <-> screen frame (cm)."""

    def __init__(self, config):
        self.width_cm = float(config["screen_width_cm"])
        self.height_cm = float(config["screen_height_cm"])
        self.camera_x_cm = float(config["camera_from_left_edge_cm"])   # lens centre, from the lit area's left edge
        self.camera_gap_cm = float(config["camera_above_top_edge_cm"])  # lens centre, above the lit area's top edge
        tilt = np.radians(float(config.get("camera_tilt_deg", 0.0)))  # > 0: the camera looks down
        c, s = np.cos(tilt), np.sin(tilt)
        self.tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])   # screen frame -> camera frame

    def norm_to_cm(self, xy):
        """Normalized point -> point of the screen frame (cm). The camera faces the user: the screen's right is x < 0."""
        xy = np.asarray(xy, dtype=float)
        u = (xy[..., 0] + 0.5) * self.width_cm
        v = (xy[..., 1] + 0.5) * self.height_cm
        return np.stack([-(u - self.camera_x_cm), self.camera_gap_cm + v, np.zeros_like(u)], axis=-1)

    def cm_to_norm(self, p):
        p = np.asarray(p, dtype=float)
        u = self.camera_x_cm - p[..., 0]
        v = p[..., 1] - self.camera_gap_cm
        return np.stack([u / self.width_cm - 0.5, v / self.height_cm - 0.5], axis=-1)

    def to_screen_frame(self, vectors):
        """Points or directions of the camera frame -> screen frame."""
        return np.einsum("ji,...j->...i", self.tilt, np.asarray(vectors, dtype=float))

    def to_camera_frame(self, vectors):
        """Points or directions of the screen frame -> camera frame."""
        return np.einsum("ij,...j->...i", self.tilt, np.asarray(vectors, dtype=float))


# --- The eyes -------------------------------------------------------------------------------

def landmarks_px(landmarks, width, height):
    return np.asarray(landmarks, dtype=float)[..., :2] * np.array([width, height])


def eye_centre_px(landmarks, width, height):
    return landmarks_px(landmarks, width, height)[..., EYE_CORNERS, :].mean(axis=-2)


def eye_position(eye_px, depth, focal, width, height):
    """E = ((u - W/2) Z / f, (v - H/2) Z / f, Z): the eyes' centre back-projected at depth Z (camera frame)."""
    eye_px = np.asarray(eye_px, dtype=float)
    z = np.asarray(depth, dtype=float)
    return np.stack([(eye_px[..., 0] - width / 2.0) * z / focal, (eye_px[..., 1] - height / 2.0) * z / focal, z], axis=-1)


EYES = [dict(corners=(33, 133), lids=(159, 145), iris=(468, 469, 470, 471, 472)),
        dict(corners=(362, 263), lids=(386, 374), iris=(473, 474, 475, 476, 477))]


def iris_features(landmarks, size):
    """Per eye: the iris centre (x, y) and the eyelids' apexes (upper, lower: y) in the eye's own frame, in eye widths (8 numbers)."""
    px = landmarks_px(np.asarray(landmarks, dtype=float).reshape(-1, 478, 3), *size)
    out = []
    for eye in EYES:
        left, right = px[:, eye["corners"][0]], px[:, eye["corners"][1]]
        origin = (left + right) / 2.0
        width = np.linalg.norm(right - left, axis=1, keepdims=True)
        ax = (right - left) / width
        ay = np.stack([-ax[:, 1], ax[:, 0]], axis=1)
        iris = px[:, list(eye["iris"])].mean(axis=1) - origin
        upper, lower = px[:, eye["lids"][0]] - origin, px[:, eye["lids"][1]] - origin
        out += [np.sum(iris * ax, 1) / width[:, 0], np.sum(iris * ay, 1) / width[:, 0],
                np.sum(upper * ay, 1) / width[:, 0], np.sum(lower * ay, 1) / width[:, 0]]
    return np.stack(out, axis=1)


# --- Angles and rays ------------------------------------------------------------------------

def direction_to_angles(d):
    """Unit direction -> (alpha_x, alpha_y) = (atan2(d_x, -d_z), atan2(d_y, -d_z)), radians."""
    d = np.asarray(d, dtype=float)
    return np.stack([np.arctan2(d[..., 0], -d[..., 2]), np.arctan2(d[..., 1], -d[..., 2])], axis=-1)


def angles_to_direction(alpha):
    alpha = np.asarray(alpha, dtype=float)
    d = np.stack([np.tan(alpha[..., 0]), np.tan(alpha[..., 1]), -np.ones(alpha.shape[:-1])], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def intersect_screen(eye, direction):
    """Point where the ray eye + t direction meets the plane z = 0 (nan if it moves away from it)."""
    eye, direction = np.asarray(eye, dtype=float), np.asarray(direction, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[..., 2] / direction[..., 2]
    t = np.where(t > 0, t, np.nan)
    return eye + t[..., None] * direction


def label_angles(point, eye, rotation):
    d = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return direction_to_angles(np.einsum("...ji,...j->...i", rotation, d))


def gaze_direction(alpha, rotation):
    """The gaze's unit direction in the screen frame, from its two angles in the ray frame."""
    return np.einsum("...ij,...j->...i", rotation, angles_to_direction(alpha))


def predict_point(alpha, eye, rotation):
    return intersect_screen(eye, gaze_direction(alpha, rotation))


def ray_frame(eye):
    """Rotation whose (0, 0, -1) points from the eye to the camera (the origin), x as close as possible to (1, 0, 0)."""
    eye = np.asarray(eye, dtype=float)
    z = eye / np.linalg.norm(eye, axis=-1, keepdims=True)
    x = np.array([1.0, 0.0, 0.0]) - z[..., :1] * z
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=-1)
