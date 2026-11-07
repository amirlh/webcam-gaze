"""The face: MediaPipe's landmarks, the lens's distortion, the head's pose, the face's normalisation and UniGaze ViT-H.

- Landmarks: MediaPipe's face landmarker (resources/face_landmarker_v2_with_blendshapes.task), one face, on the RGB image.
  Eyes open: eye aspect ratio >= 0.2 for both eyes.
- Pose: ptgaze's 3D face model (resources/ptgaze_face_model.npz, MIT licence) fitted to the 468 landmarks by solvePnP with
  the camera's true focal (gaze/checkerboard.py). The eyes' depth is the mean depth of its 4 eye corners.
- Normalisation (Zhang et al. 2018): a virtual camera f = 960 px at 60 cm looking at the face centre without roll, 224 x 224,
  RGB, ImageNet mean and deviation.
- Network: UniGaze ViT-H ("unigaze_h14_joint" weights, Hugging Face UniGaze/UniGaze-models, non-commercial licence), in full
  float32 (no TF32), or float16 after to_half(). Its (pitch, yaw) is turned back into the camera frame by R_n^T.
"""

import pathlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import geometry

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANDMARKER_PATH = ROOT / "resources" / "face_landmarker_v2_with_blendshapes.task"
_FACE = np.load(ROOT / "resources" / "ptgaze_face_model.npz")
FACE_3D, FACE_CENTRE = _FACE["landmarks"], np.concatenate([_FACE["reye"], _FACE["leye"], _FACE["nose"]])
MODEL = "unigaze_h14_joint"
K_N = np.array([[960.0, 0.0, 112.0], [0.0, 960.0, 112.0], [0.0, 0.0, 1.0]])
SIZE_N = (224, 224)
DISTANCE_N = 0.6   # m, as ptgaze's face model
EAR_THRESHOLD = 0.2
EAR_POINTS = {"left": [362, 385, 387, 263, 373, 380], "right": [133, 158, 160, 33, 144, 153]}
MEAN, SD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
JPEG_QUALITY = 100   # the recorded images, and the live ones before the network, so that it sees the same thing


def as_recorded(image_bgr):
    """The image as the recording writes it and the calibration reads it back (JPEG, JPEG_QUALITY)."""
    return cv2.imdecode(cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1], cv2.IMREAD_COLOR)


# --- The camera -----------------------------------------------------------------------------

def camera_matrix(focal, width, height):
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


class Straightener:
    """Undoes the lens's radial distortion (k1, k2 of gaze/checkerboard.py), the same camera matrix kept."""

    def __init__(self, focal, k1_k2, width, height):
        K = camera_matrix(focal, width, height)
        self.maps = cv2.initUndistortRectifyMap(K, np.array([*k1_k2, 0.0, 0.0]), None, K, (width, height), cv2.CV_32FC1)

    def __call__(self, image):
        return cv2.remap(image, self.maps[0], self.maps[1], cv2.INTER_LINEAR)


# --- MediaPipe ------------------------------------------------------------------------------

def landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)), num_faces=1))


def detect(detector, image_bgr):
    """(landmarks 478 x 3 normalized, eyes open) or None if no face."""
    import mediapipe as mp
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        return None
    landmarks = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
    px = landmarks[:, :2] * np.array([image_bgr.shape[1], image_bgr.shape[0]])
    return landmarks, min(ear(px, side) for side in ("left", "right")) >= EAR_THRESHOLD


def ear(px, side):
    """Eye aspect ratio."""
    p1, p2, p3, p4, p5, p6 = (px[i] for i in EAR_POINTS[side])
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))


# --- The pose and the normalisation ---------------------------------------------------------

def head_pose(px468, K):
    """solvePnP of the 3D face model, from an unrotated head 1 m ahead: (rotation, the model's points in the camera frame, m)."""
    _, rvec, tvec = cv2.solvePnP(FACE_3D, np.ascontiguousarray(px468, dtype=np.float64), K, np.zeros((1, 5)),
                                 np.zeros(3), np.array([0.0, 0.0, 1.0]), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    rot = Rotation.from_rotvec(np.asarray(rvec, dtype=float).reshape(3))
    return rot, FACE_3D @ rot.as_matrix().T + np.asarray(tvec, dtype=float).reshape(3)


def eye_depth_cm(landmarks, focal, width, height):
    """The eyes' depth (cm): the mean depth of the posed model's 4 eye corners."""
    _, model3d = head_pose(np.asarray(landmarks)[:468, :2] * np.array([width, height]), camera_matrix(focal, width, height))
    return float(model3d[geometry.EYE_CORNERS, 2].mean() * 100.0)


def normalising_rotation(centre, head_rot):
    """Zhang et al. 2018, section 4.2: rows x, y, z of the virtual camera in the camera frame."""
    z = centre / np.linalg.norm(centre)
    y = np.cross(z, head_rot.as_matrix()[:, 0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x / np.linalg.norm(x), y, z])


def normalised_face(image_bgr, landmarks, focal):
    """224 x 224 BGR face image and R_n."""
    h, w = image_bgr.shape[:2]
    K = camera_matrix(focal, w, h)
    rot, model3d = head_pose(np.asarray(landmarks)[:468, :2] * np.array([w, h]), K)
    centre = model3d[FACE_CENTRE].mean(axis=0)
    R_n = normalising_rotation(centre, rot)
    W = K_N @ np.diag([1.0, 1.0, DISTANCE_N / np.linalg.norm(centre)]) @ R_n @ np.linalg.inv(K)
    return cv2.warpPerspective(image_bgr, W, SIZE_N), R_n


def to_tensor(face_bgr):
    rgb = face_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / SD).transpose(2, 0, 1).astype(np.float32)


def gaze_vector(pitch_yaw):
    """(pitch, yaw) -> unit direction, looking along -z for (0, 0)."""
    pitch, yaw = pitch_yaw[..., 0], pitch_yaw[..., 1]
    return -np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=-1)


# --- The network ----------------------------------------------------------------------------

def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Model:
    """UniGaze ViT-H on the best device; predict(faces) -> unit gaze directions in each face's virtual camera."""

    def __init__(self):
        import torch
        import unigaze
        torch.backends.cuda.matmul.allow_tf32 = False   # full float32 arithmetic, as the calibration
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.half, self.device = False, device()
        self.net = unigaze.load(MODEL, device="cpu").eval().to(self.device)

    def to_half(self):
        """float16 from now on: faster, slightly different from the calibration's float32."""
        self.net, self.half = self.net.half(), True

    def predict(self, faces_bgr):
        import torch
        x = torch.from_numpy(np.stack([to_tensor(f) for f in faces_bgr])).to(self.device)
        x = x.half() if self.half else x
        with torch.no_grad():
            out = self.net(x)["pred_gaze"].float().cpu().numpy()
        return gaze_vector(out)


def to_camera(model, faces, rotations):
    """The network on normalised faces, its unit directions turned back into the camera frame (v R_n)."""
    v = np.einsum("nj,nji->ni", model.predict(list(faces)), np.stack(rotations))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def gaze_cam(model, images_bgr, landmarks, focal):
    """Unit gaze directions in the camera frame for these images (one face each)."""
    faces, rotations = zip(*(normalised_face(im, lm, focal) for im, lm in zip(images_bgr, landmarks)))
    return to_camera(model, faces, rotations)
