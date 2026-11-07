# webcam-gaze

Where you look on your screen, from an ordinary webcam.

Each camera image goes through these steps:
1. The face is found with **MediaPipe**, after the lens distortion has been undone.
2. The face is normalised with the camera's true focal length.
3. **UniGaze ViT-H** gives the gaze direction.
4. A small correction learnt for your screen and camera (a ridge) turns it into a point on the screen.

The correction comes from a calibration of about 6 minutes, done once. A 9-point calibration of about 20 seconds then adapts it to the person at the start of each session.

## What you need

- Linux (developed on Ubuntu 26.04, GNOME on Wayland), Python 3.11.
- A webcam, 1920 x 1080 at 30 images/s.
- An NVIDIA GPU is strongly advised: ViT-H takes about 30 ms per image on a laptop RTX 5090. It also runs on the processor, but slowly.
- A checkerboard to measure the camera, once: OpenCV's `pattern.png`, 9 x 6 inner corners, shown on a phone or printed.

## Install

```bash
bash install.sh
```

This creates `~/webcam_gaze_venv` (or `$GAZE_VENV`) and downloads ViT-H's weights, about 2.5 GB. Every command below is run from this folder.

## Set up the screen

```bash
cp settings/config_example.json settings/config.json
```

Then write your own measurements in `settings/config.json`:

| Key | Meaning |
|---|---|
| `screen_width_cm`, `screen_height_cm` | the lit area of the screen, in cm |
| `screen_width_px`, `screen_height_px` | its resolution, for the gaze in pixels |
| `camera_from_left_edge_cm` | the lens's centre, from the lit area's left edge |
| `camera_above_top_edge_cm` | the lens's centre, above the lit area's top edge |
| `camera_tilt_deg` | 0 for a camera in the plane of the screen (a laptop's); > 0 if it looks down |
| `camera_index`, `camera_width`, `camera_height`, `camera_fps`, `camera_mjpg` | which camera and how to open it |

## Use

```bash
PY=~/webcam_gaze_venv/bin/python   # or $GAZE_VENV/bin/python
$PY -m gaze calibrate          # once per screen and camera: about 6 min of dots, then about 5 min of computing
$PY -m gaze session            # each time: 9 points (about 20 s), then the green gaze dot on the screen
$PY -m gaze session --data     # 9 points, then the gaze's numbers in the terminal and in a CSV file, without a window
```

**`calibrate`**
- The first time, it measures the camera with the checkerboard. Tilt the board strongly (30 to 50 degrees) every way, near and far, and in every corner of the image, until 25 views are kept. Then press Escape. `--checkerboard` measures the camera again.
- It then shows 13 series of red dots at different places: your usual distance, nearer, farther, shifted to each side, and your head turned towards a grey cross. **Do not hold your head still:** move it gently all the time, as in normal use, with your eyes on the black centre of each dot.
- Space (or a click) starts a series, R redoes the previous one, Escape stops.
- Between series, the camera's image is shown in the top right corner.

**`session`**
- First, 9 points with your head still.
- The program then chooses how to adapt the calibration to you, using these 9 points only. There are three choices: none, a constant offset, or the correction fitted again with your points.
- Then the live gaze. Keys: **F** filtered or raw point, **G** reference crosses, **I** the gaze's numbers, **Escape** quit.
- To judge the dot, fix a cross and have someone else watch where the dot is. If you look at the dot yourself, it follows your gaze and seems to slide.

**`--data`**
- Without a window: twice a second, the terminal prints where you look, where your eyes are and in which direction they look.
- Every point also goes to `data/session_<date>/gaze.csv`. `--csv` writes this file during the dot view as well.
- Ctrl+C stops.

Other options: `session --calibration data/calibration_<date>` uses another calibration than the current one. `--half` runs the live network in float16: it is faster, and slightly different from the calibration's float32.

## The gaze's numbers (`gaze.csv`)

| Column | Meaning |
|---|---|
| `time_s`, `delay_ms` | the camera image's time since the start; the delay from that image to the point |
| `screen_x`, `screen_y` | the point looked at, normalized: centre 0, edges ±0.5, x to the right, y down, as you see the screen |
| `screen_x_filtered`, `screen_y_filtered` | the same after the saccade filter (the dot's F key) |
| `screen_x_px`, `screen_y_px` | the point in screen pixels, from the top-left corner |
| `screen_x_cm`, `screen_y_cm` | the point in cm, from the lit area's top-left corner |
| `eyes_x_cm`, `eyes_y_cm`, `eyes_z_cm` | the gaze's origin: the centre of the two eyes, camera frame |
| `gaze_dx`, `gaze_dy`, `gaze_dz` | the gaze's unit direction, after the correction, camera frame |
| `hit_x_cm`, `hit_y_cm`, `hit_z_cm` | where the gaze meets the screen's plane, camera frame |

**Camera frame:**
- origin at the lens, in cm;
- x towards the right of the camera's image (your left);
- y down;
- z from the camera towards you.

## Accuracy

Accuracy is given in degrees of visual angle: the angle, at the eyes, between where you look and where the program puts the point. At 57 cm from the screen, 1° is about 1 cm.

It was measured once, on the author's laptop: a 16-inch screen, a built-in 1080p webcam, about 60 cm away, evening light, one person. The figures are for this version: the moving-head calibration and all 8 eye numbers. The test's targets are at 12 places, 10 of which the calibration never shows:

- **guided:** 4 series of 24 targets, with the head held at a given place (usual place, nearer and shifted, head turned);
- **moving:** 24 targets while moving all the time;
- **free:** 2 series of 24 targets in free use.

The error is measured in two ways.

**1. Per target.** For each target, take the median error over the images while it is looked at (from 0.6 s after it appears, when the eyes have landed). Then take the mean over all targets.

**2. As Pupil Labs measures it** (Kassner et al. 2014; `accuracy_visualizer.py` of their software):
- every image while a target is shown counts;
- images more than 5° off are dropped as human error;
- **accuracy** is arccos of the mean cosine of the angle, which is about the root mean square angle;
- **precision** is the root mean square of the angle between successive images, keeping only steps under 0.5°.

| | 1. Per target | 2. Pupil Labs: accuracy | 2. Pupil Labs: precision |
|---|---|---|---|
| Guided | 1.36° | 1.63° | 0.31° |
| Moving | 1.40° | 1.71° | 0.33° |
| Free | 1.72° | 2.00° | 0.34° |

Taking instead the median over the targets of each series, then the mean over the series, gives 1.37°, 1.46° and 1.54° for the same test.

In the second way, about 11 % of the images were dropped as more than 5° off. For comparison, Pupil Labs' head-mounted tracker reaches 0.6° accuracy and 0.08° precision in its makers' own test.

- **Do the 9 points with your head still.** In that test, they were done while moving, and the offset they chose made the guided error worse. On the same test, the calibration without any adaptation gave, per target, 0.89° guided, 1.39° moving and 1.78° free.
- **The 8 eye numbers help.** They were measured on the same test, with a head-still calibration, per target. Against the network's gaze alone, they lower the error by 18 % guided (1.40 → 1.15°), 16 % moving (1.80 → 1.51°) and 14 % free (1.97 → 1.69°). Against 6 numbers (without the lower eyelids), they lower it by 8 to 10 %. Each gain is established: the 90 % interval of the paired difference per target excludes zero.
- **Other networks.** ptgaze's ETH-XGaze, MPIIFaceGaze and MPIIGaze ResNets were tried in the same chain, instead of UniGaze ViT-H. They were 0.3 to 1.7° less accurate.
- **This version has not been measured on other people.** One other person was measured once, with an earlier version of the chain.

These figures put the eyes at their true distance, found from the size of the iris. The program's own figures (the calibration's validation and the 9 points) are printed in degrees too. They are medians, and they put the eyes at the head pose's distance, a few per cent too far, so they read a few per cent low.

## How it works

Each frame goes through these steps:

1. **Undistortion.** The image is undistorted with the checkerboard's k1 and k2, before anything else (`gaze/face.py`, `gaze/checkerboard.py`).
2. **Landmarks.** MediaPipe's face landmarker gives 478 points. The eyes count as open when both eye aspect ratios are at least 0.2.
3. **Pose.** ptgaze's 3D face model is fitted to the 468 face points by solvePnP, with the camera's true focal. The eyes' centre E is MediaPipe's eye corners in pixels, placed at the depth of this pose's eye corners.
4. **Normalisation** (Zhang et al. 2018): a virtual camera, f = 960 px at 60 cm, looks at the face without roll, and gives a 224 x 224 image.
5. **Network.** UniGaze ViT-H in float32 gives the gaze's pitch and yaw. They are turned back into the camera frame.
6. **Correction.** The network's gaze, as 2 angles in the eye-to-camera frame, and MediaPipe's 8 eye numbers go into a ridge regression. The 8 numbers are each iris's centre (x, y) and the height of its upper and lower eyelid, in eye widths. The ridge gives the 2 angles of the true gaze. The ridge's lambda is chosen by cross-validation on the calibration: each series left out in turn, and 5 groups of dot positions left out in turn (`gaze/method.py`).
7. **Point.** The gaze ray from E meets the screen's plane (`gaze/geometry.py`).

The live view runs MediaPipe and the network in two threads, at about 20 points per second with a delay of about 75 ms on a laptop RTX 5090.

## Code structure

`python -m gaze` (`gaze/__main__.py`) runs the two commands:

- **`calibrate`**: if `settings/camera.json` is missing, `checkerboard.py` measures the camera first. Then `record.py` records the 13 sub-blocks, in its own process with its full-screen window: a thread reads the camera, a thread runs MediaPipe, and a few threads write the images. Back in `__main__`, `method.py` runs the network on the recorded images (`face.py`), fits the ridge and writes `model.npz`.
- **`session`**: `record.py` records the 9 points the same way, `method.py` runs the network on them and chooses the adaptation, then `live.py` shows the live gaze, with MediaPipe and the network in two threads.

`face.py` (MediaPipe, undistortion, pose, normalisation, UniGaze) and `geometry.py` (screen, eyes, angles, rays) are used by all of them. The ridge's inputs come from one function, `method.features()`, for the recordings and for the live gaze alike.

## Files

```
gaze/
  __main__.py        the commands: checkerboard, calibrate, session
  protocol.py        the calibration's dots and instructions
  record.py          the full-screen dots and the recording (images and landmarks)
  camera.py          the webcam: opening it, the reading thread, the JPEG writer, settings/camera.json
  checkerboard.py    the camera's focal and distortion
  face.py            MediaPipe, undistortion, head pose, normalisation, UniGaze
  geometry.py        the screen, the eyes, angles and rays
  method.py          frames, the network on recordings, the ridge, the 9-point adaptation
  live.py            the live gaze: the dot or the numbers
  saccade_filter.py  the dot's filter (F)
resources/           MediaPipe's face landmarker model, ptgaze's 3D face model
settings/            config_example.json; your config.json, camera.json and current calibration (not in git)
data/                your calibrations and sessions: images, landmarks, network outputs, model.npz (not in git; about
                     7 GB per calibration and 0.5 GB per session; old data/session_* folders can be deleted)
```

## Third-party parts and licences

- **UniGaze** (github.com/ut-vision/UniGaze; weights on Hugging Face, `UniGaze/UniGaze-models`, `unigaze_h14_joint`). Its weights are under a **non-commercial** licence: read it before any other use.
- **MediaPipe** face landmarker (`resources/face_landmarker_v2_with_blendshapes.task`), Google, Apache 2.0.
- **ptgaze**'s 3D face model (`resources/ptgaze_face_model.npz`), from hysts/pytorch_mpiigaze_demo, MIT.
- The saccade filter follows Kumar et al. (2008) and Feit et al. (2017).
- The normalisation follows Zhang, Sugano and Bulling (2018), *Revisiting data normalization for appearance-based gaze estimation*.
- The camera calibration is Zhang (2000), *A flexible new technique for camera calibration* (OpenCV's `calibrateCamera`).
