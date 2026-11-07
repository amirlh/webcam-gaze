"""What is shown on the screen, in two modes.

    calibration   the long calibration, once per screen and camera, about 6 min: 13 sub-blocks at different places (usual
                  distance, nearer, farther, shifted, head turned), the head moving gently all the time.
    short         the short calibration, at the start of each session, about 20 s: 9 points, head still at the usual place.

Points are normalized screen coordinates: centre 0, edges +-0.5, x to the right, y down. Each point stays POINT_S seconds;
its frames count from SKIP_S seconds after it appears (the eyes have landed).
"""

import numpy as np

POINT_S = 2.0
SKIP_S = 0.6
SEED = 20261026

_A_VALUES = (-0.4, -0.13, 0.13, 0.4)
_C_VALUES = (-0.35, 0.0, 0.35)
GRID_A = [(x, y) for y in _A_VALUES for x in _A_VALUES]
GRID_C = [(x, y) for y in _C_VALUES for x in _C_VALUES]
# the crosses of the live view (key G)
TARGETS = [(-0.12, 0.0), (-0.35, -0.3), (0.35, 0.3), (0.12, -0.3), (-0.35, 0.3), (0.12, 0.0), (0.35, -0.3),
           (-0.12, 0.3), (-0.12, -0.3), (0.12, 0.3), (-0.35, 0.0), (0.35, 0.0)]


def _shuffled(points, seed):
    return [points[i] for i in np.random.default_rng(seed).permutation(len(points))]


_MOVE = ("\nDo NOT hold your head still: move it gently and all the time around this place, as in normal use,\n"
         "your eyes always on the black centre of each red dot.")

# name -> (points, nose mark or None, instruction)
SUBBLOCKS = {
    "A": (_shuffled(GRID_A, SEED), None, "A: at your usual distance (about 60 cm). 16 points." + _MOVE),
    "C1": (_shuffled(GRID_C, SEED + 1), None, "C1: come about 15 cm closer (about 45 cm). 9 points." + _MOVE),
    "C2": (_shuffled(GRID_C, SEED + 2), None, "C2: move about 15 cm back (about 75 cm). 9 points." + _MOVE),
    "C3": (_shuffled(GRID_C, SEED + 3), None, "C3: move your head about 10 cm to YOUR left. 9 points." + _MOVE),
    "C4": (_shuffled(GRID_C, SEED + 4), None, "C4: move your head about 10 cm to YOUR right. 9 points." + _MOVE),
    "R1": (_shuffled(GRID_C, SEED + 11), (-0.25, 0.0),
           "R1: turn your head to YOUR left, your nose roughly towards the grey cross. 9 points." + _MOVE),
    "R2": (_shuffled(GRID_C, SEED + 12), (-0.5, 0.0),
           "R2: turn your head to YOUR left, your nose roughly towards the grey cross at the edge of the screen. 9 points." + _MOVE),
    "R3": (_shuffled(GRID_C, SEED + 13), (0.25, 0.0),
           "R3: turn your head to YOUR right, your nose roughly towards the grey cross. 9 points." + _MOVE),
    "R4": (_shuffled(GRID_C, SEED + 14), (0.5, 0.0),
           "R4: turn your head to YOUR right, your nose roughly towards the grey cross at the edge of the screen. 9 points." + _MOVE),
    "P1": (_shuffled(GRID_C, SEED + 21), None, "P1: come about 15 cm closer AND move about 10 cm to YOUR left. 9 points." + _MOVE),
    "P2": (_shuffled(GRID_C, SEED + 22), None, "P2: come about 15 cm closer AND move about 10 cm to YOUR right. 9 points." + _MOVE),
    "P3": (_shuffled(GRID_C, SEED + 23), None, "P3: move about 15 cm back AND about 10 cm to YOUR left. 9 points." + _MOVE),
    "P4": (_shuffled(GRID_C, SEED + 24), None, "P4: move about 15 cm back AND about 10 cm to YOUR right. 9 points." + _MOVE),
    "K": (_shuffled(GRID_C, SEED + 51), None,
          "Short calibration: sit at your usual place (about 60 cm) and keep your head still.\n"
          "Look at the black centre of each red dot. 9 points, about 20 s."),
}
CALIBRATION = ["A", "C1", "C2", "C3", "C4", "R1", "R2", "R3", "R4", "P1", "P2", "P3", "P4"]
SHORT = ["K"]
MODES = {"calibration": CALIBRATION, "short": SHORT}
