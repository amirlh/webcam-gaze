"""The saccade filter of the live view."""

import collections

import numpy as np


class SaccadeFilter:
    """Averages the gaze point during a fixation, and jumps with the eyes at a saccade.

    The output is the running mean of the current fixation. A sample farther than
    `threshold` noise standard deviations from it is held back for one frame: if the
    next sample is closer to it than to the fixation, a new fixation starts there (a
    saccade, followed with one frame of delay); otherwise it was an outlier and is
    dropped. The noise standard deviation of each axis is estimated from the median
    step between consecutive raw samples, which the few saccade steps barely move.
    Method of Kumar et al. 2008 and Feit et al. 2017.
    """

    def __init__(self, threshold=4.0, min_gain=0.15, sigma_init=(0.03, 0.03), sigma_window=90,
                 sigma_bounds=(0.005, 0.15)):
        self.threshold = threshold      # in noise standard deviations
        self.min_gain = min_gain        # gain once the fixation is long: follows slow drift in ~6 frames
        self.sigma = np.array(sigma_init, dtype=float)  # noise sd per axis, normalized screen units
        self._sigma_bounds = sigma_bounds
        self._steps = collections.deque(maxlen=sigma_window)
        self._previous = None
        self._fixation = None
        self._count = 0
        self._candidate = None

    def _distance(self, a, b):
        return float(np.linalg.norm((a - b) / self.sigma))

    def _update_sigma(self, sample):
        if self._previous is not None:
            self._steps.append(np.abs(sample - self._previous))
        self._previous = sample
        if len(self._steps) >= 10:
            # For independent noise of sd s, a step has sd s*sqrt(2), and 1.4826 * median|x| = sd.
            sigma = 1.4826 * np.median(np.array(self._steps), axis=0) / np.sqrt(2.0)
            self.sigma = np.clip(sigma, *self._sigma_bounds)

    def _start(self, point, count=1):
        self._fixation = np.array(point, dtype=float)
        self._count = count

    def step(self, sample):
        """One raw sample (x, y) in, the filtered point out."""
        sample = np.asarray(sample, dtype=float)
        self._update_sigma(sample)
        if self._fixation is None:
            self._start(sample)
        elif self._distance(sample, self._fixation) <= self.threshold:
            self._candidate = None   # a held sample that was not followed: an outlier, dropped
            self._count += 1
            self._fixation += max(self.min_gain, 1.0 / self._count) * (sample - self._fixation)
        elif (self._candidate is not None
              and self._distance(sample, self._candidate) < self._distance(sample, self._fixation)):
            # Second far sample, on the candidate's side: a saccade.
            if self._distance(sample, self._candidate) <= self.threshold:
                self._start((sample + self._candidate) / 2.0, count=2)
            else:
                self._start(sample)  # the candidate was taken in flight
            self._candidate = None
        else:
            # First far sample: hold the point for one frame.
            self._candidate = sample
        return self._fixation.copy()
